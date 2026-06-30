//! Capability-bound built-in functions for the Rhai-backed `.ragsh` session
//! (design milestones R3–R5).
//!
//! This module registers the reserved built-in functions on a session's
//! [`rhai::Engine`] as **host capabilities**: each one resolves a name through
//! the session's [`CapabilityRegistry`], enforces the [`ReplPolicy`] call and
//! recursion limits, records a [`ReplCallRecord`], and lowers to the real
//! harness/graph runtime.
//!
//! The surface registered here is:
//!
//! - model calls — `model_query`, `model_query_batched`
//! - agent calls — `agent_query`, `agent_query_batched`
//! - graph runs — `graph_run`, `graph_run_batched`
//! - tool calls — `tool_call`, `tool_call_batched`
//! - session built-ins — `emit`, `show_vars`, `answer`, plus `print`/`debug`
//!   capture
//! - graph authoring — `graph_define`, `graph_validate`, `graph_compile`,
//!   `graph_diff`, `graph_register`, which lower through the Cluster H `.rag`
//!   compiler and capability resolver. Generated topology is never installed
//!   directly: a draft only becomes `compiled` after the resolver binds it, and
//!   `graph_register` honors [`ReplPolicy::generated_graphs_require_review`].
//!
//! ## The async adapter (blocking bridge)
//!
//! The Rhai engine is synchronous, but model, tool, agent, and graph calls are
//! async. This slice uses the design's **blocking bridge** for v1: a host
//! function builds the async future and drives it to completion in place with
//! [`futures::executor::block_on`] (see [`bridge_block_on`]). This keeps the
//! capability boundary deterministic for scripted tests (a [`ScriptedModel`] or
//! [`FakeTool`] resolves without yielding to a reactor) and works for real
//! providers when the session is driven from a multi-threaded runtime, where
//! blocking one worker does not starve the call's own I/O.
//!
//! The design's longer-term direction is *command recording* (host functions
//! emit `ReplCommand` values the async runtime executes after the cell); the
//! public `ReplResult`/`ReplCallRecord` types are already shaped for it. The
//! bridge is intentionally the only blocking surface and is confined to this
//! module.
//!
//! [`ScriptedModel`]: crate::harness::testkit::ScriptedModel
//! [`FakeTool`]: crate::harness::testkit::FakeTool

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rhai::{Array, Dynamic, Engine, EvalAltResult, Map, Position};
use serde_json::{Value, json};

use super::types::{GraphBlueprintHandle, LanguageCompiler, ReplPolicy};
use super::{
    ReplCallKind, ReplCallRecord, dynamic_to_repl_value, json_to_repl_value, repl_value_to_dynamic,
};
use crate::error::TinyAgentsError;
use crate::harness::events::EventSink;
use crate::harness::ids::new_call_id;
use crate::harness::message::Message;
use crate::harness::model::ModelRequest;
use crate::harness::tool::ToolCall;
use crate::language::compiler::compile_with_provenance;
use crate::language::parser::parse_str;
use crate::language::resolver::Resolver;
use crate::language::types::Origin;
use crate::language::{Blueprint, blueprint_diff};
use crate::registry::CapabilityRegistry;

/// Session-cumulative counters for capability calls, enforced against the
/// `ReplPolicy` `max_*_calls` limits. Counts accumulate across cells (the
/// limits are documented per session) and are shared with every capability
/// closure on the engine.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CallCounters {
    /// `model_query` (and per-item `model_query_batched`) calls made.
    pub model: usize,
    /// `tool_call` (and per-item `tool_call_batched`) calls made.
    pub tool: usize,
    /// `graph_run` (and per-item `graph_run_batched`) calls made.
    pub graph: usize,
    /// `agent_query` (and per-item `agent_query_batched`) calls made.
    pub agent: usize,
    /// `graph_define` blueprints drafted.
    pub graph_def: usize,
}

/// The host-side context shared (via `Arc`) with every capability closure on a
/// session's engine. Holds the live registries, application state, policy, and
/// the shared per-cell buffers / session counters / graph drafts.
pub(super) struct HostContext<State: Send + Sync> {
    /// The unified capability catalog (models, tools, graphs, agents).
    pub registry: Arc<CapabilityRegistry<State>>,
    /// The application state capability calls are invoked against.
    pub state: Arc<State>,
    /// The session policy (call/recursion/concurrency limits).
    pub policy: ReplPolicy,
    /// Optional expressive-language compiler handle (provenance label).
    pub language: Option<LanguageCompiler>,
    /// The session id, used as the generated-graph provenance label.
    pub session_label: String,
    /// The session's run depth, the parent depth for recursive sub-runs.
    pub run_depth: usize,
    /// The event sink shared with the run context.
    pub events: EventSink,
    /// Per-cell shared buffers (stdout, calls, answer, host error, vars).
    pub buffers: super::CellBuffers,
    /// Session-cumulative call counters.
    pub counters: Arc<Mutex<CallCounters>>,
    /// Graph blueprints drafted this session, keyed by name.
    pub drafts: Arc<Mutex<BTreeMap<String, GraphBlueprintHandle>>>,
}

/// Drives an async capability future to completion synchronously — the v1
/// "blocking bridge" adapter (see the [module docs](self)).
fn bridge_block_on<F: Future>(future: F) -> F::Output {
    futures::executor::block_on(future)
}

/// One completed `model_query_batched` item: `(model, text, finish_reason,
/// structured, elapsed)`.
type ModelBatchItem = (String, String, Option<String>, bool, Duration);

/// One completed `agent_query_batched` item: `(agent, text, elapsed)`.
type AgentBatchItem = (String, String, Duration);

// ── Error / recording helpers ───────────────────────────────────────────────

/// Stashes the precise crate error so `eval_cell` can surface it verbatim, and
/// returns the stringly-typed Rhai runtime error the engine propagates.
fn raise<State: Send + Sync>(ctx: &HostContext<State>, err: TinyAgentsError) -> Box<EvalAltResult> {
    let message = err.to_string();
    ctx.buffers.set_host_error(err);
    Box::new(EvalAltResult::ErrorRuntime(
        Dynamic::from(message),
        Position::NONE,
    ))
}

/// Raises a [`TinyAgentsError::Validation`] for an invalid script argument.
fn invalid<State: Send + Sync>(
    ctx: &HostContext<State>,
    message: impl Into<String>,
) -> Box<EvalAltResult> {
    raise(ctx, TinyAgentsError::Validation(message.into()))
}

/// Records a capability call (or emitted event) into the per-cell buffer.
fn record<State: Send + Sync>(
    ctx: &HostContext<State>,
    kind: ReplCallKind,
    name: &str,
    detail: Value,
    elapsed: Duration,
) {
    ctx.buffers.push_call(ReplCallRecord {
        call_id: new_call_id(),
        kind,
        name: name.to_string(),
        detail,
        elapsed,
    });
}

// ── Map argument helpers ────────────────────────────────────────────────────

/// Reads a string field from a Rhai object map argument.
fn map_str(map: &Map, key: &str) -> Option<String> {
    map.get(key).and_then(|d| d.clone().into_string().ok())
}

/// Reads a boolean field from a Rhai object map argument.
fn map_bool(map: &Map, key: &str) -> Option<bool> {
    map.get(key).and_then(|d| d.as_bool().ok())
}

/// Converts a Rhai object map argument into a JSON value (for tool arguments
/// and structured payloads).
fn map_json(map: &Map, key: &str) -> Option<Value> {
    map.get(key)
        .map(|d| dynamic_to_repl_value(d).to_json())
        .filter(|v| !v.is_null())
}

// ── Counter limit helpers ───────────────────────────────────────────────────

/// Increments and bounds the model-call counter.
fn bump_model<State: Send + Sync>(ctx: &HostContext<State>) -> Result<(), Box<EvalAltResult>> {
    let mut counters = ctx.counters.lock().expect("counters poisoned");
    if counters.model >= ctx.policy.max_model_calls {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "model call limit ({}) exceeded",
                ctx.policy.max_model_calls
            )),
        ));
    }
    counters.model += 1;
    Ok(())
}

/// Increments and bounds the tool-call counter.
fn bump_tool<State: Send + Sync>(ctx: &HostContext<State>) -> Result<(), Box<EvalAltResult>> {
    let mut counters = ctx.counters.lock().expect("counters poisoned");
    if counters.tool >= ctx.policy.max_tool_calls {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "tool call limit ({}) exceeded",
                ctx.policy.max_tool_calls
            )),
        ));
    }
    counters.tool += 1;
    Ok(())
}

/// Increments and bounds the graph-run counter.
fn bump_graph<State: Send + Sync>(ctx: &HostContext<State>) -> Result<(), Box<EvalAltResult>> {
    let mut counters = ctx.counters.lock().expect("counters poisoned");
    if counters.graph >= ctx.policy.max_graph_calls {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "graph call limit ({}) exceeded",
                ctx.policy.max_graph_calls
            )),
        ));
    }
    counters.graph += 1;
    Ok(())
}

/// Increments and bounds the agent-call counter.
fn bump_agent<State: Send + Sync>(ctx: &HostContext<State>) -> Result<(), Box<EvalAltResult>> {
    let mut counters = ctx.counters.lock().expect("counters poisoned");
    if counters.agent >= ctx.policy.max_model_calls {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "agent call limit ({}) exceeded",
                ctx.policy.max_model_calls
            )),
        ));
    }
    counters.agent += 1;
    Ok(())
}

/// Enforces the recursion-depth bound for a sub-run (agent or graph).
///
/// Reuses the harness recursion bookkeeping (Cluster G): a sub-run executes one
/// level below the session's run depth, and a child depth past
/// [`ReplPolicy::max_depth`] fails closed with
/// [`TinyAgentsError::SubAgentDepth`].
fn check_depth<State: Send + Sync>(ctx: &HostContext<State>) -> Result<(), Box<EvalAltResult>> {
    let child_depth = ctx.run_depth + 1;
    if child_depth > ctx.policy.max_depth {
        return Err(raise(
            ctx,
            TinyAgentsError::SubAgentDepth(ctx.policy.max_depth),
        ));
    }
    Ok(())
}

// ── Request builders ────────────────────────────────────────────────────────

/// Builds a [`ModelRequest`] from a `model_query` argument map.
fn build_model_request(model: &str, params: &Map) -> ModelRequest {
    let mut messages = Vec::new();
    if let Some(system) = map_str(params, "system") {
        messages.push(Message::system(system));
    }
    if let Some(prompt) = map_str(params, "prompt") {
        messages.push(Message::user(prompt));
    }
    ModelRequest {
        messages,
        model: Some(model.to_string()),
        ..Default::default()
    }
}

/// Wraps a model response text as the script-visible value (a string by
/// default, or a structured map when `structured: true`).
fn model_value(text: String, finish_reason: Option<String>, structured: bool) -> Dynamic {
    if structured {
        let mut map = Map::new();
        map.insert("content".into(), Dynamic::from(text));
        if let Some(reason) = finish_reason {
            map.insert("finish_reason".into(), Dynamic::from(reason));
        }
        Dynamic::from_map(map)
    } else {
        Dynamic::from(text)
    }
}

// ── Single capability implementations ───────────────────────────────────────

fn model_query_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let model_name =
        map_str(params, "model").ok_or_else(|| invalid(ctx, "model_query: missing `model`"))?;
    bump_model(ctx)?;
    let model = ctx
        .registry
        .model(&model_name)
        .ok_or_else(|| raise(ctx, TinyAgentsError::ModelNotFound(model_name.clone())))?;
    let request = build_model_request(&model_name, params);
    let start = Instant::now();
    let response =
        bridge_block_on(model.invoke(&ctx.state, request)).map_err(|err| raise(ctx, err))?;
    let elapsed = start.elapsed();
    let finish_reason = response.finish_reason.clone();
    let text = Message::Assistant(response.message).text();
    record(
        ctx,
        ReplCallKind::Model,
        &model_name,
        json!({ "chars": text.len() }),
        elapsed,
    );
    Ok(model_value(
        text,
        finish_reason,
        map_bool(params, "structured").unwrap_or(false),
    ))
}

fn tool_call_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let tool_name =
        map_str(params, "tool").ok_or_else(|| invalid(ctx, "tool_call: missing `tool`"))?;
    bump_tool(ctx)?;
    let tool = ctx
        .registry
        .tool(&tool_name)
        .ok_or_else(|| raise(ctx, TinyAgentsError::ToolNotFound(tool_name.clone())))?;
    let arguments = map_json(params, "arguments").unwrap_or(Value::Null);
    let call = ToolCall {
        id: new_call_id().as_str().to_string(),
        name: tool_name.clone(),
        arguments: arguments.clone(),
    };
    let start = Instant::now();
    let result = bridge_block_on(tool.call(&ctx.state, call)).map_err(|err| raise(ctx, err))?;
    let elapsed = start.elapsed();
    record(
        ctx,
        ReplCallKind::Tool,
        &tool_name,
        json!({ "arguments": arguments }),
        elapsed,
    );
    if let Some(error) = result.error {
        return Err(raise(ctx, TinyAgentsError::Tool(error)));
    }
    let structured = map_bool(params, "structured").unwrap_or(false);
    if structured && result.raw.is_some() {
        let mut map = Map::new();
        map.insert("content".into(), Dynamic::from(result.content));
        map.insert(
            "raw".into(),
            repl_value_to_dynamic(&json_to_repl_value(&result.raw.unwrap_or(Value::Null))),
        );
        Ok(Dynamic::from_map(map))
    } else {
        Ok(Dynamic::from(result.content))
    }
}

fn agent_query_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    use crate::graph::subagent_node::SubAgentInput;
    let agent_name =
        map_str(params, "agent").ok_or_else(|| invalid(ctx, "agent_query: missing `agent`"))?;
    bump_agent(ctx)?;
    check_depth(ctx)?;
    let agent = ctx.registry.agent(&agent_name).ok_or_else(|| {
        raise(
            ctx,
            TinyAgentsError::Capability(format!("agent `{agent_name}` is not registered")),
        )
    })?;
    let prompt = map_str(params, "prompt")
        .or_else(|| map_str(params, "input"))
        .unwrap_or_default();
    let mut input = SubAgentInput::prompt(prompt);
    if let Some(data) = map_json(params, "input") {
        input = input.with_data(data);
    }
    let start = Instant::now();
    let output =
        bridge_block_on(agent.run(input, ctx.events.clone())).map_err(|err| raise(ctx, err))?;
    record(
        ctx,
        ReplCallKind::Agent,
        &agent_name,
        json!({ "model_calls": output.model_calls, "tool_calls": output.tool_calls }),
        start.elapsed(),
    );
    Ok(Dynamic::from(output.text))
}

/// Resolves a registered graph blueprint and records the run, returning a
/// reference to the resolved topology.
///
/// Resolving a registered graph routes through the capability registry; the
/// REPL hands back the resolved blueprint reference (graph id, start node, node
/// count) rather than installing or stepping topology here. Materializing a
/// `CompiledGraph` and driving its super-steps is owned by the graph runtime
/// and wired in a later slice; this keeps the REPL an orchestration surface,
/// not a topology-mutation surface.
fn graph_run_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let graph_name =
        map_str(params, "graph").ok_or_else(|| invalid(ctx, "graph_run: missing `graph`"))?;
    bump_graph(ctx)?;
    check_depth(ctx)?;
    let blueprint = ctx
        .registry
        .graph_blueprint(&graph_name)
        .ok_or_else(|| {
            raise(
                ctx,
                TinyAgentsError::Capability(format!("graph `{graph_name}` is not registered")),
            )
        })?
        .clone();
    record(
        ctx,
        ReplCallKind::Graph,
        &graph_name,
        json!({ "nodes": blueprint.nodes.len() }),
        Duration::default(),
    );
    Ok(blueprint_reference(&blueprint))
}

/// Builds the script-visible reference map for a resolved graph blueprint.
fn blueprint_reference(blueprint: &Blueprint) -> Dynamic {
    let mut map = Map::new();
    map.insert("graph".into(), Dynamic::from(blueprint.graph_id.clone()));
    map.insert("start".into(), Dynamic::from(blueprint.start.clone()));
    map.insert("nodes".into(), Dynamic::from(blueprint.nodes.len() as i64));
    map.insert("resolved".into(), Dynamic::from(true));
    Dynamic::from_map(map)
}

// ── Graph-authoring implementations ─────────────────────────────────────────

fn graph_define_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let name =
        map_str(params, "name").ok_or_else(|| invalid(ctx, "graph_define: missing `name`"))?;
    let source =
        map_str(params, "source").ok_or_else(|| invalid(ctx, "graph_define: missing `source`"))?;

    {
        let mut counters = ctx.counters.lock().expect("counters poisoned");
        if counters.graph_def >= ctx.policy.max_graph_definitions {
            return Err(raise(
                ctx,
                TinyAgentsError::LimitExceeded(format!(
                    "graph definition limit ({}) exceeded",
                    ctx.policy.max_graph_definitions
                )),
            ));
        }
        counters.graph_def += 1;
    }
    if source.len() > ctx.policy.max_script_bytes {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "graph source is {} bytes, exceeding max_script_bytes ({})",
                source.len(),
                ctx.policy.max_script_bytes
            )),
        ));
    }

    let label = ctx
        .language
        .as_ref()
        .map(|l| l.provenance_label.clone())
        .unwrap_or_else(|| ctx.session_label.clone());
    let origin = Origin::generated_by(label);
    let program = parse_str(&source).map_err(|err| raise(ctx, err))?;
    let blueprints =
        compile_with_provenance(&program, origin.clone()).map_err(|err| raise(ctx, err))?;
    let blueprint = blueprints
        .into_iter()
        .find(|b| b.graph_id == name)
        .ok_or_else(|| {
            invalid(
                ctx,
                format!("graph_define: source has no graph named `{name}`"),
            )
        })?;

    let handle = GraphBlueprintHandle {
        name: blueprint.graph_id.clone(),
        source,
        blueprint: blueprint.clone(),
        origin,
        compiled: false,
        requires_review: ctx.policy.generated_graphs_require_review,
    };
    ctx.drafts
        .lock()
        .expect("drafts poisoned")
        .insert(handle.name.clone(), handle.clone());
    record(
        ctx,
        ReplCallKind::Graph,
        "graph_define",
        json!({ "name": handle.name }),
        Duration::default(),
    );
    Ok(draft_descriptor(&handle))
}

/// Builds the script-visible descriptor map for a graph draft (carrying its
/// name, node count, and compile/review status). The opaque
/// [`GraphBlueprintHandle`] itself lives host-side in `ctx.drafts`.
fn draft_descriptor(handle: &GraphBlueprintHandle) -> Dynamic {
    let mut map = Map::new();
    map.insert("name".into(), Dynamic::from(handle.name.clone()));
    map.insert(
        "nodes".into(),
        Dynamic::from(handle.blueprint.nodes.len() as i64),
    );
    map.insert("compiled".into(), Dynamic::from(handle.compiled));
    map.insert(
        "requires_review".into(),
        Dynamic::from(handle.requires_review),
    );
    Dynamic::from_map(map)
}

/// Looks up a graph draft by the `name` field of a descriptor map.
fn lookup_draft<State: Send + Sync>(
    ctx: &HostContext<State>,
    descriptor: &Map,
    func: &str,
) -> Result<GraphBlueprintHandle, Box<EvalAltResult>> {
    let name = map_str(descriptor, "name")
        .ok_or_else(|| invalid(ctx, format!("{func}: descriptor is missing `name`")))?;
    ctx.drafts
        .lock()
        .expect("drafts poisoned")
        .get(&name)
        .cloned()
        .ok_or_else(|| invalid(ctx, format!("{func}: no graph draft named `{name}`")))
}

fn graph_validate_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    descriptor: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let handle = lookup_draft(ctx, descriptor, "graph_validate")?;
    let program = parse_str(&handle.source).map_err(|err| raise(ctx, err))?;
    let diagnostics = Resolver::from_registry(&*ctx.registry).resolve_program(&program);
    let array: Array = diagnostics
        .iter()
        .map(|d| Dynamic::from(d.message.clone()))
        .collect();
    Ok(Dynamic::from_array(array))
}

fn graph_compile_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    descriptor: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let mut handle = lookup_draft(ctx, descriptor, "graph_compile")?;
    // Bind the blueprint through the same resolver gate file-backed `.rag`
    // source passes — generated topology is never trusted blindly.
    Resolver::from_registry(&*ctx.registry)
        .resolve_blueprint(&handle.blueprint)
        .map_err(|err| raise(ctx, err))?;
    handle.compiled = true;
    handle.requires_review = ctx.policy.generated_graphs_require_review;
    ctx.drafts
        .lock()
        .expect("drafts poisoned")
        .insert(handle.name.clone(), handle.clone());
    record(
        ctx,
        ReplCallKind::Graph,
        "graph_compile",
        json!({ "name": handle.name, "requires_review": handle.requires_review }),
        Duration::default(),
    );
    Ok(draft_descriptor(&handle))
}

fn graph_diff_handles<State: Send + Sync>(
    ctx: &HostContext<State>,
    old: &Blueprint,
    new: &Blueprint,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let diff = blueprint_diff(old, new);
    let value = serde_json::to_value(&diff)
        .map_err(|err| raise(ctx, TinyAgentsError::Validation(err.to_string())))?;
    Ok(repl_value_to_dynamic(&json_to_repl_value(&value)))
}

fn graph_register_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let graph = params
        .get("graph")
        .and_then(|d| d.read_lock::<Map>().map(|m| m.clone()))
        .ok_or_else(|| {
            invalid(
                ctx,
                "graph_register: `graph` must be a compiled graph descriptor",
            )
        })?;
    let handle = lookup_draft(ctx, &graph, "graph_register")?;
    if !handle.compiled {
        return Err(raise(
            ctx,
            TinyAgentsError::Validation(
                "graph_register: graph must be compiled via graph_compile first".to_string(),
            ),
        ));
    }
    let review_id = map_str(params, "review_id").filter(|s| !s.is_empty());
    if handle.requires_review && review_id.is_none() {
        return Err(raise(
            ctx,
            TinyAgentsError::Validation(format!(
                "graph_register: generated graph `{}` requires review (no review_id)",
                handle.name
            )),
        ));
    }
    // Enforce the review gate and emit a registry intent. The compiled topology
    // is handed to the host for installation through the registry resolver —
    // the REPL never installs generated topology directly.
    record(
        ctx,
        ReplCallKind::Graph,
        "graph_register",
        json!({ "name": handle.name, "review_id": review_id }),
        Duration::default(),
    );
    Ok(Dynamic::from(handle.name))
}

// ── Batched implementations ─────────────────────────────────────────────────

/// Extracts the object-map items of a batched argument array.
fn batch_items<State: Send + Sync>(
    ctx: &HostContext<State>,
    items: &Array,
    func: &str,
) -> Result<Vec<Map>, Box<EvalAltResult>> {
    items
        .iter()
        .map(|item| {
            item.read_lock::<Map>()
                .map(|m| m.clone())
                .ok_or_else(|| invalid(ctx, format!("{func}: each item must be an object map")))
        })
        .collect()
}

fn model_query_batched_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    items: &Array,
) -> Result<Dynamic, Box<EvalAltResult>> {
    use futures::stream::{self, StreamExt};

    let items = batch_items(ctx, items, "model_query_batched")?;
    let mut prepared = Vec::with_capacity(items.len());
    for params in &items {
        let model_name = map_str(params, "model")
            .ok_or_else(|| invalid(ctx, "model_query_batched: missing `model`"))?;
        bump_model(ctx)?;
        let model = ctx
            .registry
            .model(&model_name)
            .ok_or_else(|| raise(ctx, TinyAgentsError::ModelNotFound(model_name.clone())))?;
        let request = build_model_request(&model_name, params);
        let structured = map_bool(params, "structured").unwrap_or(false);
        prepared.push((model_name, model, request, structured));
    }

    let concurrency = ctx.policy.max_concurrency.max(1);
    let results: Vec<Result<ModelBatchItem, TinyAgentsError>> = bridge_block_on(async {
        stream::iter(prepared.iter().map(|(name, model, request, structured)| {
            let name = name.clone();
            let structured = *structured;
            async move {
                let start = Instant::now();
                let response = model.invoke(&ctx.state, request.clone()).await?;
                let finish_reason = response.finish_reason.clone();
                let text = Message::Assistant(response.message).text();
                Ok((name, text, finish_reason, structured, start.elapsed()))
            }
        }))
        .buffered(concurrency)
        .collect()
        .await
    });

    let mut out = Array::with_capacity(results.len());
    for result in results {
        let (name, text, finish_reason, structured, elapsed) =
            result.map_err(|err| raise(ctx, err))?;
        record(
            ctx,
            ReplCallKind::Model,
            &name,
            json!({ "chars": text.len() }),
            elapsed,
        );
        out.push(model_value(text, finish_reason, structured));
    }
    Ok(Dynamic::from_array(out))
}

fn tool_call_batched_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    items: &Array,
) -> Result<Dynamic, Box<EvalAltResult>> {
    use futures::stream::{self, StreamExt};

    let items = batch_items(ctx, items, "tool_call_batched")?;
    let mut prepared = Vec::with_capacity(items.len());
    for params in &items {
        let tool_name = map_str(params, "tool")
            .ok_or_else(|| invalid(ctx, "tool_call_batched: missing `tool`"))?;
        bump_tool(ctx)?;
        let tool = ctx
            .registry
            .tool(&tool_name)
            .ok_or_else(|| raise(ctx, TinyAgentsError::ToolNotFound(tool_name.clone())))?;
        let arguments = map_json(params, "arguments").unwrap_or(Value::Null);
        prepared.push((tool_name, tool, arguments));
    }

    let concurrency = ctx.policy.max_concurrency.max(1);
    let results: Vec<
        Result<(String, crate::harness::tool::ToolResult, Duration), TinyAgentsError>,
    > = bridge_block_on(async {
        stream::iter(prepared.iter().map(|(name, tool, arguments)| {
            let name = name.clone();
            let call = ToolCall {
                id: new_call_id().as_str().to_string(),
                name: name.clone(),
                arguments: arguments.clone(),
            };
            async move {
                let start = Instant::now();
                let result = tool.call(&ctx.state, call).await?;
                Ok((name, result, start.elapsed()))
            }
        }))
        .buffered(concurrency)
        .collect()
        .await
    });

    let mut out = Array::with_capacity(results.len());
    for result in results {
        let (name, tool_result, elapsed) = result.map_err(|err| raise(ctx, err))?;
        record(
            ctx,
            ReplCallKind::Tool,
            &name,
            json!({ "chars": tool_result.content.len() }),
            elapsed,
        );
        if let Some(error) = tool_result.error {
            return Err(raise(ctx, TinyAgentsError::Tool(error)));
        }
        out.push(Dynamic::from(tool_result.content));
    }
    Ok(Dynamic::from_array(out))
}

fn agent_query_batched_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    items: &Array,
) -> Result<Dynamic, Box<EvalAltResult>> {
    use crate::graph::subagent_node::SubAgentInput;
    use futures::stream::{self, StreamExt};

    let items = batch_items(ctx, items, "agent_query_batched")?;
    let mut prepared = Vec::with_capacity(items.len());
    for params in &items {
        let agent_name = map_str(params, "agent")
            .ok_or_else(|| invalid(ctx, "agent_query_batched: missing `agent`"))?;
        bump_agent(ctx)?;
        check_depth(ctx)?;
        let agent = ctx.registry.agent(&agent_name).ok_or_else(|| {
            raise(
                ctx,
                TinyAgentsError::Capability(format!("agent `{agent_name}` is not registered")),
            )
        })?;
        let prompt = map_str(params, "prompt")
            .or_else(|| map_str(params, "input"))
            .unwrap_or_default();
        let mut input = SubAgentInput::prompt(prompt);
        if let Some(data) = map_json(params, "input") {
            input = input.with_data(data);
        }
        prepared.push((agent_name, agent, input));
    }

    let concurrency = ctx.policy.max_concurrency.max(1);
    let results: Vec<Result<AgentBatchItem, TinyAgentsError>> = bridge_block_on(async {
        stream::iter(prepared.iter().map(|(name, agent, input)| {
            let name = name.clone();
            async move {
                let start = Instant::now();
                let output = agent.run(input.clone(), ctx.events.clone()).await?;
                Ok((name, output.text, start.elapsed()))
            }
        }))
        .buffered(concurrency)
        .collect()
        .await
    });

    let mut out = Array::with_capacity(results.len());
    for result in results {
        let (name, text, elapsed) = result.map_err(|err| raise(ctx, err))?;
        record(ctx, ReplCallKind::Agent, &name, json!({}), elapsed);
        out.push(Dynamic::from(text));
    }
    Ok(Dynamic::from_array(out))
}

fn graph_run_batched_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    items: &Array,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let items = batch_items(ctx, items, "graph_run_batched")?;
    let mut out = Array::with_capacity(items.len());
    for params in &items {
        out.push(graph_run_impl(ctx, params)?);
    }
    Ok(Dynamic::from_array(out))
}

// ── Engine construction ─────────────────────────────────────────────────────

/// Builds a sandboxed Rhai engine for a session, registering every host-backed
/// built-in against the session's live registries and policy.
///
/// The engine is configured with the policy operation limit (fail-closed on
/// runaway scripts) and is granted no filesystem, network, or process access —
/// the only host surface is the capability functions registered here.
pub(super) fn build_engine<State: Send + Sync + 'static>(ctx: Arc<HostContext<State>>) -> Engine {
    let mut engine = Engine::new();
    engine.set_max_operations(ctx.policy.max_operations);

    // ── stdout capture ──
    let stdout_ctx = ctx.clone();
    engine.on_print(move |text| stdout_ctx.buffers.push_stdout_line(text));
    let debug_ctx = ctx.clone();
    engine.on_debug(move |text, _source, _pos| debug_ctx.buffers.push_stdout_line(text));

    // ── emit(name) / emit(name, #{ ... }) ──
    let emit_ctx = ctx.clone();
    engine.register_fn("emit", move |name: &str| {
        record(
            &emit_ctx,
            ReplCallKind::Emit,
            name,
            Value::Null,
            Duration::default(),
        );
    });
    let emit_payload_ctx = ctx.clone();
    engine.register_fn("emit", move |name: &str, data: Map| {
        let detail = dynamic_to_repl_value(&Dynamic::from_map(data)).to_json();
        record(
            &emit_payload_ctx,
            ReplCallKind::Emit,
            name,
            detail,
            Duration::default(),
        );
    });

    // ── answer(content) ──
    let answer_ctx = ctx.clone();
    engine.register_fn("answer", move |content: &str| {
        answer_ctx.buffers.set_answer(content.to_string());
    });

    // ── show_vars() ──
    let show_ctx = ctx.clone();
    engine.register_fn("show_vars", move || {
        show_ctx.buffers.push_stdout_line("# vars");
        for (name, value) in show_ctx.buffers.vars_snapshot() {
            show_ctx
                .buffers
                .push_stdout_line(&format!("{name} = {value}"));
        }
    });

    // ── model capabilities ──
    let model_ctx = ctx.clone();
    engine.register_fn("model_query", move |params: Map| {
        model_query_impl(&model_ctx, &params)
    });
    let model_batch_ctx = ctx.clone();
    engine.register_fn("model_query_batched", move |items: Array| {
        model_query_batched_impl(&model_batch_ctx, &items)
    });

    // ── tool capabilities ──
    let tool_ctx = ctx.clone();
    engine.register_fn("tool_call", move |params: Map| {
        tool_call_impl(&tool_ctx, &params)
    });
    let tool_batch_ctx = ctx.clone();
    engine.register_fn("tool_call_batched", move |items: Array| {
        tool_call_batched_impl(&tool_batch_ctx, &items)
    });

    // ── agent capabilities ──
    let agent_ctx = ctx.clone();
    engine.register_fn("agent_query", move |params: Map| {
        agent_query_impl(&agent_ctx, &params)
    });
    let agent_batch_ctx = ctx.clone();
    engine.register_fn("agent_query_batched", move |items: Array| {
        agent_query_batched_impl(&agent_batch_ctx, &items)
    });

    // ── graph run capabilities ──
    let graph_ctx = ctx.clone();
    engine.register_fn("graph_run", move |params: Map| {
        graph_run_impl(&graph_ctx, &params)
    });
    let graph_batch_ctx = ctx.clone();
    engine.register_fn("graph_run_batched", move |items: Array| {
        graph_run_batched_impl(&graph_batch_ctx, &items)
    });

    // ── graph authoring (lowering through the `.rag` compiler) ──
    let define_ctx = ctx.clone();
    engine.register_fn("graph_define", move |params: Map| {
        graph_define_impl(&define_ctx, &params)
    });
    let validate_ctx = ctx.clone();
    engine.register_fn("graph_validate", move |descriptor: Map| {
        graph_validate_impl(&validate_ctx, &descriptor)
    });
    let compile_ctx = ctx.clone();
    engine.register_fn("graph_compile", move |descriptor: Map| {
        graph_compile_impl(&compile_ctx, &descriptor)
    });
    let diff_name_ctx = ctx.clone();
    engine.register_fn(
        "graph_diff",
        move |name: &str, draft: Map| -> Result<Dynamic, Box<EvalAltResult>> {
            let old = diff_name_ctx
                .registry
                .graph_blueprint(name)
                .ok_or_else(|| {
                    invalid(
                        &diff_name_ctx,
                        format!("graph_diff: graph `{name}` is not registered"),
                    )
                })?
                .clone();
            let new = lookup_draft(&diff_name_ctx, &draft, "graph_diff")?;
            graph_diff_handles(&diff_name_ctx, &old, &new.blueprint)
        },
    );
    let diff_draft_ctx = ctx.clone();
    engine.register_fn(
        "graph_diff",
        move |old: Map, new: Map| -> Result<Dynamic, Box<EvalAltResult>> {
            let old = lookup_draft(&diff_draft_ctx, &old, "graph_diff")?;
            let new = lookup_draft(&diff_draft_ctx, &new, "graph_diff")?;
            graph_diff_handles(&diff_draft_ctx, &old.blueprint, &new.blueprint)
        },
    );
    let register_ctx = ctx.clone();
    engine.register_fn("graph_register", move |params: Map| {
        graph_register_impl(&register_ctx, &params)
    });

    engine
}
