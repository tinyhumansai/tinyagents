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

use super::types::{GraphBlueprintHandle, LanguageCompiler, ReplCancelFlag, ReplPolicy};
use super::{
    ReplCallKind, ReplCallRecord, dynamic_to_repl_value, json_to_repl_value, repl_value_to_dynamic,
};
use crate::error::TinyAgentsError;
use crate::harness::events::{AgentEvent, EventSink, ReplCallPhase};
use crate::harness::ids::{CallId, new_call_id};
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
    /// External cancellation flag, observed fail-closed by the `on_progress`
    /// hook (mid-script) and the blocking capability bridge (mid-call).
    pub cancel: ReplCancelFlag,
    /// Per-cell shared buffers (stdout, calls, answer, host error, vars).
    pub buffers: super::CellBuffers,
    /// Session-cumulative call counters.
    pub counters: Arc<Mutex<CallCounters>>,
    /// Graph blueprints drafted this session, keyed by name.
    pub drafts: Arc<Mutex<BTreeMap<String, GraphBlueprintHandle>>>,
}

/// How often the watcher thread in [`bridge_block_on_raw`] wakes to observe an
/// armed [`ReplCancelFlag`] while a capability call is in flight.
///
/// Small enough that a user cancel releases a hung call promptly, large enough
/// that watching a fast (scripted-test) call costs nothing measurable.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Why the watcher tripped a bounded [`bridge_block_on_raw`] call.
enum BridgeStop {
    /// The per-cell wall-clock deadline elapsed.
    Deadline,
    /// The external [`ReplCancelFlag`] was tripped.
    Cancelled,
}

/// Drives an async capability future to completion synchronously, bounded by an
/// optional wall-clock `deadline` **and** an external `cancel` flag — the v1
/// "blocking bridge" adapter (see the [module docs](self)), with fail-closed
/// enforcement of both [`ReplPolicy::timeout`] and host cancellation.
///
/// `on_progress` (see [`build_engine`]) only fires between Rhai
/// statements/operations, so it can never interrupt a blocked native call: this
/// is the enforcement point for that case. A detached watcher thread races the
/// capability future; when the deadline elapses or `cancel` trips first, the
/// future is dropped — canceling the underlying request, since providers are
/// built on cancel-safe `reqwest`/`futures` — and a `Timeout` or `Cancelled`
/// error is returned instead of blocking the session forever. If the future
/// finishes first, the watcher observes the dropped receiver and exits.
fn bridge_block_on_raw<F: Future>(
    deadline: Option<Instant>,
    cancel: &ReplCancelFlag,
    future: F,
) -> std::result::Result<F::Output, TinyAgentsError> {
    // Fail closed before the call even starts if either bound has already
    // tripped, so a cancel/timeout that landed between statements is honored
    // without dispatching the call at all.
    if cancel.is_cancelled() {
        return Err(TinyAgentsError::Cancelled);
    }
    if let Some(deadline) = deadline
        && Instant::now() >= deadline
    {
        return Err(TinyAgentsError::Timeout(format!(
            "{DEADLINE_EXCEEDED_TOKEN} before a host capability call could start"
        )));
    }

    let (tx, rx) = futures::channel::oneshot::channel::<BridgeStop>();
    let watcher_cancel = cancel.clone();
    // A detached watcher wakes the race below when the deadline elapses or the
    // cancel flag trips. If the capability future finishes first, `rx` is
    // dropped and `tx.is_canceled()` lets the watcher exit promptly instead of
    // polling out a full deadline.
    std::thread::spawn(move || {
        loop {
            if tx.is_canceled() {
                return;
            }
            if watcher_cancel.is_cancelled() {
                let _ = tx.send(BridgeStop::Cancelled);
                return;
            }
            match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        let _ = tx.send(BridgeStop::Deadline);
                        return;
                    }
                    std::thread::sleep((deadline - now).min(CANCEL_POLL_INTERVAL));
                }
                None => std::thread::sleep(CANCEL_POLL_INTERVAL),
            }
        }
    });

    match futures::executor::block_on(futures::future::select(Box::pin(future), rx)) {
        futures::future::Either::Left((output, _watcher)) => Ok(output),
        futures::future::Either::Right((stop, _fut)) => match stop {
            Ok(BridgeStop::Cancelled) => Err(TinyAgentsError::Cancelled),
            Ok(BridgeStop::Deadline) => Err(TinyAgentsError::Timeout(format!(
                "{DEADLINE_EXCEEDED_TOKEN} during a host capability call"
            ))),
            // The watcher dropped its sender without sending — only reachable in
            // a race the future has effectively already won; re-check the bounds
            // and prefer cancellation, never silently succeeding.
            Err(_canceled) => {
                if cancel.is_cancelled() {
                    Err(TinyAgentsError::Cancelled)
                } else {
                    Err(TinyAgentsError::Timeout(format!(
                        "{DEADLINE_EXCEEDED_TOKEN} during a host capability call"
                    )))
                }
            }
        },
    }
}

/// Convenience wrapper over [`bridge_block_on_raw`] for the common case where
/// the capability future itself resolves to a `Result`, flattening the deadline
/// / cancellation error into the same error channel as the call's own failures.
fn bridge_block_on<T, F>(
    deadline: Option<Instant>,
    cancel: &ReplCancelFlag,
    future: F,
) -> std::result::Result<T, TinyAgentsError>
where
    F: Future<Output = std::result::Result<T, TinyAgentsError>>,
{
    bridge_block_on_raw(deadline, cancel, future)?
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

/// Records a completed capability call (or emitted event) into the per-cell
/// buffer **and** streams it live on the session [`EventSink`] as an
/// [`AgentEvent::ReplCall`] with phase [`ReplCallPhase::Completed`].
///
/// `call_id` is generated by the caller up front so a preceding
/// [`emit_call_started`] event can carry the same id, letting a host pair the
/// start and completion of one call.
fn record<State: Send + Sync>(
    ctx: &HostContext<State>,
    call_id: CallId,
    kind: ReplCallKind,
    name: &str,
    detail: Value,
    elapsed: Duration,
) {
    let record = ReplCallRecord {
        call_id,
        kind,
        name: name.to_string(),
        detail,
        elapsed,
    };
    emit_repl_call(ctx, &record, ReplCallPhase::Completed);
    ctx.buffers.push_call(record);
}

/// Emits an [`AgentEvent::ReplCall`] on the session event sink so a live
/// observer sees a capability call as it happens, rather than only in
/// [`ReplResult::calls`](super::ReplResult) after the cell returns.
fn emit_repl_call<State: Send + Sync>(
    ctx: &HostContext<State>,
    record: &ReplCallRecord,
    phase: ReplCallPhase,
) {
    ctx.events.emit(AgentEvent::ReplCall {
        session_id: ctx.session_label.clone(),
        record: record.clone(),
        phase,
    });
}

/// Streams a `ReplCall` "started" event for a capability call about to be
/// dispatched, carrying `call_id` (matched by the later [`record`] completion),
/// its kind, and name — but no `detail` (arguments are only in the completed
/// record) and a zero `elapsed`.
fn emit_call_started<State: Send + Sync>(
    ctx: &HostContext<State>,
    call_id: &CallId,
    kind: ReplCallKind,
    name: &str,
) {
    let record = ReplCallRecord {
        call_id: call_id.clone(),
        kind,
        name: name.to_string(),
        detail: Value::Null,
        elapsed: Duration::default(),
    };
    emit_repl_call(ctx, &record, ReplCallPhase::Started);
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
    if counters.agent >= ctx.policy.max_agent_calls {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "agent call limit ({}) exceeded",
                ctx.policy.max_agent_calls
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
    // Funnel the depth-cap check through the shared harness guard so the REPL
    // sub-run bound stays in lock-step with SubAgent/SubAgentTool.
    crate::harness::context::RunConfig::checked_child_depth(ctx.run_depth, ctx.policy.max_depth)
        .map(|_| ())
        .map_err(|err| raise(ctx, err))
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

mod authoring;
mod batched;
mod capabilities;

use authoring::*;
use batched::*;
use capabilities::*;

// ── Engine construction ─────────────────────────────────────────────────────

/// Sentinel exception value `on_progress` terminates a script with when the
/// per-cell [`ReplPolicy::timeout`] deadline elapses. `eval_cell` recognizes
/// this exact string and maps it to `TinyAgentsError::Timeout` instead of the
/// generic runtime-error path.
pub(super) const DEADLINE_EXCEEDED_TOKEN: &str = "ragsh cell exceeded its wall-clock timeout";

/// Sentinel exception value `on_progress` terminates a script with when an
/// external [`ReplCancelFlag`] is tripped mid-script. `eval_cell`'s
/// `map_rhai_error` recognizes this exact string and maps it to
/// [`TinyAgentsError::Cancelled`] instead of the generic runtime-error path.
pub(super) const CANCELLED_TOKEN: &str = "ragsh cell cancelled by host";

/// Builds a sandboxed Rhai engine for a session, registering every host-backed
/// built-in against the session's live registries and policy.
///
/// The engine is configured with the policy operation limit (fail-closed on
/// runaway scripts) and is granted no filesystem, network, or process access —
/// the only host surface is the capability functions registered here.
pub(super) fn build_engine<State: Send + Sync + 'static>(ctx: Arc<HostContext<State>>) -> Engine {
    let mut engine = Engine::new();
    engine.set_max_operations(ctx.policy.max_operations);

    // Fail-closed wall-clock deadline: `eval_cell` arms `ctx.buffers`'s
    // per-cell deadline before running the script. `on_progress` is polled
    // between Rhai statements/operations, so this catches runaway *script*
    // loops (a busy `while true {}` with no host calls) that `max_operations`
    // alone might not bound tightly enough in wall-clock terms. Host
    // capability calls (`model_query`, `tool_call`, …) are bounded separately
    // by `bridge_block_on`, since a blocked native call never yields back to
    // `on_progress`.
    let deadline_ctx = ctx.clone();
    engine.on_progress(move |_ops| {
        // External cancellation takes precedence: a host that tripped the
        // cancel flag mid-script terminates the cell at the next
        // statement/operation with the cancellation sentinel, which
        // `map_rhai_error` maps to `TinyAgentsError::Cancelled`.
        if deadline_ctx.cancel.is_cancelled() {
            return Some(Dynamic::from(CANCELLED_TOKEN.to_string()));
        }
        // A fail-closed host check (currently: push_stdout_line's
        // max_output_bytes enforcement) may have stashed an error without
        // Rhai itself failing; abort promptly instead of letting the script
        // keep running until it happens to yield control back naturally.
        // `eval_cell` prefers the stashed error over this sentinel's text.
        if deadline_ctx.buffers.host_error_pending() {
            return Some(Dynamic::from(DEADLINE_EXCEEDED_TOKEN.to_string()));
        }
        match deadline_ctx.buffers.deadline() {
            Some(deadline) if Instant::now() >= deadline => {
                Some(Dynamic::from(DEADLINE_EXCEEDED_TOKEN.to_string()))
            }
            _ => None,
        }
    });

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
            new_call_id(),
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
            new_call_id(),
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

#[cfg(test)]
mod bridge_deadline_test {
    use super::*;

    #[test]
    fn no_deadline_awaits_to_completion() {
        let out = bridge_block_on::<u32, _>(None, &ReplCancelFlag::new(), async { Ok(7) })
            .expect("no deadline");
        assert_eq!(out, 7);
    }

    #[test]
    fn future_finishing_before_the_deadline_succeeds() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let out =
            bridge_block_on::<u32, _>(Some(deadline), &ReplCancelFlag::new(), async { Ok(9) })
                .expect("within deadline");
        assert_eq!(out, 9);
    }

    #[test]
    fn deadline_already_elapsed_fails_closed_without_starting_the_call() {
        // Regression test: `ReplPolicy::timeout` used to be parsed but never
        // enforced anywhere a host capability call could hang forever.
        let deadline = Instant::now() - Duration::from_millis(1);
        let err =
            bridge_block_on::<u32, _>(Some(deadline), &ReplCancelFlag::new(), async { Ok(1) })
                .expect_err("deadline already passed");
        assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    }

    #[test]
    fn a_hanging_call_is_cut_off_at_the_deadline_instead_of_blocking_forever() {
        // A future that never resolves models a hung provider/tool call. The
        // deadline must still return control promptly rather than hanging the
        // whole `eval_cell` (and therefore the session) forever.
        let start = Instant::now();
        let deadline = start + Duration::from_millis(30);
        let err = bridge_block_on::<u32, _>(
            Some(deadline),
            &ReplCancelFlag::new(),
            futures::future::pending::<std::result::Result<u32, TinyAgentsError>>(),
        )
        .expect_err("hanging call must be cut off at the deadline");
        assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "took {:?}, should return promptly at the 30ms deadline",
            start.elapsed()
        );
    }

    #[test]
    fn cancel_already_set_fails_closed_without_starting_the_call() {
        // A flag tripped before the bridge runs must short-circuit to
        // `Cancelled` without ever polling the (here, never-resolving) future.
        let cancel = ReplCancelFlag::new();
        cancel.cancel();
        let err = bridge_block_on::<u32, _>(
            None,
            &cancel,
            futures::future::pending::<std::result::Result<u32, TinyAgentsError>>(),
        )
        .expect_err("pre-cancelled call must not start");
        assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
    }

    #[test]
    fn a_hanging_call_is_cut_off_promptly_when_the_cancel_flag_trips() {
        // With no deadline, a hung capability future must still be released
        // once a host trips the cancel flag from another thread — the watcher
        // polls the flag and drops the future.
        let start = Instant::now();
        let cancel = ReplCancelFlag::new();
        let trigger = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            trigger.cancel();
        });
        let err = bridge_block_on::<u32, _>(
            None,
            &cancel,
            futures::future::pending::<std::result::Result<u32, TinyAgentsError>>(),
        )
        .expect_err("hanging call must be cut off on cancel");
        assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "took {:?}, should return promptly after the ~40ms cancel",
            start.elapsed()
        );
    }
}
