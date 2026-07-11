//! Single-shot capability implementations (`model_query`, `tool_call`,
//! `agent_query`, `graph_run`).
//!
//! Split out of `session/builtins/mod.rs`; see that module's doc comment
//! for the full built-in surface and the blocking-bridge design.

use super::*;

// ── Single capability implementations ───────────────────────────────────────

pub(super) fn model_query_impl<State: Send + Sync + 'static>(
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
    let call_id = new_call_id();
    emit_call_started(ctx, &call_id, ReplCallKind::Model, &model_name);
    let start = Instant::now();
    let response = bridge_block_on(
        ctx.buffers.deadline(),
        &ctx.cancel,
        model.invoke(&ctx.state, request),
    )
    .map_err(|err| raise(ctx, err))?;
    let elapsed = start.elapsed();
    let finish_reason = response.finish_reason.clone();
    let text = Message::Assistant(response.message).text();
    record(
        ctx,
        call_id,
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

pub(super) fn tool_call_impl<State: Send + Sync + 'static>(
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
    let call_id = new_call_id();
    let call = ToolCall {
        id: call_id.as_str().to_string(),
        name: tool_name.clone(),
        arguments: arguments.clone(),
        invalid: None,
    };
    emit_call_started(ctx, &call_id, ReplCallKind::Tool, &tool_name);
    let start = Instant::now();
    let result = bridge_block_on(
        ctx.buffers.deadline(),
        &ctx.cancel,
        tool.call(&ctx.state, call),
    )
    .map_err(|err| raise(ctx, err))?;
    let elapsed = start.elapsed();
    record(
        ctx,
        call_id,
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

pub(super) fn agent_query_impl<State: Send + Sync + 'static>(
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
    let call_id = new_call_id();
    emit_call_started(ctx, &call_id, ReplCallKind::Agent, &agent_name);
    let start = Instant::now();
    let output = bridge_block_on(
        ctx.buffers.deadline(),
        &ctx.cancel,
        agent.run(input, ctx.events.clone()),
    )
    .map_err(|err| raise(ctx, err))?;
    record(
        ctx,
        call_id,
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
pub(super) fn graph_run_impl<State: Send + Sync + 'static>(
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
        new_call_id(),
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
