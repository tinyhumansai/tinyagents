//! Batched capability implementations (`model_query_batched`,
//! `tool_call_batched`, `agent_query_batched`, `graph_run_batched`).
//!
//! Split out of `session/builtins/mod.rs`; see that module's doc comment
//! for the full built-in surface and the blocking-bridge design.

use super::*;

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
    let results: Vec<Result<ModelBatchItem, TinyAgentsError>> =
        bridge_block_on_raw(ctx.buffers.deadline(), async {
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
        })
        .map_err(|err| raise(ctx, err))?;

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
    > = bridge_block_on_raw(ctx.buffers.deadline(), async {
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
    })
    .map_err(|err| raise(ctx, err))?;

    // Each item's own tool-reported error is surfaced per item, matching the
    // single-call path's behavior for that one call, rather than aborting the
    // whole batch and discarding every other item's already-computed result —
    // a batch of N independent tool calls should not lose N-1 successes
    // because item N/2 failed. A `bridge_block_on_raw`/transport failure
    // above (a harness-level failure, not a tool-reported one) still aborts
    // the whole batch, since no results exist to preserve in that case.
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
        match tool_result.error {
            Some(error) => {
                let mut map = Map::new();
                map.insert("ok".into(), Dynamic::from(false));
                map.insert("error".into(), Dynamic::from(error));
                out.push(Dynamic::from_map(map));
            }
            None => {
                let mut map = Map::new();
                map.insert("ok".into(), Dynamic::from(true));
                map.insert("content".into(), Dynamic::from(tool_result.content));
                out.push(Dynamic::from_map(map));
            }
        }
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
    let results: Vec<Result<AgentBatchItem, TinyAgentsError>> =
        bridge_block_on_raw(ctx.buffers.deadline(), async {
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
        })
        .map_err(|err| raise(ctx, err))?;

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
