//! The core superstep loop body: `run_loop` drives one model call,
//! any requested tool calls, and repeats until the model finishes or a
//! configured limit is reached.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

use super::model_call::{ModelCallBase, ToolCallBase};
use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Drives the loop body, returning `Ok(())` on a clean finish or the first
    /// error encountered. The caller owns lifecycle bookkeeping (final status
    /// transition, `RunFailed`/`on_error` on error).
    pub(super) async fn run_loop(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        input: Vec<Message>,
        streaming: bool,
    ) -> Result<()> {
        let record = ctx.emit(AgentEvent::RunStarted {
            run_id: ctx.run_id().clone(),
            thread_id: ctx.thread_id().cloned(),
        });
        status.set_last_event(record.id);
        status.mark_running(HarnessPhase::Idle);

        // Reconcile the `RunConfig`-derived limit tracker with the harness's
        // `RunPolicy::limits` so model/tool call caps have one enforced
        // source of truth instead of the two silently disagreeing (see
        // `LimitTracker::sync_call_limits`).
        ctx.limits.sync_call_limits(
            self.policy.limits.max_model_calls,
            self.policy.limits.max_tool_calls,
        );

        let mut messages = input;

        // The tool set is fixed for the duration of a run, so build the sorted
        // schema vec once here instead of re-collecting, re-calling every tool's
        // `schema()`, and re-sorting on every turn (per model call).
        let tool_schemas = self.tools.schemas();

        status.mark_running(HarnessPhase::Middleware);
        self.middleware.run_before_agent(ctx, state).await?;

        loop {
            // Safe cancellation checkpoint: if an orchestrator requested
            // cooperative cancellation, stop before doing any further work
            // (steering, request build, or model call) for this turn.
            if ctx.cancellation.is_cancelled() {
                return Err(TinyAgentsError::Cancelled);
            }

            // Safe steering checkpoint: drain any orchestrator/human steering
            // commands and apply the policy-permitted ones before the next
            // model call. Cancel terminates the run; Pause short-circuits it.
            match crate::harness::steering::apply_pending_steering(ctx, &mut messages)? {
                crate::harness::steering::SteeringOutcome::Cancel => {
                    return Err(TinyAgentsError::Cancelled);
                }
                crate::harness::steering::SteeringOutcome::Pause => break,
                crate::harness::steering::SteeringOutcome::Continue => {}
            }

            // Fail-closed limit and deadline checks before each model call.
            if ctx.check_deadline().is_err() {
                ctx.emit(AgentEvent::LimitReached {
                    kind: LimitKind::WallClock,
                });
                return Err(TinyAgentsError::Timeout(format!(
                    "run `{}` exceeded its wall-clock deadline",
                    ctx.run_id()
                )));
            }
            // The context's `LimitTracker` (synced with `RunPolicy::limits`
            // above) is the single enforced source of truth for the model-call
            // cap, so the reported limit always matches the one that trips.
            if let Err(err) = ctx.record_model_call() {
                ctx.emit(AgentEvent::LimitReached {
                    kind: LimitKind::ModelCalls,
                });
                return Err(TinyAgentsError::LimitExceeded(err.to_string()));
            }

            // Build the request from the working transcript, tool schemas, and
            // policy response format.
            status.mark_running(HarnessPhase::BuildingRequest);
            let mut request = ModelRequest::new(messages.clone()).with_tools(tool_schemas.clone());
            if let Some(format) = &self.policy.default_response_format {
                request = request.with_response_format(format.clone());
            }
            if let Some(cap) = ctx.config.max_turn_output_tokens {
                request.max_tokens =
                    Some(request.max_tokens.map_or(cap, |current| current.min(cap)));
            }

            status.mark_running(HarnessPhase::Middleware);
            self.middleware
                .run_before_model(ctx, state, &mut request)
                .await?;

            // Resolve the model for the event/log name before invoking.
            let binding = self
                .models
                .resolve_request(&request, None, None)
                .ok_or_else(|| {
                    TinyAgentsError::ModelNotFound(
                        request
                            .model
                            .clone()
                            .unwrap_or_else(|| "<default>".to_string()),
                    )
                })?;
            let model_name = binding.resolved.name.clone();

            // Resolve the structured-output plan against the resolved model.
            // `Auto` consults the model profile to choose provider-native schema
            // mode versus a tool-call fallback; an explicit `JsonSchema` always
            // uses provider-native mode. The chosen strategy drives extraction of
            // the final response below.
            let structured_plan: Option<(StructuredStrategy, String, Value)> =
                match request.response_format.clone() {
                    Some(ResponseFormat::Auto { name, schema }) => {
                        let strategy = StructuredStrategy::for_profile(binding.model.profile());
                        match strategy {
                            StructuredStrategy::ProviderSchema => {
                                request.response_format =
                                    Some(ResponseFormat::json_schema(name.clone(), schema.clone()));
                            }
                            StructuredStrategy::ToolCall => {
                                request.response_format = Some(ResponseFormat::Text);
                                request.tools.push(ToolSchema {
                                    name: name.clone(),
                                    description: format!("Return the result as `{name}`."),
                                    parameters: schema.clone(),
                                    format: crate::harness::tool::ToolFormat::Json,
                                });
                                request.tool_choice = ToolChoice::Tool(name.clone());
                            }
                        }
                        Some((strategy, name, schema))
                    }
                    Some(ResponseFormat::JsonSchema { name, schema }) => {
                        Some((StructuredStrategy::ProviderSchema, name, schema))
                    }
                    _ => None,
                };

            let call_id = CallId::new(format!("{}-model-{}", ctx.run_id(), run.model_calls + 1));
            status.mark_running(HarnessPhase::Model);
            status.active_model_call = Some(call_id.clone());
            // Captured here (where the call actually starts) so the completed
            // event carries a real start time for duration-aware exporters.
            let model_started_at_ms = crate::harness::ids::now_ms();
            let record = ctx.emit(AgentEvent::ModelStarted {
                call_id: call_id.clone(),
                model: model_name,
            });
            status.set_last_event(record.id);

            // The real model call (cache + retry + fallback core) is the
            // innermost base of the model-wrap onion. Lifecycle `before_model`
            // already ran above; the wrap onion runs here; lifecycle
            // `after_model` runs below — so ordering is:
            // before_model -> wrap onion (outer..inner..base) -> after_model.
            let base = ModelCallBase {
                harness: self,
                call_id: call_id.clone(),
                resolved: binding.resolved,
                model: binding.model,
                streaming,
            };
            // Snapshot the request messages for observability before `request`
            // is moved into the model-wrap onion, gated by the capture policy so
            // payload-free runs never serialize prompt text.
            let captured_input = self
                .policy
                .capture
                .model_io
                .then(|| serde_json::to_value(&request.messages).unwrap_or(Value::Null));
            let mut response = self
                .middleware
                .run_wrapped_model(ctx, state, request, &base)
                .await?
                .into_response();

            status.mark_running(HarnessPhase::Middleware);
            self.middleware
                .run_after_model(ctx, state, &mut response)
                .await?;

            // Accounting.
            run.model_calls += 1;
            run.steps += 1;
            status.model_calls = run.model_calls;
            status.active_model_call = None;
            if let Some(usage) = response.usage {
                run.usage.record(usage);
                status.usage = run.usage;
                let record = ctx.emit(AgentEvent::UsageRecorded { usage });
                status.set_last_event(record.id);
            }
            let captured_output = self
                .policy
                .capture
                .model_io
                .then(|| serde_json::to_value(&response.message).unwrap_or(Value::Null));
            let record = ctx.emit(AgentEvent::ModelCompleted {
                call_id,
                started_at_ms: Some(model_started_at_ms),
                usage: response.usage,
                input: captured_input,
                output: captured_output,
            });
            status.set_last_event(record.id);

            messages.push(Message::Assistant(response.message.clone()));

            // Safe checkpoint: honor any control outcome a middleware requested
            // during this turn (for example an early-exit tool or a budget stop
            // hook), before executing further tools.
            if let Some(control) = ctx.take_control() {
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: control.kind().to_string(),
                    detail: match &control {
                        MiddlewareControl::StopWithFinal(text) => text.clone(),
                        MiddlewareControl::Interrupt { node, message } => {
                            format!("{node}: {message}")
                        }
                    },
                });
                status.set_last_event(record.id);
                match control {
                    MiddlewareControl::StopWithFinal(text) => {
                        run.final_response = Some(ModelResponse::assistant(text));
                        break;
                    }
                    MiddlewareControl::Interrupt { node, message } => {
                        return Err(TinyAgentsError::Interrupted { node, message });
                    }
                }
            }

            let tool_calls = response.tool_calls().to_vec();

            // A tool-call structured-output strategy produces an artificial tool
            // call that is not a registered tool; treat it as the final response
            // rather than attempting to execute it.
            let structured_tool_hit = matches!(
                &structured_plan,
                Some((StructuredStrategy::ToolCall, name, _))
                    if tool_calls.iter().any(|c| &c.name == name)
            );

            if tool_calls.is_empty() || structured_tool_hit {
                // Final response: optionally extract structured output using the
                // resolved plan (provider-native schema or tool-call arguments).
                if let Some((strategy, name, schema)) = &structured_plan {
                    let extractor =
                        StructuredExtractor::new(*strategy, name.clone(), schema.clone());
                    let output = extractor.extract(&response)?;
                    run.structured = Some(output.value);
                }
                run.final_response = Some(response);
                break;
            }

            // Execute requested tools serially.
            status.mark_running(HarnessPhase::Tools);
            for mut call in tool_calls {
                // Safe cancellation checkpoint: stop before invoking the next
                // (side-effecting) tool if cancellation was requested.
                if ctx.cancellation.is_cancelled() {
                    return Err(TinyAgentsError::Cancelled);
                }
                if ctx.check_deadline().is_err() {
                    ctx.emit(AgentEvent::LimitReached {
                        kind: LimitKind::WallClock,
                    });
                    return Err(TinyAgentsError::Timeout(format!(
                        "run `{}` exceeded its wall-clock deadline",
                        ctx.run_id()
                    )));
                }
                // The context's `LimitTracker` (synced with `RunPolicy::limits`
                // above) is the single enforced source of truth for the
                // tool-call cap, so the reported limit always matches the one
                // that trips.
                if let Err(err) = ctx.record_tool_call() {
                    ctx.emit(AgentEvent::LimitReached {
                        kind: LimitKind::ToolCalls,
                    });
                    return Err(TinyAgentsError::LimitExceeded(err.to_string()));
                }

                self.middleware
                    .run_before_tool(ctx, state, &mut call)
                    .await?;

                let tool = match self.tools.get(&call.name) {
                    Some(tool) => tool,
                    None => {
                        // The model called an unregistered tool. Apply the run's
                        // `UnknownToolPolicy` instead of unconditionally aborting.
                        let requested = call.name.clone();
                        let arguments = call.arguments.clone();
                        let call_id = CallId::new(call.id.clone());

                        // Rewrite mode: retarget to a fixed compatibility tool if
                        // that tool exists, otherwise fall through to recovery.
                        let rewrite_target = match &self.policy.unknown_tool {
                            UnknownToolPolicy::Rewrite { tool_name } => {
                                self.tools.get(tool_name).map(|t| (tool_name.clone(), t))
                            }
                            _ => None,
                        };

                        if let Some((tool_name, tool)) = rewrite_target {
                            call.name = tool_name.clone();
                            let record = ctx.emit(AgentEvent::UnknownToolCall {
                                call_id,
                                requested_name: requested,
                                arguments,
                                recovery: format!("rewrite:{tool_name}"),
                            });
                            status.set_last_event(record.id);
                            tool
                        } else if matches!(self.policy.unknown_tool, UnknownToolPolicy::Fail) {
                            return Err(TinyAgentsError::ToolNotFound(requested));
                        } else {
                            // `ReturnToolError` (or a Rewrite whose target is also
                            // missing): inject a tool-error result naming the
                            // requested tool and the valid tools, then continue so
                            // the model can correct itself. This consumed one
                            // tool-call budget slot above, bounding the loop.
                            let valid = self.tools.names().join(", ");
                            let args_repr = serde_json::to_string(&arguments)
                                .unwrap_or_else(|_| "<unserializable>".to_string());
                            let message = format!(
                                "unknown tool `{requested}` (arguments: {args_repr}); \
                                 valid tools: [{valid}]"
                            );
                            let record = ctx.emit(AgentEvent::UnknownToolCall {
                                call_id,
                                requested_name: requested.clone(),
                                arguments,
                                recovery: "tool_error".to_string(),
                            });
                            status.set_last_event(record.id);
                            run.tool_calls += 1;
                            status.tool_calls = run.tool_calls;
                            messages.push(Message::tool(call.id.clone(), message));
                            continue;
                        }
                    }
                };
                tool.schema().validate_call(&call)?;

                let tool_call_id = CallId::new(call.id.clone());
                let tool_name = call.name.clone();
                status.active_tool_calls.push(tool_call_id.clone());
                // Captured here (where the call actually starts) so the
                // completed event carries a real start time for
                // duration-aware exporters.
                let tool_started_at_ms = crate::harness::ids::now_ms();
                let record = ctx.emit(AgentEvent::ToolStarted {
                    call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                });
                status.set_last_event(record.id);

                // Snapshot the arguments for observability before `call` is
                // moved into the tool-wrap onion, gated by the capture policy.
                let captured_input = self.policy.capture.tool_io.then(|| call.arguments.clone());

                // The real tool call is the innermost base of the tool-wrap
                // onion (same before -> wrap -> after ordering as the model
                // path): lifecycle `before_tool` ran above, the wrap onion runs
                // here, and lifecycle `after_tool` runs below. Bounded by the
                // same remaining wall-clock budget as a model call, so a
                // hanging tool cannot block the run past its deadline either.
                let base = ToolCallBase { tool };
                let remaining = self.call_budget(ctx);
                let run_id = ctx.run_id().as_str().to_string();
                let fut = self.middleware.run_wrapped_tool(ctx, state, call, &base);
                let mut result = Self::with_call_budget(remaining, &run_id, "tool call", fut)
                    .await?
                    .into_result();

                self.middleware
                    .run_after_tool(ctx, state, &mut result)
                    .await?;

                run.tool_calls += 1;
                status.tool_calls = run.tool_calls;
                status.active_tool_calls.retain(|c| c != &tool_call_id);
                let captured_output = self
                    .policy
                    .capture
                    .tool_io
                    .then(|| Value::String(result.content.clone()));
                let record = ctx.emit(AgentEvent::ToolCompleted {
                    call_id: tool_call_id,
                    tool_name,
                    started_at_ms: Some(tool_started_at_ms),
                    input: captured_input,
                    output: captured_output,
                });
                status.set_last_event(record.id);

                messages.push(Message::tool(
                    result.call_id.clone(),
                    result.content.clone(),
                ));
            }
        }

        run.messages = messages;

        status.mark_running(HarnessPhase::Middleware);
        self.middleware.run_after_agent(ctx, state, run).await?;

        let record = ctx.emit(AgentEvent::RunCompleted {
            run_id: ctx.run_id().clone(),
        });
        status.set_last_event(record.id);

        Ok(())
    }

    /// Resolves the effective response-cache decision for `request`.
    ///
    /// Returns `Some((cache, key))` when a [`ResponseCache`] is attached to the
    /// harness *and* caching is enabled for this call. The per-request
    /// [`ModelRequest::cache_policy`] takes precedence over the harness-level
    /// [`RunPolicy::cache`][crate::harness::runtime::RunPolicy]; when the request
    /// carries no policy the run policy's
    /// [`response_cache_enabled`][crate::harness::cache::CachePolicy] decides.
    /// Returns `None` (caching disabled) when no cache is attached or the
    /// effective policy disables it.
    pub(super) fn response_cache_decision(
        &self,
        request: &ModelRequest,
    ) -> Option<(Arc<dyn ResponseCache>, String)> {
        let cache = self.response_cache.as_ref()?;
        let enabled = match &request.cache_policy {
            Some(policy) => policy.response_cache_enabled,
            None => self.policy.cache.response_cache_enabled,
        };
        if !enabled {
            return None;
        }
        // Skip caching multi-turn requests. Once the transcript contains a prior
        // assistant turn (or tool result), every subsequent call carries a
        // unique history and can never be re-served, so caching it only pays the
        // hashing/serialization cost and grows the cache with dead entries. The
        // first, history-free call is the only reusable one.
        if request
            .messages
            .iter()
            .any(|m| matches!(m, Message::Assistant(_) | Message::Tool(_)))
        {
            return None;
        }
        Some((Arc::clone(cache), cache_key(request)))
    }
}
