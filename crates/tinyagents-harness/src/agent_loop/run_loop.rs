//! The core superstep loop body: `run_loop` drives one model call,
//! any requested tool calls, and repeats until the model finishes or a
//! configured limit is reached.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

use super::model_call::ModelCallBase;
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
        // The tracker's wall-clock start is stamped when the context is
        // constructed (`RunContext::new`), not necessarily when the run
        // actually begins doing work — a context built ahead of time and
        // queued would otherwise burn down its deadline before the first
        // model call. Restart it here, at the true top of the run (M-8).
        ctx.limits.restart();
        let mut messages = input;
        // The body borrows the working transcript rather than owning it so the
        // transcript survives **every** exit path, not just the successful one.
        // A mid-turn tool failure used to drop everything accumulated so far,
        // leaving the caller unable to inspect, repair, or resume from the
        // partial conversation.
        let outcome = self
            .run_loop_body(state, ctx, run, status, &mut messages, streaming)
            .await;
        run.messages = std::mem::take(&mut messages);

        let exit = match outcome {
            Ok(exit) => exit,
            Err(error) => {
                tracing::debug!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    messages = run.messages.len(),
                    "[agent_loop] run failed; partial transcript preserved on the run"
                );
                return Err(error);
            }
        };

        status.mark_running(HarnessPhase::Middleware);
        self.middleware.run_after_agent(ctx, state, run).await?;

        match exit {
            LoopExit::Finished | LoopExit::LimitStop(_) => {
                if let LoopExit::LimitStop(kind) = &exit {
                    tracing::debug!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        limit_kind = ?kind,
                        messages = run.messages.len(),
                        "[agent_loop] completing with the partial run after a limit stop"
                    );
                }
                let record = ctx.emit(AgentEvent::RunCompleted {
                    run_id: ctx.run_id().clone(),
                });
                status.set_last_event(record.id);
            }
            LoopExit::Paused(pause) => {
                // A pause is not a completion: reporting `run.completed` here
                // is exactly what made "paused for a human" indistinguishable
                // from "the model produced an empty final answer". The pause
                // stays latched on the steering handle so a later `Resume`
                // lifts it.
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "paused".to_string(),
                    detail: pause.reason.clone().unwrap_or_else(|| {
                        format!("paused at checkpoint {}", pause.paused_at_checkpoint)
                    }),
                });
                status.set_last_event(record.id);
                tracing::debug!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    checkpoint = pause.paused_at_checkpoint,
                    "[agent_loop] run paused by steering"
                );
                run.paused = Some(pause);
            }
        }

        Ok(())
    }

    /// The loop body proper. Returns how the loop left off so the caller can
    /// finalize (and, on any error, still keep the working transcript).
    async fn run_loop_body(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        streaming: bool,
    ) -> Result<LoopExit> {
        let record = ctx.emit(AgentEvent::RunStarted {
            run_id: ctx.run_id().clone(),
            thread_id: ctx.thread_id().cloned(),
        });
        status.set_last_event(record.id);
        status.mark_running(HarnessPhase::Idle);

        // Reconcile the `RunConfig`-derived limit tracker with the harness's
        // `RunPolicy::limits` so model/tool call caps have one enforced source
        // of truth instead of the two silently disagreeing.
        //
        // The two directions are NOT symmetric, and telling them apart is the
        // whole reason `RunConfig`'s caps are `Option<usize>`:
        //
        // - an **explicitly set** `RunConfig` cap is the caller's ceiling, so
        //   the stricter of (config, policy) wins — fail-closed. Previously
        //   this was a plain assignment, so
        //   `RunConfig::new("r").with_max_model_calls(2)` against the default
        //   policy silently ran 25 model calls;
        // - an **unset** cap merely defaulted, so the policy is the only real
        //   source of truth and may raise the cap above that default.
        let effective_model_calls = resolve_call_cap(
            ctx.config.max_model_calls,
            self.policy.limits.max_model_calls,
        );
        let effective_tool_calls =
            resolve_call_cap(ctx.config.max_tool_calls, self.policy.limits.max_tool_calls);
        tracing::debug!(
            target: "tinyagents::agent_loop",
            run_id = %ctx.run_id(),
            config_model_calls = ?ctx.config.max_model_calls,
            config_tool_calls = ?ctx.config.max_tool_calls,
            policy_model_calls = self.policy.limits.max_model_calls,
            policy_tool_calls = self.policy.limits.max_tool_calls,
            effective_model_calls,
            effective_tool_calls,
            "[agent_loop] resolved run call caps"
        );
        // The values are already reconciled per-axis above, so the assignment
        // form (`sync_call_limits`) is the correct primitive here:
        // `tighten_call_limits` would additionally min against the tracker's
        // config-*default*-derived cap and so could not honor a policy that
        // legitimately raises an unset cap.
        ctx.limits
            .sync_call_limits(effective_model_calls, effective_tool_calls);

        // The tool set is fixed for the duration of a run, so build the sorted
        // schema vec once here instead of re-collecting, re-calling every tool's
        // `schema()`, and re-sorting on every turn (per model call).
        //
        // Only *direct* tools go on the wire. Deferred tools are indexed into
        // the run's catalogue and reached through the `tool_search` /
        // `tool_call` bridge, whose two schemas are appended *after* the
        // name-sorted direct set so the cached prefix is unchanged by them.
        // The same host allow-list gates both halves: deferral only ever
        // subtracts from what the host admitted. `resolve_tool_allowlist`
        // (not a raw read of `binding.allowed_tools`) is what applies I-9's
        // fail-closed default, so an empty declared list denies every tool
        // here exactly as it does for the direct set below.
        let allowed_tools = self.resolve_tool_allowlist(ctx)?;
        let host_allows = |name: &str| {
            allowed_tools
                .as_ref()
                .is_none_or(|allowed| allowed.contains(name))
        };
        let mut tool_schemas = self
            .tools
            .schemas()
            .into_iter()
            .filter(|schema| host_allows(&schema.name))
            .collect::<Vec<_>>();
        if let Some(preparation) = &self.policy.tool_schemas {
            tool_schemas = crate::tool::prepare_tool_schemas(&tool_schemas, preparation);
        }
        let deferred_catalog = self.deferred_catalog(&host_allows);
        if !deferred_catalog.is_empty() {
            // A host-registered `tool_search`/`tool_call` keeps its slot: the
            // intrinsic bridge only fills a name nobody registered. Check the
            // full registry (`self.tools.dispatch`), not just the direct set
            // collected into `tool_schemas` above — a `Hidden` or `Deferred`
            // registration under either name must also suppress the intrinsic
            // schema, because admission's own collision rule
            // (`self.tools.dispatch(&call.name).is_none()` in
            // `answer_discovery_bridge`) checks the same full registry. Using
            // a narrower rule here than admission uses would let this loop
            // advertise an intrinsic schema that admission then treats as
            // owned by the registered tool (or, for `Hidden`, refuses).
            let mut bridge: Vec<_> =
                crate::tool::discover::bridge_schemas(&deferred_catalog, &self.policy.discovery)
                    .into_iter()
                    .collect();
            if let Some(preparation) = &self.policy.tool_schemas {
                // The bridge schemas are generated here, after the direct set
                // was prepared above, so they need the same provider
                // projection (for example Gemini's `minimum`/`maximum`
                // removal) applied individually or they reach the wire raw.
                bridge = bridge
                    .into_iter()
                    .map(|schema| crate::tool::prepare_tool_schema(&schema, preparation))
                    .collect();
            }
            for schema in bridge {
                if self.tools.dispatch(&schema.name).is_none() {
                    tool_schemas.push(schema);
                }
            }
        }
        // Fail closed on a structured-output schema whose name collides with a
        // registered tool *or* the intrinsic discovery bridge. Under the
        // tool-call strategy the schema is sent as an extra `function` entry,
        // so a collision puts two identically-named functions in one request
        // — which OpenAI rejects outright — and makes "was this the schema or
        // the real tool?" unanswerable for every returned call. Two checks,
        // because neither alone covers every name that ends up on the wire:
        // `self.tools.names()` covers every registered tool (Direct, Deferred,
        // Hidden), but not the intrinsic `tool_search`/`tool_call` bridge,
        // which has no registry entry; `tool_schemas` covers the bridge (and
        // the Direct set) but never contains a Deferred tool's own name.
        if let Some(name) = self
            .policy
            .default_response_format
            .as_ref()
            .and_then(|format| match format {
                ResponseFormat::Auto { name, .. } | ResponseFormat::JsonSchema { name, .. } => {
                    Some(name)
                }
                _ => None,
            })
            && (self
                .tools
                .names()
                .iter()
                .any(|registered| registered == name)
                || tool_schemas.iter().any(|schema| &schema.name == name))
        {
            return Err(TinyAgentsError::Validation(format!(
                "structured-output schema name `{name}` collides with a registered tool (or the \
                 intrinsic discovery bridge) of the same name; rename one of them"
            )));
        }

        status.mark_running(HarnessPhase::Middleware);
        self.middleware.run_before_agent(ctx, state).await?;

        // Announced after `before_agent` so a listener that subscribes there
        // (the usual place) sees the run's tool surface. This is deliberately
        // the pre-middleware/pre-request baseline (see the event's doc
        // comment): per-turn `before_model` middleware and a structured-
        // output tool-call fallback can still narrow or grow what an
        // individual request actually sends.
        let record = ctx.emit(AgentEvent::ToolsAdvertised {
            direct: tool_schemas.len(),
            deferred: deferred_catalog.len(),
            schema_bytes: crate::token_estimation::tool_schema_bytes(&tool_schemas),
        });
        status.set_last_event(record.id);

        // Truncated-empty recovery state (see `RunPolicy::truncated_empty_retries`).
        // These persist across the retry `continue` within a single logical turn:
        // `boosted_max_tokens` overrides the next request's cap, `truncation_base`
        // records the original cap so growth stays clamped at 4x, and the counter
        // bounds how many times we re-issue the call.
        let mut truncated_empty_retries_used: u32 = 0;
        let mut boosted_max_tokens: Option<u32> = None;
        let mut truncation_base: Option<u32> = None;

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
            match crate::steering::apply_pending_steering(ctx, messages)? {
                crate::steering::SteeringOutcome::Cancel => {
                    return Err(TinyAgentsError::Cancelled);
                }
                crate::steering::SteeringOutcome::Pause => {
                    let pause = ctx
                        .steering
                        .as_ref()
                        .and_then(|handle| handle.pause_state())
                        .unwrap_or(crate::steering::PauseState {
                            reason: None,
                            paused_at_checkpoint: 0,
                        });
                    return Ok(LoopExit::Paused(pause));
                }
                crate::steering::SteeringOutcome::Continue => {}
            }

            // Safe checkpoint: honor a control outcome requested during the
            // *previous* turn's tool execution (or by `before_agent`) before
            // spending another model call on it. Draining only after the model
            // call meant a `StopWithFinal`/`Interrupt` raised from
            // `after_tool`/`wrap_tool` was honored one full model call late —
            // an extra billable provider round trip after a guardrail, or a
            // human gate, had already said stop.
            if let Some(exit) = self.apply_pending_control(ctx, run, status, messages)? {
                return Ok(exit);
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
            // `LimitBehavior::StopWithPartial` turns cap exhaustion into a
            // clean stop rather than an error that discards every message,
            // usage figure, and tool result the run produced up to that point.
            match ctx.limits.try_record_model_call() {
                Ok(crate::limits::LimitOutcome::Proceed) => {}
                Ok(crate::limits::LimitOutcome::Stop(_)) => {
                    ctx.emit(AgentEvent::LimitReached {
                        kind: LimitKind::ModelCalls,
                    });
                    tracing::debug!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        "[agent_loop] model-call cap reached; stopping with the partial run"
                    );
                    return Ok(LoopExit::LimitStop(LimitKind::ModelCalls));
                }
                Err(err) => {
                    ctx.emit(AgentEvent::LimitReached {
                        kind: LimitKind::ModelCalls,
                    });
                    // `RunConfig` cannot express a `LimitBehavior`, so the
                    // tracker built from it always carries the default
                    // (`Error`) even when the harness policy asks for
                    // `StopWithPartial`. Honor the policy here rather than
                    // discarding a run the operator asked to keep.
                    if matches!(
                        self.policy.limits.behavior,
                        crate::limits::LimitBehavior::StopWithPartial
                    ) {
                        tracing::debug!(
                            target: "tinyagents::agent_loop",
                            run_id = %ctx.run_id(),
                            "[agent_loop] model-call cap reached; policy asks to stop with the \
                             partial run"
                        );
                        return Ok(LoopExit::LimitStop(LimitKind::ModelCalls));
                    }
                    return Err(TinyAgentsError::LimitExceeded(err.to_string()));
                }
            }

            // Build the request from the working transcript, tool schemas, and
            // policy response format.  Go through `PromptBuilder` rather than
            // constructing `ModelRequest` directly: a provider KV cache needs
            // an explicit stable prefix, and the system instructions plus the
            // name-sorted tool schemas are stable for this whole run.
            status.mark_running(HarnessPhase::BuildingRequest);
            let system_end = messages
                .iter()
                .take_while(|message| matches!(message, Message::System(_)))
                .count();
            let mut prompt = crate::prompt::PromptBuilder::new();
            if system_end > 0 {
                prompt.push_system("system", messages[..system_end].to_vec());
            }
            if !tool_schemas.is_empty() {
                prompt.push_tools_segment("tools", tool_schemas.clone());
            }
            let mut request = prompt.build(messages[system_end..].to_vec());
            // Provider adapters that maintain an external conversation (for
            // example Claude Code's resumable CLI session) need the caller's
            // logical thread id, not a hash of prompt text. Carry the harness
            // thread through request metadata while preserving an explicit
            // caller-supplied value.
            if let Some(thread_id) = ctx.thread_id() {
                if request.metadata.is_null() {
                    request.metadata = serde_json::json!({
                        "thread_id": thread_id.as_str(),
                    });
                } else if let Some(metadata) = request.metadata.as_object_mut() {
                    metadata.entry("thread_id").or_insert_with(|| {
                        serde_json::Value::String(thread_id.as_str().to_string())
                    });
                }
            }
            if let Some(format) = &self.policy.default_response_format {
                request = request.with_response_format(format.clone());
            }
            if let Some(cap) = ctx.config.max_turn_output_tokens {
                request.max_tokens =
                    Some(request.max_tokens.map_or(cap, |current| current.min(cap)));
            }
            // Truncated-empty recovery: a prior attempt this turn exhausted its
            // token budget on the (hidden) reasoning channel and returned no
            // usable content, so re-issue the call with a larger cap. The boost
            // deliberately wins over the per-turn cap above — that cap is what
            // truncated the response — and was already clamped to 4x the
            // original budget when it was computed below.
            if let Some(boost) = boosted_max_tokens {
                request.max_tokens = Some(boost);
            }

            status.mark_running(HarnessPhase::Middleware);
            self.middleware
                .run_before_model(ctx, state, &mut request)
                .await?;

            // Resolve the model for the event/log name before invoking.
            // Hosted turns install their routing decision against this live
            // `RunContext`; explicit-model SDK calls continue to resolve only
            // through the local registry. Context-instance identity keeps two
            // same-id concurrent runs from borrowing each other's model.
            let binding = if let Some(binding) = self.resolve_host_model(ctx, &request).await? {
                binding
            } else {
                self.models
                    .resolve_request(&request, None, None)
                    .ok_or_else(|| {
                        TinyAgentsError::ModelNotFound(
                            request
                                .model
                                .clone()
                                .unwrap_or_else(|| "<default>".to_string()),
                        )
                    })?
            };
            let model_name = binding.resolved.name.clone();

            // An explicit request override that resolution skipped (unknown
            // name, missing capability, or provider-retired) falls through to
            // a lower-priority candidate by documented fail-closed semantics;
            // surface that fall-through as a diagnostic event instead of
            // silently substituting a different model.
            if let Some(requested) = &request.model
                && binding.resolved.source
                    != tinyinference_llm::model::ModelResolutionSource::RequestOverride
            {
                ctx.emit(AgentEvent::ModelOverrideSkipped {
                    requested: requested.clone(),
                    resolved: model_name.clone(),
                });
            }

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
                                let fallback_schema = ToolSchema {
                                    name: name.clone(),
                                    description: format!("Return the result as `{name}`."),
                                    parameters: schema.clone(),
                                    format: tinyinference_llm::tool::ToolFormat::Json,
                                };
                                // This schema is generated here, after the
                                // direct and bridge schemas above were
                                // prepared for the target provider, so it
                                // needs the same projection or it reaches the
                                // wire raw (see the `tool_schemas` and bridge
                                // preparation above).
                                request.tools.push(match &self.policy.tool_schemas {
                                    Some(preparation) => crate::tool::prepare_tool_schema(
                                        &fallback_schema,
                                        preparation,
                                    ),
                                    None => fallback_schema,
                                });
                                // Force the schema tool **only** when it is the
                                // sole tool available. Forcing it inside a
                                // tool-using loop makes the model emit the
                                // structured call on turn 1, which terminates
                                // the loop before any registered tool can ever
                                // run — the agent silently loses its tools, and
                                // the symptom points nowhere near this code.
                                // LangChain likewise binds a schema tool with a
                                // forced `tool_choice` only in its terminal
                                // wrapper, never in the tool-calling loop.
                                if tool_schemas.is_empty() {
                                    request.tool_choice = ToolChoice::Tool(name.clone());
                                } else {
                                    tracing::debug!(
                                        target: "tinyagents::agent_loop",
                                        run_id = %ctx.run_id(),
                                        schema_name = %name,
                                        registered_tools = tool_schemas.len(),
                                        "[agent_loop] structured tool offered but not forced; \
                                         registered tools stay callable"
                                    );
                                }
                            }
                        }
                        Some((strategy, name, schema))
                    }
                    Some(ResponseFormat::JsonSchema { name, schema }) => {
                        Some((StructuredStrategy::ProviderSchema, name, schema))
                    }
                    _ => None,
                };

            // A host budget is acquired only for an explicit host-driven run.
            // Do it after structured-output planning: a synthetic schema tool
            // is part of the provider request and must be included in its
            // estimate. The permit remains alive through response accounting,
            // so cancellation or a provider error still releases it through
            // Drop.
            let host_budget = if let Some(host_run) =
                crate::runtime::host_invocation_binding::<State, Ctx>(ctx)?
            {
                if let Some(budget) = host_run.host.budget.clone() {
                    let context_state = crate::host::ContextState {
                        message_count: request.messages.len(),
                        prompt_tokens: crate::token_estimation::estimate_slice_tokens(
                            &request.messages,
                        ),
                        context_window_tokens: binding
                            .model
                            .profile()
                            .and_then(|profile| profile.max_input_tokens),
                        iterations: run.steps,
                    };
                    let hint = budget.compression_hint(&context_state);
                    if hint.is_advised() {
                        tracing::debug!(?hint, "[host] budget gate advised context compression");
                        apply_host_budget_compression(ctx, &mut request.messages, hint)?;
                    }
                    let estimate = crate::host::CallEstimate::new(
                        &model_name,
                        crate::token_estimation::estimate_slice_tokens(&request.messages),
                        request.max_tokens.unwrap_or_default() as u64,
                    )
                    .with_agent(host_run.agent_id.clone())
                    .with_thread(
                        ctx.thread_id()
                            .cloned()
                            .unwrap_or_else(|| ctx.run_id().as_str().into()),
                    )
                    .with_tool_count(request.tools.len());
                    let permit = match self.call_budget(ctx) {
                        Some(remaining) => tokio::select! {
                            biased;
                            _ = ctx.cancellation.cancelled() => {
                                return Err(TinyAgentsError::Cancelled);
                            }
                            acquired = tokio::time::timeout(remaining, budget.acquire(&estimate)) => {
                                acquired.map_err(|_| TinyAgentsError::Timeout(format!(
                                    "budget admission for run `{}` exceeded its remaining wall-clock deadline",
                                    ctx.run_id()
                                )))??
                            }
                        },
                        None => tokio::select! {
                            biased;
                            _ = ctx.cancellation.cancelled() => {
                                return Err(TinyAgentsError::Cancelled);
                            }
                            acquired = budget.acquire(&estimate) => acquired?,
                        },
                    };
                    Some((budget.clone(), permit))
                } else {
                    None
                }
            } else {
                None
            };
            let request_has_tools = !request.tools.is_empty();

            let call_id = CallId::new(format!("{}-model-{}", ctx.run_id(), run.model_calls + 1));
            status.mark_running(HarnessPhase::Model);
            status.active_model_call = Some(call_id.clone());
            // Mirrored onto the context so `ModelMiddleware` (e.g.
            // `RetryMiddleware`) can correlate its own events with the exact
            // call id the loop uses instead of deriving an uncorrelated one
            // (I-7). Cleared right after the wrap onion returns, below.
            ctx.active_model_call = Some(call_id.clone());
            // Captured here (where the call actually starts) so the completed
            // event carries a real start time for duration-aware exporters.
            let model_started_at_ms = crate::ids::now_ms();
            let record = ctx.emit(AgentEvent::ModelStarted {
                call_id: call_id.clone(),
                model: model_name.clone(),
            });
            status.set_last_event(record.id);

            // Captured before `binding.model` moves into `base` below: decides
            // whether text-dialect recovery should even be attempted for this
            // call's response (see the call site after the model returns).
            let text_dialect_recovery_enabled = match self.policy.text_dialect_recovery {
                crate::runtime::TextDialectRecovery::Off => false,
                crate::runtime::TextDialectRecovery::On => true,
                crate::runtime::TextDialectRecovery::Auto => !binding
                    .model
                    .profile()
                    .map(|profile| profile.tool_calling)
                    .unwrap_or(false),
            };

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
                required_capabilities: request.required_capabilities.clone(),
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
            // Snapshot the effective token cap before `request` moves into the
            // model-wrap onion, so truncated-empty recovery can compute the next
            // (doubled) budget from what was actually sent.
            let attempt_max_tokens = request.max_tokens;
            let mut response = self
                .middleware
                .run_wrapped_model(ctx, state, request, &base)
                .await?
                .into_response();

            // Providers occasionally put a text-dialect call in visible
            // content even when a native tool channel was offered. Use the
            // canonical TinyTools-Agent parser rather than the retired
            // harness prompt parser, and only recover when the provider did
            // not already supply structured calls. Gated by
            // `RunPolicy::text_dialect_recovery` (computed above, before the
            // resolved model moved into the wrap onion).
            recover_text_dialect_calls(
                ctx,
                &mut response,
                &call_id,
                request_has_tools,
                text_dialect_recovery_enabled,
            );

            // Account for the completed provider response before fallible
            // response middleware. A middleware rejection must not erase
            // usage already incurred, and the host admission permit covers
            // provider work rather than post-processing.
            run.model_calls += 1;
            run.steps += 1;
            status.model_calls = run.model_calls;
            status.active_model_call = None;
            ctx.active_model_call = None;
            // A cache replay consumed no provider tokens, so folding its usage
            // into the run's totals reports spend that never happened. The
            // saving is surfaced through the cache-hit event instead of being
            // buried in the spend total.
            if let Some(usage) = response.usage {
                if response.served_from_cache {
                    tracing::debug!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        call_id = %call_id,
                        saved_input_tokens = usage.input_tokens,
                        saved_output_tokens = usage.output_tokens,
                        "[agent_loop] cache-served response; usage not billed to the run"
                    );
                } else {
                    run.usage.record(usage);
                    status.usage = run.usage;
                    let record = ctx.emit(AgentEvent::UsageRecorded { usage });
                    status.set_last_event(record.id);
                }
                if !response.served_from_cache
                    && let Some((budget, _permit)) = &host_budget
                    && let Err(error) = self.record_host_usage(ctx, budget, &usage).await
                {
                    let record = ctx.emit(AgentEvent::ModelFailed {
                        call_id: call_id.clone(),
                        model: model_name.clone(),
                        started_at_ms: Some(model_started_at_ms),
                        attempts: None,
                        error: error.to_string(),
                    });
                    status.set_last_event(record.id);
                    return Err(error);
                }
            }
            // The permit guards a provider call, not the tools it may request.
            // Keeping a parent permit while awaiting a sub-agent tool can
            // deadlock a one-slot gate: the child needs that same slot for its
            // model call while the parent waits for the child tool to return.
            drop(host_budget);
            status.mark_running(HarnessPhase::Middleware);
            self.middleware
                .run_after_model(ctx, state, &mut response)
                .await?;
            let captured_output = self
                .policy
                .capture
                .model_io
                .then(|| serde_json::to_value(&response.message).unwrap_or(Value::Null));
            let record = ctx.emit(AgentEvent::ModelCompleted {
                call_id: call_id.clone(),
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
            if let Some(exit) = self.apply_pending_control(ctx, run, status, messages)? {
                return Ok(exit);
            }

            let tool_calls = response.tool_calls().to_vec();

            // A tool-call structured-output strategy produces an artificial tool
            // call that is not a registered tool, so split the turn's calls into
            // the schema call(s) and the genuine ones. Treating "any call
            // matched the schema name" as terminal silently dropped every
            // sibling call in the same turn — a turn returning
            // `[search(...), my_schema(...)]` broke out with `search` never
            // executed and no event to say so.
            let structured_call_name = match &structured_plan {
                Some((StructuredStrategy::ToolCall, name, _)) => Some(name.clone()),
                _ => None,
            };
            let (structured_hits, real_tool_calls): (Vec<ToolCall>, Vec<ToolCall>) =
                match &structured_call_name {
                    Some(name) => tool_calls
                        .iter()
                        .cloned()
                        .partition(|call| &call.name == name),
                    None => (Vec::new(), tool_calls.clone()),
                };
            let structured_tool_hit = !structured_hits.is_empty();

            if structured_tool_hit && !real_tool_calls.is_empty() {
                // Record the structured payload the model already produced,
                // then run the real tools it asked for in the same turn and let
                // the loop continue; the model finishes on a later turn.
                if let Some((strategy, name, schema)) = &structured_plan {
                    let extractor =
                        StructuredExtractor::new(*strategy, name.clone(), schema.clone());
                    match extractor.extract(&response) {
                        Ok(output) => run.structured = Some(output.value),
                        Err(error) => tracing::debug!(
                            target: "tinyagents::agent_loop",
                            run_id = %ctx.run_id(),
                            %error,
                            "[agent_loop] structured extraction failed on a mixed turn; \
                             continuing with the real tool calls"
                        ),
                    }
                }
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "structured_with_tool_calls".to_string(),
                    detail: format!(
                        "structured output recorded alongside {} real tool call(s); \
                         the run continues",
                        real_tool_calls.len()
                    ),
                });
                status.set_last_event(record.id);

                // Every requested `tool_call_id` must be answered or the
                // transcript is malformed for the next provider call.
                for call in &structured_hits {
                    messages.push(Message::tool(
                        call.id.clone(),
                        "Structured output recorded. Continue with the remaining tool calls.",
                    ));
                }

                reset_truncated_empty_recovery(
                    &mut truncated_empty_retries_used,
                    &mut boosted_max_tokens,
                    &mut truncation_base,
                );

                status.mark_running(HarnessPhase::Tools);
                self.execute_tools(state, ctx, run, status, messages, real_tool_calls)
                    .await?;

                // Safe checkpoint: a control requested from `after_tool` /
                // `wrap_tool` is honored here, at the edge it was raised on.
                if let Some(exit) = self.apply_pending_control(ctx, run, status, messages)? {
                    return Ok(exit);
                }
                continue;
            }

            if real_tool_calls.is_empty() {
                // Truncated-empty recovery (runs before structured extraction,
                // which would otherwise fail on the empty completion). A local
                // reasoning model can burn the whole token budget on its hidden
                // reasoning channel and return `finish_reason == "length"` with
                // no visible text, no tool calls, and no structured output — a
                // result useless to every caller. Retry the call (bumping the
                // token budget when one was set) instead of surfacing the blank.
                // A structured tool hit carries a real payload, so it is never
                // treated as truncated-empty.
                let truncated_empty = tool_calls.is_empty()
                    && response.finish_reason.as_deref() == Some("length")
                    && response.text().trim().is_empty();
                if truncated_empty
                    && truncated_empty_retries_used < self.policy.truncated_empty_retries
                {
                    // Drop the useless empty assistant row appended above so the
                    // retry re-sends the identical transcript.
                    messages.pop();
                    truncated_empty_retries_used += 1;
                    // Grow the token budget when the request set one: double it,
                    // clamped at 4x the original cap. An unset budget stays unset
                    // (a plain retry is still worthwhile — the failure is
                    // stochastic).
                    if let Some(sent) = attempt_max_tokens {
                        let base = *truncation_base.get_or_insert(sent);
                        let next = boosted_max_tokens
                            .unwrap_or(sent)
                            .saturating_mul(2)
                            .min(base.saturating_mul(4));
                        boosted_max_tokens = Some(next);
                    }
                    let record = ctx.emit(AgentEvent::RetryScheduled {
                        call_id: call_id.clone(),
                        attempt: truncated_empty_retries_used as usize,
                    });
                    status.set_last_event(record.id);
                    continue;
                }

                // This turn resolved without scheduling a truncated-empty
                // retry, so the recovery state must not leak into later turns:
                // a stale `boosted_max_tokens` would override the caller's
                // per-turn cap on every subsequent call, and a spent retry
                // counter would deny recovery to a later turn that needs it.
                reset_truncated_empty_recovery(
                    &mut truncated_empty_retries_used,
                    &mut boosted_max_tokens,
                    &mut truncation_base,
                );

                // The model says it is not finished (`ModelResponse::continue_turn`).
                // Hand the floor back and ask for another reply instead of taking
                // this response as the turn's answer. Checked after truncated-empty
                // recovery — a truncated response is broken, not a deliberate
                // continue — and before structured extraction, which would treat it
                // as terminal.
                //
                // The assistant row is already on `messages` (appended above), so
                // only the nudge is needed. `max_model_calls` bounds the resulting
                // loop exactly as it bounds a tool-calling one.
                if !structured_tool_hit && let Some(nudge) = response.continue_turn.clone() {
                    messages.push(Message::user(nudge));
                    continue;
                }

                // Final response: optionally extract structured output using the
                // resolved plan (provider-native schema or tool-call arguments).
                if let Some((strategy, name, schema)) = &structured_plan {
                    let extractor =
                        StructuredExtractor::new(*strategy, name.clone(), schema.clone());
                    let output = extractor.extract(&response)?;
                    run.structured = Some(output.value);
                }
                // An empty provider completion — no text, no tool calls, and no
                // structured output — must not silently become the terminal
                // answer (openhuman#4638). When the policy opts in, drop the
                // empty assistant row appended above and fail with a typed error
                // so the caller can re-prompt instead of returning a blank
                // success. Gated off by default to preserve callers that rely on
                // empty finals.
                if self.policy.error_on_empty_response
                    && run.structured.is_none()
                    && tool_calls.is_empty()
                    && response.text().trim().is_empty()
                {
                    messages.pop();
                    return Err(TinyAgentsError::EmptyResponse);
                }
                run.final_response = Some(response);
                return Ok(LoopExit::Finished);
            }

            // A tool-calling response is a resolved turn too: clear the
            // recovery state before the tools run so the next turn starts from
            // the caller's configured cap and a full retry budget.
            reset_truncated_empty_recovery(
                &mut truncated_empty_retries_used,
                &mut boosted_max_tokens,
                &mut truncation_base,
            );

            // Execute requested tools: serial admission -> serial or
            // concurrent execution -> ordered fold. Multi-call turns run
            // concurrently when no tool-wrap middleware is registered; see
            // `agent_loop/tools.rs` for the dispatch rules and the semantics
            // preserved in each mode.
            status.mark_running(HarnessPhase::Tools);
            self.execute_tools(state, ctx, run, status, messages, real_tool_calls)
                .await?;

            // Safe checkpoint: honor a control requested from `after_tool` /
            // `wrap_tool` at the edge it was raised on, rather than a model
            // call later.
            if let Some(exit) = self.apply_pending_control(ctx, run, status, messages)? {
                return Ok(exit);
            }
        }
    }

    /// Drains any pending [`MiddlewareControl`] and turns it into a loop
    /// decision.
    ///
    /// Returns `Ok(None)` when nothing was requested, `Ok(Some(exit))` when the
    /// loop must stop, and `Err` for
    /// [`MiddlewareControl::Interrupt`]. Called at every safe checkpoint — the
    /// top of an iteration, after the model call, and after tool execution — so
    /// a control raised anywhere in a turn takes effect on that turn.
    fn apply_pending_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
    ) -> Result<Option<LoopExit>> {
        let Some(control) = ctx.take_control() else {
            return Ok(None);
        };
        let record = ctx.emit(AgentEvent::ControlApplied {
            control: control.kind().to_string(),
            detail: match &control {
                MiddlewareControl::StopWithFinal(text) => text.clone(),
                MiddlewareControl::Interrupt { node, message } => format!("{node}: {message}"),
            },
        });
        status.set_last_event(record.id);
        match control {
            MiddlewareControl::StopWithFinal(text) => {
                // The most recently appended assistant row may carry
                // `tool_calls` that were never answered — e.g. a middleware
                // requesting `StopWithFinal` right after the model turn that
                // requested them, before `execute_tools` ever ran. Left as
                // is, `run.messages`/`messages` end with an assistant row
                // whose tool calls have no matching tool message, which a
                // provider rejects (400) if the transcript is ever replayed
                // (M-1). Append a synthetic tool result for each unanswered
                // call so the transcript stays replayable.
                Self::close_unanswered_tool_calls(
                    messages,
                    "run stopped before this tool call was executed",
                );
                run.final_response = Some(ModelResponse::assistant(text));
                Ok(Some(LoopExit::Finished))
            }
            MiddlewareControl::Interrupt { node, message } => {
                Err(TinyAgentsError::Interrupted { node, message })
            }
        }
    }

    /// Appends a synthetic [`Message::tool`] result for every tool call on
    /// the last message that is still unanswered, so the transcript stays
    /// replayable through a provider that requires every `tool_calls` entry
    /// on an assistant message to have a matching tool result before the next
    /// turn (M-1). A no-op when the last message is not an unanswered
    /// assistant tool-call row.
    fn close_unanswered_tool_calls(messages: &mut Vec<Message>, reason: &str) {
        let Some(Message::Assistant(last)) = messages.last() else {
            return;
        };
        if last.tool_calls.is_empty() {
            return;
        }
        let synthetic: Vec<Message> = last
            .tool_calls
            .iter()
            .map(|call| Message::tool(call.id.clone(), reason))
            .collect();
        messages.extend(synthetic);
    }

    /// Resolves the effective response-cache decision for `request`.
    ///
    /// Returns `Some((cache, key))` when a [`ResponseCache`] is attached to the
    /// harness *and* caching is enabled for this call. The per-request
    /// [`ModelRequest::cache_policy`] takes precedence over the harness-level
    /// [`RunPolicy::cache`][crate::runtime::RunPolicy]; when the request
    /// carries no policy the run policy's
    /// [`response_cache_enabled`][crate::cache::CachePolicy] decides.
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

    /// Records realised provider usage without allowing host accounting I/O to
    /// outlive a cancelled or deadline-expired run. The local run totals are
    /// updated before this call, so a host-recording failure never erases spend
    /// that the provider has already incurred.
    async fn record_host_usage(
        &self,
        ctx: &RunContext<Ctx>,
        budget: &Arc<dyn crate::host::BudgetGate>,
        usage: &tinyinference_llm::usage::Usage,
    ) -> Result<()> {
        let cancellation = ctx.cancellation.clone();
        let recording = budget.record(usage);
        match self.call_budget(ctx) {
            Some(remaining) => tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(TinyAgentsError::Cancelled),
                result = tokio::time::timeout(remaining, recording) => result.map_err(|_| TinyAgentsError::Timeout(format!(
                    "budget usage recording for run `{}` exceeded its remaining wall-clock deadline",
                    ctx.run_id()
                )))?,
            },
            None => tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(TinyAgentsError::Cancelled),
                result = recording => result,
            },
        }
    }
}

/// Refreshes the harness-owned stable-prefix annotation at model-call dispatch.
///
/// Lifecycle and wrap middleware may add or rewrite leading system messages.
/// The request builder initially fingerprints those messages together with the
/// tool schemas, but the provider prompt-cache key is derived only after every
/// middleware layer has delegated to the innermost call. Rebuilding that
/// annotation there keeps cache routing tied to the bytes sent to the provider.
pub(super) fn refresh_prompt_cache_fingerprint(request: &mut ModelRequest) {
    let system_end = request
        .messages
        .iter()
        .take_while(|message| matches!(message, Message::System(_)))
        .count();
    let harness_layout = request.cache_segments.is_empty()
        || request.cache_segments.iter().all(|segment| {
            segment.cacheable
                && ((segment.id == "system" && segment.role == SegmentRole::System)
                    || (segment.id == "tools" && segment.role == SegmentRole::Tools))
        });

    if harness_layout {
        request.cache_segments.clear();
        if system_end > 0 {
            request.cache_segments.push(PromptSegment {
                id: "system".to_string(),
                role: SegmentRole::System,
                cacheable: true,
            });
        }
        if !request.tools.is_empty() {
            request.cache_segments.push(PromptSegment {
                id: "tools".to_string(),
                role: SegmentRole::Tools,
                cacheable: true,
            });
        }
        if request.cache_segments.is_empty() {
            request.prompt_fingerprint = None;
            return;
        }

        let mut prompt = crate::prompt::PromptBuilder::new();
        if system_end > 0 {
            prompt.push_system("system", request.messages[..system_end].to_vec());
        }
        if !request.tools.is_empty() {
            prompt.push_tools_segment("tools", request.tools.clone());
        }
        request.prompt_fingerprint = prompt.build(Vec::new()).prompt_fingerprint;
        return;
    }

    // Custom segment annotations do not carry message boundaries, so the
    // harness cannot safely rebuild their stable-prefix projection. Preserve
    // middleware ownership and use a conservative digest over the full
    // request instead: it sacrifices tail-only reuse but prevents distinct
    // prefixes from sharing a provider routing key.
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(crate::cache::cache_key(request));
    hasher.update(serde_json::to_vec(&request.cache_segments).unwrap_or_default());
    let fingerprint = hasher.finalize();
    request.prompt_fingerprint = Some(
        fingerprint
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    );
}

/// Applies a host budget hint before the provider sees the request.
///
/// This deliberately uses the harness's pairing-safe generic context reducer
/// instead of a host-specific transcript rewrite. `Soft` preserves the most
/// recent half of a multi-turn conversation (and all system messages) when it
/// can make progress. `Hard` uses a token budget and refuses a request that
/// cannot be reduced without discarding its whole conversational payload.
/// Both outcomes are observable through the canonical `context.compressed`
/// event so hosts can correlate a budget decision with the actual request.
fn apply_host_budget_compression<Ctx>(
    ctx: &mut RunContext<Ctx>,
    messages: &mut Vec<Message>,
    hint: crate::host::CompressionHint,
) -> Result<()> {
    use crate::host::CompressionHint;
    use crate::summarization::{TrimStrategy, trim_messages};

    let from_tokens = crate::token_estimation::estimate_slice_tokens(messages);
    let non_system = messages
        .iter()
        .filter(|message| !matches!(message, Message::System(_)))
        .count();
    let reduced = match hint {
        CompressionHint::None => return Ok(()),
        // Preserve a recent working window without perturbing a short prompt.
        CompressionHint::Soft if non_system < 3 => return Ok(()),
        CompressionHint::Soft => {
            trim_messages(messages, &TrimStrategy::KeepLast((non_system / 2).max(1)))
        }
        // A hard hint must create real headroom without ever treating system
        // instructions as expendable. The token trimmer removes oldest
        // conversational messages, preserves every system message verbatim,
        // and clears an orphaned tool-result prefix after its owning assistant
        // call was evicted.
        CompressionHint::Hard => crate::summarization::trim_messages_to_token_budget_with(
            messages,
            crate::summarization::TokenTrimPolicy::strict((from_tokens / 2).max(1))
                .preserve_system()
                .drop_leading_orphan_tools(),
            crate::token_estimation::estimate_message_tokens,
        ),
    };
    let to_tokens = crate::token_estimation::estimate_slice_tokens(&reduced);
    let has_conversation = reduced
        .iter()
        .any(|message| !matches!(message, Message::System(_)));
    if to_tokens >= from_tokens || reduced.is_empty() || (hint.is_required() && !has_conversation) {
        if hint.is_required() {
            return Err(TinyAgentsError::Validation(
                "host budget requires reducible conversational context before provider call".into(),
            ));
        }
        return Ok(());
    }
    *messages = reduced;
    ctx.emit(AgentEvent::Compressed {
        from_tokens,
        to_tokens,
    });
    Ok(())
}

/// Whether `recover_text_dialect_calls` should even attempt to parse `text`.
///
/// Fenced code blocks are always skipped regardless of
/// [`crate::runtime::TextDialectRecovery`]: a model demonstrating
/// `<tool_call>` syntax inside a ``` fence — explaining the format, echoing a
/// worked example — is manifestly not making a call, and recovering it would
/// silently execute quoted documentation as a real action.
fn text_dialect_markup_only_in_fenced_code(text: &str) -> bool {
    let mut in_fence = false;
    let mut saw_marker_outside_fence = false;
    let mut saw_marker_anywhere = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if line.contains("<tool_call") {
            saw_marker_anywhere = true;
            if !in_fence {
                saw_marker_outside_fence = true;
            }
        }
    }
    saw_marker_anywhere && !saw_marker_outside_fence
}

/// Recovers XML/text-dialect calls through `tinytools-agent` while preserving
/// every non-text provider content block (notably reasoning blocks).
///
/// Gated by [`crate::runtime::RunPolicy::text_dialect_recovery`] (`enabled`
/// is the caller's resolved decision from that policy — see the call site in
/// `run_loop_body`) and always skips markup that appears only inside a fenced
/// code block. Emits [`AgentEvent::ControlApplied`] when it actually rewrites
/// the response, so the recovery is auditable rather than a silent transform.
fn recover_text_dialect_calls<Ctx>(
    ctx: &RunContext<Ctx>,
    response: &mut tinyinference_llm::model::ModelResponse,
    model_call_id: &CallId,
    has_tools: bool,
    enabled: bool,
) {
    if !enabled || !has_tools || !response.message.tool_calls.is_empty() {
        return;
    }

    use tinytools_agent::dialect::{DialectResponse, ToolDialect, XmlDialect};

    let text = response.text();
    if text_dialect_markup_only_in_fenced_code(&text) {
        return;
    }

    let dialect_response = DialectResponse {
        text: Some(text),
        tool_calls: Vec::new(),
    };
    let (cleaned, parsed) = XmlDialect.parse_response(&dialect_response);
    if parsed.is_empty() {
        return;
    }

    ctx.emit(AgentEvent::ControlApplied {
        control: "text_dialect_recovered".to_string(),
        detail: format!(
            "recovered {} text-dialect tool call(s) from model call `{model_call_id}`",
            parsed.len()
        ),
    });

    response.message.tool_calls = parsed
        .into_iter()
        .enumerate()
        .map(|(position, call)| {
            ToolCall::new(
                call.id
                    .unwrap_or_else(|| format!("{model_call_id}-tool-{}", position + 1)),
                call.name,
                call.arguments,
            )
        })
        .collect();

    let mut inserted = false;
    response.message.content = response
        .message
        .content
        .drain(..)
        .filter_map(|block| match block {
            tinyinference_llm::message::ContentBlock::Text(_) if !inserted => {
                inserted = true;
                (!cleaned.is_empty())
                    .then(|| tinyinference_llm::message::ContentBlock::Text(cleaned.clone()))
            }
            tinyinference_llm::message::ContentBlock::Text(_) => None,
            other => Some(other),
        })
        .collect();
    if !inserted && !cleaned.is_empty() {
        response
            .message
            .content
            .push(tinyinference_llm::message::ContentBlock::Text(cleaned));
    }
}

/// Resolves one run-scoped call cap from the per-run [`RunConfig`] value and
/// the harness-wide [`crate::runtime::RunPolicy`] value.
///
/// An explicitly-set config cap is the caller's ceiling and can only be
/// tightened by the policy (fail-closed `min`); an unset config cap leaves the
/// policy as the single source of truth, which is what lets a policy raise a
/// cap above the crate default.
fn resolve_call_cap(config_cap: Option<usize>, policy_cap: usize) -> usize {
    match config_cap {
        Some(explicit) => explicit.min(policy_cap),
        None => policy_cap,
    }
}

/// Clears the per-turn truncated-empty recovery state (see
/// [`crate::runtime::RunPolicy::truncated_empty_retries`]).
///
/// The state is scoped to a single logical turn: the boosted token cap and the
/// retry counter must not carry over into the turns that follow a recovered one.
fn reset_truncated_empty_recovery(
    retries_used: &mut u32,
    boosted_max_tokens: &mut Option<u32>,
    truncation_base: &mut Option<u32>,
) {
    *retries_used = 0;
    *boosted_max_tokens = None;
    *truncation_base = None;
}

#[cfg(test)]
mod recovery_tests {
    use super::recover_text_dialect_calls;
    use crate::context::{RunConfig, RunContext};
    use crate::ids::CallId;
    use tinyinference_llm::model::ModelResponse;

    #[test]
    fn text_dialect_markup_is_not_recovered_when_the_request_offered_no_tools() {
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
        let mut response = ModelResponse::assistant(
            "<tool_call><name>shell</name><arguments>{\"command\":\"id\"}</arguments></tool_call>",
        );

        recover_text_dialect_calls(&ctx, &mut response, &CallId::new("model-1"), false, true);

        assert!(response.message.tool_calls.is_empty());
        assert!(response.text().contains("<tool_call>"));
    }

    /// I-2 regression: even when tools were offered, `enabled = false`
    /// (what `RunPolicy::text_dialect_recovery` resolves to for a model whose
    /// profile reports native tool calling, under the default `Auto` policy)
    /// must not execute `<tool_call>` markup the model merely quoted.
    #[test]
    fn text_dialect_markup_is_not_recovered_when_the_policy_disables_it() {
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
        let mut response = ModelResponse::assistant(
            r#"<tool_call>{"name": "shell", "arguments": {"command": "id"}}</tool_call>"#,
        );

        recover_text_dialect_calls(&ctx, &mut response, &CallId::new("model-1"), true, false);

        assert!(response.message.tool_calls.is_empty());
        assert!(response.text().contains("<tool_call>"));
    }

    /// I-2 regression: a final answer that quotes `<tool_call>` markup inside
    /// a fenced code block must never be executed, even when recovery is
    /// otherwise enabled and tools were offered.
    #[test]
    fn text_dialect_markup_inside_a_fenced_code_block_is_never_recovered() {
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
        let mut response = ModelResponse::assistant(
            "Here is the format:\n```\n<tool_call>{\"name\": \"shell\", \"arguments\": {}}</tool_call>\n```\n",
        );

        recover_text_dialect_calls(&ctx, &mut response, &CallId::new("model-1"), true, true);

        assert!(
            response.message.tool_calls.is_empty(),
            "markup quoted inside a fenced code block must not become a real call"
        );
        assert!(response.text().contains("<tool_call>"));
    }

    /// Sanity check for the fenced-code-block guard: markup outside any fence
    /// is still recovered when the policy and tool offer both allow it.
    #[test]
    fn text_dialect_markup_outside_a_fenced_code_block_is_recovered() {
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
        let mut response = ModelResponse::assistant(
            r#"<tool_call>{"name": "shell", "arguments": {"command": "id"}}</tool_call>"#,
        );

        recover_text_dialect_calls(&ctx, &mut response, &CallId::new("model-1"), true, true);

        assert_eq!(response.message.tool_calls.len(), 1);
        assert_eq!(response.message.tool_calls[0].name, "shell");
    }
}
