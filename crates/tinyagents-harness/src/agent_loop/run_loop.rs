//! The core superstep loop body: `run_loop` drives one model call,
//! any requested tool calls, and repeats until the model finishes or a
//! configured limit is reached.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

use super::handoff_transform;
use super::model_call::ModelCallBase;
use super::types::{MixedStructuredTurn, ResponseTurn, TruncationOutcome, TurnFlow, TurnRecovery};
use super::*;

/// Final-turn nudge when a repeat/no-progress guard halts the run: the model
/// was stopped mid-loop, possibly with its deliverable unwritten, so it is
/// given one last turn to produce its output before the run ends.
const DELIVER_FIRST_NUDGE: &str = "You were stopped because you kept repeating the same \
steps without making progress. This is your last turn before the run ends. If you have \
enough to finish, write your final output now to the exact file or location the task \
requires, then stop. Do not re-issue calls you already have results for -- act on what \
you already have, or state plainly what is blocking you.";

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
        ctx.reset_turn_tracker(messages.len());
        let outcome = self
            .run_loop_body(state, ctx, run, status, &mut messages, streaming)
            .await;
        // Announce whatever the final turn appended and close it, on every
        // exit path, before the transcript moves onto the run.
        ctx.close_turn(self.policy.capture, &messages);
        run.messages = std::mem::take(&mut messages);
        // A4: the `Collect` lane is delivered on the run, never on the
        // transcript, and on every exit path — a host that pushed
        // observations during a run that then failed still gets them back.
        if let Some(queue) = ctx.run_queue.clone() {
            run.collected.extend(
                queue
                    .take_admitted(crate::run_queue::QueueLane::Collect, self.policy.queue_mode)
                    .await
                    .into_iter()
                    .map(|item| item.value.clone()),
            );
        }

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

        // One typed answer to "how did the loop end", derived once here so the
        // event, `run.terminal` and the legacy fields cannot disagree.
        let mut terminal = TerminalOutcome::from_loop_exit(&exit, ctx.provider_started());
        // A repeat / no-progress guard halts by pausing; its marker says so.
        if matches!(exit, LoopExit::Paused(_))
            && let Some(summary) = ctx.halted_by_guard.take()
        {
            run.failure_cause = ctx.take_halted_by_guard_cause();
            terminal =
                TerminalOutcome::halted(summary).with_provider_started(ctx.provider_started());
        }
        run.terminal = Some(terminal.clone());

        status.mark_running(HarnessPhase::Middleware);
        let after_agent = self.middleware.run_after_agent(ctx, state, run).await;
        // `after_agent` may post-process `run.messages`; announce anything it
        // appended (even if it then failed) so a mirror built from lifecycle
        // events matches the returned transcript.
        ctx.flush_transcript(self.policy.capture, &run.messages);
        after_agent?;
        // The hook may have replaced the outcome; the event reports the final one.
        let terminal = run.terminal.clone().unwrap_or(terminal);

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
                    outcome: Some(terminal),
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
            LoopExit::Deferred(requests) => {
                // Like a pause, a deferral is not a completion: the run is
                // waiting on a human decision or host-side execution for the
                // calls listed in `requests`. The transcript already carries
                // the assistant's tool-call row and every non-deferred
                // sibling's result, so persisting `run.messages` +
                // `run.deferred` is all a host needs to resume later.
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "deferred".to_string(),
                    detail: format!(
                        "{} approval(s), {} external call(s) pending",
                        requests.approvals.len(),
                        requests.calls.len()
                    ),
                });
                status.set_last_event(record.id);
                tracing::debug!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    approvals = requests.approvals.len(),
                    calls = requests.calls.len(),
                    "[agent_loop] run deferred on pending tool calls"
                );
                run.deferred = Some(requests);
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

        // Build the tool surface once (see `tool_surface.rs`): the direct tool
        // set plus the deferred catalogue behind the `tool_search` bridge. The
        // tool gate (the host allow-list plus the run's tool rules) gates both
        // halves; `resolve_tool_allowlist` underneath it (not a raw read of
        // `binding.allowed_tools`) is what applies I-9's fail-closed default,
        // so an empty declared list denies every tool.
        let gate = self.resolve_tool_gate(ctx)?;
        let mut surface = self.build_tool_surface(ctx, messages, &gate).await?;
        self.check_structured_schema_name(&surface.tool_schemas)?;

        status.mark_running(HarnessPhase::Middleware);
        self.middleware.run_before_agent(ctx, state).await?;

        // Announced after `before_agent` so a listener that subscribes there
        // (the usual place) sees the run's tool surface. This is deliberately
        // the pre-middleware/pre-request baseline (see the event's doc
        // comment): per-turn `before_model` middleware and a structured-
        // output tool-call fallback can still narrow or grow what an
        // individual request actually sends.
        let record = ctx.emit(AgentEvent::ToolsAdvertised {
            direct: surface.direct_schema_count,
            deferred: surface.deferred_catalog.len(),
            schema_bytes: crate::token_estimation::tool_schema_bytes(&surface.tool_schemas),
        });
        status.set_last_event(record.id);

        // Resume (A2): the caller supplied decisions for the tool calls a
        // previous run left pending on this transcript. Apply them — answer
        // denials and host-supplied results, run approved calls — before
        // spending a model call, so the model's next turn sees every call
        // answered. An approved call that defers *again* is settled exactly
        // like a fresh deferral below.
        if let Some(results) = ctx.take_deferred_results() {
            let pending = pending_tool_calls(messages)?;
            status.mark_running(HarnessPhase::Tools);
            let deferred = self
                .apply_deferred_results(state, ctx, run, status, messages, pending, results)
                .await?;
            if let Some(exit) = self
                .settle_deferred(state, ctx, run, status, messages, deferred)
                .await?
            {
                return Ok(exit);
            }
        }

        // Per-turn recovery state (retry/nudge counters and the boosted output
        // cap; see `TurnRecovery`). It persists across the retry `continue`
        // within a single logical turn and is reset at each turn boundary.
        let mut turn_recovery = TurnRecovery::default();

        // Output-validation retry state (see `RunPolicy::output_retry`, A3).
        // Scoped to the whole run rather than reset per turn: `max_attempts`
        // is a run-wide ceiling on re-asks, matching `retries.output` in
        // Pydantic AI rather than a per-turn allowance.
        let mut output_retry_attempts: u8 = 0;

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
                    // A repeat/no-progress guard halt (not a human/orchestrator pause)
                    // gets one deliver-first turn before it ends the run: the model was
                    // stopped mid-loop with its deliverable possibly unwritten. Nudge it
                    // to write its final output, lift the guard's pause latch, and loop
                    // once more. One-shot per run; a second guard halt ends it for real.
                    if ctx.halted_by_guard.is_some()
                        && !ctx.deliver_first_attempted
                        && ctx.limits.remaining_model_calls() > 0
                    {
                        ctx.deliver_first_attempted = true;
                        let _ = ctx.take_halted_by_guard();
                        let _ = ctx.take_halted_by_guard_cause();
                        if let Some(handle) = ctx.steering.as_ref() {
                            handle.resume();
                        }
                        messages.push(Message::user(DELIVER_FIRST_NUDGE.to_string()));
                        let record = ctx.emit(AgentEvent::ControlApplied {
                            control: "deliver_first_nudge".to_string(),
                            detail: "repeat/no-progress guard halt; granting a final turn \
                                     to write the deliverable before ending"
                                .to_string(),
                        });
                        status.set_last_event(record.id);
                        tracing::info!(
                            target: "tinyagents::agent_loop",
                            run_id = %ctx.run_id(),
                            "[agent_loop] guard halt: granting one deliver-first turn before ending"
                        );
                        continue;
                    }
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
            match self.apply_pending_control(ctx, run, status, messages)? {
                ControlEffect::None => {}
                ControlEffect::ContinueLoop => continue,
                ControlEffect::Exit(exit) => return Ok(exit),
            }

            // Fail-closed limit and deadline checks before each model call.
            ctx.clear_last_limit();
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

            // The tool-change patch must use the hosted model's placement
            // rules. Keep this decision for middleware and dispatch while
            // the routing inputs remain unchanged.
            let resolution_cache = std::sync::Arc::new(std::sync::Mutex::new(
                None::<(
                    Option<String>,
                    Vec<tinyinference_llm::model::ModelHint>,
                    Option<tinyinference_llm::model::CapabilitySet>,
                    crate::agent_loop::ModelBinding<State>,
                )>,
            ));
            // A pending steered model switch decides the model this call
            // uses, so the preview (and the tool-change patch shaped by it)
            // must be for that model, not the default.
            let patch_request = ModelRequest {
                model: self.steered_model(ctx),
                ..ModelRequest::default()
            };
            let patch_profile =
                if let Some(binding) = self.resolve_host_model(ctx, &patch_request).await? {
                    let profile = binding.model.profile().cloned();
                    *resolution_cache.lock().unwrap() = Some((
                        patch_request.model.clone(),
                        patch_request.model_hints.clone(),
                        patch_request.required_capabilities.clone(),
                        binding,
                    ));
                    profile
                } else {
                    match self.resolve_local_model(ctx, &patch_request).await {
                        Ok(binding) => {
                            let profile = binding.model.profile().cloned();
                            *resolution_cache.lock().unwrap() = Some((
                                patch_request.model.clone(),
                                patch_request.model_hints.clone(),
                                patch_request.required_capabilities.clone(),
                                binding,
                            ));
                            profile
                        }
                        Err(TinyAgentsError::ModelNotFound(_)) => None,
                        Err(error) => return Err(error),
                    }
                };

            // B6: re-consult the toolset chain and declare any live change as a
            // transcript patch, so it is part of *this* turn's request; then
            // promote tools a successful `tool_search` returned and assemble
            // the turn's wire list.
            // Announce pending appends first so a rewrite is reported against
            // the transcript it actually rewrote.
            ctx.flush_transcript(self.policy.capture, messages);
            let rewrote = surface
                .declare_toolset_changes(self, ctx, messages, &gate, patch_profile.as_ref())
                .await?;
            let rewrote = surface.promote_discovered(messages, patch_profile.as_ref()) || rewrote;
            if rewrote {
                ctx.rebase_transcript(messages.len(), "tool_change");
            }
            surface.assemble_turn_schemas();

            // Build the request from the working transcript, tool schemas, and
            // policy response format (see `model_turn.rs`).
            let mut request = self.build_turn_request(
                ctx,
                status,
                messages,
                &surface.tool_schemas,
                turn_recovery.boosted_max_tokens,
            );

            // Apply a pending `SteeringCommand::SwitchModel` before anything
            // resolves a binding, so middleware, resolution, events and the
            // handoff/budget/dialect decisions below all see the new model.
            // Re-applied after `before_model` (see below).
            let model_before_switch = request.model.clone();
            let switch_candidate = self.steered_model(ctx).and_then(|requested| {
                resolution_cache
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|(_, _, _, binding)| {
                        (binding.resolved.name == requested
                            && binding.resolved.requested.as_deref() == Some(requested.as_str()))
                        .then(|| binding.model.clone())
                    })
            });
            self.apply_steered_model_switch(
                ctx,
                &mut request,
                &model_before_switch,
                switch_candidate,
            )
            .await?;

            // Known tool requirements must shape the hosted profile seen by
            // middleware. The later gate below still catches tools added by
            // a before_model hook.
            if matches!(
                self.policy.tool_dialect,
                crate::config::ToolDispatcher::Native
            ) && (!request.tools.is_empty()
                || matches!(request.response_format, Some(ResponseFormat::Auto { .. })))
            {
                request
                    .required_capabilities
                    .get_or_insert_default()
                    .tool_calling = true;
            }

            status.mark_running(HarnessPhase::Middleware);
            ctx.model_profile = patch_profile;
            // Stage ephemeral request content before BudgetMiddleware's
            // admission estimate. The queue commits it only if the model
            // wrap onion reaches its base call.
            self.middleware
                .run_before_model_dispatch(ctx, state, &mut request)
                .await?;
            let profile_cache = resolution_cache.clone();
            self.middleware
                .run_before_model_with_profile(
                    ctx,
                    state,
                    &mut request,
                    self,
                    move |harness, ctx, request| {
                        let resolution_cache = profile_cache.clone();
                        Box::pin(async move {
                            let key = (
                                request.model.clone(),
                                request.model_hints.clone(),
                                request.required_capabilities.clone(),
                            );
                            if let Some((cached_key, cached_hints, cached_capabilities, binding)) =
                                resolution_cache.lock().unwrap().as_ref()
                                && *cached_key == key.0
                                && *cached_hints == key.1
                                && *cached_capabilities == key.2
                            {
                                return Ok(binding.model.profile().cloned());
                            }
                            if let Some(binding) = harness.resolve_host_model(ctx, request).await? {
                                let profile = binding.model.profile().cloned();
                                *resolution_cache.lock().unwrap() =
                                    Some((key.0, key.1, key.2, binding));
                                Ok(profile)
                            } else {
                                match harness.resolve_local_model(ctx, request).await {
                                    Ok(binding) => {
                                        let profile = binding.model.profile().cloned();
                                        *resolution_cache.lock().unwrap() =
                                            Some((key.0, key.1, key.2, binding));
                                        Ok(profile)
                                    }
                                    Err(TinyAgentsError::ModelNotFound(_)) => Ok(None),
                                    Err(error) => Err(error),
                                }
                            }
                        })
                    },
                )
                .await?;

            // A forced native dialect cannot silently select a model that
            // lacks provider-native tool calling. This has to happen after
            // `before_model`, because middleware may add tools, and before
            // model resolution, because the resolver is the capability gate.
            // An automatic structured response also needs this gate: its
            // fallback may become a native schema tool after selection.
            if matches!(
                self.policy.tool_dialect,
                crate::config::ToolDispatcher::Native
            ) && (!request.tools.is_empty()
                || matches!(request.response_format, Some(ResponseFormat::Auto { .. })))
            {
                request
                    .required_capabilities
                    .get_or_insert_default()
                    .tool_calling = true;
            }

            // Middleware may have chosen another model or added capability
            // requirements; the steered switch is re-validated and wins.
            let switch_candidate = self.steered_model(ctx).and_then(|requested| {
                resolution_cache
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|(_, _, _, binding)| {
                        (binding.resolved.name == requested
                            && binding.resolved.requested.as_deref() == Some(requested.as_str()))
                        .then(|| binding.model.clone())
                    })
            });
            self.apply_steered_model_switch(
                ctx,
                &mut request,
                &model_before_switch,
                switch_candidate,
            )
            .await?;

            // Safe checkpoint: a control requested from `before_model_control`
            // (for example `BudgetMiddleware` finding the budget already
            // exhausted) is honored **before** the model is actually
            // dispatched, not one billable call late. Without this checkpoint
            // the queued control would only be drained at the next one (after
            // this response comes back), spending exactly the call the
            // control was raised to prevent.
            match self.apply_pending_control(ctx, run, status, messages)? {
                ControlEffect::None => {}
                ControlEffect::ContinueLoop => continue,
                ControlEffect::Exit(exit) => return Ok(exit),
            }

            // Request reducers run after the admission estimate and may evict
            // the staged correction. Re-apply only the original staged prefix
            // after those rewrites; instructions queued during this call stay
            // pending for the next request.
            self.middleware
                .run_after_model_rewrite(ctx, state, &mut request)
                .await?;

            // Resolve the model for the event/log name before invoking.
            // Hosted turns install their routing decision against this live
            // `RunContext`; explicit-model SDK calls continue to resolve only
            // through the local registry. Context-instance identity keeps two
            // same-id concurrent runs from borrowing each other's model.
            let cached = resolution_cache.lock().unwrap().take().and_then(
                |(model, hints, capabilities, binding)| {
                    (model == request.model
                        && hints == request.model_hints
                        && capabilities == request.required_capabilities)
                        .then_some(binding)
                },
            );
            let hosted_binding = match cached {
                Some(binding) => Some(binding),
                None => self.resolve_host_model(ctx, &request).await?,
            };
            let binding = if let Some(binding) = hosted_binding {
                binding
            } else {
                self.resolve_local_model(ctx, &request).await?
            };
            ctx.model_profile = binding.model.profile().cloned();
            crate::middleware::library::rehome_ephemeral_system_instructions(
                &mut request,
                ctx.model_profile.as_ref(),
            );
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

            // Cross-provider handoff: rewrite any part of the outgoing
            // transcript that a mid-session provider/model switch left
            // unsafe to replay verbatim (foreign signed/redacted thinking,
            // non-conforming tool-call ids, unsupported images) right before
            // this request is sent. A no-op (same-origin run, the common
            // case) allocates nothing — see `handoff_transform`. Runs before
            // the schema/reasoning adjustments below so a rewritten
            // transcript (rather than the pre-handoff one) is what those
            // adjustments and the eventual request see.
            if let Some(profile) = binding.model.profile() {
                let target_origin = handoff_transform::target_origin_for(profile);
                let outcome = handoff_transform::prepare_for_model(
                    &request.messages,
                    profile,
                    &target_origin,
                );
                let changes = outcome.changes;
                if changes > 0 {
                    request.messages = outcome.messages.into_owned();
                    ctx.emit(AgentEvent::HandoffTransformApplied { changes });
                }
            }

            // Apply the resolved model's schema transform (for example
            // stripping `$defs` a provider rejects) to every tool schema
            // already attached to the request. This is the same wire-shape
            // adjustment `SchemaPreparation::schema_transform` performs, run
            // here because it depends on the resolved binding's profile,
            // which is only known once resolution above has run.
            if let Some(transform) = binding
                .model
                .profile()
                .and_then(|profile| profile.schema_transform.as_ref())
            {
                for tool in request.tools.iter_mut() {
                    tool.parameters = transform.apply(&tool.parameters);
                }
            }

            // A caller asking for a *named* reasoning effort (for example
            // `ReasoningEffort::High`) gets whatever generic token that name
            // implies unless the resolved model's profile maps that name to
            // something more specific for this exact model (a provider-tuned
            // `budget_tokens`, typically). Only fill in a name the profile
            // actually maps and only when the caller has not already pinned
            // an explicit `budget_tokens` — an explicit budget is the
            // caller's own override and must win over the profile default.
            // The run policy's reasoning default (a host's "thinking level")
            // fills in only where nothing upstream already chose one, and is
            // attached before the profile mapping below so a named effort
            // still gets the resolved model's tuned config.
            apply_default_reasoning(&mut request, self.policy.default_reasoning.as_ref());
            if let Some(profile) = binding.model.profile()
                && let Some(reasoning) = request.reasoning.as_ref()
                && reasoning.budget_tokens.is_none()
                && let Some(effort) = reasoning.effort
                && let Some(mapped) = profile.thinking_level_map.get(effort.as_str())
            {
                request.reasoning = Some(mapped.clone());
            }
            // The call after a dead one (`RunPolicy::truncated_empty_reasoning_off`)
            // goes out without reasoning, after the default and the profile
            // mapping so neither puts it back. One call only, unless dead calls
            // kept coming and reasoning-off is now sticky for the run.
            let dead_call_retry = std::mem::take(&mut turn_recovery.reasoning_off_next);
            ctx.dead_call_retry = dead_call_retry;
            if dead_call_retry || turn_recovery.reasoning_off_sticky {
                tracing::info!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    "[agent_loop] call after a dead one goes out with reasoning off"
                );
                request.reasoning = Some(tinyinference_llm::model::ReasoningConfig::effort(
                    tinyinference_llm::model::ReasoningEffort::None,
                ));
            }

            // Resolve the structured-output plan against the resolved model (see
            // `structured_plan.rs`); the plan drives extraction of the final
            // response below. `synthesized_tools` are the schema tools the plan
            // pushed onto the request.
            let (structured_plan, synthesized_tools) = self.plan_structured_output(
                ctx,
                &mut request,
                binding.model.profile(),
                surface.tool_schemas.len(),
            );

            // What was offered is fixed here, before a text dialect strips
            // the schemas off the wire: recovery and the stream scrubber need
            // the names, and the structured-output schema tool counts. The
            // registry is extended (not just the run-level one) so a
            // per-turn synthetic tool — the structured-output fallback
            // schema just pushed above — has a positional layout to decode
            // a recovered call against; the catalogue already advertises it
            // because it is rendered fresh from `tools` on every call.
            let offered_tool_count = request.tools.len();
            // Whether this turn could possibly have accepted a tool call at
            // all: tools were offered and the effective choice is not
            // `None`. Read below by the dropped-tool-call nudge: nudging a
            // model to "issue the call" when no call could ever have been
            // accepted wastes up to `dropped_tool_call_nudges` model calls
            // asking for something impossible before falling through. Also
            // gates whether `recovery` below is populated at all: an empty
            // recovery makes every grammar in `tinytools-agent` decline to
            // recognize anything as a call, so a model that narrated
            // `<tool_call>`-shaped markup as plain text while explicitly
            // told not to call anything is never misread as a real,
            // side-effecting call.
            let tools_available_this_turn =
                offered_tool_count > 0 && request.tool_choice != ToolChoice::None;
            let dialect = super::dialect::RunDialect::resolve(
                self.policy.tool_dialect,
                &request.tools,
                binding.model.profile().map(|profile| profile.tool_calling),
            );
            let forced_text_dialect = dialect.is_text();
            // Whether text-dialect markup may become a call on this turn. A
            // forced text dialect always recovers regardless of this flag —
            // the model can only answer in text, so parsing it is the
            // protocol, not a fallback. Under `Native`, `Auto` skips a model
            // whose resolved profile reports native tool calling, since such
            // a model that still answered in prose was explaining or quoting
            // the format, not making a call (I-2).
            let text_dialect_recovery_enabled = match self.policy.text_dialect_recovery {
                crate::runtime::TextDialectRecovery::Off => false,
                crate::runtime::TextDialectRecovery::On => true,
                crate::runtime::TextDialectRecovery::Auto => !binding
                    .model
                    .profile()
                    .map(|profile| profile.tool_calling)
                    .unwrap_or(false),
            };
            // That native model still emits its own chat-template tokens
            // (DeepSeek DSML, `<|tool_call>`) when it narrates a real call,
            // and quoted prose does not contain them. So under `Auto` the
            // stream scrubber keeps recovering those and nothing else; `Off`
            // recovers nothing on either path (#6733).
            let special_tokens_only = !forced_text_dialect
                && !text_dialect_recovery_enabled
                && self.policy.text_dialect_recovery == crate::runtime::TextDialectRecovery::Auto;
            let recovery = if tools_available_this_turn
                && (forced_text_dialect || text_dialect_recovery_enabled || special_tokens_only)
            {
                super::dialect::TextRecovery {
                    offered: Arc::new(request.tools.clone()),
                    registry: dialect.registry_for(&request.tools),
                    dropped: Arc::default(),
                    withhold: false,
                    special_tokens_only,
                }
            } else if tools_available_this_turn {
                // Recovery is off: markup stays text and never becomes a call.
                super::dialect::TextRecovery::default()
            } else {
                // Nothing can be recovered as a call, but a call the model
                // writes anyway is kept out of the answer (see
                // `TextRecovery::withholding`).
                super::dialect::TextRecovery::withholding()
            };
            // Applied before budget preflight below: for a text dialect this
            // rewrite folds the protocol block and full tool catalogue into
            // `request.messages` and clears `request.tools`, and that is the
            // request whose size the budget estimate has to reflect. Doing
            // this after preflight (as before) let a prompt near
            // `max_input_tokens` pass admission on the small structured
            // request and then send a materially larger rendered-text one,
            // defeating the pre-call budget limit.
            dialect.apply_to_request(
                &mut request,
                self.policy.host_renders_tool_catalogue,
                &synthesized_tools,
            );

            // A host budget is acquired only for an explicit host-driven run (see
            // `host_budget.rs`), after structured-output planning so a synthetic
            // schema tool is part of the estimate.
            let host_budget = self
                .admit_host_budget(
                    ctx,
                    run,
                    &mut request,
                    binding.model.profile(),
                    &model_name,
                    offered_tool_count,
                )
                .await?;
            let call_id = CallId::new(format!("{}-model-{}", ctx.run_id(), run.model_calls + 1));
            status.mark_running(HarnessPhase::Model);
            status.active_model_call = Some(call_id.clone());
            // Mirrored onto the context so `ModelMiddleware` (e.g.
            // `RetryMiddleware`) can correlate its own events with the exact
            // call id the loop uses instead of deriving an uncorrelated one
            // (I-7). Cleared right after the wrap onion returns, below.
            ctx.active_model_call = Some(call_id.clone());
            ctx.begin_model_call();
            // Captured here (where the call actually starts) so the completed
            // event carries a real start time for duration-aware exporters.
            let model_started_at_ms = crate::ids::now_ms();
            ctx.start_turn(self.policy.capture, messages);
            let record = ctx.emit(AgentEvent::ModelStarted {
                call_id: call_id.clone(),
                model: model_name.clone(),
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
                required_capabilities: request.required_capabilities.clone(),
                shape: super::dialect::CallShape {
                    streaming,
                    recovery: recovery.clone(),
                    retry_empty_final: self.policy.empty_response_retries > 0
                        && structured_plan.is_none()
                        && run.structured.is_none(),
                },
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
            let (mut response, wrap_control) = match self
                .middleware
                .run_wrapped_model(ctx, state, request, &base)
                .await
            {
                Ok(outcome) => outcome.into_response_with_control(),
                Err(error) => {
                    self.middleware
                        .run_after_model_dispatch(ctx, state, ctx.model_base_called())
                        .await?;
                    status.active_model_call = None;
                    ctx.active_model_call = None;
                    ctx.mark_model_call_failed();
                    // A response discarded before a retry was billed even if
                    // the replacement call fails. Do not leave that usage in
                    // the context when the `?` below would skip normal
                    // response accounting.
                    self.account_discarded_usage(
                        ctx,
                        run,
                        status,
                        &call_id,
                        &model_name,
                        model_started_at_ms,
                        &host_budget,
                    )
                    .await?;
                    return Err(error);
                }
            };
            self.middleware
                .run_after_model_dispatch(ctx, state, ctx.model_base_called())
                .await?;
            // A `ModelMiddleware::wrap_model` that short-circuited with
            // `MiddlewareModelOutcome::Command` carries no real response (see
            // that variant's docs); queue its control the same way a
            // lifecycle hook's control-outcome return would, so the next safe
            // checkpoint (right below, after this turn's bookkeeping) applies
            // it instead of the placeholder response being mistaken for a
            // real completion.
            if let Some(control) = wrap_control {
                ctx.request_control(control);
            }

            // Providers occasionally put a text-dialect call in visible
            // content even when a native tool channel was offered, and a
            // forced text dialect always does. Read the response through
            // every grammar the protocol crate knows (`recover_text_calls`),
            // but only when the provider did not already supply structured
            // calls. `recovery` is already empty when this turn offered no
            // tools, the effective tool choice was `None`, or the policy is
            // `Off`; this gate also skips the `Auto` native model, whose
            // special-token calls only the stream scrubber recovers (#6733).
            // The audit event lives in the wrapper; which fenced code is
            // protected is `tinytools-agent`'s call.
            if forced_text_dialect || text_dialect_recovery_enabled {
                recover_text_dialect_calls(ctx, &mut response, &call_id, &recovery);
            }
            // A turn that could not take a call: whatever call markup the
            // model wrote is scrubbed from the answer and never run. Applied
            // whatever `text_dialect_recovery` says — that policy decides
            // whether prose can *become* a call, and here none can. A
            // streamed reply was already scrubbed delta by delta; this
            // catches a unary one.
            if recovery.withhold {
                super::dialect::withhold_text_calls(&mut response, &call_id, &recovery.dropped);
            }

            // The provider call has returned: a failure from here on (response
            // accounting, `after_model`) is after-turn, not an in-flight call.
            ctx.active_model_call = None;

            // Account for the completed provider response before fallible
            // response middleware (see `model_turn.rs`). A middleware rejection
            // must not erase usage already incurred, and the host admission
            // permit covers provider work rather than post-processing.
            self.account_model_response(
                ctx,
                run,
                status,
                &response,
                &call_id,
                &model_name,
                model_started_at_ms,
                &host_budget,
            )
            .await?;
            // The permit guards a provider call, not the tools it may request.
            // Keeping a parent permit while awaiting a sub-agent tool can
            // deadlock a one-slot gate: the child needs that same slot for its
            // model call while the parent waits for the child tool to return.
            drop(host_budget);
            status.mark_running(HarnessPhase::Middleware);
            let after_model = self
                .middleware
                .run_after_model(ctx, state, &mut response)
                .await;
            after_model?;
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
            ctx.flush_transcript(self.policy.capture, messages);

            // Safe checkpoint: honor any control outcome a middleware requested
            // during this turn (for example an early-exit tool or a budget stop
            // hook), before executing further tools.
            match self.apply_pending_control(ctx, run, status, messages)? {
                ControlEffect::None => {}
                ControlEffect::ContinueLoop => continue,
                ControlEffect::Exit(exit) => return Ok(exit),
            }

            let tool_calls = response.tool_calls().to_vec();

            // A tool-call structured-output strategy produces an artificial tool
            // call that is not a registered tool, so split the turn's calls into
            // the schema call(s) and the genuine ones. Treating "any call
            // matched the schema name" as terminal silently dropped every
            // sibling call in the same turn — a turn returning
            // `[search(...), my_schema(...)]` broke out with `search` never
            // executed and no event to say so.
            let structured_call_names: Vec<String> = match &structured_plan {
                Some((StructuredStrategy::ToolCall, name, _)) => vec![name.clone()],
                Some((StructuredStrategy::ToolCallUnion, _, _)) => {
                    match &self.policy.structured_strategy_override {
                        Some(crate::runtime::StructuredStrategyOverride::ToolCallUnion {
                            variants,
                        }) => variants.iter().map(|(n, _)| n.clone()).collect(),
                        _ => Vec::new(),
                    }
                }
                _ => Vec::new(),
            };
            let (structured_hits, real_tool_calls): (Vec<ToolCall>, Vec<ToolCall>) =
                if structured_call_names.is_empty() {
                    (Vec::new(), tool_calls.clone())
                } else {
                    tool_calls
                        .iter()
                        .cloned()
                        .partition(|call| structured_call_names.contains(&call.name))
                };
            let structured_tool_hit = !structured_hits.is_empty();

            let turn = ResponseTurn {
                call_id: &call_id,
                response: &response,
                tool_calls: &tool_calls,
                attempt_max_tokens,
                started_at_ms: model_started_at_ms,
                recovery: &recovery,
                tools_available: tools_available_this_turn,
                text_dialect_calls_recoverable: forced_text_dialect
                    || text_dialect_recovery_enabled,
                has_structured_plan: structured_plan.is_some(),
                structured_call_names: &structured_call_names,
            };
            // A length stop may have cut a tool call off mid-arguments: answer
            // the suspect calls with an error rather than running them.
            let turn_had_truncated_calls = match self
                .reject_truncated_tool_calls(
                    state,
                    ctx,
                    run,
                    status,
                    messages,
                    &mut turn_recovery,
                    &turn,
                )
                .await?
            {
                TruncationOutcome::Clean => false,
                TruncationOutcome::CallsRejected => true,
                TruncationOutcome::EndTurn(None) => continue,
                TruncationOutcome::EndTurn(Some(exit)) => return Ok(exit),
            };

            if structured_tool_hit && !real_tool_calls.is_empty() {
                // A6: one turn asked to both answer (the structured-output
                // schema call) and run further tools; `end_strategy` decides.
                let mixed = MixedStructuredTurn {
                    response,
                    structured_plan: structured_plan.as_ref(),
                    structured_hits,
                    real_tool_calls,
                    turn_had_truncated_calls,
                };
                match self
                    .finish_mixed_structured_turn(
                        state,
                        ctx,
                        run,
                        status,
                        messages,
                        &mut turn_recovery,
                        &mut surface.promoted_names,
                        mixed,
                    )
                    .await?
                {
                    TurnFlow::NextTurn => continue,
                    TurnFlow::Exit(exit) => return Ok(exit),
                }
            }

            if real_tool_calls.is_empty() {
                // Resolve unusable responses before finishing the turn. Recovery
                // may continue to the loop's existing limit check without
                // scheduling a retry when no model-call budget remains.
                if self.recover_unusable_response(
                    ctx,
                    run,
                    status,
                    messages,
                    &mut turn_recovery,
                    &turn,
                )? {
                    continue;
                }

                // This turn resolved without scheduling a truncated-empty
                // retry, so the recovery state must not leak into later turns:
                // a stale `boosted_max_tokens` would override the caller's
                // per-turn cap on every subsequent call, and a spent retry
                // counter would deny recovery to a later turn that needs it.
                turn_recovery.reset_after_final();

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
                //
                // A3: extraction failure (schema-invalid/unparseable) or a
                // registered `OutputValidator` rejecting an otherwise
                // schema-valid value with `TinyAgentsError::ModelRetry` no
                // longer immediately fails the run. Both feed the same
                // output-validation retry loop — re-ask the model with the
                // error as a repair prompt, bounded by
                // `RunPolicy::output_retry.max_attempts` — because a
                // schema-valid-but-wrong answer and a malformed one are the
                // same failure from the caller's perspective: the model needs
                // another turn to fix it.
                if let Some((strategy, name, schema)) = &structured_plan {
                    let extractor = self.build_structured_extractor(strategy, name, schema);
                    let outcome = extractor.extract_outcome(&response);
                    let variant = outcome.variant.clone();
                    let error = match outcome.value {
                        Some(value) => match &self.output_validator {
                            Some(validator) => match validator.validate(ctx, state, &value).await {
                                Ok(()) => {
                                    run.structured = Some(value);
                                    run.structured_variant = variant;
                                    None
                                }
                                Err(TinyAgentsError::ModelRetry(message)) => Some(message),
                                Err(other) => return Err(other),
                            },
                            None => {
                                run.structured = Some(value);
                                run.structured_variant = variant;
                                None
                            }
                        },
                        None => outcome.error,
                    };
                    if let Some(error) = error {
                        if output_retry_attempts < self.policy.output_retry.max_attempts {
                            output_retry_attempts += 1;
                            let record = ctx.emit(AgentEvent::OutputRetry {
                                attempt: output_retry_attempts,
                                error: error.clone(),
                            });
                            status.set_last_event(record.id);
                            let prompt = self
                                .policy
                                .output_retry
                                .message_template
                                .replace("{error}", &error);
                            messages.push(Message::user(prompt));
                            continue;
                        }
                        return Err(TinyAgentsError::StructuredOutput(error));
                    }
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
                    ctx.retract_transcript(messages.len());
                    return Err(TinyAgentsError::EmptyResponse);
                }
                run.final_response = Some(response);
                ctx.close_turn(self.policy.capture, messages);
                // Natural finish (A4): queued steering or a follow-up turns
                // "done" into "one more turn" instead of returning.
                match self.settle_natural_finish(ctx, status, messages).await? {
                    TurnFlow::NextTurn => continue,
                    TurnFlow::Exit(exit) => return Ok(exit),
                }
            }

            // A tool-calling response is a resolved turn too: clear the
            // recovery state before the tools run so the next turn starts from
            // the caller's configured cap and a full retry budget.
            // A turn whose call was cut off keeps its retry budget and boosted
            // output cap for the retry.
            turn_recovery.reset_after_tool_turn(turn_had_truncated_calls);

            // Execute requested tools: serial admission -> serial or
            // concurrent execution -> ordered fold. Multi-call turns run
            // concurrently when no tool-wrap middleware is registered; see
            // `agent_loop/tools.rs` for the dispatch rules and the semantics
            // preserved in each mode.
            status.mark_running(HarnessPhase::Tools);
            let deferred = self
                .execute_tools_with_promotions(
                    state,
                    ctx,
                    run,
                    status,
                    messages,
                    real_tool_calls,
                    &mut surface.promoted_names,
                )
                .await?;
            // A2: a batch that deferred calls either resolves them inline
            // (handler registered) or ends the run here with the pending
            // requests; the non-deferred siblings' results are already on
            // the transcript.
            if let Some(exit) = self
                .settle_deferred(state, ctx, run, status, messages, deferred)
                .await?
            {
                return Ok(exit);
            }

            ctx.close_turn(self.policy.capture, messages);

            // Turn boundary (A4): every tool result of this batch is on the
            // transcript, so queued steering can be applied now — never
            // mid-batch — before the next model call sees it.
            self.apply_queued_lane(ctx, status, messages, crate::run_queue::QueueLane::Steer)
                .await;

            // Turn boundary: give every middleware a chance to end the run
            // based on the whole turn's tool results rather than any single
            // call (see `Middleware::should_stop_after_turn`). A `Middleware`
            // hook could already have requested `JumpTo(End)` from
            // `after_tool_control`; this is the aggregate counterpart for a
            // decision that only makes sense once the whole turn has settled.
            if self.middleware.any_should_stop_after_turn(ctx, run) {
                ctx.request_control(MiddlewareControl::JumpTo(LoopTarget::End));
            }

            // Safe checkpoint: honor a control requested from `after_tool` /
            // `wrap_tool` at the edge it was raised on, rather than a model
            // call later.
            match self.apply_pending_control(ctx, run, status, messages)? {
                ControlEffect::None => {}
                ControlEffect::ContinueLoop => continue,
                ControlEffect::Exit(exit) => return Ok(exit),
            }
        }
    }

    /// Builds the [`StructuredExtractor`] for a resolved `structured_plan`
    /// entry (A6).
    ///
    /// [`StructuredStrategy::ToolCallUnion`] needs its variant list, which
    /// `structured_plan`'s `(strategy, name, schema)` tuple has nowhere to
    /// carry — the variants live on
    /// [`crate::runtime::RunPolicy::structured_strategy_override`] instead,
    /// which this reaches back into rather than widening the tuple. Every
    /// other strategy builds the extractor directly from the tuple as
    /// before.
    pub(super) fn build_structured_extractor(
        &self,
        strategy: &StructuredStrategy,
        name: &str,
        schema: &Value,
    ) -> StructuredExtractor {
        if matches!(strategy, StructuredStrategy::ToolCallUnion)
            && let Some(crate::runtime::StructuredStrategyOverride::ToolCallUnion { variants }) =
                &self.policy.structured_strategy_override
        {
            return StructuredExtractor::new_union(name, variants.clone());
        }
        StructuredExtractor::new(strategy.clone(), name.to_string(), schema.clone())
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
}

/// Use the frozen boundary supplied by a durable session when rebuilding a
/// model request on the next harness invocation. A later System summary is
/// model-visible history, not another stable prompt tier.
pub(super) fn cacheable_system_prefix_end(
    messages: &[Message],
    frozen_system_prefix_len: Option<usize>,
) -> usize {
    let leading_system = messages
        .iter()
        .take_while(|message| matches!(message, Message::System(_)))
        .count();
    frozen_system_prefix_len.map_or(leading_system, |count| count.min(leading_system))
}

/// Prevent the dispatch refresh from inferring a newly leading System
/// summary as stable when a session explicitly froze zero messages. An empty
/// annotation means "infer from roles" to the harness, so retain an explicit
/// noncacheable marker in this zero-prefix, no-tools case even if the summary
/// has not been inserted by a later middleware yet.
pub(super) fn mark_empty_frozen_prefix(
    request: &mut ModelRequest,
    frozen_system_prefix_len: Option<usize>,
) {
    if frozen_system_prefix_len == Some(0) && request.cache_segments.is_empty() {
        request.cache_segments.push(PromptSegment {
            id: crate::cache::VOLATILE_SYSTEM_HISTORY_SEGMENT_ID.into(),
            role: SegmentRole::Volatile,
            cacheable: false,
        });
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
    crate::cache::promote_tools_after_zero_prefix_marker(request);
    let leading_system_end = request
        .messages
        .iter()
        .take_while(|message| matches!(message, Message::System(_)))
        .count();
    // An explicit canonical layout names the cacheable system messages. A
    // compaction summary can be another leading System message without being
    // part of that frozen prefix; promoting it here re-rolls the provider's
    // prompt_cache_key on every compaction. With no explicit boundary, keep
    // the existing conservative leading-System behavior.
    let system_end =
        crate::cache::declared_system_prefix_len(request).unwrap_or(leading_system_end);
    let mut expected_layout = (0..system_end)
        .map(|index| PromptSegment {
            id: crate::prompt::system_segment_id(index),
            role: SegmentRole::System,
            cacheable: true,
        })
        .collect::<Vec<_>>();
    if !request.tools.is_empty() {
        expected_layout.push(PromptSegment {
            id: "tools".to_string(),
            role: SegmentRole::Tools,
            cacheable: true,
        });
    }
    // The canonical harness-owned trailing tools segment: only *this* exact
    // segment (including `cacheable: true`) is recognized as the harness's
    // own below, so middleware that deliberately annotated its own trailing
    // `tools` segment `cacheable: false` keeps that opt-out instead of being
    // silently promoted to cacheable once a text dialect strips the schemas.
    let canonical_tools_segment = PromptSegment {
        id: "tools".to_string(),
        role: SegmentRole::Tools,
        cacheable: true,
    };
    // A text dialect (`RunDialect::apply_to_request`) folds the catalogue
    // into the system prompt and clears `tools` *after* `before_model` ran,
    // so a middleware that declared the harness layout while the schemas
    // were still on the request legitimately carries a trailing `tools`
    // segment the rebuilt layout no longer has. That is still the harness
    // layout, not a custom annotation: demoting it to the whole-request
    // digest below would re-roll the provider routing key on every call.
    //
    // The declared head has to equal the rebuilt system-segment prefix
    // exactly. The one case that legitimately would not — no leading system
    // message at declare time, so the dialect synthesizes one — is already
    // resolved before this function ever runs, by
    // `RunDialect::sync_stripped_tools_cache_segment`, which has the
    // pre-rewrite message shape this function does not: reconstructing that
    // distinction from the rewritten request alone cannot tell an
    // actually-synthesized leading segment apart from a custom declaration
    // that deliberately left an already-present system message out of the
    // cache key.
    let declared_with_stripped_tools = request.tools.is_empty()
        && request
            .cache_segments
            .split_last()
            .is_some_and(|(last, head)| {
                *last == canonical_tools_segment && head == expected_layout
            });
    let harness_layout = request.cache_segments.is_empty()
        || request.cache_segments == expected_layout
        || declared_with_stripped_tools;

    if harness_layout {
        request.cache_segments = expected_layout;
        if request.cache_segments.is_empty() {
            request.prompt_fingerprint = None;
            return;
        }

        let mut prompt = crate::prompt::PromptBuilder::new();
        prompt.push_system_messages(&request.messages[..system_end]);
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

/// Recovers text-dialect calls through `tinytools-agent`
/// ([`super::dialect::recover_text_calls`]) while preserving every non-text
/// provider content block (notably reasoning blocks).
///
/// `recovery` carries the tools offered this turn and the P-Format registry;
/// it is already empty when nothing could be recovered — no tools offered,
/// an effective `ToolChoice::None`, or
/// [`crate::runtime::RunPolicy::text_dialect_recovery`] set to `Off`. For a
/// native model under `Auto` the caller skips this function, and only the
/// stream scrubber recovers, special-token calls only (see the `recovery`
/// binding in `run_loop_body`, #6733). Whether markup inside a fenced code
/// block is a call or a quoted example is `tinytools-agent`'s decision (its protected ranges): a
/// language-tagged fence is protected, a bare one is not. Emits
/// [`AgentEvent::ControlApplied`] when it actually rewrites the response, so
/// the recovery is auditable rather than a silent transform.
fn recover_text_dialect_calls<Ctx>(
    ctx: &RunContext<Ctx>,
    response: &mut tinyinference_llm::model::ModelResponse,
    model_call_id: &CallId,
    recovery: &super::dialect::TextRecovery,
) {
    if recovery.offered.is_empty() {
        return;
    }

    let before = response.message.tool_calls.len();
    super::dialect::recover_text_calls(
        response,
        model_call_id,
        &recovery.offered,
        recovery.registry.as_deref(),
        &recovery.dropped,
    );
    let recovered = response.message.tool_calls.len().saturating_sub(before);
    if recovered == 0 {
        return;
    }

    ctx.emit(AgentEvent::ControlApplied {
        control: "text_dialect_recovered".to_string(),
        detail: format!(
            "recovered {recovered} text-dialect tool call(s) from model call `{model_call_id}`"
        ),
    });
}

/// The re-prompt sent when a model signalled a tool call it did not make.
/// Deliberately terse and instruction-free beyond the one thing needed: the
/// task and the tools are already in the transcript.
pub(super) const DROPPED_TOOL_CALL_NUDGE: &str = "Your previous turn indicated a tool call but none was \
     included. If you meant to call a tool, issue the actual tool call now; otherwise answer \
     directly.";

/// The re-prompt sent when the model wrote a tool call on a turn that offered
/// no callable tool. The call was scrubbed and not run; the wording names that
/// plainly, because a model told only to "answer" keeps trying to act.
pub(super) const WITHHELD_TOOL_CALL_NUDGE: &str = "Your previous reply was a tool call, but tools are not \
     available for this reply, so it did not run. Do not write tool calls. Answer now in plain \
     text from the results already gathered, and state any remaining uncertainty.";

/// The re-prompt sent when a reply ran out of output tokens while reasoning,
/// produced no tool call, and the boosted retry failed the same way (see
/// [`crate::runtime::RunPolicy::truncated_empty_nudges`]). It names the cause
/// and asks for the smallest next step: a model told only to "continue"
/// deliberates again, and one writing a large file in a single call runs out
/// again.
pub(super) const TRUNCATED_EMPTY_TOOL_NUDGE: &str = "Your last reply ran out of output tokens while \
     reasoning and produced no tool call. Stop deliberating: make the next tool call now, and \
     write files incrementally in small pieces.";

/// [`TRUNCATED_EMPTY_TOOL_NUDGE`] for a turn with no callable tool (tools
/// withdrawn for a concluding answer, or `ToolChoice::None`): asking for a
/// tool call there would only get a call that cannot run.
pub(super) const TRUNCATED_EMPTY_ANSWER_NUDGE: &str = "Your last reply ran out of output tokens while \
     reasoning and produced no answer. Stop deliberating and write a short answer now from \
     what you already have.";

/// The re-prompt sent when a text-dialect tool-call block could not be
/// decoded: no tool ran, and the model should know why rather than assume
/// its call went through.
pub(super) const UNDECODABLE_TOOL_CALL_NUDGE: &str = "Your previous turn contained a tool-call block that \
     could not be parsed, so no tool ran. Re-issue the call using exactly the format from the \
     tool protocol; otherwise answer directly.";

/// The tool calls on the transcript's last assistant row that have no
/// matching tool-result row after it — the calls a previous run deferred
/// (A2). Errors when the transcript has nothing to resume.
fn pending_tool_calls(messages: &[Message]) -> Result<Vec<ToolCall>> {
    let Some(assistant_at) = messages
        .iter()
        .rposition(|message| matches!(message, Message::Assistant(_)))
    else {
        return Err(TinyAgentsError::Validation(
            "cannot resume: the transcript has no assistant tool-call row".to_string(),
        ));
    };
    let Message::Assistant(assistant) = &messages[assistant_at] else {
        unreachable!("rposition matched an assistant row");
    };
    // Count answers per id: two calls sharing an id need two tool rows, so
    // one row must not mark both answered (the earlier call takes it).
    let mut answered: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for message in &messages[assistant_at + 1..] {
        if let Message::Tool(tool) = message {
            *answered.entry(tool.tool_call_id.as_str()).or_default() += 1;
        }
    }
    let pending: Vec<ToolCall> = assistant
        .tool_calls
        .iter()
        .filter(|call| match answered.get_mut(call.id.as_str()) {
            Some(left) if *left > 0 => {
                *left -= 1;
                false
            }
            _ => true,
        })
        .cloned()
        .collect();
    if pending.is_empty() {
        return Err(TinyAgentsError::Validation(
            "cannot resume: the transcript has no unanswered tool calls".to_string(),
        ));
    }
    Ok(pending)
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

/// The positions of the calls of a length-stopped response that may be
/// incomplete: the last call (the cut landed inside it, whether it came from
/// the native tool channel or was recovered from text) and any call the
/// provider flagged `invalid` (a repair could make it look whole). Every
/// earlier call was finished before the cut.
pub(super) fn truncated_call_positions(calls: &[ToolCall]) -> std::collections::HashSet<usize> {
    let Some(last) = calls.len().checked_sub(1) else {
        return std::collections::HashSet::new();
    };
    calls
        .iter()
        .enumerate()
        .filter(|(index, call)| *index == last || call.invalid.is_some())
        .map(|(index, _)| index)
        .collect()
}

#[cfg(test)]
#[path = "run_loop_recovery_tests.rs"]
mod recovery_tests;

#[cfg(test)]
#[path = "run_loop_dead_call_tests.rs"]
mod dead_call_tests;

#[cfg(test)]
#[path = "run_loop_withheld_tests.rs"]
mod withheld_tests;

/// Attaches `default` to `request` when the request carries no reasoning
/// config of its own. A request-level config always wins.
pub(crate) fn apply_default_reasoning(
    request: &mut ModelRequest,
    default: Option<&tinyinference_llm::model::ReasoningConfig>,
) {
    if request.reasoning.is_none()
        && let Some(reasoning) = default
        && !reasoning.is_empty()
    {
        tracing::debug!(
            effort = ?reasoning.effort,
            budget_tokens = ?reasoning.budget_tokens,
            "[agent_loop] applying run-policy default reasoning"
        );
        request.reasoning = Some(reasoning.clone());
    }
}
