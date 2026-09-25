//! The core superstep loop body: `run_loop` drives one model call,
//! any requested tool calls, and repeats until the model finishes or a
//! configured limit is reached.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

use super::handoff_transform;
use super::model_call::ModelCallBase;
use super::tool_changes;
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
        // A4: the `Collect` lane is delivered on the run, never on the
        // transcript, and on every exit path — a host that pushed
        // observations during a run that then failed still gets them back.
        if let Some(queue) = ctx.run_queue.clone() {
            run.collected
                .extend(queue.drain(crate::run_queue::QueueLane::Collect).await);
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

        // Build the direct tool set once. Discovery can add typed declarations
        // after a search, while the base declarations stay stable.
        //
        // Only *direct* tools go on the initial wire request. Deferred tools
        // are indexed into the run's catalogue and reached through the
        // `tool_search` / `tool_call` bridge. Search matches are promoted on
        // the next request; the bridge schemas follow the direct set.
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
        // Composable toolset chain (gap B3, `AgentHarness::with_toolset`):
        // additive to the registry's own `Direct` schemas above — a name the
        // registry already advertises keeps the registry's declaration, so a
        // registered tool always wins a collision. This run's toolset is
        // consulted once here, matching the registry's own once-per-run
        // schema build a few lines up (the comment above explains why: the
        // resulting request tool list feeds the provider prompt cache, so
        // rebuilding it every turn would defeat that cache). A caller that
        // genuinely needs true per-turn variance can still call
        // [`crate::tool::toolset::ToolSet::tools`] directly from a
        // `before_model` middleware, which *does* run every turn.
        if let Some(toolset) = &self.toolset {
            let existing: std::collections::HashSet<&str> = tool_schemas
                .iter()
                .map(|schema| schema.name.as_str())
                .collect();
            let extra: Vec<_> = toolset
                .tools(ctx)
                .await?
                .into_iter()
                .filter(|tool| tool.exposure() == tinytools::ToolExposure::Direct)
                .filter(|tool| host_allows(tool.name()))
                .filter(|tool| !existing.contains(tool.name()))
                .map(|tool| crate::tool::provider_schema(tool.as_ref()))
                .collect();
            tool_schemas.extend(extra);
            // Keep the combined set name-sorted: every consumer of
            // `tool_schemas` below (and the provider request it feeds) relies
            // on the sort for wire-byte/prompt-cache stability.
            tool_schemas.sort_by(|left, right| left.name.cmp(&right.name));
        }
        // Provider projection applies once, to the full combined set
        // (registry + toolset), so a toolset-supplied schema reaches the
        // wire cleaned exactly like a registered one.
        if let Some(preparation) = &self.policy.tool_schemas {
            tool_schemas = crate::tool::prepare_tool_schemas(&tool_schemas, preparation);
        }
        // Captured before the bridge schemas are appended below, so
        // `ToolsAdvertised.direct` reports the actual `Direct`-exposure
        // count. Otherwise it would silently include the two intrinsic
        // bridge schemas whenever discovery is enabled, double-counting
        // relative to `deferred` and making `direct` mean different things
        // depending on whether any tool happens to be deferred.
        let direct_schema_count = tool_schemas.len();
        let mut direct_tool_schemas = tool_schemas.clone();
        // B6 (`docs/runtime-comparison/plan.md`): `declared_tool_schemas`
        // tracks what the transcript has actually been told about the
        // toolset chain's tools so far (folded or patched in, turn by turn,
        // by the loop below), so a later turn's live toolset resolution can
        // be diffed against it instead of against the wire list — the wire
        // list also carries the bridge schemas captured into
        // `bridge_schemas` next, which never change within a run and so are
        // deliberately excluded from the diff.
        //
        // Starts empty rather than seeded from the merge above: nothing has
        // been recorded on the transcript yet, so the loop's first-turn diff
        // (below) always fires when a toolset is installed, declaring the
        // full initial toolset-supplied set as one patch (turn 1 has no
        // prior cached prefix to protect, so there is no cost to always
        // recording it). This is what makes
        // [`tinyinference_llm::message::replay_system_state`] able to
        // reconstruct the *complete* effective tool set from the transcript
        // alone, not just later deltas — the alternative (seeding from the
        // merge above) would leave the initial toolset-only tools
        // permanently undeclared on the wire-only `tool_schemas` snapshot
        // computed here, which a replay can never see.
        let mut declared_tool_schemas: Vec<ToolSchema> = Vec::new();
        let mut bridge_schemas: Vec<ToolSchema> = Vec::new();
        let deferred_catalog = self.deferred_catalog(&host_allows);
        // A resumed transcript carries promoted declarations in SystemMessage
        // patches. Only restore names still admitted into this run's catalogue.
        let mut promoted_schemas: std::collections::BTreeMap<String, ToolSchema> =
            tinyinference_llm::message::replay_system_state(messages)
                .1
                .into_iter()
                .filter(|schema| deferred_catalog.get(&schema.name).is_some())
                .map(|schema| (schema.name.clone(), schema))
                .collect();
        let mut promoted_names: std::collections::BTreeSet<String> =
            promoted_schemas.keys().cloned().collect();
        let mut recorded_promotions = promoted_names.clone();
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
                    tool_schemas.push(schema.clone());
                    bridge_schemas.push(schema);
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
            direct: direct_schema_count,
            deferred: deferred_catalog.len(),
            schema_bytes: crate::token_estimation::tool_schema_bytes(&tool_schemas),
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

        // Truncated-empty recovery state (see `RunPolicy::truncated_empty_retries`).
        // These persist across the retry `continue` within a single logical turn:
        // `boosted_max_tokens` overrides the next request's cap, `truncation_base`
        // records the original cap so growth stays clamped at 4x, and the counter
        // bounds how many times we re-issue the call.
        let mut truncated_empty_retries_used: u32 = 0;
        let mut empty_response_retries_used: u32 = 0;
        // Consecutive "you said tool_calls but sent none" re-prompts
        // (see `RunPolicy::dropped_tool_call_nudges`).
        let mut dropped_tool_call_nudges_used: u32 = 0;
        let mut boosted_max_tokens: Option<u32> = None;
        let mut truncation_base: Option<u32> = None;

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

            // B6 (`docs/runtime-comparison/plan.md`, `declare_tool_changes`):
            // re-consult the toolset chain (documented as "called once per
            // turn", `ToolSet::tools`) and diff its live set against what
            // this transcript has declared so far. A caller whose toolset
            // never varies turn to turn sees no diff and pays nothing here —
            // this only fires for a genuine mid-run change. Deliberately
            // runs before the request/`ModelStarted` below, so the patch (if
            // any) is part of *this* turn's request.
            if let Some(toolset) = &self.toolset {
                let mut live_schemas: Vec<ToolSchema> = self
                    .tools
                    .schemas()
                    .into_iter()
                    .filter(|schema| host_allows(&schema.name))
                    .collect();
                let existing: std::collections::HashSet<&str> = live_schemas
                    .iter()
                    .map(|schema| schema.name.as_str())
                    .collect();
                let extra: Vec<_> = toolset
                    .tools(ctx)
                    .await?
                    .into_iter()
                    .filter(|tool| tool.exposure() == tinytools::ToolExposure::Direct)
                    .filter(|tool| host_allows(tool.name()))
                    .filter(|tool| !existing.contains(tool.name()))
                    .map(|tool| crate::tool::provider_schema(tool.as_ref()))
                    .collect();
                live_schemas.extend(extra);
                live_schemas.sort_by(|left, right| left.name.cmp(&right.name));
                if let Some(preparation) = &self.policy.tool_schemas {
                    live_schemas = crate::tool::prepare_tool_schemas(&live_schemas, preparation);
                }
                if let Some(patch) =
                    tool_changes::diff_tool_set(&declared_tool_schemas, &live_schemas)
                {
                    // A cheap, non-mutating preview resolution against the
                    // transcript as it stands (pre-patch) decides fold vs.
                    // insert. It is a pure registry lookup (no network call,
                    // see `ModelRegistry::resolve_request`), so this stays
                    // proportional to the diff it gates. An unresolved
                    // preview conservatively folds (`false`): folding is
                    // always correct, only less cache-friendly.
                    let mid_conversation = self
                        .models
                        .resolve_request(&ModelRequest::new(messages.clone()), None, None)
                        .and_then(|binding| binding.model.profile().cloned())
                        .is_some_and(|profile| profile.mid_conversation_system_messages);
                    tool_changes::apply_tool_change_patch(messages, patch, mid_conversation);
                    declared_tool_schemas = live_schemas.clone();
                    direct_tool_schemas = live_schemas;
                }
            }

            // Promote only names returned by a successful intrinsic search.
            // The patch makes the declaration recoverable from the transcript;
            // the provider receives its typed schema on this and later calls.
            let newly_promoted: Vec<ToolSchema> = promoted_names
                .difference(&recorded_promotions)
                .filter_map(|name| deferred_catalog.get(name).cloned())
                .collect();
            if !newly_promoted.is_empty() {
                if let Some(patch) = tool_changes::diff_tool_set(&[], &newly_promoted) {
                    let mid_conversation = self
                        .models
                        .resolve_request(&ModelRequest::new(messages.clone()), None, None)
                        .and_then(|binding| binding.model.profile().cloned())
                        .is_some_and(|profile| profile.mid_conversation_system_messages);
                    tool_changes::apply_tool_change_patch(messages, patch, mid_conversation);
                }
                recorded_promotions.extend(newly_promoted.iter().map(|schema| schema.name.clone()));
                promoted_schemas.extend(
                    newly_promoted
                        .into_iter()
                        .map(|schema| (schema.name.clone(), schema)),
                );
            }
            tool_schemas = direct_tool_schemas.clone();
            tool_schemas.extend(
                promoted_schemas
                    .values()
                    .filter(|schema| {
                        !direct_tool_schemas
                            .iter()
                            .any(|direct| direct.name == schema.name)
                    })
                    .cloned(),
            );
            tool_schemas.extend(bridge_schemas.clone());

            // Build the request from the working transcript, tool schemas, and
            // policy response format.  Go through `PromptBuilder` rather than
            // constructing `ModelRequest` directly: a provider KV cache needs
            // an explicit stable prefix, and the system instructions plus the
            // name-sorted tool schemas are stable between discoveries.
            status.mark_running(HarnessPhase::BuildingRequest);
            let system_end = cacheable_system_prefix_end(messages, ctx.frozen_system_prefix_len);
            let mut prompt = crate::prompt::PromptBuilder::new();
            prompt.push_system_messages(&messages[..system_end]);
            if !tool_schemas.is_empty() {
                prompt.push_tools_segment("tools", tool_schemas.clone());
            }
            let mut request = prompt.build(messages[system_end..].to_vec());
            mark_empty_frozen_prefix(&mut request, ctx.frozen_system_prefix_len);
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
            if let Some(profile) = binding.model.profile()
                && let Some(reasoning) = request.reasoning.as_ref()
                && reasoning.budget_tokens.is_none()
                && let Some(effort) = reasoning.effort
                && let Some(mapped) = profile.thinking_level_map.get(effort.as_str())
            {
                request.reasoning = Some(mapped.clone());
            }

            // Resolve the structured-output plan against the resolved model.
            // `Auto` consults the model profile to choose provider-native schema
            // mode versus a tool-call fallback; an explicit `JsonSchema` always
            // uses provider-native mode. The chosen strategy drives extraction of
            // the final response below.
            // Marks where any structured-output fallback tool gets pushed
            // below, so it can be told apart afterward from what was already
            // on `request.tools` — see `synthesized_tools`.
            let tools_before_structured_plan = request.tools.len();
            let structured_plan: Option<(StructuredStrategy, String, Value)> =
                match request.response_format.clone() {
                    Some(ResponseFormat::Auto { name, schema })
                        if matches!(
                            self.policy.structured_strategy_override,
                            Some(crate::runtime::StructuredStrategyOverride::Prompted { .. })
                        ) =>
                    {
                        let template = match &self.policy.structured_strategy_override {
                            Some(crate::runtime::StructuredStrategyOverride::Prompted {
                                template,
                            }) => template.clone(),
                            _ => unreachable!("guarded by the match arm above"),
                        };
                        let schema = crate::tool::apply_profile_schema_transform(
                            &schema,
                            binding.model.profile(),
                        );
                        request.response_format = Some(ResponseFormat::Text);
                        let instructions = template.clone().unwrap_or_else(|| {
                            crate::structured::default_prompted_template().to_string()
                        });
                        let schema_text = serde_json::to_string_pretty(&schema).unwrap_or_default();
                        crate::cache::prepend_system_message(
                            &mut request,
                            format!("{instructions}\n\nJSON Schema for `{name}`:\n{schema_text}"),
                        );
                        Some((StructuredStrategy::Prompted { template }, name, schema))
                    }
                    Some(ResponseFormat::Auto { name, schema })
                        if matches!(
                            self.policy.structured_strategy_override,
                            Some(crate::runtime::StructuredStrategyOverride::ToolCallUnion { .. })
                        ) =>
                    {
                        let variants = match &self.policy.structured_strategy_override {
                            Some(crate::runtime::StructuredStrategyOverride::ToolCallUnion {
                                variants,
                            }) => variants.clone(),
                            _ => unreachable!("guarded by the match arm above"),
                        };
                        request.response_format = Some(ResponseFormat::Text);
                        for (variant_name, variant_schema) in &variants {
                            let variant_schema = crate::tool::apply_profile_schema_transform(
                                variant_schema,
                                binding.model.profile(),
                            );
                            let schema_tool = ToolSchema {
                                name: variant_name.clone(),
                                description: format!("Return the result as `{variant_name}`."),
                                parameters: variant_schema,
                                format: tinyinference_llm::tool::ToolFormat::Json,
                            };
                            request.tools.push(match &self.policy.tool_schemas {
                                Some(preparation) => {
                                    crate::tool::prepare_tool_schema(&schema_tool, preparation)
                                }
                                None => schema_tool,
                            });
                        }
                        let _ = schema;
                        Some((StructuredStrategy::ToolCallUnion, name, Value::Null))
                    }
                    Some(ResponseFormat::Auto { name, schema }) => {
                        let schema = crate::tool::apply_profile_schema_transform(
                            &schema,
                            binding.model.profile(),
                        );
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
                            // A profile whose `default_structured_mode` is
                            // `Prompted` reaches this arm too (not only
                            // through the dedicated
                            // `structured_strategy_override` arm above): the
                            // schema goes into the system segment instead of
                            // a provider API field, mirroring the override
                            // arm's construction.
                            StructuredStrategy::Prompted { ref template } => {
                                request.response_format = Some(ResponseFormat::Text);
                                let instructions = template.clone().unwrap_or_else(|| {
                                    crate::structured::default_prompted_template().to_string()
                                });
                                let schema_text =
                                    serde_json::to_string_pretty(&schema).unwrap_or_default();
                                crate::cache::prepend_system_message(
                                    &mut request,
                                    format!(
                                        "{instructions}\n\nJSON Schema for `{name}`:\n{schema_text}"
                                    ),
                                );
                            }
                            // `for_profile` never returns `ToolCallUnion`;
                            // that strategy is reached exclusively through
                            // the dedicated `structured_strategy_override`
                            // arm above.
                            StructuredStrategy::ToolCallUnion => unreachable!(
                                "StructuredStrategy::for_profile never returns ToolCallUnion"
                            ),
                        }
                        Some((strategy, name, schema))
                    }
                    Some(ResponseFormat::JsonSchema { name, schema }) => {
                        let schema = crate::tool::apply_profile_schema_transform(
                            &schema,
                            binding.model.profile(),
                        );
                        request.response_format = Some(ResponseFormat::JsonSchema {
                            name: name.clone(),
                            schema: schema.clone(),
                        });
                        Some((StructuredStrategy::ProviderSchema, name, schema))
                    }
                    _ => None,
                };

            // Tool schemas minted by the structured-output plan above (the
            // `ToolCall` / `ToolCallUnion` fallback tools), pushed onto
            // `request.tools` after `tools_before_structured_plan` was
            // recorded. A host that renders its own static tool catalogue
            // composed it before this turn's structured-output planning ran,
            // so it cannot have advertised these; `RunDialect::apply_to_request`
            // appends their catalogue entries even in the host-rendered case.
            let synthesized_tools: Vec<ToolSchema> =
                request.tools[tools_before_structured_plan..].to_vec();

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
            let recovery = if tools_available_this_turn {
                super::dialect::TextRecovery {
                    offered: Arc::new(request.tools.clone()),
                    registry: dialect.registry_for(&request.tools),
                }
            } else {
                super::dialect::TextRecovery::default()
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
                    .with_tool_count(offered_tool_count);
                    let permit = ctx
                        .bounded(self.call_budget(ctx), budget.acquire(&estimate), || {
                            format!(
                                "budget admission for run `{}` exceeded its remaining wall-clock deadline",
                                ctx.run_id()
                            )
                        })
                        .await?;
                    Some((budget.clone(), permit))
                } else {
                    None
                }
            } else {
                None
            };
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
            // call's response (see the call site after the model returns). A
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
            let (mut response, wrap_control) = self
                .middleware
                .run_wrapped_model(ctx, state, request, &base)
                .await?
                .into_response_with_control();
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
            // tools or the effective tool choice was `None` (computed
            // above); the `forced_text_dialect || text_dialect_recovery_enabled`
            // gate additionally skips a resolved model whose profile reports
            // native tool calling under `RunPolicy::text_dialect_recovery`'s
            // `Auto` default. The fenced-code guard and the audit event live
            // in the wrapper.
            if forced_text_dialect || text_dialect_recovery_enabled {
                recover_text_dialect_calls(ctx, &mut response, &call_id, &recovery);
            }

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

            if structured_tool_hit && !real_tool_calls.is_empty() {
                // A6: one turn asked to both answer (the structured-output
                // schema call) and run further tools. `RunPolicy::end_strategy`
                // decides what happens to the two, replacing the old
                // ad-hoc "record and keep going" behavior with three named,
                // documented outcomes (`EndStrategy`).
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "structured_with_tool_calls".to_string(),
                    detail: format!(
                        "{:?} end_strategy handling {} real tool call(s) alongside a \
                         structured-output call",
                        self.policy.end_strategy,
                        real_tool_calls.len()
                    ),
                });
                status.set_last_event(record.id);

                if matches!(self.policy.end_strategy, EndStrategy::Early) {
                    // Finish immediately: the structured answer wins outright,
                    // and the accompanying tool calls never run. Every
                    // requested `tool_call_id` — structured hits and the
                    // skipped real calls alike — still needs an answer or the
                    // transcript is malformed for a future replay.
                    if let Some((strategy, name, schema)) = &structured_plan {
                        let extractor = self.build_structured_extractor(strategy, name, schema);
                        match extractor.extract(&response) {
                            Ok(output) => {
                                run.structured = Some(output.value);
                                run.structured_variant = output.variant;
                            }
                            Err(error) => tracing::debug!(
                                target: "tinyagents::agent_loop",
                                run_id = %ctx.run_id(),
                                %error,
                                "[agent_loop] structured extraction failed on a mixed turn \
                                 under EndStrategy::Early"
                            ),
                        }
                    }
                    for call in &structured_hits {
                        messages.push(Message::tool(
                            call.id.clone(),
                            "Structured output recorded.",
                        ));
                    }
                    for call in &real_tool_calls {
                        messages.push(Message::tool(
                            call.id.clone(),
                            "run stopped before this tool call was executed \
                             (EndStrategy::Early: the structured answer ends the run first)",
                        ));
                    }
                    run.final_response = Some(response);
                    if self
                        .continue_from_queue_at_finish(ctx, status, messages)
                        .await
                    {
                        continue;
                    }
                    return Ok(LoopExit::Finished);
                }

                if matches!(self.policy.end_strategy, EndStrategy::Graceful) {
                    // Record the answer now (it will not be asked for again),
                    // but let the requested tools actually run before ending
                    // the run — their side effects and results are not
                    // silently dropped, unlike `Early`.
                    if let Some((strategy, name, schema)) = &structured_plan {
                        let extractor = self.build_structured_extractor(strategy, name, schema);
                        match extractor.extract(&response) {
                            Ok(output) => {
                                run.structured = Some(output.value);
                                run.structured_variant = output.variant;
                            }
                            Err(error) => tracing::debug!(
                                target: "tinyagents::agent_loop",
                                run_id = %ctx.run_id(),
                                %error,
                                "[agent_loop] structured extraction failed on a mixed turn \
                                 under EndStrategy::Graceful"
                            ),
                        }
                    }
                    for call in &structured_hits {
                        messages.push(Message::tool(
                            call.id.clone(),
                            "Structured output recorded.",
                        ));
                    }
                    status.mark_running(HarnessPhase::Tools);
                    let deferred = self
                        .execute_tools_with_promotions(
                            state,
                            ctx,
                            run,
                            status,
                            messages,
                            real_tool_calls,
                            &mut promoted_names,
                        )
                        .await?;
                    if let Some(exit) = self
                        .settle_deferred(state, ctx, run, status, messages, deferred)
                        .await?
                    {
                        return Ok(exit);
                    }
                    if let ControlEffect::Exit(exit) =
                        self.apply_pending_control(ctx, run, status, messages)?
                    {
                        return Ok(exit);
                    }
                    run.final_response = Some(response);
                    if self
                        .continue_from_queue_at_finish(ctx, status, messages)
                        .await
                    {
                        continue;
                    }
                    return Ok(LoopExit::Finished);
                }

                // `EndStrategy::Exhaustive`: the output tool this turn is
                // ignored outright (never recorded) — the run keeps going
                // exactly as if only the real tool calls had been requested.
                // It only finishes once a later turn's output-tool call has
                // no accompanying function-tool calls.
                debug_assert!(matches!(self.policy.end_strategy, EndStrategy::Exhaustive));
                for call in &structured_hits {
                    messages.push(Message::tool(
                        call.id.clone(),
                        "Structured output noted but not final yet; finish the remaining tool \
                         calls first (EndStrategy::Exhaustive).",
                    ));
                }

                // A mixed turn (structured payload alongside real tool calls)
                // is a resolved turn exactly like an ordinary tool-calling one
                // (see the reset below at the non-mixed path): it must not
                // leave a spent `dropped_tool_call_nudges_used` counter to
                // leak into a later, unrelated dropped-call turn, which would
                // otherwise receive fewer than the policy's configured number
                // of consecutive re-prompts.
                dropped_tool_call_nudges_used = 0;
                empty_response_retries_used = 0;
                reset_truncated_empty_recovery(
                    &mut truncated_empty_retries_used,
                    &mut boosted_max_tokens,
                    &mut truncation_base,
                );

                status.mark_running(HarnessPhase::Tools);
                let deferred = self
                    .execute_tools_with_promotions(
                        state,
                        ctx,
                        run,
                        status,
                        messages,
                        real_tool_calls,
                        &mut promoted_names,
                    )
                    .await?;
                if let Some(exit) = self
                    .settle_deferred(state, ctx, run, status, messages, deferred)
                    .await?
                {
                    return Ok(exit);
                }

                // Turn boundary (A4): same steer drain as the plain tool path.
                self.apply_queued_lane(ctx, status, messages, crate::run_queue::QueueLane::Steer)
                    .await;

                // Safe checkpoint: a control requested from `after_tool` /
                // `wrap_tool` is honored here, at the edge it was raised on.
                match self.apply_pending_control(ctx, run, status, messages)? {
                    ControlEffect::None => {}
                    ControlEffect::ContinueLoop => continue,
                    ControlEffect::Exit(exit) => return Ok(exit),
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

                // A provider can also finish normally after sending only a
                // reasoning side channel (or no content at all). Retrying that
                // unusable answer is opt-in because it incurs another provider
                // call. Unlike a length-truncated reply, keep the same token
                // cap: there is no evidence that output space ran out.
                let nontruncated_empty = tool_calls.is_empty()
                    && response.text().trim().is_empty()
                    && response.continue_turn.is_none()
                    && structured_plan.is_none()
                    && run.structured.is_none()
                    && response.finish_reason.as_deref() != Some("length")
                    && response.finish_reason.as_deref() != Some("tool_calls")
                    && !response.served_from_cache;
                if nontruncated_empty
                    && empty_response_retries_used < self.policy.empty_response_retries
                    && ctx.limits.remaining_model_calls() > 0
                {
                    messages.pop();
                    empty_response_retries_used += 1;
                    tracing::info!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        call_id = %call_id,
                        attempt = empty_response_retries_used,
                        finish_reason = ?response.finish_reason,
                        content_blocks = response.message.content.len(),
                        "[agent_loop] retrying completion without visible answer"
                    );
                    let record = ctx.emit(AgentEvent::RetryScheduled {
                        call_id: call_id.clone(),
                        attempt: empty_response_retries_used as usize,
                    });
                    status.set_last_event(record.id);
                    continue;
                }

                // Dropped tool call: the provider says the model stopped to
                // call a tool, but nothing arrived — structured or in text.
                // A bounded re-prompt asks for the call itself. The assistant
                // row stays on the transcript so the model sees what it did.
                if tool_calls.is_empty()
                    && response.finish_reason.as_deref() == Some("tool_calls")
                    && tools_available_this_turn
                    && dropped_tool_call_nudges_used < self.policy.dropped_tool_call_nudges
                {
                    dropped_tool_call_nudges_used += 1;
                    messages.push(Message::user(DROPPED_TOOL_CALL_NUDGE));
                    let record = ctx.emit(AgentEvent::RetryScheduled {
                        call_id: call_id.clone(),
                        attempt: dropped_tool_call_nudges_used as usize,
                    });
                    status.set_last_event(record.id);
                    continue;
                }
                dropped_tool_call_nudges_used = 0;
                empty_response_retries_used = 0;

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
                    return Err(TinyAgentsError::EmptyResponse);
                }
                run.final_response = Some(response);
                // Natural finish (A4): queued steering or a follow-up turns
                // "done" into "one more turn" instead of returning.
                if self
                    .continue_from_queue_at_finish(ctx, status, messages)
                    .await
                {
                    continue;
                }
                return Ok(LoopExit::Finished);
            }

            // A tool-calling response is a resolved turn too: clear the
            // recovery state before the tools run so the next turn starts from
            // the caller's configured cap and a full retry budget.
            dropped_tool_call_nudges_used = 0;
            empty_response_retries_used = 0;
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
            let deferred = self
                .execute_tools_with_promotions(
                    state,
                    ctx,
                    run,
                    status,
                    messages,
                    real_tool_calls,
                    &mut promoted_names,
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

    /// Takes the pending items of `lane` from the run's queue (per
    /// [`RunPolicy::queue_mode`][crate::runtime::RunPolicy::queue_mode]),
    /// appends them to the working transcript, and emits
    /// [`AgentEvent::QueuedMessageApplied`] (A4). Returns whether anything
    /// was applied. A run without a queue never applies anything.
    async fn apply_queued_lane(
        &self,
        ctx: &mut RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        lane: crate::run_queue::QueueLane,
    ) -> bool {
        let Some(queue) = ctx.run_queue.clone() else {
            return false;
        };
        let items = queue.take(lane, self.policy.queue_mode).await;
        if items.is_empty() {
            return false;
        }
        let count = items.len();
        messages.extend(items);
        let record = ctx.emit(AgentEvent::QueuedMessageApplied { lane, count });
        status.set_last_event(record.id);
        tracing::debug!(
            target: "tinyagents::agent_loop",
            run_id = %ctx.run_id(),
            lane = lane.as_str(),
            count,
            "[agent_loop] applied queued messages to the transcript"
        );
        true
    }

    /// The natural-finish queue boundary (A4): the model produced a final
    /// answer, so pending `Steer` items (first) or, when there are none,
    /// `Followup` items are appended and the loop runs another turn instead
    /// of returning. Returns whether the loop should continue. Only reached
    /// from the paths where the *model* finished — a middleware stop, a
    /// limit stop, a pause, or a deferral is terminal and leaves the queue
    /// untouched for the host.
    async fn continue_from_queue_at_finish(
        &self,
        ctx: &mut RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
    ) -> bool {
        if ctx.run_queue.is_none() {
            return false;
        }
        self.apply_queued_lane(ctx, status, messages, crate::run_queue::QueueLane::Steer)
            .await
            || self
                .apply_queued_lane(ctx, status, messages, crate::run_queue::QueueLane::Followup)
                .await
    }

    /// Settles the calls a batch deferred (A2).
    ///
    /// Returns `Ok(None)` when nothing was deferred, or when a registered
    /// [`crate::tool::DeferredToolHandler`] resolved every pending call and
    /// the loop can continue. Returns `Ok(Some(LoopExit::Deferred))` when
    /// the caller must resolve the requests — no handler, or an approved
    /// call deferred a second time (surfaced rather than re-asked, so a
    /// handler and a tool that never agree cannot spin).
    async fn settle_deferred(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        deferred: crate::tool::DeferredToolRequests,
    ) -> Result<Option<LoopExit>> {
        if deferred.is_empty() {
            return Ok(None);
        }
        let Some(handler) = &self.deferred_tool_handler else {
            return Ok(Some(LoopExit::Deferred(deferred)));
        };
        let results = handler.handle(&deferred).await?;
        let pending: Vec<ToolCall> = deferred
            .approvals
            .iter()
            .chain(deferred.calls.iter())
            .cloned()
            .collect();
        let again = self
            .apply_deferred_results(state, ctx, run, status, messages, pending, results)
            .await?;
        if again.is_empty() {
            return Ok(None);
        }
        Ok(Some(LoopExit::Deferred(again)))
    }

    /// Applies host decisions to `pending` deferred calls (A2): every call
    /// must be resolved (`Validation` error naming the missing ids
    /// otherwise). Host-supplied results and denials are answered without
    /// running a tool; approvals run the tool now through the ordinary
    /// serial pipeline, with the model's or the approver's edited
    /// arguments. Returns whatever the approved calls deferred *again*.
    #[allow(clippy::too_many_arguments)]
    async fn apply_deferred_results(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        pending: Vec<ToolCall>,
        mut results: crate::tool::DeferredToolResults,
    ) -> Result<crate::tool::DeferredToolRequests> {
        let missing: Vec<&str> = pending
            .iter()
            .filter(|call| !results.resolves(&CallId::new(call.id.clone())))
            .map(|call| call.id.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(TinyAgentsError::Validation(format!(
                "cannot resume: deferred tool calls still unresolved: [{}]",
                missing.join(", ")
            )));
        }
        let mut deferred = crate::tool::DeferredToolRequests::default();
        // Follow-up user messages (B2) trail the whole resumed batch, for
        // the same provider-ordering reason as in `execute_tools`.
        let mut follow_ups = Vec::new();
        for mut call in pending {
            let call_id = CallId::new(call.id.clone());
            if let Some(outcome) = results.calls.remove(&call_id) {
                follow_ups.extend(
                    self.recover_tool_call(
                        state,
                        ctx,
                        run,
                        status,
                        messages,
                        &call,
                        outcome.into_tool_result(),
                    )
                    .await?,
                );
                continue;
            }
            let decision = results
                .approvals
                .remove(&call_id)
                .expect("every pending call was validated as resolved above");
            match decision {
                crate::tool::ToolApprovalDecision::Deny { message } => {
                    let record = ctx.emit(AgentEvent::ToolDenied {
                        call_id,
                        message: message.clone(),
                    });
                    status.set_last_event(record.id);
                    follow_ups.extend(
                        self.recover_tool_call(
                            state,
                            ctx,
                            run,
                            status,
                            messages,
                            &call,
                            tinytools::ToolResult::error(message),
                        )
                        .await?,
                    );
                }
                decision => {
                    if let crate::tool::ToolApprovalDecision::ApproveWithArgs(arguments) = decision
                    {
                        call.arguments = arguments;
                    }
                    let record = ctx.emit(AgentEvent::ToolApproved { call_id });
                    status.set_last_event(record.id);
                    ctx.mark_call_approved(call.id.clone());
                    let mut approved_promotions = std::collections::BTreeSet::new();
                    follow_ups.extend(
                        self.execute_tool_serially(
                            state,
                            ctx,
                            run,
                            status,
                            messages,
                            call,
                            &mut deferred,
                            &mut approved_promotions,
                        )
                        .await?,
                    );
                }
            }
        }
        super::tools::append_follow_ups(messages, follow_ups);
        Ok(deferred)
    }

    /// Drains any pending [`MiddlewareControl`] and turns it into a loop
    /// decision.
    ///
    /// Returns [`ControlEffect::None`] when nothing was requested (or the
    /// pending request needed no loop-level action, e.g.
    /// [`MiddlewareControl::UpdateState`]), [`ControlEffect::ContinueLoop`]
    /// when the current turn must be abandoned in favor of a fresh iteration
    /// (`JumpTo(Model)`), and [`ControlEffect::Exit`] when the run is done.
    /// `Err` surfaces [`MiddlewareControl::Interrupt`]. Called at every safe
    /// checkpoint — the top of an iteration, after the model call, and after
    /// tool execution — so a control raised anywhere in a turn takes effect on
    /// that turn.
    fn apply_pending_control(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
    ) -> Result<ControlEffect> {
        let Some(control) = ctx.take_control() else {
            return Ok(ControlEffect::None);
        };
        // `UpdateState` is applied (queued, really — see `RunContext::
        // push_state_update`) silently: it carries no loop-level decision, so
        // audit-logging it as a `ControlApplied` event alongside jumps and
        // stops would be noise. It still shows up wherever the host inspects
        // `RunContext::take_state_updates`.
        if let MiddlewareControl::UpdateState(update) = control {
            ctx.push_state_update(update);
            return Ok(ControlEffect::None);
        }
        let record = ctx.emit(AgentEvent::ControlApplied {
            control: control.kind().to_string(),
            detail: match &control {
                MiddlewareControl::Continue => String::new(),
                MiddlewareControl::JumpTo(target) => format!("{target:?}"),
                MiddlewareControl::UpdateState(_) => unreachable!("handled above"),
                MiddlewareControl::StopWithFinal(text) => text.clone(),
                MiddlewareControl::Interrupt { node, message } => format!("{node}: {message}"),
            },
        });
        status.set_last_event(record.id);
        match control {
            MiddlewareControl::Continue => Ok(ControlEffect::None),
            MiddlewareControl::UpdateState(_) => unreachable!("handled above"),
            MiddlewareControl::JumpTo(LoopTarget::Tools) => {
                // Tool execution already runs whenever the turn produced real
                // tool calls; there is nothing else to route to when it did
                // not. Either way, this is a no-op at the loop level.
                Ok(ControlEffect::None)
            }
            MiddlewareControl::JumpTo(LoopTarget::Model) => {
                // Abandon whatever the rest of this turn would have done
                // (typically: running tools the model just requested) and go
                // straight to a fresh model call. Close out any tool calls on
                // the last assistant row first so the transcript stays
                // replayable (see the `StopWithFinal` arm below for why).
                Self::close_unanswered_tool_calls(
                    messages,
                    "run jumped back to the model before this tool call was executed",
                );
                Ok(ControlEffect::ContinueLoop)
            }
            MiddlewareControl::JumpTo(LoopTarget::End) => {
                Self::close_unanswered_tool_calls(
                    messages,
                    "run stopped before this tool call was executed",
                );
                if run.final_response.is_none() {
                    let text = Self::last_assistant_text(messages);
                    run.final_response = Some(ModelResponse::assistant(text));
                }
                Ok(ControlEffect::Exit(LoopExit::Finished))
            }
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
                Ok(ControlEffect::Exit(LoopExit::Finished))
            }
            MiddlewareControl::Interrupt { node, message } => {
                Err(TinyAgentsError::Interrupted { node, message })
            }
        }
    }

    /// The text of the most recent assistant message, or empty when there is
    /// none. Used to synthesize a final response for
    /// [`MiddlewareControl::JumpTo`]`(`[`LoopTarget::End`]`)`, which (unlike
    /// [`MiddlewareControl::StopWithFinal`]) carries no text of its own.
    fn last_assistant_text(messages: &[Message]) -> String {
        messages
            .iter()
            .rev()
            .find(|message| matches!(message, Message::Assistant(_)))
            .map(Message::text)
            .unwrap_or_default()
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
    fn build_structured_extractor(
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
        let recording = budget.record(usage);
        ctx.bounded(self.call_budget(ctx), recording, || {
            format!(
                "budget usage recording for run `{}` exceeded its remaining wall-clock deadline",
                ctx.run_id()
            )
        })
        .await
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

/// Recovers text-dialect calls through `tinytools-agent`
/// ([`super::dialect::recover_text_calls`]) while preserving every non-text
/// provider content block (notably reasoning blocks).
///
/// `recovery` carries the tools offered this turn and the P-Format registry;
/// it is already empty when nothing could be recovered — no tools offered,
/// an effective `ToolChoice::None`, or
/// [`crate::runtime::RunPolicy::text_dialect_recovery`] resolving to off for
/// the resolved model (see the `recovery` binding in `run_loop_body`). On top
/// of that, markup that appears only inside a fenced code block is always
/// skipped. Emits [`AgentEvent::ControlApplied`] when it actually rewrites
/// the response, so the recovery is auditable rather than a silent transform.
fn recover_text_dialect_calls<Ctx>(
    ctx: &RunContext<Ctx>,
    response: &mut tinyinference_llm::model::ModelResponse,
    model_call_id: &CallId,
    recovery: &super::dialect::TextRecovery,
) {
    if recovery.offered.is_empty() {
        return;
    }

    if text_dialect_markup_only_in_fenced_code(&response.text()) {
        return;
    }

    let before = response.message.tool_calls.len();
    super::dialect::recover_text_calls(
        response,
        model_call_id,
        &recovery.offered,
        recovery.registry.as_deref(),
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
const DROPPED_TOOL_CALL_NUDGE: &str = "Your previous turn indicated a tool call but none was \
     included. If you meant to call a tool, issue the actual tool call now; otherwise answer \
     directly.";

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
    let answered: std::collections::HashSet<&str> = messages[assistant_at + 1..]
        .iter()
        .filter_map(|message| match message {
            Message::Tool(tool) => Some(tool.tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    let pending: Vec<ToolCall> = assistant
        .tool_calls
        .iter()
        .filter(|call| !answered.contains(call.id.as_str()))
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
    use std::sync::Arc;

    use super::recover_text_dialect_calls;
    use crate::agent_loop::dialect::TextRecovery;
    use crate::context::{RunConfig, RunContext};
    use crate::ids::CallId;
    use tinyinference_llm::model::ModelResponse;
    use tinyinference_llm::tool::ToolSchema;

    fn offered(names: &[&str]) -> TextRecovery {
        TextRecovery {
            offered: Arc::new(
                names
                    .iter()
                    .map(|name| ToolSchema::new(*name, "", serde_json::json!({"type": "object"})))
                    .collect(),
            ),
            registry: None,
        }
    }

    #[test]
    fn text_dialect_markup_is_not_recovered_when_the_request_offered_no_tools() {
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
        let mut response = ModelResponse::assistant(
            "<tool_call><name>shell</name><arguments>{\"command\":\"id\"}</arguments></tool_call>",
        );

        recover_text_dialect_calls(
            &ctx,
            &mut response,
            &CallId::new("model-1"),
            &TextRecovery::default(),
        );

        assert!(response.message.tool_calls.is_empty());
        assert!(response.text().contains("<tool_call>"));
    }

    /// I-2 regression: `RunPolicy::text_dialect_recovery` resolving to off
    /// (what `Auto` yields for a model whose profile reports native tool
    /// calling) is represented as an empty `TextRecovery`, exactly like a
    /// turn that offered no tools — so `<tool_call>` markup the model merely
    /// quoted must not be executed.
    #[test]
    fn text_dialect_markup_is_not_recovered_when_the_policy_disables_it() {
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
        let mut response = ModelResponse::assistant(
            r#"<tool_call>{"name": "shell", "arguments": {"command": "id"}}</tool_call>"#,
        );

        recover_text_dialect_calls(
            &ctx,
            &mut response,
            &CallId::new("model-1"),
            &TextRecovery::default(),
        );

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

        recover_text_dialect_calls(
            &ctx,
            &mut response,
            &CallId::new("model-1"),
            &offered(&["shell"]),
        );

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

        recover_text_dialect_calls(
            &ctx,
            &mut response,
            &CallId::new("model-1"),
            &offered(&["shell"]),
        );

        assert_eq!(response.message.tool_calls.len(), 1);
        assert_eq!(response.message.tool_calls[0].name, "shell");
    }
}
