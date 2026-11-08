//! Per-turn model-call plumbing for the superstep loop: building the request
//! for a turn and accounting for its response.
//!
//! Split out of `run_loop.rs`.

use super::run_loop::{cacheable_system_prefix_end, mark_empty_frozen_prefix};
use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Builds the turn's request from the working transcript, tool schemas, and
    /// policy response format.
    ///
    /// Goes through `PromptBuilder` rather than constructing `ModelRequest`
    /// directly: a provider KV cache needs an explicit stable prefix, and the
    /// system instructions plus the name-sorted tool schemas are stable between
    /// discoveries. `boosted_max_tokens` is the truncated-empty recovery cap:
    /// a prior attempt this turn exhausted its token budget on the (hidden)
    /// reasoning channel, so the call is re-issued with a larger cap. The boost
    /// deliberately wins over the per-turn cap — that cap is what truncated the
    /// response — and was already clamped to 4x the original budget.
    pub(super) fn build_turn_request(
        &self,
        ctx: &RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        messages: &[Message],
        tool_schemas: &[ToolSchema],
        boosted_max_tokens: Option<u32>,
    ) -> ModelRequest {
        status.mark_running(HarnessPhase::BuildingRequest);
        let system_end = cacheable_system_prefix_end(messages, ctx.frozen_system_prefix_len);
        let mut prompt = crate::prompt::PromptBuilder::new();
        prompt.push_system_messages(&messages[..system_end]);
        if !tool_schemas.is_empty() {
            prompt.push_tools_segment("tools", tool_schemas.to_vec());
        }
        let mut history = messages[system_end..].to_vec();
        // Omit completed turns' reasoning from the request only: `messages`,
        // and so the stored transcript, keeps every block. The current turn's
        // stays, since a provider validates it across the tool loop.
        let replay = ctx
            .config
            .replay_reasoning
            .unwrap_or(self.policy.replay_reasoning);
        if replay == crate::reasoning_replay::ReasoningReplay::CurrentTurn {
            crate::reasoning_replay::drop_past_turn_reasoning(&mut history);
        }
        let mut request = prompt.build(history);
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
                metadata
                    .entry("thread_id")
                    .or_insert_with(|| serde_json::Value::String(thread_id.as_str().to_string()));
            }
        }
        if let Some(format) = &self.policy.default_response_format {
            request = request.with_response_format(format.clone());
        }
        if let Some(cap) = ctx.config.max_turn_output_tokens {
            request.max_tokens = Some(request.max_tokens.map_or(cap, |current| current.min(cap)));
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
        request
    }

    /// Accounts for a completed provider response: bumps the run's call and
    /// step counters, folds the usage into the run totals (a cache replay
    /// consumed no provider tokens, so its usage is surfaced through the
    /// cache-hit event instead of buried in the spend total), and records it
    /// with the host budget. A host-recording failure emits `ModelFailed` and
    /// is returned; it never erases spend the provider already incurred.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn account_model_response(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        response: &ModelResponse,
        call_id: &CallId,
        model_name: &str,
        model_started_at_ms: u64,
        host_budget: &Option<(Arc<dyn crate::host::BudgetGate>, crate::host::Permit)>,
    ) -> Result<()> {
        run.model_calls += 1;
        run.steps += 1;
        status.model_calls = run.model_calls;
        status.active_model_call = None;
        // Responses a wrap middleware discarded and re-requested were billed
        // too: account them before the call that replaced them.
        let discarded_error = self
            .account_discarded_usage(
                ctx,
                run,
                status,
                call_id,
                model_name,
                model_started_at_ms,
                host_budget,
            )
            .await
            .err();
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
                    model: model_name.to_string(),
                    started_at_ms: Some(model_started_at_ms),
                    attempts: None,
                    error: error.to_string(),
                });
                status.set_last_event(record.id);
                return Err(discarded_error.unwrap_or(error));
            }
        }
        if let Some(error) = discarded_error {
            return Err(error);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn account_discarded_usage(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        call_id: &CallId,
        model_name: &str,
        model_started_at_ms: u64,
        host_budget: &Option<(Arc<dyn crate::host::BudgetGate>, crate::host::Permit)>,
    ) -> Result<()> {
        let mut first_error = None;
        for usage in ctx.take_discarded_usage() {
            run.usage.record(usage);
            status.usage = run.usage;
            let record = ctx.emit(AgentEvent::UsageRecorded { usage });
            status.set_last_event(record.id);
            if let Some((budget, _permit)) = &host_budget
                && let Err(error) = self.record_host_usage(ctx, budget, &usage).await
            {
                let record = ctx.emit(AgentEvent::ModelFailed {
                    call_id: call_id.clone(),
                    model: model_name.to_string(),
                    started_at_ms: Some(model_started_at_ms),
                    attempts: None,
                    error: error.to_string(),
                });
                status.set_last_event(record.id);
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
