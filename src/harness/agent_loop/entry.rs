//! Public entry points for the default agent loop: `invoke*`/
//! `invoke_streaming*` and the shared `drive` lifecycle wrapper.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Runs the default agent loop and returns the accumulated [`AgentRun`].
    ///
    /// `state` is shared, read-only application data passed to every model and
    /// tool call. `ctx_data` is moved into the [`RunContext`] for the run.
    /// `config` supplies the run identity and limits, and `input` seeds the
    /// working message transcript.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::LimitExceeded`] when the model- or tool-call
    /// cap is reached, [`TinyAgentsError::Timeout`] when the wall-clock deadline
    /// elapses, [`TinyAgentsError::ModelNotFound`] when no model can be
    /// resolved, [`TinyAgentsError::ToolNotFound`] when the model calls an
    /// unregistered tool, or any error surfaced by a model, tool, middleware,
    /// or structured-output extraction.
    pub async fn invoke(
        &self,
        state: &State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        self.invoke_with_status(state, ctx_data, config, input)
            .await
            .map(|result| result.run)
    }

    /// Runs the default agent loop with a generated default [`RunConfig`].
    ///
    /// Builds `RunConfig::new("run")` and a default `Ctx`. Identifiers are
    /// derived deterministically from the config (no random or time-based ids),
    /// so repeated calls with the same input behave identically.
    pub async fn invoke_default(&self, state: &State, input: Vec<Message>) -> Result<AgentRun>
    where
        Ctx: Default,
    {
        self.invoke(state, Ctx::default(), RunConfig::new("run"), input)
            .await
    }

    /// Runs the default agent loop and returns both the [`AgentRun`] and a
    /// compact [`HarnessRunStatus`] snapshot describing how the run ended.
    ///
    /// This is the underlying entry point used by [`AgentHarness::invoke`]; use
    /// it directly when you also need lifecycle/status information (phase,
    /// counters, timing, error summary). On error the returned status would have
    /// been marked failed, but the error is propagated instead so callers see
    /// the failure; use the event stream for failed-run status.
    pub async fn invoke_with_status(
        &self,
        state: &State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        let ctx = RunContext::new(config, ctx_data);
        self.drive(state, ctx, input, false).await
    }

    /// Runs the default agent loop inside a caller-supplied [`RunContext`],
    /// returning the accumulated [`AgentRun`].
    ///
    /// Use this when you need to control the run's dependencies — for example
    /// to attach your own [`crate::harness::events::EventSink`] (so an external
    /// listener or [`crate::harness::testkit::EventRecorder`] receives every
    /// event), inject a custom [`crate::harness::store::StoreRegistry`], or carry
    /// pre-populated `Ctx` data. The context's [`RunConfig`] supplies the run
    /// identity and limits, exactly as for [`AgentHarness::invoke`].
    ///
    /// # Errors
    ///
    /// Identical to [`AgentHarness::invoke`].
    pub async fn invoke_in_context(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        self.drive(state, ctx, input, false)
            .await
            .map(|result| result.run)
    }

    /// Like [`AgentHarness::invoke_in_context`] but also returns the compact
    /// [`HarnessRunStatus`] snapshot.
    pub async fn invoke_in_context_with_status(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        self.drive(state, ctx, input, false).await
    }

    /// Streaming counterpart of [`AgentHarness::invoke`].
    ///
    /// Behaves exactly like [`AgentHarness::invoke`] except each model call is
    /// driven through [`crate::harness::model::ChatModel::stream`] rather than
    /// [`crate::harness::model::ChatModel::invoke`]: incremental message deltas
    /// are emitted as [`AgentEvent::ModelDelta`] events and threaded through
    /// every middleware's
    /// [`on_model_delta`][crate::harness::middleware::Middleware::on_model_delta]
    /// hook before the chunks are merged back into the final
    /// [`crate::harness::model::ModelResponse`]. Tool execution, limits, retry,
    /// fallback, structured output, and all other lifecycle behavior are
    /// identical to the non-streaming path.
    pub async fn invoke_streaming(
        &self,
        state: &State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        let ctx = RunContext::new(config, ctx_data);
        self.drive(state, ctx, input, true)
            .await
            .map(|result| result.run)
    }

    /// Streaming counterpart of [`AgentHarness::invoke_default`].
    pub async fn invoke_streaming_default(
        &self,
        state: &State,
        input: Vec<Message>,
    ) -> Result<AgentRun>
    where
        Ctx: Default,
    {
        self.invoke_streaming(state, Ctx::default(), RunConfig::new("run"), input)
            .await
    }

    /// Streaming counterpart of [`AgentHarness::invoke_in_context`].
    pub async fn invoke_streaming_in_context(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        self.drive(state, ctx, input, true)
            .await
            .map(|result| result.run)
    }

    /// Streaming counterpart of [`AgentHarness::invoke_in_context_with_status`].
    pub async fn invoke_streaming_in_context_with_status(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        self.drive(state, ctx, input, true).await
    }

    /// Shared driver: runs the loop inside `ctx` and owns lifecycle
    /// bookkeeping (status transitions plus `RunFailed`/`on_error` on error).
    ///
    /// `streaming` selects whether each model call is driven through
    /// [`crate::harness::model::ChatModel::stream`] (firing `on_model_delta`
    /// middleware per delta) or the unary
    /// [`crate::harness::model::ChatModel::invoke`] path.
    async fn drive(
        &self,
        state: &State,
        mut ctx: RunContext<Ctx>,
        input: Vec<Message>,
        streaming: bool,
    ) -> Result<AgentLoopResult> {
        let run_id = ctx.config.run_id.clone();
        let thread_id = ctx.config.thread_id.clone();

        let mut status = HarnessRunStatus::new(run_id.clone(), ComponentId::new("agent_loop"));
        if let Some(thread) = thread_id {
            status = status.with_thread(thread);
        }

        let mut run = AgentRun::new();

        match self
            .run_loop(state, &mut ctx, &mut run, &mut status, input, streaming)
            .await
        {
            Ok(()) => {
                status.mark_completed();
                Ok(AgentLoopResult { run, status })
            }
            Err(error) => {
                let record = ctx.emit(AgentEvent::RunFailed {
                    run_id,
                    error: error.to_string(),
                });
                status.set_last_event(record.id);
                status.mark_failed(error.to_string());
                // Surface the failure to every middleware. Inner errors are
                // ignored so the originating error is never masked.
                let _ = self.middleware.run_on_error(&mut ctx, &error).await;
                Err(error)
            }
        }
    }
}
