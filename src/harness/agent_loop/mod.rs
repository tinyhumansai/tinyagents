//! Default model-tool-model agent loop.
//!
//! This module implements the harness's standard execution loop as inherent
//! methods on [`crate::harness::runtime::AgentHarness`]: build a model request,
//! invoke the model (with retry and fallback), execute any requested tools,
//! append the tool results, and repeat until the model produces a final
//! assistant message with no tool calls or a configured limit is reached.
//!
//! # Lifecycle
//!
//! 1. Build a [`RunContext`] from the [`RunConfig`] and emit
//!    [`AgentEvent::RunStarted`].
//! 2. Run `before_agent` middleware.
//! 3. Repeatedly:
//!    - enforce the model-call cap and wall-clock deadline (fail-closed),
//!    - build the [`ModelRequest`] from the working messages, registered tool
//!      schemas, and the policy's default response format,
//!    - run `before_model` middleware, emit [`AgentEvent::ModelStarted`],
//!    - resolve and invoke the model with retry + fallback,
//!    - run `after_model` middleware, emit [`AgentEvent::ModelCompleted`], fold
//!      usage into the [`AgentRun`], append the assistant message,
//!    - if the assistant requested tools, execute each (enforcing the tool-call
//!      cap, running `before_tool`/`after_tool`, emitting tool events) and
//!      append the tool results, then continue,
//!    - otherwise extract structured output when configured and break.
//! 4. Run `after_agent` middleware and emit [`AgentEvent::RunCompleted`].
//!
//! On any error the loop emits [`AgentEvent::RunFailed`], fans the error out
//! through `on_error` middleware, and returns the error.
//!
//! # Limits
//!
//! Model and tool caps come from [`RunPolicy::limits`][crate::harness::runtime::RunPolicy]
//! and are enforced *before* each call, returning
//! [`RustAgentsError::LimitExceeded`]. The wall-clock deadline (from the run
//! config) is checked each iteration and surfaces as
//! [`RustAgentsError::Timeout`]. The run context's own [`crate::harness::limits::LimitTracker`]
//! is also advanced so its counters stay consistent.
//!
//! # Backoff
//!
//! Retry backoff durations are *computed* via
//! [`crate::harness::retry::RetryPolicy::backoff_for_attempt`] but the loop does
//! **not** sleep: keeping the loop sleep-free makes tests fast and
//! deterministic. A real provider integration may choose to sleep for the
//! computed duration before retrying.

mod types;

pub use types::*;

use crate::error::{Result, RustAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, HarnessRunStatus};
use crate::harness::ids::{CallId, ComponentId, HarnessPhase};
use crate::harness::message::Message;
use crate::harness::middleware::AgentRun;
use crate::harness::model::{
    ModelRequest, ModelResolutionSource, ModelResponse, ResolvedModel, ResolvedModelBinding,
    ResponseFormat,
};
use crate::harness::retry::is_retryable;
use crate::harness::runtime::AgentHarness;
use crate::harness::structured::{StructuredExtractor, StructuredStrategy};

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
    /// Returns [`RustAgentsError::LimitExceeded`] when the model- or tool-call
    /// cap is reached, [`RustAgentsError::Timeout`] when the wall-clock deadline
    /// elapses, [`RustAgentsError::ModelNotFound`] when no model can be
    /// resolved, [`RustAgentsError::ToolNotFound`] when the model calls an
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
        self.drive(state, ctx, input).await
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
        self.drive(state, ctx, input).await.map(|result| result.run)
    }

    /// Like [`AgentHarness::invoke_in_context`] but also returns the compact
    /// [`HarnessRunStatus`] snapshot.
    pub async fn invoke_in_context_with_status(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        self.drive(state, ctx, input).await
    }

    /// Shared driver: runs the loop inside `ctx` and owns lifecycle
    /// bookkeeping (status transitions plus `RunFailed`/`on_error` on error).
    async fn drive(
        &self,
        state: &State,
        mut ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> Result<AgentLoopResult> {
        let run_id = ctx.config.run_id.clone();
        let thread_id = ctx.config.thread_id.clone();

        let mut status = HarnessRunStatus::new(run_id.clone(), ComponentId::new("agent_loop"));
        if let Some(thread) = thread_id {
            status = status.with_thread(thread);
        }

        let mut run = AgentRun::new();

        match self
            .run_loop(state, &mut ctx, &mut run, &mut status, input)
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

    /// Drives the loop body, returning `Ok(())` on a clean finish or the first
    /// error encountered. The caller owns lifecycle bookkeeping (final status
    /// transition, `RunFailed`/`on_error` on error).
    async fn run_loop(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        input: Vec<Message>,
    ) -> Result<()> {
        let record = ctx.emit(AgentEvent::RunStarted {
            run_id: ctx.run_id().clone(),
            thread_id: ctx.thread_id().cloned(),
        });
        status.set_last_event(record.id);
        status.mark_running(HarnessPhase::Idle);

        let mut messages = input;

        status.mark_running(HarnessPhase::Middleware);
        self.middleware.run_before_agent(ctx, state).await?;

        loop {
            // Fail-closed limit and deadline checks before each model call.
            ctx.check_deadline().map_err(|_| {
                RustAgentsError::Timeout(format!(
                    "run `{}` exceeded its wall-clock deadline",
                    ctx.run_id()
                ))
            })?;
            if run.model_calls >= self.policy.limits.max_model_calls {
                return Err(RustAgentsError::LimitExceeded(format!(
                    "max model calls ({}) reached",
                    self.policy.limits.max_model_calls
                )));
            }

            // Build the request from the working transcript, tool schemas, and
            // policy response format.
            status.mark_running(HarnessPhase::BuildingRequest);
            let mut request = ModelRequest::new(messages.clone()).with_tools(self.tools.schemas());
            if let Some(format) = &self.policy.default_response_format {
                request = request.with_response_format(format.clone());
            }

            status.mark_running(HarnessPhase::Middleware);
            self.middleware
                .run_before_model(ctx, state, &mut request)
                .await?;

            // Record against the context tracker too (keeps its counters
            // consistent); map its error onto the deterministic limit error.
            ctx.record_model_call().map_err(|_| {
                RustAgentsError::LimitExceeded(format!(
                    "max model calls ({}) reached",
                    self.policy.limits.max_model_calls
                ))
            })?;

            // Resolve the model for the event/log name before invoking.
            let binding = self
                .models
                .resolve_request(&request, None, None)
                .ok_or_else(|| {
                    RustAgentsError::ModelNotFound(
                        request
                            .model
                            .clone()
                            .unwrap_or_else(|| "<default>".to_string()),
                    )
                })?;
            let model_name = binding.resolved.name.clone();

            let call_id = CallId::new(format!("{}-model-{}", ctx.run_id(), run.model_calls + 1));
            status.mark_running(HarnessPhase::Model);
            status.active_model_call = Some(call_id.clone());
            let record = ctx.emit(AgentEvent::ModelStarted {
                call_id: call_id.clone(),
                model: model_name,
            });
            status.set_last_event(record.id);

            let mut response = self
                .invoke_model_with_retry(state, ctx, &request, &call_id, binding)
                .await?;

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
            }
            let record = ctx.emit(AgentEvent::ModelCompleted {
                call_id,
                usage: response.usage,
            });
            status.set_last_event(record.id);

            messages.push(Message::Assistant(response.message.clone()));

            let tool_calls = response.tool_calls().to_vec();
            if tool_calls.is_empty() {
                // Final response: optionally extract structured output.
                if let Some(ResponseFormat::JsonSchema { name, schema }) =
                    &self.policy.default_response_format
                {
                    let extractor = StructuredExtractor::new(
                        StructuredStrategy::ProviderSchema,
                        name.clone(),
                        schema.clone(),
                    );
                    let output = extractor.extract(&response)?;
                    run.structured = Some(output.value);
                }
                run.final_response = Some(response);
                break;
            }

            // Execute requested tools serially.
            status.mark_running(HarnessPhase::Tools);
            for mut call in tool_calls {
                ctx.check_deadline().map_err(|_| {
                    RustAgentsError::Timeout(format!(
                        "run `{}` exceeded its wall-clock deadline",
                        ctx.run_id()
                    ))
                })?;
                if run.tool_calls >= self.policy.limits.max_tool_calls {
                    return Err(RustAgentsError::LimitExceeded(format!(
                        "max tool calls ({}) reached",
                        self.policy.limits.max_tool_calls
                    )));
                }
                ctx.record_tool_call().map_err(|_| {
                    RustAgentsError::LimitExceeded(format!(
                        "max tool calls ({}) reached",
                        self.policy.limits.max_tool_calls
                    ))
                })?;

                self.middleware
                    .run_before_tool(ctx, state, &mut call)
                    .await?;

                let tool = self
                    .tools
                    .get(&call.name)
                    .ok_or_else(|| RustAgentsError::ToolNotFound(call.name.clone()))?;

                let tool_call_id = CallId::new(call.id.clone());
                let tool_name = call.name.clone();
                status.active_tool_calls.push(tool_call_id.clone());
                let record = ctx.emit(AgentEvent::ToolStarted {
                    call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                });
                status.set_last_event(record.id);

                let mut result = tool.call(state, call).await?;

                self.middleware
                    .run_after_tool(ctx, state, &mut result)
                    .await?;

                run.tool_calls += 1;
                status.tool_calls = run.tool_calls;
                status.active_tool_calls.retain(|c| c != &tool_call_id);
                let record = ctx.emit(AgentEvent::ToolCompleted {
                    call_id: tool_call_id,
                    tool_name,
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

    /// Invokes a model with retry and fallback.
    ///
    /// Retries are governed by [`RunPolicy::retry`][crate::harness::runtime::RunPolicy]
    /// and apply only to retryable errors (see [`is_retryable`]); each scheduled
    /// retry emits [`AgentEvent::RetryScheduled`]. When retries are exhausted
    /// (or the error is non-retryable) and a [`crate::harness::retry::FallbackPolicy`]
    /// is configured, the next model in the chain is tried. The computed backoff
    /// duration is intentionally not slept on (see the module docs).
    async fn invoke_model_with_retry(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        call_id: &CallId,
        binding: ResolvedModelBinding<State>,
    ) -> Result<ModelResponse> {
        let mut current_name = binding.resolved.name.clone();
        let mut model = binding.model;
        let mut resolved = binding.resolved;

        loop {
            // Retry loop for the current model.
            let mut attempt = 0usize;
            let outcome = loop {
                match model.invoke(state, request.clone()).await {
                    Ok(response) => break Ok(response),
                    Err(error) => {
                        if is_retryable(&error) && self.policy.retry.should_retry(attempt) {
                            attempt += 1;
                            ctx.emit(AgentEvent::RetryScheduled {
                                call_id: call_id.clone(),
                                attempt,
                            });
                            // Compute (but do not sleep on) the backoff so the
                            // loop stays fast and deterministic in tests.
                            let _ = self.policy.retry.backoff_for_attempt(attempt);
                            continue;
                        }
                        break Err(error);
                    }
                }
            };

            match outcome {
                Ok(mut response) => {
                    if response.resolved_model.is_none() {
                        response.resolved_model = Some(resolved);
                    }
                    return Ok(response);
                }
                Err(error) => {
                    // Retries exhausted (or non-retryable): try the next model
                    // in the fallback chain, if any.
                    let next = self
                        .policy
                        .fallback
                        .as_ref()
                        .and_then(|fallback| fallback.next_after(&current_name))
                        .map(str::to_owned);
                    match next.and_then(|name| self.models.get(&name).map(|m| (name, m))) {
                        Some((name, next_model)) => {
                            resolved = ResolvedModel {
                                name: name.clone(),
                                requested: Some(name.clone()),
                                source: ModelResolutionSource::Hint,
                            };
                            current_name = name;
                            model = next_model;
                            continue;
                        }
                        None => return Err(error),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test;
