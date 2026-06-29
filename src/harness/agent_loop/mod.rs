//! Default model-tool-model agent loop.
//!
//! This loop is the innermost turn of the recursive-language-model (RLM)
//! runtime: it is where one model call is driven to completion, and because a
//! whole harness can be exposed as a tool
//! ([`crate::harness::subagent::SubAgentTool`]), the very tools this loop
//! executes may themselves be other agents — so "a model calling a model" is
//! just this loop nested inside one of its own tool calls. Each invocation runs
//! inside a [`RunContext`] that tracks recursion depth, fans usage/cost up to a
//! parent run, and observes cooperative cancellation and steering at safe
//! checkpoints.
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
//! [`TinyAgentsError::LimitExceeded`]. The wall-clock deadline (from the run
//! config) is checked each iteration and surfaces as
//! [`TinyAgentsError::Timeout`]. The run context's own [`crate::harness::limits::LimitTracker`]
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

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{Result, TinyAgentsError};
use crate::harness::cache::{ResponseCache, cache_key};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, HarnessRunStatus};
use crate::harness::ids::{CallId, ComponentId, HarnessPhase};
use crate::harness::message::{Message, MessageDelta};
use crate::harness::middleware::{AgentRun, BoxModelFuture, BoxToolFuture, ModelBaseCall, ToolBaseCall};
use crate::harness::model::{
    ChatModel, ModelDelta, ModelRequest, ModelResolutionSource, ModelResponse, ModelStreamItem,
    ResolvedModel, ResolvedModelBinding, ResponseFormat, StreamAccumulator, ToolChoice,
};
use crate::harness::retry::is_retryable;
use crate::harness::runtime::AgentHarness;
use crate::harness::structured::{StructuredExtractor, StructuredStrategy};
use crate::harness::tool::{Tool, ToolCall, ToolSchema};
use futures::StreamExt;
use serde_json::Value;

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
        streaming: bool,
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
            ctx.check_deadline().map_err(|_| {
                TinyAgentsError::Timeout(format!(
                    "run `{}` exceeded its wall-clock deadline",
                    ctx.run_id()
                ))
            })?;
            if run.model_calls >= self.policy.limits.max_model_calls {
                return Err(TinyAgentsError::LimitExceeded(format!(
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
                TinyAgentsError::LimitExceeded(format!(
                    "max model calls ({}) reached",
                    self.policy.limits.max_model_calls
                ))
            })?;

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
            }
            let record = ctx.emit(AgentEvent::ModelCompleted {
                call_id,
                usage: response.usage,
            });
            status.set_last_event(record.id);

            messages.push(Message::Assistant(response.message.clone()));

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
                ctx.check_deadline().map_err(|_| {
                    TinyAgentsError::Timeout(format!(
                        "run `{}` exceeded its wall-clock deadline",
                        ctx.run_id()
                    ))
                })?;
                if run.tool_calls >= self.policy.limits.max_tool_calls {
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "max tool calls ({}) reached",
                        self.policy.limits.max_tool_calls
                    )));
                }
                ctx.record_tool_call().map_err(|_| {
                    TinyAgentsError::LimitExceeded(format!(
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
                    .ok_or_else(|| TinyAgentsError::ToolNotFound(call.name.clone()))?;

                let tool_call_id = CallId::new(call.id.clone());
                let tool_name = call.name.clone();
                status.active_tool_calls.push(tool_call_id.clone());
                let record = ctx.emit(AgentEvent::ToolStarted {
                    call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                });
                status.set_last_event(record.id);

                // The real tool call is the innermost base of the tool-wrap
                // onion (same before -> wrap -> after ordering as the model
                // path): lifecycle `before_tool` ran above, the wrap onion runs
                // here, and lifecycle `after_tool` runs below.
                let base = ToolCallBase { tool };
                let mut result = self
                    .middleware
                    .run_wrapped_tool(ctx, state, call, &base)
                    .await?
                    .into_result();

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
    fn response_cache_decision(
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
        Some((Arc::clone(cache), cache_key(request)))
    }

    /// Invokes a model, consulting the local response cache around the
    /// retry/fallback path.
    ///
    /// When caching is enabled for this call (see
    /// [`Self::response_cache_decision`]) the cache is checked **before** any
    /// provider call: on a hit an [`AgentEvent::CacheHit`] is emitted and the
    /// cached [`ModelResponse`] is returned *without* invoking the underlying
    /// [`ChatModel`] (the retry/fallback path is skipped entirely); on a miss an
    /// [`AgentEvent::CacheMiss`] is emitted, the provider is invoked normally,
    /// and the successful response is written back to the cache.
    ///
    /// # Accounting
    ///
    /// A cache hit is still counted as a model "step"/call by the caller
    /// ([`Self::run_loop`] increments `model_calls`/`steps` and emits
    /// [`AgentEvent::ModelCompleted`] after this returns) so usage and limit
    /// bookkeeping stay consistent whether or not a call was served from cache.
    /// The behavioral guarantee is only that the underlying provider is not
    /// contacted on a hit.
    async fn invoke_model_with_retry(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        call_id: &CallId,
        binding: ResolvedModelBinding<State>,
        streaming: bool,
    ) -> Result<ModelResponse> {
        let decision = self.response_cache_decision(request);

        if let Some((cache, key)) = decision.as_ref() {
            if let Some(mut cached) = cache.get(key).await? {
                ctx.emit(AgentEvent::CacheHit {
                    call_id: call_id.clone(),
                    key: key.clone(),
                });
                if cached.resolved_model.is_none() {
                    cached.resolved_model = Some(binding.resolved.clone());
                }
                return Ok(cached);
            }
            ctx.emit(AgentEvent::CacheMiss {
                call_id: call_id.clone(),
                key: key.clone(),
            });
        }

        let response = self
            .invoke_model_resolving(state, ctx, request, call_id, binding, streaming)
            .await?;

        if let Some((cache, key)) = decision.as_ref() {
            cache.put(key, response.clone()).await?;
        }

        Ok(response)
    }

    /// Invokes a model with retry and fallback (no caching).
    ///
    /// Retries are governed by [`RunPolicy::retry`][crate::harness::runtime::RunPolicy]
    /// and apply only to retryable errors (see [`is_retryable`]); each scheduled
    /// retry emits [`AgentEvent::RetryScheduled`]. When retries are exhausted
    /// (or the error is non-retryable) and a [`crate::harness::retry::FallbackPolicy`]
    /// is configured, the next model in the chain is tried. The computed backoff
    /// duration is intentionally not slept on (see the module docs).
    async fn invoke_model_resolving(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        call_id: &CallId,
        binding: ResolvedModelBinding<State>,
        streaming: bool,
    ) -> Result<ModelResponse> {
        let mut current_name = binding.resolved.name.clone();
        let mut model = binding.model;
        let mut resolved = binding.resolved;
        let run_id = ctx.run_id().clone();

        loop {
            // Retry loop for the current model.
            let mut attempt = 0usize;
            let outcome = loop {
                // Observe cancellation before (re)issuing a model attempt so a
                // cancel requested during a retry/rate-limit wait stops the run
                // promptly instead of firing another provider call or falling
                // through to the fallback chain.
                if ctx.cancellation.is_cancelled() {
                    return Err(TinyAgentsError::Cancelled);
                }
                // Bound this individual provider call by the run's *remaining*
                // wall-clock budget so a hung or slow model call is interrupted
                // mid-flight, not merely detected by the between-call deadline
                // check. reqwest/futures are cancel-safe, so dropping the future
                // on elapse cancels the underlying request. When neither the run
                // config nor the harness policy configures a timeout the call is
                // awaited unbounded.
                let remaining = self.call_budget(ctx);
                let attempt_result = if streaming {
                    let fut =
                        self.invoke_model_streaming_once(state, ctx, &model, request, call_id);
                    Self::with_call_budget(remaining, run_id.as_str(), fut).await
                } else {
                    let fut = model.invoke(state, request.clone());
                    Self::with_call_budget(remaining, run_id.as_str(), fut).await
                };
                match attempt_result {
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

    /// Computes the wall-clock budget for the next individual model call.
    ///
    /// The budget is the *tighter* of two remaining-time sources:
    ///
    /// - the run config's `timeout_ms` (the same deadline the between-call
    ///   [`RunContext::check_deadline`] enforces), tracked by the run's
    ///   [`crate::harness::limits::LimitTracker`], and
    /// - the harness policy's
    ///   [`RunLimits::max_wall_clock_ms`][crate::harness::limits::RunLimits::max_wall_clock_ms],
    ///   measured against the same tracker start.
    ///
    /// Either source may be absent; when both are absent the call is unbounded
    /// (`None`). Honoring the policy source lets a sub-agent whose child
    /// [`RunConfig`] carries no per-run timeout still be bounded by its
    /// harness's policy-level wall-clock cap.
    fn call_budget(&self, ctx: &RunContext<Ctx>) -> Option<Duration> {
        let config_budget = ctx.remaining_wall_clock();
        let policy_budget = self.policy.limits.max_wall_clock_ms.map(|ms| {
            Duration::from_millis(ms)
                .checked_sub(ctx.limits.elapsed())
                .unwrap_or(Duration::ZERO)
        });
        match (config_budget, policy_budget) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// Awaits a single model-call future, optionally bounded by `budget`.
    ///
    /// When `budget` is `Some`, the future is wrapped in
    /// [`tokio::time::timeout`]; if it elapses the future is dropped (cancelling
    /// the in-flight provider request) and a
    /// [`TinyAgentsError::Timeout`] is returned. When `budget` is `None` (no
    /// run timeout configured) the future is awaited without a bound.
    ///
    /// `budget` is the run's *remaining* wall-clock budget at the time the call
    /// is issued, so each successive call gets a tighter bound as the deadline
    /// approaches.
    async fn with_call_budget<F>(budget: Option<Duration>, run_id: &str, fut: F) -> F::Output
    where
        F: Future<Output = Result<ModelResponse>>,
    {
        match budget {
            Some(budget) => match tokio::time::timeout(budget, fut).await {
                Ok(result) => result,
                Err(_) => Err(TinyAgentsError::Timeout(format!(
                    "model call for run `{run_id}` exceeded its remaining wall-clock budget \
                     ({} ms)",
                    budget.as_millis()
                ))),
            },
            None => fut.await,
        }
    }

    /// Drives one streaming model call to completion.
    ///
    /// Consumes [`crate::harness::model::ChatModel::stream`], emitting an
    /// [`AgentEvent::ModelDelta`] and running every middleware's
    /// [`on_model_delta`][crate::harness::middleware::Middleware::on_model_delta]
    /// hook for each [`ModelStreamItem::MessageDelta`] (and standalone
    /// [`ModelStreamItem::ToolCallDelta`]), then folds the items into the final
    /// [`ModelResponse`] via [`StreamAccumulator`]. The merged response is
    /// equivalent to what the unary [`crate::harness::model::ChatModel::invoke`]
    /// path would have produced, so the rest of the loop is unaffected.
    async fn invoke_model_streaming_once(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        model: &Arc<dyn ChatModel<State>>,
        request: &ModelRequest,
        call_id: &CallId,
    ) -> Result<ModelResponse> {
        let mut stream = model.stream(state, request.clone()).await?;
        let mut accumulator = StreamAccumulator::new();

        // Clone the cheap token so the cancellation future does not borrow
        // `ctx` for the duration of the stream loop (the body still needs
        // `&mut ctx` for events and middleware).
        let cancellation = ctx.cancellation.clone();

        loop {
            // Race the next provider chunk against cooperative cancellation. If
            // cancellation wins we drop the partially consumed stream and unwind
            // with `Cancelled`; the `cancelled()` future is cancel-safe.
            let item = tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    return Err(TinyAgentsError::Cancelled);
                }
                next = stream.next() => match next {
                    Some(item) => item,
                    None => break,
                },
            };

            // Surface incremental message/tool-call fragments through events and
            // the `on_model_delta` middleware hook before merging them.
            let message_delta = match &item {
                ModelStreamItem::MessageDelta(delta) => Some(delta.clone()),
                ModelStreamItem::ToolCallDelta(tool_delta) => Some(MessageDelta {
                    text: String::new(),
                    tool_call: Some(tool_delta.clone()),
                }),
                _ => None,
            };

            if let Some(message_delta) = message_delta {
                let record = ctx.emit(AgentEvent::ModelDelta {
                    call_id: call_id.clone(),
                    delta: message_delta.clone(),
                });
                let _ = record;

                let mut model_delta = ModelDelta {
                    call_id: call_id.as_str().to_string(),
                    content: message_delta.text.clone(),
                    tool_call: message_delta.tool_call.clone(),
                };
                self.middleware
                    .run_on_model_delta(ctx, state, &mut model_delta)
                    .await?;
            }

            accumulator.push(&item);
        }

        accumulator.finish()
    }
}

/// The innermost model call wrapped by the model-wrap onion.
///
/// Implements [`ModelBaseCall`] over the harness's cache + retry + fallback core
/// ([`AgentHarness::invoke_model_with_retry`]) so a [`crate::harness::middleware::ModelMiddleware`]
/// can proceed, short-circuit, retry, or fall back around the *whole* real model
/// call. The resolved binding is rebuilt per invocation so a wrap middleware
/// that retries `next` issues a fresh provider call each time.
struct ModelCallBase<'h, State: Send + Sync, Ctx: Send + Sync> {
    harness: &'h AgentHarness<State, Ctx>,
    call_id: CallId,
    resolved: ResolvedModel,
    model: Arc<dyn ChatModel<State>>,
    streaming: bool,
}

impl<State: Send + Sync, Ctx: Send + Sync> ModelBaseCall<State, Ctx>
    for ModelCallBase<'_, State, Ctx>
{
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        Box::pin(async move {
            let binding = ResolvedModelBinding {
                resolved: self.resolved.clone(),
                model: Arc::clone(&self.model),
            };
            self.harness
                .invoke_model_with_retry(state, ctx, &request, &self.call_id, binding, self.streaming)
                .await
        })
    }
}

/// The innermost tool call wrapped by the tool-wrap onion.
///
/// Implements [`ToolBaseCall`] over a single resolved [`Tool`] so a
/// [`crate::harness::middleware::ToolMiddleware`] can wrap the real tool
/// invocation.
struct ToolCallBase<State: Send + Sync> {
    tool: Arc<dyn Tool<State>>,
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolBaseCall<State, Ctx> for ToolCallBase<State> {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move { self.tool.call(state, call).await })
    }
}

#[cfg(test)]
mod test;
