//! Model invocation: cache-aware retry/fallback dispatch
//! (`invoke_model_with_retry`, `invoke_model_resolving`), the streaming
//! variant, and the innermost `ModelBaseCall`/`ToolBaseCall` impls that
//! the middleware wrap-onion terminates into.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
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
    /// and apply only to retryable errors (see
    /// [`is_retryable`][crate::harness::retry::is_retryable]); each scheduled
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
        // Tracks every model name already attempted in this fallback chain so
        // a chain containing a repeated name (e.g. `[primary, backup,
        // primary]`) cannot alternate between the same two models forever;
        // once a name has been tried it is never tried again.
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(current_name.clone());

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
                    Self::with_call_budget(remaining, run_id.as_str(), "model call", fut).await
                } else {
                    // Race the wall-clock-bounded unary call against cooperative
                    // cancellation, mirroring the streaming path: a cancel
                    // requested while a buffered (non-streamed) provider call is
                    // in flight drops the future — reqwest cancels the underlying
                    // request — and unwinds with `Cancelled` instead of paying
                    // for the call to run to completion. `cancelled()` is
                    // cancel-safe, and the pre-call `is_cancelled()` check above
                    // still short-circuits before the request is ever issued.
                    let cancellation = ctx.cancellation.clone();
                    let fut = model.invoke(state, request.clone());
                    let budgeted =
                        Self::with_call_budget(remaining, run_id.as_str(), "model call", fut);
                    tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => Err(TinyAgentsError::Cancelled),
                        result = budgeted => result,
                    }
                };
                match attempt_result {
                    Ok(response) => break Ok(response),
                    Err(error) => {
                        // `RunLimits::max_retries_per_call` is a hard ceiling
                        // that a looser `RetryPolicy::max_attempts` cannot
                        // exceed; whichever is stricter wins.
                        let max_attempts = self
                            .policy
                            .retry
                            .max_attempts_capped_at(self.policy.limits.max_retries_per_call);
                        // Route the retry decision through the shared
                        // `RetryPolicy::should_retry_error` engine (same
                        // classification + attempt-cap logic RetryMiddleware
                        // uses), applying the harness ceiling by capping a
                        // cloned policy first so the two sites cannot drift.
                        let capped = self.policy.retry.clone().with_max_attempts(max_attempts);
                        if capped.should_retry_error(attempt, &error) {
                            // Compute the backoff from the *pre-increment*
                            // attempt number: `attempt == 0` is the first
                            // retry and must sleep `initial_backoff_ms`
                            // (`RetryPolicy::backoff_for_attempt(0)`). Sleeping
                            // on the post-increment value skipped
                            // `initial_backoff_ms` entirely and shifted the
                            // whole exponential schedule one step too high.
                            let backoff_attempt = attempt;
                            attempt += 1;
                            ctx.emit(AgentEvent::RetryScheduled {
                                call_id: call_id.clone(),
                                attempt,
                            });
                            // Sleep for the backoff only when the policy opts in
                            // (`with_backoff_sleep`); otherwise this is a no-op so
                            // the loop stays fast and deterministic in tests.
                            self.policy.retry.sleep_backoff(backoff_attempt).await;
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
                    // A non-retryable, deadline-driven timeout must not feed
                    // back into the fallback chain: the run itself is out of
                    // wall-clock budget, so trying another model would just
                    // spin until the *next* deadline check fails identically.
                    if matches!(error, TinyAgentsError::Timeout(_)) {
                        return Err(error);
                    }
                    // Retries exhausted (or non-retryable): walk the fallback
                    // chain for the next model, skipping any name already
                    // visited in this chain (so a chain with a repeated name
                    // cannot alternate between the same models forever) and any
                    // candidate that fails the request's capability/lifecycle
                    // gate. Initial resolution gates the primary selection
                    // through `model_eligible`; without the same gate here a
                    // primary failure could silently fall back to a model that
                    // can't call tools, lacks vision, or has a smaller context
                    // window (issue #4641). `allow_retired` is `false` to match
                    // `ModelRegistry::resolve_request`.
                    let required = request.required_capabilities.as_ref();
                    let mut cursor = current_name.clone();
                    let selected = loop {
                        let next = self
                            .policy
                            .fallback
                            .as_ref()
                            .and_then(|fallback| fallback.next_after(&cursor))
                            .map(str::to_owned)
                            .filter(|name| !visited.contains(name));
                        let Some((name, next_model)) =
                            next.and_then(|name| self.models.get(&name).map(|m| (name, m)))
                        else {
                            break None;
                        };
                        if !model_eligible(next_model.as_ref(), required, false) {
                            // Ineligible candidate: record it as visited, make
                            // the skip observable, and keep walking the chain.
                            visited.insert(name.clone());
                            ctx.emit(AgentEvent::FallbackSkipped {
                                model: name.clone(),
                            });
                            cursor = name;
                            continue;
                        }
                        break Some((name, next_model));
                    };
                    match selected {
                        Some((name, next_model)) => {
                            visited.insert(name.clone());
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
    pub(super) fn call_budget(&self, ctx: &RunContext<Ctx>) -> Option<Duration> {
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

    /// Awaits a single call future (model or tool), optionally bounded by
    /// `budget`.
    ///
    /// When `budget` is `Some`, the future is wrapped in
    /// [`tokio::time::timeout`]; if it elapses the future is dropped (cancelling
    /// the in-flight provider/tool request) and a
    /// [`TinyAgentsError::Timeout`] is returned. When `budget` is `None` (no
    /// run timeout configured) the future is awaited without a bound.
    ///
    /// `budget` is the run's *remaining* wall-clock budget at the time the call
    /// is issued, so each successive call gets a tighter bound as the deadline
    /// approaches. `what` names the kind of call in the timeout message (e.g.
    /// `"model call"`, `"tool call"`).
    pub(super) async fn with_call_budget<T, F>(
        budget: Option<Duration>,
        run_id: &str,
        what: &str,
        fut: F,
    ) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match budget {
            Some(budget) => match tokio::time::timeout(budget, fut).await {
                Ok(result) => result,
                Err(_) => Err(TinyAgentsError::Timeout(format!(
                    "{what} for run `{run_id}` exceeded its remaining wall-clock budget \
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
                    reasoning: String::new(),
                    tool_call: Some(tool_delta.clone()),
                }),
                _ => None,
            };

            if let Some(message_delta) = message_delta {
                // Build the middleware-facing delta first (it needs owned
                // copies of the fields), then move `message_delta` into the
                // event so the hot path clones the payload once instead of
                // twice per streamed token.
                let mut model_delta = ModelDelta {
                    call_id: call_id.as_str().to_string(),
                    content: message_delta.text.clone(),
                    reasoning: message_delta.reasoning.clone(),
                    tool_call: message_delta.tool_call.clone(),
                };
                ctx.emit(AgentEvent::ModelDelta {
                    run_id: ctx.config.run_id.clone(),
                    call_id: call_id.clone(),
                    delta: message_delta,
                });
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
pub(super) struct ModelCallBase<'h, State: Send + Sync, Ctx: Send + Sync> {
    pub(super) harness: &'h AgentHarness<State, Ctx>,
    pub(super) call_id: CallId,
    pub(super) resolved: ResolvedModel,
    pub(super) model: Arc<dyn ChatModel<State>>,
    pub(super) streaming: bool,
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
                .invoke_model_with_retry(
                    state,
                    ctx,
                    &request,
                    &self.call_id,
                    binding,
                    self.streaming,
                )
                .await
        })
    }
}

/// The innermost tool call wrapped by the tool-wrap onion.
///
/// Implements [`ToolBaseCall`] over a single resolved [`Tool`] so a
/// [`crate::harness::middleware::ToolMiddleware`] can wrap the real tool
/// invocation.
pub(super) struct ToolCallBase<State: Send + Sync> {
    pub(super) tool: Arc<dyn Tool<State>>,
    pub(super) timeout_settings: Option<crate::harness::tool::ToolTimeoutSettings>,
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolBaseCall<State, Ctx> for ToolCallBase<State> {
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let timeout = self
                .timeout_settings
                .as_ref()
                .map(|settings| settings.resolve(self.tool.timeout_policy(&call)));
            let timeout_result = super::tools::timeout_result(&call, timeout);
            let future = self.tool.call_with_context(
                state,
                call,
                crate::harness::tool::ToolExecutionContext::from_run_context(ctx),
            );
            match timeout.and_then(|resolved| resolved.deadline) {
                Some(deadline) => match tokio::time::timeout(deadline, future).await {
                    Ok(result) => result,
                    Err(_) => Ok(timeout_result),
                },
                None => future.await,
            }
        })
    }
}
