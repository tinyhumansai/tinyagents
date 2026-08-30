//! Model invocation: cache-aware retry/fallback dispatch
//! (`invoke_model_with_retry`, `invoke_model_resolving`), the streaming
//! variant, and the innermost `ModelBaseCall`/`ToolBaseCall` impls that
//! the middleware wrap-onion terminates into.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

/// Timeout-message label for a call bounded by the run's remaining wall-clock
/// budget: the run is out of time, not (necessarily) this call wedged.
pub(super) const RUN_BOUND_LABEL: &str = "remaining wall-clock budget";

/// Timeout-message label for a model call bounded by the per-call ceiling
/// ([`RunLimits::max_model_call_ms`][crate::limits::RunLimits::max_model_call_ms]):
/// this one call ran past the time any single call is allowed, with run time
/// still left.
pub(super) const PER_CALL_BOUND_LABEL: &str = "per-model-call ceiling";

use super::*;
use crate::cache::{
    CacheSkipReason, apply_prompt_cache_breakpoints, scoped_cache_key,
};
use tinyinference::cache::CachePolicy;

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
    /// # Key composition
    ///
    /// The key is **not** the request hash alone. [`Self::response_cache_decision`]
    /// produces the request half ([`cache_key`]); this method folds in the
    /// *resolved* model's
    /// [`cache_identity`][tinyinference::model::ChatModel::cache_identity], the
    /// `streaming` flag, and the policy namespace via
    /// [`scoped_cache_key`][crate::cache::scoped_cache_key]. All three
    /// were previously absent from the key:
    ///
    /// * the request's `model` field is never set by the loop and the endpoint
    ///   and credential live inside the `Arc<dyn ChatModel>`, so one shared
    ///   cache served a hosted harness's answer to a local one;
    /// * `streaming` is a parameter of this function, not a request field, so a
    ///   warm streaming run could be served an entry written by a unary run.
    ///
    /// # Accounting
    ///
    /// A cache hit is still counted as a model "step"/call by the caller
    /// ([`Self::run_loop`] increments `model_calls`/`steps` and emits
    /// [`AgentEvent::ModelCompleted`] after this returns) so limit bookkeeping
    /// stays consistent whether or not a call was served from cache. The hit is
    /// stamped [`ModelResponse::served_from_cache`] so *token/cost* accounting
    /// can tell a replay from a real call and not re-bill it.
    ///
    /// # Failure policy
    ///
    /// A cache is an optimization. Neither a failed lookup nor a failed write
    /// fails the run: a read error degrades to a miss, and a write error is
    /// logged and dropped — the provider call has already succeeded and been
    /// paid for, so discarding its answer because the cache was unavailable is
    /// strictly worse than not caching at all.
    async fn invoke_model_with_retry(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        call_id: &CallId,
        binding: ResolvedModelBinding<State>,
        streaming: bool,
    ) -> Result<ModelResponse> {
        let policy = self.effective_cache_policy(request);
        // The identity of the model that is actually about to be called — known
        // only *after* resolution, which is why the key cannot be finalized by
        // the request-hashing half alone.
        let identity = binding.model.cache_identity();
        let primary_name = binding.resolved.name.clone();

        let decision = self.response_cache_decision(request).map(|(cache, base)| {
            let key = scoped_cache_key(
                &base,
                identity.as_deref(),
                streaming,
                policy.namespace.as_deref(),
            );
            (cache, key)
        });

        if decision.is_none() {
            let reason = self.cache_skip_reason(request);
            tinyagents_tracing::debug!(
                call_id = %call_id.as_str(),
                reason = reason.as_str(),
                "[cache] response cache not consulted for this model call"
            );
        }

        if let Some((cache, key)) = decision.as_ref() {
            // A read failure is a miss, not a run failure: `InMemoryResponseCache`
            // reports a poisoned mutex as a `Validation` error, and the trait is
            // explicitly designed for third-party implementations whose failure
            // modes we do not control.
            let looked_up = match cache.get(key).await {
                Ok(hit) => hit,
                Err(error) => {
                    tinyagents_tracing::warn!(
                        call_id = %call_id.as_str(),
                        %error,
                        "[cache] response-cache lookup failed; treating as a miss"
                    );
                    None
                }
            };
            if let Some(mut cached) = looked_up {
                ctx.emit(AgentEvent::CacheHit {
                    call_id: call_id.clone(),
                    key: key.clone(),
                });
                cached.served_from_cache = true;
                if cached.resolved_model.is_none() {
                    cached.resolved_model = Some(binding.resolved.clone());
                }
                if streaming {
                    self.replay_cached_response_as_deltas(state, ctx, call_id, &cached)
                        .await?;
                }
                return Ok(cached);
            }
            ctx.emit(AgentEvent::CacheMiss {
                call_id: call_id.clone(),
                key: key.clone(),
            });
        }

        // Provider prompt-cache breakpoints are injected *after* the key is
        // derived (they mutate `provider_options`, which the key covers) and
        // only when the policy asks for prefix protection, so the common path
        // never pays for a request clone.
        let mut breakpointed;
        let effective_request = if policy.protect_prompt_prefix {
            breakpointed = request.clone();
            apply_prompt_cache_breakpoints(&mut breakpointed);
            &breakpointed
        } else {
            request
        };

        let response = self
            .invoke_model_resolving(state, ctx, effective_request, call_id, binding, streaming)
            .await?;

        if let Some((cache, key)) = decision.as_ref() {
            // Only the *primary* model's answer may be stored under this key.
            // `invoke_model_resolving` walks the fallback chain on failure and
            // can return a different model's response; writing that under the
            // primary's key poisons it — permanently, when no TTL is set — so
            // every later run of the primary silently gets the fallback's
            // answer, and reports model B after `ModelStarted` announced A.
            let served_by = response
                .resolved_model
                .as_ref()
                .map(|resolved| resolved.name.as_str());
            if served_by.is_some_and(|name| name != primary_name) {
                tinyagents_tracing::debug!(
                    call_id = %call_id.as_str(),
                    primary = %primary_name,
                    served_by = served_by.unwrap_or_default(),
                    "[cache] skipping write: a fallback model answered, not the keyed model"
                );
            } else if let Err(error) = cache
                .put_with_ttl(key, response.clone(), policy.ttl())
                .await
            {
                // The provider call already succeeded and was paid for.
                // Discarding its answer because the cache is unavailable would
                // be strictly worse than not caching.
                tinyagents_tracing::warn!(
                    call_id = %call_id.as_str(),
                    %error,
                    "[cache] response-cache write failed; returning the response uncached"
                );
            }
        }

        Ok(response)
    }

    /// Resolves the effective [`CachePolicy`] for `request`: the per-request
    /// policy when present, otherwise the harness-level
    /// [`RunPolicy::cache`][crate::runtime::RunPolicy].
    pub(super) fn effective_cache_policy(&self, request: &ModelRequest) -> CachePolicy {
        request
            .cache_policy
            .clone()
            .unwrap_or_else(|| self.policy.cache.clone())
    }

    /// Explains why [`Self::response_cache_decision`] declined to consult the
    /// cache, for the diagnostic log line on the skip path.
    ///
    /// `docs/modules/harness/cache.md` specifies a richer decision surface than
    /// the bare `CacheHit`/`CacheMiss` pair; without this a caller seeing a 0%
    /// hit rate cannot tell "no cache attached" from "policy off" from "every
    /// request was multi-turn".
    pub(super) fn cache_skip_reason(&self, request: &ModelRequest) -> CacheSkipReason {
        if self.response_cache.is_none() {
            return CacheSkipReason::NoCacheAttached;
        }
        if !self.effective_cache_policy(request).response_cache_enabled {
            return CacheSkipReason::PolicyDisabled;
        }
        CacheSkipReason::MultiTurnTranscript
    }

    /// Replays a cache hit as synthetic stream deltas so a warm streaming run
    /// is observationally identical to a cold one.
    ///
    /// A cache hit short-circuits before [`Self::invoke_model_streaming_once`],
    /// so a streaming run served from cache used to emit **zero**
    /// [`AgentEvent::ModelDelta`] events and run **zero**
    /// [`on_model_delta`][crate::middleware::Middleware::on_model_delta]
    /// hooks — a UI rendering deltas showed nothing at all, contradicting the
    /// streaming contract documented on the harness entry points. LangChain
    /// replays hits as synthetic stream events for exactly this reason.
    ///
    /// The replay emits one text delta (when the cached message has text) and
    /// one delta per cached tool call, mirroring what the provider stream would
    /// have produced. It is a replay, not a re-derivation: no provider is
    /// contacted.
    async fn replay_cached_response_as_deltas(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        call_id: &CallId,
        cached: &ModelResponse,
    ) -> Result<()> {
        let text = cached.text();
        let tool_calls = cached.tool_calls().to_vec();
        tinyagents_tracing::debug!(
            call_id = %call_id.as_str(),
            text_len = text.len(),
            tool_calls = tool_calls.len(),
            "[cache] replaying a cache hit as synthetic stream deltas"
        );

        let mut deltas: Vec<MessageDelta> = Vec::new();
        if !text.is_empty() {
            deltas.push(MessageDelta {
                text,
                reasoning: String::new(),
                tool_call: None,
            });
        }
        for call in &tool_calls {
            deltas.push(MessageDelta {
                text: String::new(),
                reasoning: String::new(),
                tool_call: Some(crate::tool::ToolDelta {
                    call_id: call.id.clone(),
                    content: serde_json::to_string(&call.arguments).unwrap_or_default(),
                    tool_name: Some(call.name.clone()),
                }),
            });
        }

        for delta in deltas {
            let mut model_delta = ModelDelta {
                call_id: call_id.as_str().to_string(),
                content: delta.text.clone(),
                reasoning: delta.reasoning.clone(),
                tool_call: delta.tool_call.clone(),
            };
            ctx.emit(AgentEvent::ModelDelta {
                run_id: ctx.config.run_id.clone(),
                call_id: call_id.clone(),
                delta,
            });
            self.middleware
                .run_on_model_delta(ctx, state, &mut model_delta)
                .await?;
        }
        Ok(())
    }

    /// Invokes a model with retry and fallback (no caching).
    ///
    /// Retries are governed by [`RunPolicy::retry`][crate::runtime::RunPolicy]
    /// and apply only to retryable errors (see
    /// [`is_retryable`][crate::retry::is_retryable]); each scheduled
    /// retry emits [`AgentEvent::RetryScheduled`]. When retries are exhausted
    /// (or the error is non-retryable) and a [`crate::retry::FallbackPolicy`]
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
            // Counts the deltas the *current* streaming attempt has already
            // handed to consumers. A stream that dies after 200 tokens has
            // already delivered them; the retry replays from scratch, so a UI
            // concatenating `ModelDelta.text` renders partial garbage followed
            // by the full answer. `StreamAccumulator` is discarded internally,
            // but consumers are never told to discard too — see the warning
            // below and the `AgentEvent` handoff noted in the module docs.
            let mut deltas_emitted = 0usize;
            let outcome = loop {
                // Observe cancellation before (re)issuing a model attempt so a
                // cancel requested during a retry/rate-limit wait stops the run
                // promptly instead of firing another provider call or falling
                // through to the fallback chain.
                if ctx.cancellation.is_cancelled() {
                    return Err(TinyAgentsError::Cancelled);
                }
                // Bound this individual provider call by the tighter of the
                // run's *remaining* wall-clock budget and the per-model-call
                // ceiling, so a hung or slow model call is interrupted
                // mid-flight, not merely detected by the between-call deadline
                // check. reqwest/futures are cancel-safe, so dropping the future
                // on elapse cancels the underlying request. Recomputed here,
                // inside the attempt loop, so every retry attempt gets a fresh
                // per-call window. When neither the run config nor the harness
                // policy configures any timeout the call is awaited unbounded.
                let (remaining, bound) = self.model_call_budget(ctx);
                let attempt_result = if streaming {
                    let fut = self.invoke_model_streaming_once(
                        state,
                        ctx,
                        &model,
                        request,
                        call_id,
                        &mut deltas_emitted,
                    );
                    Self::with_call_budget(remaining, run_id.as_str(), "model call", bound, fut)
                        .await
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
                    let fut = async {
                        model
                            .invoke(state, request.clone())
                            .await
                            .map_err(TinyAgentsError::from)
                    };
                    let budgeted = Self::with_call_budget(
                        remaining,
                        run_id.as_str(),
                        "model call",
                        bound,
                        fut,
                    );
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
                            if streaming && deltas_emitted > 0 {
                                // The retry re-emits the whole response from
                                // the beginning. Until `AgentEvent` grows a
                                // dedicated discard marker, `RetryScheduled`
                                // for a streaming call *is* the signal that
                                // every delta seen so far for this `call_id`
                                // must be dropped.
                                tinyagents_tracing::warn!(
                                    call_id = %call_id.as_str(),
                                    discarded_deltas = deltas_emitted,
                                    attempt,
                                    "[stream] retrying a streaming call that already emitted \
                                     deltas; consumers must discard everything received so far \
                                     for this call_id"
                                );
                            }
                            deltas_emitted = 0;
                            ctx.emit(AgentEvent::RetryScheduled {
                                call_id: call_id.clone(),
                                attempt,
                            });
                            // Sleep for the backoff only when the policy opts in
                            // (`with_backoff_sleep`); otherwise this is a no-op so
                            // the loop stays fast and deterministic in tests.
                            self.policy
                                .retry
                                .sleep_backoff_for_error(backoff_attempt, &error)
                                .await;
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
    ///   [`crate::limits::LimitTracker`], and
    /// - the harness policy's
    ///   [`RunLimits::max_wall_clock_ms`][crate::limits::RunLimits::max_wall_clock_ms],
    ///   measured against the same tracker start.
    ///
    /// Either source may be absent; when both are absent the call is unbounded
    /// (`None`). Honoring the policy source lets a sub-agent whose child
    /// [`RunConfig`] carries no per-run timeout still be bounded by its
    /// harness's policy-level wall-clock cap.
    ///
    /// This is the budget for **tool calls**, which are bounded only by the
    /// run's remaining time (a sub-agent delegation is a tool call wrapping an
    /// entire child run). Model calls go through
    /// [`model_call_budget`](Self::model_call_budget), which additionally
    /// clamps to the per-call ceiling
    /// [`RunLimits::max_model_call_ms`][crate::limits::RunLimits::max_model_call_ms].
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

    /// Computes the wall-clock budget for the next individual **model** call:
    /// the tighter of the run's remaining budget ([`call_budget`](Self::call_budget))
    /// and the per-call ceiling
    /// [`RunLimits::max_model_call_ms`][crate::limits::RunLimits::max_model_call_ms].
    ///
    /// Computed afresh for every call *and every retry attempt*, so each
    /// attempt gets its own full per-call window rather than inheriting an
    /// earlier attempt's consumption — while still never overshooting the run
    /// deadline. Returns the budget together with the label of whichever bound
    /// is in force, so the timeout error names the ceiling that actually fired
    /// (a per-call ceiling means "this one call wedged"; the run's remainder
    /// means "the run is out of time") — field triage needs to tell them apart.
    pub(super) fn model_call_budget(
        &self,
        ctx: &RunContext<Ctx>,
    ) -> (Option<Duration>, &'static str) {
        let run_budget = self.call_budget(ctx);
        let per_call = self
            .policy
            .limits
            .max_model_call_ms
            .map(Duration::from_millis);
        match (run_budget, per_call) {
            (Some(run), Some(cap)) if cap < run => (Some(cap), PER_CALL_BOUND_LABEL),
            (None, Some(cap)) => (Some(cap), PER_CALL_BOUND_LABEL),
            (run, _) => (run, RUN_BOUND_LABEL),
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
    /// `budget` is either the run's *remaining* wall-clock budget at the time
    /// the call is issued (each successive call gets a tighter bound as the
    /// deadline approaches) or, for model calls, the per-call ceiling when that
    /// is tighter. `what` names the kind of call in the timeout message (e.g.
    /// `"model call"`, `"tool call"`); `bound` names which budget source is in
    /// force ([`RUN_BOUND_LABEL`] / [`PER_CALL_BOUND_LABEL`]) so the message
    /// says which ceiling fired.
    pub(super) async fn with_call_budget<T, F>(
        budget: Option<Duration>,
        run_id: &str,
        what: &str,
        bound: &str,
        fut: F,
    ) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match budget {
            Some(budget) => match tokio::time::timeout(budget, fut).await {
                Ok(result) => result,
                Err(_) => Err(TinyAgentsError::Timeout(format!(
                    "{what} for run `{run_id}` exceeded its {bound} ({} ms)",
                    budget.as_millis()
                ))),
            },
            None => fut.await,
        }
    }

    /// Drives one streaming model call to completion.
    ///
    /// Consumes [`tinyinference::model::ChatModel::stream`], emitting an
    /// [`AgentEvent::ModelDelta`] and running every middleware's
    /// [`on_model_delta`][crate::middleware::Middleware::on_model_delta]
    /// hook for each [`ModelStreamItem::MessageDelta`] (and standalone
    /// [`ModelStreamItem::ToolCallDelta`]), then folds the items into the final
    /// [`ModelResponse`] via [`StreamAccumulator`]. The merged response is
    /// equivalent to what the unary [`tinyinference::model::ChatModel::invoke`]
    /// path would have produced, so the rest of the loop is unaffected.
    ///
    /// `deltas_emitted` is incremented for every delta actually handed to
    /// consumers, so the retry path can tell whether a failed attempt already
    /// published output that now has to be discarded.
    async fn invoke_model_streaming_once(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        model: &Arc<dyn ChatModel<State>>,
        request: &ModelRequest,
        call_id: &CallId,
        deltas_emitted: &mut usize,
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
                *deltas_emitted += 1;
                self.middleware
                    .run_on_model_delta(ctx, state, &mut model_delta)
                    .await?;
            }

            accumulator.push(&item);
        }

        Ok(accumulator.finish()?)
    }
}
/// The innermost model call wrapped by the model-wrap onion.
///
/// Implements [`ModelBaseCall`] over the harness's cache + retry + fallback core
/// ([`AgentHarness::invoke_model_with_retry`]) so a [`crate::middleware::ModelMiddleware`]
/// can proceed, short-circuit, retry, or fall back around the *whole* real model
/// call. The resolved binding is rebuilt per invocation so a wrap middleware
/// that retries `next` issues a fresh provider call each time.
///
/// # The binding is re-resolved from the request
///
/// [`ModelCallBase::call`] used to rebuild the binding purely from
/// [`Self::resolved`] / [`Self::model`], both captured **before** the wrap onion
/// ran, and ignore [`ModelRequest::model`] entirely. A wrap middleware steers by
/// mutating that field — it is the only lever it has — so
/// [`ModelFallbackMiddleware`][crate::middleware::ModelFallbackMiddleware]
/// re-invoked *the same failing model* once per configured fallback name,
/// emitting a misleading `FallbackSelected { from, to }` for each, and then
/// returned the original error. The asymmetry was easy to miss because
/// `before_model` **does** honour `request.model`: lifecycle resolution happens
/// after that hook, but before this one.
pub(super) struct ModelCallBase<'h, State: Send + Sync, Ctx: Send + Sync> {
    pub(super) harness: &'h AgentHarness<State, Ctx>,
    pub(super) call_id: CallId,
    pub(super) resolved: ResolvedModel,
    pub(super) model: Arc<dyn ChatModel<State>>,
    pub(super) streaming: bool,
}

impl<State: Send + Sync, Ctx: Send + Sync> ModelCallBase<'_, State, Ctx> {
    /// Produces the binding for one invocation, honouring a model override that
    /// a wrap middleware wrote into `request.model`.
    ///
    /// * `request.model` absent, or equal to the already-resolved name: reuse
    ///   the captured binding (the common path — no registry lookup).
    /// * `request.model` names something the registry resolves as a genuine
    ///   [`ModelResolutionSource::RequestOverride`]: use it. This is what makes
    ///   a wrap-layer fallback actually switch models.
    /// * `request.model` names something unresolvable (unregistered, missing a
    ///   required capability, provider-retired): fall back to the captured
    ///   binding and emit [`AgentEvent::ModelOverrideSkipped`], matching the
    ///   fail-closed behaviour `run_loop` already has for a pre-wrap override.
    ///   Silently substituting a different model is the one outcome that is
    ///   never acceptable.
    fn rebind(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
    ) -> ResolvedModelBinding<State> {
        let captured = || ResolvedModelBinding {
            resolved: self.resolved.clone(),
            model: Arc::clone(&self.model),
        };
        let Some(requested) = request.model.as_deref() else {
            return captured();
        };
        if requested == self.resolved.name {
            return captured();
        }
        match self.harness.models.resolve_request(request, None, None) {
            Some(binding)
                if binding.resolved.source == ModelResolutionSource::RequestOverride
                    && binding.resolved.name == requested =>
            {
                tinyagents_tracing::debug!(
                    call_id = %self.call_id.as_str(),
                    from = %self.resolved.name,
                    to = %binding.resolved.name,
                    "[model] wrap layer overrode the model; re-resolved the binding"
                );
                binding
            }
            _ => {
                tinyagents_tracing::warn!(
                    call_id = %self.call_id.as_str(),
                    requested = %requested,
                    resolved = %self.resolved.name,
                    "[model] wrap layer named an unresolvable model; keeping the resolved binding"
                );
                ctx.emit(AgentEvent::ModelOverrideSkipped {
                    requested: requested.to_string(),
                    resolved: self.resolved.name.clone(),
                });
                captured()
            }
        }
    }
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
            let binding = self.rebind(ctx, &request);
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
/// [`crate::middleware::ToolMiddleware`] can wrap the real tool
/// invocation.
pub(super) struct ToolCallBase<State: Send + Sync> {
    pub(super) tool: Arc<dyn Tool<State>>,
    pub(super) timeout_settings: Option<crate::tool::ToolTimeoutSettings>,
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
                crate::tool::ToolExecutionContext::from_run_context(ctx),
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
