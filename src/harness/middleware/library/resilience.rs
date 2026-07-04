//! Resilience middleware: retry, timeout, model fallback, rate limiting.
//!
//! Split out of `library/mod.rs`; see that module's doc comment for the
//! full built-in middleware library overview.

use super::*;

// ── RetryMiddleware ───────────────────────────────────────────────────────────

impl RetryMiddleware {
    /// Creates a retry middleware using the given [`RetryPolicy`].
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            label: "retry",
            policy,
        }
    }

    /// Creates a retry middleware with the default [`RetryPolicy`].
    pub fn with_default_policy() -> Self {
        Self::new(RetryPolicy::default())
    }

    /// Returns the policy-derived backoff for the given retry `attempt`.
    ///
    /// Exposed for callers that want to inspect the backoff. The middleware
    /// itself sleeps between retries only when the policy opts in via
    /// [`RetryPolicy::with_backoff_sleep`]; otherwise it retries back-to-back.
    pub fn backoff_for_attempt(&self, attempt: usize) -> Duration {
        self.policy.backoff_for_attempt(attempt)
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for RetryMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut attempt = 0usize;
        loop {
            match next.run(ctx, state, request.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(error) => {
                    if self.policy.should_retry_error(attempt, &error) {
                        // Compute the backoff from the *pre-increment* attempt
                        // number: `attempt == 0` is the first retry and must
                        // sleep `initial_backoff_ms`
                        // (`RetryPolicy::backoff_for_attempt(0)`). Sleeping on
                        // the post-increment value skipped `initial_backoff_ms`
                        // entirely and shifted the whole exponential schedule
                        // one step too high.
                        let backoff_attempt = attempt;
                        attempt += 1;
                        let call_id = CallId::new(format!("{}-model", ctx.run_id()));
                        ctx.emit(AgentEvent::RetryScheduled { call_id, attempt });
                        // Sleep for the backoff only when the policy opts in
                        // (`with_backoff_sleep`); a no-op otherwise.
                        self.policy.sleep_backoff(backoff_attempt).await;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
    }
}

// ── TimeoutMiddleware ─────────────────────────────────────────────────────────

impl TimeoutMiddleware {
    /// Creates a timeout middleware bounding each model call to `timeout`.
    pub fn new(timeout: Duration) -> Self {
        Self {
            label: "timeout",
            timeout,
        }
    }

    /// Creates a timeout middleware from a millisecond duration.
    pub fn from_millis(ms: u64) -> Self {
        Self::new(Duration::from_millis(ms))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for TimeoutMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        let run_id = ctx.run_id().as_str().to_string();
        let fut = next.run(ctx, state, request);
        match tokio::time::timeout(self.timeout, fut).await {
            Ok(result) => result,
            Err(_) => Err(TinyAgentsError::Timeout(format!(
                "model call for run `{run_id}` exceeded the {} ms middleware timeout",
                self.timeout.as_millis()
            ))),
        }
    }
}

// ── ModelFallbackMiddleware ───────────────────────────────────────────────────

impl ModelFallbackMiddleware {
    /// Creates a fallback middleware that tries each model name in order after
    /// the primary call fails.
    pub fn new(fallbacks: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "model_fallback",
            fallbacks: fallbacks.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for ModelFallbackMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        match next.run(ctx, state, request.clone()).await {
            Ok(outcome) => Ok(outcome),
            Err(mut last_error) => {
                let mut current = request.model.clone().unwrap_or_default();
                for fallback in &self.fallbacks {
                    // Only a *transient* failure justifies trying another model:
                    // a non-retryable error (auth/validation/schema) will fail
                    // the same way on every backend, so switching burns quota
                    // and latency for nothing. Classification is shared with the
                    // rest of the harness via `is_retryable`.
                    if !is_retryable(&last_error) {
                        break;
                    }
                    ctx.emit(AgentEvent::FallbackSelected {
                        from: current.clone(),
                        to: fallback.clone(),
                    });
                    let mut req = request.clone();
                    req.model = Some(fallback.clone());
                    match next.run(ctx, state, req).await {
                        Ok(outcome) => return Ok(outcome),
                        Err(error) => {
                            last_error = error;
                            current = fallback.clone();
                        }
                    }
                }
                Err(last_error)
            }
        }
    }
}

// ── RateLimitMiddleware ───────────────────────────────────────────────────────

impl RateLimitMiddleware {
    /// Creates a rate-limit middleware gating one token per call through
    /// `limiter`, failing immediately when the bucket is empty
    /// ([`RateLimitBehavior::Error`]).
    pub fn new(limiter: Arc<RateLimiter>) -> Self {
        Self {
            label: "rate_limit",
            limiter,
            tokens: 1,
            behavior: RateLimitBehavior::Error,
            poll_interval: Duration::from_millis(50),
            now: Arc::new(Instant::now),
        }
    }

    /// Sets the number of tokens each call consumes.
    pub fn with_tokens(mut self, tokens: u64) -> Self {
        self.tokens = tokens;
        self
    }

    /// Sets the behavior when the bucket lacks capacity.
    pub fn with_behavior(mut self, behavior: RateLimitBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Switches to [`RateLimitBehavior::Wait`] with the given poll interval.
    pub fn waiting(mut self, poll_interval: Duration) -> Self {
        self.behavior = RateLimitBehavior::Wait;
        self.poll_interval = poll_interval;
        self
    }

    /// Replaces the clock used to read the current instant (for deterministic
    /// tests).
    pub fn with_clock(mut self, now: NowFn) -> Self {
        self.now = now;
        self
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for RateLimitMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut wait_start: Option<Instant> = None;
        loop {
            let now = (self.now)();
            if self.limiter.try_acquire(self.tokens, now) {
                // Report the *actual* wall-clock time spent waiting (per the
                // injectable clock), not the intended per-poll interval.
                if let Some(start) = wait_start {
                    ctx.emit(AgentEvent::RateLimitWaited {
                        waited_ms: now.saturating_duration_since(start).as_millis() as u64,
                    });
                }
                break;
            }
            match self.behavior {
                RateLimitBehavior::Error => {
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "rate limit: could not acquire {} token(s)",
                        self.tokens
                    )));
                }
                RateLimitBehavior::Wait => {
                    // A bucket that never refills (refill rate <= 0) or a
                    // request larger than the bucket capacity can never be
                    // satisfied by waiting: fail fast instead of livelocking
                    // in the poll loop forever.
                    if !self.limiter.can_ever_acquire(self.tokens) {
                        return Err(TinyAgentsError::LimitExceeded(format!(
                            "rate limit: waiting for {} token(s) can never succeed \
                             (bucket capacity {}, refill {}/s)",
                            self.tokens,
                            self.limiter.capacity(),
                            self.limiter.refill_per_sec()
                        )));
                    }
                    wait_start.get_or_insert(now);
                    tokio::time::sleep(self.poll_interval).await;
                }
            }
        }
        next.run(ctx, state, request).await
    }
}
