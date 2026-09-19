//! Retry, fallback, and rate-limiting policy implementations.
//!
//! These policies make the recursive harness durable under transient failure:
//! because every level of the recursion bottoms out in the same model call,
//! retry/backoff, model fallback, and rate limiting apply uniformly to a
//! top-level agent and to any nested sub-agent or graph-node call, so a flaky
//! provider does not collapse a deep recursion.
//!
//! Three independent policies live here:
//!
//! - [`RetryPolicy`] — exponential backoff with optional jitter and a per-call
//!   attempt cap.
//! - [`FallbackPolicy`] — ordered list of model names to try in sequence.
//! - [`RateLimiter`] — token-bucket limiter for pacing provider calls.
//!
//! A free function [`is_retryable`] classifies a [`TinyAgentsError`] so callers
//! can decide whether to retry or propagate immediately.
//! Provider error taxonomy and `Retry-After` parsing are consumed directly
//! from [`tinyinference_llm::failure`]; this module owns only harness retry policy.
//!
//! # Testability note
//!
//! [`RateLimiter`] and [`RetryPolicy::backoff_for_attempt`] accept an explicit
//! `now: Instant` / `rand01: f64` so tests can drive time and randomness
//! deterministically without injecting a clock trait.

mod jitter;
mod types;

pub use types::*;

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::TinyAgentsError;
use tinyinference_llm::failure::{classify_provider_failure, parse_retry_after_ms};

/// Fraction of the base backoff the additive jitter band spans in each
/// direction: with jitter enabled the effective delay lands uniformly in
/// `[base * (1 - JITTER_FRACTION), base * (1 + JITTER_FRACTION)]`.
///
/// Matches LangChain's `_retry.py` (`delay ± 25%`). LangGraph instead adds a
/// flat `uniform(0, 1)` second; the proportional form generalizes better across
/// the sub-second and multi-second ends of the same schedule. What both
/// references share, and what matters here, is that jitter is **additive** — it
/// never scales the delay toward zero.
pub const JITTER_FRACTION: f64 = 0.25;

// ── RetryPolicy ──────────────────────────────────────────────────────────────

impl RetryPolicy {
    /// Sets the total number of attempts (first try + retries).
    ///
    /// A value of `1` disables retries entirely.
    pub fn with_max_attempts(mut self, n: usize) -> Self {
        self.max_attempts = n;
        self
    }

    /// Sets the initial backoff in milliseconds.
    pub fn with_initial_backoff_ms(mut self, ms: u64) -> Self {
        self.initial_backoff_ms = ms;
        self
    }

    /// Sets the maximum backoff cap in milliseconds.
    pub fn with_max_backoff_ms(mut self, ms: u64) -> Self {
        self.max_backoff_ms = ms;
        self
    }

    /// Sets the exponential backoff multiplier.
    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.multiplier = m;
        self
    }

    /// Enables or disables jitter.
    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Enables or disables actually sleeping for the computed backoff between
    /// retries.
    ///
    /// **On by default.** Pass `false` to opt out — which tests that assert on
    /// retry counts without wanting real elapsed time should do explicitly. See
    /// [`RetryPolicy::backoff_sleep`] for why the default is on.
    pub fn with_backoff_sleep(mut self, sleep: bool) -> Self {
        self.backoff_sleep = sleep;
        self
    }

    /// Sets the ceiling applied to a server-supplied `Retry-After` delay. See
    /// [`RetryPolicy::max_retry_after_ms`].
    pub fn with_max_retry_after_ms(mut self, ms: u64) -> Self {
        self.max_retry_after_ms = ms;
        self
    }

    /// Replaces the built-in [`is_retryable`] classification with `predicate`.
    ///
    /// The predicate decides *only* whether an error is transient; the attempt
    /// cap still applies on top. Ported from LangGraph's
    /// `RetryPolicy.retry_on`.
    pub fn with_retry_on(mut self, predicate: RetryPredicate) -> Self {
        self.retry_on = Some(predicate);
        self
    }

    /// Clears any custom [`RetryPolicy::retry_on`] predicate, restoring the
    /// built-in [`is_retryable`] classification.
    pub fn with_default_retry_on(mut self) -> Self {
        self.retry_on = None;
        self
    }

    /// Classifies `error` using this policy's [`RetryPolicy::retry_on`]
    /// predicate when one is set, and the crate-wide [`is_retryable`] otherwise.
    ///
    /// This is the single classification entry point every retry loop should
    /// use; calling the free [`is_retryable`] directly silently ignores a
    /// caller's custom predicate.
    pub fn is_retryable_error(&self, error: &TinyAgentsError) -> bool {
        match &self.retry_on {
            Some(predicate) => predicate(error),
            None => is_retryable(error),
        }
    }

    /// Sleeps for this policy's backoff before the given retry `attempt`, but
    /// only when [`RetryPolicy::backoff_sleep`] is enabled.
    ///
    /// A single, reusable helper so every retry loop that honors a
    /// [`RetryPolicy`] gets identical backoff behavior. A no-op (returns
    /// immediately) when sleeping is disabled or the computed backoff is zero.
    ///
    /// Prefer [`RetryPolicy::sleep_backoff_for_error`] where the failure is in
    /// hand: it additionally honors a server-supplied `Retry-After`.
    pub async fn sleep_backoff(&self, attempt: usize) {
        self.sleep_for(attempt, self.backoff_for_attempt(attempt), None)
            .await;
    }

    /// Sleeps for `max(computed backoff, server-supplied Retry-After)` before
    /// retry `attempt`, but only when [`RetryPolicy::backoff_sleep`] is enabled.
    ///
    /// A `429` carrying `Retry-After: 30` means the provider will keep refusing
    /// for 30 seconds; retrying after the policy's 200 ms merely burns the
    /// remaining attempts. Taking the **max** (rather than replacing the
    /// backoff) means a server hint can only ever lengthen the wait, so a bogus
    /// `Retry-After: 0` cannot defeat backoff. The hint is clamped at
    /// [`RetryPolicy::max_retry_after_ms`].
    pub async fn sleep_backoff_for_error(&self, attempt: usize, error: &TinyAgentsError) {
        let hint = self.clamped_retry_after(error);
        self.sleep_for(attempt, self.backoff_for_error(attempt, error), hint)
            .await;
    }

    /// The delay this policy would wait before retry `attempt` given `error`:
    /// the larger of the computed backoff and the clamped server-supplied
    /// `Retry-After`. Computed regardless of [`RetryPolicy::backoff_sleep`], so
    /// observability can report the intended delay even when sleeping is off.
    pub fn backoff_for_error(&self, attempt: usize, error: &TinyAgentsError) -> Duration {
        let computed = self.backoff_for_attempt(attempt);
        match self.clamped_retry_after(error) {
            Some(hint) => computed.max(hint),
            None => computed,
        }
    }

    /// Extracts and clamps the server-supplied `Retry-After` carried by `error`.
    fn clamped_retry_after(&self, error: &TinyAgentsError) -> Option<Duration> {
        retry_after_hint(error).map(|hint| hint.min(Duration::from_millis(self.max_retry_after_ms)))
    }

    /// Shared sleep body: logs the decision, then waits when enabled.
    async fn sleep_for(&self, attempt: usize, backoff: Duration, hint: Option<Duration>) {
        if !self.backoff_sleep {
            tracing::debug!(
                target: "tinyagents::retry",
                attempt,
                backoff_ms = backoff.as_millis() as u64,
                "[retry] backoff sleep disabled; retrying immediately"
            );
            return;
        }
        if backoff > Duration::ZERO {
            tracing::debug!(
                target: "tinyagents::retry",
                attempt,
                backoff_ms = backoff.as_millis() as u64,
                retry_after_ms = hint.map(|h| h.as_millis() as u64),
                jitter = self.jitter,
                "[retry] sleeping before retry"
            );
            tokio::time::sleep(backoff).await;
        }
    }

    /// Returns `true` when another attempt should be made.
    ///
    /// `attempt` is zero-indexed: `0` is the first attempt, `1` is the first
    /// retry, and so on. The policy permits another retry when
    /// `attempt + 1 < max_attempts`.
    pub fn should_retry(&self, attempt: usize) -> bool {
        attempt + 1 < self.max_attempts
    }

    /// The single retry decision shared by every retry loop in the harness.
    ///
    /// Returns `true` only when `error` is transient ([`is_retryable`]) *and*
    /// the policy still permits another attempt ([`RetryPolicy::should_retry`]).
    /// Both the agent loop and [`crate::middleware::library::RetryMiddleware`]
    /// route their per-attempt decision through here so the retry classification
    /// and attempt-cap logic live in exactly one place and cannot drift apart.
    ///
    /// Callers that need to fold in a harness-level ceiling
    /// ([`RetryPolicy::max_attempts_capped_at`]) should apply
    /// [`RetryPolicy::with_max_attempts`] first and call this on the capped
    /// policy.
    pub fn should_retry_error(&self, attempt: usize, error: &TinyAgentsError) -> bool {
        self.is_retryable_error(error) && self.should_retry(attempt)
    }

    /// Reconciles this policy's own `max_attempts` with a harness-level
    /// ceiling expressed as a *retry* count (not counting the first attempt)
    /// — [`crate::limits::RunLimits::max_retries_per_call`] — and
    /// returns whichever total-attempt cap is stricter.
    ///
    /// Without this, a `RunPolicy` could configure a looser `RetryPolicy`
    /// than its own `RunLimits`, silently making the "hard" limit
    /// unenforceable; the harness's agent loop always calls this instead of
    /// consulting `max_attempts` directly.
    pub fn max_attempts_capped_at(&self, max_retries_per_call: usize) -> usize {
        self.max_attempts
            .min(max_retries_per_call.saturating_add(1))
    }

    /// Computes the backoff for the given retry `attempt`.
    ///
    /// - `attempt = 0` → `initial_backoff_ms`
    /// - `attempt = 1` → `initial_backoff_ms * multiplier`
    /// - …capped at `max_backoff_ms`
    ///
    /// When [`RetryPolicy::jitter`] is `false` this is fully deterministic. When
    /// jitter is enabled it draws a **real** random value and spreads the result
    /// additively around the base (see
    /// [`backoff_for_attempt_with`][Self::backoff_for_attempt_with]); tests that
    /// need a fixed spread should call that method with an explicit `rand01`
    /// rather than this one.
    pub fn backoff_for_attempt(&self, attempt: usize) -> Duration {
        // 0.5 is the band midpoint, so the no-jitter and jitter paths agree
        // exactly when jitter is off — and the RNG is not touched at all then.
        let rand01 = if self.jitter { jitter::rand01() } else { 0.5 };
        self.backoff_for_attempt_with(attempt, rand01)
    }

    /// Computes backoff for `attempt` using the supplied `rand01 ∈ [0, 1)` for
    /// jitter. This is the deterministic seam: tests inject a fixed value here,
    /// production goes through [`backoff_for_attempt`][Self::backoff_for_attempt].
    ///
    /// When [`RetryPolicy::jitter`] is `false`, `rand01` is ignored and the
    /// result is the exact exponential schedule.
    ///
    /// When jitter is enabled the delay is spread **additively** around the base:
    /// `base * (1 + JITTER_FRACTION * (2 * rand01 - 1))`, i.e. uniformly over
    /// `[base * 0.75, base * 1.25]` at the default
    /// [`JITTER_FRACTION`], then clamped to `max_backoff_ms`. `rand01 == 0.5`
    /// reproduces the un-jittered value exactly.
    ///
    /// This is deliberately **not** the old `base * rand01` form. That one
    /// scaled the delay *down* toward zero, and because the production path
    /// passed a hardcoded `rand01 = 0.0`, turning jitter on produced a zero
    /// delay and disabled backoff entirely. Both reference implementations are
    /// additive: LangGraph adds `uniform(0, 1)` seconds, LangChain applies
    /// `delay ± 25%` clamped at zero.
    pub fn backoff_for_attempt_with(&self, attempt: usize, rand01: f64) -> Duration {
        // `powi` wants `i32`; an `attempt` this large would already dwarf any
        // realistic `max_attempts`, so saturate rather than truncate/wrap
        // silently (M-13).
        let exponent = i32::try_from(attempt).unwrap_or(i32::MAX);
        let base = (self.initial_backoff_ms as f64) * self.multiplier.powi(exponent);
        let jittered = if self.jitter {
            // Map [0, 1) onto [-1, 1) then scale by the band width.
            let offset = JITTER_FRACTION * (2.0 * rand01.clamp(0.0, 1.0) - 1.0);
            (base * (1.0 + offset)).max(0.0)
        } else {
            base
        };
        let capped = jittered.min(self.max_backoff_ms as f64);
        Duration::from_millis(saturating_millis(capped))
    }
}

/// Converts a millisecond duration held as `f64` to `u64`, saturating a
/// negative or non-finite value to `0` instead of relying on the cast's
/// implicit (if well-defined since Rust 1.45) saturating behavior — the
/// saturation is now spelled out at the call site rather than implicit in a
/// bare `as` cast (M-13).
fn saturating_millis(value: f64) -> u64 {
    if value.is_finite() && value > 0.0 {
        value as u64
    } else {
        0
    }
}

/// Extracts a server-supplied `Retry-After` delay carried by `error`, if any.
///
/// A `429` or `503` that names how long the client must wait is authoritative:
/// retrying sooner burns an attempt for certain. [`RetryPolicy::backoff_for_error`]
/// folds this into the delay by taking the larger of the two.
///
/// # Two sources, in priority order
///
/// 1. **The structured field.**
///    [`ProviderError::retry_after_ms`][tinyinference_llm::model::ProviderError::retry_after_ms]
///    is populated by the provider adapter directly from the HTTP `Retry-After`
///    response header (both the delta-seconds and HTTP-date forms). This is the
///    contract; it is read first.
/// 2. **The error message text**, parsed with [`parse_retry_after_ms`]. Hosted
///    providers generally echo the header into the error body, so this is a
///    useful fallback for adapters that do not yet populate the field — but it
///    is string matching, not a contract, and a provider that sends the header
///    without echoing it into the body is served only by source 1.
pub fn retry_after_hint(error: &TinyAgentsError) -> Option<Duration> {
    let message = match error {
        TinyAgentsError::Provider(provider_error) => {
            if let Some(ms) = provider_error.retry_after_ms {
                return Some(Duration::from_millis(ms));
            }
            provider_error.message.as_str()
        }
        TinyAgentsError::Model(message) | TinyAgentsError::Tool(message) => message.as_str(),
        _ => return None,
    };
    parse_retry_after_ms(message).map(Duration::from_millis)
}

// ── is_retryable ─────────────────────────────────────────────────────────────

/// Classifies a [`TinyAgentsError`] as retryable or not.
///
/// ## Heuristic
///
/// | Variant | Retryable | Rationale |
/// |---|---|---|
/// | `Provider` | depends | Classified from [`tinyinference_llm::model::ProviderError::retryable`] — a 429/408/409/5xx is retryable, a 4xx like 401/400 is not. |
/// | `Model` | depends | No structured `ProviderError` to read, so the message text is run through [`classify_provider_failure`] — a 5xx / 429 / timeout is retryable, an `invalid api key` or `model not found` is not. |
/// | `Tool` | yes | Tool execution may have hit a transient dependency. |
/// | `CallTimeout` | **yes** | A per-call ceiling fired with run time still left; unlike `Timeout`, the run is not out of budget. |
/// | `Validation` | **no** | Caller-side schema or policy error; retrying will not help. |
/// | `Serialization` | **no** | Malformed data; retrying will not help. |
/// | `RecursionLimit` | **no** | Structural loop cap; not transient. |
/// | `MissingStart` / `MissingNode` / `MissingEdgeTarget` / `MissingRoute` | **no** | Graph configuration errors; not transient. |
/// | `ToolNotFound` / `ModelNotFound` | **no** | Registry errors; not transient. |
/// | `StructuredOutput` | **no** | Schema mismatch; retrying the same call will likely fail again. |
///
/// Callers holding a [`RetryPolicy`] should call
/// [`RetryPolicy::is_retryable_error`] instead, so a caller-supplied
/// [`RetryPolicy::retry_on`] predicate is honored.
pub fn is_retryable(err: &TinyAgentsError) -> bool {
    match err {
        TinyAgentsError::Provider(provider_error) => provider_error.retryable,
        // A bare `Model(String)` used to be retried unconditionally, so a
        // permanent `401 invalid api key` burned every attempt with a
        // guaranteed-identical failure. There is no structured
        // `ProviderError` here, but the message text is the same text
        // `classify_provider_failure` already knows how to read — so use it
        // rather than assuming transience.
        TinyAgentsError::Model(message) => {
            classify_provider_failure(None, None, message).is_retryable()
        }
        // Tool failures stay unconditionally retryable: a tool's error text is
        // arbitrary caller-authored content with no shared vocabulary to
        // classify against, so the HTTP-shaped heuristics above would be
        // guessing. Callers that know better narrow this with
        // [`RetryPolicy::retry_on`].
        TinyAgentsError::Tool(_) => true,
        // A per-model-call ceiling firing means this one call wedged, with
        // run time still left — retryable, unlike a run-deadline `Timeout`
        // (see that variant's own retryability rationale above).
        TinyAgentsError::CallTimeout(_) => true,
        _ => false,
    }
}

// ── FallbackPolicy ───────────────────────────────────────────────────────────

impl FallbackPolicy {
    /// Creates a new policy from an ordered list of model identifiers.
    pub fn new(models: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            models: models.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the next model to try after `current`, or `None` if `current`
    /// is the last entry or is not present in the list.
    ///
    /// Lookup is O(n) but the list is expected to be very short (2–4 entries).
    pub fn next_after<'a>(&'a self, current: &str) -> Option<&'a str> {
        let mut iter = self.models.iter();
        while let Some(m) = iter.next() {
            if m == current {
                return iter.next().map(String::as_str);
            }
        }
        None
    }
}

// ── RateLimiter ──────────────────────────────────────────────────────────────

impl RateLimiter {
    /// Creates a new token-bucket limiter with the given `capacity` (maximum
    /// burst) and `refill_per_sec` (sustained request rate).
    ///
    /// The bucket starts full.
    pub fn new(capacity: u64, refill_per_sec: f64) -> Self {
        let cap = capacity as f64;
        Self {
            inner: Mutex::new(types::RateLimiterState {
                tokens: cap,
                capacity: cap,
                refill_per_sec,
                last_refill: Instant::now(),
            }),
        }
    }

    /// Refills the bucket based on elapsed time then attempts to consume
    /// `tokens`.
    ///
    /// Returns `true` if the tokens were successfully consumed, `false` if the
    /// bucket does not have enough tokens at the provided `now`.
    ///
    /// The caller supplies `now` so that this method is testable without real
    /// time progression.
    pub fn try_acquire(&self, tokens: u64, now: Instant) -> bool {
        let mut state = self.inner.lock().unwrap();
        self.refill(&mut state, now);
        if state.tokens >= tokens as f64 {
            state.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }

    /// Returns the number of whole tokens available in the bucket at `now`.
    pub fn available(&self, now: Instant) -> u64 {
        let mut state = self.inner.lock().unwrap();
        self.refill(&mut state, now);
        state.tokens.floor() as u64
    }

    /// Returns the bucket capacity (maximum burst) in tokens.
    pub fn capacity(&self) -> u64 {
        self.inner.lock().unwrap().capacity as u64
    }

    /// Returns the sustained refill rate in tokens per second.
    pub fn refill_per_sec(&self) -> f64 {
        self.inner.lock().unwrap().refill_per_sec
    }

    /// Returns `true` when waiting could ever satisfy an acquisition of
    /// `tokens`: the request fits the bucket capacity and the bucket actually
    /// refills. With a zero (or negative) refill rate, or a request larger
    /// than the capacity, a failed acquire can never succeed later.
    pub fn can_ever_acquire(&self, tokens: u64) -> bool {
        let state = self.inner.lock().unwrap();
        tokens as f64 <= state.capacity && state.refill_per_sec > 0.0
    }

    /// Adds tokens to the bucket based on time elapsed since the last refill.
    fn refill(&self, state: &mut types::RateLimiterState, now: Instant) {
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            state.tokens = (state.tokens + elapsed * state.refill_per_sec).min(state.capacity);
            state.last_refill = now;
        }
    }
}

#[cfg(test)]
mod test;
