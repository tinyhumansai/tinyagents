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
//!
//! # Testability note
//!
//! [`RateLimiter`] and [`RetryPolicy::backoff_for_attempt`] accept an explicit
//! `now: Instant` / `rand01: f64` so tests can drive time and randomness
//! deterministically without injecting a clock trait.

mod types;

pub use types::*;

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::TinyAgentsError;
use crate::harness::model::ProviderError;

// ── Provider failure classification ─────────────────────────────────────────

fn parse_status_at(text: &str, start: usize) -> Option<u16> {
    let digits: String = text
        .get(start..)?
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if digits.len() == 3 {
        digits.parse().ok()
    } else {
        None
    }
}

fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

/// Extracts an HTTP status from structured provider error text.
///
/// Recognized positions are provider envelopes like `API error (404): ...`,
/// explicit `HTTP 404`, `status: 404` / `status 404`, or a status code at the
/// beginning of the message. Free-text digit runs such as latency values or
/// model ids are ignored.
pub fn structured_http_status(message: &str) -> Option<u16> {
    let trimmed = message.trim_start();
    if let Some(status) = parse_status_at(trimmed, 0) {
        return Some(status);
    }

    for (idx, _) in message.match_indices('(') {
        if let Some(status) = parse_status_at(message, idx + 1) {
            return Some(status);
        }
    }

    for marker in ["http ", "status:", "status "] {
        if let Some(idx) = find_case_insensitive(message, marker)
            && let Some(status) = parse_status_at(message, idx + marker.len())
        {
            return Some(status);
        }
    }

    None
}

fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 429) || status >= 500
}

fn is_upstream_unhealthy_status(status: u16) -> bool {
    matches!(status, 408 | 409) || status >= 500
}

fn text_indicates_rate_limit(lower: &str) -> bool {
    lower.contains("429")
        && (lower.contains("too many") || lower.contains("rate") || lower.contains("limit"))
}

fn text_indicates_upstream_unhealthy(lower: &str) -> bool {
    lower.contains("no healthy upstream")
        || lower.contains("upstream unavailable")
        || lower.contains("service unavailable")
        || lower.contains("408 request timeout")
        || lower.contains("409 conflict")
        || lower.contains("500 internal server error")
        || lower.contains("502 bad gateway")
        || lower.contains("503 service unavailable")
        || lower.contains("504 gateway timeout")
}

fn text_indicates_non_retryable(lower: &str) -> bool {
    let auth_failure_hints = [
        "invalid api key",
        "incorrect api key",
        "missing api key",
        "api key not set",
        "authentication failed",
        "auth failed",
        "unauthorized",
        "forbidden",
        "permission denied",
        "access denied",
        "invalid token",
    ];
    if auth_failure_hints.iter().any(|hint| lower.contains(hint)) {
        return true;
    }

    lower.contains("model")
        && (lower.contains("not found")
            || lower.contains("unknown")
            || lower.contains("unsupported")
            || lower.contains("does not exist")
            || lower.contains("invalid"))
}

fn text_indicates_non_retryable_rate_limit(lower: &str) -> bool {
    let business_hints = [
        "plan does not include",
        "doesn't include",
        "not include",
        "insufficient balance",
        "insufficient_balance",
        "insufficient quota",
        "insufficient_quota",
        "quota exhausted",
        "out of credits",
        "no available package",
        "package not active",
        "purchase package",
        "model not available for your plan",
    ];
    if business_hints.iter().any(|hint| lower.contains(hint)) {
        return true;
    }

    lower.split(|ch: char| !ch.is_ascii_digit()).any(|token| {
        token
            .parse::<u16>()
            .is_ok_and(|code| matches!(code, 1113 | 1311))
    })
}

/// Classifies a normalized provider failure from generic HTTP status, provider
/// error code/type, and message details.
///
/// Host applications may layer their own account, billing, or product-specific
/// terminal rules before or after this helper. TinyAgents only classifies
/// provider-neutral retry behavior.
pub fn classify_provider_failure(
    status: Option<u16>,
    code: Option<&str>,
    message: &str,
) -> ProviderFailureClass {
    let status = status.or_else(|| structured_http_status(message));
    let lower = match code {
        Some(code) if !code.trim().is_empty() => format!("{message} {code}").to_ascii_lowercase(),
        _ => message.to_ascii_lowercase(),
    };

    if status == Some(429) || text_indicates_rate_limit(&lower) {
        if text_indicates_non_retryable_rate_limit(&lower) {
            return ProviderFailureClass::NonRetryableRateLimit;
        }
        return ProviderFailureClass::RateLimited;
    }

    if let Some(status) = status {
        if is_upstream_unhealthy_status(status) {
            return ProviderFailureClass::UpstreamUnhealthy;
        }
        if (400..500).contains(&status) && !is_retryable_status(status) {
            return ProviderFailureClass::NonRetryable;
        }
    }

    if text_indicates_upstream_unhealthy(&lower) {
        return ProviderFailureClass::UpstreamUnhealthy;
    }
    if text_indicates_non_retryable(&lower) {
        return ProviderFailureClass::NonRetryable;
    }

    ProviderFailureClass::Retryable
}

/// Classifies a normalized [`ProviderError`].
pub fn classify_provider_error(error: &ProviderError) -> ProviderFailureClass {
    classify_provider_failure(error.status, error.code.as_deref(), &error.message)
}

/// Computes the retryability flag for a normalized [`ProviderError`].
pub fn provider_error_is_retryable(error: &ProviderError) -> bool {
    classify_provider_error(error).is_retryable()
}

/// Parses a `Retry-After` / `retry_after` value from provider error text into
/// milliseconds. Integer and fractional seconds are accepted.
pub fn parse_retry_after_ms(message: &str) -> Option<u64> {
    let lower = message.to_ascii_lowercase();
    for prefix in &[
        "retry-after:",
        "retry_after:",
        "retry-after ",
        "retry_after ",
    ] {
        if let Some(pos) = lower.find(prefix) {
            let after = &message[pos + prefix.len()..];
            let number: String = after
                .trim_start()
                .chars()
                .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
                .collect();
            if let Ok(seconds) = number.parse::<f64>()
                && seconds.is_finite()
                && seconds >= 0.0
            {
                return u64::try_from(Duration::from_secs_f64(seconds).as_millis()).ok();
            }
        }
    }
    None
}

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
    /// Off by default so tests stay deterministic and fast. Enable it in
    /// production so a transient failure is retried after a real, growing delay
    /// rather than back-to-back. See [`RetryPolicy::backoff_sleep`].
    pub fn with_backoff_sleep(mut self, sleep: bool) -> Self {
        self.backoff_sleep = sleep;
        self
    }

    /// Sleeps for this policy's backoff before the given retry `attempt`, but
    /// only when [`RetryPolicy::backoff_sleep`] is enabled.
    ///
    /// A single, reusable helper so every retry loop that honors a
    /// [`RetryPolicy`] gets identical, opt-in backoff behavior. A no-op (returns
    /// immediately) when sleeping is disabled or the computed backoff is zero.
    pub async fn sleep_backoff(&self, attempt: usize) {
        if !self.backoff_sleep {
            return;
        }
        let backoff = self.backoff_for_attempt(attempt);
        if backoff > Duration::ZERO {
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
    /// Both the agent loop and [`crate::harness::middleware::library::RetryMiddleware`]
    /// route their per-attempt decision through here so the retry classification
    /// and attempt-cap logic live in exactly one place and cannot drift apart.
    ///
    /// Callers that need to fold in a harness-level ceiling
    /// ([`RetryPolicy::max_attempts_capped_at`]) should apply
    /// [`RetryPolicy::with_max_attempts`] first and call this on the capped
    /// policy.
    pub fn should_retry_error(&self, attempt: usize, error: &TinyAgentsError) -> bool {
        is_retryable(error) && self.should_retry(attempt)
    }

    /// Reconciles this policy's own `max_attempts` with a harness-level
    /// ceiling expressed as a *retry* count (not counting the first attempt)
    /// — [`crate::harness::limits::RunLimits::max_retries_per_call`] — and
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

    /// Computes the deterministic (no-jitter) backoff for the given retry
    /// `attempt`.
    ///
    /// - `attempt = 0` → `initial_backoff_ms`
    /// - `attempt = 1` → `initial_backoff_ms * multiplier`
    /// - …capped at `max_backoff_ms`
    ///
    /// When [`RetryPolicy::jitter`] is `true`, prefer
    /// [`backoff_for_attempt_with`][Self::backoff_for_attempt_with] and supply a
    /// caller-controlled `[0, 1)` random value so the implementation remains
    /// testable.
    pub fn backoff_for_attempt(&self, attempt: usize) -> Duration {
        self.backoff_for_attempt_with(attempt, 0.0)
    }

    /// Computes backoff for `attempt` using the supplied `rand01 ∈ [0, 1)` for
    /// jitter.
    ///
    /// When [`RetryPolicy::jitter`] is `false`, `rand01` is ignored and the
    /// result is fully deterministic. When jitter is enabled, the backoff is
    /// uniformly distributed over `[0, base_backoff]`.
    pub fn backoff_for_attempt_with(&self, attempt: usize, rand01: f64) -> Duration {
        let base = (self.initial_backoff_ms as f64) * self.multiplier.powi(attempt as i32);
        let capped = base.min(self.max_backoff_ms as f64);
        let effective = if self.jitter {
            capped * rand01.clamp(0.0, 1.0)
        } else {
            capped
        };
        Duration::from_millis(effective as u64)
    }
}

// ── is_retryable ─────────────────────────────────────────────────────────────

/// Classifies a [`TinyAgentsError`] as retryable or not.
///
/// ## Heuristic
///
/// | Variant | Retryable | Rationale |
/// |---|---|---|
/// | `Provider` | depends | Classified from [`crate::harness::model::ProviderError::retryable`] — a 429/408/409/5xx is retryable, a 4xx like 401/400 is not. |
/// | `Model` | yes | No structured detail to classify from (transport/parse failure); transient provider 5xx / rate-limit / network glitch is the common case. |
/// | `Tool` | yes | Tool execution may have hit a transient dependency. |
/// | `Validation` | **no** | Caller-side schema or policy error; retrying will not help. |
/// | `Serialization` | **no** | Malformed data; retrying will not help. |
/// | `RecursionLimit` | **no** | Structural loop cap; not transient. |
/// | `MissingStart` / `MissingNode` / `MissingEdgeTarget` / `MissingRoute` | **no** | Graph configuration errors; not transient. |
/// | `ToolNotFound` / `ModelNotFound` | **no** | Registry errors; not transient. |
/// | `StructuredOutput` | **no** | Schema mismatch; retrying the same call will likely fail again. |
pub fn is_retryable(err: &TinyAgentsError) -> bool {
    match err {
        TinyAgentsError::Provider(provider_error) => provider_error.retryable,
        TinyAgentsError::Model(_) | TinyAgentsError::Tool(_) => true,
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
