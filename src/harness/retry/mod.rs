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

    /// Returns `true` when another attempt should be made.
    ///
    /// `attempt` is zero-indexed: `0` is the first attempt, `1` is the first
    /// retry, and so on. The policy permits another retry when
    /// `attempt + 1 < max_attempts`.
    pub fn should_retry(&self, attempt: usize) -> bool {
        attempt + 1 < self.max_attempts
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
/// | `Model` | yes | Transient provider 5xx / rate-limit / network glitch. |
/// | `Tool` | yes | Tool execution may have hit a transient dependency. |
/// | `Validation` | **no** | Caller-side schema or policy error; retrying will not help. |
/// | `Serialization` | **no** | Malformed data; retrying will not help. |
/// | `RecursionLimit` | **no** | Structural loop cap; not transient. |
/// | `MissingStart` / `MissingNode` / `MissingEdgeTarget` / `MissingRoute` | **no** | Graph configuration errors; not transient. |
/// | `ToolNotFound` / `ModelNotFound` | **no** | Registry errors; not transient. |
/// | `StructuredOutput` | **no** | Schema mismatch; retrying the same call will likely fail again. |
pub fn is_retryable(err: &TinyAgentsError) -> bool {
    matches!(err, TinyAgentsError::Model(_) | TinyAgentsError::Tool(_))
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
