//! Types for retry, fallback, and rate-limiting policies.

/// Configures how a harness call is retried on transient failure.
///
/// Backoff grows exponentially: `initial_backoff_ms * multiplier^attempt`, then
/// capped at `max_backoff_ms`. When `jitter` is `true` the caller should use
/// [`RetryPolicy::backoff_for_attempt_with`] and supply a `[0, 1)` random
/// value; this avoids thundering-herd without making the implementation
/// non-deterministic for tests.
///
/// # Examples
///
/// ```
/// use rustagents::harness::retry::RetryPolicy;
///
/// let policy = RetryPolicy::default();
/// assert!(policy.should_retry(0));
/// assert!(!policy.should_retry(3));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RetryPolicy {
    /// Total number of attempts (first try + retries). A value of `1` means no
    /// retries.
    pub max_attempts: usize,
    /// Backoff duration before the first retry in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds.
    pub max_backoff_ms: u64,
    /// Multiplicative factor applied to the backoff on each attempt.
    pub multiplier: f64,
    /// When `true`, the caller should supply a `[0, 1)` random value to
    /// [`RetryPolicy::backoff_for_attempt_with`] to add jitter.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4, // 1 try + 3 retries
            initial_backoff_ms: 200,
            max_backoff_ms: 30_000,
            multiplier: 2.0,
            jitter: false,
        }
    }
}

/// Ordered list of model identifiers to try in sequence when the current model
/// fails.
///
/// The harness will move to `next_after` the current model on non-retryable
/// errors (or after retries are exhausted).
///
/// # Examples
///
/// ```
/// use rustagents::harness::retry::FallbackPolicy;
///
/// let policy = FallbackPolicy { models: vec!["claude-3-5-sonnet".into(), "claude-3-haiku".into()] };
/// assert_eq!(policy.next_after("claude-3-5-sonnet"), Some("claude-3-haiku"));
/// assert_eq!(policy.next_after("claude-3-haiku"), None);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FallbackPolicy {
    /// Ordered model identifiers. The harness uses the first entry by default
    /// and falls back to subsequent entries on failure.
    pub models: Vec<String>,
}

/// A simple token-bucket rate limiter.
///
/// State is kept inside the struct behind a [`std::sync::Mutex`] so the same
/// instance can be shared across threads without wrapping in an `Arc<Mutex<...>>`
/// by the caller.
///
/// The caller must supply the current time (`now: Instant`) to every method so
/// the limiter is fully testable without real sleeping or timer injection.
///
/// # Examples
///
/// ```
/// use std::time::Instant;
/// use rustagents::harness::retry::RateLimiter;
///
/// let mut limiter = RateLimiter::new(10, 5.0);
/// let now = Instant::now();
/// assert!(limiter.try_acquire(1, now));
/// ```
pub struct RateLimiter {
    pub(crate) inner: std::sync::Mutex<RateLimiterState>,
}

/// Inner mutable state of a [`RateLimiter`].
pub(crate) struct RateLimiterState {
    /// Current token count (may be fractional due to sub-second refills).
    pub(crate) tokens: f64,
    /// Maximum tokens the bucket can hold.
    pub(crate) capacity: f64,
    /// Tokens added to the bucket per second.
    pub(crate) refill_per_sec: f64,
    /// Last time the bucket was refilled.
    pub(crate) last_refill: std::time::Instant,
}
