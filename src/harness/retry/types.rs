//! Types for retry, fallback, and rate-limiting policies.
//!
//! These declarative policy structs ([`RetryPolicy`], [`FallbackPolicy`],
//! [`RateLimiter`]) are the durability knobs the recursive harness applies to
//! the model call that every level of the recursion ultimately reduces to, so
//! the same resilience settings govern parent and nested calls alike. Each is
//! deterministically testable: backoff and the limiter take an explicit
//! `rand01` / `now` rather than reading a global clock or RNG.

use std::sync::Arc;

use crate::error::TinyAgentsError;

/// A caller-supplied predicate deciding whether a [`TinyAgentsError`] should be
/// retried, replacing the crate's built-in
/// [`is_retryable`][crate::harness::retry::is_retryable] classification.
///
/// Ported from LangGraph's `RetryPolicy.retry_on`, which accepts an exception
/// type, a sequence of types, or a callable. Rust has no exception hierarchy to
/// match on, so the callable form is the only one that generalizes — a caller
/// wanting the "sequence of types" form writes a `matches!` over
/// [`TinyAgentsError`] variants inside the closure.
///
/// The predicate answers *classification only* ("is this error transient?"); the
/// attempt cap is still applied on top by
/// [`RetryPolicy::should_retry`][crate::harness::retry::RetryPolicy::should_retry].
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tinyagents::error::TinyAgentsError;
/// use tinyagents::harness::retry::RetryPolicy;
///
/// // Retry transport failures but never a failing tool.
/// let policy = RetryPolicy::default().with_retry_on(Arc::new(|err: &TinyAgentsError| {
///     matches!(err, TinyAgentsError::Model(_))
/// }));
/// assert!(policy.is_retryable_error(&TinyAgentsError::Model("connection reset".into())));
/// assert!(!policy.is_retryable_error(&TinyAgentsError::Tool("boom".into())));
/// ```
pub type RetryPredicate = Arc<dyn Fn(&TinyAgentsError) -> bool + Send + Sync>;

/// Configures how a harness call is retried on transient failure.
///
/// Backoff grows exponentially: `initial_backoff_ms * multiplier^attempt`, then
/// capped at `max_backoff_ms`. When `jitter` is `true` the delay is spread
/// **additively** around that base (`base * (1 ± JITTER_FRACTION)`), so jitter
/// can never collapse the delay to zero — see
/// [`RetryPolicy::backoff_for_attempt_with`].
///
/// # Examples
///
/// ```
/// use tinyagents::harness::retry::RetryPolicy;
///
/// let policy = RetryPolicy::default();
/// assert!(policy.should_retry(0));
/// assert!(!policy.should_retry(3));
/// ```
#[derive(Clone)]
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
    /// When `true`, the computed backoff is spread additively around its base
    /// (`base * (1 ± `[`JITTER_FRACTION`][crate::harness::retry::JITTER_FRACTION]`)`)
    /// so concurrent clients that failed together do not retry together.
    ///
    /// Jitter **widens** the delay band; it never removes the delay. (It used to:
    /// the implementation multiplied the base by `rand01`, and the production
    /// call path passed a hardcoded `0.0`, so enabling jitter disabled backoff
    /// entirely.)
    pub jitter: bool,
    /// When `true`, retry loops that honor this policy actually
    /// [`tokio::time::sleep`] for the computed backoff between attempts.
    ///
    /// **Defaults to `true`.** A transport failure is classified retryable, so a
    /// `false` default means the default four attempts fire back-to-back — which
    /// is exactly the wrong behaviour against a rate-limited hosted provider or
    /// a local runtime (Ollama, LM Studio) that is merely loading a model. Tests
    /// that need instant retries opt *out* explicitly with
    /// [`RetryPolicy::with_backoff_sleep(false)`][RetryPolicy::with_backoff_sleep].
    pub backoff_sleep: bool,
    /// Upper bound, in milliseconds, on a server-supplied `Retry-After` delay
    /// that [`RetryPolicy::backoff_for_error`] will honor.
    ///
    /// A provider (or a proxy in front of one) can send an arbitrarily large
    /// `Retry-After`; without a ceiling a single header could park a run for
    /// hours. Defaults to [`RetryPolicy::DEFAULT_MAX_RETRY_AFTER_MS`].
    pub max_retry_after_ms: u64,
    /// Optional caller-supplied retry classification, replacing
    /// [`is_retryable`][crate::harness::retry::is_retryable].
    ///
    /// `None` (the default) uses the crate's built-in classification, so this
    /// field is purely additive: an existing policy behaves exactly as before.
    pub retry_on: Option<RetryPredicate>,
}

impl RetryPolicy {
    /// Default ceiling applied to a server-supplied `Retry-After` (2 minutes).
    pub const DEFAULT_MAX_RETRY_AFTER_MS: u64 = 120_000;
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4, // 1 try + 3 retries
            initial_backoff_ms: 200,
            max_backoff_ms: 30_000,
            multiplier: 2.0,
            jitter: false,
            backoff_sleep: true,
            max_retry_after_ms: Self::DEFAULT_MAX_RETRY_AFTER_MS,
            retry_on: None,
        }
    }
}

/// Hand-written so [`RetryPolicy::retry_on`] (a boxed closure, which has no
/// `Debug`) does not block the derive. The predicate is rendered as a presence
/// marker.
impl std::fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_attempts", &self.max_attempts)
            .field("initial_backoff_ms", &self.initial_backoff_ms)
            .field("max_backoff_ms", &self.max_backoff_ms)
            .field("multiplier", &self.multiplier)
            .field("jitter", &self.jitter)
            .field("backoff_sleep", &self.backoff_sleep)
            .field("max_retry_after_ms", &self.max_retry_after_ms)
            .field(
                "retry_on",
                &self.retry_on.as_ref().map(|_| "<custom predicate>"),
            )
            .finish()
    }
}

/// Hand-written for the same reason as [`Debug`]. Two predicates compare equal
/// only when they are the *same* `Arc` (or both absent) — closures have no
/// meaningful structural equality.
impl PartialEq for RetryPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.max_attempts == other.max_attempts
            && self.initial_backoff_ms == other.initial_backoff_ms
            && self.max_backoff_ms == other.max_backoff_ms
            && self.multiplier == other.multiplier
            && self.jitter == other.jitter
            && self.backoff_sleep == other.backoff_sleep
            && self.max_retry_after_ms == other.max_retry_after_ms
            && match (&self.retry_on, &other.retry_on) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
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
/// use tinyagents::harness::retry::FallbackPolicy;
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
/// use tinyagents::harness::retry::RateLimiter;
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

/// Provider-failure class used for retry, fallback, and telemetry decisions.
///
/// The class is intentionally provider-neutral: callers can layer product,
/// billing, or account-specific rules on top, while TinyAgents supplies the
/// generic HTTP/status/error-message behavior shared by hosted model adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderFailureClass {
    /// A transient upstream failure such as timeout, rate limit, or 5xx.
    Retryable,
    /// A permanent caller/account/model error such as 400/401/403/404.
    NonRetryable,
    /// A generic 429 rate limit where retrying after backoff may succeed.
    RateLimited,
    /// A 429 carrying quota, balance, plan, or package exhaustion detail.
    NonRetryableRateLimit,
    /// A provider-side outage/capacity failure such as 408/409/5xx.
    UpstreamUnhealthy,
}

impl ProviderFailureClass {
    /// Whether retrying the same request may succeed.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::Retryable | Self::RateLimited | Self::UpstreamUnhealthy
        )
    }

    /// Stable reason label suitable for logs and telemetry dimensions.
    pub fn reason(self) -> &'static str {
        match self {
            Self::Retryable => "retryable",
            Self::NonRetryable => "non_retryable",
            Self::RateLimited => "rate_limited",
            Self::NonRetryableRateLimit => "rate_limited_non_retryable",
            Self::UpstreamUnhealthy => "upstream_unhealthy",
        }
    }
}
