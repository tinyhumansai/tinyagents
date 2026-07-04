//! Tests for the retry/fallback/rate-limit policies: exponential backoff growth
//! and capping, jitter scaling and clamping, `should_retry` boundaries,
//! `is_retryable` error classification, `FallbackPolicy::next_after` traversal,
//! and token-bucket acquisition, time-based refill, and capacity capping.

use std::time::{Duration, Instant};

use super::{FallbackPolicy, RateLimiter, RetryPolicy, is_retryable};
use crate::error::TinyAgentsError;

#[test]
fn smoke_retry_policy_compiles() {
    let policy = RetryPolicy::default();
    assert!(policy.should_retry(0));
    assert!(!policy.should_retry(3));

    assert!(is_retryable(&TinyAgentsError::Model("timeout".into())));
    assert!(!is_retryable(&TinyAgentsError::Validation(
        "bad input".into()
    )));
}

// ── RetryPolicy::backoff_for_attempt ──────────────────────────────────────────

#[test]
fn backoff_grows_exponentially_then_caps() {
    // initial=200, multiplier=2.0, cap=30_000 (defaults).
    let policy = RetryPolicy::default();

    assert_eq!(policy.backoff_for_attempt(0), Duration::from_millis(200));
    assert_eq!(policy.backoff_for_attempt(1), Duration::from_millis(400));
    assert_eq!(policy.backoff_for_attempt(2), Duration::from_millis(800));
    assert_eq!(policy.backoff_for_attempt(3), Duration::from_millis(1_600));

    // Monotonic non-decreasing up to the cap.
    let mut prev = Duration::ZERO;
    for attempt in 0..20 {
        let cur = policy.backoff_for_attempt(attempt);
        assert!(cur >= prev, "backoff must be monotonic non-decreasing");
        assert!(
            cur <= Duration::from_millis(30_000),
            "must never exceed cap"
        );
        prev = cur;
    }

    // Large attempt is capped exactly at the maximum.
    assert_eq!(
        policy.backoff_for_attempt(50),
        Duration::from_millis(30_000)
    );
}

#[test]
fn backoff_jitter_scales_by_rand01() {
    let policy = RetryPolicy::default().with_jitter(true);

    // attempt 2 base = 800ms. With jitter the result is base * rand01.
    assert_eq!(
        policy.backoff_for_attempt_with(2, 0.0),
        Duration::from_millis(0)
    );
    assert_eq!(
        policy.backoff_for_attempt_with(2, 0.5),
        Duration::from_millis(400)
    );
    // rand01 is clamped into [0, 1).
    assert_eq!(
        policy.backoff_for_attempt_with(2, 5.0),
        Duration::from_millis(800)
    );
    assert_eq!(
        policy.backoff_for_attempt_with(2, -3.0),
        Duration::from_millis(0)
    );
}

#[test]
fn backoff_without_jitter_ignores_rand01() {
    let policy = RetryPolicy::default(); // jitter = false
    assert_eq!(
        policy.backoff_for_attempt_with(1, 0.99),
        Duration::from_millis(400)
    );
}

// ── RetryPolicy::should_retry ─────────────────────────────────────────────────

#[test]
fn should_retry_boundary_at_max_attempts() {
    let policy = RetryPolicy::default().with_max_attempts(3);
    assert!(policy.should_retry(0));
    assert!(policy.should_retry(1));
    assert!(!policy.should_retry(2)); // 2 + 1 == max_attempts → stop
    assert!(!policy.should_retry(3));

    // max_attempts == 1 disables retries entirely.
    let no_retry = RetryPolicy::default().with_max_attempts(1);
    assert!(!no_retry.should_retry(0));
}

// ── RetryPolicy::max_attempts_capped_at ───────────────────────────────────────

#[test]
fn max_attempts_capped_at_takes_the_stricter_of_the_two_caps() {
    // A looser `RunLimits::max_retries_per_call` never widens the policy's
    // own cap.
    let policy = RetryPolicy::default().with_max_attempts(3);
    assert_eq!(policy.max_attempts_capped_at(10), 3);

    // A stricter `max_retries_per_call` (a *retry* count, so +1 for the first
    // attempt) overrides a looser policy.
    assert_eq!(policy.max_attempts_capped_at(1), 2);

    // Zero retries permitted means exactly one attempt, same as
    // `max_attempts == 1`.
    assert_eq!(policy.max_attempts_capped_at(0), 1);
}

// ── is_retryable per error class ──────────────────────────────────────────────

#[test]
fn is_retryable_classification() {
    assert!(is_retryable(&TinyAgentsError::Model("5xx".into())));
    assert!(is_retryable(&TinyAgentsError::Tool("transient".into())));

    assert!(!is_retryable(&TinyAgentsError::Validation("bad".into())));
    assert!(!is_retryable(&TinyAgentsError::RecursionLimit(10)));

    let serde_err = serde_json::from_str::<i32>("not-json").unwrap_err();
    assert!(!is_retryable(&TinyAgentsError::Serialization(serde_err)));
}

// ── TinyAgentsError::Provider classification ──────────────────────────────────

#[test]
fn provider_error_retryability_is_read_from_the_structured_flag_not_assumed() {
    use crate::harness::model::ProviderError;

    // Regression test: the unary/streaming provider path used to flatten a
    // structured `ProviderError` into a plain `Model(String)`, so retry could
    // not distinguish a retryable 429 from a non-retryable 401 and retried
    // both. `TinyAgentsError::Provider` preserves the `retryable` flag a real
    // provider adapter computes from the HTTP status, and `is_retryable` must
    // consult it instead of assuming every provider failure is transient.
    let rate_limited = ProviderError {
        provider: "openai".to_string(),
        status: Some(429),
        retryable: true,
        message: "rate limited".to_string(),
        ..ProviderError::default()
    };
    assert!(is_retryable(&TinyAgentsError::Provider(Box::new(
        rate_limited
    ))));

    let unauthorized = ProviderError {
        provider: "openai".to_string(),
        status: Some(401),
        retryable: false,
        message: "invalid api key".to_string(),
        ..ProviderError::default()
    };
    assert!(!is_retryable(&TinyAgentsError::Provider(Box::new(
        unauthorized
    ))));
}

// ── FallbackPolicy::next_after ────────────────────────────────────────────────

#[test]
fn fallback_next_after_semantics() {
    let policy = FallbackPolicy::new(["a", "b", "c"]);

    // Middle entry returns the following one.
    assert_eq!(policy.next_after("a"), Some("b"));
    assert_eq!(policy.next_after("b"), Some("c"));

    // Last entry has no successor.
    assert_eq!(policy.next_after("c"), None);

    // Unknown entry returns None.
    assert_eq!(policy.next_after("missing"), None);

    // Empty policy returns None for anything.
    let empty = FallbackPolicy::default();
    assert_eq!(empty.next_after("a"), None);
}

// ── RateLimiter ───────────────────────────────────────────────────────────────

#[test]
fn rate_limiter_acquire_until_empty() {
    let limiter = RateLimiter::new(3, 1.0);
    let now = Instant::now();

    assert_eq!(limiter.available(now), 3);
    assert!(limiter.try_acquire(1, now));
    assert!(limiter.try_acquire(2, now));
    assert_eq!(limiter.available(now), 0);

    // Bucket is empty; further acquisition fails at the same instant.
    assert!(!limiter.try_acquire(1, now));
}

#[test]
fn rate_limiter_refills_over_time() {
    let limiter = RateLimiter::new(10, 5.0); // 5 tokens/sec
    let start = Instant::now();

    // Drain the bucket.
    assert!(limiter.try_acquire(10, start));
    assert_eq!(limiter.available(start), 0);

    // After 1 second, 5 tokens have refilled.
    let after_1s = start + Duration::from_secs(1);
    assert_eq!(limiter.available(after_1s), 5);

    // Partial refill: 0.5s → 2 whole tokens (2.5 floored).
    let limiter2 = RateLimiter::new(10, 5.0);
    let s2 = Instant::now();
    assert!(limiter2.try_acquire(10, s2));
    let after_half = s2 + Duration::from_millis(500);
    assert_eq!(limiter2.available(after_half), 2);
}

#[test]
fn rate_limiter_refill_caps_at_capacity() {
    let limiter = RateLimiter::new(5, 100.0);
    let start = Instant::now();
    // Bucket starts full; a long elapsed time cannot exceed capacity.
    let later = start + Duration::from_secs(60);
    assert_eq!(limiter.available(later), 5);
}

#[test]
fn backoff_sleep_defaults_off_and_is_opt_in() {
    // Default policy does not sleep, keeping retry loops deterministic in tests.
    assert!(!RetryPolicy::default().backoff_sleep);
    // The builder flips it on for production callers.
    assert!(
        RetryPolicy::default()
            .with_backoff_sleep(true)
            .backoff_sleep
    );
}

#[tokio::test(start_paused = true)]
async fn sleep_backoff_waits_only_when_enabled() {
    use tokio::time::Instant as TokioInstant;

    // Disabled (default): returns immediately with no virtual time elapsed.
    let policy = RetryPolicy::default();
    let t0 = TokioInstant::now();
    policy.sleep_backoff(1).await;
    assert_eq!(t0.elapsed(), Duration::ZERO);

    // Enabled: advances virtual time by the computed backoff (attempt 1 =
    // initial_backoff_ms with the default multiplier applied at attempt^power).
    let sleeping = RetryPolicy::default().with_backoff_sleep(true);
    let expected = sleeping.backoff_for_attempt(1);
    let t1 = TokioInstant::now();
    sleeping.sleep_backoff(1).await;
    assert_eq!(t1.elapsed(), expected);
    assert!(expected > Duration::ZERO);
}

#[test]
fn should_retry_error_combines_classification_and_attempt_cap() {
    // 1 try + 2 retries: attempts 0 and 1 may retry, attempt 2 may not.
    let policy = RetryPolicy::default().with_max_attempts(3);

    // Retryable error, attempts left → retry.
    let retryable = TinyAgentsError::Model("5xx".into());
    assert!(policy.should_retry_error(0, &retryable));
    assert!(policy.should_retry_error(1, &retryable));
    // Retryable error, attempts exhausted → stop.
    assert!(!policy.should_retry_error(2, &retryable));

    // Non-retryable error is never retried regardless of remaining attempts.
    let non_retryable = TinyAgentsError::Validation("bad".into());
    assert!(!policy.should_retry_error(0, &non_retryable));
}
