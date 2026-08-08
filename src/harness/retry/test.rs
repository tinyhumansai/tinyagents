//! Tests for the retry/fallback/rate-limit policies: exponential backoff growth
//! and capping, jitter scaling and clamping, `should_retry` boundaries,
//! `is_retryable` error classification, `FallbackPolicy::next_after` traversal,
//! and token-bucket acquisition, time-based refill, and capacity capping.

use std::time::{Duration, Instant};

use super::{
    FallbackPolicy, ProviderFailureClass, RateLimiter, RetryPolicy, classify_provider_error,
    classify_provider_failure, is_retryable, parse_retry_after_ms, structured_http_status,
};
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
fn backoff_jitter_spreads_additively_around_the_base() {
    let policy = RetryPolicy::default().with_jitter(true);

    // attempt 2 base = 800ms. Jitter spreads it over ±25% → [600, 1000].
    assert_eq!(
        policy.backoff_for_attempt_with(2, 0.0),
        Duration::from_millis(600)
    );
    // The band midpoint reproduces the un-jittered value exactly.
    assert_eq!(
        policy.backoff_for_attempt_with(2, 0.5),
        Duration::from_millis(800)
    );
    // rand01 is clamped into [0, 1].
    assert_eq!(
        policy.backoff_for_attempt_with(2, 5.0),
        Duration::from_millis(1_000)
    );
    assert_eq!(
        policy.backoff_for_attempt_with(2, -3.0),
        Duration::from_millis(600)
    );
}

#[test]
fn jitter_never_collapses_the_backoff_to_zero() {
    // Regression test (LOOP-2): jitter used to be *multiplicative*
    // (`base * rand01`), and the production path — `backoff_for_attempt`, which
    // `sleep_backoff` calls — passed a hardcoded `rand01 = 0.0`. So the
    // production-hardened-looking `.with_backoff_sleep(true).with_jitter(true)`
    // computed a ZERO delay, `sleep_backoff`'s `> Duration::ZERO` guard never
    // fired, and nothing ever slept: a rate-limited provider got hammered
    // back-to-back. Jitter must only ever *widen* the delay band.
    let policy = RetryPolicy::default().with_jitter(true);

    for attempt in 0..8 {
        let plain = RetryPolicy::default().backoff_for_attempt(attempt);
        let floor = plain.mul_f64(1.0 - super::JITTER_FRACTION);
        let ceiling = plain.mul_f64(1.0 + super::JITTER_FRACTION);

        // The deterministic seam across the whole [0, 1) input range.
        for step in 0..=20 {
            let jittered = policy.backoff_for_attempt_with(attempt, f64::from(step) / 20.0);
            assert!(jittered > Duration::ZERO, "jitter collapsed the delay");
            assert!(jittered >= floor && jittered <= ceiling, "outside the band");
        }

        // And the production path, which now draws real randomness.
        for _ in 0..64 {
            let jittered = policy.backoff_for_attempt(attempt);
            assert!(
                jittered > Duration::ZERO,
                "production backoff_for_attempt returned a zero delay with jitter on"
            );
            assert!(jittered >= floor && jittered <= ceiling, "outside the band");
        }
    }
}

#[tokio::test(start_paused = true)]
async fn jittered_sleep_backoff_actually_sleeps() {
    // The end-to-end shape of LOOP-2: the config that reads as
    // production-hardened must wait, not spin.
    use tokio::time::Instant as TokioInstant;

    let policy = RetryPolicy::default()
        .with_backoff_sleep(true)
        .with_jitter(true);

    let t0 = TokioInstant::now();
    policy.sleep_backoff(0).await;
    assert!(
        t0.elapsed() > Duration::ZERO,
        "jitter + backoff_sleep must still sleep"
    );
}

#[test]
fn production_backoff_is_random_when_jitter_is_enabled() {
    // A stuck RNG would reproduce the original defect in a subtler form.
    let policy = RetryPolicy::default().with_jitter(true);
    let first = policy.backoff_for_attempt(4);
    assert!(
        (0..128).any(|_| policy.backoff_for_attempt(4) != first),
        "jittered backoff never varied — the randomness source is not wired up"
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

#[test]
fn structured_http_status_uses_only_anchored_positions() {
    assert_eq!(
        structured_http_status("custom_openai API error (403 Forbidden): nope"),
        Some(403)
    );
    assert_eq!(structured_http_status("HTTP 404 Not Found"), Some(404));
    assert_eq!(structured_http_status("status: 401"), Some(401));
    assert_eq!(structured_http_status("408 Request Timeout"), Some(408));

    assert_eq!(
        structured_http_status("upstream took 450ms to respond, retrying"),
        None
    );
    assert_eq!(
        structured_http_status("gpt-4-0409 returned an empty completion"),
        None
    );
    assert_eq!(
        structured_http_status("received 412 partial bytes before reset"),
        None
    );
}

#[test]
fn provider_failure_classifies_generic_http_statuses() {
    assert_eq!(
        classify_provider_failure(Some(401), None, "invalid api key"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(Some(404), None, "model not found"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "too many requests"),
        ProviderFailureClass::RateLimited
    );
    assert_eq!(
        classify_provider_failure(Some(408), None, "request timeout"),
        ProviderFailureClass::UpstreamUnhealthy
    );
    assert_eq!(
        classify_provider_failure(Some(502), None, "bad gateway"),
        ProviderFailureClass::UpstreamUnhealthy
    );
}

#[test]
fn provider_failure_classifies_message_hints_without_status() {
    assert_eq!(
        classify_provider_failure(None, None, "authentication failed"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(None, None, "model glm-4.7 is unsupported"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(None, None, "no healthy upstream available"),
        ProviderFailureClass::UpstreamUnhealthy
    );
    assert_eq!(
        classify_provider_failure(None, None, "429 Too Many Requests: rate limit exceeded"),
        ProviderFailureClass::RateLimited
    );
}

#[test]
fn provider_failure_classifies_non_retryable_rate_limits() {
    assert_eq!(
        classify_provider_failure(
            Some(429),
            Some("1311"),
            "the current account plan does not include glm-5"
        ),
        ProviderFailureClass::NonRetryableRateLimit
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "insufficient balance"),
        ProviderFailureClass::NonRetryableRateLimit
    );
}

#[test]
fn provider_failure_class_controls_retryability_and_reason_labels() {
    assert!(ProviderFailureClass::RateLimited.is_retryable());
    assert!(ProviderFailureClass::UpstreamUnhealthy.is_retryable());
    assert!(!ProviderFailureClass::NonRetryable.is_retryable());
    assert!(!ProviderFailureClass::NonRetryableRateLimit.is_retryable());

    assert_eq!(ProviderFailureClass::Retryable.reason(), "retryable");
    assert_eq!(ProviderFailureClass::NonRetryable.reason(), "non_retryable");
    assert_eq!(ProviderFailureClass::RateLimited.reason(), "rate_limited");
    assert_eq!(
        ProviderFailureClass::NonRetryableRateLimit.reason(),
        "rate_limited_non_retryable"
    );
    assert_eq!(
        ProviderFailureClass::UpstreamUnhealthy.reason(),
        "upstream_unhealthy"
    );
}

#[test]
fn classify_provider_error_reads_structured_error_fields() {
    use crate::harness::model::ProviderError;

    let provider_error = ProviderError {
        provider: "openai".to_string(),
        model: Some("gpt-4o".to_string()),
        status: Some(429),
        code: Some("insufficient_quota".to_string()),
        message: "insufficient quota".to_string(),
        ..ProviderError::default()
    };

    assert_eq!(
        classify_provider_error(&provider_error),
        ProviderFailureClass::NonRetryableRateLimit
    );
}

#[test]
fn retry_after_parser_accepts_integer_float_and_space_separators() {
    assert_eq!(
        parse_retry_after_ms("429 Too Many Requests, Retry-After: 5"),
        Some(5_000)
    );
    assert_eq!(
        parse_retry_after_ms("Rate limited. retry_after: 2.5 seconds"),
        Some(2_500)
    );
    assert_eq!(parse_retry_after_ms("Retry-After 7"), Some(7_000));
    assert_eq!(parse_retry_after_ms("500 Internal Server Error"), None);
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
fn backoff_sleep_defaults_on_and_is_opt_out() {
    // Regression test (LOCAL-4): the default used to be `false` "so tests stay
    // deterministic and fast", which made test convenience the *production*
    // policy. Transport failures are classified retryable, so the default four
    // attempts fired back-to-back against a local runtime that was merely
    // loading a multi-gigabyte model. LangGraph's reference default
    // (`initial_interval=0.5, backoff_factor=2.0, jitter=True`) is on.
    assert!(
        RetryPolicy::default().backoff_sleep,
        "backoff must sleep by default"
    );
    // Tests and other latency-sensitive callers opt out explicitly.
    assert!(
        !RetryPolicy::default()
            .with_backoff_sleep(false)
            .backoff_sleep
    );
}

#[tokio::test(start_paused = true)]
async fn sleep_backoff_waits_unless_explicitly_disabled() {
    use tokio::time::Instant as TokioInstant;

    // Explicitly disabled: returns immediately with no virtual time elapsed.
    let policy = RetryPolicy::default().with_backoff_sleep(false);
    let t0 = TokioInstant::now();
    policy.sleep_backoff(1).await;
    assert_eq!(t0.elapsed(), Duration::ZERO);

    // Default (enabled): advances virtual time by the computed backoff.
    let sleeping = RetryPolicy::default();
    let expected = sleeping.backoff_for_attempt(1);
    let t1 = TokioInstant::now();
    sleeping.sleep_backoff(1).await;
    assert_eq!(t1.elapsed(), expected);
    assert!(expected > Duration::ZERO);
}

// ── Retry-After (LOOP-5) ──────────────────────────────────────────────────────

#[test]
fn retry_after_hint_is_read_from_every_error_shape_that_can_carry_one() {
    use crate::harness::model::ProviderError;
    use crate::harness::retry::retry_after_hint;

    assert_eq!(
        retry_after_hint(&TinyAgentsError::Model(
            "429 Too Many Requests, Retry-After: 30".into()
        )),
        Some(Duration::from_secs(30))
    );
    assert_eq!(
        retry_after_hint(&TinyAgentsError::Provider(Box::new(ProviderError {
            provider: "openai".into(),
            status: Some(429),
            retryable: true,
            message: "rate limited; retry_after: 12.5 seconds".into(),
            ..ProviderError::default()
        }))),
        Some(Duration::from_millis(12_500))
    );
    // Nothing to read → no hint, and the plain backoff applies.
    assert_eq!(
        retry_after_hint(&TinyAgentsError::Model("500 Internal Server Error".into())),
        None
    );
    assert_eq!(
        retry_after_hint(&TinyAgentsError::Validation("bad".into())),
        None
    );
}

#[test]
fn backoff_for_error_takes_the_max_of_backoff_and_retry_after() {
    // Regression test (LOOP-5): `parse_retry_after_ms` existed but had only
    // test callers, so a 429 saying `Retry-After: 30` was retried after the
    // policy's 200ms, three times, and then gave up.
    let policy = RetryPolicy::default();
    let rate_limited = TinyAgentsError::Model("429 rate limited, Retry-After: 30".into());

    assert_eq!(
        policy.backoff_for_error(0, &rate_limited),
        Duration::from_secs(30),
        "a server-supplied Retry-After must win over a shorter computed backoff"
    );

    // A hint shorter than the computed backoff can never shorten the wait, so a
    // bogus `Retry-After: 0` cannot defeat backoff.
    let tiny_hint = TinyAgentsError::Model("429 rate limited, Retry-After: 0".into());
    assert_eq!(
        policy.backoff_for_error(3, &tiny_hint),
        policy.backoff_for_attempt(3)
    );

    // No hint at all → identical to the plain schedule.
    let plain = TinyAgentsError::Model("502 bad gateway".into());
    assert_eq!(
        policy.backoff_for_error(1, &plain),
        policy.backoff_for_attempt(1)
    );
}

#[test]
fn retry_after_is_clamped_so_a_hostile_header_cannot_park_the_run() {
    let policy = RetryPolicy::default().with_max_retry_after_ms(5_000);
    let absurd = TinyAgentsError::Model("429 rate limited, Retry-After: 86400".into());
    assert_eq!(
        policy.backoff_for_error(0, &absurd),
        Duration::from_millis(5_000)
    );
}

#[tokio::test(start_paused = true)]
async fn sleep_backoff_for_error_honors_retry_after() {
    use tokio::time::Instant as TokioInstant;

    let policy = RetryPolicy::default();
    let rate_limited = TinyAgentsError::Model("429 rate limited, Retry-After: 30".into());

    let t0 = TokioInstant::now();
    policy.sleep_backoff_for_error(0, &rate_limited).await;
    assert_eq!(t0.elapsed(), Duration::from_secs(30));
}

// ── retry_on predicate (LOOP-5b) ──────────────────────────────────────────────

#[test]
fn model_errors_are_classified_from_their_message_not_assumed_transient() {
    // Regression test (LOOP-5b): the `Model(_)` arm returned `true`
    // unconditionally, so a permanent auth failure that never got a structured
    // `ProviderError` burned every attempt.
    assert!(!is_retryable(&TinyAgentsError::Model(
        "401 Unauthorized: invalid api key".into()
    )));
    assert!(!is_retryable(&TinyAgentsError::Model(
        "model gpt-9 does not exist".into()
    )));

    // Transient shapes stay retryable.
    assert!(is_retryable(&TinyAgentsError::Model(
        "502 Bad Gateway".into()
    )));
    assert!(is_retryable(&TinyAgentsError::Model(
        "429 Too Many Requests: rate limit exceeded".into()
    )));
    // Unclassifiable transport text keeps the permissive default.
    assert!(is_retryable(&TinyAgentsError::Model(
        "connection reset by peer".into()
    )));
}

#[test]
fn retry_on_predicate_overrides_the_builtin_classification() {
    use std::sync::Arc;

    // Default: no predicate → built-in classification, unchanged.
    let default_policy = RetryPolicy::default();
    assert!(default_policy.is_retryable_error(&TinyAgentsError::Tool("flaky".into())));
    assert!(!default_policy.is_retryable_error(&TinyAgentsError::Validation("bad".into())));

    // LangGraph's curated default shape: connection errors and 5xx, never a
    // programming error (a failing tool is the closest analogue here).
    let narrowed = RetryPolicy::default().with_retry_on(Arc::new(|err: &TinyAgentsError| {
        matches!(
            err,
            TinyAgentsError::Model(_) | TinyAgentsError::Provider(_)
        )
    }));
    assert!(narrowed.is_retryable_error(&TinyAgentsError::Model("timeout".into())));
    assert!(
        !narrowed.is_retryable_error(&TinyAgentsError::Tool("flaky".into())),
        "a custom predicate must be able to *narrow* the built-in set"
    );

    // And it can widen it too.
    let widened = RetryPolicy::default().with_retry_on(Arc::new(|_: &TinyAgentsError| true));
    assert!(widened.is_retryable_error(&TinyAgentsError::Validation("bad".into())));

    // Clearing restores the built-in classification.
    assert!(
        !widened
            .clone()
            .with_default_retry_on()
            .is_retryable_error(&TinyAgentsError::Validation("bad".into()))
    );
}

#[test]
fn should_retry_error_consults_the_custom_predicate() {
    use std::sync::Arc;

    // Regression test: `should_retry_error` called the free `is_retryable`
    // directly, so a caller's `retry_on` would have been silently ignored by
    // every retry loop in the harness.
    let policy = RetryPolicy::default()
        .with_max_attempts(3)
        .with_retry_on(Arc::new(|err: &TinyAgentsError| {
            matches!(err, TinyAgentsError::Validation(_))
        }));

    assert!(policy.should_retry_error(0, &TinyAgentsError::Validation("retry me".into())));
    assert!(!policy.should_retry_error(0, &TinyAgentsError::Tool("do not".into())));
    // The attempt cap still applies on top of the predicate.
    assert!(!policy.should_retry_error(2, &TinyAgentsError::Validation("retry me".into())));
}

#[test]
fn retry_policy_equality_and_debug_survive_the_predicate_field() {
    use std::sync::Arc;

    assert_eq!(RetryPolicy::default(), RetryPolicy::default());

    let predicate: crate::harness::retry::RetryPredicate = Arc::new(|_: &TinyAgentsError| true);
    let a = RetryPolicy::default().with_retry_on(predicate.clone());
    let b = RetryPolicy::default().with_retry_on(predicate);
    assert_eq!(a, b, "the same Arc compares equal");
    assert_ne!(a, RetryPolicy::default());
    assert_ne!(
        a,
        RetryPolicy::default().with_retry_on(Arc::new(|_: &TinyAgentsError| true)),
        "distinct closures are not equal"
    );

    assert!(format!("{a:?}").contains("custom predicate"));
    assert!(format!("{:?}", RetryPolicy::default()).contains("retry_on: None"));
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
