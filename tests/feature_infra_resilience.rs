//! Feature/integration tests for the harness resilience infrastructure
//! (`harness::retry` + `harness::limits`).
//!
//! Covers the durability knobs every level of a recursive run bottoms out in:
//! exponential backoff, transient-error classification, provider-failure
//! taxonomy (HTTP status parsing, rate-limit vs quota, `Retry-After`), model
//! fallback ordering, the token-bucket rate limiter, and the fail-closed
//! run-limit tracker.
//!
//! Deterministic and offline: backoff and the limiter take an explicit
//! `rand01` / `now`, so no wall-clock or RNG dependence.

use std::time::{Duration, Instant};

use tinyagents::error::TinyAgentsError;
use tinyagents::harness::limits::{LimitTracker, RunLimits};
use tinyagents::harness::model::ProviderError;
use tinyagents::harness::retry::{
    FallbackPolicy, ProviderFailureClass, RateLimiter, RetryPolicy, classify_provider_error,
    classify_provider_failure, is_retryable, parse_retry_after_ms, provider_error_is_retryable,
    structured_http_status,
};

// ── RetryPolicy backoff ─────────────────────────────────────────────────────

#[test]
fn backoff_grows_exponentially_then_caps() {
    let policy = RetryPolicy::default()
        .with_initial_backoff_ms(100)
        .with_multiplier(2.0)
        .with_max_backoff_ms(500);
    assert_eq!(policy.backoff_for_attempt(0), Duration::from_millis(100));
    assert_eq!(policy.backoff_for_attempt(1), Duration::from_millis(200));
    assert_eq!(policy.backoff_for_attempt(2), Duration::from_millis(400));
    // 800 would be next but the cap holds it at 500.
    assert_eq!(policy.backoff_for_attempt(3), Duration::from_millis(500));
}

#[test]
fn jitter_scales_backoff_by_supplied_random_value() {
    let policy = RetryPolicy::default()
        .with_initial_backoff_ms(1000)
        .with_multiplier(1.0)
        .with_jitter(true);
    // rand01 = 0.0 collapses the window to zero; 1.0 keeps the full base.
    assert_eq!(policy.backoff_for_attempt_with(0, 0.0), Duration::ZERO);
    assert_eq!(
        policy.backoff_for_attempt_with(0, 0.25),
        Duration::from_millis(250)
    );
    assert_eq!(
        policy.backoff_for_attempt_with(0, 1.0),
        Duration::from_millis(1000)
    );
}

#[test]
fn should_retry_honours_the_attempt_cap() {
    let policy = RetryPolicy::default().with_max_attempts(3);
    assert!(policy.should_retry(0));
    assert!(policy.should_retry(1));
    assert!(!policy.should_retry(2)); // third attempt is the last
}

#[test]
fn should_retry_error_combines_transience_and_attempt_cap() {
    let policy = RetryPolicy::default().with_max_attempts(2);
    let transient = TinyAgentsError::Tool("flaky dependency".into());
    let permanent = TinyAgentsError::Validation("bad schema".into());

    // Transient + attempts remaining → retry.
    assert!(policy.should_retry_error(0, &transient));
    // Transient but the cap is reached → no retry.
    assert!(!policy.should_retry_error(1, &transient));
    // Permanent error is never retried even with attempts remaining.
    assert!(!policy.should_retry_error(0, &permanent));
}

#[test]
fn max_attempts_capped_at_takes_the_stricter_ceiling() {
    let loose = RetryPolicy::default().with_max_attempts(10);
    // A run limit of 2 retries → 3 total attempts wins over the looser 10.
    assert_eq!(loose.max_attempts_capped_at(2), 3);
    // A looser run limit does not raise the policy's own cap.
    let tight = RetryPolicy::default().with_max_attempts(2);
    assert_eq!(tight.max_attempts_capped_at(100), 2);
}

// ── is_retryable classification ─────────────────────────────────────────────

#[test]
fn is_retryable_matches_documented_variants() {
    assert!(is_retryable(&TinyAgentsError::Model(
        "transport blip".into()
    )));
    assert!(is_retryable(&TinyAgentsError::Tool("timeout".into())));
    assert!(!is_retryable(&TinyAgentsError::Validation("nope".into())));
    assert!(!is_retryable(&TinyAgentsError::Memory("gone".into())));
}

#[test]
fn provider_error_retryability_follows_its_flag() {
    let retryable = ProviderError {
        provider: "openai".into(),
        model: None,
        status: Some(503),
        code: None,
        message: "service unavailable".into(),
        retryable: true,
        raw: None,
    };
    assert!(is_retryable(&TinyAgentsError::Provider(Box::new(
        retryable.clone()
    ))));
    assert!(provider_error_is_retryable(&retryable));

    let terminal = ProviderError {
        retryable: false,
        status: Some(401),
        message: "invalid api key".into(),
        ..retryable.clone()
    };
    assert!(!is_retryable(&TinyAgentsError::Provider(Box::new(
        terminal
    ))));
}

// ── Provider-failure taxonomy ───────────────────────────────────────────────

#[test]
fn structured_http_status_extracts_codes_from_envelopes() {
    assert_eq!(
        structured_http_status("API error (404): missing"),
        Some(404)
    );
    assert_eq!(structured_http_status("HTTP 503 upstream"), Some(503));
    assert_eq!(structured_http_status("status: 429 slow down"), Some(429));
    assert_eq!(structured_http_status("429 Too Many Requests"), Some(429));
    // A free-text digit run (a latency value) is not mistaken for a status.
    assert_eq!(structured_http_status("took 1234 ms"), None);
}

#[test]
fn classify_distinguishes_rate_limit_from_quota_exhaustion() {
    // A plain 429 that could succeed after backoff.
    assert_eq!(
        classify_provider_failure(Some(429), None, "too many requests"),
        ProviderFailureClass::RateLimited
    );
    // A 429 carrying quota/balance detail is terminal.
    let quota = classify_provider_failure(Some(429), None, "insufficient quota for this month");
    assert_eq!(quota, ProviderFailureClass::NonRetryableRateLimit);
    assert!(!quota.is_retryable());
}

#[test]
fn classify_maps_status_families_to_classes() {
    assert_eq!(
        classify_provider_failure(Some(503), None, "boom"),
        ProviderFailureClass::UpstreamUnhealthy
    );
    assert_eq!(
        classify_provider_failure(Some(400), None, "bad request"),
        ProviderFailureClass::NonRetryable
    );
    // No status but transient text falls through to Retryable.
    assert_eq!(
        classify_provider_failure(None, None, "connection reset by peer"),
        ProviderFailureClass::Retryable
    );
}

#[test]
fn classify_provider_error_and_reason_labels_are_stable() {
    let err = ProviderError {
        provider: "anthropic".into(),
        model: Some("claude".into()),
        status: Some(500),
        code: None,
        message: "internal server error".into(),
        retryable: true,
        raw: None,
    };
    let class = classify_provider_error(&err);
    assert_eq!(class, ProviderFailureClass::UpstreamUnhealthy);
    assert_eq!(class.reason(), "upstream_unhealthy");
    assert!(class.is_retryable());
}

#[test]
fn parse_retry_after_reads_seconds_into_millis() {
    assert_eq!(parse_retry_after_ms("Retry-After: 3"), Some(3_000));
    assert_eq!(parse_retry_after_ms("retry_after 0.5 seconds"), Some(500));
    assert_eq!(parse_retry_after_ms("no hint here"), None);
}

// ── FallbackPolicy ──────────────────────────────────────────────────────────

#[test]
fn fallback_walks_the_model_chain_then_stops() {
    let policy = FallbackPolicy::new(["primary", "secondary", "tertiary"]);
    assert_eq!(policy.next_after("primary"), Some("secondary"));
    assert_eq!(policy.next_after("secondary"), Some("tertiary"));
    assert_eq!(policy.next_after("tertiary"), None);
    // An unknown current model has no successor.
    assert_eq!(policy.next_after("unknown"), None);
}

// ── RateLimiter (token bucket) ──────────────────────────────────────────────

#[test]
fn rate_limiter_drains_then_refills_over_time() {
    let limiter = RateLimiter::new(2, 1.0); // capacity 2, 1 token/sec
    let t0 = Instant::now();
    assert!(limiter.try_acquire(1, t0));
    assert!(limiter.try_acquire(1, t0));
    // Bucket empty: the third acquire at the same instant fails.
    assert!(!limiter.try_acquire(1, t0));

    // One second later exactly one token has refilled.
    let t1 = t0 + Duration::from_secs(1);
    assert_eq!(limiter.available(t1), 1);
    assert!(limiter.try_acquire(1, t1));
    assert!(!limiter.try_acquire(1, t1));
}

#[test]
fn rate_limiter_reports_when_acquisition_can_never_succeed() {
    let limiter = RateLimiter::new(5, 2.0);
    assert_eq!(limiter.capacity(), 5);
    assert_eq!(limiter.refill_per_sec(), 2.0);
    // A request larger than capacity can never be satisfied.
    assert!(!limiter.can_ever_acquire(6));
    assert!(limiter.can_ever_acquire(5));

    // With no refill, a drained bucket can never recover a spent token.
    let stalled = RateLimiter::new(1, 0.0);
    assert!(!stalled.can_ever_acquire(1));
}

// ── LimitTracker (fail-closed run limits) ───────────────────────────────────

#[test]
fn model_and_tool_caps_are_inclusive_and_fail_closed() {
    let limits = RunLimits::default()
        .with_max_model_calls(2)
        .with_max_tool_calls(1);
    let mut tracker = LimitTracker::new(limits);

    assert!(tracker.record_model_call().is_ok());
    assert!(tracker.record_model_call().is_ok()); // exactly 2 allowed
    assert!(tracker.record_model_call().is_err()); // the 3rd trips
    assert_eq!(tracker.model_calls(), 3);

    assert!(tracker.record_tool_call().is_ok());
    assert!(tracker.record_tool_call().is_err());
}

#[test]
fn remaining_model_calls_saturates_at_zero() {
    let mut tracker = LimitTracker::new(RunLimits::default().with_max_model_calls(1));
    assert_eq!(tracker.remaining_model_calls(), 1);
    let _ = tracker.record_model_call();
    assert_eq!(tracker.remaining_model_calls(), 0);
    // Over the cap: remaining stays at 0 rather than wrapping.
    let _ = tracker.record_model_call();
    assert_eq!(tracker.remaining_model_calls(), 0);
}

#[test]
fn wall_clock_check_passes_when_no_deadline_configured() {
    let tracker = LimitTracker::new(RunLimits::default());
    assert!(tracker.limits().max_wall_clock_ms.is_none());
    assert!(tracker.check_wall_clock().is_ok());
    // No deadline → no remaining-budget bound.
    assert!(tracker.remaining_wall_clock().is_none());
}

#[test]
fn wall_clock_deadline_yields_a_shrinking_remaining_budget() {
    let tracker = LimitTracker::new(RunLimits::default().with_max_wall_clock_ms(Some(60_000)));
    let remaining = tracker.remaining_wall_clock().expect("deadline configured");
    // Immediately after construction the whole budget is (nearly) intact.
    assert!(remaining <= Duration::from_millis(60_000));
    assert!(remaining > Duration::from_millis(59_000));
    // Not yet elapsed, so the between-call check still passes.
    assert!(tracker.check_wall_clock().is_ok());
}

#[test]
fn sync_call_limits_overrides_caps_while_preserving_counts() {
    let mut tracker = LimitTracker::new(RunLimits::default().with_max_model_calls(1));
    let _ = tracker.record_model_call();
    // Reconcile to a higher ceiling; the already-recorded count survives.
    tracker.sync_call_limits(5, 10);
    assert_eq!(tracker.model_calls(), 1);
    assert!(tracker.record_model_call().is_ok());
    assert_eq!(tracker.limits().max_model_calls, 5);
    assert_eq!(tracker.limits().max_tool_calls, 10);
}
