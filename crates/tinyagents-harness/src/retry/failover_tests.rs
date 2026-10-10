use super::*;
use tinyinference_llm::model::ProviderError;

fn provider(
    status: Option<u16>,
    code: Option<&str>,
    message: &str,
    retryable: bool,
) -> TinyAgentsError {
    TinyAgentsError::Provider(Box::new(ProviderError {
        provider: "test".into(),
        model: None,
        status,
        code: code.map(str::to_owned),
        message: message.into(),
        retryable,
        retry_after_ms: None,
        raw: None,
        partial_message: None,
        stop_reason: None,
    }))
}

fn reason_of(error: &TinyAgentsError) -> FailoverReason {
    FailoverReason::classify(error)
}

// ── classification ───────────────────────────────────────────────────────────

#[test]
fn classifies_http_statuses_on_provider_errors() {
    let cases = [
        (401, "unauthorized", FailoverReason::Auth),
        (403, "forbidden", FailoverReason::Auth),
        (402, "payment required", FailoverReason::Billing),
        (
            404,
            "model `gpt-x` does not exist",
            FailoverReason::ModelNotFound,
        ),
        (408, "request timeout", FailoverReason::Timeout),
        (429, "too many requests", FailoverReason::RateLimit),
        (503, "service unavailable", FailoverReason::Overloaded),
        (529, "overloaded", FailoverReason::Overloaded),
        (500, "internal error", FailoverReason::Transport),
        (502, "bad gateway", FailoverReason::Transport),
        (404, "route not found", FailoverReason::Format),
        (400, "bad request", FailoverReason::Format),
        (422, "unprocessable", FailoverReason::Format),
    ];
    for (status, message, expected) in cases {
        let error = provider(Some(status), None, message, false);
        assert_eq!(reason_of(&error), expected, "status {status}");
    }
}

#[test]
fn business_rate_limits_and_exhausted_quota_are_billing_not_rate_limit() {
    let quota = provider(Some(429), None, "You exceeded your current quota", false);
    assert_eq!(reason_of(&quota), FailoverReason::Billing);
    let credits = provider(
        Some(400),
        None,
        "Insufficient credits on this account",
        false,
    );
    assert_eq!(reason_of(&credits), FailoverReason::Billing);
}

#[test]
fn revoked_or_deactivated_keys_are_permanent_auth() {
    for message in [
        "API key has been revoked",
        "This account has been deactivated",
        "organization suspended",
    ] {
        let error = provider(Some(401), None, message, false);
        assert_eq!(
            reason_of(&error),
            FailoverReason::AuthPermanent,
            "{message}"
        );
    }
    let plain = provider(Some(401), None, "invalid api key", false);
    assert_eq!(reason_of(&plain), FailoverReason::Auth);
}

#[test]
fn fixable_or_per_project_auth_states_are_not_permanent() {
    for message in [
        "API key disabled for this project",
        "access_terminated_error: endpoint disabled",
        "key permanently scoped to another project",
    ] {
        let error = provider(Some(403), None, message, false);
        assert_eq!(reason_of(&error), FailoverReason::Auth, "{message}");
    }
}

#[test]
fn a_timeout_parameter_in_a_4xx_body_is_not_a_timeout() {
    let error = provider(
        Some(400),
        None,
        "invalid value for 'timeout': expected integer",
        false,
    );
    assert_eq!(reason_of(&error), FailoverReason::Format);
    let real = provider(Some(504), None, "gateway timeout", true);
    assert_eq!(reason_of(&real), FailoverReason::Timeout);
    let flattened = TinyAgentsError::Model("request timed out".into());
    assert_eq!(reason_of(&flattened), FailoverReason::Timeout);
}

#[test]
fn overloaded_wording_wins_over_a_generic_status() {
    let error = provider(Some(500), None, "The model is overloaded, try later", true);
    assert_eq!(reason_of(&error), FailoverReason::Overloaded);
}

#[test]
fn classifies_flattened_model_errors_from_their_text() {
    let cases = [
        ("invalid api key", FailoverReason::Auth),
        ("model not found: foo", FailoverReason::ModelNotFound),
        ("429 too many requests", FailoverReason::RateLimit),
        ("connection reset by peer", FailoverReason::Transport),
    ];
    for (message, expected) in cases {
        assert_eq!(
            reason_of(&TinyAgentsError::Model(message.into())),
            expected,
            "{message}"
        );
    }
}

#[test]
fn context_window_text_is_context_overflow_even_under_a_400() {
    let error = provider(
        Some(400),
        None,
        "This model's maximum context length is 8192 tokens",
        false,
    );
    assert_eq!(reason_of(&error), FailoverReason::ContextOverflow);
}

#[test]
fn classifies_dedicated_error_variants() {
    assert_eq!(
        reason_of(&TinyAgentsError::CallTimeout("slow".into())),
        FailoverReason::Timeout
    );
    assert_eq!(
        reason_of(&TinyAgentsError::Timeout("deadline".into())),
        FailoverReason::Timeout
    );
    assert_eq!(
        reason_of(&TinyAgentsError::EmptyResponse),
        FailoverReason::EmptyResponse
    );
    assert_eq!(
        reason_of(&TinyAgentsError::ModelNotFound("x".into())),
        FailoverReason::ModelNotFound
    );
    assert_eq!(
        reason_of(&TinyAgentsError::Validation("bad".into())),
        FailoverReason::Format
    );
    assert_eq!(
        reason_of(&TinyAgentsError::LimitExceeded("breaker".into())),
        FailoverReason::Unknown
    );
    assert_eq!(
        reason_of(&TinyAgentsError::RateLimited("refillable".into())),
        FailoverReason::RateLimit
    );
    assert_eq!(
        reason_of(&TinyAgentsError::StreamIdleTimeout("stalled".into())),
        FailoverReason::Timeout
    );
    assert_eq!(
        reason_of(&TinyAgentsError::ContextOverflow {
            provider: "openai".into(),
            model: None,
            message: "too long".into(),
        }),
        FailoverReason::ContextOverflow
    );
}

// ── decision table ───────────────────────────────────────────────────────────

fn state(
    retryable: bool,
    attempts_remaining: bool,
    larger_window_available: bool,
) -> FailoverState {
    FailoverState {
        retryable,
        attempts_remaining,
        larger_window_available,
    }
}

#[test]
fn transient_reasons_retry_the_same_model_then_fall_back() {
    for reason in [
        FailoverReason::RateLimit,
        FailoverReason::Overloaded,
        FailoverReason::Timeout,
        FailoverReason::Transport,
        FailoverReason::EmptyResponse,
        FailoverReason::Unknown,
    ] {
        assert_eq!(
            decide(reason, state(true, true, false)),
            FailoverDecision::RetrySame,
            "{reason:?} with attempts left"
        );
        assert_eq!(
            decide(reason, state(true, false, false)),
            FailoverDecision::Fallback,
            "{reason:?} exhausted"
        );
        assert_eq!(
            decide(reason, state(false, true, false)),
            FailoverDecision::Fallback,
            "{reason:?} the retry policy calls non-transient"
        );
    }
}

#[test]
fn auth_billing_and_model_not_found_skip_retries_and_fall_back() {
    for reason in [
        FailoverReason::Auth,
        FailoverReason::AuthPermanent,
        FailoverReason::Billing,
        FailoverReason::ModelNotFound,
    ] {
        // Even a (wrongly) retryable flag with attempts left must not retry.
        assert_eq!(
            decide(reason, state(true, true, false)),
            FailoverDecision::Fallback,
            "{reason:?}"
        );
    }
}

#[test]
fn format_errors_fall_back_because_provider_4xx_is_often_model_specific() {
    for retryable in [false, true] {
        assert_eq!(
            decide(FailoverReason::Format, state(retryable, true, false)),
            FailoverDecision::Fallback
        );
    }
}

#[test]
fn context_overflow_falls_back_only_when_a_larger_window_exists() {
    assert_eq!(
        decide(FailoverReason::ContextOverflow, state(true, true, true)),
        FailoverDecision::Fallback
    );
    assert_eq!(
        decide(FailoverReason::ContextOverflow, state(true, true, false)),
        FailoverDecision::Surface
    );
}

#[test]
fn only_permanent_auth_marks_a_model_skipped_for_the_run() {
    assert!(FailoverReason::AuthPermanent.skips_model_for_run());
    for reason in [
        FailoverReason::Auth,
        FailoverReason::Billing,
        FailoverReason::RateLimit,
        FailoverReason::ModelNotFound,
        FailoverReason::Unknown,
    ] {
        assert!(!reason.skips_model_for_run(), "{reason:?}");
    }
}

#[test]
fn reason_labels_are_stable_snake_case() {
    assert_eq!(FailoverReason::AuthPermanent.as_str(), "auth_permanent");
    assert_eq!(FailoverReason::ContextOverflow.as_str(), "context_overflow");
    assert_eq!(FailoverReason::RateLimit.as_str(), "rate_limit");
}

#[test]
fn state_for_reads_the_retry_policy_and_the_error() {
    let policy = RetryPolicy::default().with_max_attempts(2);
    let transient = TinyAgentsError::Model("connection reset".into());
    let state = FailoverState::for_error(&policy, 0, &transient);
    assert!(state.retryable && state.attempts_remaining);
    let exhausted = FailoverState::for_error(&policy, 1, &transient);
    assert!(exhausted.retryable && !exhausted.attempts_remaining);
    assert!(
        !exhausted.larger_window_available,
        "defaults to no larger window"
    );
}
