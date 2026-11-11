use super::{HostedErrorKind, hosted_error};
use crate::error::TinyAgentsError;
use crate::middleware::AgentRun;
use crate::retry::FailoverReason;
use crate::terminal::{TerminalOutcome, TimeoutPhase};
use tinyinference_llm::model::ProviderError;

fn provider(status: u16, message: &str) -> TinyAgentsError {
    TinyAgentsError::Provider(Box::new(ProviderError {
        provider: "test".into(),
        model: None,
        status: Some(status),
        code: None,
        message: message.into(),
        retryable: false,
        retry_after_ms: None,
        raw: None,
        partial_message: None,
        stop_reason: None,
    }))
}

fn failed_run(error: &TinyAgentsError) -> AgentRun {
    let mut run = AgentRun::new();
    run.terminal = Some(TerminalOutcome::from_error(error, TimeoutPhase::AfterTurn));
    run
}

#[test]
fn a_provider_401_keeps_its_class_but_not_its_text() {
    let error = provider(401, "invalid x-api-key sk-secret");
    let hosted = hosted_error(&error, failed_run(&error));

    assert_eq!(hosted.kind, HostedErrorKind::Provider);
    assert!(!hosted.message.contains("sk-secret"));
    assert!(matches!(
        hosted.provider_failure(),
        Some(FailoverReason::Auth | FailoverReason::AuthPermanent)
    ));
}

#[test]
fn a_missing_model_and_a_rate_limit_are_told_apart() {
    let missing = provider(404, "model `nope` not found");
    let limited = provider(429, "too many requests");

    assert_eq!(
        hosted_error(&missing, failed_run(&missing)).provider_failure(),
        Some(FailoverReason::ModelNotFound)
    );
    assert_eq!(
        hosted_error(&limited, failed_run(&limited)).provider_failure(),
        Some(FailoverReason::RateLimit)
    );
}

#[test]
fn a_non_provider_failure_has_no_provider_class() {
    let error = TinyAgentsError::Cancelled;
    assert_eq!(
        hosted_error(&error, failed_run(&error)).provider_failure(),
        None
    );
}

#[test]
fn a_local_rate_limiter_denial_is_not_blamed_on_the_provider() {
    let local = TinyAgentsError::RateLimited("token bucket empty".into());
    let hosted = hosted_error(&local, failed_run(&local));
    assert_eq!(hosted.kind, HostedErrorKind::RateLimited);
    assert_eq!(hosted.provider_failure(), None);

    let upstream = provider(429, "too many requests");
    assert_eq!(
        hosted_error(&upstream, failed_run(&upstream)).provider_failure(),
        Some(FailoverReason::RateLimit)
    );
}

#[test]
fn a_summarizer_wrapped_failure_is_classified_by_what_it_wraps() {
    let wrap = |inner: TinyAgentsError| TinyAgentsError::SummarizationUsage {
        error: Box::new(inner),
        usage: tinyinference_llm::usage::Usage::default(),
    };
    let local = wrap(TinyAgentsError::RateLimited("token bucket empty".into()));
    let hosted = hosted_error(&local, failed_run(&local));
    assert_eq!(hosted.kind, HostedErrorKind::RateLimited);
    assert_eq!(hosted.provider_failure(), None);

    let upstream = wrap(provider(429, "too many requests"));
    let hosted = hosted_error(&upstream, failed_run(&upstream));
    assert_eq!(hosted.kind, HostedErrorKind::Provider);
    assert_eq!(hosted.provider_failure(), Some(FailoverReason::RateLimit));
}
