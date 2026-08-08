//! Wave 2 — LOOP-5b: the HTTP `Retry-After` header is read.
//!
//! Wave 1 landed the retry side (`retry_after_hint`, `backoff_for_error`, the
//! `max_retry_after_ms` clamp) but it read only the error's **message text**.
//! A provider that sends the header without echoing it into the JSON body was
//! simply not honored. This pins the structured path: the transport parses the
//! header into [`ProviderError::retry_after_ms`], and `retry_after_hint` reads
//! that field ahead of the text fallback.

use std::time::Duration;

use tinyagents::TinyAgentsError;
use tinyagents::harness::model::ProviderError;
use tinyagents::harness::retry::{RetryPolicy, retry_after_hint};

fn provider_error(message: &str, retry_after_ms: Option<u64>) -> TinyAgentsError {
    TinyAgentsError::Provider(Box::new(ProviderError {
        provider: "openai".to_string(),
        model: Some("gpt-5".to_string()),
        status: Some(429),
        code: Some("rate_limit_exceeded".to_string()),
        message: message.to_string(),
        retryable: true,
        retry_after_ms,
        raw: None,
    }))
}

#[test]
fn the_structured_field_is_honored_when_the_body_says_nothing() {
    // The header-only case: the provider sent `Retry-After: 30` but the JSON
    // body carries no wait at all. Before the field existed there was nothing
    // to read and the hint was `None`.
    let error = provider_error("Rate limit reached for gpt-5.", Some(30_000));
    assert_eq!(
        retry_after_hint(&error),
        Some(Duration::from_millis(30_000)),
        "the header value must be honored even when the body is silent"
    );
}

#[test]
fn the_structured_field_wins_over_the_message_text() {
    // Both present and disagreeing: the header is the contract, the text is a
    // string-matching fallback.
    let error = provider_error("Rate limited. retry-after: 2", Some(45_000));
    assert_eq!(retry_after_hint(&error), Some(Duration::from_millis(45_000)));
}

#[test]
fn the_message_text_fallback_still_works() {
    // Adapters that do not yet populate the field keep the wave-1 behaviour.
    let error = provider_error("Rate limited. retry-after: 2", None);
    assert_eq!(retry_after_hint(&error), Some(Duration::from_secs(2)));
}

#[test]
fn a_server_supplied_wait_drives_the_backoff_and_stays_clamped() {
    let policy = RetryPolicy::default();
    let honored = policy.backoff_for_error(0, &provider_error("rate limited", Some(30_000)));
    assert!(
        honored >= Duration::from_millis(30_000),
        "the computed backoff must be at least the server-supplied wait, got {honored:?}"
    );

    // The wave-1 clamp still applies to an absurd server value.
    let clamped = policy.backoff_for_error(0, &provider_error("rate limited", Some(86_400_000)));
    assert!(
        clamped <= Duration::from_millis(policy.max_retry_after_ms),
        "an absurd Retry-After must stay clamped, got {clamped:?}"
    );
}

#[test]
fn provider_error_round_trips_without_the_new_field() {
    // `#[serde(default)]` keeps payloads written before the field existed
    // decodable — important for anything that persists provider errors.
    let legacy: ProviderError = serde_json::from_str(
        r#"{"provider":"openai","message":"boom","retryable":true}"#,
    )
    .expect("legacy payload decodes");
    assert_eq!(legacy.retry_after_ms, None);
}
