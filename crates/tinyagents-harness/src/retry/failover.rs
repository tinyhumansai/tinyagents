//! Reason-aware failover: *why* a model call failed decides what happens next.
//!
//! Retrying and falling back used to be driven by one boolean
//! ([`is_retryable`]) and applied to every error alike: a permanent `401`
//! walked the whole fallback chain with a guaranteed-identical failure per
//! hop, while a malformed request was replayed against every model on the
//! chain. This module separates the two questions:
//!
//! 1. [`FailoverReason::classify`] names the cause, reusing
//!    [`classify_provider_failure`] and the provider-body matchers from
//!    [`tinyinference_llm::failure`] rather than inventing new heuristics.
//! 2. [`decide`] is a pure function from `(reason, state)` to a
//!    [`FailoverDecision`] — [`RetrySame`](FailoverDecision::RetrySame),
//!    [`Fallback`](FailoverDecision::Fallback) or
//!    [`Surface`](FailoverDecision::Surface).
//!
//! The agent loop's model call consults [`decide`] on every failed attempt.
//!
//! # Decision table
//!
//! | Reason | Decision |
//! |---|---|
//! | `RateLimit`, `Overloaded`, `Timeout`, `Transport`, `EmptyResponse`, `Unknown` | `RetrySame` while the retry policy calls the error transient **and** attempts remain; then `Fallback` |
//! | `Auth`, `Billing`, `ModelNotFound` | `Fallback` immediately (a retry cannot change the answer) |
//! | `AuthPermanent` | `Fallback` immediately, and the model is skipped for the rest of the run ([`FailoverReason::skips_model_for_run`]) |
//! | `Format` | `Fallback` (never retried on the same model). A 4xx is often *provider*-specific — OpenAI strict-schema, Gemini `Unknown name`, Anthropic `input_schema` — so another model may accept the request. Nothing is provably model-independent, so nothing surfaces here |
//! | `ContextOverflow` | `Fallback` only to a candidate whose profile `max_input_tokens` is strictly larger than the current model's ([`FailoverState::larger_window_available`]); otherwise `Surface` (compaction is the remedy) |
//! | `LimitExceeded` (run/provider budget) | Surfaces immediately; fallback cannot bypass the exhausted budget |
//! | `StreamIdleTimeout` | Falls back after the current model reaches its breaker threshold |
//!
//! A custom [`RetryPolicy::retry_on`] predicate keeps authority over the
//! *transient* reasons (it can veto a retry), but it cannot turn a permanent
//! reason such as `Auth` or `Format` back into a same-model retry.

use super::{FailoverDecision, FailoverReason, FailoverState, RetryPolicy};
use crate::error::TinyAgentsError;
use tinyinference_llm::failure::{
    ProviderFailureClass, body_indicates_auth_key_error, body_indicates_insufficient_credits,
    body_indicates_no_model_loaded, body_indicates_provider_access_policy_denied,
    body_indicates_quota_exhausted, classify_provider_failure, is_context_window_exceeded_message,
    structured_http_status,
};

impl FailoverState {
    /// Builds the state for `error` on the zero-indexed `attempt` under
    /// `policy`. Pass the policy already capped by the harness's
    /// `max_retries_per_call`.
    pub fn for_error(policy: &RetryPolicy, attempt: usize, error: &TinyAgentsError) -> Self {
        Self {
            retryable: policy.is_retryable_error(error),
            attempts_remaining: policy.should_retry(attempt),
            larger_window_available: false,
        }
    }
}

impl FailoverReason {
    /// Stable snake_case label for logs and telemetry.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::AuthPermanent => "auth_permanent",
            Self::Billing => "billing",
            Self::RateLimit => "rate_limit",
            Self::Overloaded => "overloaded",
            Self::LimitExceeded => "limit_exceeded",
            Self::StreamIdleTimeout => "stream_idle_timeout",
            Self::Timeout => "timeout",
            Self::Format => "format",
            Self::ContextOverflow => "context_overflow",
            Self::ModelNotFound => "model_not_found",
            Self::EmptyResponse => "empty_response",
            Self::Transport => "transport",
            Self::Unknown => "unknown",
        }
    }

    /// `true` when a model that failed for this reason should be skipped for
    /// the remainder of the run (the cross-call skip hint). Only
    /// [`FailoverReason::AuthPermanent`] qualifies: every other reason can
    /// clear on its own (rate limits, outages) or be fixed by the host
    /// (a refreshed token, a topped-up balance).
    pub fn skips_model_for_run(self) -> bool {
        matches!(self, Self::AuthPermanent)
    }

    /// Classifies `error`.
    pub fn classify(error: &TinyAgentsError) -> Self {
        match error {
            TinyAgentsError::Provider(provider) => classify_text(
                provider.status,
                provider.code.as_deref(),
                &provider.message,
                provider.retryable,
            ),
            TinyAgentsError::Model(message) => classify_text(None, None, message, true),
            TinyAgentsError::ContextOverflow { .. } => Self::ContextOverflow,
            TinyAgentsError::ModelNotFound(_) => Self::ModelNotFound,
            TinyAgentsError::EmptyResponse => Self::EmptyResponse,
            TinyAgentsError::CallTimeout(_) | TinyAgentsError::Timeout(_) => Self::Timeout,
            TinyAgentsError::RateLimited(_) => Self::RateLimit,
            TinyAgentsError::LimitExceeded(_) => Self::LimitExceeded,
            TinyAgentsError::StreamIdleTimeout(_) => Self::StreamIdleTimeout,
            TinyAgentsError::Validation(_) => Self::Format,
            _ => Self::Unknown,
        }
    }
}

/// The failover table. Pure; see the [module docs](self) for the rationale.
pub fn decide(reason: FailoverReason, state: FailoverState) -> FailoverDecision {
    use FailoverReason::*;
    match reason {
        RateLimit | Overloaded | Timeout | Transport | EmptyResponse | Unknown => {
            if state.retryable && state.attempts_remaining {
                FailoverDecision::RetrySame
            } else {
                FailoverDecision::Fallback
            }
        }
        Auth | AuthPermanent | Billing | ModelNotFound | Format => FailoverDecision::Fallback,
        LimitExceeded => FailoverDecision::Surface,
        StreamIdleTimeout => FailoverDecision::Fallback,
        ContextOverflow if state.larger_window_available => FailoverDecision::Fallback,
        ContextOverflow => FailoverDecision::Surface,
    }
}

/// Markers of a credential that will not recover on its own. Deliberately
/// narrow: "disabled", "terminated" and the like also describe fixable or
/// per-project/per-endpoint states (a key disabled for one project, an access
/// policy on one endpoint), which must not write a model off for the run.
const PERMANENT_AUTH_MARKERS: &[&str] = &["revoked", "deactivated", "suspended", "banned"];

fn classify_text(
    status: Option<u16>,
    code: Option<&str>,
    message: &str,
    provider_retryable: bool,
) -> FailoverReason {
    use FailoverReason::*;
    let lower = match code {
        Some(code) if !code.trim().is_empty() => format!("{message} {code}").to_ascii_lowercase(),
        _ => message.to_ascii_lowercase(),
    };
    let status = status.or_else(|| structured_http_status(message));

    if is_context_window_exceeded_message(message) || lower.contains("context_length_exceeded") {
        return ContextOverflow;
    }

    let failure = classify_provider_failure(status, code, message);
    if status == Some(402)
        || failure == ProviderFailureClass::NonRetryableRateLimit
        || body_indicates_insufficient_credits(message)
        || body_indicates_quota_exhausted(message)
        || lower.contains("insufficient_quota")
        || lower.contains("current quota")
    {
        return Billing;
    }

    let key_error = body_indicates_auth_key_error(message)
        || body_indicates_provider_access_policy_denied(message);
    if matches!(status, Some(401 | 403)) || key_error {
        return if PERMANENT_AUTH_MARKERS.iter().any(|m| lower.contains(m)) {
            AuthPermanent
        } else {
            Auth
        };
    }

    if lower.contains("model_not_found")
        || lower.contains("model not found")
        || body_indicates_no_model_loaded(message)
        || (lower.contains("model")
            && (lower.contains("does not exist") || lower.contains("unknown model"))
            && !matches!(status, Some(429 | 500..=599)))
        || (status == Some(404) && lower.contains("model"))
    {
        return ModelNotFound;
    }

    if status == Some(429) || failure == ProviderFailureClass::RateLimited {
        return RateLimit;
    }
    // Text-only timeout detection must not fire on a 4xx body that merely names
    // a `timeout` request parameter ("invalid value for 'timeout'").
    let client_error = matches!(status, Some(400..=499)) && status != Some(408);
    if status == Some(408)
        || (!client_error && (lower.contains("timed out") || lower.contains("timeout")))
    {
        return Timeout;
    }
    if matches!(status, Some(503 | 529))
        || lower.contains("overloaded")
        || lower.contains("over capacity")
    {
        return Overloaded;
    }
    match status {
        Some(409) | Some(500..=599) => return Transport,
        Some(400..=499) => return Format,
        _ => {}
    }

    match failure {
        ProviderFailureClass::UpstreamUnhealthy => Transport,
        ProviderFailureClass::NonRetryable if !provider_retryable || status.is_none() => Format,
        _ => Transport,
    }
}

#[cfg(test)]
#[path = "failover_tests.rs"]
mod test;
