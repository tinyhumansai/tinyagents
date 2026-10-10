//! Typed terminal outcome of a run.
//!
//! A run used to end as one of three loosely related shapes: a
//! [`LoopExit`](crate::agent_loop) inside the loop, a
//! [`TinyAgentsError`] on the failure channel, and a free-form `String` on
//! [`AgentEvent::RunFailed`](crate::events::AgentEvent::RunFailed). A host that
//! wanted to know *why* a run ended — a deadline, a cancel, a provider outage,
//! a tripped guard — had to parse the string. [`TerminalOutcome`] is the one
//! structured answer, and this module is the one place that maps the loop's
//! exits and errors onto it.
//!
//! The legacy string is kept everywhere the outcome is attached
//! ([`TerminalOutcome::message`] mirrors it), so existing hosts are unaffected.
//!
//! # Mapping
//!
//! | Source | [`TerminalReason`] | [`TerminalClass`] |
//! | --- | --- | --- |
//! | model finished / `StopWithFinal` | `Completed` | `Success` |
//! | `StopWithPartial` call cap | `LimitReached(kind)` | `Failure` |
//! | `TinyAgentsError::Cancelled` | `Cancelled` | `Cancellation` |
//! | `TinyAgentsError::Timeout` (run deadline) | `Timeout` | `Timeout` |
//! | `TinyAgentsError::CallTimeout` (one call wedged) | `ProviderFailed(Some(Timeout))` | `Timeout` |
//! | `TinyAgentsError::RateLimited` (refillable bucket) | `ProviderFailed(Some(RateLimit))` | `Failure` |
//! | `TinyAgentsError::StreamIdleTimeout` (breaker threshold) | `ProviderFailed(Some(StreamIdleTimeout))` | `Timeout` |
//! | provider / model / overflow / empty-response errors | `ProviderFailed(reason)` | `Failure` |
//! | tool errors | `ToolFailed` | `Failure` |
//! | run caps, depth caps | `LimitReached(..)` | `Failure` |
//! | steering pause | `Paused` | `Suspended` |
//! | deferred tool calls | `Deferred` | `Suspended` |
//! | repeat / no-progress guard | `Halted` ([`TerminalOutcome::halted`]) | `Failure` |
//! | anything else | `Internal` | `Failure` |
//!
//! # Precedence
//!
//! When two outcomes compete for one run — a cancel races a deadline, a tool
//! failure surfaces while a limit trips — [`TerminalOutcome::merge`] keeps the
//! more authoritative one. Highest first:
//!
//! 1. external cancel (`Cancelled`)
//! 2. run deadline (`Timeout`)
//! 3. idle / provider-call timeout (`ProviderFailed(Some(Timeout))`)
//! 4. limits (`LimitReached`)
//! 5. no-progress / repeat guard (`Halted`)
//! 6. other failures (`ProviderFailed`, `ToolFailed`, `Internal`)
//! 7. suspension (`Paused`, `Deferred`)
//! 8. success (`Completed`)
//!
//! Ties keep the outcome already held (the earlier one). The merged outcome's
//! `provider_started` is the OR of both, so "did the provider ever run" is not
//! lost when a higher-precedence outcome wins.

use serde::{Deserialize, Serialize};

use crate::agent_loop::LoopExit;
use crate::error::TinyAgentsError;
use crate::events::LimitKind;
use crate::retry::FailoverReason;

/// Broad family of a [`TerminalReason`], for hosts that only branch on
/// success / timeout / cancel / failure / suspended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TerminalClass {
    /// The run reached a final answer.
    Success,
    /// A deadline or per-call timeout ended the run.
    Timeout,
    /// The run was cancelled from outside.
    Cancellation,
    /// The run ended without a usable result for any other reason.
    Failure,
    /// The run is resumable, not finished (paused or deferred).
    Suspended,
}

/// Where in the run a timeout landed.
///
/// Also used as the *site* a failure surfaced at ([`TerminalOutcome::from_error`]):
/// [`TerminalOutcome::from_error`] derives `provider_started` as `false` exactly
/// for [`TimeoutPhase::BeforeProvider`]; the loop drivers then overwrite it with
/// the run-wide dispatch history (a later call rejected before dispatch still
/// reports that an earlier call reached the provider).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TimeoutPhase {
    /// No model call had started yet (preflight, host preparation, resolution).
    BeforeProvider,
    /// A provider call was in flight.
    Provider,
    /// At least one provider call had started and none was in flight: a turn
    /// boundary, a tool batch, or post-turn middleware.
    AfterTurn,
}

/// Why a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TerminalReason {
    /// The model produced a final answer.
    Completed,
    /// A run cap tripped; `None` when the cap's kind is not known.
    LimitReached(Option<LimitKind>),
    /// The run's own wall-clock deadline elapsed.
    Timeout,
    /// The run was cancelled from outside.
    Cancelled,
    /// Steering latched a pause; the run is resumable.
    Paused,
    /// Tool calls are waiting on a human decision or host execution.
    Deferred,
    /// A no-progress / repeat guard stopped the run.
    Halted,
    /// The model provider failed, classified when possible.
    ProviderFailed(Option<FailoverReason>),
    /// A tool failed in a way that ended the run.
    ToolFailed,
    /// Anything the other reasons do not name.
    Internal,
}

impl TerminalReason {
    /// The [`TerminalClass`] this reason belongs to.
    pub fn class(self) -> TerminalClass {
        match self {
            Self::Completed => TerminalClass::Success,
            Self::Timeout
            | Self::ProviderFailed(Some(
                FailoverReason::Timeout | FailoverReason::StreamIdleTimeout,
            )) => TerminalClass::Timeout,
            Self::Cancelled => TerminalClass::Cancellation,
            Self::Paused | Self::Deferred => TerminalClass::Suspended,
            Self::LimitReached(_)
            | Self::Halted
            | Self::ProviderFailed(_)
            | Self::ToolFailed
            | Self::Internal => TerminalClass::Failure,
        }
    }

    /// Precedence rank used by [`TerminalOutcome::merge`]; lower wins.
    fn rank(&self) -> u8 {
        match self {
            Self::Cancelled => 0,
            Self::Timeout => 1,
            Self::ProviderFailed(Some(FailoverReason::Timeout)) => 2,
            Self::LimitReached(_) => 3,
            Self::Halted => 4,
            Self::ProviderFailed(_) | Self::ToolFailed | Self::Internal => 5,
            Self::Paused | Self::Deferred => 6,
            Self::Completed => 7,
        }
    }
}

/// The structured answer to "how and why did this run end".
///
/// Attached to [`AgentEvent::RunCompleted`](crate::events::AgentEvent::RunCompleted),
/// [`AgentEvent::RunFailed`](crate::events::AgentEvent::RunFailed) and
/// [`AgentRun::terminal`](crate::middleware::AgentRun::terminal). See the
/// [module docs](self) for the mapping and the merge precedence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TerminalOutcome {
    /// Why the run ended.
    pub reason: TerminalReason,
    /// The broad family of [`Self::reason`]; always `reason.class()`.
    pub class: TerminalClass,
    /// Where the timeout landed; `Some` only for timeout-class outcomes built
    /// from an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_phase: Option<TimeoutPhase>,
    /// Whether any provider call had started before the run ended.
    #[serde(default)]
    pub provider_started: bool,
    /// Human-readable description; mirrors the legacy `RunFailed::error`
    /// string for failures.
    pub message: String,
}

impl TerminalOutcome {
    /// Builds an outcome for `reason`, deriving the class from it.
    pub fn new(reason: TerminalReason, message: impl Into<String>) -> Self {
        Self {
            reason,
            class: reason.class(),
            timeout_phase: None,
            provider_started: false,
            message: message.into(),
        }
    }

    /// A successful completion.
    pub fn completed() -> Self {
        Self::new(TerminalReason::Completed, "")
    }

    /// A run stopped by a no-progress / repeat guard, with the guard's summary.
    pub fn halted(message: impl Into<String>) -> Self {
        Self::new(TerminalReason::Halted, message)
    }

    /// A tripped run cap. [`LimitKind::WallClock`] is the run deadline, so it
    /// maps to [`TerminalReason::Timeout`] rather than a limit.
    pub fn limit_reached(kind: Option<LimitKind>, message: impl Into<String>) -> Self {
        match kind {
            Some(LimitKind::WallClock) => Self::new(TerminalReason::Timeout, message),
            kind => Self::new(TerminalReason::LimitReached(kind), message),
        }
    }

    /// Fills in the kind of a [`TerminalReason::LimitReached`] whose kind was
    /// not known when it was classified (a `LimitExceeded` error carries only
    /// text). Other outcomes are returned unchanged.
    pub fn with_limit_kind(mut self, kind: Option<LimitKind>) -> Self {
        if self.reason == TerminalReason::LimitReached(None) {
            self.reason = match kind {
                Some(LimitKind::WallClock) => TerminalReason::Timeout,
                kind => TerminalReason::LimitReached(kind),
            };
            self.class = self.reason.class();
        }
        self
    }

    /// Records whether a provider call had started.
    pub fn with_provider_started(mut self, started: bool) -> Self {
        self.provider_started = started;
        self
    }

    /// Records where a timeout landed.
    pub fn with_timeout_phase(mut self, phase: TimeoutPhase) -> Self {
        self.timeout_phase = Some(phase);
        self
    }

    /// Classifies `error`. `site` is where the run stood when the error
    /// surfaced; it sets `provider_started` and, for timeouts, the phase.
    pub fn from_error(error: &TinyAgentsError, site: TimeoutPhase) -> Self {
        use TinyAgentsError as E;
        let message = error.to_string();
        let provider_started = site != TimeoutPhase::BeforeProvider;
        let outcome = match error {
            E::Cancelled => Self::new(TerminalReason::Cancelled, message),
            E::Timeout(_) => Self::new(TerminalReason::Timeout, message).with_timeout_phase(site),
            E::CallTimeout(_) => Self::new(
                TerminalReason::ProviderFailed(Some(FailoverReason::Timeout)),
                message,
            )
            .with_timeout_phase(site),
            E::RateLimited(_) | E::StreamIdleTimeout(_) => {
                let reason = FailoverReason::classify(error);
                let outcome = Self::new(TerminalReason::ProviderFailed(Some(reason)), message);
                if reason == FailoverReason::StreamIdleTimeout {
                    outcome.with_timeout_phase(site)
                } else {
                    outcome
                }
            }
            E::Provider(_)
            | E::Model(_)
            | E::ContextOverflow { .. }
            | E::ModelNotFound(_)
            | E::EmptyResponse
            | E::GenerationStalled
            | E::SummarizationUsage { .. } => {
                // A summarizer failure wraps the real error; classify that.
                let reason = match error {
                    E::SummarizationUsage { error: inner, .. } => FailoverReason::classify(inner),
                    other => FailoverReason::classify(other),
                };
                let outcome = Self::new(TerminalReason::ProviderFailed(Some(reason)), message);
                if reason == FailoverReason::Timeout {
                    outcome.with_timeout_phase(site)
                } else {
                    outcome
                }
            }
            E::Tool(_) | E::ToolFailed(_) | E::ToolNotFound(_) | E::ModelRetry(_) => {
                Self::new(TerminalReason::ToolFailed, message)
            }
            E::LimitExceeded(_)
            | E::RecursionLimit(_)
            | E::SubAgentDepth(_)
            | E::NodeVisitLimit { .. } => Self::new(TerminalReason::LimitReached(None), message),
            E::ApprovalRequired { .. } | E::CallDeferred { .. } => {
                Self::new(TerminalReason::Deferred, message)
            }
            E::Interrupted { node, .. } if node == "steering-pause" => {
                Self::new(TerminalReason::Paused, message)
            }
            E::Interrupted { .. } => Self::new(TerminalReason::Internal, message),
            _ => Self::new(TerminalReason::Internal, message),
        };
        // A provider-call timeout implies a provider call started.
        let started = provider_started || outcome.timeout_phase == Some(TimeoutPhase::Provider);
        outcome.with_provider_started(started)
    }

    /// Maps the loop's own deliberate exits. `provider_started` is whether any
    /// model call had been dispatched.
    pub(crate) fn from_loop_exit(exit: &LoopExit, provider_started: bool) -> Self {
        let outcome = match exit {
            LoopExit::Finished => Self::completed(),
            LoopExit::LimitStop(kind) => Self::limit_reached(
                Some(*kind),
                format!(
                    "stopped with the partial run: {} limit reached",
                    kind.as_str()
                ),
            ),
            LoopExit::Paused(pause) => Self::new(
                TerminalReason::Paused,
                pause.reason.clone().unwrap_or_else(|| {
                    format!("paused at checkpoint {}", pause.paused_at_checkpoint)
                }),
            ),
            LoopExit::Deferred(requests) => Self::new(
                TerminalReason::Deferred,
                format!(
                    "{} approval(s), {} external call(s) pending",
                    requests.approvals.len(),
                    requests.calls.len()
                ),
            ),
        };
        outcome.with_provider_started(provider_started)
    }

    /// Merges two competing outcomes, keeping the one with higher precedence
    /// (see the [module docs](self)). `self` wins ties. `provider_started` is
    /// the OR of both.
    pub fn merge(self, other: Self) -> Self {
        let started = self.provider_started || other.provider_started;
        let winner = if other.reason.rank() < self.reason.rank() {
            other
        } else {
            self
        };
        winner.with_provider_started(started)
    }
}

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod tests;
