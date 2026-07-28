//! Host classification of raw tool failures.
//!
//! See [`ToolOutcomeClassifier`] for the trait contract and
//! [`NoopToolOutcomeClassifier`] for the inert default.

use serde::{Deserialize, Serialize};

use crate::harness::ids::CallId;

/// Turns a raw tool failure into a structured, host-defined classification.
///
/// Synchronous and pure by contract — implementations inspect text and return,
/// with no I/O — mirroring
/// [`EventListener`][crate::harness::events::EventListener], the crate's other
/// non-async extension point. Every string in [`ToolFailure`] is authored by
/// the embedder; the crate defines no failure taxonomy and no user-facing copy.
pub trait ToolOutcomeClassifier<State: Send + Sync>: Send + Sync {
    /// Classifies a failed call, or returns `None` to leave it unclassified.
    fn classify(&self, state: &State, failure: &ToolFailureContext<'_>) -> Option<ToolFailure>;
}

/// A failed tool call awaiting classification.
#[derive(Clone, Debug)]
pub struct ToolFailureContext<'a> {
    /// Provider-assigned identity of the call that failed.
    pub call_id: &'a CallId,
    /// Name of the tool that failed.
    pub tool_name: &'a str,
    /// The failure text, with any separate error field and result body already
    /// combined so a classifier need not guess where the signal is.
    pub error: &'a str,
    /// `true` when the runtime aborted the call on its own deadline, which a
    /// classifier cannot infer reliably from text alone.
    pub timed_out: bool,
}

/// A host-defined classification of one failure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolFailure {
    /// Stable host-defined discriminant.
    ///
    /// Display and logging only. Runtime code must never branch on this value:
    /// the taxonomy behind it is the embedder's, and matching on its strings
    /// would pull host vocabulary into crate control flow. Branch on
    /// [`retry`][Self::retry] instead.
    pub class: String,
    /// Coarser host-defined grouping the class belongs to. Display only, for
    /// the same reason as [`class`][Self::class].
    pub category: String,
    /// Host-authored description of what went wrong.
    pub cause: String,
    /// Host-authored description of what to do next.
    pub next_action: String,
    /// How the runtime should treat a retry of this call.
    #[serde(default)]
    pub retry: RetryDisposition,
}

/// How the runtime should treat a retry of a failed call.
///
/// A crate-owned, product-neutral vocabulary so retry policy can live in crate
/// code without matching on host class names — and so "we do not know" stays
/// distinguishable from "yes, retry", which a plain `bool` collapses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryDisposition {
    /// Nothing is known about whether a retry would help. Runtimes should treat
    /// this conservatively rather than as an invitation to retry.
    #[default]
    Unknown,
    /// A retry cannot succeed; the failure is deterministic.
    Never,
    /// A retry may succeed straight away.
    Immediate,
    /// A retry may succeed after a delay — the failure is transient (a
    /// timeout, an unavailable dependency, a dropped connection).
    Backoff,
}

impl RetryDisposition {
    /// Whether the runtime may retry the call at all.
    ///
    /// [`Unknown`][Self::Unknown] answers `false`: an unclassified failure is
    /// not evidence that retrying is safe.
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::Immediate | Self::Backoff)
    }
}

/// A [`ToolOutcomeClassifier`] that classifies nothing.
///
/// `classify` returns `None` for every input, leaving failures unclassified.
/// Sync, so it needs no async machinery at all.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopToolOutcomeClassifier;

impl NoopToolOutcomeClassifier {
    /// Creates the classifier.
    pub fn new() -> Self {
        Self
    }
}

impl<State: Send + Sync> ToolOutcomeClassifier<State> for NoopToolOutcomeClassifier {
    fn classify(&self, _state: &State, _failure: &ToolFailureContext<'_>) -> Option<ToolFailure> {
        None
    }
}
