//! Host seam for deciding what a tool result *means*.
//!
//! The runtime knows whether a tool call returned; it does not know whether
//! what came back should be treated as a success, retried, or given up on.
//! That judgement is product knowledge, and it is tool-specific: a `404` from a
//! CRM lookup usually means "no such record" — a perfectly good empty result —
//! while a `404` from a document fetch means the run cannot proceed. The same
//! transport-level shape lands in two different classes depending on which
//! integration produced it, so no rule the crate could hard-code would be right
//! for both. Hence a trait: the host, which owns the integration catalogue,
//! answers.
//!
//! This capability is **optional** (RFC §3.8). A host that has no per-tool
//! knowledge passes `None` and the runtime skips classification entirely,
//! rather than being handed a stub that always answers
//! [`OutcomeClass::Success`]. Absence and "classified as fine" are different
//! facts, and conflating them is exactly the failure mode RFC §2.3 forbids.
//! [`ErrorFieldClassifier`] exists for hosts that want the obvious baseline
//! *deliberately*, not as a silent default.
//!
//! **Reuses [`ToolResult`] on purpose.** There is no parallel
//! "classifiable result" type here. A second result struct would have to be
//! populated by the runtime from the real one, and every field added to
//! [`ToolResult`] would then either need mirroring or would be invisible to
//! classifiers — a drift surface with no upside. Classifiers read the real
//! result, including `raw`, which is where a host's structured error codes
//! live.
//!
//! **Dependency rule:** `serde` + `std` only, matching the inert-value-type
//! carve-out in [`crate::config::types`]. A host can implement this
//! trait without pulling an engine.

use serde::{Deserialize, Serialize};

use crate::tool::ToolResult;

// ── OutcomeClass ──────────────────────────────────────────────────────────────

/// What the runtime should do with a tool result.
///
/// Three variants, not a boolean, because "failed" and "worth trying again"
/// are independent facts and the loop reacts differently to each: a retryable
/// failure may be re-dispatched without telling the model anything new, while a
/// permanent failure must be surfaced into the transcript so the model can pick
/// a different approach instead of burning iterations on the same call.
///
/// Deliberately closed and coarse. A richer taxonomy (rate-limited,
/// unauthorized, not-found, …) would be transport vocabulary leaking into a
/// decision the host has already made — the host maps its own detail *into*
/// these three and keeps the detail on its side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeClass {
    /// The call did what was asked. Includes legitimately empty results — a
    /// search that matched nothing succeeded.
    #[default]
    Success,
    /// The call failed for a reason that may not recur: a timeout, a throttle,
    /// a transient upstream error. The runtime may re-dispatch the identical
    /// call.
    ///
    /// A classifier must only return this for calls it believes are safe to
    /// repeat. The runtime cannot know whether a tool had side effects, so this
    /// variant is an assertion by the host that repeating is acceptable.
    RetryableFailure,
    /// The call failed in a way that repeating will not fix: bad arguments,
    /// missing permissions, an entity that does not exist. The runtime should
    /// surface it to the model rather than retry.
    PermanentFailure,
}

impl OutcomeClass {
    /// Whether the result should be treated as a successful outcome.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Whether the runtime may re-dispatch the identical call.
    ///
    /// Only [`RetryableFailure`](Self::RetryableFailure) is retryable —
    /// [`Success`](Self::Success) has nothing to retry, and re-running a
    /// [`PermanentFailure`](Self::PermanentFailure) spends an iteration to
    /// obtain the same answer.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RetryableFailure)
    }

    /// Whether this class represents a failure of either kind.
    ///
    /// Useful for reporting paths that care that something went wrong but not
    /// what the loop should do about it.
    pub fn is_failure(&self) -> bool {
        !self.is_success()
    }
}

// ── ToolOutcomeClassifier ─────────────────────────────────────────────────────

/// Classifies a completed tool result (RFC §3.8).
///
/// Synchronous by design. Classification runs on the turn's critical path once
/// per tool result, and an `async` signature would invite implementations that
/// perform I/O there — a network round trip to decide whether a network round
/// trip failed. A host needing external knowledge should load it up front and
/// consult it from memory here.
///
/// Implementations must be pure with respect to the runtime: no side effects
/// the loop depends on, and the same `(name, result)` pair should classify the
/// same way every time. The runtime may call `classify` more than once for one
/// result (for example when both the retry decision and a reporting path need
/// it) and does not memoize.
pub trait ToolOutcomeClassifier: Send + Sync {
    /// Classifies `result`, which was produced by the tool named `name`.
    ///
    /// `name` is passed separately from `result.name` because the per-tool
    /// judgement is the whole point of this seam — the classifier is expected
    /// to branch on it — and because the runtime knows the name it dispatched
    /// even when a misbehaving tool returns a result stamped with a different
    /// one. Treat `name` as authoritative.
    fn classify(&self, name: &str, result: &ToolResult) -> OutcomeClass;
}

// ── ErrorFieldClassifier ──────────────────────────────────────────────────────

/// The baseline classifier: reads [`ToolResult::error`] and nothing else.
///
/// `Some(_)` becomes [`OutcomeClass::PermanentFailure`], `None` becomes
/// [`OutcomeClass::Success`]. It ignores `name`, `content`, and `raw`, because
/// interpreting any of those would require knowing which tool ran — precisely
/// the knowledge this crate does not have.
///
/// **Why `PermanentFailure` rather than `RetryableFailure`.** Retrying is the
/// answer that can cause harm: the runtime does not know whether the tool had
/// side effects, so a blanket retry can double-send a message or double-charge
/// a card, and at best it spends turn iterations re-earning the same error.
/// Classifying as permanent surfaces the error to the model, which can then
/// choose to call again with different arguments — a strictly safer default,
/// and the retryable half is exactly what a host is expected to override.
///
/// Suitable for tests, local runs, and hosts that genuinely have no per-tool
/// error taxonomy. A host with one should implement
/// [`ToolOutcomeClassifier`] itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ErrorFieldClassifier;

impl ErrorFieldClassifier {
    /// Creates the classifier. Stateless — every instance behaves identically.
    pub fn new() -> Self {
        Self
    }
}

impl ToolOutcomeClassifier for ErrorFieldClassifier {
    fn classify(&self, _name: &str, result: &ToolResult) -> OutcomeClass {
        match result.error {
            Some(_) => OutcomeClass::PermanentFailure,
            None => OutcomeClass::Success,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_with_error(error: Option<&str>) -> ToolResult {
        ToolResult {
            call_id: "call-1".to_string(),
            name: "search".to_string(),
            content: "some content".to_string(),
            raw: None,
            error: error.map(str::to_string),
            elapsed_ms: 3,
        }
    }

    // ── OutcomeClass invariants ───────────────────────────────────────────────

    #[test]
    fn default_class_is_success() {
        assert_eq!(OutcomeClass::default(), OutcomeClass::Success);
    }

    #[test]
    fn only_retryable_failure_is_retryable() {
        assert!(!OutcomeClass::Success.is_retryable());
        assert!(OutcomeClass::RetryableFailure.is_retryable());
        assert!(!OutcomeClass::PermanentFailure.is_retryable());
    }

    #[test]
    fn success_and_failure_partition_the_enum() {
        for class in [
            OutcomeClass::Success,
            OutcomeClass::RetryableFailure,
            OutcomeClass::PermanentFailure,
        ] {
            assert_ne!(
                class.is_success(),
                class.is_failure(),
                "{class:?} must be exactly one of success/failure"
            );
        }
    }

    #[test]
    fn class_round_trips_through_serde_as_snake_case() {
        let json = serde_json::to_string(&OutcomeClass::RetryableFailure).expect("serialize");
        assert_eq!(json, "\"retryable_failure\"");
        let back: OutcomeClass = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, OutcomeClass::RetryableFailure);
    }

    // ── ErrorFieldClassifier behaviour ────────────────────────────────────────

    #[test]
    fn absent_error_classifies_as_success() {
        let classifier = ErrorFieldClassifier::new();
        assert_eq!(
            classifier.classify("search", &result_with_error(None)),
            OutcomeClass::Success
        );
    }

    #[test]
    fn present_error_classifies_as_permanent_failure() {
        let classifier = ErrorFieldClassifier::new();
        assert_eq!(
            classifier.classify("search", &result_with_error(Some("boom"))),
            OutcomeClass::PermanentFailure
        );
    }

    #[test]
    fn empty_error_string_still_counts_as_a_failure() {
        // `Some("")` is a tool reporting failure without a message; the field's
        // presence is the signal, so it must not be mistaken for success.
        let classifier = ErrorFieldClassifier::new();
        assert_eq!(
            classifier.classify("search", &result_with_error(Some(""))),
            OutcomeClass::PermanentFailure
        );
    }

    #[test]
    fn never_returns_retryable() {
        // Pinned deliberately: the safe-by-default choice is load-bearing, and
        // a host relying on it must not have retries appear under it silently.
        let classifier = ErrorFieldClassifier::new();
        for error in [None, Some("timed out"), Some("429 rate limited")] {
            assert!(
                !classifier
                    .classify("any_tool", &result_with_error(error))
                    .is_retryable()
            );
        }
    }

    #[test]
    fn tool_name_does_not_change_the_verdict() {
        let classifier = ErrorFieldClassifier::new();
        let failed = result_with_error(Some("boom"));
        assert_eq!(
            classifier.classify("alpha", &failed),
            classifier.classify("omega", &failed)
        );
    }

    #[test]
    fn content_and_raw_are_ignored() {
        let classifier = ErrorFieldClassifier::new();
        let mut noisy = result_with_error(None);
        noisy.content = "Error: everything is on fire".to_string();
        noisy.raw = Some(serde_json::json!({ "status": 500 }));
        assert_eq!(
            classifier.classify("search", &noisy),
            OutcomeClass::Success,
            "only the error field is consulted"
        );
    }

    #[test]
    fn usable_as_a_trait_object() {
        // The runtime holds this capability as `Option<Arc<dyn …>>`, so the
        // trait must stay object-safe.
        let classifier: std::sync::Arc<dyn ToolOutcomeClassifier> =
            std::sync::Arc::new(ErrorFieldClassifier);
        assert_eq!(
            classifier.classify("search", &result_with_error(Some("nope"))),
            OutcomeClass::PermanentFailure
        );
    }
}
