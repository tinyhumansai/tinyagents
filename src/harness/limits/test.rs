//! Unit tests for run-scoped limit enforcement.
//!
//! Covers the counter/cap smoke path, the fail-open vs fail-closed
//! reconciliation of two limit sources (LOOP-1), and the
//! error-vs-stop-with-partial exhaustion behaviour (LOOP-9b).

use super::{LimitBehavior, LimitKind, LimitOutcome, LimitTracker, RunLimits};
use crate::error::TinyAgentsError;

#[test]
fn smoke_default_limits_compile() {
    let limits = RunLimits::default();
    let mut tracker = LimitTracker::new(limits);
    tracker.record_model_call().unwrap();
    assert_eq!(tracker.model_calls(), 1);
}

// ── LOOP-1: reconciling two limit sources ────────────────────────────────────

#[test]
fn tighten_call_limits_keeps_the_stricter_cap_in_both_directions() {
    // Regression test (LOOP-1): `sync_call_limits` is a plain assignment with
    // no `min`, so the agent loop's per-run call overwrote a caller's explicit
    // `RunConfig` cap *upward* — `RunConfig::new("r").with_max_model_calls(2)`
    // against the default policy ran 25. The existing coverage only pinned the
    // loosening direction, so nothing caught it.

    // A looser candidate must NOT widen an existing cap.
    let mut strict = LimitTracker::new(RunLimits::default().with_max_model_calls(2));
    strict.tighten_call_limits(25, 50);
    assert_eq!(
        strict.limits().max_model_calls,
        2,
        "a looser second source silently widened an explicit cap"
    );

    // A stricter candidate does tighten it.
    let mut loose = LimitTracker::new(RunLimits::default());
    loose.tighten_call_limits(5, 10);
    assert_eq!(loose.limits().max_model_calls, 5);
    assert_eq!(loose.limits().max_tool_calls, 10);

    // Each axis is reconciled independently.
    let mut mixed = LimitTracker::new(
        RunLimits::default()
            .with_max_model_calls(3)
            .with_max_tool_calls(999),
    );
    mixed.tighten_call_limits(100, 7);
    assert_eq!(mixed.limits().max_model_calls, 3);
    assert_eq!(mixed.limits().max_tool_calls, 7);
}

#[test]
fn tighten_call_limits_preserves_recorded_counts() {
    let mut tracker = LimitTracker::new(RunLimits::default());
    tracker.record_model_call().unwrap();
    tracker.record_tool_call().unwrap();
    tracker.tighten_call_limits(5, 10);
    assert_eq!(tracker.model_calls(), 1);
    assert_eq!(tracker.tool_calls(), 1);
}

#[test]
fn tighten_call_limits_actually_trips_at_the_stricter_cap() {
    // The cap must be enforced, not merely reported.
    let mut tracker = LimitTracker::new(RunLimits::default().with_max_model_calls(2));
    tracker.tighten_call_limits(25, 50);
    tracker.record_model_call().unwrap();
    tracker.record_model_call().unwrap();
    let err = tracker
        .record_model_call()
        .expect_err("the caller's cap of 2 must still trip");
    assert!(err.to_string().contains('2'), "got {err}");
}

#[test]
fn sync_call_limits_remains_the_documented_fail_open_override() {
    // Kept deliberately, for the case a `RunPolicy` must raise a cap above the
    // `RunConfig` *default*. Pinned so the two methods cannot be confused.
    let mut tracker = LimitTracker::new(RunLimits::default().with_max_model_calls(2));
    tracker.sync_call_limits(30, 1_000);
    assert_eq!(tracker.limits().max_model_calls, 30);
    assert_eq!(tracker.limits().max_tool_calls, 1_000);
}

// ── LOOP-9b: exhaustion behaviour ────────────────────────────────────────────

#[test]
fn limit_behavior_defaults_to_error() {
    assert_eq!(RunLimits::default().behavior, LimitBehavior::Error);
    assert_eq!(LimitBehavior::Error.as_str(), "error");
    assert_eq!(
        LimitBehavior::StopWithPartial.as_str(),
        "stop_with_partial"
    );
}

#[test]
fn error_behavior_preserves_the_historical_hard_failure() {
    let mut tracker = LimitTracker::new(RunLimits::default().with_max_model_calls(1));
    assert_eq!(
        tracker.try_record_model_call().unwrap(),
        LimitOutcome::Proceed
    );
    let err = tracker.try_record_model_call().unwrap_err();
    assert!(matches!(err, TinyAgentsError::Validation(_)), "got {err:?}");
    assert!(err.to_string().contains("max model calls (1) exceeded"));
}

#[test]
fn stop_with_partial_reports_a_clean_stop_instead_of_discarding_the_run() {
    // Regression test (LOOP-9b): every cap used to raise `LimitExceeded`,
    // throwing away the whole run and all the work already done. LangChain's
    // `ModelCallLimitMiddleware` returns a jump-to-end instead.
    let mut tracker = LimitTracker::new(
        RunLimits::default()
            .with_max_model_calls(1)
            .with_max_tool_calls(2)
            .with_behavior(LimitBehavior::StopWithPartial),
    );

    assert_eq!(
        tracker.try_record_model_call().unwrap(),
        LimitOutcome::Proceed
    );
    assert_eq!(
        tracker
            .try_record_model_call()
            .expect("stop_with_partial must not error"),
        LimitOutcome::Stop(LimitKind::ModelCalls)
    );

    tracker.try_record_tool_call().unwrap();
    tracker.try_record_tool_call().unwrap();
    assert_eq!(
        tracker.try_record_tool_call().unwrap(),
        LimitOutcome::Stop(LimitKind::ToolCalls)
    );
}

#[test]
fn rollback_tool_calls_uncounts_calls_that_never_ran() {
    // LangChain's `ToolCallLimitMiddleware` rolls the thread count back for
    // every remaining call it answered with a "stopped before this could run"
    // message rather than executing.
    let mut tracker = LimitTracker::new(
        RunLimits::default().with_behavior(LimitBehavior::StopWithPartial),
    );
    for _ in 0..5 {
        tracker.try_record_tool_call().unwrap();
    }
    tracker.rollback_tool_calls(3);
    assert_eq!(tracker.tool_calls(), 2);

    // Saturates rather than wrapping.
    tracker.rollback_tool_calls(99);
    assert_eq!(tracker.tool_calls(), 0);
}

#[test]
fn limit_kind_labels_match_the_event_layer() {
    // The limits module keeps its own `LimitKind` so it need not depend on the
    // observability layer; the labels must not drift apart.
    assert_eq!(
        LimitKind::ModelCalls.as_str(),
        crate::harness::events::LimitKind::ModelCalls.as_str()
    );
    assert_eq!(
        LimitKind::ToolCalls.as_str(),
        crate::harness::events::LimitKind::ToolCalls.as_str()
    );
}
