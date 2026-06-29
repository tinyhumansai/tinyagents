//! Unit tests for run-scoped limit enforcement.
//!
//! Smoke-checks that default [`RunLimits`] build a [`LimitTracker`] and that
//! recording a model call advances the counter.

#[test]
fn smoke_default_limits_compile() {
    use super::{LimitTracker, RunLimits};
    let limits = RunLimits::default();
    let mut tracker = LimitTracker::new(limits);
    tracker.record_model_call().unwrap();
    assert_eq!(tracker.model_calls(), 1);
}
