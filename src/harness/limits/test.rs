//! Tests added in a later pass.

#[test]
fn smoke_default_limits_compile() {
    use super::{LimitTracker, RunLimits};
    let limits = RunLimits::default();
    let mut tracker = LimitTracker::new(limits);
    tracker.record_model_call().unwrap();
    assert_eq!(tracker.model_calls(), 1);
}
