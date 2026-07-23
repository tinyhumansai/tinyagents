use std::time::Duration;

use super::{ToolTimeout, ToolTimeoutSettings};

#[test]
fn inherited_timeout_updates_across_clones() {
    let settings = ToolTimeoutSettings::new(2_000, 1_000, 10_000, 250);
    let clone = settings.clone();

    assert_eq!(
        clone.resolve(ToolTimeout::Inherit).deadline,
        Some(Duration::from_millis(2_000))
    );
    assert_eq!(settings.set_inherited_timeout_ms(4_000), 4_000);
    assert_eq!(
        clone.resolve(ToolTimeout::Inherit).deadline,
        Some(Duration::from_millis(4_000))
    );
}

#[test]
fn zero_disables_only_the_inherited_deadline() {
    let settings = ToolTimeoutSettings::new(0, 1_000, 10_000, 250);

    assert_eq!(settings.inherited_timeout(), None);
    assert_eq!(settings.resolve(ToolTimeout::Inherit).deadline, None);
    assert_eq!(settings.resolve(ToolTimeout::Unbounded).deadline, None);
    assert_eq!(
        settings.resolve(ToolTimeout::Millis(2_000)).deadline,
        Some(Duration::from_millis(2_250))
    );
}

#[test]
fn explicit_timeout_clamps_and_reports_unpadded_budget() {
    let settings = ToolTimeoutSettings::new(5_000, 1_000, 10_000, 250);

    let below = settings.resolve(ToolTimeout::Millis(1));
    assert_eq!(below.budget_ms, 1_000);
    assert_eq!(below.deadline, Some(Duration::from_millis(1_250)));

    let above = settings.resolve(ToolTimeout::Millis(99_000));
    assert_eq!(above.budget_ms, 10_000);
    assert_eq!(above.deadline, Some(Duration::from_millis(10_250)));
}

#[test]
fn invalid_bounds_are_normalized() {
    let settings = ToolTimeoutSettings::new(9, 0, 0, 0);
    let resolved = settings.resolve(ToolTimeout::Inherit);
    assert_eq!(resolved.budget_ms, 1);
    assert_eq!(resolved.deadline, Some(Duration::from_millis(1)));
}
