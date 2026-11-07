use super::*;
use tinytools::ToolSpec;

fn spec(name: &str, description: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: description.into(),
        parameters: serde_json::json!({"type":"object"}),
    }
}

#[test]
fn snapshot_deduplicates_equal_declarations_and_sorts_names() {
    let snapshot = ToolSnapshot::new(vec![spec("z", "z"), spec("a", "a"), spec("a", "a")]).unwrap();
    assert_eq!(
        snapshot
            .specs()
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "z"]
    );
}

#[test]
fn snapshot_rejects_conflicting_declarations() {
    assert_eq!(
        ToolSnapshot::new(vec![spec("read", "first"), spec("read", "second")]).unwrap_err(),
        ToolSnapshotError::NameCollision("read".into())
    );
}

#[test]
fn snapshot_json_round_trip_preserves_exact_flag_semantics() {
    let snapshot = ToolSnapshot::new(vec![spec("read", "read")]).unwrap();
    let restored = ToolSnapshot::from_json(&snapshot.to_json()).unwrap();
    assert_eq!(restored.to_json(), snapshot.to_json());
    assert!(!restored.is_exact());
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HostAlpha(u8);
#[derive(Clone, Debug, PartialEq, Eq)]
struct HostBeta(&'static str);

#[test]
fn typed_extensions_survive_snapshot_copies_without_changing_json() {
    let original_json = serde_json::json!([]);
    let snapshot = ToolSnapshot::from_json(&original_json)
        .unwrap()
        .with_extension(HostAlpha(7))
        .with_extension(HostBeta("host-owned"));
    assert_eq!(snapshot.extension::<HostAlpha>(), Some(&HostAlpha(7)));
    assert_eq!(
        snapshot.extension::<HostBeta>(),
        Some(&HostBeta("host-owned"))
    );
    assert_eq!(snapshot.to_json(), original_json);

    let clone = snapshot.clone();
    let narrowed = snapshot.retaining(|_| true);
    let (retained, count) = snapshot.with_retained(&ToolSnapshot::default()).unwrap();
    assert_eq!(count, 0);
    for copy in [&clone, &narrowed, &retained] {
        assert_eq!(copy.extension::<HostAlpha>(), Some(&HostAlpha(7)));
        assert_eq!(copy.extension::<HostBeta>(), Some(&HostBeta("host-owned")));
        assert_eq!(copy.to_json(), original_json);
    }
    let replaced = clone.with_extension(HostAlpha(8));
    assert_eq!(replaced.extension::<HostAlpha>(), Some(&HostAlpha(8)));
    assert_eq!(snapshot.extension::<HostAlpha>(), Some(&HostAlpha(7)));
}

#[test]
fn exact_snapshot_never_retains_recorded_declarations() {
    let exact = ToolSnapshot::new(vec![spec("live", "live")])
        .unwrap()
        .exact();
    let recorded = ToolSnapshot::new(vec![spec("recorded", "recorded")]).unwrap();
    let (merged, count) = exact.with_retained(&recorded).unwrap();
    assert_eq!(count, 0);
    assert!(merged.is_exact());
    assert_eq!(
        merged
            .specs()
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["live"]
    );
}

fn names(snapshot: &ToolSnapshot) -> Vec<&str> {
    snapshot
        .specs()
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[test]
fn live_retention_keeps_only_recorded_declarations_the_host_still_offers() {
    let live = ToolSnapshot::new(vec![spec("live", "new description")])
        .unwrap()
        .with_offered_names(["deferred"]);
    let recorded = ToolSnapshot::new(vec![
        spec("live", "old description"),
        spec("deferred", "deferred"),
        spec("retired", "retired"),
    ])
    .unwrap();

    let (merged, count) = live.with_retained_offered(&recorded).unwrap();

    assert_eq!(count, 1);
    assert_eq!(names(&merged), ["deferred", "live"]);
    let live_spec = merged.specs().iter().find(|s| s.name == "live").unwrap();
    assert_eq!(live_spec.description, "new description");
}

#[test]
fn live_retention_drops_everything_when_nothing_is_offered() {
    let live = ToolSnapshot::new(vec![spec("live", "live")]).unwrap();
    let recorded = ToolSnapshot::new(vec![spec("retired", "retired")]).unwrap();

    let (merged, count) = live.with_retained_offered(&recorded).unwrap();

    assert_eq!(count, 0);
    assert_eq!(names(&merged), ["live"]);
}

#[test]
fn exact_snapshot_ignores_live_retention() {
    let exact = ToolSnapshot::new(vec![spec("live", "live")])
        .unwrap()
        .with_offered_names(["recorded"])
        .exact();
    let recorded = ToolSnapshot::new(vec![spec("recorded", "recorded")]).unwrap();

    let (merged, count) = exact.with_retained_offered(&recorded).unwrap();

    assert_eq!(count, 0);
    assert_eq!(names(&merged), ["live"]);
}

#[test]
fn offered_names_survive_copies_and_stay_out_of_the_json() {
    let snapshot = ToolSnapshot::new(vec![spec("live", "live")])
        .unwrap()
        .with_offered_names(["deferred"]);
    let narrowed = snapshot.retaining(|_| true);

    for copy in [&snapshot.clone(), &narrowed] {
        assert!(copy.offers("deferred"));
        assert!(copy.offers("live"));
        assert!(!copy.offers("retired"));
    }
    assert_eq!(snapshot.to_json().as_array().unwrap().len(), 1);
}
