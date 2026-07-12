//! Feature tests for blueprint diffing
//! ([`tinyagents::language::diff::blueprint_diff`]).
//!
//! The existing suite covers additions and identity (`graph_id`/`start`) plus a
//! channel-reducer change. These add the under-covered halves: node/channel/edge
//! *removals*, every graph-level policy change (`defaults`, `input`, `output`,
//! `checkpoint`, `interrupt`, and `joins`), a broad set of node field changes,
//! and the rendered/serialized forms of a removal-heavy diff.

use tinyagents::blueprint_diff;
use tinyagents::language::testkit::blueprint;

const OLD: &str = r#"
graph flow {
  start a
  checkpoint always
  interrupt manual
  input { question string }
  output { answer string }
  defaults { mode "fast" }
  channel messages append
  channel gone overwrite
  node a { next b }
  node b { next END }
  node removed_node { next END }
  join [a] -> b
}
"#;

const NEW: &str = r#"
graph flow {
  start a
  interrupt auto
  input { question string extra string }
  defaults { mode "slow" }
  channel messages append
  node a { next b }
  node b { next END }
}
"#;

#[test]
fn graph_level_policy_changes_are_all_reported() {
    let diff = blueprint_diff(&blueprint(OLD), &blueprint(NEW));

    assert!(diff.checkpoint_changed.is_some(), "{diff:#?}");
    assert!(diff.interrupt_changed.is_some(), "{diff:#?}");
    assert!(diff.input_changed.is_some(), "{diff:#?}");
    assert!(diff.output_changed.is_some(), "{diff:#?}");
    assert!(diff.defaults_changed.is_some(), "{diff:#?}");
    assert!(diff.joins_changed.is_some(), "{diff:#?}");

    let (old_interrupt, new_interrupt) = diff.interrupt_changed.clone().unwrap();
    assert_eq!(old_interrupt, "manual");
    assert_eq!(new_interrupt, "auto");
}

#[test]
fn node_channel_and_edge_removals_are_reported() {
    let diff = blueprint_diff(&blueprint(OLD), &blueprint(NEW));

    assert!(diff.nodes_removed.contains(&"removed_node".to_string()));
    assert!(diff.channels_removed.contains(&"gone".to_string()));
    assert!(diff.nodes_added.is_empty());
    assert!(diff.channels_added.is_empty());
}

#[test]
fn a_removal_heavy_diff_renders_and_round_trips_through_serde() {
    let diff = blueprint_diff(&blueprint(OLD), &blueprint(NEW));
    assert!(!diff.is_empty());

    let rendered = diff.to_string();
    assert!(rendered.contains("- node removed_node"), "{rendered}");
    assert!(rendered.contains("- channel gone"), "{rendered}");
    assert!(rendered.contains("~ checkpoint"), "{rendered}");
    assert!(
        rendered.contains("~ interrupt: manual -> auto"),
        "{rendered}"
    );

    let json = serde_json::to_string(&diff).expect("serializes");
    let round_trip: tinyagents::BlueprintDiff = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(round_trip, diff);
}

#[test]
fn node_field_changes_are_reported_per_field() {
    let old = blueprint(
        r#"graph g { start a node a { kind model model "m1" prompt "p1" tools ["t1"] timeout "5s" metadata { owner "x" } next b } node b { next END } }"#,
    );
    let new = blueprint(
        r#"graph g { start a node a { kind agent model "m2" prompt "p2" tools ["t1", "t2"] timeout "10s" metadata { owner "y" } routes { ok -> b } } node b { next END } }"#,
    );

    let diff = blueprint_diff(&old, &new);
    let node = diff
        .nodes_changed
        .iter()
        .find(|n| n.name == "a")
        .expect("node `a` changed");
    let changed_fields: Vec<&str> = node.fields.iter().map(|f| f.field.as_str()).collect();

    for field in [
        "kind", "model", "prompt", "tools", "routing", "timeout", "metadata",
    ] {
        assert!(
            changed_fields.contains(&field),
            "missing {field}: {changed_fields:?}"
        );
    }

    let kind_change = node.fields.iter().find(|f| f.field == "kind").unwrap();
    assert_eq!(kind_change.old, "model");
    assert_eq!(kind_change.new, "agent");
}

#[test]
fn an_edge_removal_is_reported_and_rendered() {
    let old = blueprint("graph g { start a node a {} node b {} a -> b }");
    let new = blueprint("graph g { start a node a { next END } node b {} }");

    let diff = blueprint_diff(&old, &new);
    assert!(
        diff.edges_removed
            .iter()
            .any(|e| e.from == "a" && e.to == "b"),
        "{diff:#?}"
    );
    assert!(diff.to_string().contains("- edge a -> b"), "{diff}");
}

#[test]
fn an_identical_blueprint_diffs_to_nothing() {
    let bp = blueprint(NEW);
    let diff = blueprint_diff(&bp, &bp);
    assert!(diff.is_empty());
    assert_eq!(diff.to_string(), "no changes");
}
