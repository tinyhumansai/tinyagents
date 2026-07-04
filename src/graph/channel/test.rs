//! Unit tests for the channel-per-field state model: each channel's merge rule,
//! `ChannelSet` dispatch/snapshot, `ChannelState` round-trips through a real
//! graph (fan-out into aggregate channels), and concurrent-write conflict
//! detection on non-aggregate channels.

use super::*;
use crate::TinyAgentsError;
use crate::graph::builder::{GraphBuilder, NodeContext};
use crate::graph::command::{Command, NodeResult};
use serde_json::{Value, json};

// --- per-channel merge rules ---

#[test]
fn last_value_overwrites() {
    let c = LastValue;
    assert_eq!(c.merge(None, json!(1)).unwrap(), json!(1));
    assert_eq!(c.merge(Some(&json!(1)), json!(2)).unwrap(), json!(2));
    assert!(!c.allows_concurrent());
}

#[test]
fn topic_appends_scalars_and_arrays() {
    let c = Topic;
    let v = c.merge(None, json!("a")).unwrap();
    let v = c.merge(Some(&v), json!("b")).unwrap();
    let v = c.merge(Some(&v), json!(["c", "d"])).unwrap();
    assert_eq!(v, json!(["a", "b", "c", "d"]));
    assert!(c.allows_concurrent());
}

#[test]
fn delta_accumulates_numbers() {
    let c = Delta;
    let v = c.merge(None, json!(2)).unwrap();
    let v = c.merge(Some(&v), json!(3)).unwrap();
    assert_eq!(v, json!(5));
    // integer + float promotes to float
    let v = c.merge(Some(&v), json!(0.5)).unwrap();
    assert_eq!(v, json!(5.5));
    assert!(c.merge(Some(&json!(1)), json!("x")).is_err());
}

#[test]
fn messages_merge_by_id() {
    let c = Messages;
    let v = c.merge(None, json!([{"id": "1", "text": "hi"}])).unwrap();
    let v = c
        .merge(Some(&v), json!([{"id": "2", "text": "yo"}]))
        .unwrap();
    // upsert existing id 1
    let v = c
        .merge(Some(&v), json!({"id": "1", "text": "hello"}))
        .unwrap();
    assert_eq!(
        v,
        json!([{"id": "1", "text": "hello"}, {"id": "2", "text": "yo"}])
    );
    assert!(c.allows_concurrent());
}

#[test]
fn messages_merge_dedup_map_preserves_order_and_appends_unkeyed() {
    // Exercise the id->index map path: a single batch that upserts an existing
    // id, appends a new id, and appends an unkeyed message, then a follow-up
    // upsert of the id introduced by that batch. Existing order is preserved and
    // dedup is by id only.
    let c = Messages;
    let v = c
        .merge(
            None,
            json!([{"id": "a", "text": "1"}, {"id": "b", "text": "2"}]),
        )
        .unwrap();
    let v = c
        .merge(
            Some(&v),
            json!([
                {"id": "a", "text": "1-updated"},
                {"id": "c", "text": "3"},
                {"text": "no-id"},
            ]),
        )
        .unwrap();
    // Upserting an id first seen in the previous batch must land on that message.
    let v = c
        .merge(Some(&v), json!({"id": "c", "text": "3-updated"}))
        .unwrap();
    assert_eq!(
        v,
        json!([
            {"id": "a", "text": "1-updated"},
            {"id": "b", "text": "2"},
            {"id": "c", "text": "3-updated"},
            {"text": "no-id"},
        ])
    );
}

#[test]
fn ephemeral_overwrites_and_is_marked() {
    let c = Ephemeral;
    assert_eq!(c.merge(Some(&json!(1)), json!(2)).unwrap(), json!(2));
    assert!(c.is_ephemeral());
    assert!(!c.allows_concurrent());
}

#[test]
fn untracked_is_not_tracked() {
    let c = Untracked;
    assert!(!c.is_tracked());
    assert_eq!(c.merge(None, json!("x")).unwrap(), json!("x"));
}

#[test]
fn binary_aggregate_folds_with_closure() {
    let c = BinaryAggregate::new(|a: Value, b: Value| {
        Ok(json!(a.as_i64().unwrap() * b.as_i64().unwrap()))
    });
    let v = c.merge(None, json!(2)).unwrap();
    let v = c.merge(Some(&v), json!(3)).unwrap();
    let v = c.merge(Some(&v), json!(4)).unwrap();
    assert_eq!(v, json!(24));
    assert!(c.allows_concurrent());
}

#[test]
fn binary_aggregate_from_reducer() {
    // Any `Reducer<Value>` can back an aggregate channel; here a closure reducer
    // that keeps the numeric maximum.
    let c =
        BinaryAggregate::from_reducer(crate::graph::ClosureReducer::new(|a: Value, b: Value| {
            Ok(if b.as_f64() > a.as_f64() { b } else { a })
        }));
    let v = c.merge(None, json!(3)).unwrap();
    let v = c.merge(Some(&v), json!(7)).unwrap();
    let v = c.merge(Some(&v), json!(1)).unwrap();
    assert_eq!(v, json!(7));
}

#[test]
fn count_barrier_readiness() {
    let c = Barrier::new(2);
    let v = c.merge(None, json!("a")).unwrap();
    assert!(!c.is_ready(Some(&v)));
    let v = c.merge(Some(&v), json!("b")).unwrap();
    assert!(c.is_ready(Some(&v)));
}

#[test]
fn named_barrier_readiness() {
    let c = NamedBarrier::new(["left", "right"]);
    let v = c.merge(None, json!({"left": 1})).unwrap();
    assert!(!c.is_ready(Some(&v)));
    let v = c.merge(Some(&v), json!({"right": 2})).unwrap();
    assert!(c.is_ready(Some(&v)));
    assert!(c.allows_concurrent());
}

// --- ChannelSet ---

#[test]
fn channel_set_apply_and_snapshot_excludes_untracked() {
    let mut set = ChannelSet::new()
        .with_channel("count", Delta)
        .with_channel("scratch", Untracked);
    set.apply_update("count", json!(2)).unwrap();
    set.apply_update("count", json!(5)).unwrap();
    set.apply_update("scratch", json!("temp")).unwrap();

    assert_eq!(set.get("count"), Some(&json!(7)));
    let snap = set.snapshot();
    assert_eq!(snap.get("count"), Some(&json!(7)));
    assert!(
        !snap.contains_key("scratch"),
        "untracked excluded from snapshot"
    );
}

#[test]
fn channel_set_unknown_channel_errors() {
    let mut set = ChannelSet::new();
    assert!(matches!(
        set.apply_update("nope", json!(1)),
        Err(TinyAgentsError::Graph(_))
    ));
}

// --- ChannelState graph round-trips ---

fn aggregate_state() -> ChannelState {
    ChannelState::new()
        .with_channel("results", Topic)
        .with_channel("total", Delta)
}

/// `start` fans out to `a` and `b` in one parallel superstep; both write the
/// shared aggregate channels and their writes merge deterministically.
#[tokio::test]
async fn fan_out_merges_both_branches_into_aggregate_channels() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .with_parallel(true)
        .add_node("start", |_s: ChannelState, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["a", "b"])))
        })
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new()
                    .set("results", "a")
                    .set("total", 10)
                    .at_step(c.step),
            ))
        })
        .add_node("b", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new()
                    .set("results", "b")
                    .set("total", 5)
                    .at_step(c.step),
            ))
        })
        .mark_command_routing("start")
        .set_entry("start")
        .set_finish("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let exec = graph.run(aggregate_state()).await.unwrap();
    // Topic merges both branches in active-set index order; Delta sums them.
    assert_eq!(exec.state.get("results"), Some(&json!(["a", "b"])));
    assert_eq!(exec.state.get("total"), Some(&json!(15)));
}

/// Two concurrent branches writing the same `LastValue` channel in one step is
/// a conflict.
#[tokio::test]
async fn concurrent_last_value_writes_conflict() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .with_parallel(true)
        .add_node("start", |_s: ChannelState, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["a", "b"])))
        })
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("winner", "a").at_step(c.step),
            ))
        })
        .add_node("b", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("winner", "b").at_step(c.step),
            ))
        })
        .mark_command_routing("start")
        .set_entry("start")
        .set_finish("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let initial = ChannelState::new().with_channel("winner", LastValue);
    let err = graph.run(initial).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::InvalidConcurrentUpdate(_)));
}

/// A `LastValue` channel overwritten across *different* steps is fine — only
/// same-step concurrent writes conflict.
#[tokio::test]
async fn sequential_last_value_overwrites_across_steps() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("v", "first").at_step(c.step),
            ))
        })
        .add_node("b", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("v", "second").at_step(c.step),
            ))
        })
        .add_edge("a", "b")
        .set_entry("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let initial = ChannelState::new().with_channel("v", LastValue);
    let exec = graph.run(initial).await.unwrap();
    assert_eq!(exec.state.get("v"), Some(&json!("second")));
}

/// An `Ephemeral` channel written in one step is visible to the next step's
/// node, then cleared when the step advances.
#[tokio::test]
async fn ephemeral_channel_is_cleared_on_next_step() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("tmp", 1).at_step(c.step),
            ))
        })
        .add_node("b", |s: ChannelState, c: NodeContext| async move {
            // tmp is still visible to this (next-step) node.
            let seen = s.get("tmp").cloned().unwrap_or(json!(null));
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("saw", seen).at_step(c.step),
            ))
        })
        .add_edge("a", "b")
        .set_entry("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let initial = ChannelState::new()
        .with_channel("tmp", Ephemeral)
        .with_channel("saw", LastValue);
    let exec = graph.run(initial).await.unwrap();
    assert_eq!(exec.state.get("saw"), Some(&json!(1)), "tmp was visible");
    assert_eq!(
        exec.state.get("tmp"),
        None,
        "tmp cleared after step advanced"
    );
    // Snapshot reflects only the tracked, surviving channels.
    let snap = exec.state.snapshot();
    assert_eq!(snap.get("saw"), Some(&json!(1)));
    assert!(!snap.contains_key("tmp"));
}

/// A single node writing the same non-aggregate channel twice in one update is
/// last-wins, not a conflict.
#[test]
fn single_update_repeat_write_is_last_wins() {
    let state = ChannelState::new().with_channel("v", LastValue);
    let merged = state
        .merge(ChannelUpdate::new().set("v", 1).set("v", 2).at_step(1))
        .unwrap();
    assert_eq!(merged.get("v"), Some(&json!(2)));
}
