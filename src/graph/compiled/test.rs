//! Unit tests for the superstep executor: sequential and parallel runs,
//! reducer fan-in ordering, conditional/command routing, checkpoint
//! persistence, interrupt/resume, and recursion-limit enforcement.

use super::*;
use crate::graph::builder::{GraphBuilder, GraphDefaults, NodeContext, Route};
use crate::graph::checkpoint::{Checkpointer, InMemoryCheckpointer};
use crate::graph::command::{Command, Interrupt, NodeResult, Send};
use crate::graph::reducer::ClosureStateReducer;
use crate::graph::stream::{CollectingSink, GraphEvent};
use crate::harness::ids::ExecutionStatus;
use crate::harness::retry::RetryPolicy;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
struct Counter {
    value: i32,
    log: Vec<String>,
}

/// Builds a graph whose nodes return partial `i32` updates merged by a custom
/// reducer that adds to `value` and records a log entry.
fn adding_graph() -> CompiledGraph<Counter, i32> {
    GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            s.log.push(format!("+{u}"));
            Ok(s)
        }))
        .add_node("inc", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("double", |s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(s.value))
        })
        .set_entry("inc")
        .add_edge("inc", "double")
        .set_finish("double")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn partial_updates_and_reducer() {
    let graph = adding_graph();
    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();
    // inc -> value=1 ; double -> +1 (value snapshot) -> value=2
    assert_eq!(run.state.value, 2);
    assert_eq!(run.state.log, vec!["+1", "+1"]);
    assert_eq!(run.steps, 2);
    assert_eq!(
        run.visited
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>(),
        vec!["inc", "double"]
    );
    assert_eq!(run.status.status, ExecutionStatus::Completed);
    assert!(!run.is_interrupted());
}

#[tokio::test]
async fn conditional_routing_selects_branch() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("start", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(0))
        })
        .add_node("even", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(100))
        })
        .add_node("odd", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(200))
        })
        .set_entry("start")
        .add_conditional_edges(
            "start",
            |s: &i32| {
                if *s % 2 == 0 {
                    "even".to_string()
                } else {
                    "odd".to_string()
                }
            },
            [("even", "even"), ("odd", "odd")],
        )
        .set_finish("even")
        .set_finish("odd")
        .compile()
        .unwrap();

    let run = graph.run(0).await.unwrap();
    assert_eq!(run.state, 100);
}

#[tokio::test]
async fn command_goto_overrides_edges() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("router", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::update(5).with_goto(["target"]),
            ))
        })
        .add_node("target", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("router")
        .mark_command_routing("router")
        .set_finish("target")
        .compile()
        .unwrap();

    let run = graph.run(0).await.unwrap();
    assert_eq!(run.state, 6);
}

#[tokio::test]
async fn command_goto_rejects_unknown_target_immediately() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("router", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["missing"])))
        })
        .set_entry("router")
        .mark_command_routing("router")
        .compile()
        .unwrap();

    let err = graph.run(0).await.unwrap_err();
    match err {
        TinyAgentsError::MissingNode(node) => assert_eq!(node, "missing"),
        other => panic!("expected MissingNode, got {other:?}"),
    }
}

#[tokio::test]
async fn command_goto_rejects_start_target() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("router", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["__start__"])))
        })
        .set_entry("router")
        .mark_command_routing("router")
        .compile()
        .unwrap();

    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Graph(_)), "got {err:?}");
    assert!(err.to_string().contains("START"), "{err}");
}

#[tokio::test]
async fn invalid_command_goto_is_not_persisted_as_next_node() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("router", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::update(1).with_goto(["missing"]),
            ))
        })
        .set_entry("router")
        .mark_command_routing("router")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let err = graph.run_with_thread("bad-goto", 0).await.unwrap_err();
    assert!(
        matches!(err, TinyAgentsError::MissingNode(_)),
        "got {err:?}"
    );
    assert_eq!(
        cp.count("bad-goto"),
        0,
        "invalid runtime route must fail before boundary checkpoint persistence"
    );
}

#[tokio::test]
async fn recursion_limit_is_deterministic() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .with_recursion_limit(3)
        .add_node("loop", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("loop")
        .add_edge("loop", "loop")
        .compile()
        .unwrap();

    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::RecursionLimit(3)));
}

#[tokio::test]
async fn superstep_count_matches_path_length() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("c", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_finish("c")
        .compile()
        .unwrap();
    let run = graph.run(0).await.unwrap();
    assert_eq!(run.steps, 3);
    assert_eq!(run.state, 3);
}

#[tokio::test]
async fn checkpoints_persist_at_boundaries() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let run = graph.run_with_thread("t1", 0).await.unwrap();
    assert_eq!(run.state, 2);
    assert!(run.checkpoint_id.is_some());

    // one checkpoint per superstep boundary
    let list = cp.list("t1").await.unwrap();
    assert_eq!(list.len(), 2);
    // lineage is chained
    assert!(list[0].parent_checkpoint_id.is_none());
    assert_eq!(
        list[1].parent_checkpoint_id.as_deref(),
        Some(list[0].checkpoint_id.as_str())
    );
}

#[tokio::test]
async fn exit_durability_persists_only_terminal_checkpoint() {
    use crate::graph::checkpoint::DurabilityMode;

    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone())
        .with_durability(DurabilityMode::Exit);

    let run = graph.run_with_thread("t1", 0).await.unwrap();
    assert_eq!(run.state, 2);
    // Only the terminal boundary is persisted under Exit durability.
    assert_eq!(cp.count("t1"), 1);
    assert!(run.checkpoint_id.is_some());
    let list = cp.list("t1").await.unwrap();
    assert_eq!(list.len(), 1);
    // The single record is the terminal boundary: no pending next nodes.
    assert!(list[0].next_nodes.is_empty());
}

#[tokio::test]
async fn interrupt_then_resume_reruns_node() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("approve", |s, ctx: NodeContext| async move {
            match ctx.resume {
                // resumed: apply the approved increment
                Some(value) => {
                    let bump = value.get("bump").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                    Ok(NodeResult::Update(s + bump))
                }
                // first run: pause for human approval
                None => Ok(NodeResult::Interrupt(Interrupt::new(
                    "approve",
                    json!({ "ask": "approve?" }),
                ))),
            }
        })
        .add_node("done", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("approve")
        .add_edge("approve", "done")
        .set_finish("done")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    // first run pauses
    let paused = graph.run_with_thread("hitl", 10).await.unwrap();
    assert!(paused.is_interrupted());
    assert_eq!(paused.status.status, ExecutionStatus::Interrupted);
    assert_eq!(paused.interrupts.len(), 1);

    // resume re-runs the interrupted node with the resume value
    let resumed = graph
        .resume("hitl", Command::resume(json!({ "bump": 5 })))
        .await
        .unwrap();
    assert!(!resumed.is_interrupted());
    assert_eq!(resumed.state, 15);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
}

#[tokio::test]
async fn resume_emits_restore_not_save_for_the_loaded_checkpoint() {
    // Resuming loads a checkpoint; that read must surface as CheckpointRestored,
    // never CheckpointSaved (which would inflate persisted-checkpoint counts).
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let sink = Arc::new(CollectingSink::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("approve", |s, ctx: NodeContext| async move {
            match ctx.resume {
                Some(_) => Ok(NodeResult::Update(s + 1)),
                None => Ok(NodeResult::Interrupt(Interrupt::new("approve", json!({})))),
            }
        })
        .set_entry("approve")
        .set_finish("approve")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone())
        .with_event_sink(sink.clone());

    let paused = graph.run_with_thread("t", 0).await.unwrap();
    let loaded = paused
        .checkpoint_id
        .clone()
        .expect("interrupt persisted a checkpoint");

    // Only inspect events emitted during the resume (the initial run genuinely
    // saved the interrupt checkpoint).
    let before = sink.events().len();
    graph
        .resume("t", Command::resume(json!(null)))
        .await
        .unwrap();
    let resume_events = sink.events();
    let resume_events = &resume_events[before..];

    assert!(
        resume_events.iter().any(|e| matches!(
            e,
            GraphEvent::CheckpointRestored { checkpoint_id } if *checkpoint_id == loaded
        )),
        "resume must emit CheckpointRestored for the loaded checkpoint"
    );
    assert!(
        !resume_events.iter().any(|e| matches!(
            e,
            GraphEvent::CheckpointSaved { checkpoint_id } if *checkpoint_id == loaded
        )),
        "loading a checkpoint on resume must not re-emit it as saved"
    );
}

#[tokio::test]
async fn resume_preserves_parent_checkpoint_lineage() {
    // A run that boundary-checkpoints, interrupts, then resumes to completion
    // must keep a single connected lineage: the first post-resume checkpoint
    // chains onto the loaded one instead of orphaning the pre-interrupt
    // history. Without it, get_state_history stops at the resume point and
    // prune deletes the ancestors it should protect.
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("start", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .add_node("approve", |s, ctx: NodeContext| async move {
            match ctx.resume {
                Some(_) => Ok(NodeResult::Update(s + 1)),
                None => Ok(NodeResult::Interrupt(Interrupt::new("approve", json!({})))),
            }
        })
        .add_node("done", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("start")
        .add_edge("start", "approve")
        .add_edge("approve", "done")
        .set_finish("done")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let paused = graph.run_with_thread("hitl", 10).await.unwrap();
    assert!(paused.is_interrupted());
    let resumed = graph
        .resume("hitl", Command::resume(json!(null)))
        .await
        .unwrap();
    assert!(!resumed.is_interrupted());

    // Four boundary checkpoints: start, approve(interrupt), approve(resumed),
    // done — all reachable through the parent lineage from the latest.
    let history = graph.get_state_history("hitl", None).await.unwrap();
    assert_eq!(
        history.len(),
        4,
        "full lineage must walk past the resume point; got steps {:?}",
        history.iter().map(|s| s.metadata.step).collect::<Vec<_>>()
    );
    // Connected chain: exactly one root, every parent present.
    let ids: std::collections::HashSet<&str> = history
        .iter()
        .map(|s| s.metadata.checkpoint_id.as_str())
        .collect();
    let roots = history
        .iter()
        .filter(|s| s.metadata.parent_checkpoint_id.is_none())
        .count();
    assert_eq!(roots, 1, "a connected lineage has exactly one root");
    for s in &history {
        if let Some(parent) = &s.metadata.parent_checkpoint_id {
            assert!(
                ids.contains(parent.as_str()),
                "parent `{parent}` must be present in the walked history"
            );
        }
    }

    // Prune protects the ancestor chain of the retained window: keeping the
    // latest still keeps the pre-interrupt checkpoints it depends on.
    cp.prune("hitl", 1).await.unwrap();
    assert_eq!(
        cp.list("hitl").await.unwrap().len(),
        4,
        "prune must retain the full ancestor chain across the resume boundary"
    );
}

#[tokio::test]
async fn interrupt_without_checkpointer_errors_instead_of_pausing() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("approve", |_s, _ctx: NodeContext| async move {
            Ok(NodeResult::Interrupt(Interrupt::new(
                "approve",
                json!({ "ask": "approve?" }),
            )))
        })
        .set_entry("approve")
        .set_finish("approve")
        .compile()
        .unwrap();

    let err = graph.run_with_thread("hitl", 10).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Resume(_)), "got {err:?}");
    assert!(err.to_string().contains("checkpointer"), "{err}");
}

#[tokio::test]
async fn interrupt_without_thread_errors_instead_of_pausing() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("approve", |_s, _ctx: NodeContext| async move {
            Ok(NodeResult::Interrupt(Interrupt::new(
                "approve",
                json!({ "ask": "approve?" }),
            )))
        })
        .set_entry("approve")
        .set_finish("approve")
        .compile()
        .unwrap()
        .with_checkpointer(Arc::new(InMemoryCheckpointer::<i32>::new()));

    let err = graph.run(10).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Resume(_)), "got {err:?}");
    assert!(err.to_string().contains("thread id"), "{err}");
}

#[tokio::test]
async fn resume_without_checkpointer_errors() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap();
    let err = graph
        .resume("t", Command::resume(json!(null)))
        .await
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Resume(_)));
}

#[tokio::test]
async fn events_are_emitted() {
    let sink = Arc::new(CollectingSink::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap()
        .with_event_sink(sink.clone());

    graph.run(1).await.unwrap();
    let events = sink.events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, GraphEvent::StepStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, GraphEvent::NodeCompleted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, GraphEvent::StepCompleted { .. }))
    );
}

// --- State inspection & time travel ----------------------------------------

/// A linear `a -> b -> c` counter graph (each node `+1`) wired to `cp`, used by
/// the inspection/time-travel tests.
fn chain_graph(cp: Arc<InMemoryCheckpointer<i32>>) -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("c", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_finish("c")
        .compile()
        .unwrap()
        .with_checkpointer(cp)
}

#[tokio::test]
async fn get_state_and_history_walk_the_lineage() {
    use crate::graph::CheckpointSource;

    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = chain_graph(cp.clone());
    graph.run_with_thread("t", 0).await.unwrap();

    // Latest snapshot is the terminal boundary: state 3, no pending nodes.
    let latest = graph.get_state("t", None).await.unwrap().unwrap();
    assert_eq!(latest.values, 3);
    assert!(latest.next_nodes.is_empty());
    assert_eq!(latest.metadata.source, CheckpointSource::Loop);

    // History is newest-first along the parent chain: 3 boundaries.
    let history = graph.get_state_history("t", None).await.unwrap();
    assert_eq!(
        history.iter().map(|s| s.values).collect::<Vec<_>>(),
        vec![3, 2, 1]
    );
    // The oldest snapshot has no parent; younger ones chain to their parent.
    assert!(history.last().unwrap().parent_config.is_none());
    assert_eq!(
        history[0].parent_config.as_ref().unwrap().checkpoint_id,
        history[1].config.checkpoint_id,
    );

    // limit caps to the most recent snapshots.
    let limited = graph.get_state_history("t", Some(2)).await.unwrap();
    assert_eq!(limited.len(), 2);
    assert_eq!(limited[0].values, 3);

    // Unknown thread / missing checkpointer behave as documented.
    assert!(graph.get_state("missing", None).await.unwrap().is_none());
}

#[tokio::test]
async fn update_state_goes_through_the_reducer() {
    use crate::graph::CheckpointSource;

    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = adding_graph().with_checkpointer(cp.clone());
    graph
        .run_with_thread(
            "t",
            Counter {
                value: 0,
                log: vec![],
            },
        )
        .await
        .unwrap();

    // Manual write: the reducer adds 10 and records a log entry (proving it is
    // not a raw overwrite).
    let config = graph.update_state("t", 10, None).await.unwrap();
    let snap = graph
        .get_state("t", Some(&config.checkpoint_id.unwrap()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(snap.values.value, 12);
    assert_eq!(snap.values.log, vec!["+1", "+1", "+10"]);
    assert_eq!(snap.metadata.source, CheckpointSource::Update);

    // Attributing to a missing node is rejected.
    let err = graph
        .update_state("t", 1, Some("nope".into()))
        .await
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::MissingNode(_)));
}

#[tokio::test]
async fn update_state_as_node_sets_successor_pending_nodes() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = chain_graph(cp.clone());
    graph.run_with_thread("t", 0).await.unwrap();

    // Attribute a write to `a`: the new checkpoint's pending nodes become a's
    // successor (`b`), so a resume continues from there.
    let config = graph.update_state("t", 5, Some("a".into())).await.unwrap();
    let snap = graph
        .get_state("t", Some(&config.checkpoint_id.unwrap()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        snap.next_nodes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>(),
        vec!["b".to_string()]
    );
}

#[tokio::test]
async fn update_state_as_command_node_is_rejected() {
    // A command node routes dynamically, so it has no static successors. Using
    // it as `as_node` would persist an empty `next_nodes` and silently render
    // the thread non-resumable; the write must be rejected instead.
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("router", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::update(5).with_goto(["target"]),
            ))
        })
        .add_node("target", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("router")
        .mark_command_routing("router")
        .set_finish("target")
        .compile()
        .unwrap()
        .with_checkpointer(cp);
    graph.run_with_thread("t", 0).await.unwrap();

    let err = graph
        .update_state("t", 1, Some("router".into()))
        .await
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Graph(_)), "got {err:?}");
    assert!(err.to_string().contains("non-resumable"), "{err}");

    // A plain node is still accepted.
    graph
        .update_state("t", 1, Some("target".into()))
        .await
        .unwrap();
}

#[tokio::test]
async fn bulk_update_state_applies_successive_updates() {
    use crate::graph::CheckpointSource;

    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = chain_graph(cp.clone());
    graph.run_with_thread("t", 0).await.unwrap();
    let before = cp.count("t");

    let last = graph
        .bulk_update_state("t", [(10, None), (100, None)])
        .await
        .unwrap();
    // Two new update checkpoints were appended.
    assert_eq!(cp.count("t"), before + 2);
    let snap = graph
        .get_state("t", Some(&last.checkpoint_id.unwrap()))
        .await
        .unwrap()
        .unwrap();
    // overwrite reducer: 3 -> 10 -> 100 (last write wins each step).
    assert_eq!(snap.values, 100);
    assert_eq!(snap.metadata.source, CheckpointSource::Update);

    // Empty bulk is rejected (no resulting config).
    let err = graph.bulk_update_state("t", []).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Checkpoint(_)));
}

#[tokio::test]
async fn fork_state_does_not_mutate_source() {
    use crate::graph::CheckpointSource;

    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = chain_graph(cp.clone());
    graph.run_with_thread("src", 0).await.unwrap();
    let src_before = cp.count("src");
    let src_latest = graph.get_state("src", None).await.unwrap().unwrap();

    let forked = graph.fork_state("src", None, "dst").await.unwrap();
    // Source thread is untouched: same count, same latest state/source.
    assert_eq!(cp.count("src"), src_before);
    let src_after = graph.get_state("src", None).await.unwrap().unwrap();
    assert_eq!(src_after.values, src_latest.values);
    assert_eq!(src_after.metadata.source, src_latest.metadata.source);

    // Target carries the forked state as a fresh root (no parent), source=fork.
    let dst = graph
        .get_state("dst", Some(&forked.checkpoint_id.unwrap()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(dst.values, src_latest.values);
    assert_eq!(dst.metadata.source, CheckpointSource::Fork);
    assert!(dst.parent_config.is_none());
}

#[tokio::test]
async fn resume_from_older_checkpoint_replays_forward() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = chain_graph(cp.clone());
    graph.run_with_thread("t", 0).await.unwrap();

    // The first boundary (after `a`) has state 1 and pending node `b`.
    let list = cp.list("t").await.unwrap();
    let after_a = &list[0];
    assert_eq!(
        after_a
            .next_nodes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>(),
        vec!["b".to_string()]
    );

    // Time-travel resume from that older checkpoint replays b -> c forward.
    let replayed = graph
        .resume_from(
            "t",
            ResumeTarget::Checkpoint(after_a.checkpoint_id.clone()),
            Command::new(),
        )
        .await
        .unwrap();
    assert!(!replayed.is_interrupted());
    assert_eq!(replayed.state, 3);

    // Resuming an unknown checkpoint id errors.
    let err = graph
        .resume_from("t", ResumeTarget::Checkpoint("nope".into()), Command::new())
        .await
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Resume(_)));
}

// --- Parallel (fan-out / fan-in) execution ---------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
struct Fan {
    /// Values contributed by branches, in reducer-application order.
    values: Vec<i32>,
    /// Fork branch indices observed by branches, in reducer-application order.
    forks: Vec<usize>,
    /// Sum a downstream join node computed over the merged `values`.
    joined_sum: Option<i32>,
}

#[derive(Clone, Debug)]
enum FanUpdate {
    Branch { value: i32, fork: usize },
    Join(i32),
}

/// Shared instrumentation proving how many branches were in flight at once.
#[derive(Clone)]
struct Concurrency {
    inflight: Arc<AtomicUsize>,
    max: Arc<AtomicUsize>,
}

impl Concurrency {
    fn new() -> Self {
        Self {
            inflight: Arc::new(AtomicUsize::new(0)),
            max: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn max_observed(&self) -> usize {
        self.max.load(AtomicOrdering::SeqCst)
    }

    async fn track<T>(&self, sleep: Duration, value: T) -> T {
        let now = self.inflight.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        self.max.fetch_max(now, AtomicOrdering::SeqCst);
        tokio::time::sleep(sleep).await;
        self.inflight.fetch_sub(1, AtomicOrdering::SeqCst);
        value
    }
}

/// Builds a fan-out/fan-in graph: `super` routes to three branches that each
/// contribute a value and their fork index; all three converge on `join`, which
/// observes the merged state. `parallel` toggles concurrent branch execution.
/// Branch sleeps are deliberately reversed (a shortest, c longest) so reducer
/// ordering cannot accidentally match completion order.
fn fanout_graph(parallel: bool, conc: Concurrency) -> CompiledGraph<Fan, FanUpdate> {
    let (c_a, c_b, c_c) = (conc.clone(), conc.clone(), conc);
    GraphBuilder::<Fan, FanUpdate>::new()
        .with_parallel(parallel)
        .set_reducer(ClosureStateReducer::new(|mut s: Fan, u: FanUpdate| {
            match u {
                FanUpdate::Branch { value, fork } => {
                    s.values.push(value);
                    s.forks.push(fork);
                }
                FanUpdate::Join(sum) => s.joined_sum = Some(sum),
            }
            Ok(s)
        }))
        .add_node("super", |_s: Fan, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::default().with_goto(["a", "b", "c"]),
            ))
        })
        .add_node("a", move |_s: Fan, c: NodeContext| {
            let conc = c_a.clone();
            let fork = c
                .fork
                .as_ref()
                .map(|f| f.branch_index)
                .unwrap_or(usize::MAX);
            async move {
                Ok(NodeResult::Update(
                    conc.track(
                        Duration::from_millis(20),
                        FanUpdate::Branch { value: 1, fork },
                    )
                    .await,
                ))
            }
        })
        .add_node("b", move |_s: Fan, c: NodeContext| {
            let conc = c_b.clone();
            let fork = c
                .fork
                .as_ref()
                .map(|f| f.branch_index)
                .unwrap_or(usize::MAX);
            async move {
                Ok(NodeResult::Update(
                    conc.track(
                        Duration::from_millis(60),
                        FanUpdate::Branch { value: 2, fork },
                    )
                    .await,
                ))
            }
        })
        .add_node("c", move |_s: Fan, c: NodeContext| {
            let conc = c_c.clone();
            let fork = c
                .fork
                .as_ref()
                .map(|f| f.branch_index)
                .unwrap_or(usize::MAX);
            async move {
                Ok(NodeResult::Update(
                    conc.track(
                        Duration::from_millis(100),
                        FanUpdate::Branch { value: 4, fork },
                    )
                    .await,
                ))
            }
        })
        .add_node("join", |s: Fan, _c: NodeContext| async move {
            Ok(NodeResult::Update(FanUpdate::Join(s.values.iter().sum())))
        })
        .set_entry("super")
        .mark_command_routing("super")
        .add_edge("a", "join")
        .add_edge("b", "join")
        .add_edge("c", "join")
        .set_finish("join")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn parallel_runs_branches_concurrently_and_merges() {
    let conc = Concurrency::new();
    let graph = fanout_graph(true, conc.clone());
    let run = graph.run(Fan::default()).await.unwrap();

    // All three branches ran at the same time.
    assert_eq!(conc.max_observed(), 3);
    // Reducer merged every branch's contribution.
    assert_eq!(run.state.values, vec![1, 2, 4]);
    // Fork indices are deterministic active-set positions, not completion order.
    assert_eq!(run.state.forks, vec![0, 1, 2]);
    // Downstream join observed the merged state.
    assert_eq!(run.state.joined_sum, Some(7));
    // super | (a,b,c) | join == 3 supersteps.
    assert_eq!(run.steps, 3);
}

#[tokio::test]
async fn sequential_mode_runs_one_branch_at_a_time() {
    let conc = Concurrency::new();
    let graph = fanout_graph(false, conc.clone());
    let run = graph.run(Fan::default()).await.unwrap();

    // Never more than one branch in flight in sequential mode.
    assert_eq!(conc.max_observed(), 1);
    // Same deterministic merge as the parallel run.
    assert_eq!(run.state.values, vec![1, 2, 4]);
    assert_eq!(run.state.joined_sum, Some(7));
    // Sequential branches get no fork identity.
    assert_eq!(run.state.forks, vec![usize::MAX, usize::MAX, usize::MAX]);
    assert_eq!(run.steps, 3);
}

#[tokio::test]
async fn parallel_merge_is_reproducible() {
    // Run the same parallel fan-out repeatedly; the merged order must be stable
    // regardless of which branch's sleep finishes first.
    for _ in 0..5 {
        let graph = fanout_graph(true, Concurrency::new());
        let run = graph.run(Fan::default()).await.unwrap();
        assert_eq!(run.state.values, vec![1, 2, 4]);
        assert_eq!(run.state.forks, vec![0, 1, 2]);
        assert_eq!(run.state.joined_sum, Some(7));
    }
}

#[tokio::test]
async fn recursion_limit_is_deterministic_in_parallel() {
    // A self-looping fan-out in parallel mode must still hit the recursion limit
    // deterministically at the configured number of supersteps.
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .with_parallel(true)
        .with_recursion_limit(3)
        .add_node("loop", |s, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::update(s + 1).with_goto(["loop"]),
            ))
        })
        .set_entry("loop")
        .mark_command_routing("loop")
        .compile()
        .unwrap();

    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::RecursionLimit(3)));
}

#[tokio::test]
async fn parallel_interrupt_pauses_at_lowest_index_branch() {
    // When a parallel branch interrupts, the step pauses; the interrupted branch
    // and every later active node become the checkpoint's pending nodes, while
    // lower-index successful branches' updates are still applied.
    let cp = Arc::new(InMemoryCheckpointer::<Fan>::new());
    let graph = GraphBuilder::<Fan, FanUpdate>::new()
        .with_parallel(true)
        .set_reducer(ClosureStateReducer::new(|mut s: Fan, u: FanUpdate| {
            if let FanUpdate::Branch { value, fork } = u {
                s.values.push(value);
                s.forks.push(fork);
            }
            Ok(s)
        }))
        .add_node("super", |_s: Fan, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::default().with_goto(["a", "b"]),
            ))
        })
        .add_node("a", |_s: Fan, _c: NodeContext| async move {
            Ok(NodeResult::Update(FanUpdate::Branch { value: 1, fork: 0 }))
        })
        .add_node("b", |_s: Fan, _c: NodeContext| async move {
            Ok(NodeResult::Interrupt(Interrupt::new("b", json!({}))))
        })
        .set_entry("super")
        .mark_command_routing("super")
        .set_finish("a")
        .set_finish("b")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let paused = graph.run_with_thread("fan", Fan::default()).await.unwrap();
    assert!(paused.is_interrupted());
    // Lower-index branch `a` committed before the pause.
    assert_eq!(paused.state.values, vec![1]);
    // The interrupting branch `b` is the head of the pending set.
    assert_eq!(
        paused.status.active_nodes.first().map(|n| n.to_string()),
        Some("b".to_string())
    );
}

#[tokio::test]
async fn parallel_interrupt_schedules_completed_branch_successors() {
    // Parallel [a, b]: a routes to successor `x` and completes; b interrupts.
    // After resume, x (a's successor) must still run — its scheduling used to be
    // dropped at the interrupt boundary, so x silently never executed.
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = GraphBuilder::<Counter, i32>::new()
        .with_parallel(true)
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            s.log.push(format!("+{u}"));
            Ok(s)
        }))
        .add_node("super", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::default().with_goto(["a", "b"]),
            ))
        })
        .add_node("a", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("b", |_s: Counter, c: NodeContext| async move {
            match c.resume {
                Some(_) => Ok(NodeResult::Update(100)),
                None => Ok(NodeResult::Interrupt(Interrupt::new("b", json!({})))),
            }
        })
        .add_node("x", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(10))
        })
        .set_entry("super")
        .mark_command_routing("super")
        .add_edge("a", "x")
        .set_finish("b")
        .set_finish("x")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let paused = graph
        .run_with_thread(
            "t",
            Counter {
                value: 0,
                log: vec![],
            },
        )
        .await
        .unwrap();
    assert!(paused.is_interrupted());
    assert_eq!(paused.state.value, 1, "branch a committed before the pause");

    let done = graph
        .resume("t", Command::resume(json!(null)))
        .await
        .unwrap();
    assert!(
        done.visited.iter().any(|n| n.as_str() == "x"),
        "a's successor x must run after resume"
    );
    // 1 (a) + 100 (b resume) + 10 (x) — every scheduled branch ran once.
    assert_eq!(done.state.value, 111);
}

#[tokio::test]
async fn send_args_survive_interrupt_and_resume() {
    // A `Send` fanout schedules three workers (args 1, 2, 3); the arg-1 worker
    // interrupts on its first activation. On resume every pending worker must
    // still carry its own send arg — before the fix they resumed with `None`.
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = GraphBuilder::<Counter, i32>::new()
        .with_parallel(true)
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            s.log.push(format!("w:{u}"));
            Ok(s)
        }))
        .add_node("dispatch", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::send([
                Send::new("worker", json!(1)),
                Send::new("worker", json!(2)),
                Send::new("worker", json!(3)),
            ])))
        })
        .add_node("worker", |_s: Counter, c: NodeContext| async move {
            let arg = c
                .send_arg
                .clone()
                .expect("worker scheduled via Send must carry its arg")
                .as_i64()
                .unwrap() as i32;
            if arg == 1 && c.resume.is_none() {
                return Ok(NodeResult::Interrupt(Interrupt::new("worker", json!({}))));
            }
            Ok(NodeResult::Update(arg))
        })
        .set_entry("dispatch")
        .mark_command_routing("dispatch")
        .set_finish("worker")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let paused = graph
        .run_with_thread(
            "fan",
            Counter {
                value: 0,
                log: vec![],
            },
        )
        .await
        .unwrap();
    assert!(paused.is_interrupted());

    // Resume: the arg-1 worker unblocks and the other two re-run with their
    // preserved args. With the arg lost, `expect(...)` above would panic.
    let done = graph
        .resume("fan", Command::resume(json!(null)))
        .await
        .unwrap();
    assert_eq!(done.state.value, 6, "all three worker args (1+2+3) applied");
    let mut log = done.state.log.clone();
    log.sort();
    assert_eq!(log, vec!["w:1", "w:2", "w:3"]);
}

#[tokio::test]
async fn barrier_arrivals_survive_interrupt_and_resume() {
    // Diamond join: p1 arrives at the barrier before an interrupt; p2 arrives
    // only after resume. The join must still fire — the p1 arrival has to
    // survive the checkpoint boundary or the join's precondition is never met.
    let cp = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let graph = GraphBuilder::<Counter, i32>::new()
        .with_parallel(true)
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            s.log.push(format!("+{u}"));
            Ok(s)
        }))
        .add_node("super", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::default().with_goto(["p1", "hold"]),
            ))
        })
        .add_node("p1", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("p2", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(2))
        })
        // `hold` interrupts first; on resume it routes to p2 (the second
        // barrier predecessor).
        .add_node("hold", |_s: Counter, c: NodeContext| async move {
            match c.resume {
                Some(_) => Ok(NodeResult::Command(Command::new().with_goto(["p2"]))),
                None => Ok(NodeResult::Interrupt(Interrupt::new("hold", json!({})))),
            }
        })
        .add_node("join", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(100))
        })
        .set_entry("super")
        .mark_command_routing("super")
        .mark_command_routing("hold")
        .add_waiting_edge("p1", "join")
        .add_waiting_edge("p2", "join")
        .set_finish("join")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let paused = graph
        .run_with_thread(
            "diamond",
            Counter {
                value: 0,
                log: vec![],
            },
        )
        .await
        .unwrap();
    assert!(paused.is_interrupted());
    assert_eq!(
        paused.state.value, 1,
        "p1 committed (arrived at the barrier)"
    );

    let done = graph
        .resume("diamond", Command::resume(json!(null)))
        .await
        .unwrap();
    assert!(
        done.visited.iter().any(|n| n.as_str() == "join"),
        "join must fire once both barrier predecessors have arrived across the resume"
    );
    // 1 (p1) + 2 (p2) + 100 (join).
    assert_eq!(done.state.value, 103);
}

#[tokio::test]
async fn barrier_relief_fires_when_source_skips_relief_node() {
    // Mixed fan-in: `m` waits on both `a` and `c`. `condition` never routes to
    // `a` — it always takes the `skip` route to END, simulating an untaken
    // conditional branch — so without a barrier relief `m` would deadlock
    // forever waiting on a predecessor that never runs.
    // `add_barrier_relief("condition", "a", "m")` registers `a`'s phantom
    // arrival at `m` whenever `condition` completes without activating `a`,
    // so `m` still fires once `c`'s real arrival lands.
    let graph = GraphBuilder::<Vec<String>, String>::new()
        .with_parallel(true)
        .set_reducer(ClosureStateReducer::new(|mut s: Vec<String>, u: String| {
            s.push(u);
            Ok(s)
        }))
        .add_node("start", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::default().with_goto(["condition", "c"]),
            ))
        })
        .add_node("condition", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update("condition".to_string()))
        })
        .add_node("a", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update("a".to_string()))
        })
        .add_node("c", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update("c".to_string()))
        })
        .add_node("m", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update("m".to_string()))
        })
        .set_entry("start")
        .mark_command_routing("start")
        // `condition` always takes the `skip` route to END — `a` is never
        // reached via a real edge.
        .add_conditional_edges(
            "condition",
            |_s: &Vec<String>| "skip".to_string(),
            [("skip", END)],
        )
        .add_waiting_edge("a", "m")
        .add_waiting_edge("c", "m")
        .add_barrier_relief("condition", "a", "m")
        .set_finish("m")
        .compile()
        .unwrap();

    let run = graph.run(Vec::new()).await.unwrap();

    assert!(
        run.visited.iter().any(|n| n.as_str() == "m"),
        "m must activate via the barrier relief even though `a` never ran"
    );
    assert!(
        !run.visited.iter().any(|n| n.as_str() == "a"),
        "a must never have run (condition always skipped it)"
    );
    assert_eq!(
        run.state,
        vec!["condition".to_string(), "c".to_string(), "m".to_string()],
        "m fires off condition+c's real contributions, with no phantom `a` update"
    );
}

#[tokio::test]
async fn reducer_error_at_boundary_transitions_run_to_failed() {
    // A reducer error raised at the step boundary (after the node ran) must
    // still fail the run — emit RunFailed / a Failed status — rather than
    // unwinding and leaving observers to see the run stuck in Running.
    let sink = Arc::new(CollectingSink::new());
    let graph = GraphBuilder::<i32, i32>::new()
        .set_reducer(ClosureStateReducer::new(|_s: i32, u: i32| {
            if u == 999 {
                Err(TinyAgentsError::Graph("reducer boom".to_string()))
            } else {
                Ok(u)
            }
        }))
        .add_node("boom", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(999))
        })
        .set_entry("boom")
        .set_finish("boom")
        .compile()
        .unwrap()
        .with_event_sink(sink.clone());

    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Graph(_)), "got {err:?}");
    assert!(
        sink.events()
            .iter()
            .any(|e| matches!(e, GraphEvent::RunFailed { .. })),
        "a boundary reducer error must transition the run to Failed (RunFailed emitted)"
    );
}

#[tokio::test]
async fn status_snapshot_reports_run() {
    let graph = adding_graph();
    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();
    let status = &run.status;
    assert_eq!(status.status, ExecutionStatus::Completed);
    assert_eq!(status.current_step, 2);
    assert!(status.ended_at.is_some());
    assert!(status.error.is_none());
    assert_eq!(status.graph_id, *graph.graph_id());
}

/// A `Send` fan-out delivers a distinct per-branch argument to N parallel
/// activations of the *same* node, and the reducer merges their results.
#[tokio::test]
async fn send_fanout_delivers_distinct_args_to_parallel_branches() {
    let graph = GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            s.log.push(format!("worker:{u}"));
            Ok(s)
        }))
        .with_parallel(true)
        // dispatch fans out three custom inputs to the same worker node.
        .add_node("dispatch", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::send([
                Send::new("worker", json!(10)),
                Send::new("worker", json!(20)),
                Send::new("worker", json!(30)),
            ])))
        })
        // each worker invocation consumes its own send arg as the update.
        .add_node("worker", |_s: Counter, c: NodeContext| async move {
            let arg = c
                .send_arg
                .expect("worker scheduled via Send carries an arg");
            let v = arg.as_i64().unwrap() as i32;
            Ok(NodeResult::Update(v))
        })
        .mark_command_routing("dispatch")
        .set_entry("dispatch")
        .set_finish("worker")
        .compile()
        .unwrap();

    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();

    // All three distinct args merged: 10 + 20 + 30.
    assert_eq!(run.state.value, 60);
    // The worker ran three times (one activation per Send packet).
    let worker_runs = run
        .visited
        .iter()
        .filter(|n| n.as_str() == "worker")
        .count();
    assert_eq!(worker_runs, 3);
    let mut log = run.state.log.clone();
    log.sort();
    assert_eq!(log, vec!["worker:10", "worker:20", "worker:30"]);
}

#[tokio::test]
async fn repeated_send_activations_keep_distinct_commands() {
    // Regression: two `Send` activations of the *same* node each return a
    // distinct `Command::goto`. A node-keyed goto map let the second clobber
    // the first, so both branches routed to the survivor's target (and one
    // sink was dropped). Keyed per activation, each keeps its own routing.
    let graph = GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            s.log.push(format!("n:{u}"));
            Ok(s)
        }))
        .with_parallel(true)
        .add_node("dispatch", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::send([
                Send::new("worker", json!(1)),
                Send::new("worker", json!(2)),
            ])))
        })
        // Each worker routes to a different sink based on its own send arg.
        .add_node("worker", |_s: Counter, c: NodeContext| async move {
            let arg = c.send_arg.expect("worker carries a send arg");
            let target = if arg.as_i64() == Some(1) {
                "sink_a"
            } else {
                "sink_b"
            };
            Ok(NodeResult::Command(Command::new().with_goto([target])))
        })
        .add_node("sink_a", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(10))
        })
        .add_node("sink_b", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(20))
        })
        .mark_command_routing("dispatch")
        .mark_command_routing("worker")
        .set_entry("dispatch")
        .set_finish("sink_a")
        .set_finish("sink_b")
        .compile()
        .unwrap();

    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();

    // Both sinks must have run — one per activation's own goto.
    assert!(
        run.visited.iter().any(|n| n.as_str() == "sink_a"),
        "sink_a (worker arg 1's target) must run"
    );
    assert!(
        run.visited.iter().any(|n| n.as_str() == "sink_b"),
        "sink_b (worker arg 2's target) must run"
    );
    assert_eq!(run.state.value, 30, "both sinks contributed (10 + 20)");
}

#[tokio::test]
async fn run_with_inputs_seeds_start_and_peer_node() {
    let graph = GraphBuilder::<Counter, String>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: String| {
            s.log.push(u);
            Ok(s)
        }))
        .with_parallel(true)
        .add_node("user_loop", |_s: Counter, c: NodeContext| async move {
            let input = c
                .send_arg
                .expect("start input should be delivered to entry node")
                .as_str()
                .expect("user payload is a string")
                .to_string();
            Ok(NodeResult::Update(format!("user:{input}")))
        })
        .add_node("tool_loop", |_s: Counter, c: NodeContext| async move {
            let tool = c
                .send_arg
                .expect("tool input should be delivered to peer node")
                .get("tool")
                .and_then(|v| v.as_str())
                .expect("tool payload names the tool")
                .to_string();
            Ok(NodeResult::Update(format!("tool:{tool}")))
        })
        .set_entry("user_loop")
        .set_finish("user_loop")
        .set_finish("tool_loop")
        .compile()
        .unwrap();

    let run = graph
        .run_with_inputs(
            Counter {
                value: 0,
                log: vec![],
            },
            [
                GraphInput::start(json!("hello")),
                GraphInput::new("tool_loop", json!({ "tool": "search" })),
            ],
        )
        .await
        .unwrap();

    assert_eq!(run.steps, 1);
    assert_eq!(run.state.log, vec!["user:hello", "tool:search"]);
    assert_eq!(
        run.visited.iter().map(|n| n.as_str()).collect::<Vec<_>>(),
        vec!["user_loop", "tool_loop"]
    );
}

#[tokio::test]
async fn run_with_inputs_preserves_repeated_inputs_to_same_node() {
    let graph = GraphBuilder::<Counter, String>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: String| {
            s.log.push(u);
            Ok(s)
        }))
        .add_node("worker", |_s: Counter, c: NodeContext| async move {
            let item = c
                .send_arg
                .expect("external input should carry an item")
                .as_i64()
                .expect("item payload is an integer");
            Ok(NodeResult::Update(format!("item:{item}")))
        })
        .set_entry("worker")
        .set_finish("worker")
        .compile()
        .unwrap();

    let run = graph
        .run_with_inputs(
            Counter {
                value: 0,
                log: vec![],
            },
            [
                GraphInput::new("worker", json!(1)),
                GraphInput::new("worker", json!(2)),
                GraphInput::new("worker", json!(3)),
            ],
        )
        .await
        .unwrap();

    assert_eq!(run.steps, 1);
    assert_eq!(run.state.log, vec!["item:1", "item:2", "item:3"]);
    assert_eq!(
        run.visited
            .iter()
            .filter(|node| node.as_str() == "worker")
            .count(),
        3
    );
}

/// A node with normal `goto` (no `send_arg`) gets `None`, while the same node
/// reached via `Send` gets the packet's argument — proving the two coexist.
#[tokio::test]
async fn goto_activation_has_no_send_arg() {
    let graph = GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            Ok(s)
        }))
        .add_node("start", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["sink"])))
        })
        .add_node("sink", |_s: Counter, c: NodeContext| async move {
            // Plain goto activation: no per-invocation argument.
            assert!(c.send_arg.is_none());
            Ok(NodeResult::Update(1))
        })
        .mark_command_routing("start")
        .set_entry("start")
        .set_finish("sink")
        .compile()
        .unwrap();
    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();
    assert_eq!(run.state.value, 1);
}

/// A user route enum with `Display` can label conditional edges directly
/// (typed routes), and the [`Route`] newtype is accepted interchangeably.
#[tokio::test]
async fn typed_enum_conditional_route() {
    #[derive(Clone, Copy)]
    enum Decision {
        Approve,
        Reject,
    }
    impl std::fmt::Display for Decision {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Decision::Approve => f.write_str("approve"),
                Decision::Reject => f.write_str("reject"),
            }
        }
    }

    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("gate", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .add_node("approved", |_s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(100))
        })
        .add_node("rejected", |_s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(-1))
        })
        .set_entry("gate")
        // Router returns the enum directly (impl ToString); the route table is
        // keyed by the enum variant and the `Route` newtype interchangeably.
        .add_conditional_edges(
            "gate",
            |s: &i32| {
                if *s > 0 {
                    Decision::Approve
                } else {
                    Decision::Reject
                }
            },
            [
                (Route::new(Decision::Approve), "approved"),
                (Route::new(Decision::Reject), "rejected"),
            ],
        )
        .set_finish("approved")
        .set_finish("rejected")
        .compile()
        .unwrap();

    assert_eq!(graph.run(5).await.unwrap().state, 100);
    assert_eq!(graph.run(-3).await.unwrap().state, -1);
}

/// A barrier/waiting node activates exactly once, only after *all* of its
/// registered predecessors have completed — even when they finish in different
/// supersteps.
#[tokio::test]
async fn waiting_edge_barrier_joins_staggered_predecessors() {
    let graph = GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            Ok(s)
        }))
        .add_node("start", |_s: Counter, _c: NodeContext| async move {
            // Fan out to a fast predecessor and a one-hop chain.
            Ok(NodeResult::Command(Command::goto(["p1", "inter"])))
        })
        .add_node("p1", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("inter", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("p2", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("join", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(10))
        })
        .mark_command_routing("start")
        .set_entry("start")
        // p1 completes in step 2; p2 only after inter (step 3). The barrier
        // holds `join` until both have arrived.
        .add_waiting_edge("p1", "join")
        .add_edge("inter", "p2")
        .add_waiting_edge("p2", "join")
        .set_finish("join")
        .compile()
        .unwrap();

    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();

    // join ran exactly once (not once per predecessor arrival).
    let join_runs = run.visited.iter().filter(|n| n.as_str() == "join").count();
    assert_eq!(join_runs, 1);
    // p1 + inter + p2 + join = 1 + 1 + 1 + 10.
    assert_eq!(run.state.value, 13);
    // join is the last node visited, proving it waited for both branches.
    assert_eq!(run.visited.last().unwrap().as_str(), "join");
}

/// `add_sequence` is sugar for a chain of direct edges.
#[tokio::test]
async fn add_sequence_chains_direct_edges() {
    let graph = GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            s.log.push(format!("+{u}"));
            Ok(s)
        }))
        .add_node("a", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("b", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("c", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .set_entry("a")
        .add_sequence(["a", "b", "c"])
        .set_finish("c")
        .compile()
        .unwrap();

    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();
    assert_eq!(run.state.value, 3);
    assert_eq!(
        run.visited
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
}

/// `with_max_concurrency` (via `set_defaults`) bounds the number of node
/// handlers in flight at once within a parallel superstep.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_concurrency_bounds_in_flight_branches() {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));

    let worker_in_flight = in_flight.clone();
    let worker_max = max_seen.clone();

    let graph = GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            Ok(s)
        }))
        .set_defaults(GraphDefaults {
            parallel: Some(true),
            max_concurrency: Some(2),
            ..Default::default()
        })
        .add_node("dispatch", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::send([
                Send::new("worker", json!(1)),
                Send::new("worker", json!(1)),
                Send::new("worker", json!(1)),
                Send::new("worker", json!(1)),
            ])))
        })
        .add_node("worker", move |_s: Counter, _c: NodeContext| {
            let in_flight = worker_in_flight.clone();
            let max_seen = worker_max.clone();
            async move {
                let now = in_flight.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                max_seen.fetch_max(now, AtomicOrdering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                in_flight.fetch_sub(1, AtomicOrdering::SeqCst);
                Ok(NodeResult::Update(1))
            }
        })
        .mark_command_routing("dispatch")
        .set_entry("dispatch")
        .set_finish("worker")
        .compile()
        .unwrap();

    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();

    // All four workers ran and contributed.
    assert_eq!(run.state.value, 4);
    // Never more than the configured bound of 2 in flight simultaneously.
    assert!(
        max_seen.load(AtomicOrdering::SeqCst) <= 2,
        "max in-flight {} exceeded bound",
        max_seen.load(AtomicOrdering::SeqCst)
    );
    // And concurrency actually happened (a chunk of 2 overlapped).
    assert_eq!(max_seen.load(AtomicOrdering::SeqCst), 2);
}

/// With `max_concurrency`, the executor uses a rolling `buffered(limit)` window
/// rather than fixed `join_all` chunks, so a slow branch does not head-of-line
/// block later branches: a new branch starts as soon as any in-flight one
/// finishes. A fixed-chunk executor would run the long branch's chunk to
/// completion before starting the next chunk, so the long branch would overlap
/// at most its single chunk-mate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_concurrency_uses_rolling_window_not_chunks() {
    // A shared flag marks the long branch as running; short branches count how
    // many of them start while the long branch is still in flight.
    let long_running = Arc::new(AtomicBool::new(false));
    let overlapped_with_long = Arc::new(AtomicUsize::new(0));

    let w_long = long_running.clone();
    let w_overlap = overlapped_with_long.clone();

    let graph = GraphBuilder::<Counter, i32>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Counter, u: i32| {
            s.value += u;
            Ok(s)
        }))
        .set_defaults(GraphDefaults {
            parallel: Some(true),
            max_concurrency: Some(2),
            ..Default::default()
        })
        .add_node("dispatch", |_s: Counter, _c: NodeContext| async move {
            // One long branch (arg 100) plus three short branches (arg 5).
            Ok(NodeResult::Command(Command::send([
                Send::new("worker", json!(100)),
                Send::new("worker", json!(5)),
                Send::new("worker", json!(5)),
                Send::new("worker", json!(5)),
            ])))
        })
        .add_node("worker", move |_s: Counter, c: NodeContext| {
            let long_running = w_long.clone();
            let overlapped = w_overlap.clone();
            async move {
                let ms = c.send_arg.and_then(|v| v.as_u64()).unwrap_or(0);
                if ms >= 50 {
                    // The long branch: flag itself running for its whole life.
                    long_running.store(true, AtomicOrdering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                    long_running.store(false, AtomicOrdering::SeqCst);
                } else {
                    // A short branch: did it get to start while the long branch
                    // was still running? Only possible with a rolling window.
                    if long_running.load(AtomicOrdering::SeqCst) {
                        overlapped.fetch_add(1, AtomicOrdering::SeqCst);
                    }
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                }
                Ok(NodeResult::Update(1))
            }
        })
        .mark_command_routing("dispatch")
        .set_entry("dispatch")
        .set_finish("worker")
        .compile()
        .unwrap();

    let run = graph
        .run(Counter {
            value: 0,
            log: vec![],
        })
        .await
        .unwrap();

    assert_eq!(run.state.value, 4, "all four workers ran");
    // With a rolling window the two short branches that start after the initial
    // pair (slot freed as each short one finishes) run while the long branch is
    // still going. A fixed-chunk executor would finish the long branch's chunk
    // first, so at most one short branch could overlap it.
    assert!(
        overlapped_with_long.load(AtomicOrdering::SeqCst) >= 2,
        "expected the rolling window to overlap the long branch with >=2 short \
         branches, saw {}",
        overlapped_with_long.load(AtomicOrdering::SeqCst)
    );
}

/// A per-node default timeout fails the run with [`TinyAgentsError::Timeout`]
/// when a handler does not resolve in time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_timeout_fails_slow_handler() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .with_node_timeout(Duration::from_millis(20))
        .add_node("slow", |s: i32, _c: NodeContext| async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(NodeResult::Update(s))
        })
        .set_entry("slow")
        .set_finish("slow")
        .compile()
        .unwrap();

    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Timeout(_)));
}

// ── Whole-run wall-clock deadline ────────────────────────────────────────────

/// A per-run deadline stops the run *between* super-steps once the elapsed run
/// time reaches it, surfacing [`TinyAgentsError::Timeout`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_deadline_stops_between_supersteps() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s: i32, _c: NodeContext| async move {
            tokio::time::sleep(Duration::from_millis(40)).await;
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap()
        .with_run_deadline(Duration::from_millis(20));

    // The first boundary (elapsed ~0) admits node `a`; the next boundary
    // (elapsed ~40ms ≥ 20ms) trips the deadline before `b` ever runs.
    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
}

/// A run that finishes within its deadline is unaffected — no false trip.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_deadline_allows_a_run_that_finishes_in_time() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap()
        .with_run_deadline(Duration::from_secs(30));

    let run = graph.run(0).await.unwrap();
    assert_eq!(run.state, 2);
}

/// On a checkpointed thread, a deadline trip leaves the last committed boundary
/// checkpoint intact — so the run can be resumed to completion rather than lost
/// (the durability win over an external `tokio::time::timeout`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_deadline_leaves_last_checkpoint_resumable() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let topology = || {
        GraphBuilder::<i32, i32>::overwrite()
            .add_node("a", |s: i32, _c: NodeContext| async move {
                tokio::time::sleep(Duration::from_millis(40)).await;
                Ok(NodeResult::Update(s + 1))
            })
            .add_node("b", |s: i32, _c: NodeContext| async move {
                Ok(NodeResult::Update(s + 1))
            })
            .add_node("c", |s: i32, _c: NodeContext| async move {
                Ok(NodeResult::Update(s + 1))
            })
            .set_entry("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish("c")
            .compile()
            .unwrap()
    };

    // Trips after `a`'s boundary (state=1, next=[b]) but before `b` runs.
    let deadlined = topology()
        .with_checkpointer(cp.clone())
        .with_run_deadline(Duration::from_millis(20));
    let err = deadlined.run_with_thread("t", 0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");

    // The boundary checkpoint from the completed super-step survived intact.
    let list = cp.list("t").await.unwrap();
    assert!(
        !list.is_empty(),
        "the pre-deadline boundary checkpoint is intact"
    );

    // Resuming (no deadline) continues from that checkpoint to completion.
    let resumed = topology().with_checkpointer(cp.clone());
    let run = resumed
        .resume("t", Command::resume(json!(null)))
        .await
        .unwrap();
    assert_eq!(run.state, 3, "resume ran the remaining super-steps b and c");
}

// ── Network resilience: node retry + resumable failures ──────────────────────

/// A single-node graph whose handler fails (with a retryable model error) the
/// first `fail_times` invocations, then succeeds with `+1`. The shared counter
/// lets a test observe how many attempts were made.
fn flaky_graph(fail_times: usize, attempts: Arc<AtomicUsize>) -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("flaky", move |s, _c: NodeContext| {
            let attempts = attempts.clone();
            async move {
                let n = attempts.fetch_add(1, AtomicOrdering::SeqCst);
                if n < fail_times {
                    Err(TinyAgentsError::Model(format!("transient blip {n}")))
                } else {
                    Ok(NodeResult::Update(s + 1))
                }
            }
        })
        .set_entry("flaky")
        .set_finish("flaky")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn node_retry_recovers_transient_failure() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let sink = Arc::new(CollectingSink::new());
    // Fail twice, succeed on the third attempt; the policy allows 1 try + 3
    // retries, so recovery is within budget.
    let graph = flaky_graph(2, attempts.clone())
        .with_node_retry(RetryPolicy::default().with_max_attempts(4))
        .with_event_sink(sink.clone());

    let run = graph.run(10).await.unwrap();
    assert_eq!(run.state, 11);
    assert_eq!(run.status.status, ExecutionStatus::Completed);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 3);

    let retries = sink
        .events()
        .into_iter()
        .filter(|e| matches!(e, GraphEvent::NodeRetryScheduled { .. }))
        .count();
    assert_eq!(retries, 2, "one retry per transient failure");
}

#[tokio::test]
async fn exhausted_retries_leave_a_resumable_failure_checkpoint() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let attempts = Arc::new(AtomicUsize::new(0));
    // Fail the first 3 invocations. With 1 try + 1 retry the first run exhausts
    // its budget (2 attempts: n=0,1) and aborts, leaving a resumable checkpoint.
    let graph = flaky_graph(3, attempts.clone())
        .with_node_retry(RetryPolicy::default().with_max_attempts(2))
        .with_checkpointer(cp.clone());

    let err = graph.run_with_thread("net", 100).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2, "1 try + 1 retry");

    // The failure boundary is durable: the checkpoint schedules the failed node
    // for re-run at the first superstep.
    let list = cp.list("net").await.unwrap();
    let last = list.last().expect("a failure checkpoint was persisted");
    assert_eq!(last.next_nodes, vec![NodeId::from("flaky")]);
    let snapshot = graph.get_state("net", None).await.unwrap().unwrap();
    assert_eq!(
        snapshot.metadata.step, 1,
        "failure boundary is at the first superstep"
    );

    // Retry: attempt n=2 fails, retry n=3 (3<3 is false) succeeds — the run
    // completes without losing the earlier progress.
    let resumed = graph.retry("net").await.unwrap();
    assert_eq!(
        resumed.state, 101,
        "resume re-runs the failed node to success"
    );
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 4);
}

#[tokio::test]
async fn node_failure_without_retry_policy_is_resumable() {
    // No node-retry policy configured: the first failure aborts the run, but a
    // checkpointed thread still leaves a resumable failure boundary (the
    // "resumable abort" default) rather than losing the run.
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let attempts = Arc::new(AtomicUsize::new(0));
    let graph = flaky_graph(1, attempts.clone()).with_checkpointer(cp.clone());

    let err = graph.run_with_thread("once", 7).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
    assert_eq!(
        attempts.load(AtomicOrdering::SeqCst),
        1,
        "no retries attempted"
    );

    // Failed status carries the resumable checkpoint id.
    let status = graph.get_state("once", None).await.unwrap().unwrap();
    assert_eq!(status.next_nodes, vec![NodeId::from("flaky")]);

    // The transient condition has cleared; retry completes the run.
    let resumed = graph.retry("once").await.unwrap();
    assert_eq!(resumed.state, 8);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
}

#[tokio::test]
async fn edit_state_then_retry_uses_the_edited_state() {
    // User-feedback continuation: after a failure, the operator edits committed
    // state via update_state, then retries; the re-run sees the edited value.
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let attempts = Arc::new(AtomicUsize::new(0));
    let graph = flaky_graph(1, attempts.clone()).with_checkpointer(cp.clone());

    graph.run_with_thread("feedback", 0).await.unwrap_err();

    // Operator bumps the committed state by +40, inheriting the failure
    // boundary's pending nodes (`flaky`), then retries. The node adds +1 to
    // whatever state it now sees.
    graph.update_state("feedback", 40, None).await.unwrap();
    let resumed = graph.retry("feedback").await.unwrap();
    assert_eq!(
        resumed.state, 41,
        "retry runs against the edited state (40) + 1"
    );
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
}

#[tokio::test]
async fn parallel_partial_progress_is_preserved_on_failure() {
    // A parallel step where one branch succeeds and a lower/higher-index branch
    // fails: the successful branch's update is folded into committed state and
    // the failure checkpoint schedules only the failed branch for re-run.
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let attempts = Arc::new(AtomicUsize::new(0));
    let graph = GraphBuilder::<i32, i32>::new()
        .with_parallel(true)
        .set_reducer(ClosureStateReducer::new(|s: i32, u: i32| Ok(s + u)))
        .add_node("seed", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["ok", "flaky"])))
        })
        .add_node("ok", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("flaky", move |_s, _c: NodeContext| {
            let attempts = attempts.clone();
            async move {
                // Fail on the first invocation, succeed on resume.
                if attempts.fetch_add(1, AtomicOrdering::SeqCst) == 0 {
                    Err(TinyAgentsError::Model("branch blip".into()))
                } else {
                    Ok(NodeResult::Update(10))
                }
            }
        })
        .set_entry("seed")
        .mark_command_routing("seed")
        .set_finish("ok")
        .set_finish("flaky")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    // First run: seed fans out; the "ok" branch commits +1, "flaky" aborts.
    graph.run_with_thread("fanout", 0).await.unwrap_err();
    let snapshot = graph.get_state("fanout", None).await.unwrap().unwrap();
    assert_eq!(
        snapshot.values, 1,
        "the successful branch's +1 is preserved"
    );
    assert!(
        snapshot.next_nodes.contains(&NodeId::from("flaky")),
        "only the failed branch is scheduled for re-run: {:?}",
        snapshot.next_nodes
    );

    // Resume: the flaky branch now succeeds (+10) without re-running "ok".
    let resumed = graph.retry("fanout").await.unwrap();
    assert_eq!(resumed.state, 11, "1 (preserved) + 10 (re-run branch)");
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
}

// ---------------------------------------------------------------------------
// DurabilityMode::Async
// ---------------------------------------------------------------------------

/// Delegating checkpointer whose `put` sleeps first, then records completion,
/// so tests can observe whether the executor awaited the write inline.
struct SlowCheckpointer {
    inner: Arc<InMemoryCheckpointer<i32>>,
    delay: Duration,
    completed_puts: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl Checkpointer<i32> for SlowCheckpointer {
    async fn put(
        &self,
        checkpoint: crate::graph::checkpoint::Checkpoint<i32>,
    ) -> crate::error::Result<crate::harness::ids::CheckpointId> {
        tokio::time::sleep(self.delay).await;
        let id = self.inner.put(checkpoint).await?;
        self.completed_puts.fetch_add(1, AtomicOrdering::SeqCst);
        Ok(id)
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> crate::error::Result<Option<crate::graph::checkpoint::Checkpoint<i32>>> {
        self.inner.get(thread_id, checkpoint_id).await
    }

    async fn list(
        &self,
        thread_id: &str,
    ) -> crate::error::Result<Vec<crate::graph::checkpoint::CheckpointMetadata>> {
        self.inner.list(thread_id).await
    }

    async fn list_threads(&self) -> crate::error::Result<Vec<String>> {
        self.inner.list_threads().await
    }

    async fn delete_thread(&self, thread_id: &str) -> crate::error::Result<()> {
        self.inner.delete_thread(thread_id).await
    }

    async fn delete_checkpoints(
        &self,
        thread_id: &str,
        ids: &[String],
    ) -> crate::error::Result<usize> {
        self.inner.delete_checkpoints(thread_id, ids).await
    }
}

/// Delegating checkpointer that fails every *non-terminal* boundary `put`
/// (records with pending next nodes), simulating a broken store while the
/// terminal write still succeeds.
struct FailNonTerminalCheckpointer {
    inner: Arc<InMemoryCheckpointer<i32>>,
}

#[async_trait::async_trait]
impl Checkpointer<i32> for FailNonTerminalCheckpointer {
    async fn put(
        &self,
        checkpoint: crate::graph::checkpoint::Checkpoint<i32>,
    ) -> crate::error::Result<crate::harness::ids::CheckpointId> {
        if !checkpoint.next_nodes.is_empty() {
            return Err(crate::error::TinyAgentsError::Checkpoint(
                "injected background write failure".to_string(),
            ));
        }
        self.inner.put(checkpoint).await
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> crate::error::Result<Option<crate::graph::checkpoint::Checkpoint<i32>>> {
        self.inner.get(thread_id, checkpoint_id).await
    }

    async fn list(
        &self,
        thread_id: &str,
    ) -> crate::error::Result<Vec<crate::graph::checkpoint::CheckpointMetadata>> {
        self.inner.list(thread_id).await
    }

    async fn list_threads(&self) -> crate::error::Result<Vec<String>> {
        self.inner.list_threads().await
    }

    async fn delete_thread(&self, thread_id: &str) -> crate::error::Result<()> {
        self.inner.delete_thread(thread_id).await
    }

    async fn delete_checkpoints(
        &self,
        thread_id: &str,
        ids: &[String],
    ) -> crate::error::Result<usize> {
        self.inner.delete_checkpoints(thread_id, ids).await
    }
}

fn two_step_graph() -> crate::graph::GraphBuilder<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
}

#[tokio::test]
async fn async_durability_persists_every_boundary_with_intact_lineage() {
    use crate::graph::checkpoint::DurabilityMode;

    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = two_step_graph()
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone())
        .with_durability(DurabilityMode::Async);

    let run = graph.run_with_thread("t-async", 0).await.unwrap();
    assert_eq!(run.state, 2);
    // Both boundaries are durable by run end: the run drained its background
    // writes before writing the terminal checkpoint.
    let list = cp.list("t-async").await.unwrap();
    assert_eq!(list.len(), 2);
    // Lineage stays chained even though the first write ran in the background
    // (its id was minted before the write was handed off).
    assert_eq!(
        list[1].parent_checkpoint_id.as_deref(),
        Some(list[0].checkpoint_id.as_str())
    );
    assert_eq!(
        run.checkpoint_id.as_ref().map(|id| id.as_str()),
        Some(list[1].checkpoint_id.as_str())
    );
}

#[tokio::test]
async fn async_durability_does_not_await_non_terminal_writes_inline() {
    use crate::graph::checkpoint::DurabilityMode;

    let completed_puts = Arc::new(AtomicUsize::new(0));
    let observed_at_b = Arc::new(AtomicUsize::new(usize::MAX));
    let cp = Arc::new(SlowCheckpointer {
        inner: Arc::new(InMemoryCheckpointer::new()),
        delay: Duration::from_millis(100),
        completed_puts: completed_puts.clone(),
    });

    let puts_for_b = completed_puts.clone();
    let seen = observed_at_b.clone();
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", move |s, _c: NodeContext| {
            let puts = puts_for_b.clone();
            let seen = seen.clone();
            async move {
                // Record how many checkpoint writes had *completed* when this
                // node ran. Under Sync durability the step-1 boundary write
                // (100ms) would have finished first; under Async it is still
                // in flight.
                seen.store(puts.load(AtomicOrdering::SeqCst), AtomicOrdering::SeqCst);
                Ok(NodeResult::Update(s + 1))
            }
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap()
        .with_checkpointer(cp)
        .with_durability(DurabilityMode::Async);

    let run = graph.run_with_thread("t-async-slow", 0).await.unwrap();
    assert_eq!(run.state, 2);
    assert_eq!(
        observed_at_b.load(AtomicOrdering::SeqCst),
        0,
        "node b must start while the step-1 boundary write is still in flight"
    );
    // ...but by run end every write has been drained and is durable.
    assert_eq!(completed_puts.load(AtomicOrdering::SeqCst), 2);
}

#[tokio::test]
async fn async_durability_surfaces_background_write_failure_in_run_result() {
    use crate::graph::checkpoint::DurabilityMode;

    let cp = Arc::new(FailNonTerminalCheckpointer {
        inner: Arc::new(InMemoryCheckpointer::new()),
    });
    let graph = two_step_graph()
        .compile()
        .unwrap()
        .with_checkpointer(cp)
        .with_durability(DurabilityMode::Async);

    // The step-1 boundary write fails in the background; the run must not
    // report success — the failure surfaces at the next durability boundary
    // or, at the latest, at the terminal drain.
    let err = graph
        .run_with_thread("t-async-fail", 0)
        .await
        .expect_err("a lost background checkpoint must fail the run");
    assert!(
        err.to_string()
            .contains("injected background write failure"),
        "unexpected error: {err}"
    );
}
