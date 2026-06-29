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
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
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
