//! Unit tests for the superstep executor: sequential and parallel runs,
//! reducer fan-in ordering, conditional/command routing, checkpoint
//! persistence, interrupt/resume, and recursion-limit enforcement.

use super::*;
use crate::graph::builder::{GraphBuilder, NodeContext};
use crate::graph::checkpoint::{Checkpointer, InMemoryCheckpointer};
use crate::graph::command::{Command, Interrupt, NodeResult};
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
