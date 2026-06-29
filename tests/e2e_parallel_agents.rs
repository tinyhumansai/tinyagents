//! TRUE end-to-end, fully OFFLINE: parallel branch execution + forking on a
//! durable [`CompiledGraph`].
//!
//! This composes the **durable graph builder/executor** (`with_parallel`),
//! the **command** model (`Command::goto` fan-out), the **reducer**
//! (concurrent partial-update merge), and the **fork identity** carried on
//! [`NodeContext`]. No model or network is involved, so this test runs (and
//! must pass) under any feature set, including `--features openai` with no key.
//!
//! Topology built by [`fanout_graph`]:
//!
//! ```text
//!            ┌──> worker-a ──┐
//!   START -> dispatch ──> worker-b ──> aggregate -> END
//!            └──> worker-c ──┘
//!            └──> worker-d ──┘
//! ```
//!
//! `dispatch` fans out to four worker branches via a single
//! `Command::goto(["worker-a", .., "worker-d"])`. Each worker contributes a
//! partial update (its produced value, the fork branch index it observed, and
//! its own name) that the reducer folds into shared channels at the superstep
//! boundary. The downstream `aggregate` node then reads the *merged* state and
//! computes a derived total, proving fan-in visibility.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tinyagents::graph::ClosureStateReducer;
use tinyagents::{Command, CompiledGraph, GraphBuilder, NodeContext, NodeResult};

/// Committed graph state. Every field is filled by branches merging through the
/// reducer; `total` is derived downstream from the merged `values`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct AgentState {
    /// Values produced by each worker branch, in reducer-application order.
    values: Vec<i32>,
    /// The `branch_index` each worker observed on its `NodeContext::fork`.
    /// `usize::MAX` is the sentinel for "no fork identity" (sequential mode).
    forks: Vec<usize>,
    /// Names of the workers that contributed, in reducer-application order.
    workers: Vec<String>,
    /// Sum computed downstream over the merged `values` (fan-in proof).
    total: Option<i32>,
}

/// Partial-update type merged through the [`ClosureStateReducer`].
#[derive(Clone, Debug)]
enum AgentUpdate {
    /// One worker branch's contribution.
    Work {
        value: i32,
        fork: usize,
        worker: String,
    },
    /// The downstream aggregate's derived total.
    Total(i32),
}

/// Shared instrumentation that records the maximum number of branches that were
/// ever in flight simultaneously. In parallel mode this should reach the branch
/// count; in sequential mode it must never exceed one.
#[derive(Clone)]
struct Inflight {
    current: Arc<AtomicUsize>,
    max: Arc<AtomicUsize>,
}

impl Inflight {
    fn new() -> Self {
        Self {
            current: Arc::new(AtomicUsize::new(0)),
            max: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn max_observed(&self) -> usize {
        self.max.load(Ordering::SeqCst)
    }

    /// Marks a branch as in flight for `sleep`, updating the high-water mark,
    /// then yields `value`.
    async fn run<T>(&self, sleep: Duration, value: T) -> T {
        let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(sleep).await;
        self.current.fetch_sub(1, Ordering::SeqCst);
        value
    }
}

/// Reads the fork branch index from a [`NodeContext`], using `usize::MAX` as
/// the "no fork identity" sentinel (sequential mode / single-node steps).
fn fork_index(ctx: &NodeContext) -> usize {
    ctx.fork
        .as_ref()
        .map(|f| f.branch_index)
        .unwrap_or(usize::MAX)
}

/// Builds the fan-out/fan-in agent graph. `parallel` toggles concurrent branch
/// execution; `inflight` instruments concurrency. Worker sleeps are reversed
/// (a longest, d shortest) so reducer-merge order can never accidentally match
/// completion order — any deterministic ordering must come from the active-set
/// position, not from which branch finishes first.
fn fanout_graph(parallel: bool, inflight: Inflight) -> CompiledGraph<AgentState, AgentUpdate> {
    // Each worker: (name, produced value, sleep millis).
    let workers = [
        ("worker-a", 1, 80u64),
        ("worker-b", 2, 60),
        ("worker-c", 4, 40),
        ("worker-d", 8, 20),
    ];

    let mut builder = GraphBuilder::<AgentState, AgentUpdate>::new()
        .with_parallel(parallel)
        .set_reducer(ClosureStateReducer::new(
            |mut s: AgentState, u: AgentUpdate| {
                match u {
                    AgentUpdate::Work {
                        value,
                        fork,
                        worker,
                    } => {
                        s.values.push(value);
                        s.forks.push(fork);
                        s.workers.push(worker);
                    }
                    AgentUpdate::Total(t) => s.total = Some(t),
                }
                Ok(s)
            },
        ))
        // dispatch fans out to all workers in one command.
        .add_node("dispatch", |_s: AgentState, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::default().with_goto([
                "worker-a", "worker-b", "worker-c", "worker-d",
            ])))
        });

    for (name, value, sleep_ms) in workers {
        let inflight = inflight.clone();
        builder = builder.add_node(name, move |_s: AgentState, c: NodeContext| {
            let inflight = inflight.clone();
            let fork = fork_index(&c);
            async move {
                let update = AgentUpdate::Work {
                    value,
                    fork,
                    worker: name.to_string(),
                };
                Ok(NodeResult::Update(
                    inflight.run(Duration::from_millis(sleep_ms), update).await,
                ))
            }
        });
    }

    builder = builder
        // aggregate reads the merged state after fan-in.
        .add_node("aggregate", |s: AgentState, _c: NodeContext| async move {
            Ok(NodeResult::Update(AgentUpdate::Total(
                s.values.iter().sum(),
            )))
        })
        .set_entry("dispatch")
        .mark_command_routing("dispatch")
        .add_edge("worker-a", "aggregate")
        .add_edge("worker-b", "aggregate")
        .add_edge("worker-c", "aggregate")
        .add_edge("worker-d", "aggregate")
        .set_finish("aggregate");

    builder.compile().expect("graph compiles")
}

#[tokio::test]
async fn parallel_fanout_merges_all_branches_and_downstream_sees_them() {
    let inflight = Inflight::new();
    let graph = fanout_graph(true, inflight.clone());
    let run = graph.run(AgentState::default()).await.unwrap();

    // All four worker branches were in flight at the same time.
    assert_eq!(
        inflight.max_observed(),
        4,
        "parallel mode should run every branch concurrently"
    );

    // The reducer folded in every branch's contribution, in deterministic
    // active-set order (goto order), NOT completion order.
    assert_eq!(run.state.values, vec![1, 2, 4, 8]);
    assert_eq!(
        run.state.workers,
        vec!["worker-a", "worker-b", "worker-c", "worker-d"],
    );

    // Each fork saw its stable branch index from the active set.
    assert_eq!(run.state.forks, vec![0, 1, 2, 3]);

    // The downstream aggregate node observed the fully merged state.
    assert_eq!(run.state.total, Some(15));

    // dispatch | (a,b,c,d) | aggregate == 3 supersteps.
    assert_eq!(run.steps, 3);

    // Every node was visited; workers executed within a single superstep.
    for node in [
        "dispatch",
        "worker-a",
        "worker-b",
        "worker-c",
        "worker-d",
        "aggregate",
    ] {
        assert!(
            run.visited.iter().any(|n| n.as_str() == node),
            "expected `{node}` in visited history {:?}",
            run.visited,
        );
    }
}

#[tokio::test]
async fn sequential_mode_runs_one_branch_at_a_time_without_fork_identity() {
    let inflight = Inflight::new();
    let graph = fanout_graph(false, inflight.clone());
    let run = graph.run(AgentState::default()).await.unwrap();

    // Never more than one branch in flight in sequential mode.
    assert_eq!(
        inflight.max_observed(),
        1,
        "sequential mode must serialize branches"
    );

    // Sequential branches receive no fork identity.
    assert_eq!(
        run.state.forks,
        vec![usize::MAX, usize::MAX, usize::MAX, usize::MAX],
    );
}

#[tokio::test]
async fn parallel_and_sequential_reach_the_same_final_state() {
    // Final committed state (modulo fork identity) must be identical whether
    // branches run concurrently or one-at-a-time. This is the core determinism
    // guarantee of the executor: parallelism is an execution strategy, not a
    // semantic change.
    let parallel = fanout_graph(true, Inflight::new())
        .run(AgentState::default())
        .await
        .unwrap();
    let sequential = fanout_graph(false, Inflight::new())
        .run(AgentState::default())
        .await
        .unwrap();

    assert_eq!(parallel.state.values, sequential.state.values);
    assert_eq!(parallel.state.workers, sequential.state.workers);
    assert_eq!(parallel.state.total, sequential.state.total);
    assert_eq!(parallel.steps, sequential.steps);

    // The only intentional difference is fork identity.
    assert_ne!(parallel.state.forks, sequential.state.forks);
}

#[tokio::test]
async fn parallel_merge_order_is_reproducible_across_runs() {
    // Re-running the same concurrent fan-out must always produce the same merge
    // order, regardless of which branch's sleep finishes first.
    for _ in 0..5 {
        let run = fanout_graph(true, Inflight::new())
            .run(AgentState::default())
            .await
            .unwrap();
        assert_eq!(run.state.values, vec![1, 2, 4, 8]);
        assert_eq!(run.state.forks, vec![0, 1, 2, 3]);
        assert_eq!(run.state.total, Some(15));
    }
}
