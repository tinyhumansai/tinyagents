//! Feature tests for graph-level `Send` fan-out joined through a state reducer.
//!
//! Where `tests/e2e_complex_graph.rs` fans out with a static `goto` to two
//! *named* branch nodes, this covers the dynamic `Send` primitive: one worker
//! node scheduled once per argument, each activation receiving its own
//! `send_arg`, with the per-branch updates folded back into committed state by
//! the reducer in deterministic branch order — under both sequential and
//! bounded-parallel execution — and (in parallel mode) a distinct fork identity
//! per branch.

use std::sync::{Arc, Mutex};

use serde_json::json;

use tinyagents::{
    ClosureStateReducer, GraphBuilder, NodeContext, NodeResult, fanout_node, run_recorded,
};

/// Committed state: the append-only list of squared work items.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Squares {
    values: Vec<i64>,
}

/// Records each worker's `(fork branch index, arg)`; the fork is `None` in
/// sequential mode and `Some(_)` under parallel execution.
type ForkLog = Arc<Mutex<Vec<(Option<usize>, i64)>>>;

/// Builds a `spread -> square (xN) -> collect` graph. `spread` fans out one
/// `Send` per argument; each `square` activation squares its own `send_arg` and
/// records the fork it ran on. `collect` is a join that runs once after every
/// branch has folded in.
fn fanout_graph(
    parallel: bool,
    max_concurrency: Option<usize>,
    forks: ForkLog,
) -> tinyagents::CompiledGraph<Squares, i64> {
    let mut builder = GraphBuilder::<Squares, i64>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Squares, u: i64| {
            s.values.push(u);
            Ok(s)
        }))
        .with_parallel(parallel);
    if let Some(cap) = max_concurrency {
        builder = builder.with_max_concurrency(cap);
    }
    builder
        .add_node(
            "spread",
            fanout_node("square", [json!(1), json!(2), json!(3), json!(4)]),
        )
        .add_node("square", move |_s: Squares, ctx: NodeContext| {
            let forks = forks.clone();
            async move {
                let arg = ctx
                    .send_arg
                    .as_ref()
                    .and_then(|v| v.as_i64())
                    .expect("each worker carries a send arg");
                let branch = ctx.fork.as_ref().map(|f| f.branch_index);
                forks.lock().unwrap().push((branch, arg));
                Ok(NodeResult::Update(arg * arg))
            }
        })
        .add_node("collect", tinyagents::noop_node())
        .set_entry("spread")
        .add_edge("square", "collect")
        .set_finish("collect")
        .compile()
        .expect("fan-out graph compiles")
}

#[tokio::test]
async fn send_fanout_folds_each_branch_update_through_the_reducer() {
    let forks: ForkLog = Arc::new(Mutex::new(Vec::new()));
    let run = run_recorded(
        &fanout_graph(false, None, forks.clone()),
        None,
        Squares::default(),
    )
    .await
    .expect("run succeeds");

    // The reducer folds the four branch updates in deterministic branch order.
    assert_eq!(run.execution.state.values, vec![1, 4, 9, 16]);

    // The worker was scheduled once per argument, each with its own send arg.
    let square_visits = run
        .execution
        .visited
        .iter()
        .filter(|n| n.as_str() == "square")
        .count();
    assert_eq!(square_visits, 4);

    let mut args: Vec<i64> = forks.lock().unwrap().iter().map(|(_, a)| *a).collect();
    args.sort_unstable();
    assert_eq!(args, vec![1, 2, 3, 4]);
}

#[tokio::test]
async fn each_parallel_branch_receives_a_distinct_fork_identity() {
    let forks: ForkLog = Arc::new(Mutex::new(Vec::new()));
    let _ = run_recorded(
        &fanout_graph(true, None, forks.clone()),
        None,
        Squares::default(),
    )
    .await
    .expect("run succeeds");

    let mut recorded = forks.lock().unwrap().clone();
    recorded.sort_by_key(|(branch, _)| branch.unwrap_or(usize::MAX));
    // Four distinct branch indices (0..4), each paired with its own send arg.
    assert_eq!(
        recorded,
        vec![(Some(0), 1), (Some(1), 2), (Some(2), 3), (Some(3), 4)]
    );
}

#[tokio::test]
async fn parallel_fanout_is_deterministic_under_a_bounded_concurrency_cap() {
    let forks: ForkLog = Arc::new(Mutex::new(Vec::new()));
    // Run in parallel with at most two branches in flight at once.
    let graph = fanout_graph(true, Some(2), forks.clone());

    let run = run_recorded(&graph, None, Squares::default())
        .await
        .expect("parallel run succeeds");

    // Reducer application stays in stable branch order regardless of the
    // parallel completion order, so the joined state matches the sequential run.
    assert_eq!(run.execution.state.values, vec![1, 4, 9, 16]);

    // The join runs exactly once after all branches drained.
    let collect_visits = run
        .execution
        .visited
        .iter()
        .filter(|n| n.as_str() == "collect")
        .count();
    assert_eq!(collect_visits, 1);
}
