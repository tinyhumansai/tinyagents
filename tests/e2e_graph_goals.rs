//! End-to-end coverage for the per-thread goal (`graph::goals`) exercised
//! through the public crate surface: a self-driving graph loop wired with
//! `goal_gate_node` keeps re-running a work node while the thread's goal is
//! active, and stops when the work node marks the goal complete.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tinyagents::graph::command::NodeResult;
use tinyagents::graph::{NodeContext, NodeFuture};
use tinyagents::harness::store::{InMemoryStore, Store};
use tinyagents::{GoalProgress, GraphBuilder, ThreadGoalStatus, goal_gate_node, goal_store};

/// State overwritten by each work iteration.
#[derive(Clone, Debug, Default, PartialEq)]
struct LoopState {
    iters: usize,
}

#[tokio::test]
async fn self_driving_goal_loop_runs_until_complete() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());
    goal_store::set(&store, "thread-goal", "process every item", None)
        .await
        .expect("set goal");

    // The work node advances a counter and, on the third iteration, marks the
    // goal complete (the graph analogue of a model calling `goal_complete`).
    let counter = Arc::new(AtomicUsize::new(0));
    let work_store = store.clone();
    let work_node = move |_state: LoopState, _ctx: NodeContext| {
        let counter = counter.clone();
        let work_store = work_store.clone();
        Box::pin(async move {
            let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
            if n >= 3 {
                goal_store::complete(&work_store, "thread-goal")
                    .await
                    .expect("complete goal");
            }
            Ok(NodeResult::Update(LoopState { iters: n }))
        }) as NodeFuture<LoopState>
    };

    // The gate accounts a fixed cost per iteration and reports progress so the
    // loop is only stopped by the goal reaching `Complete`.
    let gate = goal_gate_node::<LoopState, LoopState>(store.clone(), "work", |_s: &LoopState| {
        GoalProgress {
            tokens_used: 10,
            elapsed_secs: 1,
            made_progress: true,
        }
    });

    let graph = GraphBuilder::<LoopState, LoopState>::overwrite()
        .with_recursion_limit(64)
        .add_node("work", work_node)
        .add_node("gate", gate)
        .set_entry("work")
        .add_edge("work", "gate")
        .with_command_destinations("gate", ["work", tinyagents::END])
        .compile()
        .expect("graph compiles");

    let exec = graph
        .run_with_thread("thread-goal", LoopState::default())
        .await
        .expect("graph runs to completion");

    assert_eq!(exec.state.iters, 3, "loops until the goal is completed");

    let goal = goal_store::get(&store, "thread-goal")
        .await
        .expect("load goal")
        .expect("goal exists");
    assert_eq!(goal.status, ThreadGoalStatus::Complete);
    // Usage was accounted across the iterations that ran through the gate.
    assert!(
        goal.tokens_used >= 20,
        "usage accrued: {}",
        goal.tokens_used
    );
}
