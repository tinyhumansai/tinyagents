//! Per-thread **goal + task board** working together on one thread.
//!
//! This offline example wires both `graph::goals` and `graph::todos` on a single
//! thread and lets the goal *drive* the board:
//!
//! - A durable [`ThreadGoal`] ("ship the v2 release") is the completion
//!   contract, with a token budget.
//! - A [`TaskBoard`] holds the concrete work items (three cards).
//! - A `goal_gate_node` forms a self-driving loop: each iteration the `work`
//!   node advances the board by one kanban transition (Todo → InProgress →
//!   Done), and once every card is Done it marks the goal `Complete`. The gate
//!   keeps looping while the goal is Active and under budget, accounting the
//!   iteration's token usage, and routes to `END` when the goal completes.
//!
//! Both primitives persist on one shared [`InMemoryStore`], addressed by the
//! run's thread id.
//!
//! Run with:
//!
//! ```text
//! cargo run --example goals_and_todos
//! ```

use std::sync::Arc;

use tinyagents::graph::command::NodeResult;
use tinyagents::graph::{NodeContext, NodeFuture};
use tinyagents::harness::store::{InMemoryStore, Store};
use tinyagents::{
    END, GoalProgress, GraphBuilder, Result, TaskCardStatus, goal_gate_node, goal_store, todo_store,
};

/// The thread both primitives are scoped to.
const THREAD: &str = "release-thread";

/// Roughly the tokens each work iteration "spends", accounted against the goal.
const TOKENS_PER_ITERATION: u64 = 500;

/// State overwritten by each work iteration — just a step counter for display.
#[derive(Clone, Debug, Default)]
struct ReleaseState {
    iteration: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // One store backs both the goal and the board for this thread.
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());

    // 1. Set the durable objective with a generous token budget.
    goal_store::set(
        &store,
        THREAD,
        "Ship the v2 release",
        Some(100_000), // token budget
    )
    .await?;

    // 2. Seed the board with the concrete work items.
    for title in [
        "Write the changelog",
        "Tag the release",
        "Publish the crate",
    ] {
        todo_store::add(&store, THREAD, title, Default::default()).await?;
    }

    println!(
        "Initial board:\n{}\n",
        todo_store::list(&store, THREAD).await?.markdown
    );

    // 3a. The work node: advance the board by ONE kanban transition per
    // iteration, then complete the goal once every card is Done.
    let work_store = store.clone();
    let work_node = move |mut state: ReleaseState, _ctx: NodeContext| {
        let store = work_store.clone();
        Box::pin(async move {
            state.iteration += 1;
            let cards = todo_store::list(&store, THREAD).await?.cards;

            if let Some(active) = cards
                .iter()
                .find(|c| c.status == TaskCardStatus::InProgress)
            {
                // Finish the card currently in progress.
                todo_store::update_status(&store, THREAD, &active.id, TaskCardStatus::Done).await?;
                println!("  ✓ done: {}", active.title);
            } else if let Some(next) = cards.iter().find(|c| c.status == TaskCardStatus::Todo) {
                // Pull the next card into progress (single-in-progress invariant).
                todo_store::update_status(&store, THREAD, &next.id, TaskCardStatus::InProgress)
                    .await?;
                println!("  → started: {}", next.title);
            } else {
                // Every card is Done — the objective is satisfied.
                goal_store::complete(&store, THREAD).await?;
                println!("  ★ all cards done → goal complete");
            }

            Ok(NodeResult::Update(state))
        }) as NodeFuture<ReleaseState>
    };

    // 3b. The gate: account each iteration's usage and loop while the goal is
    // Active and under budget, else route to END.
    let gate = goal_gate_node::<ReleaseState, ReleaseState>(
        store.clone(),
        "work",
        |_state: &ReleaseState| GoalProgress {
            tokens_used: TOKENS_PER_ITERATION,
            elapsed_secs: 1,
            made_progress: true,
        },
    );

    // 4. Wire the self-driving loop: START → work → gate → (work | END).
    let graph = GraphBuilder::<ReleaseState, ReleaseState>::overwrite()
        .with_recursion_limit(64)
        .add_node("work", work_node)
        .add_node("gate", gate)
        .set_entry("work")
        .add_edge("work", "gate")
        .with_command_destinations("gate", ["work", END])
        .compile()?;

    println!("Running the goal-driven loop:");
    let exec = graph
        .run_with_thread(THREAD, ReleaseState::default())
        .await?;

    // 5. Report the final state of both primitives.
    let goal = goal_store::get(&store, THREAD).await?.expect("goal exists");
    let board = todo_store::list(&store, THREAD).await?;

    println!(
        "\nFinished after {} work iterations.\n",
        exec.state.iteration
    );
    println!(
        "Goal: {} — status={}, tokens_used={}/{}",
        goal.objective,
        goal.status.as_str(),
        goal.tokens_used,
        goal.token_budget
            .map(|b| b.to_string())
            .unwrap_or_else(|| "∞".into()),
    );
    println!("\nFinal board:\n{}", board.markdown);

    Ok(())
}
