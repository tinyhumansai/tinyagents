//! Graph-native continuation for the thread goal.
//!
//! OpenHuman drives a goal from an out-of-band heartbeat: when a thread carries
//! an active goal and goes idle, the heartbeat injects one continuation turn,
//! marks the goal one-shot `continuation_suppressed`, and a later user turn
//! clears it. TinyAgents has no heartbeat and no ambient agent, so the same
//! behaviour is re-expressed on the graph runtime three ways:
//!
//! - [`goal_gate_node`] (**primary**) — a self-driving bounded loop. A
//!   command-routing node that, after each work iteration, folds the iteration's
//!   usage into the goal and routes back to the work node while the goal is
//!   Active and under budget, else routes to [`END`]. The graph's
//!   `recursion_limit` is the hard backstop.
//! - [`run_continuation_tick`] (**driver**) — a faithful port of the heartbeat
//!   for callers that *do* have an external scheduler: it selects idle active
//!   goals and runs one turn each through a caller-supplied closure.
//! - [`note_user_turn`] — call at the start of a user-initiated run to clear the
//!   one-shot suppression (and reactivate a paused goal), the analogue of
//!   OpenHuman's post-turn `account_turn_against_goal` suppression reset.
//!
//! # Token accounting boundary
//!
//! The graph runtime is provider-neutral and does not meter tokens per node.
//! Accounting is therefore **explicit**: the work node records what it spent
//! into `State` and the caller's `progress`/`run_turn` closure reports it as a
//! [`GoalProgress`]/[`TurnOutcome`]. `made_progress == false` stops the loop
//! (one-shot suppression) exactly as "the turn produced no tool calls" does in
//! OpenHuman.

use std::sync::Arc;
use std::time::Duration;

use super::store;
use super::types::{GoalProgress, ThreadGoal, ThreadGoalStatus, TurnOutcome};
use crate::error::Result;
use crate::graph::builder::{END, NodeContext, NodeFuture};
use crate::graph::command::{Command, NodeResult};
use crate::harness::store::Store;

/// Builds a **goal gate** node handler: a self-driving bounded loop that keeps
/// re-running `work_node` while the thread's goal is Active and under budget.
///
/// Wire it as `work_node -> gate` (a static edge) and register `gate` as a
/// command node whose destinations are `[work_node, END]`
/// (`with_command_destinations`). On each activation the gate:
///
/// 1. reads the thread id from the [`NodeContext`] (absent → routes to [`END`]);
/// 2. loads the goal (`None` → routes to [`END`]);
/// 3. folds the just-finished iteration's [`GoalProgress`] via
///    [`store::account_usage`], which flips an over-budget goal to
///    [`BudgetLimited`](ThreadGoalStatus::BudgetLimited);
/// 4. routes a single `goto`:
///    - not Active, or `continuation_suppressed` → [`END`];
///    - the iteration made no progress → set one-shot suppression, then [`END`];
///    - otherwise → back to `work_node` (loop).
///
/// The handler only routes (it never updates state), so `Update` carries no
/// bound beyond `Send`. Pass the returned closure straight to
/// [`GraphBuilder::add_node`](crate::graph::GraphBuilder::add_node).
pub fn goal_gate_node<State, Update>(
    store: Arc<dyn Store>,
    work_node: impl Into<crate::harness::ids::NodeId>,
    progress: impl Fn(&State) -> GoalProgress + Send + Sync + 'static,
) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Send + 'static,
{
    let work_node = work_node.into();
    let progress = Arc::new(progress);
    move |state, ctx| {
        let store = store.clone();
        let work_node = work_node.clone();
        let progress = progress.clone();
        Box::pin(async move {
            let Some(thread_id) = ctx.thread_id.as_ref().map(|t| t.as_str().to_string()) else {
                return Ok(NodeResult::Command(Command::goto([END])));
            };
            let Some(goal) = store::get(&store, &thread_id).await? else {
                return Ok(NodeResult::Command(Command::goto([END])));
            };

            let p = progress(&state);
            let goal = match store::account_usage(
                &store,
                &thread_id,
                &goal.goal_id,
                p.tokens_used,
                p.elapsed_secs,
            )
            .await?
            {
                Some(g) => g,
                None => return Ok(NodeResult::Command(Command::goto([END]))),
            };

            if !goal.status.is_active() || goal.continuation_suppressed {
                return Ok(NodeResult::Command(Command::goto([END])));
            }
            if !p.made_progress {
                // One-shot: a zero-progress iteration stops the loop until a
                // user-initiated run clears the flag (see `note_user_turn`).
                store::set_continuation_suppressed_if(&store, &thread_id, &goal.goal_id, true)
                    .await?;
                return Ok(NodeResult::Command(Command::goto([END])));
            }
            Ok(NodeResult::Command(Command::goto([work_node])))
        })
    }
}

/// One continuation pass for callers with an external scheduler: run at most
/// `max_per_tick` idle active goals, oldest-idle first, one turn each.
///
/// A goal is a candidate when it is Active, not already suppressed, and has been
/// idle (no `updated_at` change) for at least `idle`. For each selected goal the
/// closure `run_turn` runs one turn; its [`TurnOutcome`] is folded via
/// [`store::account_usage`], and a no-progress turn sets the one-shot
/// suppression. Returns the number of turns actually run.
///
/// A `run_turn` that errors is skipped (best-effort — a failed turn must not
/// stop the whole tick); [`Store`] errors from accounting propagate. This is a
/// single sequential async fn, so it already serializes; concurrent scheduling
/// is the caller's responsibility.
pub async fn run_continuation_tick<F, Fut>(
    store: &Arc<dyn Store>,
    idle: Duration,
    max_per_tick: usize,
    run_turn: F,
) -> Result<usize>
where
    F: Fn(ThreadGoal) -> Fut,
    Fut: Future<Output = Result<TurnOutcome>>,
{
    let now = store::now_ms();
    let idle_ms = idle.as_millis() as u64;
    let mut candidates: Vec<ThreadGoal> = store::list_all(store)
        .await?
        .into_iter()
        .filter(|g| g.status.is_active())
        .filter(|g| !g.continuation_suppressed)
        .filter(|g| now.saturating_sub(g.updated_at_ms) >= idle_ms)
        .collect();
    // Oldest-idle first so the most-neglected goal advances under the cap.
    candidates.sort_by_key(|g| g.updated_at_ms);
    candidates.truncate(max_per_tick);

    let mut ran = 0;
    for goal in candidates {
        let outcome = match run_turn(goal.clone()).await {
            Ok(o) => o,
            Err(_) => continue,
        };
        store::account_usage(
            store,
            &goal.thread_id,
            &goal.goal_id,
            outcome.tokens_used,
            outcome.elapsed_secs,
        )
        .await?;
        if !outcome.made_progress {
            store::set_continuation_suppressed_if(store, &goal.thread_id, &goal.goal_id, true)
                .await?;
        }
        ran += 1;
    }
    Ok(ran)
}

/// Notes the start of a **user-initiated** run for `thread_id`: clears the
/// one-shot continuation suppression on an active goal and reactivates a paused
/// one. Returns the goal as it stands afterward, or `None` when the thread has
/// none.
///
/// This is what distinguishes a user turn from a self-driving loop iteration —
/// the loop never calls it, so it can never clear its own suppression.
pub async fn note_user_turn(store: &Arc<dyn Store>, thread_id: &str) -> Result<Option<ThreadGoal>> {
    let Some(goal) = store::get(store, thread_id).await? else {
        return Ok(None);
    };
    match goal.status {
        ThreadGoalStatus::Paused => store::resume(store, thread_id).await.map(Some),
        ThreadGoalStatus::Active if goal.continuation_suppressed => {
            store::set_continuation_suppressed_if(store, thread_id, &goal.goal_id, false).await
        }
        _ => Ok(Some(goal)),
    }
}
