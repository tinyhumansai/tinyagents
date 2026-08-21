//! Feature coverage for goal budget enforcement (`graph::goals::budget`) through
//! the public crate surface.
//!
//! Two halves of the same policy: [`account_turn`] charges a finished turn
//! against the thread's goal, and a [`GoalBudgetGuard`] decides mid-turn
//! whether the turn should stop before it overruns. Together they are what
//! keeps an autonomous run from spending without a ceiling — and what keeps a
//! *user-present* conversation from being hard-stopped once there is no live
//! budget left to protect.

use std::sync::Arc;

use tinyagents::harness::store::{InMemoryStore, Store};
use tinyagents::{
    BudgetVerdict, GoalBudgetGuard, ThreadGoalStatus, account_turn, goal_store, turn_tokens,
};

fn store() -> Arc<dyn Store> {
    Arc::new(InMemoryStore::default())
}

#[tokio::test]
async fn an_autonomous_run_spends_down_its_budget_and_stops() {
    let store = store();
    let goal = goal_store::set(&store, "thread-1", "reindex everything", Some(300))
        .await
        .expect("set goal");
    let guard = GoalBudgetGuard::for_goal(&goal).expect("budgeted goal is enforceable");

    // Two turns inside the ceiling: charged, still active, still runnable.
    for _ in 0..2 {
        let updated = account_turn(&store, "thread-1", 60, 40, 5, false)
            .await
            .expect("account")
            .expect("goal charged");
        assert_eq!(updated.status, ThreadGoalStatus::Active);
        assert_eq!(
            guard.check(&store, 0).await.unwrap(),
            BudgetVerdict::Continue
        );
    }
    let goal = goal_store::get(&store, "thread-1").await.unwrap().unwrap();
    assert_eq!(goal.tokens_used, 200);
    assert_eq!(goal.budget_remaining(), Some(100));

    // Mid-turn, the guard sees the projected spend cross the ceiling and calls
    // a stop before the turn burns through it.
    assert_eq!(
        guard.check(&store, 50).await.unwrap(),
        BudgetVerdict::Continue
    );
    let verdict = guard.check(&store, 100).await.unwrap();
    assert!(verdict.is_stop(), "{verdict:?}");

    // The turn wraps up and is accounted; the goal becomes budget-limited.
    let final_goal = account_turn(&store, "thread-1", 60, 60, 3, false)
        .await
        .expect("account")
        .expect("goal charged");
    assert_eq!(final_goal.status, ThreadGoalStatus::BudgetLimited);
    assert!(final_goal.over_budget());
    assert_eq!(final_goal.budget_remaining(), Some(0));
}

#[tokio::test]
async fn a_limited_goal_stops_charging_and_stops_hard_stopping() {
    let store = store();
    let goal = goal_store::set(&store, "thread-1", "ship it", Some(100))
        .await
        .expect("set goal");
    let guard = GoalBudgetGuard::for_goal(&goal).expect("guard");
    account_turn(&store, "thread-1", 80, 40, 1, false)
        .await
        .expect("account");

    let limited = goal_store::get(&store, "thread-1").await.unwrap().unwrap();
    assert_eq!(limited.status, ThreadGoalStatus::BudgetLimited);
    let used = limited.tokens_used;

    // The user keeps talking. Nothing further is charged against the exhausted
    // goal, and the guard stands down — the injected goal context is what
    // steers the model now, not a hard stop on a user-present turn.
    assert!(
        account_turn(&store, "thread-1", 500, 500, 60, true)
            .await
            .expect("account")
            .is_none()
    );
    assert_eq!(
        goal_store::get(&store, "thread-1")
            .await
            .unwrap()
            .unwrap()
            .tokens_used,
        used
    );
    assert_eq!(
        guard.check(&store, 10_000).await.unwrap(),
        BudgetVerdict::Continue
    );
}

#[tokio::test]
async fn a_goal_with_no_budget_is_charged_but_never_stopped() {
    let store = store();
    let goal = goal_store::set(&store, "thread-1", "keep an eye on things", None)
        .await
        .expect("set goal");

    // Nothing to enforce, so there is no guard to arm.
    assert!(GoalBudgetGuard::for_goal(&goal).is_none());

    // Usage is still tracked — a host may want the number even with no cap.
    let updated = account_turn(&store, "thread-1", 5_000, 5_000, 600, true)
        .await
        .expect("account")
        .expect("goal charged");
    assert_eq!(updated.tokens_used, turn_tokens(5_000, 5_000));
    assert_eq!(updated.time_used_seconds, 600);
    assert_eq!(updated.status, ThreadGoalStatus::Active);
    assert_eq!(updated.budget_remaining(), None);
}

#[tokio::test]
async fn a_replaced_objective_starts_a_fresh_budget_and_disarms_the_old_guard() {
    let store = store();
    let first = goal_store::set(&store, "thread-1", "objective one", Some(100))
        .await
        .expect("set goal");
    let stale_guard = GoalBudgetGuard::for_goal(&first).expect("guard");
    account_turn(&store, "thread-1", 50, 40, 1, true)
        .await
        .expect("account");

    // A new objective mints a new goal id and resets the counters.
    let second = goal_store::set(&store, "thread-1", "objective two", Some(100))
        .await
        .expect("replace goal");
    assert_ne!(second.goal_id, first.goal_id);
    assert_eq!(second.tokens_used, 0);

    // The guard armed for the previous objective quietly stands down instead of
    // enforcing a ceiling that no longer describes the work.
    assert_eq!(
        stale_guard.check(&store, 10_000).await.unwrap(),
        BudgetVerdict::Continue
    );

    // A guard for the current goal enforces the fresh budget.
    let guard = GoalBudgetGuard::for_goal(&second).expect("guard");
    assert!(guard.check(&store, 100).await.unwrap().is_stop());
}

#[tokio::test]
async fn a_user_turn_re_arms_continuation_but_the_continuation_itself_does_not() {
    let store = store();
    let goal = goal_store::set(&store, "thread-1", "watch the queue", None)
        .await
        .expect("set goal");

    // An idle-period continuation fired and suppressed itself one-shot.
    goal_store::set_continuation_suppressed_if(&store, "thread-1", &goal.goal_id, true)
        .await
        .expect("suppress");

    // Accounting for the continuation's own turn must not clear that flag, or
    // the loop would drive itself forever.
    account_turn(&store, "thread-1", 100, 50, 10, false)
        .await
        .expect("account");
    assert!(
        goal_store::get(&store, "thread-1")
            .await
            .unwrap()
            .unwrap()
            .continuation_suppressed
    );

    // A person replying re-arms it: the next idle period may continue again.
    account_turn(&store, "thread-1", 100, 50, 10, true)
        .await
        .expect("account");
    let goal = goal_store::get(&store, "thread-1").await.unwrap().unwrap();
    assert!(!goal.continuation_suppressed);
    assert_eq!(goal.tokens_used, 300, "both turns were charged");
}

#[tokio::test]
async fn a_thread_with_no_goal_is_untouched_and_unguarded() {
    let store = store();
    assert!(
        account_turn(&store, "no-goal-here", 100, 100, 10, true)
            .await
            .expect("account")
            .is_none()
    );
    assert!(
        goal_store::get(&store, "no-goal-here")
            .await
            .unwrap()
            .is_none()
    );
}
