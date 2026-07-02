//! Unit tests for the thread-goal domain types.

use super::types::*;

fn goal(status: ThreadGoalStatus, token_budget: Option<u64>, tokens_used: u64) -> ThreadGoal {
    ThreadGoal {
        thread_id: "t".into(),
        goal_id: "goal-0".into(),
        objective: "ship the release".into(),
        status,
        token_budget,
        tokens_used,
        time_used_seconds: 0,
        created_at_ms: 0,
        updated_at_ms: 0,
        continuation_suppressed: false,
    }
}

#[test]
fn status_strings_match_serialized() {
    assert_eq!(ThreadGoalStatus::Active.as_str(), "active");
    assert_eq!(ThreadGoalStatus::Paused.as_str(), "paused");
    assert_eq!(ThreadGoalStatus::BudgetLimited.as_str(), "budget_limited");
    assert_eq!(ThreadGoalStatus::Complete.as_str(), "complete");
}

#[test]
fn active_and_terminal_predicates() {
    assert!(ThreadGoalStatus::Active.is_active());
    assert!(!ThreadGoalStatus::Paused.is_active());
    assert!(ThreadGoalStatus::Complete.is_terminal());
    assert!(ThreadGoalStatus::BudgetLimited.is_terminal());
    assert!(!ThreadGoalStatus::Active.is_terminal());
    assert!(!ThreadGoalStatus::Paused.is_terminal());
}

#[test]
fn budget_helpers() {
    let mut g = goal(ThreadGoalStatus::Active, Some(100), 40);
    assert_eq!(g.budget_remaining(), Some(60));
    assert!(!g.over_budget());
    g.tokens_used = 120;
    assert_eq!(g.budget_remaining(), Some(0));
    assert!(g.over_budget());
    g.token_budget = None;
    assert_eq!(g.budget_remaining(), None);
    assert!(!g.over_budget());
}

#[test]
fn context_block_present_only_for_active_and_budget_limited() {
    let active = goal(ThreadGoalStatus::Active, Some(500), 100);
    let block = active_goal_context_block(&active).expect("active goal has a context block");
    assert!(block.contains("ship the release"));
    assert!(block.contains("goal_complete"));
    assert!(block.contains("400"), "should name remaining budget");

    let limited = goal(ThreadGoalStatus::BudgetLimited, Some(100), 100);
    let block = active_goal_context_block(&limited).expect("budget-limited has a stop block");
    assert!(block.contains("budget"));

    assert!(active_goal_context_block(&goal(ThreadGoalStatus::Paused, None, 0)).is_none());
    assert!(active_goal_context_block(&goal(ThreadGoalStatus::Complete, None, 0)).is_none());
}

#[test]
fn thread_goal_round_trips_through_json() {
    let g = goal(ThreadGoalStatus::Active, Some(1000), 250);
    let json = serde_json::to_value(&g).unwrap();
    // camelCase field names on the wire.
    assert_eq!(json["threadId"], "t");
    assert_eq!(json["goalId"], "goal-0");
    assert_eq!(json["tokenBudget"], 1000);
    let back: ThreadGoal = serde_json::from_value(json).unwrap();
    assert_eq!(back, g);
}

mod store_tests {
    use std::sync::Arc;

    use super::super::store;
    use super::ThreadGoalStatus;
    use crate::harness::store::{InMemoryStore, Store};

    fn store() -> Arc<dyn Store> {
        Arc::new(InMemoryStore::default())
    }

    #[tokio::test]
    async fn set_get_clear_round_trip() {
        let s = store();
        assert!(store::get(&s, "t1").await.unwrap().is_none());

        let g = store::set(&s, "t1", "ship the feature", None)
            .await
            .unwrap();
        assert_eq!(g.objective, "ship the feature");
        assert_eq!(g.status, ThreadGoalStatus::Active);
        assert_eq!(g.tokens_used, 0);

        let loaded = store::get(&s, "t1").await.unwrap().unwrap();
        assert_eq!(loaded.goal_id, g.goal_id);

        assert!(store::clear(&s, "t1").await.unwrap());
        assert!(store::get(&s, "t1").await.unwrap().is_none());
        assert!(!store::clear(&s, "t1").await.unwrap());
    }

    #[tokio::test]
    async fn set_same_objective_preserves_goal_id_and_counters() {
        let s = store();
        let g1 = store::set(&s, "t", "objective A", Some(100)).await.unwrap();
        store::account_usage(&s, "t", &g1.goal_id, 30, 5)
            .await
            .unwrap()
            .unwrap();
        let g2 = store::set(&s, "t", "objective A", Some(200)).await.unwrap();
        assert_eq!(g1.goal_id, g2.goal_id, "same objective keeps goal_id");
        assert_eq!(g2.tokens_used, 30, "counters preserved");
        assert_eq!(g2.token_budget, Some(200), "budget refreshed");
    }

    #[tokio::test]
    async fn set_same_objective_stays_budget_limited_when_over_budget() {
        let s = store();
        let g = store::set(&s, "t", "obj", Some(100)).await.unwrap();
        let limited = store::account_usage(&s, "t", &g.goal_id, 120, 0)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(limited.status, ThreadGoalStatus::BudgetLimited);
        let resed = store::set(&s, "t", "obj", Some(100)).await.unwrap();
        assert_eq!(resed.tokens_used, 120, "same-objective preserves counters");
        assert_eq!(
            resed.status,
            ThreadGoalStatus::BudgetLimited,
            "still over budget → must stay budget_limited, not active"
        );
        let raised = store::set(&s, "t", "obj", Some(1000)).await.unwrap();
        assert_eq!(raised.status, ThreadGoalStatus::Active);
    }

    #[tokio::test]
    async fn set_changed_objective_mints_new_goal_id_and_resets() {
        let s = store();
        let g1 = store::set(&s, "t", "objective A", None).await.unwrap();
        store::account_usage(&s, "t", &g1.goal_id, 30, 5)
            .await
            .unwrap()
            .unwrap();
        let g2 = store::set(&s, "t", "objective B", None).await.unwrap();
        assert_ne!(g1.goal_id, g2.goal_id);
        assert_eq!(g2.tokens_used, 0, "counters reset on new objective");
        assert_eq!(g2.created_at_ms, g1.created_at_ms, "created_at preserved");
    }

    #[tokio::test]
    async fn set_if_absent_only_bootstraps_when_empty() {
        let s = store();
        let created = store::set_if_absent(&s, "t", "scout goal", None)
            .await
            .unwrap();
        assert!(created.is_some());
        let again = store::set_if_absent(&s, "t", "different goal", None)
            .await
            .unwrap();
        assert!(again.is_none());
        assert_eq!(
            store::get(&s, "t").await.unwrap().unwrap().objective,
            "scout goal"
        );
    }

    #[tokio::test]
    async fn account_usage_ignores_stale_goal_id() {
        let s = store();
        let g = store::set(&s, "t", "obj", None).await.unwrap();
        let after = store::account_usage(&s, "t", "not-the-goal-id", 50, 1)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.tokens_used, 0);
        let after = store::account_usage(&s, "t", &g.goal_id, 50, 1)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.tokens_used, 50);
    }

    #[tokio::test]
    async fn account_usage_trips_budget_limited() {
        let s = store();
        let g = store::set(&s, "t", "obj", Some(100)).await.unwrap();
        let after = store::account_usage(&s, "t", &g.goal_id, 120, 2)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.status, ThreadGoalStatus::BudgetLimited);
        assert_eq!(after.budget_remaining(), Some(0));
    }

    #[tokio::test]
    async fn pause_resume_complete_transitions() {
        let s = store();
        store::set(&s, "t", "obj", None).await.unwrap();
        assert_eq!(
            store::pause(&s, "t").await.unwrap().status,
            ThreadGoalStatus::Paused
        );
        assert_eq!(
            store::resume(&s, "t").await.unwrap().status,
            ThreadGoalStatus::Active
        );
        let done = store::complete(&s, "t").await.unwrap();
        assert_eq!(done.status, ThreadGoalStatus::Complete);
        // Resume does not reactivate a completed goal.
        assert_eq!(
            store::resume(&s, "t").await.unwrap().status,
            ThreadGoalStatus::Complete
        );
    }

    #[tokio::test]
    async fn mutators_error_without_a_goal() {
        let s = store();
        assert!(store::complete(&s, "missing").await.is_err());
        assert!(store::pause(&s, "missing").await.is_err());
    }

    #[tokio::test]
    async fn empty_objective_and_blank_thread_id_rejected() {
        let s = store();
        assert!(store::set(&s, "t", "   ", None).await.is_err());
        assert!(store::set(&s, "  ", "obj", None).await.is_err());
    }

    #[tokio::test]
    async fn list_all_returns_every_thread_goal() {
        let s = store();
        store::set(&s, "alpha", "a", None).await.unwrap();
        store::set(&s, "beta", "b", None).await.unwrap();
        let mut ids: Vec<String> = store::list_all(&s)
            .await
            .unwrap()
            .into_iter()
            .map(|g| g.thread_id)
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["alpha".to_string(), "beta".to_string()]);
    }
}
