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
