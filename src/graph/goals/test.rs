//! Unit tests for the thread-goal domain types.

use super::prompt::*;
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

mod tool_tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::super::tool::{GoalTool, GoalToolKind, goal_tools};
    use crate::harness::events::EventSink;
    use crate::harness::ids::{RunId, ThreadId};
    use crate::harness::store::{InMemoryStore, Store};
    use crate::harness::tool::{Tool, ToolCall, ToolExecutionContext};

    fn store() -> Arc<dyn Store> {
        Arc::new(InMemoryStore::default())
    }

    fn ctx(thread_id: Option<&str>) -> ToolExecutionContext {
        ToolExecutionContext {
            run_id: RunId::new("run-1"),
            thread_id: thread_id.map(ThreadId::new),
            depth: 0,
            max_turn_output_tokens: None,
            events: EventSink::new(),
            streaming: false,
            workspace: None,
        }
    }

    fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: args,
        }
    }

    #[test]
    fn goal_tools_builds_the_model_facing_set() {
        let tools = goal_tools(store());
        let names: Vec<&str> = tools.iter().map(|t| Tool::<()>::name(t.as_ref())).collect();
        assert_eq!(names, vec!["goal_get", "goal_set", "goal_complete"]);
    }

    #[tokio::test]
    async fn set_get_complete_via_tools_in_thread_scope() {
        let s = store();
        let set = GoalTool::new(GoalToolKind::Set, s.clone());
        let res = Tool::<()>::call_with_context(
            &set,
            &(),
            call(
                "c1",
                "goal_set",
                json!({ "objective": "land the PR", "token_budget": 5000 }),
            ),
            ctx(Some("thread-tools")),
        )
        .await
        .unwrap();
        assert!(res.error.is_none(), "{res:?}");
        assert!(res.content.contains("land the PR"));

        let get = GoalTool::new(GoalToolKind::Get, s.clone());
        let res = Tool::<()>::call_with_context(
            &get,
            &(),
            call("c2", "goal_get", json!({})),
            ctx(Some("thread-tools")),
        )
        .await
        .unwrap();
        assert!(res.content.contains("status: active"));

        let done = GoalTool::new(GoalToolKind::Complete, s.clone());
        let res = Tool::<()>::call_with_context(
            &done,
            &(),
            call("c3", "goal_complete", json!({})),
            ctx(Some("thread-tools")),
        )
        .await
        .unwrap();
        assert!(res.content.contains("status: complete"));
    }

    #[tokio::test]
    async fn tools_error_without_thread_scope() {
        let s = store();
        let set = GoalTool::new(GoalToolKind::Set, s);
        // Bare call (no context) errors.
        let res = Tool::<()>::call(
            &set,
            &(),
            call("c1", "goal_set", json!({ "objective": "x" })),
        )
        .await
        .unwrap();
        assert!(res.error.is_some());
        assert!(res.error.unwrap().contains("active thread"));
        // Context with no thread id also errors.
        let set = GoalTool::new(GoalToolKind::Set, store());
        let res = Tool::<()>::call_with_context(
            &set,
            &(),
            call("c2", "goal_set", json!({ "objective": "x" })),
            ctx(None),
        )
        .await
        .unwrap();
        assert!(res.error.is_some());
    }

    #[tokio::test]
    async fn get_reports_absent_goal() {
        let get = GoalTool::new(GoalToolKind::Get, store());
        let res = Tool::<()>::call_with_context(
            &get,
            &(),
            call("c1", "goal_get", json!({})),
            ctx(Some("empty-thread")),
        )
        .await
        .unwrap();
        assert!(res.content.contains("no goal set"));
    }

    #[tokio::test]
    async fn set_missing_objective_is_a_soft_error() {
        let set = GoalTool::new(GoalToolKind::Set, store());
        let res = Tool::<()>::call_with_context(
            &set,
            &(),
            call("c1", "goal_set", json!({})),
            ctx(Some("t")),
        )
        .await
        .unwrap();
        assert!(res.content.contains("missing 'objective'"));
    }
}

mod continuation_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::super::store;
    use super::super::types::{GoalProgress, ThreadGoalStatus, TurnOutcome};
    use super::super::{goal_gate_node, note_user_turn, run_continuation_tick};
    use crate::error::TinyAgentsError;
    use crate::graph::GraphBuilder;
    use crate::graph::command::NodeResult;
    use crate::harness::store::{InMemoryStore, Store};

    fn store() -> Arc<dyn Store> {
        Arc::new(InMemoryStore::default())
    }

    /// State overwritten by each work iteration: the tokens it "spent" and
    /// whether it made progress. The gate's `progress` closure reads these.
    #[derive(Clone, Debug, Default, PartialEq)]
    struct GateState {
        iters: usize,
        tokens: u64,
        progress: bool,
    }

    /// Builds and runs `work_node -> gate` under `thread_id`, looping until the
    /// gate routes to END or the recursion limit trips. `work` is the per-iteration
    /// behavior producing the next [`GateState`].
    async fn run_gate<W, Fut>(
        s: &Arc<dyn Store>,
        thread_id: &str,
        recursion_limit: usize,
        per_iter_tokens: u64,
        made_progress: bool,
        work: W,
    ) -> crate::error::Result<GateState>
    where
        W: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let counter = Arc::new(AtomicUsize::new(0));
        let work = Arc::new(work);
        let work_node = {
            let counter = counter.clone();
            let work = work.clone();
            move |_state: GateState, _ctx| {
                let counter = counter.clone();
                let work = work.clone();
                Box::pin(async move {
                    let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    work(n).await;
                    Ok(NodeResult::Update(GateState {
                        iters: n,
                        tokens: per_iter_tokens,
                        progress: made_progress,
                    }))
                }) as crate::graph::NodeFuture<GateState>
            }
        };

        let gate = goal_gate_node::<GateState, GateState>(s.clone(), "work", |st: &GateState| {
            GoalProgress {
                tokens_used: st.tokens,
                elapsed_secs: 0,
                made_progress: st.progress,
            }
        });

        let graph = GraphBuilder::<GateState, GateState>::overwrite()
            .with_recursion_limit(recursion_limit)
            .add_node("work", work_node)
            .add_node("gate", gate)
            .set_entry("work")
            .add_edge("work", "gate")
            .with_command_destinations("gate", ["work", crate::graph::END])
            .compile()?;

        let exec = graph
            .run_with_thread(thread_id, GateState::default())
            .await?;
        Ok(exec.state)
    }

    #[tokio::test]
    async fn gate_loops_while_active_then_stops_on_complete() {
        let s = store();
        store::set(&s, "t", "obj", None).await.unwrap();
        let s2 = s.clone();
        let final_state = run_gate(&s, "t", 100, 10, true, move |n| {
            let s2 = s2.clone();
            async move {
                if n >= 3 {
                    store::complete(&s2, "t").await.unwrap();
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(
            final_state.iters, 3,
            "runs the work node until goal completes"
        );
        let goal = store::get(&s, "t").await.unwrap().unwrap();
        assert_eq!(goal.status, ThreadGoalStatus::Complete);
    }

    #[tokio::test]
    async fn gate_stops_on_budget_cap() {
        let s = store();
        store::set(&s, "t", "obj", Some(100)).await.unwrap();
        let final_state = run_gate(&s, "t", 100, 40, true, |_n| async {})
            .await
            .unwrap();
        assert_eq!(final_state.iters, 3, "40*3 = 120 crosses the 100 budget");
        let goal = store::get(&s, "t").await.unwrap().unwrap();
        assert_eq!(goal.status, ThreadGoalStatus::BudgetLimited);
        assert_eq!(goal.tokens_used, 120);
    }

    #[tokio::test]
    async fn gate_stops_and_suppresses_on_zero_progress() {
        let s = store();
        store::set(&s, "t", "obj", None).await.unwrap();
        let final_state = run_gate(&s, "t", 100, 0, false, |_n| async {})
            .await
            .unwrap();
        assert_eq!(
            final_state.iters, 1,
            "a no-progress iteration stops the loop"
        );
        let goal = store::get(&s, "t").await.unwrap().unwrap();
        assert!(goal.continuation_suppressed, "one-shot suppression set");
    }

    #[tokio::test]
    async fn gate_recursion_limit_is_the_backstop() {
        let s = store();
        store::set(&s, "t", "obj", None).await.unwrap();
        // Never completes, never stalls, no budget → the loop only stops at the
        // recursion limit (work+gate = 2 supersteps per iteration).
        let err = run_gate(&s, "t", 6, 5, true, |_n| async {})
            .await
            .unwrap_err();
        assert!(matches!(err, TinyAgentsError::RecursionLimit(_)));
    }

    #[tokio::test]
    async fn gate_stops_when_thread_has_no_goal() {
        let s = store();
        // No goal set for the thread → gate routes straight to END after one work run.
        let final_state = run_gate(&s, "t", 100, 5, true, |_n| async {})
            .await
            .unwrap();
        assert_eq!(final_state.iters, 1);
    }

    #[tokio::test]
    async fn driver_runs_up_to_max_per_tick() {
        let s = store();
        store::set(&s, "a", "a", None).await.unwrap();
        store::set(&s, "b", "b", None).await.unwrap();
        store::set(&s, "c", "c", None).await.unwrap();
        let ran = run_continuation_tick(&s, Duration::ZERO, 2, |_g| async {
            Ok(TurnOutcome {
                tokens_used: 5,
                elapsed_secs: 0,
                made_progress: true,
            })
        })
        .await
        .unwrap();
        assert_eq!(ran, 2, "capped at max_per_tick");
    }

    #[tokio::test]
    async fn driver_skips_non_idle_goals() {
        let s = store();
        store::set(&s, "fresh", "obj", None).await.unwrap();
        // A 1-hour idle window over a just-created goal → nothing runs.
        let ran = run_continuation_tick(&s, Duration::from_secs(3600), 5, |_g| async {
            Ok(TurnOutcome::default())
        })
        .await
        .unwrap();
        assert_eq!(ran, 0);
    }

    #[tokio::test]
    async fn driver_suppresses_after_a_no_progress_turn() {
        let s = store();
        store::set(&s, "t", "obj", None).await.unwrap();
        let ran = run_continuation_tick(&s, Duration::ZERO, 5, |_g| async {
            Ok(TurnOutcome {
                tokens_used: 3,
                elapsed_secs: 0,
                made_progress: false,
            })
        })
        .await
        .unwrap();
        assert_eq!(ran, 1);
        assert!(
            store::get(&s, "t")
                .await
                .unwrap()
                .unwrap()
                .continuation_suppressed
        );
        // A second tick finds the goal suppressed and runs nothing.
        let ran2 = run_continuation_tick(&s, Duration::ZERO, 5, |_g| async {
            Ok(TurnOutcome::default())
        })
        .await
        .unwrap();
        assert_eq!(ran2, 0);
    }

    #[tokio::test]
    async fn note_user_turn_clears_suppression_and_resumes() {
        let s = store();
        let g = store::set(&s, "t", "obj", None).await.unwrap();
        // Suppress, then a user turn clears it.
        store::set_continuation_suppressed_if(&s, "t", &g.goal_id, true)
            .await
            .unwrap();
        let after = note_user_turn(&s, "t").await.unwrap().unwrap();
        assert!(!after.continuation_suppressed);

        // A paused goal is reactivated by a user turn.
        store::pause(&s, "t").await.unwrap();
        let after = note_user_turn(&s, "t").await.unwrap().unwrap();
        assert_eq!(after.status, ThreadGoalStatus::Active);

        // No goal → None.
        assert!(note_user_turn(&s, "missing").await.unwrap().is_none());
    }
}
