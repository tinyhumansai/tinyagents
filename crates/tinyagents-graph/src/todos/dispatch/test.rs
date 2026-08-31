//! Unit tests for the dispatch policy: selection, approval, cadence, prompts,
//! and the in-flight run registry.

use std::time::Duration;

use serde_json::json;

use super::prompt::{TaskPromptTools, build_progress_instruction, build_task_prompt};
use super::registry::{ActiveRun, ActiveRunRegistry};
use super::select::{
    PollCadence, card_urgency, has_card_in_progress, pick_next_card, requires_plan_approval,
};
use crate::todos::types::{TaskApprovalMode, TaskBoardCard, TaskCardStatus};

fn card(id: &str, status: TaskCardStatus, order: u32) -> TaskBoardCard {
    TaskBoardCard {
        id: id.to_string(),
        status,
        order,
        ..TaskBoardCard::new(id)
    }
}

fn with_urgency(mut card: TaskBoardCard, urgency: f64) -> TaskBoardCard {
    card.source_metadata = Some(json!({ "urgency": urgency }));
    card
}

fn assigned(mut card: TaskBoardCard, agent: &str) -> TaskBoardCard {
    card.assigned_agent = Some(agent.to_string());
    card
}

// ── Selection ───────────────────────────────────────────────────────────────

#[test]
fn only_todo_and_ready_cards_are_dispatchable() {
    let cards = vec![
        card("done", TaskCardStatus::Done, 0),
        card("blocked", TaskCardStatus::Blocked, 1),
        card("awaiting", TaskCardStatus::AwaitingApproval, 2),
        card("rejected", TaskCardStatus::Rejected, 3),
        card("running", TaskCardStatus::InProgress, 4),
    ];
    assert!(pick_next_card(&cards, false).is_none());

    let mut cards = cards;
    cards.push(card("ready", TaskCardStatus::Ready, 5));
    assert_eq!(pick_next_card(&cards, false).unwrap().id, "ready");
}

#[test]
fn the_most_urgent_card_wins() {
    let cards = vec![
        with_urgency(card("low", TaskCardStatus::Todo, 0), 0.1),
        with_urgency(card("high", TaskCardStatus::Todo, 1), 0.9),
        card("none", TaskCardStatus::Todo, 2),
    ];
    assert_eq!(pick_next_card(&cards, false).unwrap().id, "high");
}

#[test]
fn equal_urgency_runs_in_board_order() {
    let cards = vec![
        with_urgency(card("second", TaskCardStatus::Todo, 5), 0.5),
        with_urgency(card("first", TaskCardStatus::Todo, 1), 0.5),
    ];
    assert_eq!(pick_next_card(&cards, false).unwrap().id, "first");

    // Unscored cards tie at 0.0 and follow the same rule.
    let cards = vec![
        card("later", TaskCardStatus::Todo, 9),
        card("earlier", TaskCardStatus::Todo, 2),
    ];
    assert_eq!(pick_next_card(&cards, false).unwrap().id, "earlier");
}

#[test]
fn agent_assigned_only_skips_human_authored_cards() {
    let cards = vec![
        with_urgency(card("mine", TaskCardStatus::Todo, 0), 0.9),
        with_urgency(
            assigned(card("agents", TaskCardStatus::Todo, 1), "researcher"),
            0.1,
        ),
    ];

    // Unfiltered, urgency wins; filtered, the unassigned card is invisible even
    // though it is the more urgent one.
    assert_eq!(pick_next_card(&cards, false).unwrap().id, "mine");
    assert_eq!(pick_next_card(&cards, true).unwrap().id, "agents");
}

#[test]
fn a_blank_assignee_does_not_count_as_assigned() {
    let cards = vec![assigned(card("blank", TaskCardStatus::Todo, 0), "   ")];
    assert!(pick_next_card(&cards, true).is_none());
    assert_eq!(pick_next_card(&cards, false).unwrap().id, "blank");
}

#[test]
fn an_empty_board_has_nothing_to_dispatch() {
    assert!(pick_next_card(&[], false).is_none());
    assert!(!has_card_in_progress(&[]));
}

#[test]
fn in_progress_detection_gates_a_sweep() {
    let idle = vec![card("a", TaskCardStatus::Todo, 0)];
    let busy = vec![card("a", TaskCardStatus::InProgress, 0)];
    assert!(!has_card_in_progress(&idle));
    assert!(has_card_in_progress(&busy));
}

#[test]
fn urgency_defaults_to_zero_for_odd_metadata() {
    assert_eq!(card_urgency(&card("plain", TaskCardStatus::Todo, 0)), 0.0);

    let mut wrong_type = card("odd", TaskCardStatus::Todo, 0);
    wrong_type.source_metadata = Some(json!({ "urgency": "very" }));
    assert_eq!(card_urgency(&wrong_type), 0.0);

    let mut absent = card("absent", TaskCardStatus::Todo, 0);
    absent.source_metadata = Some(json!({ "provider": "github" }));
    assert_eq!(card_urgency(&absent), 0.0);
}

// ── Approval ────────────────────────────────────────────────────────────────

#[test]
fn a_cards_own_approval_mode_outranks_the_global_default() {
    // Required holds even when the global switch is off — otherwise a plan
    // stamped for review would execute before anyone saw it.
    assert!(requires_plan_approval(
        false,
        Some(&TaskApprovalMode::Required)
    ));
    assert!(requires_plan_approval(
        true,
        Some(&TaskApprovalMode::Required)
    ));

    // NotRequired means review already happened.
    assert!(!requires_plan_approval(
        true,
        Some(&TaskApprovalMode::NotRequired)
    ));
    assert!(!requires_plan_approval(
        false,
        Some(&TaskApprovalMode::NotRequired)
    ));
}

#[test]
fn a_card_with_no_preference_follows_the_global_default() {
    assert!(requires_plan_approval(true, None));
    assert!(!requires_plan_approval(false, None));
}

// ── Cadence ─────────────────────────────────────────────────────────────────

#[test]
fn cadence_holds_the_base_interval_through_the_grace_window() {
    let cadence = PollCadence::default();
    assert_eq!(cadence.next_delay(0), cadence.base);
    assert_eq!(cadence.next_delay(cadence.grace_ticks), cadence.base);
}

#[test]
fn cadence_doubles_past_the_grace_window() {
    let cadence = PollCadence::default();
    let base = cadence.base.as_secs();
    assert_eq!(
        cadence.next_delay(cadence.grace_ticks + 1),
        Duration::from_secs(base * 2)
    );
    assert_eq!(
        cadence.next_delay(cadence.grace_ticks + 2),
        Duration::from_secs(base * 4)
    );
    assert_eq!(
        cadence.next_delay(cadence.grace_ticks + 3),
        Duration::from_secs(base * 8)
    );
}

#[test]
fn cadence_is_monotonic_and_never_exceeds_its_ceiling() {
    let cadence = PollCadence::default();
    assert_eq!(cadence.next_delay(50), cadence.max_backoff);
    // A long idle streak saturates rather than overflowing back to a tiny delay.
    assert_eq!(cadence.next_delay(u32::MAX), cadence.max_backoff);

    let mut previous = cadence.next_delay(0);
    for idle in 1..40u32 {
        let delay = cadence.next_delay(idle);
        assert!(delay >= previous, "backoff must not shrink as idle grows");
        assert!(
            delay <= cadence.max_backoff,
            "backoff must not exceed the ceiling"
        );
        previous = delay;
    }
}

#[test]
fn a_zero_grace_cadence_backs_off_from_the_first_idle_tick() {
    let cadence = PollCadence {
        base: Duration::from_secs(10),
        max_backoff: Duration::from_secs(40),
        grace_ticks: 0,
    };
    assert_eq!(cadence.next_delay(0), Duration::from_secs(10));
    assert_eq!(cadence.next_delay(1), Duration::from_secs(20));
    assert_eq!(cadence.next_delay(2), Duration::from_secs(40));
    assert_eq!(cadence.next_delay(3), Duration::from_secs(40));
}

// ── Prompts ─────────────────────────────────────────────────────────────────

#[test]
fn the_prompt_leads_with_the_objective_and_falls_back_to_the_title() {
    let mut card = TaskBoardCard::new("Fix the flaky test");
    let tools = TaskPromptTools::default();

    let prompt = build_task_prompt(&card, &tools);
    assert!(prompt.contains("Fix the flaky test"), "{prompt}");

    card.objective = Some("Make CI green on main".to_string());
    let prompt = build_task_prompt(&card, &tools);
    assert!(prompt.contains("Make CI green on main"), "{prompt}");

    // A whitespace-only objective is not an objective.
    card.objective = Some("   ".to_string());
    assert!(build_task_prompt(&card, &tools).contains("Fix the flaky test"));
}

#[test]
fn the_prompt_numbers_plan_steps_and_lists_acceptance_criteria() {
    let mut card = TaskBoardCard::new("Ship it");
    card.plan = vec!["Reproduce".to_string(), "Fix".to_string()];
    card.acceptance_criteria = vec!["CI is green".to_string()];

    let prompt = build_task_prompt(&card, &TaskPromptTools::default());
    assert!(prompt.contains("1. Reproduce"), "{prompt}");
    assert!(prompt.contains("2. Fix"), "{prompt}");
    assert!(prompt.contains("- CI is green"), "{prompt}");
    assert!(prompt.contains("Acceptance criteria"), "{prompt}");
}

#[test]
fn a_sourced_card_gets_provenance_and_a_write_back_instruction() {
    let mut card = TaskBoardCard::new("Triage issue");
    card.source_metadata = Some(json!({
        "provider": "github",
        "repo": "tinyhumansai/tinyagents",
        "external_id": "412",
        "url": "https://example.invalid/412",
    }));

    let prompt = build_task_prompt(&card, &TaskPromptTools::default());
    assert!(
        prompt.contains("github tinyhumansai/tinyagents#412"),
        "{prompt}"
    );
    assert!(prompt.contains("memory_recall"), "{prompt}");
    assert!(prompt.contains("https://example.invalid/412"), "{prompt}");
    assert!(
        prompt.contains("record the outcome on the upstream source"),
        "{prompt}"
    );
}

#[test]
fn an_id_only_card_gets_no_provenance_line() {
    // Without a provider the origin would render as a bare "#7", which tells
    // the model nothing — so the whole block is skipped.
    let mut card = TaskBoardCard::new("Mystery task");
    card.source_metadata = Some(json!({ "external_id": "7" }));

    let prompt = build_task_prompt(&card, &TaskPromptTools::default());
    assert!(!prompt.contains("#7"), "{prompt}");
    assert!(!prompt.contains("originates from"), "{prompt}");
}

#[test]
fn a_host_without_a_memory_tool_still_gets_provenance() {
    let mut card = TaskBoardCard::new("Triage issue");
    card.source_metadata = Some(json!({ "provider": "linear", "external_id": "ENG-1" }));
    let tools = TaskPromptTools {
        memory_recall: None,
        ..TaskPromptTools::default()
    };

    let prompt = build_task_prompt(&card, &tools);
    assert!(prompt.contains("originates from linear#ENG-1"), "{prompt}");
    assert!(!prompt.contains("memory_recall"), "{prompt}");
}

#[test]
fn the_progress_instruction_names_the_card_board_and_tool() {
    let tools = TaskPromptTools {
        update_task: "board_update".to_string(),
        ..TaskPromptTools::default()
    };
    let instruction = build_progress_instruction("task-9", "user-tasks", &tools);

    assert!(instruction.contains("task-9"), "{instruction}");
    assert!(instruction.contains("user-tasks"), "{instruction}");
    assert!(instruction.contains("board_update"), "{instruction}");
    // Blocking is the sanctioned way out, and it must be spelled out.
    assert!(instruction.contains("status: blocked"), "{instruction}");
    assert!(instruction.contains("do NOT guess"), "{instruction}");
}

// ── Registry ────────────────────────────────────────────────────────────────

fn spawn_pending() -> tokio::task::JoinHandle<()> {
    tokio::spawn(async { std::future::pending::<()>().await })
}

async fn active_run(
    run_id: &str,
    card_id: &str,
) -> (ActiveRun<&'static str>, tokio::task::JoinHandle<()>) {
    let handle = spawn_pending();
    let (heartbeat_cancel, _rx) = tokio::sync::watch::channel(false);
    (
        ActiveRun {
            run_id: run_id.to_string(),
            card_id: card_id.to_string(),
            abort: handle.abort_handle(),
            heartbeat_cancel,
            context: "board",
        },
        handle,
    )
}

#[tokio::test]
async fn only_one_taker_owns_a_runs_cleanup() {
    let registry = ActiveRunRegistry::new();
    let (run, _handle) = active_run("run-1", "task-1").await;
    assert!(registry.register("thread-1", run).is_none());
    assert!(registry.contains("thread-1"));
    assert_eq!(registry.len(), 1);

    // The natural completion and a concurrent cancel both call `take`; exactly
    // one gets the entry, so the terminal write-back happens once.
    let first = registry.take("thread-1");
    let second = registry.take("thread-1");
    assert!(first.is_some());
    assert!(second.is_none());
    assert!(registry.is_empty());
}

#[tokio::test]
async fn a_scoped_cancel_ignores_a_superseded_run() {
    let registry = ActiveRunRegistry::new();
    let (run, _handle) = active_run("run-new", "task-1").await;
    registry.register("thread-1", run);

    // A cancel for a run that has already been replaced must not tear down the
    // run that took its place.
    assert!(registry.take_if("thread-1", Some("run-old")).is_none());
    assert!(registry.contains("thread-1"));

    let taken = registry
        .take_if("thread-1", Some("run-new"))
        .expect("matching run");
    assert_eq!(taken.run_id, "run-new");
}

#[tokio::test]
async fn an_unscoped_cancel_takes_whatever_is_running() {
    let registry = ActiveRunRegistry::new();
    let (run, _handle) = active_run("run-1", "task-1").await;
    registry.register("thread-1", run);

    assert!(registry.take_if("thread-1", None).is_some());
    // And on an idle thread it is simply a no-op.
    assert!(registry.take_if("thread-1", None).is_none());
    assert!(registry.take_if("unknown-thread", Some("run-1")).is_none());
}

#[tokio::test]
async fn registering_over_a_live_run_hands_back_the_displaced_one() {
    let registry = ActiveRunRegistry::new();
    let (first, _h1) = active_run("run-1", "task-1").await;
    let (second, _h2) = active_run("run-2", "task-2").await;
    registry.register("thread-1", first);

    let displaced = registry
        .register("thread-1", second)
        .expect("displaced run");
    assert_eq!(displaced.run_id, "run-1");
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.thread_ids(), vec!["thread-1".to_string()]);
}

#[tokio::test]
async fn cancelling_aborts_the_task_and_stops_the_heartbeat() {
    let registry = ActiveRunRegistry::new();
    let handle = spawn_pending();
    let (heartbeat_cancel, mut heartbeat_rx) = tokio::sync::watch::channel(false);
    registry.register(
        "thread-1",
        ActiveRun {
            run_id: "run-1".to_string(),
            card_id: "task-1".to_string(),
            abort: handle.abort_handle(),
            heartbeat_cancel,
            context: (),
        },
    );

    let run = registry.take("thread-1").expect("live run");
    run.cancel();

    assert!(handle.await.unwrap_err().is_cancelled());
    assert!(heartbeat_rx.changed().await.is_ok());
    assert!(*heartbeat_rx.borrow());
}

#[tokio::test]
async fn draining_returns_every_live_run() {
    let registry = ActiveRunRegistry::new();
    let (first, _h1) = active_run("run-1", "task-1").await;
    let (second, _h2) = active_run("run-2", "task-2").await;
    registry.register("thread-1", first);
    registry.register("thread-2", second);

    let mut drained: Vec<String> = registry.drain().into_iter().map(|(id, _)| id).collect();
    drained.sort();
    assert_eq!(
        drained,
        vec!["thread-1".to_string(), "thread-2".to_string()]
    );
    assert!(registry.is_empty());
}
