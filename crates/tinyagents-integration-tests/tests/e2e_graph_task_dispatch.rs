//! End-to-end coverage for autonomous task dispatch, assembled from the public
//! crate surface exactly as a host would assemble it:
//!
//! ```text
//! pick_next_card → requires_plan_approval → claim_card → create_run
//!   → build_task_prompt → agent turn → complete_run → card write-back
//! ```
//!
//! The agent side is a `MockModel` driving the real `todo` tool, so the card a
//! run reports on is the same card the dispatcher claimed. What these tests
//! pin down is the *loop*: the right card is picked, an unapproved plan never
//! runs, a claimed card is invisible to the next tick, a cancelled run does not
//! strand its card `in_progress`, and an abandoned worker's card comes back.

use std::sync::Arc;

use serde_json::json;

use tinyagents_graph::*;
use tinyagents_harness::context::RunConfig;
use tinyagents_harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents_harness::model::ModelResponse;
use tinyagents_harness::providers::MockModel;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::store::{InMemoryStore, Store};
use tinyagents_harness::tool::ToolCall;
use tinyagents_harness::usage::Usage;
use tinyagents_harness::*;
use tinyagents_language::*;
use tinyagents_registry::*;

const THREAD: &str = "user-tasks";

fn tool_call_response(id: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, "todo", arguments)],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
    }
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, 2)),
        },
        usage: Some(Usage::new(4, 2)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
    }
}

/// Add a card, returning its id.
async fn add_card(store: &Arc<dyn Store>, title: &str, patch: CardPatch) -> String {
    let snapshot = todo_store::add(store, THREAD, title, patch)
        .await
        .expect("add card");
    snapshot
        .cards
        .iter()
        .find(|card| card.title == title)
        .expect("card present")
        .id
        .clone()
}

fn agent_card(agent: &str, urgency: f64) -> CardPatch {
    CardPatch {
        assigned_agent: Some(agent.to_string()),
        source_metadata: Some(json!({ "urgency": urgency })),
        ..CardPatch::default()
    }
}

/// What one dispatcher tick decided to do.
#[derive(Debug, PartialEq)]
enum Tick {
    /// Nothing to claim: the board is empty, busy, or holds only work this
    /// dispatcher may not run.
    Idle,
    /// The card was parked for a human to approve its plan.
    Parked(String),
    /// The card was claimed and a run opened for it.
    Dispatched { card_id: String, run_id: String },
}

/// One sweep of the board, wired from the crate's dispatch policy: reclaim
/// what has gone stale, refuse to double-book a busy board, pick the most
/// urgent agent-assigned card, and either park it for approval or claim it.
async fn tick(store: &Arc<dyn Store>, approval_required: bool) -> Tick {
    task_run_store::reclaim_stale(store, THREAD, &RunLimits::default())
        .await
        .expect("sweep");

    let board = todo_store::list(store, THREAD).await.expect("list board");
    if has_card_in_progress(&board.cards) {
        return Tick::Idle;
    }
    let Some(card) = pick_next_card(&board.cards, true) else {
        return Tick::Idle;
    };

    if card.status == TaskCardStatus::Todo
        && requires_plan_approval(approval_required, card.approval_mode.as_ref())
    {
        todo_store::update_status(store, THREAD, &card.id, TaskCardStatus::AwaitingApproval)
            .await
            .expect("park for approval");
        return Tick::Parked(card.id);
    }

    todo_store::claim_card(
        store,
        THREAD,
        &card.id,
        &[TaskCardStatus::Todo, TaskCardStatus::Ready],
        TaskCardStatus::InProgress,
    )
    .await
    .expect("claim card");
    let run = task_run_store::create_run(store, THREAD, None, &card.id, "dispatcher")
        .await
        .expect("open run");
    Tick::Dispatched {
        card_id: card.id,
        run_id: run.run_id,
    }
}

/// Run one card through a mock agent that marks it done through the `todo`
/// tool, then close the run. Returns the prompt the agent was given.
async fn run_card(store: &Arc<dyn Store>, card: &TaskBoardCard, run_id: &str) -> String {
    let tools = TaskPromptTools::default();
    let prompt = format!(
        "{}{}",
        build_task_prompt(card, &tools),
        build_progress_instruction(&card.id, THREAD, &tools)
    );

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response(
                    "call-1",
                    json!({
                        "op": "edit",
                        "id": card.id,
                        "evidence": ["ran the migration"],
                    }),
                ),
                tool_call_response(
                    "call-2",
                    json!({ "op": "update_status", "id": card.id, "status": "done" }),
                ),
                text_response("migration applied"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(TodoTool::new(store.clone())));

    let outcome = harness
        .invoke(
            &(),
            (),
            RunConfig::new(run_id).with_thread(THREAD),
            vec![Message::user(prompt.clone())],
        )
        .await
        .expect("agent run succeeds");
    assert_eq!(outcome.tool_calls, 2);

    task_run_store::complete_run(
        store,
        THREAD,
        run_id,
        RunOutcome::Success,
        None,
        vec!["migration applied".to_string()],
    )
    .await
    .expect("close run");

    prompt
}

#[tokio::test]
async fn the_dispatcher_runs_the_most_urgent_agent_card_and_leaves_human_work_alone() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());

    // A human's own todo (unassigned, most urgent), and two agent-assigned
    // cards. Only the agent cards are the dispatcher's to run.
    let mine = add_card(
        &store,
        "call the dentist",
        CardPatch {
            source_metadata: Some(json!({ "urgency": 0.99 })),
            ..CardPatch::default()
        },
    )
    .await;
    let low = add_card(&store, "tidy the changelog", agent_card("scribe", 0.1)).await;
    let high = add_card(&store, "apply the migration", agent_card("dba", 0.8)).await;

    // Tick 1: the urgent agent card is claimed; the human's card is not touched.
    let Tick::Dispatched { card_id, run_id } = tick(&store, false).await else {
        panic!("expected a dispatch");
    };
    assert_eq!(card_id, high);

    // While it runs the board is busy, so the next tick claims nothing —
    // the single-`in_progress` rule holds across ticks, not just within one.
    assert_eq!(tick(&store, false).await, Tick::Idle);

    let board = todo_store::list(&store, THREAD).await.expect("board");
    let card = board
        .cards
        .iter()
        .find(|card| card.id == card_id)
        .expect("claimed card")
        .clone();
    let prompt = run_card(&store, &card, &run_id).await;
    assert!(prompt.contains("apply the migration"), "{prompt}");
    assert!(
        prompt.contains(&card_id),
        "the run is told which card it owns"
    );

    // Tick 2: with the first card done, the remaining agent card goes next.
    let Tick::Dispatched { card_id, .. } = tick(&store, false).await else {
        panic!("expected the second agent card");
    };
    assert_eq!(card_id, low);

    let board = todo_store::list(&store, THREAD).await.expect("board");
    let statuses: Vec<_> = board
        .cards
        .iter()
        .map(|card| (card.id.clone(), card.status))
        .collect();
    assert!(statuses.contains(&(mine.clone(), TaskCardStatus::Todo)));
    assert!(statuses.contains(&(high, TaskCardStatus::Done)));
    assert!(statuses.contains(&(low, TaskCardStatus::InProgress)));

    // The evidence the run reported is on the card it was working.
    let done = board
        .cards
        .iter()
        .find(|c| c.status == TaskCardStatus::Done)
        .unwrap();
    assert_eq!(done.evidence, vec!["ran the migration".to_string()]);
}

#[tokio::test]
async fn a_plan_awaiting_approval_never_runs_until_it_is_approved() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());
    let card_id = add_card(&store, "delete the old bucket", agent_card("ops", 0.5)).await;

    // With approval on, the tick parks the card instead of claiming it, and
    // keeps parking nothing afterwards: an awaiting card is not dispatchable.
    assert_eq!(tick(&store, true).await, Tick::Parked(card_id.clone()));
    assert_eq!(tick(&store, true).await, Tick::Idle);
    assert!(
        task_run_store::list_runs(&store, THREAD, None)
            .await
            .expect("runs")
            .is_empty(),
        "no run is opened for an unapproved plan"
    );

    // A human approves it; now the same tick claims it.
    todo_store::decide_plan(&store, THREAD, &card_id, true)
        .await
        .expect("approve plan");
    let Tick::Dispatched {
        card_id: claimed, ..
    } = tick(&store, true).await
    else {
        panic!("an approved plan runs");
    };
    assert_eq!(claimed, card_id);
}

#[tokio::test]
async fn a_card_stamped_required_is_parked_even_with_the_global_gate_off() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());
    let card_id = add_card(
        &store,
        "email the customer",
        CardPatch {
            approval_mode: Some(Some(TaskApprovalMode::Required)),
            ..agent_card("support", 0.5)
        },
    )
    .await;

    // The card's own stamp outranks the global default — an interactive plan
    // review must hold regardless of how the host is configured.
    assert_eq!(tick(&store, false).await, Tick::Parked(card_id));
}

#[tokio::test]
async fn a_cancelled_run_leaves_its_card_blocked_rather_than_stranded() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());
    let card_id = add_card(&store, "long crawl", agent_card("crawler", 0.5)).await;
    let registry: ActiveRunRegistry<String> = ActiveRunRegistry::new();

    let Tick::Dispatched { run_id, .. } = tick(&store, false).await else {
        panic!("expected a dispatch");
    };

    // The run is a detached task; the registry is how a cancel reaches it.
    let work = tokio::spawn(async { std::future::pending::<()>().await });
    let (heartbeat_cancel, mut heartbeat_rx) = tokio::sync::watch::channel(false);
    registry.register(
        THREAD,
        ActiveRun {
            run_id: run_id.clone(),
            card_id: card_id.clone(),
            abort: work.abort_handle(),
            heartbeat_cancel,
            context: THREAD.to_string(),
        },
    );

    // A cancel for some *other* run must not tear this one down.
    assert!(
        registry
            .take_if(THREAD, Some("run-from-a-previous-request"))
            .is_none()
    );

    let active = registry
        .take_if(THREAD, Some(&run_id))
        .expect("the live run");
    active.cancel();
    assert!(work.await.unwrap_err().is_cancelled());
    assert!(heartbeat_rx.changed().await.is_ok());

    // The aborted task never reaches its own write-back, so the canceller owns
    // it: close the run and park the card, rather than leaving it in progress.
    task_run_store::complete_run(
        &store,
        THREAD,
        &run_id,
        RunOutcome::Failed,
        Some("cancelled by user".to_string()),
        vec![],
    )
    .await
    .expect("close run");
    todo_store::edit(
        &store,
        THREAD,
        &card_id,
        CardPatch {
            status: Some(TaskCardStatus::Blocked),
            blocker: Some("cancelled by user".to_string()),
            ..CardPatch::default()
        },
    )
    .await
    .expect("park card");

    let board = todo_store::list(&store, THREAD).await.expect("board");
    assert_eq!(board.cards[0].status, TaskCardStatus::Blocked);
    assert_eq!(board.cards[0].blocker.as_deref(), Some("cancelled by user"));
    assert!(registry.is_empty());

    // And the board is free again: a later tick is not blocked by a ghost.
    assert_eq!(
        tick(&store, false).await,
        Tick::Idle,
        "a blocked card is not re-run"
    );
}

#[tokio::test]
async fn an_abandoned_run_is_reclaimed_by_the_next_tick() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());
    let card_id = add_card(&store, "flaky job", agent_card("runner", 0.5)).await;

    let Tick::Dispatched { run_id, .. } = tick(&store, false).await else {
        panic!("expected a dispatch");
    };

    // The worker vanishes: age its stamps so the next sweep judges it dead.
    let mut runs = task_run_store::list_runs(&store, THREAD, None)
        .await
        .expect("runs");
    for run in runs.iter_mut() {
        run.started_at = "0".to_string();
        run.last_heartbeat_at = "0".to_string();
    }
    let key: String = THREAD
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    store
        .put(
            task_run_store::RUNS_NAMESPACE,
            &key,
            serde_json::to_value(&runs).expect("serialize"),
        )
        .await
        .expect("write runs");

    // The next tick reclaims the dead run and re-dispatches the card in the
    // same sweep — recovery needs no operator intervention.
    let Tick::Dispatched {
        card_id: reclaimed,
        run_id: fresh,
    } = tick(&store, false).await
    else {
        panic!("expected a re-dispatch");
    };
    assert_eq!(reclaimed, card_id);
    assert_ne!(fresh, run_id, "a new claim, not the dead one");

    let history = task_run_store::list_runs(&store, THREAD, Some(&card_id))
        .await
        .expect("history");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].outcome, Some(RunOutcome::Reclaimed));
    assert!(history[1].is_active());
}

#[tokio::test]
async fn an_idle_board_backs_the_sweep_off_and_fresh_work_resets_it() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());
    let cadence = PollCadence::default();

    // Nothing to do: the tick is idle and the interval starts stretching once
    // the grace window is used up.
    let mut idle_ticks = 0u32;
    for _ in 0..6 {
        if tick(&store, false).await == Tick::Idle {
            idle_ticks += 1;
        }
    }
    assert_eq!(idle_ticks, 6);
    assert!(
        cadence.next_delay(idle_ticks) > cadence.base,
        "a persistently empty board is not swept at full rate forever"
    );

    // Work arrives, the tick dispatches, and the cadence snaps back.
    add_card(&store, "new work", agent_card("worker", 0.5)).await;
    assert!(matches!(tick(&store, false).await, Tick::Dispatched { .. }));
    idle_ticks = 0;
    assert_eq!(cadence.next_delay(idle_ticks), cadence.base);
}
