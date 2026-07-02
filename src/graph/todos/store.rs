//! CRUD for the per-thread task board, on the harness
//! [`Store`](crate::harness::store::Store).
//!
//! Each thread's board is a single serialized [`TaskBoard`] value under the
//! [`TODOS_NAMESPACE`] namespace, keyed by the hex-encoded thread id. Every
//! mutation runs `load → mutate → normalise → put` under a **per-thread async
//! mutex** ([`thread_lock`]) so the read-modify-write is atomic within the
//! process (the same single-process caveat as
//! [`graph::goals::store`](crate::graph::goals::store)).
//!
//! Each mutator returns a [`TodosSnapshot`] — the normalised cards plus a
//! markdown rendering — so an agent transcript and a UI stay in lock-step.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use tokio::sync::Mutex;

use super::types::{
    CardPatch, TaskBoard, TaskBoardCard, TaskCardStatus, TodosSnapshot, non_empty, normalise_board,
    now_stamp, render_markdown,
};
use crate::error::{Result, TinyAgentsError};
use crate::harness::store::Store;

/// The [`Store`] namespace holding one [`TaskBoard`] per thread.
pub const TODOS_NAMESPACE: &str = "graph.todos";

/// Serialises `load → mutate → put` per thread so a read-modify-write is atomic
/// within the process.
fn thread_lock(thread_id: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<StdMutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    let map = LOCKS.get_or_init(|| StdMutex::new(HashMap::new()));
    let mut guard = map.lock().expect("todo lock map poisoned");
    guard.entry(thread_id.to_string()).or_default().clone()
}

/// Hex-encodes the thread id into a [`Store`]-safe key.
fn key(thread_id: &str) -> String {
    thread_id
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn validate_thread_id(thread_id: &str) -> Result<String> {
    let trimmed = thread_id.trim();
    if trimmed.is_empty() {
        return Err(TinyAgentsError::Validation(
            "task board thread_id must not be empty or whitespace".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

/// Loads the raw cards for `thread_id` (empty when the thread has no board).
async fn load_cards(store: &Arc<dyn Store>, thread_id: &str) -> Result<Vec<TaskBoardCard>> {
    match store.get(TODOS_NAMESPACE, &key(thread_id)).await? {
        Some(value) => {
            let board: TaskBoard = serde_json::from_value(value)?;
            Ok(board.cards)
        }
        None => Ok(Vec::new()),
    }
}

/// Normalises and persists `cards` for `thread_id`, returning the normalised set.
async fn save_cards(
    store: &Arc<dyn Store>,
    thread_id: &str,
    cards: Vec<TaskBoardCard>,
) -> Result<Vec<TaskBoardCard>> {
    let mut board = TaskBoard {
        thread_id: thread_id.to_string(),
        cards,
        updated_at: now_stamp(),
    };
    normalise_board(&mut board);
    let value = serde_json::to_value(&board)?;
    store.put(TODOS_NAMESPACE, &key(thread_id), value).await?;
    Ok(board.cards)
}

fn snapshot(thread_id: &str, cards: Vec<TaskBoardCard>) -> TodosSnapshot {
    let markdown = render_markdown(&cards);
    TodosSnapshot {
        thread_id: thread_id.to_string(),
        cards,
        markdown,
    }
}

/// At most one card may be `InProgress` at a time. Returns a
/// [`Validation`](TinyAgentsError::Validation) error otherwise (never silently
/// fixes it).
fn enforce_single_in_progress(cards: &[TaskBoardCard]) -> Result<()> {
    let in_progress = cards
        .iter()
        .filter(|c| matches!(c.status, TaskCardStatus::InProgress))
        .count();
    if in_progress > 1 {
        return Err(TinyAgentsError::Validation(format!(
            "only one todo may be `in_progress` at a time (got {in_progress})"
        )));
    }
    Ok(())
}

/// Snapshot the current board without mutating.
pub async fn list(store: &Arc<dyn Store>, thread_id: &str) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let cards = load_cards(store, &thread_id).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Append a new card. `content` is the required title; `patch` supplies the rest.
pub async fn add(
    store: &Arc<dyn Store>,
    thread_id: &str,
    content: &str,
    patch: CardPatch,
) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let content = content.trim();
    if content.is_empty() {
        return Err(TinyAgentsError::Validation(
            "todo content must not be empty".to_string(),
        ));
    }
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut cards = load_cards(store, &thread_id).await?;
    let order = cards.len() as u32;
    cards.push(TaskBoardCard {
        title: content.to_string(),
        status: patch.status.unwrap_or(TaskCardStatus::Todo),
        objective: patch.objective.and_then(non_empty),
        plan: patch.plan.unwrap_or_default(),
        assigned_agent: patch.assigned_agent.and_then(non_empty),
        allowed_tools: patch.allowed_tools.unwrap_or_default(),
        approval_mode: patch.approval_mode.flatten(),
        acceptance_criteria: patch.acceptance_criteria.unwrap_or_default(),
        evidence: patch.evidence.unwrap_or_default(),
        notes: patch.notes.and_then(non_empty),
        blocker: patch.blocker.and_then(non_empty),
        source_metadata: patch.source_metadata,
        order,
        ..TaskBoardCard::new(content)
    });
    enforce_single_in_progress(&cards)?;
    let cards = save_cards(store, &thread_id, cards).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Edit an existing card. Fields left `None` in `patch` are untouched. Errors if
/// `id` is unknown.
pub async fn edit(
    store: &Arc<dyn Store>,
    thread_id: &str,
    id: &str,
    patch: CardPatch,
) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut cards = load_cards(store, &thread_id).await?;
    let card = cards
        .iter_mut()
        .find(|c| c.id == id)
        .ok_or_else(|| TinyAgentsError::Validation(format!("todo id '{id}' not found")))?;
    if let Some(content) = patch.content {
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            return Err(TinyAgentsError::Validation(
                "todo content must not be empty".to_string(),
            ));
        }
        card.title = trimmed;
    }
    if let Some(status) = patch.status {
        card.status = status;
    }
    if let Some(objective) = patch.objective {
        card.objective = non_empty(objective);
    }
    if let Some(plan) = patch.plan {
        card.plan = plan;
    }
    if let Some(assigned_agent) = patch.assigned_agent {
        card.assigned_agent = non_empty(assigned_agent);
    }
    if let Some(allowed_tools) = patch.allowed_tools {
        card.allowed_tools = allowed_tools;
    }
    if let Some(approval_mode) = patch.approval_mode {
        card.approval_mode = approval_mode;
    }
    if let Some(acceptance_criteria) = patch.acceptance_criteria {
        card.acceptance_criteria = acceptance_criteria;
    }
    if let Some(evidence) = patch.evidence {
        card.evidence = evidence;
    }
    if let Some(notes) = patch.notes {
        card.notes = non_empty(notes);
    }
    if let Some(blocker) = patch.blocker {
        card.blocker = non_empty(blocker);
    }
    if let Some(source_metadata) = patch.source_metadata {
        card.source_metadata = Some(source_metadata);
    }
    card.updated_at = now_stamp();
    enforce_single_in_progress(&cards)?;
    let cards = save_cards(store, &thread_id, cards).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Update only the status of a card.
pub async fn update_status(
    store: &Arc<dyn Store>,
    thread_id: &str,
    id: &str,
    status: TaskCardStatus,
) -> Result<TodosSnapshot> {
    edit(
        store,
        thread_id,
        id,
        CardPatch {
            status: Some(status),
            ..Default::default()
        },
    )
    .await
}

/// Stamp (or clear, with a blank id) a card's `session_thread_id` — the
/// conversation thread of its live/last run. Pure session-link bookkeeping,
/// orthogonal to the lifecycle (does not touch status or the invariant).
pub async fn set_session_thread(
    store: &Arc<dyn Store>,
    thread_id: &str,
    id: &str,
    session_thread_id: Option<String>,
) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut cards = load_cards(store, &thread_id).await?;
    let card = cards
        .iter_mut()
        .find(|c| c.id == id)
        .ok_or_else(|| TinyAgentsError::Validation(format!("todo id '{id}' not found")))?;
    card.session_thread_id = session_thread_id.and_then(non_empty);
    card.updated_at = now_stamp();
    let cards = save_cards(store, &thread_id, cards).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Resolve a plan-approval decision: approve (→ `Ready`) or reject
/// (→ `Rejected`). Errors unless the card is currently `AwaitingApproval`, so a
/// stale/duplicate decision can't resurrect a card that already moved on.
pub async fn decide_plan(
    store: &Arc<dyn Store>,
    thread_id: &str,
    id: &str,
    approve: bool,
) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut cards = load_cards(store, &thread_id).await?;
    let card = cards
        .iter_mut()
        .find(|c| c.id == id)
        .ok_or_else(|| TinyAgentsError::Validation(format!("todo id '{id}' not found")))?;
    if card.status != TaskCardStatus::AwaitingApproval {
        return Err(TinyAgentsError::Validation(format!(
            "card '{id}' is not awaiting approval (status: {})",
            card.status.as_str()
        )));
    }
    card.status = if approve {
        TaskCardStatus::Ready
    } else {
        TaskCardStatus::Rejected
    };
    card.updated_at = now_stamp();
    let cards = save_cards(store, &thread_id, cards).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Reject **every** `AwaitingApproval` card so none stays runnable, clearing a
/// parked plan for re-planning. Lenient when nothing is awaiting (a benign
/// no-op rather than an error).
pub async fn revise_plan(store: &Arc<dyn Store>, thread_id: &str) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut cards = load_cards(store, &thread_id).await?;
    for card in cards.iter_mut() {
        if card.status == TaskCardStatus::AwaitingApproval {
            card.status = TaskCardStatus::Rejected;
            card.updated_at = now_stamp();
        }
    }
    let cards = save_cards(store, &thread_id, cards).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Remove a card by id. Errors if `id` is unknown.
pub async fn remove(store: &Arc<dyn Store>, thread_id: &str, id: &str) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut cards = load_cards(store, &thread_id).await?;
    let before = cards.len();
    cards.retain(|c| c.id != id);
    if cards.len() == before {
        return Err(TinyAgentsError::Validation(format!(
            "todo id '{id}' not found"
        )));
    }
    let cards = save_cards(store, &thread_id, cards).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Wholesale-replace the board's cards. Cards missing ids get server-generated
/// ones on normalise.
pub async fn replace(
    store: &Arc<dyn Store>,
    thread_id: &str,
    cards: Vec<TaskBoardCard>,
) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    enforce_single_in_progress(&cards)?;
    let cards = save_cards(store, &thread_id, cards).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Empty the board.
pub async fn clear(store: &Arc<dyn Store>, thread_id: &str) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let cards = save_cards(store, &thread_id, Vec::new()).await?;
    Ok(snapshot(&thread_id, cards))
}

/// Atomic compare-and-set claim: transition a card from one of `expected` to
/// `target` under the per-thread lock, returning the fresh card on success. If
/// the card's current status is not in `expected`, the claim is rejected — the
/// caller lost the race or the card already moved on.
pub async fn claim_card(
    store: &Arc<dyn Store>,
    thread_id: &str,
    card_id: &str,
    expected: &[TaskCardStatus],
    target: TaskCardStatus,
) -> Result<TaskBoardCard> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut cards = load_cards(store, &thread_id).await?;
    let card = cards.iter_mut().find(|c| c.id == card_id).ok_or_else(|| {
        TinyAgentsError::Validation(format!("claim_card: card '{card_id}' not found on board"))
    })?;
    if !expected.contains(&card.status) {
        return Err(TinyAgentsError::Validation(format!(
            "claim_card: card '{card_id}' status is '{}', expected one of [{}]; claim rejected",
            card.status.as_str(),
            expected
                .iter()
                .map(TaskCardStatus::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    card.status = target;
    card.updated_at = now_stamp();
    let claimed_id = card.id.clone();
    enforce_single_in_progress(&cards)?;
    let cards = save_cards(store, &thread_id, cards).await?;
    cards
        .into_iter()
        .find(|c| c.id == claimed_id)
        .ok_or_else(|| {
            TinyAgentsError::Graph(format!("claim_card: card '{claimed_id}' lost after save"))
        })
}
