//! Domain types for the per-thread task board (kanban todos).
//!
//! A **task board** is a per-thread list of [`TaskBoardCard`]s — the concrete
//! work items a graph tracks, distinct from the single per-thread
//! [`ThreadGoal`](crate::graph::goals::ThreadGoal). Ported from OpenHuman's
//! task board / `todos` modules, minus the app-specific coupling (progress
//! events, RPC envelopes, scratch fallback).

use serde::{Deserialize, Serialize};

use crate::harness::ids::next_seq;

/// Lifecycle state of a task card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCardStatus {
    /// Not started.
    Todo,
    /// Plan approval required and pending; will not run until approved
    /// (→ `Ready`) or rejected (→ `Rejected`).
    AwaitingApproval,
    /// Approved for execution — runnable without a further approval check.
    Ready,
    /// Currently being worked. At most one card may be `InProgress` at a time.
    InProgress,
    /// Blocked on an external dependency; carries a `blocker` reason.
    Blocked,
    /// Finished.
    Done,
    /// Plan approval was denied; the card is not executed.
    Rejected,
}

impl TaskCardStatus {
    /// The stable lower-snake-case status label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Todo => "todo",
            Self::AwaitingApproval => "awaiting_approval",
            Self::Ready => "ready",
            Self::InProgress => "in_progress",
            Self::Blocked => "blocked",
            Self::Done => "done",
            Self::Rejected => "rejected",
        }
    }
}

/// Whether a card requires human plan approval before it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskApprovalMode {
    /// A human must approve the plan before execution.
    Required,
    /// No approval gate.
    NotRequired,
}

impl TaskApprovalMode {
    /// The stable lower-snake-case mode label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Required => "required",
            Self::NotRequired => "not_required",
        }
    }
}

/// A single task card on a thread's board.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskBoardCard {
    /// Stable card id (`task-<n>`); server-generated when blank.
    pub id: String,
    /// One-line title / summary.
    pub title: String,
    /// Lifecycle state.
    pub status: TaskCardStatus,
    /// Optional richer objective for the card.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    /// Ordered plan steps.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plan: Vec<String>,
    /// The agent assigned to run this card, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assigned_agent: Option<String>,
    /// Tools the assigned agent is allowed to use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_tools: Vec<String>,
    /// Plan-approval mode, if the card is gated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_mode: Option<TaskApprovalMode>,
    /// Acceptance criteria that define "done" for this card.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acceptance_criteria: Vec<String>,
    /// Evidence gathered toward completion.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<String>,
    /// Free-form notes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// Blocker reason (populated from `notes` on normalise when a `Blocked` card
    /// has none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocker: Option<String>,
    /// Conversation thread id of the card's live/last run, for UI cross-linking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_thread_id: Option<String>,
    /// Provider/source identifiers for a card ingested from an external source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_metadata: Option<serde_json::Value>,
    /// Position on the board (recomputed on normalise).
    #[serde(default)]
    pub order: u32,
    /// Last-mutation timestamp (unix-epoch milliseconds, as a string).
    #[serde(default)]
    pub updated_at: String,
}

impl TaskBoardCard {
    /// Creates a minimal `Todo` card with `title`, a fresh id, and no metadata.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            id: new_card_id(),
            title: title.into(),
            status: TaskCardStatus::Todo,
            objective: None,
            plan: Vec::new(),
            assigned_agent: None,
            allowed_tools: Vec::new(),
            approval_mode: None,
            acceptance_criteria: Vec::new(),
            evidence: Vec::new(),
            notes: None,
            blocker: None,
            session_thread_id: None,
            source_metadata: None,
            order: 0,
            updated_at: now_stamp(),
        }
    }
}

/// A per-thread board: an ordered list of cards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskBoard {
    /// The thread this board belongs to.
    pub thread_id: String,
    /// The cards, in board order.
    pub cards: Vec<TaskBoardCard>,
    /// Last-mutation timestamp (unix-epoch milliseconds, as a string).
    pub updated_at: String,
}

impl TaskBoard {
    /// An empty board for `thread_id`.
    pub fn empty(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            cards: Vec::new(),
            updated_at: now_stamp(),
        }
    }
}

/// A single CRUD outcome: the post-mutation cards plus a markdown rendering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodosSnapshot {
    /// The thread the board belongs to.
    pub thread_id: String,
    /// The cards after the mutation.
    pub cards: Vec<TaskBoardCard>,
    /// GitHub-flavored markdown rendering of the cards.
    pub markdown: String,
}

/// Optional fields supplied by `add` / `edit`.
///
/// `approval_mode` is doubly-optional: `None` leaves it untouched, `Some(None)`
/// clears it, `Some(Some(_))` sets it.
#[derive(Debug, Default, Clone)]
pub struct CardPatch {
    /// New title (the model-facing "content").
    pub content: Option<String>,
    /// New status.
    pub status: Option<TaskCardStatus>,
    /// New objective (empty clears).
    pub objective: Option<String>,
    /// New plan steps.
    pub plan: Option<Vec<String>>,
    /// New assigned agent (empty clears).
    pub assigned_agent: Option<String>,
    /// New allowed-tools list.
    pub allowed_tools: Option<Vec<String>>,
    /// New approval mode (`Some(None)` clears).
    pub approval_mode: Option<Option<TaskApprovalMode>>,
    /// New acceptance criteria.
    pub acceptance_criteria: Option<Vec<String>>,
    /// New evidence list.
    pub evidence: Option<Vec<String>>,
    /// New notes (empty clears).
    pub notes: Option<String>,
    /// New blocker (empty clears).
    pub blocker: Option<String>,
    /// New source metadata (`Some` sets; `None` leaves untouched).
    pub source_metadata: Option<serde_json::Value>,
}

/// Parses a stable string status alias into a [`TaskCardStatus`].
pub fn parse_status(raw: &str) -> Result<TaskCardStatus, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "todo" | "pending" => Ok(TaskCardStatus::Todo),
        "awaiting_approval" | "awaiting-approval" => Ok(TaskCardStatus::AwaitingApproval),
        "ready" | "approved" => Ok(TaskCardStatus::Ready),
        "in_progress" | "in-progress" | "inprogress" | "started" => Ok(TaskCardStatus::InProgress),
        "blocked" => Ok(TaskCardStatus::Blocked),
        "done" | "completed" | "complete" => Ok(TaskCardStatus::Done),
        "rejected" | "denied" => Ok(TaskCardStatus::Rejected),
        other => Err(format!(
            "invalid status '{other}' (expected todo|awaiting_approval|ready|in_progress|blocked|done|rejected)"
        )),
    }
}

/// Renders a card list as GitHub-flavored markdown: one `- [marker] title`
/// line per card (`[ ]` todo/ready, `[?]` awaiting approval, `[~]` in progress,
/// `[!]` blocked, `[x]` done, `[-]` rejected) followed by indented metadata.
pub fn render_markdown(cards: &[TaskBoardCard]) -> String {
    if cards.is_empty() {
        return "_No todos yet._".to_string();
    }
    let mut out = String::new();
    for card in cards {
        let marker = match card.status {
            TaskCardStatus::Todo | TaskCardStatus::Ready => "[ ]",
            TaskCardStatus::AwaitingApproval => "[?]",
            TaskCardStatus::InProgress => "[~]",
            TaskCardStatus::Blocked => "[!]",
            TaskCardStatus::Done => "[x]",
            TaskCardStatus::Rejected => "[-]",
        };
        out.push_str("- ");
        out.push_str(marker);
        out.push(' ');
        out.push_str(&card.title);
        out.push_str(&format!("  `({})`", card.id));
        out.push('\n');

        if let Some(objective) = card.objective.as_deref() {
            out.push_str("  - objective: ");
            out.push_str(objective);
            out.push('\n');
        }
        if let Some(agent) = card.assigned_agent.as_deref() {
            out.push_str("  - agent: ");
            out.push_str(agent);
            out.push('\n');
        }
        if !card.allowed_tools.is_empty() {
            out.push_str("  - tools: ");
            out.push_str(&card.allowed_tools.join(", "));
            out.push('\n');
        }
        if let Some(mode) = card.approval_mode.as_ref() {
            out.push_str("  - approval: ");
            out.push_str(mode.as_str());
            out.push('\n');
        }
        if !card.plan.is_empty() {
            out.push_str("  - plan:\n");
            for step in &card.plan {
                out.push_str("    - ");
                out.push_str(step);
                out.push('\n');
            }
        }
        if !card.acceptance_criteria.is_empty() {
            out.push_str("  - acceptance criteria:\n");
            for criterion in &card.acceptance_criteria {
                out.push_str("    - ");
                out.push_str(criterion);
                out.push('\n');
            }
        }
        if !card.evidence.is_empty() {
            out.push_str("  - evidence:\n");
            for item in &card.evidence {
                out.push_str("    - ");
                out.push_str(item);
                out.push('\n');
            }
        }

        if matches!(card.status, TaskCardStatus::Blocked) {
            if let Some(reason) = card.blocker.as_deref().or(card.notes.as_deref()) {
                out.push_str("  - _blocked:_ ");
                out.push_str(reason);
                out.push('\n');
            }
        } else if let Some(notes) = card.notes.as_deref() {
            out.push_str("  - ");
            out.push_str(notes);
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

/// Normalises a board in place: trims fields, generates missing card ids, drops
/// empty-title cards, backfills a `Blocked` card's blocker from its notes, and
/// recomputes `order` / `updated_at`.
pub fn normalise_board(board: &mut TaskBoard) {
    let now = now_stamp();
    board.thread_id = board.thread_id.trim().to_string();
    board.updated_at = now.clone();

    for card in board.cards.iter_mut() {
        card.title = card.title.trim().to_string();
        if card.id.trim().is_empty() {
            card.id = new_card_id();
        } else {
            card.id = card.id.trim().to_string();
        }
        card.notes = trim_opt(card.notes.take());
        card.objective = trim_opt(card.objective.take());
        card.assigned_agent = trim_opt(card.assigned_agent.take());
        trim_string_vec(&mut card.plan);
        trim_string_vec(&mut card.allowed_tools);
        trim_string_vec(&mut card.acceptance_criteria);
        trim_string_vec(&mut card.evidence);
        card.blocker = trim_opt(card.blocker.take());
        card.session_thread_id = trim_opt(card.session_thread_id.take());
        if card.status == TaskCardStatus::Blocked && card.blocker.is_none() {
            card.blocker = card.notes.clone();
        }
    }

    board.cards.retain(|card| !card.title.is_empty());

    for (idx, card) in board.cards.iter_mut().enumerate() {
        card.order = idx as u32;
        card.updated_at = now.clone();
    }
}

/// Trims a string, returning `None` when the result is empty.
pub(crate) fn non_empty(s: String) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn trim_opt(value: Option<String>) -> Option<String> {
    value.and_then(non_empty)
}

fn trim_string_vec(values: &mut Vec<String>) {
    values.retain_mut(|value| {
        *value = value.trim().to_string();
        !value.is_empty()
    });
}

/// Mints a fresh, process-unique card id (`task-<n>`).
pub(crate) fn new_card_id() -> String {
    format!("task-{}", next_seq())
}

/// Current unix time in milliseconds, as a string. Dependency-free (no `chrono`).
pub(crate) fn now_stamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
        .to_string()
}
