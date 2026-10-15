//! Data model for the durable completion router.

use std::collections::BTreeMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// How a finished child announces itself to its parent.
///
/// Chosen per spawn. The default, [`NotifyMode::Followup`], matches a
/// fire-and-forget background child whose result should reach the parent
/// without it asking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotifyMode {
    /// Push onto the parent's follow-up lane when the parent is live; otherwise
    /// stay pending for [`CompletionRouter::claim_pending`](super::CompletionRouter::claim_pending).
    #[default]
    Followup,
    /// Push onto the parent's collect lane when the parent is live; otherwise
    /// stay pending for [`CompletionRouter::claim_pending`](super::CompletionRouter::claim_pending).
    Collect,
    /// Stay pending until the parent's next turn starts
    /// ([`CompletionRouter::begin_turn`](super::CompletionRouter::begin_turn)).
    HoldForNextTurn,
    /// Record only. The parent pulls it
    /// ([`CompletionRouter::pull`](super::CompletionRouter::pull)).
    Off,
}

impl NotifyMode {
    /// Stable snake_case label for logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Followup => "followup",
            Self::Collect => "collect",
            Self::HoldForNextTurn => "hold_for_next_turn",
            Self::Off => "off",
        }
    }
}

/// How the child ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    /// Ran to a usable result.
    Success,
    /// Errored before producing a result.
    Failed,
    /// Cancelled before producing a result.
    Cancelled,
    /// Stopped without a complete result: timeout, budget, or waiting on input.
    Incomplete,
}

impl CompletionStatus {
    /// Stable snake_case label for logs and neutral formatting.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Incomplete => "incomplete",
        }
    }
}

/// Where a completion is in its delivery life.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionState {
    /// Waiting to be delivered (or redelivered after a failed attempt or a
    /// restart).
    Pending,
    /// The parent has it.
    Delivered,
    /// The parent collected the child explicitly, so it is not pushed from then on.
    Tombstoned,
    /// Too many failed attempts. The host applies its own give-up policy.
    GaveUp,
}

impl CompletionState {
    /// Stable snake_case label for logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Delivered => "delivered",
            Self::Tombstoned => "tombstoned",
            Self::GaveUp => "gave_up",
        }
    }
}

/// A neutral reference to a host-owned artifact holding the full output.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionArtifact {
    /// Stable host artifact id.
    pub id: String,
    /// Optional media-type hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Opaque host metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// The child's output as the parent should see it.
///
/// This is already bounded by the result policy that ran when the child
/// finished; the router stores it as given.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionResult {
    /// Visible text.
    pub text: String,
    /// Characters the result policy dropped.
    #[serde(default)]
    pub omitted_chars: usize,
    /// The full output, when the policy stored it elsewhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<CompletionArtifact>,
}

impl CompletionResult {
    /// A text-only result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }
}

/// One finished child's durable delivery record. Keyed by `task_id`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionRecord {
    /// The child's task id. The dedupe key.
    pub task_id: String,
    /// The parent this is for: a thread id or session key that survives a
    /// restart. A tombstone written before the completion exists carries an
    /// empty key.
    pub parent_key: String,
    /// The child's agent id.
    pub agent_id: String,
    /// Optional human label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// How the child ended.
    pub status: CompletionStatus,
    /// The child's output. Emptied once the record settles (`Delivered`,
    /// `Tombstoned` or `GaveUp`): the store then keeps only the bookkeeping
    /// that dedupe and tombstones need, so a withdrawn or delivered result does
    /// not outlive its parent's interest in it.
    pub result: CompletionResult,
    /// When the child finished.
    pub finished_at: SystemTime,
    /// When this record last changed.
    pub updated_at: SystemTime,
    /// Delivery attempts so far.
    #[serde(default)]
    pub attempts: u32,
    /// Delivery state.
    pub state: CompletionState,
    /// How the parent wants to hear about it.
    #[serde(default)]
    pub notify_mode: NotifyMode,
}

impl CompletionRecord {
    /// A pending record finishing now, with the default [`NotifyMode`].
    pub fn new(
        task_id: impl Into<String>,
        parent_key: impl Into<String>,
        agent_id: impl Into<String>,
        status: CompletionStatus,
        result: CompletionResult,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            task_id: task_id.into(),
            parent_key: parent_key.into(),
            agent_id: agent_id.into(),
            label: None,
            status,
            result,
            finished_at: now,
            updated_at: now,
            attempts: 0,
            state: CompletionState::Pending,
            notify_mode: NotifyMode::default(),
        }
    }

    /// Sets the human label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the notify mode.
    pub fn with_notify_mode(mut self, mode: NotifyMode) -> Self {
        self.notify_mode = mode;
        self
    }

    /// Sets when the child finished.
    pub fn with_finished_at(mut self, finished_at: SystemTime) -> Self {
        self.finished_at = finished_at;
        self
    }

    /// Moves the record to a terminal `state` now, dropping its result. The
    /// task id, parent key, status, attempts and timestamps stay, which is all
    /// dedupe, tombstones and compaction read; the payload has no reader once
    /// the record is settled.
    pub(crate) fn settle(&mut self, state: CompletionState) {
        debug_assert_ne!(
            state,
            CompletionState::Pending,
            "settle needs a terminal state"
        );
        self.state = state;
        self.updated_at = SystemTime::now();
        self.result = CompletionResult::default();
    }

    /// A tombstone for a task whose completion has not been recorded yet.
    pub(crate) fn tombstone_stub(task_id: &str) -> Self {
        let mut stub = Self::new(
            task_id,
            "",
            "",
            CompletionStatus::Incomplete,
            CompletionResult::default(),
        );
        stub.state = CompletionState::Tombstoned;
        stub
    }
}
