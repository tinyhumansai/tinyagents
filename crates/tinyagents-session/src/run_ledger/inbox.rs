//! Durable addressed input admission, independent of a live worker registry.
//!
//! A receipt means the input is durably queued, not that a model has read it.
//! Recipients must commit input to their own durable session before acknowledging
//! delivery. This admission API deliberately exposes no ledger-only "delivered"
//! mutation. A host must not substitute live steering for durable admission.

use serde::{Deserialize, Serialize};

/// Intended delivery boundary in the recipient's existing session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    /// Admit at the next safe boundary of running work.
    Steer,
    /// Queue another turn after the current work.
    FollowUp,
    /// Answer the recipient's pending question.
    Answer,
}

/// Operator input addressed to one exact durable run under a root conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddressedInput {
    /// Unique within the root conversation; retry the same payload with this id.
    pub submission_id: String,
    /// Root conversation whose persisted lineage authorizes the target.
    pub root_thread_id: String,
    /// Exact run-ledger identity, never an agent definition or display name.
    pub recipient_run_id: String,
    /// Required recipient execution boundary.
    pub mode: InputMode,
    /// Original user text; not trimmed or rewritten after admission.
    pub text: String,
}

/// Durable queued input. Repeated admission returns the original receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedInput {
    /// Original addressed input.
    pub input: AddressedInput,
    /// Recipient event sequence allocated atomically with admission.
    pub sequence: u64,
    /// Admission time at storage millisecond precision.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Optional durable inbox capability of an already scoped ledger handle.
#[async_trait::async_trait]
pub trait RunInboxStore: Send + Sync {
    /// Validate persisted root lineage and atomically insert input plus receipt.
    /// Reusing an id for different content is an error, never another delivery.
    async fn enqueue_input(&self, input: AddressedInput) -> super::LedgerResult<QueuedInput>;

    /// Read queued input in admission order after validating root ownership.
    /// A cursor excludes sequences at or below its value, allowing bounded paging.
    /// Limit is clamped to 1..=500; this never consumes or acknowledges input.
    async fn pending_inputs(
        &self,
        root_thread_id: &str,
        recipient_run_id: &str,
        after_sequence: Option<u64>,
        limit: u32,
    ) -> super::LedgerResult<Vec<QueuedInput>>;
}
