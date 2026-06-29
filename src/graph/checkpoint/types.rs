//! Checkpoint records and metadata.
//!
//! Checkpoints are graph-runtime persistence, separate from harness memory and
//! long-term stores. They are written at superstep boundaries only — never
//! mid-node — because rerunning a node from its start is far easier to reason
//! about than suspending an async Rust stack, and it matches interrupt/resume
//! semantics exactly.

use crate::graph::command::Interrupt;
use crate::harness::ids::NodeId;

/// A persisted snapshot of a graph run at a superstep boundary.
#[derive(Clone, Debug)]
pub struct Checkpoint<State> {
    /// Checkpoint lineage key for a conversation/workflow/tenant run series.
    pub thread_id: String,
    /// This checkpoint's id within the thread.
    pub checkpoint_id: String,
    /// The previous checkpoint id in the thread lineage.
    pub parent_checkpoint_id: Option<String>,
    /// Namespace scoping for nested subgraph checkpoints.
    pub namespace: Vec<String>,
    /// Committed graph state at this boundary.
    pub state: State,
    /// Nodes that should run when resuming from this checkpoint.
    pub next_nodes: Vec<NodeId>,
    /// Nodes that completed in the step that produced this checkpoint.
    pub completed_tasks: Vec<NodeId>,
    /// Per-task partial writes preserved when a step partially completes.
    pub pending_writes: Vec<PendingWrite>,
    /// Interrupts that paused the run at this boundary.
    pub interrupts: Vec<Interrupt>,
    /// Free-form metadata (source, step, etc.).
    pub metadata: serde_json::Value,
}

/// A partial write produced by a completed task, preserved across reruns.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PendingWrite {
    /// The node that produced the write.
    pub node: NodeId,
    /// The serialized write payload.
    pub payload: serde_json::Value,
}

/// Lightweight checkpoint summary returned by `Checkpointer::list`.
///
/// Listing must not require deserializing full graph state, so metadata is kept
/// separate from the [`Checkpoint`] state payload.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Thread lineage key.
    pub thread_id: String,
    /// Checkpoint id.
    pub checkpoint_id: String,
    /// Parent checkpoint id.
    pub parent_checkpoint_id: Option<String>,
    /// Namespace scoping.
    pub namespace: Vec<String>,
    /// Nodes to run on resume.
    pub next_nodes: Vec<NodeId>,
    /// Whether the checkpoint carries pending interrupts.
    pub has_interrupts: bool,
    /// Checkpoint source: `input`, `loop`, `update`, or `fork`.
    pub source: String,
    /// The superstep number that produced the checkpoint.
    pub step: usize,
}
