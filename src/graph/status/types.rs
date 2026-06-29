//! Compact, observable graph run status records.

use std::time::SystemTime;

use crate::harness::ids::{
    CheckpointId, EventId, ExecutionStatus, GraphId, InterruptId, NodeId, RunId, ThreadId,
};

/// A compact, readable summary of a graph run at an execution boundary.
///
/// Status records are **not** checkpoints. A [`crate::graph::Checkpoint`]
/// preserves the resumable state of a run; a `GraphRunStatus` summarizes live
/// and recent execution so observers (UIs, supervisors, tests) can answer "is
/// this run active?", "which node is executing?", and "which interrupt is
/// waiting?" without deserializing full graph state.
#[derive(Clone, Debug)]
pub struct GraphRunStatus {
    /// This run's id.
    pub run_id: RunId,
    /// The root run id of the run tree (equal to `run_id` for top-level runs).
    pub root_run_id: RunId,
    /// The parent run id when this run is a subgraph/sub-agent child.
    pub parent_run_id: Option<RunId>,
    /// The thread id when checkpointing is enabled.
    pub thread_id: Option<ThreadId>,
    /// The graph this run belongs to.
    pub graph_id: GraphId,
    /// The latest checkpoint id, if any.
    pub checkpoint_id: Option<CheckpointId>,
    /// The checkpoint namespace (for nested subgraph runs).
    pub checkpoint_namespace: Vec<String>,
    /// Coarse lifecycle status.
    pub status: ExecutionStatus,
    /// The current (or final) superstep number.
    pub current_step: usize,
    /// Nodes active at this boundary.
    pub active_nodes: Vec<NodeId>,
    /// Interrupts awaiting a resume command.
    pub pending_interrupts: Vec<InterruptId>,
    /// The last emitted event id, if event ids are tracked.
    pub last_event_id: Option<EventId>,
    /// When the run started.
    pub started_at: SystemTime,
    /// When this status was last updated.
    pub updated_at: SystemTime,
    /// When the run reached a terminal state, if it has.
    pub ended_at: Option<SystemTime>,
    /// A rendered error summary for failed runs.
    pub error: Option<String>,
}
