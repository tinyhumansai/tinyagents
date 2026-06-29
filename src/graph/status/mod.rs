//! Graph run status snapshots.
//!
//! See [`types`] for the [`GraphRunStatus`] definition.

mod types;

pub use types::GraphRunStatus;

use std::time::SystemTime;

use crate::harness::ids::{ExecutionStatus, GraphId, RunId};

impl GraphRunStatus {
    /// Creates a fresh status for a top-level run with no recorded progress yet.
    pub fn new(run_id: RunId, graph_id: GraphId, status: ExecutionStatus) -> Self {
        let now = SystemTime::now();
        Self {
            root_run_id: run_id.clone(),
            run_id,
            parent_run_id: None,
            thread_id: None,
            graph_id,
            checkpoint_id: None,
            checkpoint_namespace: Vec::new(),
            status,
            current_step: 0,
            active_nodes: Vec::new(),
            pending_interrupts: Vec::new(),
            last_event_id: None,
            started_at: now,
            updated_at: now,
            ended_at: None,
            error: None,
        }
    }

    /// Returns true when the run is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            ExecutionStatus::Completed | ExecutionStatus::Failed | ExecutionStatus::Cancelled
        )
    }
}

#[cfg(test)]
mod test;
