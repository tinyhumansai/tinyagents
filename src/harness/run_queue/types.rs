//! Public types for the active-run queue.

/// A queue lane consumed by the agent runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueLane {
    /// Inject at the next safe iteration boundary as an instruction.
    Steer,
    /// Dispatch as a fresh turn after the active run completes.
    Followup,
    /// Inject at the next safe boundary as additional context.
    Collect,
}

/// Snapshot of the queue depth per lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct QueueStatus {
    /// Number of pending steer items.
    pub steers: usize,
    /// Number of pending follow-up items.
    pub followups: usize,
    /// Number of pending collected-context items.
    pub collects: usize,
    /// Total number of pending items across all lanes.
    pub total: usize,
}
