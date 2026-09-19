//! Generic multi-lane queue for messages arriving during an active run.
//!
//! Hosts decide which incoming events should be queued and retain ownership of
//! the queued payload. TinyAgents owns the reusable FIFO mechanics for the
//! three lanes an agent runtime can consume at safe iteration boundaries:
//! immediate steering, deferred follow-up work, and collected context.
//!
//! # Not on the agent loop path (M-10)
//!
//! [`RunQueue`] is exported for hosts to use, but the built-in
//! [`crate::agent_loop`] does not drain it at any checkpoint today — a host
//! that wants queued input to actually reach a running agent must poll
//! `RunQueue` itself (typically between turns) and feed what it dequeues into
//! [`crate::steering::SteeringHandle::send`] or the next `invoke` call. Wiring
//! `RunQueue` directly into the loop (a `QueueMode` the loop drains after tool
//! results and before returning) is tracked as future work in the runtime
//! comparison plan's Phase 2.

mod types;

use tokio::sync::Mutex;

pub use types::{QueueLane, QueueStatus};

/// Thread-safe FIFO queue split into steer, follow-up, and collect lanes.
#[derive(Debug)]
pub struct RunQueue<T> {
    inner: Mutex<RunQueueInner<T>>,
}

/// The three lanes behind [`RunQueue`]'s lock, each an append-only FIFO until
/// drained.
#[derive(Debug)]
struct RunQueueInner<T> {
    /// Immediate steering messages, consumed as soon as the loop reaches a
    /// safe iteration boundary.
    steers: Vec<T>,
    /// Deferred follow-up work, consumed once the current run settles.
    followups: Vec<T>,
    /// Collected context (e.g. observations) accumulated for later use.
    collects: Vec<T>,
}

impl<T> RunQueue<T> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(RunQueueInner {
                steers: Vec::new(),
                followups: Vec::new(),
                collects: Vec::new(),
            }),
        }
    }

    /// Appends `item` to `lane`.
    pub async fn push(&self, lane: QueueLane, item: T) {
        let mut inner = self.inner.lock().await;
        match lane {
            QueueLane::Steer => inner.steers.push(item),
            QueueLane::Followup => inner.followups.push(item),
            QueueLane::Collect => inner.collects.push(item),
        }
    }

    /// Drains one lane in FIFO order.
    pub async fn drain(&self, lane: QueueLane) -> Vec<T> {
        let mut inner = self.inner.lock().await;
        match lane {
            QueueLane::Steer => std::mem::take(&mut inner.steers),
            QueueLane::Followup => std::mem::take(&mut inner.followups),
            QueueLane::Collect => std::mem::take(&mut inner.collects),
        }
    }

    /// Returns the current queue depth per lane.
    pub async fn status(&self) -> QueueStatus {
        let inner = self.inner.lock().await;
        let steers = inner.steers.len();
        let followups = inner.followups.len();
        let collects = inner.collects.len();
        QueueStatus {
            steers,
            followups,
            collects,
            total: steers + followups + collects,
        }
    }

    /// Clears every lane and returns the number of dropped items.
    pub async fn clear(&self) -> usize {
        let mut inner = self.inner.lock().await;
        let total = inner.steers.len() + inner.followups.len() + inner.collects.len();
        inner.steers.clear();
        inner.followups.clear();
        inner.collects.clear();
        total
    }
}

impl<T> Default for RunQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
