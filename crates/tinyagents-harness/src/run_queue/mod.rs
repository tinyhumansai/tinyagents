//! Generic multi-lane queue for messages arriving during an active run.
//!
//! Hosts decide which incoming events should be queued and retain ownership of
//! the queued payload. TinyAgents owns the reusable FIFO mechanics for the
//! three lanes an agent runtime can consume at safe iteration boundaries:
//! immediate steering, deferred follow-up work, and collected context.

mod types;

use tokio::sync::Mutex;

pub use types::{QueueLane, QueueStatus};

/// Thread-safe FIFO queue split into steer, follow-up, and collect lanes.
#[derive(Debug)]
pub struct RunQueue<T> {
    inner: Mutex<RunQueueInner<T>>,
}

#[derive(Debug)]
struct RunQueueInner<T> {
    steers: Vec<T>,
    followups: Vec<T>,
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
