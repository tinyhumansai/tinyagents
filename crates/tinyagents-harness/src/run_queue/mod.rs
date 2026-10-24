//! Generic multi-lane queue for messages arriving during an active run.
//!
//! Hosts decide which incoming events should be queued and retain ownership of
//! the queued payload. TinyAgents owns the reusable FIFO mechanics for the
//! three lanes an agent runtime can consume at safe iteration boundaries:
//! immediate steering, deferred follow-up work, and collected context.
//!
//! # On the agent loop path (A4)
//!
//! Attach a [`RunQueueHandle`] (an `Arc<RunQueue<Message>>`) to a run with
//! [`crate::context::RunContext::with_run_queue`] and the built-in
//! [`crate::agent_loop`] drains it at its safe turn boundaries, taking
//! [`QueueMode::All`] or [`QueueMode::OneAtATime`] items per boundary as
//! [`crate::runtime::RunPolicy::queue_mode`] says:
//!
//! - [`QueueLane::Steer`] — appended to the transcript right after a tool
//!   batch's results (never mid-batch), and at a natural finish before any
//!   follow-up. A steer that arrives after the model's final answer still
//!   gets one more turn.
//! - [`QueueLane::Followup`] — appended only when the model has finished and
//!   no steer is pending; the loop runs another turn instead of returning.
//! - [`QueueLane::Collect`] — never enters the transcript; drained once at
//!   run end onto [`crate::middleware::AgentRun::collected`].
//!
//! A middleware stop, limit stop, pause, or deferral is terminal: whatever is
//! still queued stays queued for the host. Every application emits
//! [`crate::events::AgentEvent::QueuedMessageApplied`]. The existing
//! [`crate::steering::SteeringHandle`] control channel (pause/resume/cancel/
//! inject) is unchanged and independent — `RunQueue` is content injection,
//! not run control.
//!
//! `RunQueue<T>` itself stays generic: hosts may keep using it with any `T`
//! for their own bookkeeping; only a `RunQueue<Message>` is loop-consumable.

mod forwarder;
mod types;

use std::sync::{Mutex, MutexGuard, PoisonError};

pub use forwarder::{
    AGENT_COLLECT_PREFIX, AGENT_STEER_PREFIX, COLLECT_PREFIX, ForwardEvent, ForwardEventSink,
    ForwarderCleanup, MessageSteeringForwarderGuard, QueuedMessage, STEER_PREFIX,
    SteeringForwarderGuard, fence_agent_message, forward_collects, forward_message_collects,
    forward_message_steers, forward_steers, lane_prefix,
};

pub use types::{QueueLane, QueueMode, QueueStatus, RunQueueHandle};

/// Thread-safe FIFO queue split into steer, follow-up, and collect lanes.
///
/// The lock guards only in-memory vector operations and is never held across
/// an `.await`, so every operation also has a synchronous form
/// ([`Self::push_now`], [`Self::drain_now`]) usable from `Drop` or other
/// non-async code without spawning a task.
#[derive(Debug)]
pub struct RunQueue<T> {
    inner: Mutex<RunQueueInner<T>>,
}

/// The three FIFO lanes behind [`RunQueue`]'s lock.
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

    /// Locks the lanes, recovering a poisoned lock. Vector invariants remain
    /// valid after unwinding; an editing callback must keep its payload valid
    /// even when it panics because callback changes are not rolled back.
    fn lock(&self) -> MutexGuard<'_, RunQueueInner<T>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Appends `item` to `lane`.
    pub async fn push(&self, lane: QueueLane, item: T) {
        self.push_now(lane, item);
    }

    /// Appends `item` to `lane` without awaiting: the synchronous form of
    /// [`Self::push`], for `Drop` impls and other non-async callers.
    pub fn push_now(&self, lane: QueueLane, item: T) {
        let mut inner = self.lock();
        match lane {
            QueueLane::Steer => inner.steers.push(item),
            QueueLane::Followup => inner.followups.push(item),
            QueueLane::Collect => inner.collects.push(item),
        }
    }

    /// Puts `items` back at the **front** of `lane`, in their given order, as
    /// one atomic step. Used to return items that were drained but never
    /// delivered, so they stay ahead of anything queued after them.
    pub async fn requeue_front(&self, lane: QueueLane, items: Vec<T>) {
        if items.is_empty() {
            return;
        }
        let mut inner = self.lock();
        let target = match lane {
            QueueLane::Steer => &mut inner.steers,
            QueueLane::Followup => &mut inner.followups,
            QueueLane::Collect => &mut inner.collects,
        };
        target.splice(0..0, items);
    }

    /// Drains one lane in FIFO order.
    pub async fn drain(&self, lane: QueueLane) -> Vec<T> {
        self.drain_now(lane)
    }

    /// Drains one lane in FIFO order without awaiting: the synchronous form
    /// of [`Self::drain`].
    pub fn drain_now(&self, lane: QueueLane) -> Vec<T> {
        let mut inner = self.lock();
        match lane {
            QueueLane::Steer => std::mem::take(&mut inner.steers),
            QueueLane::Followup => std::mem::take(&mut inner.followups),
            QueueLane::Collect => std::mem::take(&mut inner.collects),
        }
    }

    /// Drains matching items from one lane in FIFO order and leaves all other
    /// items in place. This is useful for typed host adapters that can consume
    /// only a subset of a heterogeneous queue without discarding unsupported
    /// payloads.
    pub fn drain_where_now(&self, lane: QueueLane, mut pred: impl FnMut(&T) -> bool) -> Vec<T> {
        let mut inner = self.lock();
        let items = match lane {
            QueueLane::Steer => &mut inner.steers,
            QueueLane::Followup => &mut inner.followups,
            QueueLane::Collect => &mut inner.collects,
        };
        let mut drained = Vec::new();
        let mut remaining = Vec::with_capacity(items.len());
        for item in std::mem::take(items) {
            if pred(&item) {
                drained.push(item);
            } else {
                remaining.push(item);
            }
        }
        *items = remaining;
        drained
    }

    /// Takes items from `lane` in FIFO order according to `mode`: the oldest
    /// item only under [`QueueMode::OneAtATime`], or the whole lane under
    /// [`QueueMode::All`]. Returns an empty vec when the lane is empty.
    pub async fn take(&self, lane: QueueLane, mode: QueueMode) -> Vec<T> {
        match mode {
            QueueMode::All => self.drain(lane).await,
            QueueMode::OneAtATime => {
                let mut inner = self.lock();
                let items = match lane {
                    QueueLane::Steer => &mut inner.steers,
                    QueueLane::Followup => &mut inner.followups,
                    QueueLane::Collect => &mut inner.collects,
                };
                if items.is_empty() {
                    Vec::new()
                } else {
                    vec![items.remove(0)]
                }
            }
        }
    }

    /// Returns the current queue depth per lane.
    pub async fn status(&self) -> QueueStatus {
        let inner = self.lock();
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
        let mut inner = self.lock();
        let total = inner.steers.len() + inner.followups.len() + inner.collects.len();
        inner.steers.clear();
        inner.followups.clear();
        inner.collects.clear();
        total
    }

    /// Returns a snapshot of every queued item, tagged with its lane, in
    /// lane/delivery order: [`QueueLane::Steer`], then
    /// [`QueueLane::Followup`], then [`QueueLane::Collect`], each lane
    /// preserving its own FIFO order. Non-destructive — the queue is
    /// unchanged, so a host can inspect pending work (e.g. to render it, or
    /// to decide whether [`Self::remove_where`] applies) without racing the
    /// agent loop's own drain.
    pub async fn snapshot(&self) -> Vec<(QueueLane, T)>
    where
        T: Clone,
    {
        let inner = self.lock();
        let mut items =
            Vec::with_capacity(inner.steers.len() + inner.followups.len() + inner.collects.len());
        items.extend(
            inner
                .steers
                .iter()
                .cloned()
                .map(|item| (QueueLane::Steer, item)),
        );
        items.extend(
            inner
                .followups
                .iter()
                .cloned()
                .map(|item| (QueueLane::Followup, item)),
        );
        items.extend(
            inner
                .collects
                .iter()
                .cloned()
                .map(|item| (QueueLane::Collect, item)),
        );
        items
    }

    /// Removes every queued item across all lanes for which `pred` returns
    /// `true`, preserving the relative order of the items that remain in
    /// each lane. Returns the number of items removed.
    ///
    /// Use to retract a specific queued item (e.g. one the host decided not
    /// to apply after all) without clearing the rest of the queue.
    pub async fn remove_where(&self, pred: impl Fn(&T) -> bool) -> usize {
        let mut inner = self.lock();
        let before = inner.steers.len() + inner.followups.len() + inner.collects.len();
        inner.steers.retain(|item| !pred(item));
        inner.followups.retain(|item| !pred(item));
        inner.collects.retain(|item| !pred(item));
        let after = inner.steers.len() + inner.followups.len() + inner.collects.len();
        before - after
    }

    /// Edits the first matching pending item in delivery order, atomically
    /// with respect to delivery, removal, and other queue mutations.
    ///
    /// Returns `false` if the item has already been consumed. `edit` runs
    /// under the queue lock: it must be brief and must not call this queue.
    /// Hosts retain responsibility for validating the replacement payload.
    pub async fn edit_first_where(
        &self,
        pred: impl Fn(&T) -> bool,
        edit: impl FnOnce(&mut T),
    ) -> bool {
        let mut inner = self.lock();
        let RunQueueInner {
            steers,
            followups,
            collects,
        } = &mut *inner;
        let item = steers
            .iter_mut()
            .chain(followups.iter_mut())
            .chain(collects.iter_mut())
            .find(|item| pred(item));
        match item {
            Some(item) => {
                edit(item);
                true
            }
            None => false,
        }
    }

    /// Moves the first matching pending item to `lane` at `index`, atomically
    /// with respect to delivery. `index` addresses the destination after the
    /// item is removed and is clamped to that lane's length.
    ///
    /// Moving a follow-up into the steer lane requests delivery at the next
    /// safe boundary; it never interrupts a tool batch. Returns `false` when
    /// the item has already been consumed. All other items retain their order.
    pub async fn move_first_where(
        &self,
        pred: impl Fn(&T) -> bool,
        lane: QueueLane,
        index: usize,
    ) -> bool {
        let mut inner = self.lock();
        let item = {
            let RunQueueInner {
                steers,
                followups,
                collects,
            } = &mut *inner;
            [steers, followups, collects].into_iter().find_map(|items| {
                items
                    .iter()
                    .position(&pred)
                    .map(|position| items.remove(position))
            })
        };
        let Some(item) = item else {
            return false;
        };
        let target = match lane {
            QueueLane::Steer => &mut inner.steers,
            QueueLane::Followup => &mut inner.followups,
            QueueLane::Collect => &mut inner.collects,
        };
        target.insert(index.min(target.len()), item);
        true
    }
}

impl<T> Default for RunQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
