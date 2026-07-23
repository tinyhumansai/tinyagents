//! Process-local runtime handles for detached orchestration tasks.
//!
//! [`TaskStore`](super::TaskStore) remains the durable source of lifecycle
//! truth. This registry owns the executor-only pieces that cannot survive a
//! process restart: status watch channels, cooperative cancellation tokens,
//! hard-abort handles, ownership checks, and live steering lookup.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::watch;
use tokio::task::AbortHandle;

use crate::harness::ids::TaskId;
use crate::harness::steering::SteeringHandle;
use crate::{CancellationToken, Result, TinyAgentsError};

use super::{
    CancelledDetachedTask, DetachedTaskRegistryError, DetachedTaskSnapshot,
    DetachedTaskWaitOutcome, SteeringRegistry,
};

struct DetachedTaskEntry<Metadata, Status> {
    owner_id: String,
    metadata: Metadata,
    status: watch::Receiver<Status>,
    cancellation: CancellationToken,
    abort: AbortHandle,
}

/// Ownership-aware registry for live detached task executors.
///
/// The registry is generic over application metadata and status payloads. A
/// caller supplies the terminal-status predicate once, then registers the
/// runtime handles created by its executor. Every removal deregisters the
/// corresponding [`SteeringHandle`] from the shared [`SteeringRegistry`].
///
/// This type intentionally does not duplicate durable state. Applications
/// should insert and transition the matching task in a [`TaskStore`](super::TaskStore)
/// as statuses are published. On restart, any non-terminal store record without
/// a runtime entry is an orphan for the application to reconcile.
#[derive(Clone)]
pub struct DetachedTaskRegistry<Metadata, Status> {
    inner: Arc<Mutex<HashMap<TaskId, DetachedTaskEntry<Metadata, Status>>>>,
    steering: SteeringRegistry,
    is_terminal: Arc<dyn Fn(&Status) -> bool + Send + Sync>,
    soft_cap: usize,
}

impl<Metadata, Status> DetachedTaskRegistry<Metadata, Status>
where
    Metadata: Clone + Send + Sync + 'static,
    Status: Clone + Send + Sync + 'static,
{
    /// Creates a registry using `is_terminal` to decide when a task can be
    /// pruned. `soft_cap` triggers a terminal-entry sweep during registration;
    /// live entries are never evicted merely to satisfy the cap.
    pub fn new(
        steering: SteeringRegistry,
        soft_cap: usize,
        is_terminal: impl Fn(&Status) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::default(),
            steering,
            is_terminal: Arc::new(is_terminal),
            soft_cap: soft_cap.max(1),
        }
    }

    /// Registers the executor handles for a detached task.
    ///
    /// Duplicate live ids are rejected. Before insertion, a registry at its
    /// soft cap prunes entries whose latest watched status is terminal.
    pub fn register(
        &self,
        task_id: TaskId,
        owner_id: impl Into<String>,
        metadata: Metadata,
        status: watch::Receiver<Status>,
        cancellation: CancellationToken,
        abort: AbortHandle,
    ) -> Result<()> {
        if self.len() >= self.soft_cap {
            self.sweep_terminal();
        }
        let mut guard = self.lock()?;
        if guard.contains_key(&task_id) {
            return Err(TinyAgentsError::DuplicateComponent(format!(
                "detached task runtime `{task_id}`"
            )));
        }
        guard.insert(
            task_id,
            DetachedTaskEntry {
                owner_id: owner_id.into(),
                metadata,
                status,
                cancellation,
                abort,
            },
        );
        Ok(())
    }

    /// Number of process-local task runtimes currently registered.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|guard| guard.len()).unwrap_or(0)
    }

    /// Whether no process-local task runtimes are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshots every registered task, optionally filtering by owner.
    pub fn snapshots(&self, owner_id: Option<&str>) -> Vec<DetachedTaskSnapshot<Metadata, Status>> {
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let mut snapshots: Vec<_> = guard
            .iter()
            .filter(|(_, entry)| owner_id.is_none_or(|owner| entry.owner_id == owner))
            .map(|(task_id, entry)| DetachedTaskSnapshot {
                task_id: task_id.clone(),
                owner_id: entry.owner_id.clone(),
                metadata: entry.metadata.clone(),
                status: entry.status.borrow().clone(),
            })
            .collect();
        snapshots.sort_by(|left, right| left.task_id.as_str().cmp(right.task_id.as_str()));
        snapshots
    }

    /// Returns a live steering handle after enforcing ownership and terminal
    /// status. The handle is looked up in the supplied [`SteeringRegistry`], so
    /// executors may register it independently when their run starts.
    pub fn steering_handle(
        &self,
        task_id: &TaskId,
        owner_id: &str,
    ) -> std::result::Result<SteeringHandle, DetachedTaskRegistryError> {
        self.ensure_live_owned(task_id, owner_id)?;
        self.steering
            .get(task_id)
            .ok_or(DetachedTaskRegistryError::NoSteeringHandle)
    }

    /// Trusted-control variant of [`Self::steering_handle`] that does not
    /// require an owner id. It still rejects unknown and terminal tasks.
    pub fn steering_handle_trusted(
        &self,
        task_id: &TaskId,
    ) -> std::result::Result<SteeringHandle, DetachedTaskRegistryError> {
        self.ensure_live(task_id)?;
        self.steering
            .get(task_id)
            .ok_or(DetachedTaskRegistryError::NoSteeringHandle)
    }

    /// Waits for a terminal status or `timeout`. A terminal result prunes the
    /// process-local runtime; a timeout leaves it available for another wait.
    pub async fn wait(
        &self,
        task_id: &TaskId,
        owner_id: &str,
        timeout: Duration,
    ) -> std::result::Result<DetachedTaskWaitOutcome<Status>, DetachedTaskRegistryError> {
        let mut status = {
            let guard = self
                .inner
                .lock()
                .map_err(|_| DetachedTaskRegistryError::Unknown)?;
            let entry = guard
                .get(task_id)
                .ok_or(DetachedTaskRegistryError::Unknown)?;
            if entry.owner_id != owner_id {
                return Err(DetachedTaskRegistryError::NotOwned);
            }
            entry.status.clone()
        };

        let current = status.borrow_and_update().clone();
        if (self.is_terminal)(&current) {
            self.remove(task_id);
            return Ok(DetachedTaskWaitOutcome::Terminal(current));
        }

        let waited = async {
            loop {
                status
                    .changed()
                    .await
                    .map_err(|_| DetachedTaskRegistryError::StatusChannelClosed)?;
                let current = status.borrow().clone();
                if (self.is_terminal)(&current) {
                    return Ok(current);
                }
            }
        };

        match tokio::time::timeout(timeout, waited).await {
            Ok(Ok(terminal)) => {
                self.remove(task_id);
                Ok(DetachedTaskWaitOutcome::Terminal(terminal))
            }
            Ok(Err(error)) => Err(error),
            Err(_) => Ok(DetachedTaskWaitOutcome::TimedOut(status.borrow().clone())),
        }
    }

    /// Cancels and removes an owned task, returning its application metadata.
    pub fn cancel(
        &self,
        task_id: &TaskId,
        owner_id: &str,
    ) -> std::result::Result<CancelledDetachedTask<Metadata, Status>, DetachedTaskRegistryError>
    {
        self.take_and_cancel(task_id, Some(owner_id))
    }

    /// Trusted-control variant of [`Self::cancel`] that does not require an
    /// owner id.
    pub fn cancel_trusted(
        &self,
        task_id: &TaskId,
    ) -> std::result::Result<CancelledDetachedTask<Metadata, Status>, DetachedTaskRegistryError>
    {
        self.take_and_cancel(task_id, None)
    }

    /// Cancels every task whose metadata matches `predicate`.
    pub fn cancel_where(
        &self,
        predicate: impl Fn(&Metadata) -> bool,
    ) -> Vec<CancelledDetachedTask<Metadata, Status>> {
        let task_ids: Vec<TaskId> = self
            .inner
            .lock()
            .map(|guard| {
                guard
                    .iter()
                    .filter(|(_, entry)| predicate(&entry.metadata))
                    .map(|(task_id, _)| task_id.clone())
                    .collect()
            })
            .unwrap_or_default();
        task_ids
            .into_iter()
            .filter_map(|task_id| self.cancel_trusted(&task_id).ok())
            .collect()
    }

    /// Cancels and removes every registered detached task.
    pub fn cancel_all(&self) -> Vec<CancelledDetachedTask<Metadata, Status>> {
        self.cancel_where(|_| true)
    }

    /// Prunes every entry whose latest watched status is terminal.
    pub fn sweep_terminal(&self) -> usize {
        let removed = {
            let Ok(mut guard) = self.inner.lock() else {
                return 0;
            };
            let task_ids: Vec<_> = guard
                .iter()
                .filter(|(_, entry)| (self.is_terminal)(&entry.status.borrow()))
                .map(|(task_id, _)| task_id.clone())
                .collect();
            for task_id in &task_ids {
                guard.remove(task_id);
            }
            task_ids
        };
        for task_id in &removed {
            self.steering.deregister(task_id);
        }
        removed.len()
    }

    fn ensure_live_owned(
        &self,
        task_id: &TaskId,
        owner_id: &str,
    ) -> std::result::Result<(), DetachedTaskRegistryError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| DetachedTaskRegistryError::Unknown)?;
        let entry = guard
            .get(task_id)
            .ok_or(DetachedTaskRegistryError::Unknown)?;
        if entry.owner_id != owner_id {
            return Err(DetachedTaskRegistryError::NotOwned);
        }
        if (self.is_terminal)(&entry.status.borrow()) {
            return Err(DetachedTaskRegistryError::AlreadyDone);
        }
        Ok(())
    }

    fn ensure_live(&self, task_id: &TaskId) -> std::result::Result<(), DetachedTaskRegistryError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| DetachedTaskRegistryError::Unknown)?;
        let entry = guard
            .get(task_id)
            .ok_or(DetachedTaskRegistryError::Unknown)?;
        if (self.is_terminal)(&entry.status.borrow()) {
            return Err(DetachedTaskRegistryError::AlreadyDone);
        }
        Ok(())
    }

    fn take_and_cancel(
        &self,
        task_id: &TaskId,
        owner_id: Option<&str>,
    ) -> std::result::Result<CancelledDetachedTask<Metadata, Status>, DetachedTaskRegistryError>
    {
        let entry = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| DetachedTaskRegistryError::Unknown)?;
            let entry = guard
                .get(task_id)
                .ok_or(DetachedTaskRegistryError::Unknown)?;
            if owner_id.is_some_and(|owner| entry.owner_id != owner) {
                return Err(DetachedTaskRegistryError::NotOwned);
            }
            guard
                .remove(task_id)
                .ok_or(DetachedTaskRegistryError::Unknown)?
        };
        self.steering.deregister(task_id);
        entry.cancellation.cancel();
        entry.abort.abort();
        Ok(CancelledDetachedTask {
            task_id: task_id.clone(),
            owner_id: entry.owner_id,
            metadata: entry.metadata,
            status: entry.status.borrow().clone(),
        })
    }

    fn remove(&self, task_id: &TaskId) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.remove(task_id);
        }
        self.steering.deregister(task_id);
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<TaskId, DetachedTaskEntry<Metadata, Status>>>>
    {
        self.inner.lock().map_err(|_| {
            TinyAgentsError::Graph("detached task runtime registry lock poisoned".into())
        })
    }
}
