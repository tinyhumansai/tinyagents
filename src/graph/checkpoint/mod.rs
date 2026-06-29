//! Checkpointer trait and in-memory backend.
//!
//! See [`types`] for the checkpoint record definitions. Checkpoints are written
//! at superstep boundaries only.

mod types;

pub use types::{Checkpoint, CheckpointMetadata, PendingWrite};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::harness::ids::CheckpointId;
use crate::{Result, RustAgentsError};

/// Persists and retrieves graph checkpoints keyed by thread.
#[async_trait]
pub trait Checkpointer<State>: Send + Sync
where
    State: Send + Sync + 'static,
{
    /// Persists a checkpoint and returns its id.
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId>;

    /// Loads a checkpoint for a thread. When `checkpoint_id` is `None`, returns
    /// the latest checkpoint for the thread.
    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint<State>>>;

    /// Lists checkpoint metadata for a thread in insertion order.
    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>>;
}

/// An in-memory [`Checkpointer`] backed by an `Arc<Mutex<..>>`.
///
/// Cheap to clone; clones share the same underlying store.
pub struct InMemoryCheckpointer<State> {
    inner: Arc<Mutex<HashMap<String, Vec<Checkpoint<State>>>>>,
}

impl<State> InMemoryCheckpointer<State> {
    /// Creates an empty checkpointer.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Returns the number of checkpoints stored for a thread.
    pub fn count(&self, thread_id: &str) -> usize {
        self.inner
            .lock()
            .map(|m| m.get(thread_id).map(|v| v.len()).unwrap_or(0))
            .unwrap_or(0)
    }
}

impl<State> Default for InMemoryCheckpointer<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Clone for InMemoryCheckpointer<State> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

fn lock_err() -> RustAgentsError {
    RustAgentsError::Checkpoint("in-memory checkpointer lock poisoned".to_string())
}

fn metadata_of<State>(c: &Checkpoint<State>) -> CheckpointMetadata {
    let source = c
        .metadata
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("loop")
        .to_string();
    let step = c.metadata.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    CheckpointMetadata {
        thread_id: c.thread_id.clone(),
        checkpoint_id: c.checkpoint_id.clone(),
        parent_checkpoint_id: c.parent_checkpoint_id.clone(),
        namespace: c.namespace.clone(),
        next_nodes: c.next_nodes.clone(),
        has_interrupts: !c.interrupts.is_empty(),
        source,
        step,
    }
}

#[async_trait]
impl<State> Checkpointer<State> for InMemoryCheckpointer<State>
where
    State: Clone + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId> {
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        let mut map = self.inner.lock().map_err(|_| lock_err())?;
        map.entry(checkpoint.thread_id.clone())
            .or_default()
            .push(checkpoint);
        Ok(id)
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint<State>>> {
        let map = self.inner.lock().map_err(|_| lock_err())?;
        let Some(list) = map.get(thread_id) else {
            return Ok(None);
        };
        let found = match checkpoint_id {
            Some(id) => list.iter().find(|c| c.checkpoint_id == id),
            None => list.last(),
        };
        Ok(found.cloned())
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        let map = self.inner.lock().map_err(|_| lock_err())?;
        Ok(map
            .get(thread_id)
            .map(|list| list.iter().map(metadata_of).collect())
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod test;
