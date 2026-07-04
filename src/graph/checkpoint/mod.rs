//! Checkpointer trait and in-memory backend — the durability layer that makes
//! the recursive graph runtime resumable and time-travelable.
//!
//! In a recursive-language-model harness, runs nest: a graph node can run
//! another compiled graph, which can run another, each producing its own state.
//! Checkpointing snapshots every level of that tree at superstep boundaries and
//! keys them by `thread_id`/`namespace` so a parent and its embedded subgraphs
//! never collide (see [`crate::graph::subgraph`]). Persisting committed state at
//! each boundary is what lets a run be paused on an interrupt, resumed later,
//! forked, or replayed for time-travel debugging.
//!
//! See [`types`] for the checkpoint record definitions. Checkpoints are written
//! at superstep boundaries only — never mid-node — so resuming always reruns a
//! node from its start.

mod file;
#[cfg(feature = "sqlite")]
mod sqlite;
mod types;

pub use file::FileCheckpointer;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCheckpointer;
pub use types::{
    BarrierArrivals, Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSource,
    CheckpointTuple, DurabilityMode, PendingActivation, PendingWrite,
};

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::harness::ids::CheckpointId;
use crate::{Result, TinyAgentsError};

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

    /// Loads a checkpoint for a thread scoped to `namespace`.
    ///
    /// Like [`Checkpointer::get`], but only considers checkpoints whose stored
    /// namespace equals `namespace`. This is what keeps a parent run and the
    /// subgraphs it embeds — which share a thread id but differ in namespace —
    /// from loading each other's checkpoints on resume/inspection. With
    /// `checkpoint_id == None` the latest checkpoint *in that namespace* is
    /// returned (last-write-wins, consistent with [`Checkpointer::get`]).
    ///
    /// Composed from [`Checkpointer::list`] + [`Checkpointer::get`] so every
    /// backend inherits it; override only for a cheaper scoped query.
    async fn get_scoped(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &[String],
    ) -> Result<Option<Checkpoint<State>>> {
        let metas = self.list(thread_id).await?;
        let target: Option<String> = match checkpoint_id {
            Some(id) => metas
                .iter()
                .rev()
                .find(|m| m.checkpoint_id == id && m.namespace.as_slice() == namespace)
                .map(|m| m.checkpoint_id.clone()),
            None => metas
                .iter()
                .rev()
                .find(|m| m.namespace.as_slice() == namespace)
                .map(|m| m.checkpoint_id.clone()),
        };
        match target {
            Some(id) => self.get(thread_id, Some(&id)).await,
            None => Ok(None),
        }
    }

    /// Lists checkpoint metadata for a thread in insertion order.
    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>>;

    /// Loads every checkpoint stored under `thread_id`, in listing order.
    ///
    /// This is the bulk-read companion to [`Checkpointer::list`]: it returns
    /// full [`Checkpoint`] records (including state) rather than metadata, so
    /// whole-thread operations such as [`Checkpointer::copy_thread`] can read
    /// a thread once instead of issuing one [`Checkpointer::get`] per
    /// checkpoint.
    ///
    /// The default is composed from [`Checkpointer::list`] +
    /// [`Checkpointer::get`], which re-resolves each id individually (and, on
    /// a backend whose `get` scans the whole thread, is O(H²)). Every bundled
    /// backend overrides it with a single-pass read; other backends should do
    /// the same when they can.
    async fn get_thread(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        let metas = self.list(thread_id).await?;
        let mut out = Vec::with_capacity(metas.len());
        for meta in metas {
            if let Some(checkpoint) = self.get(thread_id, Some(&meta.checkpoint_id)).await? {
                out.push(checkpoint);
            }
        }
        Ok(out)
    }

    /// Loads a [`CheckpointTuple`] — the checkpoint plus its addressing config,
    /// its parent's config, and the pending writes carried with it.
    ///
    /// Composed from [`Checkpointer::get`] so every backend gets it for free;
    /// override it only when a backend can build the tuple more cheaply. When
    /// `config.checkpoint_id` is `None` the latest checkpoint is returned.
    async fn get_tuple(&self, config: CheckpointConfig) -> Result<Option<CheckpointTuple<State>>> {
        let Some(checkpoint) = self
            .get_scoped(
                &config.thread_id,
                config.checkpoint_id.as_deref(),
                &config.namespace,
            )
            .await?
        else {
            return Ok(None);
        };
        let resolved = CheckpointConfig {
            thread_id: checkpoint.thread_id.clone(),
            checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
            namespace: checkpoint.namespace.clone(),
        };
        let parent_config =
            checkpoint
                .parent_checkpoint_id
                .as_ref()
                .map(|parent| CheckpointConfig {
                    thread_id: checkpoint.thread_id.clone(),
                    checkpoint_id: Some(parent.clone()),
                    namespace: checkpoint.namespace.clone(),
                });
        let pending_writes = checkpoint.pending_writes.clone();
        Ok(Some(CheckpointTuple {
            config: resolved,
            checkpoint,
            parent_config,
            pending_writes,
        }))
    }

    /// Returns a thread's checkpoint lineage newest-first, following each
    /// checkpoint's `parent_checkpoint_id` from the latest checkpoint in
    /// `namespace`. `limit` caps the number of tuples returned (the most recent
    /// ones).
    ///
    /// The default walks [`Checkpointer::get_tuple`] once per hop, so a backend
    /// that re-reads the whole thread per lookup (the file/JSONL backend) is
    /// O(H²) over the lineage. Such backends override this to read the thread
    /// once and walk the lineage in memory (O(H)). The observable result is
    /// identical to iterating `get_tuple` by parent pointer.
    async fn state_history(
        &self,
        thread_id: &str,
        namespace: &[String],
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple<State>>> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            if let Some(limit) = limit
                && out.len() >= limit
            {
                break;
            }
            let config = CheckpointConfig {
                thread_id: thread_id.to_string(),
                checkpoint_id: cursor.clone(),
                namespace: namespace.to_vec(),
            };
            let Some(tuple) = self.get_tuple(config).await? else {
                break;
            };
            let parent = tuple.checkpoint.parent_checkpoint_id.clone();
            out.push(tuple);
            match parent {
                Some(parent) => cursor = Some(parent),
                None => break,
            }
        }
        Ok(out)
    }

    // ---- Thread operations -------------------------------------------------
    //
    // Three storage-specific primitives (`list_threads`, `delete_thread`,
    // `delete_checkpoints`) have no default body. The higher-level operations
    // (`delete_by_run`, `copy_thread`, `prune`) are composed from those plus
    // the existing `list`/`get_thread`/`put` surface, so every backend
    // inherits them for free and only implements the three storage primitives
    // (overriding `get_thread` when a single-pass bulk read is available).

    /// Lists the ids of every thread that currently has at least one checkpoint.
    ///
    /// Order is backend-defined. Storage-specific: there is no default body.
    async fn list_threads(&self) -> Result<Vec<String>>;

    /// Deletes every checkpoint stored under `thread_id`.
    ///
    /// A no-op (still `Ok`) when the thread is unknown. Storage-specific.
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;

    /// Low-level primitive: removes the named checkpoints from `thread_id`,
    /// returning how many were actually removed.
    ///
    /// Ids not present are ignored. The default thread operations
    /// ([`Checkpointer::delete_by_run`], [`Checkpointer::prune`]) are built on
    /// top of this. Storage-specific: there is no default body.
    async fn delete_checkpoints(&self, thread_id: &str, ids: &[String]) -> Result<usize>;

    /// Deletes every checkpoint in `thread_id` stamped with `run_id`, returning
    /// the number removed.
    ///
    /// Run ids are recorded on checkpoints by the executor; records that predate
    /// run-id stamping (or were written manually) carry `None` and are never
    /// matched. Composed from [`Checkpointer::list`] +
    /// [`Checkpointer::delete_checkpoints`].
    async fn delete_by_run(&self, thread_id: &str, run_id: &str) -> Result<usize> {
        let ids: Vec<String> = self
            .list(thread_id)
            .await?
            .into_iter()
            .filter(|m| m.run_id.as_deref() == Some(run_id))
            .map(|m| m.checkpoint_id)
            .collect();
        self.delete_checkpoints(thread_id, &ids).await
    }

    /// Deep-copies every checkpoint from `source_thread` into `target_thread`,
    /// rewriting only the `thread_id` while preserving each record's
    /// `checkpoint_id` and `parent_checkpoint_id`.
    ///
    /// Because checkpoint ids are unique only within a thread, reusing them in
    /// the target keeps the parent lineage spine intact, so time-travel and
    /// resume walk the copied thread exactly as they would the source. Records
    /// are copied in listing order so parents always precede their children.
    /// Composed from [`Checkpointer::get_thread`] + [`Checkpointer::put`], so
    /// the source thread is read once (a bulk read, not one
    /// [`Checkpointer::get`] per checkpoint).
    async fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        for mut checkpoint in self.get_thread(source_thread).await? {
            checkpoint.thread_id = target_thread.to_string();
            self.put(checkpoint).await?;
        }
        Ok(())
    }

    /// Prunes old checkpoints from `thread_id`, retaining the most recent
    /// `keep_last` plus everything they depend on, and returns the number
    /// deleted.
    ///
    /// Strategy (lineage- and delta-safe):
    ///
    /// 1. Protect the most recent `keep_last` checkpoints (listing order).
    /// 2. Walk the `parent_checkpoint_id` chain of every protected checkpoint
    ///    and protect every ancestor reached. This is what honors the
    ///    delta-channel warning: a kept checkpoint that only stores a delta (or
    ///    depends on an ancestor's pending writes / snapshot) keeps its entire
    ///    ancestor chain, so it can never be left dangling without the state it
    ///    needs to be reconstructed or resumed.
    /// 3. Delete every checkpoint not in the protected set.
    ///
    /// `keep_last == 0` is treated as `keep_last == 1`: the latest checkpoint
    /// (and its ancestors) is always retained so the thread stays resumable.
    /// Composed from [`Checkpointer::list`] + [`Checkpointer::delete_checkpoints`].
    async fn prune(&self, thread_id: &str, keep_last: usize) -> Result<usize> {
        let metas = self.list(thread_id).await?;
        if metas.is_empty() {
            return Ok(0);
        }
        let keep_last = keep_last.max(1).min(metas.len());

        // Index by id so ancestor walks are O(depth).
        let mut parent_of: HashMap<&str, Option<&str>> = HashMap::new();
        for m in &metas {
            parent_of.insert(m.checkpoint_id.as_str(), m.parent_checkpoint_id.as_deref());
        }

        let mut protected: HashSet<String> = HashSet::new();
        // Step 1: the recency window.
        for m in metas.iter().rev().take(keep_last) {
            protected.insert(m.checkpoint_id.clone());
        }
        // Step 2: expand to every ancestor of a protected checkpoint.
        let window: Vec<String> = protected.iter().cloned().collect();
        for start in window {
            let mut cursor = parent_of.get(start.as_str()).copied().flatten();
            while let Some(parent) = cursor {
                if !protected.insert(parent.to_string()) {
                    break; // already protected — its chain is too.
                }
                cursor = parent_of.get(parent).copied().flatten();
            }
        }

        // Step 3: delete the rest.
        let to_delete: Vec<String> = metas
            .iter()
            .map(|m| m.checkpoint_id.clone())
            .filter(|id| !protected.contains(id))
            .collect();
        self.delete_checkpoints(thread_id, &to_delete).await
    }
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

fn lock_err() -> TinyAgentsError {
    TinyAgentsError::Checkpoint("in-memory checkpointer lock poisoned".to_string())
}

fn metadata_of<State>(c: &Checkpoint<State>) -> CheckpointMetadata {
    c.to_metadata()
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
        // Duplicate-id lookup resolves to the *last* written record, matching
        // the append-only file/sqlite backends (and `get(None)`, which returns
        // the latest). Pinning one semantic keeps the three backends
        // interchangeable — see the checkpointer conformance suite.
        let found = match checkpoint_id {
            Some(id) => list.iter().rfind(|c| c.checkpoint_id == id),
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

    async fn get_thread(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        // Single-pass bulk read: clone the thread's records in insertion
        // order, instead of the default's one `get` per listed id.
        let map = self.inner.lock().map_err(|_| lock_err())?;
        Ok(map.get(thread_id).cloned().unwrap_or_default())
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        let map = self.inner.lock().map_err(|_| lock_err())?;
        Ok(map
            .iter()
            .filter(|(_, list)| !list.is_empty())
            .map(|(thread, _)| thread.clone())
            .collect())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let mut map = self.inner.lock().map_err(|_| lock_err())?;
        map.remove(thread_id);
        Ok(())
    }

    async fn delete_checkpoints(&self, thread_id: &str, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let drop: HashSet<&str> = ids.iter().map(String::as_str).collect();
        let mut map = self.inner.lock().map_err(|_| lock_err())?;
        let Some(list) = map.get_mut(thread_id) else {
            return Ok(0);
        };
        let before = list.len();
        list.retain(|c| !drop.contains(c.checkpoint_id.as_str()));
        Ok(before - list.len())
    }
}

#[cfg(test)]
mod test;
