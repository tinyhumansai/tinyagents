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
    CheckpointTuple, DurabilityMode, PendingActivation, PendingWrite, WRITES_IDX_ERROR,
    WRITES_IDX_INTERRUPT, WRITES_IDX_RESUME, merge_writes,
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
    /// backend inherits it; override for a cheaper scoped query — both durable
    /// backends do, because the default costs a full thread scan per call and
    /// [`Checkpointer::state_history`] issues one per lineage hop.
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

    // ---- Pending writes ----------------------------------------------------
    //
    // The partial-failure protocol. A superstep can fail after some of its
    // tasks have already run; without a per-task record of what they wrote, a
    // resume cannot tell "already ran" from "not yet run" and re-executes their
    // side effects. `put_writes` records that work against the checkpoint it
    // belongs to; `get_writes` reads it back, and `get_tuple` surfaces it as
    // `CheckpointTuple::pending_writes` so resume can skip completed tasks.
    //
    // Both carry default no-op bodies so an out-of-tree `Checkpointer` keeps
    // compiling: such a backend simply never persists writes, which is exactly
    // the behaviour every backend had before the protocol existed.

    /// Records `writes` against the checkpoint addressed by `config`.
    ///
    /// `config.checkpoint_id` must name a specific checkpoint — writes belong
    /// to the boundary they were produced at, so a `None` id has no meaning and
    /// backends reject it.
    ///
    /// Idempotency follows [`PendingWrite`]'s replace-vs-ignore rule: a data
    /// write (`idx >= 0`) whose `(task_id, idx)` is already stored is ignored,
    /// while a control-plane write (`idx < 0`) replaces the stored value. Both
    /// are implemented through [`merge_writes`], so every backend agrees.
    ///
    /// The default body is a no-op returning `Ok(())`.
    async fn put_writes(&self, _config: &CheckpointConfig, _writes: &[PendingWrite]) -> Result<()> {
        Ok(())
    }

    /// Reads back the writes recorded against the checkpoint addressed by
    /// `config`, in insertion order.
    ///
    /// Returns an empty vec for an unknown checkpoint or one that has no
    /// writes. When `config.checkpoint_id` is `None` the latest checkpoint in
    /// `config.namespace` is resolved first.
    ///
    /// The default body returns an empty vec.
    async fn get_writes(&self, _config: &CheckpointConfig) -> Result<Vec<PendingWrite>> {
        Ok(Vec::new())
    }

    /// Resolves the checkpoint id a **read** of writes addresses.
    ///
    /// Unlike [`Checkpointer::put_writes`] (where an unaddressed id is a caller
    /// bug), a read may legitimately mean "the latest checkpoint in this
    /// namespace" — the same relaxation [`Checkpointer::get`] makes. Returns
    /// `None` when the thread/namespace has no checkpoint at all.
    async fn resolve_write_target(&self, config: &CheckpointConfig) -> Result<Option<String>> {
        match &config.checkpoint_id {
            Some(id) => Ok(Some(id.clone())),
            None => Ok(self
                .get_scoped(&config.thread_id, None, &config.namespace)
                .await?
                .map(|c| c.checkpoint_id)),
        }
    }

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
        let pending_writes = self.resolved_writes(&resolved, &checkpoint).await?;
        Ok(Some(CheckpointTuple {
            config: resolved,
            checkpoint,
            parent_config,
            pending_writes,
        }))
    }

    /// The writes to surface on a tuple for `checkpoint`.
    ///
    /// Prefers the separately persisted [`Checkpointer::get_writes`] records —
    /// the authoritative partial-failure ledger — and falls back to the
    /// checkpoint record's own inline `pending_writes` for backends that do not
    /// implement the write protocol (whose `get_writes` default returns empty)
    /// and for records written before it existed.
    async fn resolved_writes(
        &self,
        config: &CheckpointConfig,
        checkpoint: &Checkpoint<State>,
    ) -> Result<Vec<PendingWrite>> {
        let stored = self.get_writes(config).await?;
        if stored.is_empty() {
            Ok(checkpoint.pending_writes.clone())
        } else {
            Ok(stored)
        }
    }

    /// Returns a thread's checkpoint lineage newest-first, following each
    /// checkpoint's `parent_checkpoint_id` from the latest checkpoint in
    /// `namespace`. `limit` caps the number of tuples returned (the most recent
    /// ones).
    ///
    /// The default walks [`Checkpointer::get_tuple`] once per hop, so a backend
    /// whose scoped lookup re-reads the whole thread is O(H²) over the lineage.
    /// Every bundled backend overrides it to read the thread (or the
    /// namespace's rows) once and walk the lineage in memory, so none of the
    /// three is in that class: the JSONL backend parses its file once, and the
    /// SQLite backend issues one indexed range query — the
    /// `(thread_id, namespace, seq)` index is what makes the namespace scope
    /// expressible in SQL at all, and without it that backend silently fell
    /// back to this default. The observable result is identical to iterating
    /// `get_tuple` by parent pointer.
    ///
    /// The walk carries a **visited set**. `parent_checkpoint_id` is caller-set
    /// data, not a structurally enforced acyclic pointer: a hand-written
    /// checkpoint, a bad fork, or a `copy_thread` that reused ids can point a
    /// record at itself or at one of its descendants. With `limit == None` an
    /// unguarded walk then never terminates — it does not merely return a wrong
    /// answer, it hangs the caller. Revisiting an id ends the walk.
    /// [`FileCheckpointer`] has always had this guard (its in-memory `remove`
    /// doubles as one); this makes it uniform.
    async fn state_history(
        &self,
        thread_id: &str,
        namespace: &[String],
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple<State>>> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        let mut visited: HashSet<String> = HashSet::new();
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
            if !visited.insert(tuple.checkpoint.checkpoint_id.clone()) {
                tracing::warn!(
                    "[checkpoint] state_history: lineage cycle at checkpoint `{}` \
                     (thread `{thread_id}`); truncating the walk",
                    tuple.checkpoint.checkpoint_id
                );
                break;
            }
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
    ///
    /// # The target must be empty
    ///
    /// Copying preserves each record's `checkpoint_id` — that is what keeps the
    /// lineage spine intact — so appending a lineage onto a thread that already
    /// has one produces a file/table containing two disjoint lineages *with
    /// reused ids*. Every subsequent `get(Some(id))` then resolves to whichever
    /// copy was written last and the parent walk crosses between lineages: the
    /// thread is silently corrupt, with no error at the point of damage.
    ///
    /// So a non-empty target is **rejected** rather than merged into. Callers
    /// that genuinely want to overwrite call [`Checkpointer::delete_thread`]
    /// first, which makes the destructive intent explicit. Copying an empty or
    /// unknown source thread is a no-op (still `Ok`).
    async fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        let existing = self.list(target_thread).await?;
        if !existing.is_empty() {
            return Err(TinyAgentsError::Checkpoint(format!(
                "copy_thread: target thread `{target_thread}` already has {} checkpoint(s); \
                 copying would interleave two lineages with reused checkpoint ids. \
                 Delete the target first if replacing it is intended.",
                existing.len()
            )));
        }
        for mut checkpoint in self.get_thread(source_thread).await? {
            let source_config = CheckpointConfig {
                thread_id: source_thread.to_string(),
                checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
                namespace: checkpoint.namespace.clone(),
            };
            let writes = self.get_writes(&source_config).await?;
            checkpoint.thread_id = target_thread.to_string();
            let target_config = CheckpointConfig {
                thread_id: target_thread.to_string(),
                checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
                namespace: checkpoint.namespace.clone(),
            };
            self.put(checkpoint).await?;
            if !writes.is_empty() {
                self.put_writes(&target_config, &writes).await?;
            }
        }
        Ok(())
    }

    /// Prunes old checkpoints from `thread_id`, retaining the most recent
    /// `keep_last` plus everything they depend on, and returns the number
    /// deleted.
    ///
    /// Strategy (lineage- and delta-safe):
    ///
    /// 1. Protect the most recent `keep_last` checkpoints (listing order) *of
    ///    every namespace present in the thread*. An embedded subgraph writes
    ///    its checkpoints under the parent's thread id but its own namespace,
    ///    and its lineage is disjoint from the parent's (no parent-namespace
    ///    record ever references a child-namespace id), so a thread-wide
    ///    recency window would delete the child lineage outright and leave the
    ///    thread unresumable.
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
        let keep_last = keep_last.max(1);

        // Index by id so ancestor walks are O(depth).
        let mut parent_of: HashMap<&str, Option<&str>> = HashMap::new();
        for m in &metas {
            parent_of.insert(m.checkpoint_id.as_str(), m.parent_checkpoint_id.as_deref());
        }
        // Group by namespace so each lineage (the root run and every embedded
        // subgraph run sharing this thread) gets its own recency window.
        let mut by_namespace: HashMap<&Vec<String>, Vec<&CheckpointMetadata>> = HashMap::new();
        for m in &metas {
            by_namespace.entry(&m.namespace).or_default().push(m);
        }

        let mut protected: HashSet<String> = HashSet::new();
        // Step 1: the recency window, per namespace.
        for group in by_namespace.values() {
            for m in group.iter().rev().take(keep_last) {
                protected.insert(m.checkpoint_id.clone());
            }
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
    /// Pending writes keyed by `(thread_id, namespace, checkpoint_id)` — the
    /// same identity the SQL backends use as a primary key prefix.
    writes: Arc<Mutex<HashMap<WriteKey, Vec<PendingWrite>>>>,
}

/// The address a batch of pending writes is filed under.
type WriteKey = (String, Vec<String>, String);

impl<State> InMemoryCheckpointer<State> {
    /// Creates an empty checkpointer.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            writes: Arc::new(Mutex::new(HashMap::new())),
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
            writes: self.writes.clone(),
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
        // Writes are keyed by thread too, and deleting a thread must not leave
        // its write ledger behind for a later thread of the same name to
        // inherit. The conformance suite asserts exactly this.
        let mut writes = self.writes.lock().map_err(|_| lock_err())?;
        writes.retain(|(thread, _, _), _| thread != thread_id);
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
        let removed = before - list.len();
        drop_writes_for(&self.writes, thread_id, &drop)?;
        Ok(removed)
    }

    async fn put_writes(&self, config: &CheckpointConfig, writes: &[PendingWrite]) -> Result<()> {
        let checkpoint_id = require_checkpoint_id(config)?;
        if writes.is_empty() {
            return Ok(());
        }
        let key: WriteKey = (
            config.thread_id.clone(),
            config.namespace.clone(),
            checkpoint_id,
        );
        let mut map = self.writes.lock().map_err(|_| lock_err())?;
        let slot = map.entry(key).or_default();
        let changed = merge_writes(slot, writes);
        tracing::debug!(
            "[checkpoint:memory] put_writes thread={} checkpoint={:?} offered={} stored={}",
            config.thread_id,
            config.checkpoint_id,
            writes.len(),
            changed
        );
        Ok(())
    }

    async fn get_writes(&self, config: &CheckpointConfig) -> Result<Vec<PendingWrite>> {
        let Some(checkpoint_id) = self.resolve_write_target(config).await? else {
            return Ok(Vec::new());
        };
        let key: WriteKey = (
            config.thread_id.clone(),
            config.namespace.clone(),
            checkpoint_id,
        );
        let map = self.writes.lock().map_err(|_| lock_err())?;
        Ok(map.get(&key).cloned().unwrap_or_default())
    }
}

/// Removes the write ledgers of `ids` within `thread_id`.
fn drop_writes_for(
    writes: &Mutex<HashMap<WriteKey, Vec<PendingWrite>>>,
    thread_id: &str,
    ids: &HashSet<&str>,
) -> Result<()> {
    let mut map = writes.lock().map_err(|_| lock_err())?;
    map.retain(|(thread, _, checkpoint), _| {
        thread != thread_id || !ids.contains(checkpoint.as_str())
    });
    Ok(())
}

/// Extracts the checkpoint id a `put_writes` call addresses.
///
/// A write belongs to the boundary that produced it, so an unaddressed
/// (`None`) id is a caller bug rather than "the latest": silently filing the
/// writes against whatever checkpoint happens to be newest is precisely the
/// corruption the protocol exists to prevent.
pub(crate) fn require_checkpoint_id(config: &CheckpointConfig) -> Result<String> {
    config.checkpoint_id.clone().ok_or_else(|| {
        TinyAgentsError::Checkpoint(format!(
            "put_writes requires an explicit checkpoint_id (thread `{}`)",
            config.thread_id
        ))
    })
}

#[cfg(test)]
mod test;
