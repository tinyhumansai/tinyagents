//! File-backed [`Checkpointer`] — a durable JSON/JSONL backend that survives
//! process restarts.
//!
//! Each thread maps to one append-only JSONL file under a base directory: one
//! checkpoint record (a serialized [`Checkpoint`]) per line, written in
//! insertion order. Reads stream the thread file line by line; deletes rewrite
//! (or remove) it; [`Checkpointer::copy_thread`] copies a thread's file while
//! rewriting only the `thread_id` on each record, so the parent lineage spine is
//! preserved exactly as in memory.
//!
//! The backend is generic over `State`, but only requires
//! `State: Serialize + DeserializeOwned` on the [`Checkpointer`] impl block — the
//! trait itself stays bound-free so the in-memory path keeps working for states
//! that are not serializable.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Minimal projection used to read a checkpoint's id without deserializing its
/// `State` payload, so `get` can pick the target line and decode only that one.
#[derive(serde::Deserialize)]
struct CheckpointIdHeader {
    checkpoint_id: String,
}

use super::{Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointTuple, Checkpointer};
use crate::harness::ids::CheckpointId;
use crate::{Result, TinyAgentsError};

/// File extension for per-thread checkpoint logs.
const THREAD_EXT: &str = "jsonl";

/// A [`Checkpointer`] that persists checkpoints as JSONL files under a base
/// directory, one file per thread.
///
/// Cheap to clone; clones address the same base directory. The base directory
/// is created lazily on the first write.
pub struct FileCheckpointer<State> {
    base_dir: PathBuf,
    _marker: PhantomData<fn() -> State>,
}

impl<State> FileCheckpointer<State> {
    /// Creates a checkpointer rooted at `base_dir`.
    ///
    /// The directory is not touched until the first checkpoint is written.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            _marker: PhantomData,
        }
    }

    /// Returns the base directory backing this checkpointer.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Resolves the JSONL file path for `thread_id`.
    ///
    /// The thread id is percent-escaped so it is a safe, injective single path
    /// component (no separators, no collisions between distinct ids).
    fn thread_path(&self, thread_id: &str) -> PathBuf {
        self.base_dir
            .join(format!("{}.{THREAD_EXT}", escape_thread_id(thread_id)))
    }
}

impl<State> Clone for FileCheckpointer<State> {
    fn clone(&self) -> Self {
        Self {
            base_dir: self.base_dir.clone(),
            _marker: PhantomData,
        }
    }
}

/// Percent-escapes any byte outside `[A-Za-z0-9._-]` so a thread id maps to a
/// single, collision-free filename component.
fn escape_thread_id(thread_id: &str) -> String {
    let mut out = String::with_capacity(thread_id.len());
    for &b in thread_id.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{b:02X}"));
        }
    }
    out
}

fn io_err(context: &str, err: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Checkpoint(format!("file checkpointer: {context}: {err}"))
}

/// Builds a [`CheckpointTuple`] from an owned checkpoint, mirroring the
/// addressing/parent/pending-writes wiring of the default
/// [`Checkpointer::get_tuple`].
fn tuple_from_checkpoint<State>(checkpoint: Checkpoint<State>) -> CheckpointTuple<State> {
    let config = CheckpointConfig {
        thread_id: checkpoint.thread_id.clone(),
        checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
        namespace: checkpoint.namespace.clone(),
    };
    let parent_config = checkpoint
        .parent_checkpoint_id
        .as_ref()
        .map(|parent| CheckpointConfig {
            thread_id: checkpoint.thread_id.clone(),
            checkpoint_id: Some(parent.clone()),
            namespace: checkpoint.namespace.clone(),
        });
    let pending_writes = checkpoint.pending_writes.clone();
    CheckpointTuple {
        config,
        checkpoint,
        parent_config,
        pending_writes,
    }
}

impl<State> FileCheckpointer<State>
where
    State: DeserializeOwned,
{
    /// Reads every record in `thread_id`'s file, in insertion order.
    ///
    /// Returns an empty vec when the thread file does not exist.
    fn read_records(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        let path = self.thread_path(thread_id);
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_err("open thread file", e)),
        };
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| io_err("read line", e))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: Checkpoint<State> =
                serde_json::from_str(&line).map_err(|e| io_err("decode record", e))?;
            records.push(record);
        }
        Ok(records)
    }
}

impl<State> FileCheckpointer<State>
where
    State: Serialize,
{
    /// Overwrites `thread_id`'s file with `records` (one JSON line each).
    ///
    /// When `records` is empty the file is removed so empty threads disappear
    /// from [`Checkpointer::list_threads`].
    fn write_records(&self, thread_id: &str, records: &[Checkpoint<State>]) -> Result<()> {
        let path = self.thread_path(thread_id);
        if records.is_empty() {
            match fs::remove_file(&path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(io_err("remove empty thread file", e)),
            }
        } else {
            fs::create_dir_all(&self.base_dir).map_err(|e| io_err("create base dir", e))?;
            let mut buf = String::new();
            for record in records {
                let line = serde_json::to_string(record).map_err(|e| io_err("encode record", e))?;
                buf.push_str(&line);
                buf.push('\n');
            }
            fs::write(&path, buf).map_err(|e| io_err("write thread file", e))
        }
    }
}

#[async_trait]
impl<State> Checkpointer<State> for FileCheckpointer<State>
where
    State: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId> {
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        fs::create_dir_all(&self.base_dir).map_err(|e| io_err("create base dir", e))?;
        let path = self.thread_path(&checkpoint.thread_id);
        let mut line =
            serde_json::to_string(&checkpoint).map_err(|e| io_err("encode record", e))?;
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| io_err("open thread file for append", e))?;
        file.write_all(line.as_bytes())
            .map_err(|e| io_err("append record", e))?;
        Ok(id)
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint<State>>> {
        // Stream lines and fully decode only the single target line, instead of
        // deserializing every record's `State` just to pick one. Selection
        // matches the previous `rev().find` / `next_back` semantics: the last
        // matching line (or the last line, for `None`) wins.
        let path = self.thread_path(thread_id);
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_err("open thread file", e)),
        };
        let reader = BufReader::new(file);
        let mut target: Option<String> = None;
        for line in reader.lines() {
            let line = line.map_err(|e| io_err("read line", e))?;
            if line.trim().is_empty() {
                continue;
            }
            match checkpoint_id {
                Some(id) => {
                    // Decode only the id header to test the match, not `State`.
                    let header: CheckpointIdHeader =
                        serde_json::from_str(&line).map_err(|e| io_err("decode header", e))?;
                    if header.checkpoint_id == id {
                        target = Some(line);
                    }
                }
                None => target = Some(line),
            }
        }
        match target {
            Some(line) => Ok(Some(
                serde_json::from_str(&line).map_err(|e| io_err("decode record", e))?,
            )),
            None => Ok(None),
        }
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        Ok(self
            .read_records(thread_id)?
            .iter()
            .map(Checkpoint::to_metadata)
            .collect())
    }

    async fn state_history(
        &self,
        thread_id: &str,
        namespace: &[String],
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple<State>>> {
        // Read the whole thread once, then walk the parent lineage in memory
        // (O(H)), instead of re-reading and re-parsing the file per hop (O(H²)).
        let records = self.read_records(thread_id)?;
        if records.is_empty() {
            return Ok(Vec::new());
        }

        // id -> checkpoint, last write wins for duplicate ids (matching `get`,
        // which takes the last matching record). Track the latest checkpoint in
        // the target namespace as the walk's starting point.
        let mut by_id: std::collections::HashMap<String, Checkpoint<State>> =
            std::collections::HashMap::with_capacity(records.len());
        let mut cursor: Option<String> = None;
        for record in records {
            if record.namespace.as_slice() == namespace {
                cursor = Some(record.checkpoint_id.clone());
            }
            by_id.insert(record.checkpoint_id.clone(), record);
        }

        let mut out = Vec::new();
        while let Some(id) = cursor {
            if let Some(limit) = limit
                && out.len() >= limit
            {
                break;
            }
            // `remove` doubles as a cycle guard: each id is visited at most once.
            let Some(checkpoint) = by_id.remove(&id) else {
                break;
            };
            // A checkpoint outside the target namespace is not visible under
            // namespace-scoped lookup, so the lineage walk stops (matching the
            // `get_scoped`-based default).
            if checkpoint.namespace.as_slice() != namespace {
                break;
            }
            cursor = checkpoint.parent_checkpoint_id.clone();
            out.push(tuple_from_checkpoint(checkpoint));
        }
        Ok(out)
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        let entries = match fs::read_dir(&self.base_dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_err("read base dir", e)),
        };
        let mut threads = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| io_err("read dir entry", e))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some(THREAD_EXT) {
                continue;
            }
            // Recover the canonical thread id from the first record rather than
            // un-escaping the filename, so the value always matches what was
            // persisted.
            let file = File::open(&path).map_err(|e| io_err("open thread file", e))?;
            let mut reader = BufReader::new(file);
            let mut first = String::new();
            loop {
                first.clear();
                let read = reader
                    .read_line(&mut first)
                    .map_err(|e| io_err("read line", e))?;
                if read == 0 {
                    break; // empty file — skip
                }
                if first.trim().is_empty() {
                    continue;
                }
                let record: Checkpoint<serde::de::IgnoredAny> =
                    serde_json::from_str(&first).map_err(|e| io_err("decode header", e))?;
                threads.push(record.thread_id);
                break;
            }
        }
        Ok(threads)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let path = self.thread_path(thread_id);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(io_err("delete thread file", e)),
        }
    }

    async fn delete_checkpoints(&self, thread_id: &str, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let drop: std::collections::HashSet<&str> = ids.iter().map(String::as_str).collect();
        let mut records = self.read_records(thread_id)?;
        let before = records.len();
        records.retain(|c| !drop.contains(c.checkpoint_id.as_str()));
        let removed = before - records.len();
        if removed > 0 {
            self.write_records(thread_id, &records)?;
        }
        Ok(removed)
    }

    async fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        let mut records = self.read_records(source_thread)?;
        if records.is_empty() {
            return Ok(());
        }
        for record in &mut records {
            record.thread_id = target_thread.to_string();
        }
        // Append onto any existing target file to match `put` semantics.
        let mut existing = self.read_records(target_thread)?;
        existing.extend(records);
        self.write_records(target_thread, &existing)
    }
}
