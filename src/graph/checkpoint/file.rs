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

use super::{Checkpoint, CheckpointMetadata, Checkpointer};
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
        let records = self.read_records(thread_id)?;
        let found = match checkpoint_id {
            Some(id) => records.into_iter().rev().find(|c| c.checkpoint_id == id),
            None => records.into_iter().next_back(),
        };
        Ok(found)
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        Ok(self
            .read_records(thread_id)?
            .iter()
            .map(Checkpoint::to_metadata)
            .collect())
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
