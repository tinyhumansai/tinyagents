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

use super::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointTuple, Checkpointer, PendingWrite,
    merge_writes,
};
use crate::harness::ids::CheckpointId;
use crate::{Result, TinyAgentsError};

/// File extension for per-thread checkpoint logs.
const THREAD_EXT: &str = "jsonl";

/// Filename suffix for a thread's **pending-writes** sidecar.
///
/// Writes are recorded after their checkpoint is already durable, so they
/// cannot live in the append-only checkpoint log without turning it into a
/// mixed-record format that every reader would have to discriminate. A sibling
/// file keeps the checkpoint log exactly as it was.
const WRITES_SUFFIX: &str = ".writes.jsonl";

/// Process-wide counter making temp-file names unique so concurrent atomic
/// rewrites of the same thread never collide on their scratch file.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One line of a thread's pending-writes sidecar: the write plus the
/// `(namespace, checkpoint_id)` it is filed under.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct WriteRecord {
    #[serde(default)]
    namespace: Vec<String>,
    checkpoint_id: String,
    write: PendingWrite,
}

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
        let canonical = self.canonical_thread_path(thread_id);
        if canonical.exists() {
            canonical
        } else {
            let legacy = self.legacy_thread_path(thread_id);
            if legacy.exists() { legacy } else { canonical }
        }
    }

    /// Resolves the pending-writes sidecar path for `thread_id`.
    fn writes_path(&self, thread_id: &str) -> PathBuf {
        let canonical = self.canonical_writes_path(thread_id);
        if canonical.exists() {
            canonical
        } else {
            let legacy = self.legacy_writes_path(thread_id);
            if legacy.exists() { legacy } else { canonical }
        }
    }

    fn canonical_thread_path(&self, thread_id: &str) -> PathBuf {
        self.base_dir
            .join(format!("{}.{THREAD_EXT}", escape_thread_id(thread_id)))
    }

    fn canonical_writes_path(&self, thread_id: &str) -> PathBuf {
        self.base_dir
            .join(format!("{}{WRITES_SUFFIX}", escape_thread_id(thread_id)))
    }

    fn legacy_thread_path(&self, thread_id: &str) -> PathBuf {
        self.base_dir.join(format!(
            "{}.{THREAD_EXT}",
            legacy_escape_thread_id(thread_id)
        ))
    }

    fn legacy_writes_path(&self, thread_id: &str) -> PathBuf {
        self.base_dir.join(format!(
            "{}{WRITES_SUFFIX}",
            legacy_escape_thread_id(thread_id)
        ))
    }

    /// Reads a thread's write sidecar, tolerating a torn trailing line exactly
    /// as [`FileCheckpointer::read_records`] does.
    fn read_write_records(path: &Path, thread_id: &str) -> Result<Vec<WriteRecord>> {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_err("open writes file", e)),
        };
        decode_lines(&text, &format!("writes for thread `{thread_id}`"), |line| {
            serde_json::from_str::<WriteRecord>(line)
        })
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

/// Percent-escapes any byte outside `[a-z0-9._-]` so a thread id maps to a
/// single filename component that is injective **even on a case-insensitive
/// filesystem**.
///
/// # Why uppercase is escaped
///
/// The obvious safe set is `[A-Za-z0-9._-]`, and that is what this used to use.
/// It is injective on a case-*sensitive* filesystem and silently is not on
/// APFS, HFS+ or NTFS: threads `"Alice"` and `"alice"` map to `Alice.jsonl` and
/// `alice.jsonl`, which are the *same file*. Two unrelated runs then append into
/// one lineage, and reads hand each of them the other's checkpoints.
///
/// Escaping `A-Z` fixes it while staying case-*preserving* (the id is still
/// recoverable byte-for-byte from the name). The only uppercase characters left
/// in the output are the hex digits `A-F` of an escape, and escapes are always
/// emitted as `%` + exactly two uppercase hex digits, so no two outputs can
/// differ only by letter case: lowercasing the whole name is injective on the
/// image, which is exactly what case-insensitive collision-freedom means.
///
fn escape_thread_id(thread_id: &str) -> String {
    let mut out = String::with_capacity(thread_id.len());
    for &b in thread_id.as_bytes() {
        if b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{b:02X}"));
        }
    }
    out
}

/// The filename escaping used before uppercase letters were made explicit.
/// Kept only as a read/write fallback so persisted threads remain reachable
/// after an upgrade; new thread files always use [`escape_thread_id`].
fn legacy_escape_thread_id(thread_id: &str) -> String {
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
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_err("open thread file", e)),
        };
        decode_lines(&text, &format!("thread `{thread_id}`"), |line| {
            serde_json::from_str::<Checkpoint<State>>(line)
        })
    }
}

/// Decodes one JSON object per line, tolerating a **torn trailing line**.
///
/// A crash between `write_all` and the OS flushing the tail of the buffer
/// leaves a partial final line. It can only ever be the last one — the file is
/// append-only — so that is the only line whose decode failure is forgiven, and
/// only when the file does not end in a newline (a complete record always
/// does). Anything else is real corruption and still errors.
///
/// This matters more than "one lost record": the previous behaviour failed the
/// whole read, so a single torn byte made a thread permanently unreadable, with
/// no way to get at the hundreds of intact checkpoints in front of it.
fn decode_lines<T, F>(text: &str, what: &str, mut decode: F) -> Result<Vec<T>>
where
    F: FnMut(&str) -> std::result::Result<T, serde_json::Error>,
{
    let complete = text.is_empty() || text.ends_with('\n');
    let lines: Vec<&str> = text.lines().collect();
    let last_index = lines.len().saturating_sub(1);
    let mut out = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match decode(line) {
            Ok(record) => out.push(record),
            Err(e) if !complete && i == last_index => {
                tracing::warn!(
                    "[checkpoint:file] {what}: discarding torn trailing line \
                     ({} bytes, no terminating newline): {e}",
                    line.len()
                );
            }
            Err(e) => return Err(io_err("decode record", e)),
        }
    }
    Ok(out)
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
            write_atomic(&path, buf.as_bytes())
        }
    }
}

/// Writes `bytes` to `path` atomically: a uniquely named temp file in the same
/// directory, fsynced, then renamed over the destination.
///
/// The prune/delete path used to rewrite the thread file **in place** with
/// `fs::write`, which truncates first: a crash anywhere in the following write
/// leaves a truncated or empty file, and the whole history is gone — not the
/// pruned tail, all of it. Rename is atomic for same-directory paths on POSIX
/// and Windows, so a reader sees either the old file or the new one, and a
/// crash leaves the old one intact. This is the same shape `FileStore::put`
/// already uses.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().ok_or_else(|| {
        TinyAgentsError::Checkpoint(format!(
            "file checkpointer: path has no parent directory: {}",
            path.display()
        ))
    })?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("thread");
    let tmp = dir.join(format!(
        "{file_name}.tmp.{}.{}",
        std::process::id(),
        TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let write_and_sync = || -> std::io::Result<()> {
        let mut file = File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()
    };
    if let Err(e) = write_and_sync() {
        let _ = fs::remove_file(&tmp);
        return Err(io_err("write temp thread file", e));
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(io_err("rename temp thread file", e));
    }
    Ok(())
}

#[async_trait]
impl<State> Checkpointer<State> for FileCheckpointer<State>
where
    State: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId> {
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        // The serialize + filesystem append is synchronous, blocking work; run
        // it on the blocking pool so it never stalls a tokio worker on the
        // step-critical path.
        let base_dir = self.base_dir.clone();
        let path = self.thread_path(&checkpoint.thread_id);
        tokio::task::spawn_blocking(move || -> Result<()> {
            fs::create_dir_all(&base_dir).map_err(|e| io_err("create base dir", e))?;
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
            // Without an explicit flush to stable storage a "persisted"
            // checkpoint is only in the page cache: a host crash loses
            // boundaries the executor has already reported as durable, and can
            // leave a torn trailing line behind (which `read_records` now
            // tolerates, but should not have to see).
            file.sync_all().map_err(|e| io_err("fsync record", e))
        })
        .await
        .map_err(|e| io_err("join blocking put task", e))??;
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

    async fn get_thread(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        // Single-pass bulk read: parse the thread file once, instead of the
        // default's one whole-file `get` scan per listed id (O(H²)).
        self.read_records(thread_id)
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
            // Match on the filename suffix rather than `Path::extension()`.
            // The empty thread id escapes to the empty string, so its file is
            // literally `.jsonl` — a dotfile whose `extension()` is `None`,
            // which made that thread invisible to listing (and to everything
            // built on listing) while `get`/`put` addressed it perfectly well.
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.ends_with(&format!(".{THREAD_EXT}")) || name.ends_with(WRITES_SUFFIX) {
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
                // One unreadable file must not take down the whole listing.
                // `list_threads` decodes the first line of *every* file, so an
                // error here made a single poisoned thread break listing —
                // and therefore every operation built on it — globally.
                match serde_json::from_str::<Checkpoint<serde::de::IgnoredAny>>(&first) {
                    Ok(record) => threads.push(record.thread_id),
                    Err(e) => tracing::warn!(
                        "[checkpoint:file] list_threads: skipping unreadable thread file {}: {e}",
                        path.display()
                    ),
                }
                break;
            }
        }
        Ok(threads)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        // The write sidecar goes with the thread: leaving it behind would let a
        // later thread of the same id inherit a dead ledger.
        for path in [self.thread_path(thread_id), self.writes_path(thread_id)] {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_err("delete thread file", e)),
            }
        }
        Ok(())
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
            // Drop the deleted checkpoints' write ledgers with them.
            let writes_path = self.writes_path(thread_id);
            let write_records = Self::read_write_records(&writes_path, thread_id)?;
            let kept: Vec<&WriteRecord> = write_records
                .iter()
                .filter(|r| !drop.contains(r.checkpoint_id.as_str()))
                .collect();
            if kept.len() != write_records.len() {
                let mut buf = String::new();
                for record in kept {
                    let line =
                        serde_json::to_string(record).map_err(|e| io_err("encode write", e))?;
                    buf.push_str(&line);
                    buf.push('\n');
                }
                if buf.is_empty() {
                    match fs::remove_file(&writes_path) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(io_err("remove empty writes file", e)),
                    }
                } else {
                    write_atomic(&writes_path, buf.as_bytes())?;
                }
            }
        }
        Ok(removed)
    }

    async fn put_writes(&self, config: &CheckpointConfig, writes: &[PendingWrite]) -> Result<()> {
        let checkpoint_id = super::require_checkpoint_id(config)?;
        if writes.is_empty() {
            return Ok(());
        }
        let path = self.writes_path(&config.thread_id);
        let mut records = Self::read_write_records(&path, &config.thread_id)?;

        // Split out this checkpoint's ledger, merge, then rebuild the file.
        let (mut mine, others): (Vec<WriteRecord>, Vec<WriteRecord>) = records
            .drain(..)
            .partition(|r| r.checkpoint_id == checkpoint_id && r.namespace == config.namespace);
        let mut existing: Vec<PendingWrite> = mine.drain(..).map(|r| r.write).collect();
        let changed = merge_writes(&mut existing, writes);

        let mut buf = String::new();
        for record in others.iter() {
            let line = serde_json::to_string(record).map_err(|e| io_err("encode write", e))?;
            buf.push_str(&line);
            buf.push('\n');
        }
        for write in existing {
            let record = WriteRecord {
                namespace: config.namespace.clone(),
                checkpoint_id: checkpoint_id.clone(),
                write,
            };
            let line = serde_json::to_string(&record).map_err(|e| io_err("encode write", e))?;
            buf.push_str(&line);
            buf.push('\n');
        }
        fs::create_dir_all(&self.base_dir).map_err(|e| io_err("create base dir", e))?;
        write_atomic(&path, buf.as_bytes())?;
        tracing::debug!(
            "[checkpoint:file] put_writes thread={} checkpoint={checkpoint_id} offered={} stored={changed}",
            config.thread_id,
            writes.len()
        );
        Ok(())
    }

    async fn get_writes(&self, config: &CheckpointConfig) -> Result<Vec<PendingWrite>> {
        let Some(checkpoint_id) = self.resolve_write_target(config).await? else {
            return Ok(Vec::new());
        };
        let path = self.writes_path(&config.thread_id);
        Ok(Self::read_write_records(&path, &config.thread_id)?
            .into_iter()
            .filter(|r| r.checkpoint_id == checkpoint_id && r.namespace == config.namespace)
            .map(|r| r.write)
            .collect())
    }
}
