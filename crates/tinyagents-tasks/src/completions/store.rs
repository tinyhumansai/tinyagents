//! Completion stores: in-memory and append-only JSONL.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use tinyagents_harness::error::{Result, TinyAgentsError};

use super::types::{CompletionRecord, CompletionState};
use crate::store::run_blocking;

const LOG_PREFIX: &str = "[completion-store]";

fn store_err(what: &str, err: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Graph(format!("completion store: {what}: {err}"))
}

/// Durable bookkeeping behind [`CompletionRouter`](super::CompletionRouter).
///
/// A store holds the latest [`CompletionRecord`] per task id. It is a plain
/// persistence seam: the router serialises its read-modify-write sequences, so
/// an implementation only has to make `put` durable before it returns.
/// Synchronous on purpose, like [`TaskStore`](crate::TaskStore).
pub trait CompletionStore: Send + Sync {
    /// The latest record for `task_id`.
    fn get(&self, task_id: &str) -> Option<CompletionRecord>;

    /// Upserts a record. Durable stores persist it before returning.
    fn put(&self, record: &CompletionRecord) -> Result<()>;

    /// Latest records, for one parent or (`None`) for all. Unordered.
    fn list(&self, parent_key: Option<&str>) -> Vec<CompletionRecord>;

    /// Drops settled records (anything but `Pending`) last updated more than
    /// `retain` ago, returning how many went. Dropping a settled record also
    /// drops its dedupe, so pick a window longer than any child can outlive its
    /// parent's interest. The default keeps everything.
    fn compact(&self, retain: Duration) -> Result<usize> {
        let _ = retain;
        Ok(0)
    }
}

fn expired(record: &CompletionRecord, now: SystemTime, retain: Duration) -> bool {
    record.state != CompletionState::Pending
        && now
            .duration_since(record.updated_at)
            .is_ok_and(|age| age > retain)
}

/// Thread-safe in-memory [`CompletionStore`].
#[derive(Clone, Debug, Default)]
pub struct InMemoryCompletionStore {
    inner: Arc<Mutex<HashMap<String, CompletionRecord>>>,
}

impl InMemoryCompletionStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<String, CompletionRecord>>> {
        self.inner.lock().map_err(|_| store_err("lock", "poisoned"))
    }
}

impl CompletionStore for InMemoryCompletionStore {
    fn get(&self, task_id: &str) -> Option<CompletionRecord> {
        self.inner.lock().ok()?.get(task_id).cloned()
    }

    fn put(&self, record: &CompletionRecord) -> Result<()> {
        self.lock()?.insert(record.task_id.clone(), record.clone());
        Ok(())
    }

    fn list(&self, parent_key: Option<&str>) -> Vec<CompletionRecord> {
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        guard
            .values()
            .filter(|r| parent_key.is_none_or(|p| r.parent_key == p))
            .cloned()
            .collect()
    }

    fn compact(&self, retain: Duration) -> Result<usize> {
        let now = SystemTime::now();
        let mut guard = self.lock()?;
        let before = guard.len();
        guard.retain(|_, r| !expired(r, now, retain));
        Ok(before - guard.len())
    }
}

/// Append-only JSONL [`CompletionStore`].
///
/// Every `put` appends one JSON line and `fsync`s it, so a record the router
/// reported as stored survives a crash. Opening replays the log to the latest
/// record per task id.
///
/// A crash can leave a torn final line. Opening drops it (the write never
/// returned, so the router never reported it stored) and truncates the file
/// back to the last whole line; a complete line that does not parse is skipped
/// with a warning rather than blocking every other completion. `compact`
/// rewrites the log to one line per task through a temp file and an atomic
/// rename.
pub struct JsonlCompletionStore {
    path: PathBuf,
    inner: InMemoryCompletionStore,
    file: Mutex<std::fs::File>,
    /// Set when a failed append could not be rolled back; the log tail is then
    /// unrepaired, so every later write fails until the store is reopened.
    broken: std::sync::atomic::AtomicBool,
}

impl JsonlCompletionStore {
    /// Opens (creating if needed) the log at `path` and replays it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| store_err("create dir", e))?;
        }
        let latest = replay(&path)?;
        let file = open_append(&path)?;
        let inner = InMemoryCompletionStore::new();
        {
            let mut guard = inner.lock()?;
            for record in latest {
                guard.insert(record.task_id.clone(), record);
            }
            tracing::debug!(
                path = %path.display(),
                records = guard.len(),
                "{LOG_PREFIX} opened"
            );
        }
        Ok(Self {
            path,
            inner,
            file: Mutex::new(file),
            broken: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// The log's path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn open_append(path: &Path) -> Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| store_err("open for append", e))
}

/// Replays the log, repairing a torn tail in place.
fn replay(path: &Path) -> Result<Vec<CompletionRecord>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(store_err("read", e)),
    };
    let mut latest: HashMap<String, CompletionRecord> = HashMap::new();
    let mut good_len = 0usize;
    let mut needs_newline = false;
    let mut offset = 0usize;
    while offset < bytes.len() {
        let (line, next, terminated) = match bytes[offset..].iter().position(|b| *b == b'\n') {
            Some(i) => (&bytes[offset..offset + i], offset + i + 1, true),
            None => (&bytes[offset..], bytes.len(), false),
        };
        let parsed = serde_json::from_slice::<CompletionRecord>(line);
        match parsed {
            Ok(record) => {
                latest.insert(record.task_id.clone(), record);
                good_len = next;
                needs_newline = !terminated;
            }
            Err(_) if terminated && line.iter().all(u8::is_ascii_whitespace) => {
                good_len = next;
            }
            Err(err) if terminated => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "{LOG_PREFIX} skipping unreadable line"
                );
                good_len = next;
            }
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    dropped_bytes = line.len(),
                    "{LOG_PREFIX} dropping torn tail"
                );
            }
        }
        offset = next;
    }
    if good_len < bytes.len() || needs_newline {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|e| store_err("open for repair", e))?;
        file.set_len(good_len as u64)
            .map_err(|e| store_err("truncate torn tail", e))?;
        if needs_newline {
            let mut file = open_append(path)?;
            file.write_all(b"\n")
                .map_err(|e| store_err("terminate last line", e))?;
            file.sync_data().map_err(|e| store_err("sync repair", e))?;
        } else {
            file.sync_data().map_err(|e| store_err("sync repair", e))?;
        }
    }
    Ok(latest.into_values().collect())
}

impl CompletionStore for JsonlCompletionStore {
    fn get(&self, task_id: &str) -> Option<CompletionRecord> {
        self.inner.get(task_id)
    }

    fn put(&self, record: &CompletionRecord) -> Result<()> {
        let mut line =
            serde_json::to_string(record).map_err(|e| store_err("serialize record", e))?;
        line.push('\n');
        run_blocking(|| -> Result<()> {
            let mut file = self
                .file
                .lock()
                .map_err(|_| store_err("file lock", "poisoned"))?;
            // One write_all of the whole line, then fsync: a crash leaves either
            // the full line or a torn tail that `open` discards.
            if self.broken.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(store_err(
                    "append",
                    "log tail is unrepaired after an earlier failed write; reopen the store",
                ));
            }
            let start = file.metadata().map_err(|e| store_err("stat", e))?.len();
            let written = file
                .write_all(line.as_bytes())
                .and_then(|()| file.sync_data());
            if let Err(e) = written {
                // Drop a partial line so the next append cannot fuse with it.
                if let Err(rollback) = file.set_len(start) {
                    self.broken.store(true, std::sync::atomic::Ordering::SeqCst);
                    return Err(store_err(
                        "append (rollback also failed)",
                        format!("{e}; {rollback}"),
                    ));
                }
                return Err(store_err("append", e));
            }
            // Still under the file lock, so `compact` never sees the line
            // without the map entry (or the reverse).
            self.inner.put(record)
        })
    }

    fn list(&self, parent_key: Option<&str>) -> Vec<CompletionRecord> {
        self.inner.list(parent_key)
    }

    fn compact(&self, retain: Duration) -> Result<usize> {
        let now = SystemTime::now();
        run_blocking(|| -> Result<usize> {
            let mut file = self
                .file
                .lock()
                .map_err(|_| store_err("file lock", "poisoned"))?;
            let mut map = self.inner.lock()?;
            let before = map.len();
            let kept: Vec<&CompletionRecord> =
                map.values().filter(|r| !expired(r, now, retain)).collect();
            // A name no other compaction or store instance can be using.
            let tmp = self.path.with_extension(format!(
                "jsonl.{}.{}.tmp",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ));
            let out = {
                // Append mode, so the handle that becomes the live log keeps
                // O_APPEND semantics (a rolled-back write cannot leave a hole).
                let mut out = std::fs::OpenOptions::new()
                    .create_new(true)
                    .append(true)
                    .open(&tmp)
                    .map_err(|e| store_err("create compaction file", e))?;
                for record in &kept {
                    let mut line = serde_json::to_string(record)
                        .map_err(|e| store_err("serialize record", e))?;
                    line.push('\n');
                    out.write_all(line.as_bytes())
                        .map_err(|e| store_err("write compaction file", e))?;
                }
                out.sync_all()
                    .map_err(|e| store_err("sync compaction file", e))?;
                out
            };
            if let Err(e) = std::fs::rename(&tmp, &self.path) {
                let _ = std::fs::remove_file(&tmp);
                return Err(store_err("swap compacted log", e));
            }
            if let Some(dir) = self.path.parent().filter(|p| !p.as_os_str().is_empty())
                && let Ok(dir) = std::fs::File::open(dir)
            {
                let _ = dir.sync_all();
            }
            // The handle that wrote the compacted file follows its inode across
            // the rename, so the swap and the handle change are one step.
            *file = out;
            let keep: std::collections::HashSet<String> =
                kept.iter().map(|r| r.task_id.clone()).collect();
            map.retain(|id, _| keep.contains(id));
            let dropped = before - map.len();
            tracing::debug!(dropped, "{LOG_PREFIX} compacted");
            Ok(dropped)
        })
    }
}
