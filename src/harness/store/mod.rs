//! Harness store module — long-term key-value storage backends.
//!
//! In the recursive architecture the store is the durable, shared substrate
//! that outlives any single run: parent and child runs, sub-agents, and
//! REPL/blueprint executions read and write the same namespaced values, so a
//! deeply nested call can persist a result that a sibling or a later turn picks
//! up. It is the harness-side persistence layer, distinct from graph
//! checkpointing.
//!
//! The store is the persistence layer for harness runtime data: events, model
//! call records, tool call records, message history, artifacts, and memory. It
//! is intentionally separate from graph checkpointing (which belongs to the
//! graph module) and from prompt/model context assembly (which belongs to the
//! model and prompt modules).
//!
//! # Primary types
//! - [`Store`] — the core async trait every backend implements.
//! - [`namespaced::NamespacedStore`] — the hierarchical, TTL-aware,
//!   batch-oriented long-term store. Additive: `Store` is unchanged, and
//!   [`namespaced::FlatNamespacedStore`] adapts the new trait back to it.
//! - [`InMemoryStore`] — ephemeral in-process store for tests and examples.
//! - [`FileStore`] — file-system-backed store for local development.
//! - [`StoreRegistry`] — named bag of stores injected into `RunContext`.
//!
//! # Namespace convention
//! Use slash-free, lowercase names like `"threads"`, `"events"`, `"cache"`,
//! `"artifacts"`. The registry does not enforce a naming scheme, but
//! consistent names make multi-store applications easier to audit.

pub mod namespaced;
mod types;

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

pub use types::*;

use crate::error::{Result, TinyAgentsError};
use crate::harness::ids::now_ms;

/// Process-wide counter making [`FileStore`] temp-file names unique so
/// concurrent atomic writes to the same key never collide on their scratch file.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ── InMemoryStore ─────────────────────────────────────────────────────────────

impl InMemoryStore {
    /// Creates a new, empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Store for InMemoryStore {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<Value>> {
        let data = self
            .data
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("store lock poisoned: {e}")))?;
        Ok(data.get(namespace).and_then(|ns| ns.get(key)).cloned())
    }

    async fn put(&self, namespace: &str, key: &str, value: Value) -> Result<()> {
        let mut data = self
            .data
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("store lock poisoned: {e}")))?;
        data.entry(namespace.to_string())
            .or_default()
            .insert(key.to_string(), value);
        Ok(())
    }

    async fn delete(&self, namespace: &str, key: &str) -> Result<()> {
        let mut data = self
            .data
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("store lock poisoned: {e}")))?;
        if let Some(ns) = data.get_mut(namespace) {
            ns.remove(key);
        }
        Ok(())
    }

    async fn list(&self, namespace: &str) -> Result<Vec<String>> {
        let data = self
            .data
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("store lock poisoned: {e}")))?;
        Ok(data
            .get(namespace)
            .map(|ns| ns.keys().cloned().collect())
            .unwrap_or_default())
    }
}

// ── FileStore ─────────────────────────────────────────────────────────────────

impl FileStore {
    /// Creates a file store rooted at `root_dir`.
    ///
    /// The directory is created lazily on the first write, so constructing a
    /// `FileStore` for a path that does not yet exist is not an error.
    pub fn new(root_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
        }
    }

    /// Validates that `name` (a namespace or key) contains only safe
    /// characters: ASCII alphanumerics, hyphens, underscores, and dots.
    ///
    /// Returns a [`TinyAgentsError::Validation`] if the name is empty or
    /// contains any other byte, preventing path-traversal attacks.
    fn sanitize(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(TinyAgentsError::Validation(
                "store namespace and key must not be empty".into(),
            ));
        }
        // Reject names composed solely of dots (`.`, `..`, `...`). A namespace
        // is joined onto `root_dir` without a suffix, so `".."` would resolve to
        // the parent directory and escape the store root (path traversal).
        if name.bytes().all(|b| b == b'.') {
            return Err(TinyAgentsError::Validation(format!(
                "store name must not be all dots: {name:?} (path-traversal guard)"
            )));
        }
        if name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
        {
            Ok(())
        } else {
            Err(TinyAgentsError::Validation(format!(
                "store name contains invalid characters: {name:?} \
                 (only ASCII alphanumerics, hyphens, underscores, dots allowed)"
            )))
        }
    }

    /// Returns the canonical path for `key` within `namespace`.
    fn key_path(&self, namespace: &str, key: &str) -> std::path::PathBuf {
        self.root_dir.join(namespace).join(format!("{key}.json"))
    }
}

#[async_trait]
impl Store for FileStore {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<Value>> {
        Self::sanitize(namespace)?;
        Self::sanitize(key)?;
        let path = self.key_path(namespace, key);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)
            .map_err(|e| TinyAgentsError::Validation(format!("store read error: {e}")))?;
        let value: Value = serde_json::from_slice(&bytes)?;
        Ok(Some(value))
    }

    async fn put(&self, namespace: &str, key: &str, value: Value) -> Result<()> {
        Self::sanitize(namespace)?;
        Self::sanitize(key)?;
        let dir = self.root_dir.join(namespace);
        fs::create_dir_all(&dir)
            .map_err(|e| TinyAgentsError::Validation(format!("store mkdir error: {e}")))?;
        let path = dir.join(format!("{key}.json"));
        let bytes = serde_json::to_vec_pretty(&value)?;
        // Write to a uniquely named temp file in the same directory, then rename
        // over the destination. Rename is atomic on POSIX/Windows for same-dir
        // paths, so a reader never observes a partially written file and a crash
        // mid-write leaves the previous value intact (as the type docs promise).
        let tmp = dir.join(format!(
            "{key}.json.tmp.{}.{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::write(&tmp, &bytes)
            .map_err(|e| TinyAgentsError::Validation(format!("store write error: {e}")))?;
        if let Err(e) = fs::rename(&tmp, &path) {
            // Best-effort cleanup of the temp file so a failed rename does not
            // leak partial files into the namespace directory.
            let _ = fs::remove_file(&tmp);
            return Err(TinyAgentsError::Validation(format!(
                "store rename error: {e}"
            )));
        }
        Ok(())
    }

    async fn delete(&self, namespace: &str, key: &str) -> Result<()> {
        Self::sanitize(namespace)?;
        Self::sanitize(key)?;
        let path = self.key_path(namespace, key);
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| TinyAgentsError::Validation(format!("store delete error: {e}")))?;
        }
        Ok(())
    }

    async fn list(&self, namespace: &str) -> Result<Vec<String>> {
        Self::sanitize(namespace)?;
        let dir = self.root_dir.join(namespace);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&dir)
            .map_err(|e| TinyAgentsError::Validation(format!("store readdir error: {e}")))?;
        let mut keys = Vec::new();
        for entry in entries {
            let entry = entry
                .map_err(|e| TinyAgentsError::Validation(format!("store entry error: {e}")))?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if let Some(stem) = name.strip_suffix(".json") {
                keys.push(stem.to_string());
            }
        }
        Ok(keys)
    }
}

// ── InMemoryAppendStore ─────────────────────────────────────────────────────────

impl InMemoryAppendStore {
    /// Creates a new, empty, **unbounded** in-memory append store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Caps every stream at `max` retained entries, evicting the **oldest**
    /// entries once a stream grows past the cap.
    ///
    /// Offsets stay monotonically increasing across eviction: an evicted
    /// offset is no longer readable (`read_from` silently starts at the oldest
    /// retained entry when asked for an earlier offset), and `len` keeps
    /// returning the logical stream length. A `max` of `0` retains nothing —
    /// every append is immediately evicted (offsets still advance).
    ///
    /// The default (via [`Self::new`] / [`Default`]) is unbounded, preserving
    /// full journal semantics for tests and replay.
    pub fn with_max_entries_per_stream(mut self, max: usize) -> Self {
        self.max_entries_per_stream = Some(max);
        self
    }
}

#[async_trait]
impl AppendStore for InMemoryAppendStore {
    async fn append(&self, stream: &str, value: Value) -> Result<u64> {
        let mut streams = self
            .streams
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("append store lock poisoned: {e}")))?;
        let buf = streams.entry(stream.to_string()).or_default();
        let offset = buf.base_offset + buf.entries.len() as u64;
        buf.entries.push_back((now_ms(), value));
        if let Some(max) = self.max_entries_per_stream {
            while buf.entries.len() > max {
                buf.entries.pop_front();
                buf.base_offset += 1;
            }
        }
        Ok(offset)
    }

    async fn read_from(&self, stream: &str, offset: u64) -> Result<Vec<(u64, Value)>> {
        let streams = self
            .streams
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("append store lock poisoned: {e}")))?;
        let Some(buf) = streams.get(stream) else {
            return Ok(Vec::new());
        };
        // Entries before `base_offset` have been evicted; reading from an
        // evicted offset resumes at the oldest retained entry.
        let skip = offset.saturating_sub(buf.base_offset) as usize;
        Ok(buf
            .entries
            .iter()
            .enumerate()
            .skip(skip)
            .map(|(i, (_ts, value))| (buf.base_offset + i as u64, value.clone()))
            .collect())
    }

    async fn len(&self, stream: &str) -> Result<u64> {
        let streams = self
            .streams
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("append store lock poisoned: {e}")))?;
        Ok(streams
            .get(stream)
            .map(|buf| buf.base_offset + buf.entries.len() as u64)
            .unwrap_or(0))
    }
}

// ── JsonlAppendStore ──────────────────────────────────────────────────────────

impl JsonlAppendStore {
    /// Creates a JSONL append store rooted at `root_dir`.
    ///
    /// The directory is created lazily on the first append, so constructing a
    /// store for a path that does not yet exist is not an error.
    pub fn new(root_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
            append_guard: Default::default(),
        }
    }

    /// Returns the offset the next append to `path` should carry.
    ///
    /// Reads the **last complete line** of the file rather than parsing all of
    /// it: an exponentially growing tail window is pulled from the end until a
    /// line boundary is found. A trailing partial line (a torn write) is
    /// skipped, so a crash mid-append cannot make the stream restart its
    /// numbering. An absent or empty file starts at `0`.
    fn next_offset(path: &std::path::Path) -> Result<u64> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => {
                return Err(TinyAgentsError::Validation(format!(
                    "append store open error: {e}"
                )));
            }
        };
        let len = file
            .metadata()
            .map_err(|e| TinyAgentsError::Validation(format!("append store stat error: {e}")))?
            .len();
        if len == 0 {
            return Ok(0);
        }
        let mut window = 4096u64;
        loop {
            let start = len.saturating_sub(window);
            file.seek(SeekFrom::Start(start)).map_err(|e| {
                TinyAgentsError::Validation(format!("append store seek error: {e}"))
            })?;
            // Read BYTES, not a String. `read_to_string` requires everything
            // from `start` onward to be valid UTF-8, and `start` is an arbitrary
            // byte offset — so whenever it lands inside a multi-byte character
            // the read fails with "stream did not contain valid UTF-8" even
            // though the file is perfectly well formed.
            //
            // This code already anticipated a torn LINE, one byte further on.
            // A torn CHARACTER is the same hazard at finer grain, and it was
            // fatal rather than skipped: the `?` returns before the window can
            // grow, so the identical offset fails identically forever. Observed
            // on a hosted tenant as 584 consecutive failures and counting, with
            // observations dropped for as long as it ran; 104 of its 557 journal
            // files had a last-4096-byte window that could not decode.
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).map_err(|e| {
                TinyAgentsError::Validation(format!("append store read error: {e}"))
            })?;
            // Decode from the first newline when the window did not reach the
            // start of the file. That discards the possibly-half first line,
            // which this function wanted anyway — and because a newline is
            // always a character boundary in UTF-8, whatever follows one is
            // guaranteed to decode. The two problems have one answer.
            let decodable = if start > 0 {
                let Some(newline) = bytes.iter().position(|b| *b == b'\n') else {
                    // No newline in the whole window: every byte belongs to one
                    // unterminated line, so there is nothing complete to read
                    // and no guaranteed boundary to decode from. Widen rather
                    // than guess. Load-bearing precisely because the decode
                    // below is strict.
                    window = window.saturating_mul(4);
                    continue;
                };
                &bytes[newline + 1..]
            } else {
                &bytes[..]
            };
            // STRICT, not lossy. The boundary is guaranteed above, so a failure
            // here means a genuinely corrupt byte rather than a torn character,
            // and that must stay a hard error exactly as it was before.
            //
            // Tolerating it would be worse than the bug being fixed: a corrupt
            // byte inside a record's structural region makes that line
            // unparseable, `next_offset` then returns the PREVIOUS record's
            // offset, and the next append silently reuses an offset already on
            // disk. Measured on a 30-record stream with one byte set to 0xFF —
            // lossy decoding returned `Ok(29)` with a record at offset 29
            // already written, while `read_from` still refused the stream
            // outright, since `read_records` decodes strictly. A stream that
            // keeps accepting appends while being permanently unreadable is not
            // a tolerance anyone asked for.
            let buf = std::str::from_utf8(decodable).map_err(|e| {
                TinyAgentsError::Validation(format!("append store read error: {e}"))
            })?;
            let lines: Vec<&str> = buf.lines().collect();
            for line in lines.iter().rev() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(record) = serde_json::from_str::<StoreRecord>(line) {
                    return Ok(record.offset + 1);
                }
                // A torn trailing line: keep walking backwards.
            }
            if start == 0 {
                // Whole file scanned and nothing decoded: treat it as empty.
                return Ok(0);
            }
            window = window.saturating_mul(4);
        }
    }

    /// Returns the `<stream>.jsonl` path for `stream`, validating the name.
    fn stream_path(&self, stream: &str) -> Result<std::path::PathBuf> {
        FileStore::sanitize(stream)?;
        Ok(self.root_dir.join(format!("{stream}.jsonl")))
    }

    /// Reads and decodes every [`StoreRecord`] line in `path`, in file order.
    fn read_records(path: &std::path::Path) -> Result<Vec<StoreRecord>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(path)
            .map_err(|e| TinyAgentsError::Validation(format!("append store read error: {e}")))?;
        let mut records = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let record: StoreRecord = serde_json::from_str(line)?;
            records.push(record);
        }
        Ok(records)
    }
}

#[async_trait]
impl AppendStore for JsonlAppendStore {
    async fn append(&self, stream: &str, value: Value) -> Result<u64> {
        let path = self.stream_path(stream)?;
        let root_dir = self.root_dir.clone();
        let guard = Arc::clone(&self.append_guard);

        // The append is pure blocking file I/O. Run it off the async runtime so
        // it never stalls a tokio worker (`spawn_blocking` when a runtime is
        // present, inline otherwise — e.g. a synchronous sink draining outside a
        // runtime).
        let work = move || -> Result<u64> {
            fs::create_dir_all(&root_dir).map_err(|e| {
                TinyAgentsError::Validation(format!("append store mkdir error: {e}"))
            })?;
            // Hold the guard across read-tail + write so concurrent appends
            // through this instance get distinct, ordered offsets.
            let _guard = guard.lock().map_err(|e| {
                TinyAgentsError::Validation(format!("append store lock poisoned: {e}"))
            })?;
            let offset = Self::next_offset(&path)?;
            let record = StoreRecord {
                offset,
                value,
                created_at_ms: now_ms(),
            };
            let mut line = serde_json::to_string(&record)?;
            line.push('\n');
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| {
                    TinyAgentsError::Validation(format!("append store open error: {e}"))
                })?;
            std::io::Write::write_all(&mut file, line.as_bytes()).map_err(|e| {
                TinyAgentsError::Validation(format!("append store write error: {e}"))
            })?;
            Ok(offset)
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.spawn_blocking(work).await.map_err(|e| {
                TinyAgentsError::Validation(format!("append store task error: {e}"))
            })?,
            Err(_) => work(),
        }
    }

    async fn read_from(&self, stream: &str, offset: u64) -> Result<Vec<(u64, Value)>> {
        let path = self.stream_path(stream)?;
        // Resolve BY OFFSET, not by position. `.skip(offset)` is positional, so
        // it disagreed with the offsets it then handed back the moment a stream
        // contained anything other than a dense 0..n sequence — which is exactly
        // what the documented contract (`entries whose offset is >= offset`)
        // promises to handle, and what `InMemoryAppendStore` already did.
        Ok(Self::read_records(&path)?
            .into_iter()
            .filter(|r| r.offset >= offset)
            .map(|r| (r.offset, r.value))
            .collect())
    }

    async fn len(&self, stream: &str) -> Result<u64> {
        // The offset the next append will receive, per the trait contract.
        let path = self.stream_path(stream)?;
        Self::next_offset(&path)
    }
}

// ── StoreRegistry ─────────────────────────────────────────────────────────────

impl StoreRegistry {
    /// Creates a registry with a built-in default in-memory store.
    ///
    /// Named stores can be added with [`Self::register`].
    pub fn new() -> Self {
        Self {
            stores: HashMap::new(),
            default_store: Arc::new(InMemoryStore::new()),
        }
    }

    /// Registers `store` under `name`.
    ///
    /// Replaces any previously registered store with the same name. Returns
    /// `&mut self` for convenient builder-style chaining.
    pub fn register(&mut self, name: impl Into<String>, store: Arc<dyn Store>) -> &mut Self {
        self.stores.insert(name.into(), store);
        self
    }

    /// Looks up a named store by `name`, returning `None` if no store with
    /// that name has been registered.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Store>> {
        self.stores.get(name).cloned()
    }

    /// Returns the built-in default in-memory store.
    ///
    /// This store is always available regardless of registered backends.
    pub fn default_store(&self) -> Arc<dyn Store> {
        Arc::clone(&self.default_store)
    }
}

impl Default for StoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
