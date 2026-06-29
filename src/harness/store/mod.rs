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
//! - [`InMemoryStore`] — ephemeral in-process store for tests and examples.
//! - [`FileStore`] — file-system-backed store for local development.
//! - [`StoreRegistry`] — named bag of stores injected into `RunContext`.
//!
//! # Namespace convention
//! Use slash-free, lowercase names like `"threads"`, `"events"`, `"cache"`,
//! `"artifacts"`. The registry does not enforce a naming scheme, but
//! consistent names make multi-store applications easier to audit.

mod types;

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

pub use types::*;

use crate::error::{Result, TinyAgentsError};

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
        fs::write(&path, &bytes)
            .map_err(|e| TinyAgentsError::Validation(format!("store write error: {e}")))?;
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
    /// Creates a new, empty in-memory append store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl AppendStore for InMemoryAppendStore {
    async fn append(&self, stream: &str, value: Value) -> Result<u64> {
        let mut streams = self
            .streams
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("append store lock poisoned: {e}")))?;
        let entries = streams.entry(stream.to_string()).or_default();
        let offset = entries.len() as u64;
        entries.push((now_ms(), value));
        Ok(offset)
    }

    async fn read_from(&self, stream: &str, offset: u64) -> Result<Vec<(u64, Value)>> {
        let streams = self
            .streams
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("append store lock poisoned: {e}")))?;
        let Some(entries) = streams.get(stream) else {
            return Ok(Vec::new());
        };
        Ok(entries
            .iter()
            .enumerate()
            .skip(offset as usize)
            .map(|(i, (_ts, value))| (i as u64, value.clone()))
            .collect())
    }

    async fn len(&self, stream: &str) -> Result<u64> {
        let streams = self
            .streams
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("append store lock poisoned: {e}")))?;
        Ok(streams.get(stream).map(|e| e.len() as u64).unwrap_or(0))
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
        fs::create_dir_all(&self.root_dir)
            .map_err(|e| TinyAgentsError::Validation(format!("append store mkdir error: {e}")))?;
        // The offset is the count of existing lines; re-read to stay correct
        // across separate store instances on the same directory.
        let offset = Self::read_records(&path)?.len() as u64;
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
            .map_err(|e| TinyAgentsError::Validation(format!("append store open error: {e}")))?;
        std::io::Write::write_all(&mut file, line.as_bytes())
            .map_err(|e| TinyAgentsError::Validation(format!("append store write error: {e}")))?;
        Ok(offset)
    }

    async fn read_from(&self, stream: &str, offset: u64) -> Result<Vec<(u64, Value)>> {
        let path = self.stream_path(stream)?;
        Ok(Self::read_records(&path)?
            .into_iter()
            .skip(offset as usize)
            .map(|r| (r.offset, r.value))
            .collect())
    }

    async fn len(&self, stream: &str) -> Result<u64> {
        let path = self.stream_path(stream)?;
        Ok(Self::read_records(&path)?.len() as u64)
    }
}

/// Returns the current time in Unix-epoch milliseconds, saturating at `0` for
/// clocks set before the epoch.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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
