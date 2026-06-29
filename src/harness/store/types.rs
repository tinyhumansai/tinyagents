//! Storage backend traits and types for the harness store module.
//!
//! All public types in this module are re-exported through [`super`] so
//! callers import from `crate::harness::store` directly.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

// ── Core trait ──────────────────────────────────────────────────────────────

/// Long-term key-value store for the harness.
///
/// Stores are **namespaced**: each namespace is a logically independent bucket.
/// Keys within a namespace map to [`serde_json::Value`]s. Implementations must
/// be `Send + Sync` so they can be shared freely across async task boundaries.
///
/// This is NOT graph checkpointing. Graph checkpoints belong to the graph
/// module. Harness stores record application / runtime data around LLM
/// orchestration (events, tool call records, artifacts, memory, etc.).
#[async_trait]
pub trait Store: Send + Sync {
    /// Returns the value stored at `key` within `namespace`, or `None` if the
    /// key has never been written or was deleted.
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<Value>>;

    /// Inserts or overwrites `key` in `namespace` with `value`.
    async fn put(&self, namespace: &str, key: &str, value: Value) -> Result<()>;

    /// Removes `key` from `namespace`. This is a no-op if the key does not
    /// exist; it does not return an error in that case.
    async fn delete(&self, namespace: &str, key: &str) -> Result<()>;

    /// Returns all keys present in `namespace` in unspecified order.
    ///
    /// Returns an empty `Vec` if the namespace has never received a write.
    async fn list(&self, namespace: &str) -> Result<Vec<String>>;
}

// ── InMemoryStore ────────────────────────────────────────────────────────────

/// Thread-safe in-memory store backed by a nested [`HashMap`].
///
/// # Use cases
/// - Unit tests
/// - Examples and local prototyping
/// - Deterministic replay scenarios
///
/// # Caveats
/// There is **no durability**: data is lost when the value is dropped. The
/// store is cheaply clonable through the inner [`Arc`]; clones share the same
/// underlying data.
#[derive(Clone, Debug, Default)]
pub struct InMemoryStore {
    /// `namespace → (key → value)` map protected by a standard mutex.
    pub(crate) data: Arc<Mutex<HashMap<String, HashMap<String, Value>>>>,
}

// ── FileStore ────────────────────────────────────────────────────────────────

/// File-system-backed key-value store.
///
/// Each **namespace** maps to a subdirectory of `root_dir`. Each **key** maps
/// to a file `<key>.json` inside that subdirectory.
///
/// # Key and namespace sanitization
/// Both namespace names and key names are validated: only ASCII alphanumerics,
/// hyphens (`-`), underscores (`_`), and dots (`.`) are allowed. This blocks
/// path traversal attacks (e.g., `..`, `/`, `\`).
///
/// # Concurrency
/// Operations use [`std::fs`], which is blocking. Concurrent writes to the
/// same key from separate tasks are serialised at the OS level through
/// atomic-write semantics, but no advisory lock is held. For high-concurrency
/// workloads prefer `InMemoryStore` or a future `MongoStore`.
#[derive(Clone, Debug)]
pub struct FileStore {
    /// The root directory under which namespace subdirectories live.
    pub(crate) root_dir: PathBuf,
}

// ── StoreRegistry ────────────────────────────────────────────────────────────

/// Registry of named [`Store`] backends.
///
/// `RunContext` holds a `StoreRegistry` and exposes it to agent loops and
/// tools so they can read/write without knowing which backend is in use.
///
/// A built-in default in-memory store is always available even when no named
/// store has been registered, so code that targets the default store always
/// compiles and runs.
///
/// # Example
/// ```rust,ignore
/// let mut reg = StoreRegistry::new();
/// reg.register("events", Arc::new(FileStore::new("./data/events")));
/// reg.register("cache", Arc::new(InMemoryStore::new()));
/// ```
pub struct StoreRegistry {
    /// Named stores keyed by their registration name.
    pub(crate) stores: HashMap<String, Arc<dyn Store>>,
    /// The built-in default in-memory store, always present.
    pub(crate) default_store: Arc<dyn Store>,
}
