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

// ── AppendStore trait ─────────────────────────────────────────────────────────

/// Append-only stream storage for the harness.
///
/// Where [`Store`] answers "what is the current value of this key?", an
/// `AppendStore` answers "what happened, in order?". It is the durable backbone
/// for event journals: each `stream` is an independent, ordered, append-only
/// log of [`serde_json::Value`] entries addressed by a monotonically increasing
/// **offset**.
///
/// # Offsets
/// Offsets are zero-based positional indexes within a stream. The first entry
/// appended to a fresh stream is stored at offset `0`, the next at `1`, and so
/// on, so the offset returned by [`append`](Self::append) equals the number of
/// entries that preceded it. [`len`](Self::len) returns the count of entries
/// (equivalently, the offset the *next* append will receive). Consumers should
/// persist the last processed offset and resume with
/// [`read_from`](Self::read_from) rather than re-reading the whole stream.
///
/// Implementations must be `Send + Sync` so they can be shared across async
/// task boundaries.
#[async_trait]
pub trait AppendStore: Send + Sync {
    /// Appends `value` to the end of `stream` and returns the offset it was
    /// stored at.
    ///
    /// The returned offset is the zero-based position of the new entry, which
    /// is also the stream length immediately before the append.
    async fn append(&self, stream: &str, value: Value) -> Result<u64>;

    /// Returns every entry in `stream` whose offset is `>= offset`, in offset
    /// order, paired with its offset.
    ///
    /// Reading from `0` returns the whole stream; reading from
    /// [`len`](Self::len) (or any larger offset) returns an empty `Vec`.
    /// Reading from a stream that has never been written returns an empty
    /// `Vec` rather than an error.
    async fn read_from(&self, stream: &str, offset: u64) -> Result<Vec<(u64, Value)>>;

    /// Returns the number of entries currently stored in `stream`.
    ///
    /// This equals the offset the next [`append`](Self::append) will receive.
    /// Returns `0` for a stream that has never been written.
    async fn len(&self, stream: &str) -> Result<u64>;
}

// ── StoreRecord ───────────────────────────────────────────────────────────────

/// A single decoded entry read back from an [`AppendStore`].
///
/// The trait methods deliberately hand back the lighter `(offset, value)` tuple
/// shape, but `StoreRecord` is provided as a convenience for callers that want
/// to carry the append timestamp alongside the value (for example when
/// rendering an event journal). It is intentionally simple — richer
/// observability envelopes live in the events module.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StoreRecord {
    /// The zero-based offset of this entry within its stream.
    pub offset: u64,
    /// The stored JSON payload.
    pub value: Value,
    /// Unix-epoch milliseconds at which the entry was appended.
    pub created_at_ms: u64,
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

// ── InMemoryAppendStore ───────────────────────────────────────────────────────

/// Thread-safe, in-memory [`AppendStore`].
///
/// Each stream is a deque of `(created_at_ms, value)` entries held behind a
/// single [`Mutex`], plus the offset of its oldest retained entry so offsets
/// stay stable when old entries are evicted.
///
/// # Use cases
/// - Unit tests and deterministic replay of event journals.
/// - Examples and local prototyping.
///
/// # Retention
/// The store is **unbounded by default** — existing callers keep the full
/// journal semantics. Long-lived processes should opt into a per-stream cap
/// with [`InMemoryAppendStore::with_max_entries_per_stream`]: once a stream
/// exceeds the cap, its **oldest** entries are dropped. Offsets remain
/// monotonically increasing across eviction — an evicted offset is simply no
/// longer readable, and [`AppendStore::read_from`] returns only the retained
/// entries at or after the requested offset. [`AppendStore::len`] keeps
/// returning the *logical* stream length (the offset the next append gets),
/// which after eviction can exceed the number of retained entries.
///
/// # Caveats
/// There is **no durability**: entries are lost when the value is dropped. The
/// store is cheaply clonable through the inner [`Arc`]; clones share the same
/// underlying streams *and* the same retention cap.
#[derive(Clone, Debug, Default)]
pub struct InMemoryAppendStore {
    /// `stream → buffered entries + base offset`.
    pub(crate) streams: Arc<Mutex<HashMap<String, StreamBuffer>>>,
    /// Maximum retained entries per stream; `None` means unbounded (default).
    pub(crate) max_entries_per_stream: Option<usize>,
}

/// One in-memory stream: retained entries plus the offset of the oldest one.
#[derive(Debug, Default)]
pub(crate) struct StreamBuffer {
    /// Offset of the entry at the front of `entries` (the oldest retained
    /// entry). Starts at `0` and advances by one per evicted entry, keeping
    /// offsets monotonically increasing across eviction.
    pub(crate) base_offset: u64,
    /// Retained entries, oldest first, each entry being `(created_at_ms, value)`.
    pub(crate) entries: std::collections::VecDeque<AppendEntry>,
}

/// A single in-memory append entry: `(created_at_ms, value)`.
pub(crate) type AppendEntry = (u64, Value);

// ── JsonlAppendStore ──────────────────────────────────────────────────────────

/// JSONL-file-backed [`AppendStore`] for local development and durable journals.
///
/// Each **stream** maps to a `<stream>.jsonl` file inside `root_dir`. An append
/// writes exactly one JSON line — a [`StoreRecord`] object holding the offset,
/// the value, and the append timestamp — and `read_from` reads lines back from
/// the requested offset. The files are append-only and easy to tail or inspect
/// with ordinary shell tools.
///
/// # Stream-name sanitization
/// Stream names are validated the same way [`FileStore`] validates namespaces:
/// only ASCII alphanumerics, hyphens (`-`), underscores (`_`), and dots (`.`)
/// are allowed, and all-dot names are rejected. This blocks path traversal.
///
/// # Concurrency
/// Operations use blocking [`std::fs`] (no async-fs dependency is pulled in for
/// this local backend), but `append` runs that I/O on a blocking thread
/// (`spawn_blocking`) when a tokio runtime is present so it never stalls an
/// async worker. Appends use `OpenOptions::append`, which is atomic per line on
/// POSIX for small writes, but no advisory lock is held. To avoid re-parsing the
/// whole file on every append, each store instance caches the next offset per
/// stream (see [`Self::offsets`]); this assumes a single writing process per
/// directory. For multiple concurrent writers, funnel appends through one store
/// instance (its offset guard serialises them) or prefer a server backend.
#[derive(Clone, Debug, Default)]
pub struct JsonlAppendStore {
    /// The root directory under which `<stream>.jsonl` files live.
    pub(crate) root_dir: PathBuf,
    /// Per-stream cache of the *next* offset to write, so an append does not
    /// have to re-read and re-parse the whole file to learn its length (which
    /// made appends O(n²) per stream). Initialised lazily from disk on the
    /// first append for a stream and incremented in memory thereafter; the
    /// guard is held across the write so concurrent appends to the same store
    /// instance stay correctly ordered. Clones share the same cache.
    pub(crate) offsets: Arc<Mutex<HashMap<String, u64>>>,
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
