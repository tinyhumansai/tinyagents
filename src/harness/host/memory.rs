//! Retrieval-oriented long-term memory supplied by the embedder.
//!
//! See [`MemoryProvider`] for the trait contract and
//! [`InMemoryMemoryProvider`] for the inert default.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, TinyAgentsError};
use crate::harness::ids::ThreadId;

/// Number of records [`MemoryProvider::recall`] returns when
/// [`MemoryQuery::limit`] is `None`.
pub const DEFAULT_RECALL_LIMIT: usize = 10;

/// Retrieval-oriented long-term memory supplied by the embedder.
///
/// This is **not** conversation history: thread message logs are
/// [`ChatHistory`][crate::harness::memory::ChatHistory] and opaque key/value or
/// append-only streams are [`Store`][crate::harness::store::Store] /
/// [`AppendStore`][crate::harness::store::AppendStore]. A `MemoryProvider`
/// answers "what does the host know that is relevant to this text?" over a
/// namespaced corpus with scoring; it never replays, rewrites, or compacts a
/// thread's message list.
#[async_trait]
pub trait MemoryProvider<State: Send + Sync>: Send + Sync {
    /// Returns scored records relevant to `query`, most relevant first.
    ///
    /// An empty or non-matching query yields `Ok(vec![])`, never an error.
    async fn recall(&self, state: &State, query: &MemoryQuery<'_>) -> Result<Vec<MemoryRecord>>;

    /// Returns records matching an exact scope, without relevance scoring.
    async fn list(&self, state: &State, filter: &MemoryFilter<'_>) -> Result<Vec<MemoryRecord>>;

    /// Inserts or overwrites one record, keyed by `(namespace, key)`.
    async fn write(&self, state: &State, record: MemoryWrite) -> Result<()>;

    /// Returns a bounded per-namespace digest of the corpus.
    ///
    /// The default returns an empty `Vec`; backends without a rollup layer opt
    /// out rather than erroring.
    async fn namespace_digests(
        &self,
        state: &State,
        caps: DigestCaps,
    ) -> Result<Vec<NamespaceDigest>> {
        let _ = (state, caps);
        Ok(Vec::new())
    }
}

/// A relevance query against the host's corpus.
#[derive(Clone, Debug, Default)]
pub struct MemoryQuery<'a> {
    /// Free text the backend scores records against.
    pub text: &'a str,
    /// Maximum records to return. `None` means the backend's own default
    /// (`DEFAULT_RECALL_LIMIT` for the in-crate implementation).
    pub limit: Option<usize>,
    /// Restrict to one namespace. `None` searches every namespace.
    pub namespace: Option<&'a str>,
    /// Thread this query runs inside; combined with `cross_thread` it selects
    /// same-thread, other-thread, or unscoped recall.
    pub thread_id: Option<&'a ThreadId>,
    /// When `true`, records from other threads are eligible. When `false` and
    /// `thread_id` is set, only that thread's records are eligible.
    pub cross_thread: bool,
    /// Drop hits scoring below this value.
    pub min_score: Option<f64>,
}

impl<'a> MemoryQuery<'a> {
    /// Creates a query over `text` with every other field at its default.
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            ..Default::default()
        }
    }

    /// Sets the maximum number of records to return.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Restricts the query to one namespace.
    pub fn with_namespace(mut self, namespace: &'a str) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Scopes the query to `thread_id`, optionally allowing other threads in.
    pub fn with_thread(mut self, thread_id: &'a ThreadId, cross_thread: bool) -> Self {
        self.thread_id = Some(thread_id);
        self.cross_thread = cross_thread;
        self
    }

    /// Drops hits scoring below `min_score`.
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

/// An exact-match scope for [`MemoryProvider::list`].
///
/// Distinct from [`crate::harness::memory::MemoryScope`], which labels the
/// short-term/long-term conversation layers rather than selecting records.
#[derive(Clone, Debug, Default)]
pub struct MemoryFilter<'a> {
    /// Restrict to one namespace. `None` matches every namespace.
    pub namespace: Option<&'a str>,
    /// Free-form host category label; the crate assigns it no meaning.
    pub category: Option<&'a str>,
    /// Restrict to records recorded against one thread.
    pub thread_id: Option<&'a ThreadId>,
    /// Maximum records to return. `None` returns every match.
    pub limit: Option<usize>,
}

impl<'a> MemoryFilter<'a> {
    /// Creates a filter matching every record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the filter to one namespace.
    pub fn with_namespace(mut self, namespace: &'a str) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Restricts the filter to one host category label.
    pub fn with_category(mut self, category: &'a str) -> Self {
        self.category = Some(category);
        self
    }
}

/// One record returned by [`MemoryProvider::recall`] or
/// [`MemoryProvider::list`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Backend-assigned identity, stable for the life of the record.
    pub id: String,
    /// Host-assigned key, unique within `namespace`.
    pub key: String,
    /// The stored text.
    pub content: String,
    /// Namespace the record belongs to, when the backend tracks one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Free-form host category label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Thread the record was recorded against, when thread-scoped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Relevance score for this hit. Absent for unscored `list` results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Backend-formatted timestamp, when known. The crate never parses it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_at: Option<String>,
    /// Backend-defined extras carried opaquely so hosts can round-trip
    /// provenance without the crate modelling it.
    #[serde(default)]
    pub attributes: Value,
}

/// An insert-or-overwrite request, keyed by `(namespace, key)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryWrite {
    /// Namespace the record belongs to.
    pub namespace: String,
    /// Key, unique within `namespace`. Writing an existing key overwrites it.
    pub key: String,
    /// The text to store.
    pub content: String,
    /// Free-form host category label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Thread to record this against, when thread-scoped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
}

impl MemoryWrite {
    /// Creates a write with no category and no thread scope.
    pub fn new(
        namespace: impl Into<String>,
        key: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
            content: content.into(),
            category: None,
            thread_id: None,
        }
    }

    /// Tags the write with a host category label.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Scopes the write to one thread.
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }
}

/// Size ceilings for [`MemoryProvider::namespace_digests`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DigestCaps {
    /// Maximum characters any single namespace digest may contribute.
    pub per_namespace_max_chars: usize,
    /// Maximum characters the whole digest set may contribute.
    pub total_max_chars: usize,
}

/// A bounded rollup of one namespace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamespaceDigest {
    /// Namespace this digest summarizes.
    pub namespace: String,
    /// Host-rendered digest text, used verbatim.
    pub body: String,
    /// Backend-formatted timestamp of the last update, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

/// Ephemeral, in-process [`MemoryProvider`] backed by a shared map.
///
/// Mirrors [`InMemoryChatHistory`][crate::harness::memory::InMemoryChatHistory]
/// in shape and durability guarantees (there are none). Clones share the same
/// underlying data through the inner [`Arc`].
///
/// Scoring is deliberately trivial and dependency-free: the query is split on
/// whitespace and each record scores `matched_tokens / total_tokens` by
/// case-insensitive substring containment. Ties break on `key` ascending so
/// results are deterministic despite the backing [`HashMap`].
/// [`MemoryProvider::namespace_digests`] takes the trait default (empty) — this
/// backend has no rollup layer.
#[derive(Clone, Default)]
pub struct InMemoryMemoryProvider {
    /// `(namespace, key) → record` map protected by a standard mutex.
    records: Arc<Mutex<HashMap<(String, String), MemoryRecord>>>,
}

impl InMemoryMemoryProvider {
    /// Creates a new, empty provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored records.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.len())
    }

    /// Returns `true` when no records are stored.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.lock()?.is_empty())
    }

    /// Removes every stored record.
    pub fn clear(&self) -> Result<()> {
        self.lock()?.clear();
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<(String, String), MemoryRecord>>> {
        self.records
            .lock()
            .map_err(|e| TinyAgentsError::Memory(format!("memory provider lock poisoned: {e}")))
    }

    /// Returns `true` when `record` is eligible for a query scoped to
    /// `thread_id` with the given `cross_thread` setting.
    fn thread_matches(record: &MemoryRecord, thread_id: Option<&ThreadId>, cross: bool) -> bool {
        match thread_id {
            // Unscoped, or explicitly allowed to reach other threads.
            None => true,
            Some(_) if cross => true,
            Some(wanted) => record.thread_id.as_deref() == Some(wanted.as_str()),
        }
    }
}

#[async_trait]
impl<State: Send + Sync> MemoryProvider<State> for InMemoryMemoryProvider {
    async fn recall(&self, _state: &State, query: &MemoryQuery<'_>) -> Result<Vec<MemoryRecord>> {
        let tokens: Vec<String> = query
            .text
            .split_whitespace()
            .map(|token| token.to_lowercase())
            .collect();
        // Short-circuit before any division: an empty query has no signal, and
        // the contract says that is an empty result rather than an error.
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let total = tokens.len() as f64;

        let mut hits: Vec<MemoryRecord> = self
            .lock()?
            .values()
            .filter(|record| match query.namespace {
                Some(namespace) => record.namespace.as_deref() == Some(namespace),
                None => true,
            })
            .filter(|record| Self::thread_matches(record, query.thread_id, query.cross_thread))
            .filter_map(|record| {
                let haystack = record.content.to_lowercase();
                let matched = tokens
                    .iter()
                    .filter(|token| haystack.contains(token.as_str()))
                    .count();
                if matched == 0 {
                    return None;
                }
                let score = matched as f64 / total;
                if query.min_score.is_some_and(|floor| score < floor) {
                    return None;
                }
                let mut hit = record.clone();
                hit.score = Some(score);
                Some(hit)
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.key.cmp(&b.key))
        });
        hits.truncate(query.limit.unwrap_or(DEFAULT_RECALL_LIMIT));
        Ok(hits)
    }

    async fn list(&self, _state: &State, filter: &MemoryFilter<'_>) -> Result<Vec<MemoryRecord>> {
        let mut matches: Vec<MemoryRecord> = self
            .lock()?
            .values()
            .filter(|record| match filter.namespace {
                Some(namespace) => record.namespace.as_deref() == Some(namespace),
                None => true,
            })
            .filter(|record| match filter.category {
                Some(category) => record.category.as_deref() == Some(category),
                None => true,
            })
            .filter(|record| match filter.thread_id {
                Some(thread_id) => record.thread_id.as_deref() == Some(thread_id.as_str()),
                None => true,
            })
            .cloned()
            .collect();

        matches.sort_by(|a, b| a.key.cmp(&b.key));
        if let Some(limit) = filter.limit {
            matches.truncate(limit);
        }
        Ok(matches)
    }

    async fn write(&self, _state: &State, record: MemoryWrite) -> Result<()> {
        let stored = MemoryRecord {
            id: format!("{}/{}", record.namespace, record.key),
            key: record.key.clone(),
            content: record.content,
            namespace: Some(record.namespace.clone()),
            category: record.category,
            thread_id: record.thread_id,
            score: None,
            recorded_at: None,
            attributes: Value::Null,
        };
        self.lock()?.insert((record.namespace, record.key), stored);
        Ok(())
    }
}
