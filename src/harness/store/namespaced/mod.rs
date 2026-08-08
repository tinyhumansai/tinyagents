//! Hierarchical, batch-oriented long-term store.
//!
//! # Why this exists next to [`Store`]
//!
//! The original [`Store`](crate::harness::store::Store) trait is get/put/
//! delete/list over a **flat** `&str` namespace. That is enough to key values
//! by bucket, and not enough for anything a long-term memory layer actually
//! needs: there is no way to ask for everything under `users/alice`, no way to
//! filter by a field, no way to enumerate what namespaces exist, no expiry
//! (so a store only ever grows), and no way to issue several reads as one
//! round trip.
//!
//! [`NamespacedStore`] adds those. It does **not** replace `Store`: the flat
//! trait keeps working unchanged, and [`FlatNamespacedStore`] adapts any
//! `NamespacedStore` back to it, so existing callers are untouched.
//!
//! # `batch` is the one method that matters
//!
//! [`NamespacedStore::batch`] is the single required method; `get`, `put`,
//! `delete`, `search` and `list_namespaces` are convenience wrappers that
//! submit a one-operation batch. This is deliberate, and it is the shape
//! LangGraph converged on: a remote or pooled backend wants to coalesce
//! concurrent operations into one round trip, and it can only do that if every
//! path through the API funnels into one place. Implement `batch` well and
//! every other method is correct by construction; implement six methods
//! separately and the batched path drifts from the single path.
//!
//! Results are returned **positionally aligned** with the request: result `i`
//! answers operation `i`.
//!
//! # TTL
//!
//! [`TtlConfig`] gives items an expiry. An expired item is invisible to reads
//! and searches immediately, and is reclaimed by
//! [`NamespacedStore::sweep_expired`]. This is the bounded-growth answer the
//! flat store never had.
//!
//! # Semantic search
//!
//! [`SearchQuery::query`] is the seam for vector/semantic search. This crate
//! has no embedding dependency, so the bundled backend does a plain substring
//! scan; a backend with an index is expected to interpret it properly.

mod types;

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

pub use types::*;

use crate::error::{Result, TinyAgentsError};
use crate::harness::ids::now_ms;
use crate::harness::store::Store;

/// A hierarchical, TTL-aware, batch-oriented long-term store.
///
/// Implement [`NamespacedStore::batch`]; everything else has a default body
/// built on it.
#[async_trait]
pub trait NamespacedStore: Send + Sync {
    /// Executes `ops` and returns one result per operation, in order.
    ///
    /// The **only** required method. See the module docs for why.
    async fn batch(&self, ops: &[StoreOp]) -> Result<Vec<StoreResult>>;

    /// The TTL policy in force. Defaults to "nothing expires".
    fn ttl_config(&self) -> TtlConfig {
        TtlConfig::default()
    }

    /// Removes expired items, returning how many were reclaimed.
    ///
    /// Expiry is enforced on read regardless, so sweeping is about reclaiming
    /// space rather than correctness. The default is a no-op for backends that
    /// expire lazily.
    async fn sweep_expired(&self) -> Result<usize> {
        Ok(0)
    }

    /// Reads one item, or `None` when it is absent or expired.
    async fn get(&self, namespace: &Namespace, key: &str) -> Result<Option<Item>> {
        one(
            self,
            StoreOp::Get {
                namespace: namespace.clone(),
                key: key.to_string(),
                refresh_ttl: None,
            },
        )
        .await?
        .into_item()
    }

    /// Writes one item, applying the store's default TTL.
    async fn put(&self, namespace: &Namespace, key: &str, value: Value) -> Result<()> {
        self.put_with_ttl(namespace, key, value, None).await
    }

    /// Writes one item with an explicit lifetime in minutes.
    async fn put_with_ttl(
        &self,
        namespace: &Namespace,
        key: &str,
        value: Value,
        ttl_minutes: Option<f64>,
    ) -> Result<()> {
        one(
            self,
            StoreOp::Put {
                namespace: namespace.clone(),
                key: key.to_string(),
                value: Some(value),
                ttl_minutes,
            },
        )
        .await
        .map(|_| ())
    }

    /// Deletes one item. Deleting an absent key is not an error.
    async fn delete(&self, namespace: &Namespace, key: &str) -> Result<()> {
        one(
            self,
            StoreOp::Put {
                namespace: namespace.clone(),
                key: key.to_string(),
                value: None,
                ttl_minutes: None,
            },
        )
        .await
        .map(|_| ())
    }

    /// Searches a namespace subtree.
    async fn search(&self, query: SearchQuery) -> Result<Vec<Item>> {
        one(self, StoreOp::Search(query)).await?.into_items()
    }

    /// Lists namespaces matching `query`.
    async fn list_namespaces(&self, query: ListNamespacesQuery) -> Result<Vec<Namespace>> {
        one(self, StoreOp::ListNamespaces(query))
            .await?
            .into_namespaces()
    }
}

/// Submits a single operation and unwraps the one-element result vector.
async fn one<S>(store: &S, op: StoreOp) -> Result<StoreResult>
where
    S: NamespacedStore + ?Sized,
{
    let mut results = store.batch(std::slice::from_ref(&op)).await?;
    if results.len() != 1 {
        return Err(TinyAgentsError::Validation(format!(
            "store batch returned {} results for 1 operation — results must be \
             positionally aligned with the request",
            results.len()
        )));
    }
    Ok(results.remove(0))
}

// ── InMemoryNamespacedStore ──────────────────────────────────────────────────

/// In-process [`NamespacedStore`] backed by a map, with working TTL.
///
/// Cheap to clone; clones share the same data and TTL policy.
#[derive(Clone, Debug, Default)]
pub struct InMemoryNamespacedStore {
    items: Arc<Mutex<HashMap<(Namespace, String), Item>>>,
    ttl: TtlConfig,
}

impl InMemoryNamespacedStore {
    /// Creates an empty store with no expiry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the TTL policy.
    pub fn with_ttl(mut self, ttl: TtlConfig) -> Self {
        self.ttl = ttl;
        self
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<(Namespace, String), Item>>> {
        self.items
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("store lock poisoned: {e}")))
    }
}

/// Computes an expiry timestamp from a lifetime in minutes.
fn expiry_at(now: u64, ttl_minutes: Option<f64>) -> Option<u64> {
    ttl_minutes
        .filter(|m| m.is_finite() && *m > 0.0)
        .map(|m| now.saturating_add((m * 60_000.0) as u64))
}

/// Reads a dotted field path out of a JSON value.
fn field<'v>(value: &'v Value, path: &str) -> Option<&'v Value> {
    let mut cursor = value;
    for segment in path.split('.') {
        cursor = cursor.get(segment)?;
    }
    Some(cursor)
}

/// Whether `item` satisfies every condition in `filter`.
fn matches_filter(item: &Item, filter: &HashMap<String, FilterOp>) -> bool {
    filter
        .iter()
        .all(|(path, op)| op.matches(field(&item.value, path)))
}

/// Whether `item`'s rendered JSON contains `needle`, case-insensitively.
fn matches_query(item: &Item, needle: &str) -> bool {
    item.value
        .to_string()
        .to_lowercase()
        .contains(&needle.to_lowercase())
}

/// Whether `segments` matches `pattern`, where `*` matches any one segment.
fn matches_wildcards(segments: &[String], pattern: &[String]) -> bool {
    segments.len() == pattern.len()
        && segments
            .iter()
            .zip(pattern)
            .all(|(actual, expected)| expected == "*" || actual == expected)
}

#[async_trait]
impl NamespacedStore for InMemoryNamespacedStore {
    fn ttl_config(&self) -> TtlConfig {
        self.ttl
    }

    async fn sweep_expired(&self) -> Result<usize> {
        let now = now_ms();
        let mut items = self.lock()?;
        let before = items.len();
        items.retain(|_, item| !item.is_expired(now));
        let reclaimed = before - items.len();
        if reclaimed > 0 {
            tracing::debug!("[store:namespaced] sweep_expired reclaimed={reclaimed}");
        }
        Ok(reclaimed)
    }

    async fn batch(&self, ops: &[StoreOp]) -> Result<Vec<StoreResult>> {
        let now = now_ms();
        let mut items = self.lock()?;
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                StoreOp::Get {
                    namespace,
                    key,
                    refresh_ttl,
                } => {
                    namespace.validate()?;
                    let refresh = refresh_ttl.unwrap_or(self.ttl.refresh_on_read);
                    let addr = (namespace.clone(), key.clone());
                    let found = match items.get_mut(&addr) {
                        Some(item) if item.is_expired(now) => {
                            // Expiry is enforced on read, not only by the
                            // sweeper, so a stale item is never observable.
                            items.remove(&addr);
                            None
                        }
                        Some(item) => {
                            if refresh && let Some(minutes) = self.ttl.default_ttl_minutes {
                                item.expires_at_ms = expiry_at(now, Some(minutes));
                                item.updated_at_ms = now;
                            }
                            Some(item.clone())
                        }
                        None => None,
                    };
                    out.push(StoreResult::Item(found.map(Box::new)));
                }
                StoreOp::Put {
                    namespace,
                    key,
                    value,
                    ttl_minutes,
                } => {
                    namespace.validate()?;
                    let addr = (namespace.clone(), key.clone());
                    match value {
                        None => {
                            items.remove(&addr);
                        }
                        Some(value) => {
                            let ttl = ttl_minutes.or(self.ttl.default_ttl_minutes);
                            let created = items.get(&addr).map_or(now, |i| i.created_at_ms);
                            items.insert(
                                addr,
                                Item {
                                    namespace: namespace.clone(),
                                    key: key.clone(),
                                    value: value.clone(),
                                    created_at_ms: created,
                                    updated_at_ms: now,
                                    expires_at_ms: expiry_at(now, ttl),
                                },
                            );
                        }
                    }
                    out.push(StoreResult::Ack);
                }
                StoreOp::Search(query) => {
                    let mut found: Vec<Item> = items
                        .values()
                        .filter(|item| !item.is_expired(now))
                        .filter(|item| item.namespace.starts_with(&query.namespace_prefix))
                        .filter(|item| matches_filter(item, &query.filter))
                        .filter(|item| {
                            query
                                .query
                                .as_ref()
                                .is_none_or(|needle| matches_query(item, needle))
                        })
                        .cloned()
                        .collect();
                    // A HashMap has no order, so impose a deterministic one:
                    // paging is meaningless without it.
                    found.sort_by(|a, b| {
                        a.namespace
                            .cmp(&b.namespace)
                            .then_with(|| a.key.cmp(&b.key))
                    });
                    let windowed: Vec<Item> = found
                        .into_iter()
                        .skip(query.offset)
                        .take(query.limit.unwrap_or(usize::MAX))
                        .collect();
                    out.push(StoreResult::Items(windowed));
                }
                StoreOp::ListNamespaces(query) => {
                    let mut seen: BTreeSet<Namespace> = BTreeSet::new();
                    for item in items.values().filter(|i| !i.is_expired(now)) {
                        let ns = &item.namespace;
                        if let Some(prefix) = &query.prefix
                            && !(ns.segments().len() >= prefix.len()
                                && matches_wildcards(&ns.segments()[..prefix.len()], prefix))
                        {
                            continue;
                        }
                        if let Some(suffix) = &query.suffix {
                            let len = ns.segments().len();
                            if !(len >= suffix.len()
                                && matches_wildcards(&ns.segments()[len - suffix.len()..], suffix))
                            {
                                continue;
                            }
                        }
                        // `max_depth` truncates then deduplicates, which is how
                        // one level of the hierarchy gets enumerated.
                        let truncated = match query.max_depth {
                            Some(depth) if ns.segments().len() > depth => {
                                Namespace(ns.segments()[..depth].to_vec())
                            }
                            _ => ns.clone(),
                        };
                        seen.insert(truncated);
                    }
                    let windowed: Vec<Namespace> = seen
                        .into_iter()
                        .skip(query.offset)
                        .take(query.limit.unwrap_or(usize::MAX))
                        .collect();
                    out.push(StoreResult::Namespaces(windowed));
                }
            }
        }
        Ok(out)
    }
}

// ── Compatibility with the flat `Store` trait ────────────────────────────────

/// Exposes any [`NamespacedStore`] through the flat [`Store`] trait.
///
/// The flat namespace becomes a single-segment [`Namespace`], so existing
/// callers keep working byte-for-byte while the data lives in a store that also
/// supports hierarchy, filtering and TTL. This is what makes the new trait
/// additive rather than a migration.
#[derive(Clone, Debug)]
pub struct FlatNamespacedStore<S> {
    inner: S,
}

impl<S> FlatNamespacedStore<S> {
    /// Wraps `inner`.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Returns the wrapped store.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

#[async_trait]
impl<S: NamespacedStore> Store for FlatNamespacedStore<S> {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<Value>> {
        let ns = Namespace::new([namespace])?;
        Ok(self.inner.get(&ns, key).await?.map(|item| item.value))
    }

    async fn put(&self, namespace: &str, key: &str, value: Value) -> Result<()> {
        let ns = Namespace::new([namespace])?;
        self.inner.put(&ns, key, value).await
    }

    async fn delete(&self, namespace: &str, key: &str) -> Result<()> {
        let ns = Namespace::new([namespace])?;
        self.inner.delete(&ns, key).await
    }

    async fn list(&self, namespace: &str) -> Result<Vec<String>> {
        let ns = Namespace::new([namespace])?;
        Ok(self
            .inner
            .search(SearchQuery {
                namespace_prefix: ns.0.clone(),
                ..SearchQuery::default()
            })
            .await?
            .into_iter()
            // `starts_with` is a subtree match; the flat `list` contract is
            // "keys in exactly this namespace".
            .filter(|item| item.namespace == ns)
            .map(|item| item.key)
            .collect())
    }
}

#[cfg(test)]
mod test;
