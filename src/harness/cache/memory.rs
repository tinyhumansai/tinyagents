//! The in-process [`InMemoryResponseCache`] implementation.

use std::time::{Duration, Instant};

use async_trait::async_trait;

use super::types::{CacheEntry, CacheStats, InMemoryResponseCache, LruResponseMap, ResponseCache};
use crate::error::{Result, TinyAgentsError};
use crate::harness::model::ModelResponse;

impl InMemoryResponseCache {
    /// Default LRU capacity, in entries, when constructed via
    /// [`new`](Self::new) or [`Default`].
    pub const DEFAULT_CAPACITY: usize = 1024;

    /// Default approximate byte budget (64 MiB).
    ///
    /// An entry count alone does not bound memory: 1024 long-context responses
    /// carrying large tool payloads is hundreds of megabytes.
    pub const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;

    /// Creates a new, empty in-memory response cache bounded by
    /// [`DEFAULT_CAPACITY`](Self::DEFAULT_CAPACITY) entries and
    /// [`DEFAULT_MAX_BYTES`](Self::DEFAULT_MAX_BYTES) bytes.
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Creates a new, empty in-memory response cache retaining at most
    /// `capacity` entries (least-recently-used evicted first). A `capacity` of
    /// zero is treated as `1` so the cache always retains the last write.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_bounds(capacity, Self::DEFAULT_MAX_BYTES)
    }

    /// Creates a new, empty in-memory response cache bounded by **both** an
    /// entry count and an approximate byte budget. Whichever bound trips first
    /// evicts the least-recently-used entry.
    ///
    /// Both bounds are clamped to a minimum of `1` so the cache always retains
    /// the most recent write.
    pub fn with_bounds(capacity: usize, max_bytes: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(LruResponseMap {
                data: std::collections::HashMap::new(),
                order: std::collections::BTreeMap::new(),
                next_recency: 0,
                capacity: capacity.max(1),
                max_bytes: max_bytes.max(1),
                bytes: 0,
                stats: CacheStats::default(),
            })),
        }
    }

    /// Locks the inner map, mapping a poisoned mutex to a validation error.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, LruResponseMap>> {
        self.inner
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("cache lock poisoned: {e}")))
    }
}

impl Default for InMemoryResponseCache {
    fn default() -> Self {
        Self::new()
    }
}

impl LruResponseMap {
    /// Hands out the next monotonic recency ticket.
    fn tick(&mut self) -> u64 {
        self.next_recency = self.next_recency.wrapping_add(1);
        self.next_recency
    }

    /// Moves `key` to the most-recently-used end.
    ///
    /// `O(log n)`: one `BTreeMap` removal plus one insertion. The previous
    /// implementation scanned a `VecDeque` linearly — up to `capacity` `String`
    /// comparisons plus a memmove — on **every** hit.
    fn touch(&mut self, key: &str) {
        let Some(entry) = self.data.get(key) else {
            return;
        };
        let old = entry.recency;
        let next = self.tick();
        self.order.remove(&old);
        self.order.insert(next, key.to_string());
        if let Some(entry) = self.data.get_mut(key) {
            entry.recency = next;
        }
    }

    /// Removes `key`, keeping the byte accounting and order index consistent.
    fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        let entry = self.data.remove(key)?;
        self.order.remove(&entry.recency);
        self.bytes = self.bytes.saturating_sub(entry.bytes);
        Some(entry)
    }

    /// Evicts least-recently-used entries until both bounds are satisfied.
    fn evict_to_fit(&mut self) {
        while self.data.len() > self.capacity
            || (self.bytes > self.max_bytes && self.data.len() > 1)
        {
            let Some((_, victim)) = self.order.iter().next().map(|(k, v)| (*k, v.clone())) else {
                break;
            };
            self.remove(&victim);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            tracing::trace!(key = %victim, "[cache] evicted least-recently-used entry");
        }
    }

    /// Refreshes the derived size counters exposed through
    /// [`ResponseCache::stats`].
    fn sync_size_stats(&mut self) {
        self.stats.entries = self.data.len() as u64;
        self.stats.bytes = self.bytes as u64;
    }
}

#[async_trait]
impl ResponseCache for InMemoryResponseCache {
    async fn get(&self, key: &str) -> Result<Option<ModelResponse>> {
        let mut inner = self.lock()?;
        // Lazy expiry: an entry whose TTL elapsed is a miss and is dropped on
        // the way past, so a cache that is read but never written still sheds
        // stale entries.
        if let Some(entry) = inner.data.get(key)
            && entry.expires_at.is_some_and(|at| at <= Instant::now())
        {
            inner.remove(key);
            inner.stats.expirations = inner.stats.expirations.saturating_add(1);
            inner.stats.misses = inner.stats.misses.saturating_add(1);
            inner.sync_size_stats();
            tracing::debug!(key = %key, "[cache] entry expired; treating as miss");
            return Ok(None);
        }
        let hit = inner.data.get(key).map(|entry| entry.value.clone());
        if hit.is_some() {
            inner.touch(key);
            inner.stats.hits = inner.stats.hits.saturating_add(1);
        } else {
            inner.stats.misses = inner.stats.misses.saturating_add(1);
        }
        Ok(hit)
    }

    async fn put(&self, key: &str, value: ModelResponse) -> Result<()> {
        self.put_with_ttl(key, value, None).await
    }

    async fn put_with_ttl(
        &self,
        key: &str,
        value: ModelResponse,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let bytes = serde_json::to_vec(&value).map(|v| v.len()).unwrap_or(0);
        let mut inner = self.lock()?;
        inner.remove(key);
        let recency = inner.tick();
        inner.order.insert(recency, key.to_string());
        inner.bytes = inner.bytes.saturating_add(bytes);
        inner.data.insert(
            key.to_string(),
            CacheEntry {
                value,
                recency,
                bytes,
                expires_at: ttl.map(|ttl| Instant::now() + ttl),
            },
        );
        inner.stats.writes = inner.stats.writes.saturating_add(1);
        inner.evict_to_fit();
        inner.sync_size_stats();
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut inner = self.lock()?;
        let dropped = inner.data.len();
        inner.data.clear();
        inner.order.clear();
        inner.bytes = 0;
        inner.sync_size_stats();
        tracing::debug!(dropped, "[cache] cleared every in-memory response entry");
        Ok(())
    }

    fn stats(&self) -> CacheStats {
        self.lock().map(|inner| inner.stats).unwrap_or_default()
    }
}
