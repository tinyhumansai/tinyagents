//! Cache types for the harness cache module.
//!
//! These types let the recursive runtime answer a recurring request without
//! re-contacting the provider (response cache) and keep a provider's own
//! KV-cache prefix stable across the many requests a nested run produces
//! (layout protection).
//!
//! Two distinct caching concerns are modelled here:
//!
//! 1. **Local response cache** ([`ResponseCache`], [`InMemoryResponseCache`]):
//!    a harness-side cache that lets the harness skip provider calls entirely
//!    when the identical request has already been answered.
//!
//! 2. **Provider prompt / KV-cache layout protection** ([`PromptCacheLayout`],
//!    [`CacheLayoutEvent`], [`CachePolicy`]): tooling for preserving the
//!    stable byte-level prefix that the provider will cache in its own KV
//!    store, without caching the actual response locally.
//!
//! All public types in this module are re-exported through [`super`].

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::harness::model::ModelResponse;

// ── ResponseCache ─────────────────────────────────────────────────────────────

/// Local response cache that lets the harness skip provider calls entirely.
///
/// Keys should be produced by [`super::cache_key`] for consistency. Callers
/// are responsible for deciding when caching is safe (e.g., not caching
/// side-effecting tool calls).
#[async_trait]
pub trait ResponseCache: Send + Sync {
    /// Returns the cached [`ModelResponse`] for `key`, or `None` on a miss.
    async fn get(&self, key: &str) -> Result<Option<ModelResponse>>;

    /// Stores `value` under `key`.
    async fn put(&self, key: &str, value: ModelResponse) -> Result<()>;
}

/// Thread-safe in-memory response cache.
///
/// Intended for unit tests and short-lived local runs. Contains no durable
/// storage: all entries are lost when the value is dropped.
///
/// Entries are bounded by an LRU eviction policy (default
/// [`InMemoryResponseCache::DEFAULT_CAPACITY`]) so a long-lived cache attached
/// to a busy harness cannot grow without limit. Reads and writes move a key to
/// the most-recently-used end; once the map is full the least-recently-used key
/// is evicted on insert.
#[derive(Clone, Debug)]
pub struct InMemoryResponseCache {
    pub(crate) inner: Arc<Mutex<LruResponseMap>>,
}

/// LRU-ordered map backing [`InMemoryResponseCache`].
#[derive(Debug)]
pub(crate) struct LruResponseMap {
    /// Cached responses keyed by cache key.
    pub(crate) data: HashMap<String, ModelResponse>,
    /// Keys in least- to most-recently-used order.
    pub(crate) order: VecDeque<String>,
    /// Maximum number of entries retained before LRU eviction.
    pub(crate) capacity: usize,
}

// ── PromptCacheLayout ─────────────────────────────────────────────────────────

/// A snapshot of the ordered cacheable prompt-segment prefix that the provider
/// will see and may cache in its own KV store.
///
/// The harness computes a `PromptCacheLayout` before and after each middleware
/// pass so it can detect and report accidental prefix invalidations.
///
/// # Provider KV-cache stability rules
/// - Never insert timestamps, run ids, or dynamic retrieval output into the
///   stable prefix.
/// - Volatile content (latest user turn, tool results, scratchpads) should
///   always follow stable segments.
/// - Segment ordering must be preserved unless a middleware explicitly declares
///   a cache-layout migration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptCacheLayout {
    /// Ordered ids of cacheable (stable) prefix segments.
    pub(crate) prefix_ids: Vec<String>,
    /// Deterministic fingerprint of the ordered prefix ids.
    pub(crate) fingerprint: String,
}

// ── CacheLayoutEvent ──────────────────────────────────────────────────────────

/// Describes a change to the prompt cache layout that middleware can emit.
///
/// Consumers (observability sinks, cost accounting, regression tests) can
/// inspect this struct to understand why a provider prompt-cache prefix was
/// preserved or invalidated.
#[derive(Clone, Debug)]
pub struct CacheLayoutEvent {
    /// `true` if the cacheable prefix changed between `segment_ids_before` and
    /// `segment_ids_after`.
    pub changed_prefix: bool,
    /// `true` if `segment_ids_after` contains only volatile (non-cacheable)
    /// segments, meaning no stable prefix is present.
    pub volatile_only: bool,
    /// The ordered cacheable prefix ids before the middleware pass.
    pub segment_ids_before: Vec<String>,
    /// The ordered cacheable prefix ids after the middleware pass.
    pub segment_ids_after: Vec<String>,
}

// ── CachePolicy ───────────────────────────────────────────────────────────────

/// Policy knobs controlling both response caching and provider prompt-cache
/// layout protection.
///
/// Both flags default to `false` (no caching / no protection) so the harness
/// is safe-by-default and opts must be explicit.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePolicy {
    /// When `true`, the harness will look up (and write) local response cache
    /// entries via [`ResponseCache`] before calling the provider.
    pub response_cache_enabled: bool,
    /// When `true`, middleware must preserve the order and content of cacheable
    /// prefix segments. Violations are reported as [`CacheLayoutEvent`]s.
    pub protect_prompt_prefix: bool,
}
