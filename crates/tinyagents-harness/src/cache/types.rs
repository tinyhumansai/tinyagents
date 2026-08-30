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

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use tinyinference::model::ModelResponse;

// ── CacheStats ────────────────────────────────────────────────────────────────

/// Point-in-time counters describing a [`ResponseCache`]'s behaviour.
///
/// A caller whose hit rate is unexpectedly 0% has no way to tell an
/// over-inclusive key from a policy that never enabled caching at all; these
/// counters plus [`CacheSkipReason`] are the diagnostic. Implementations that
/// cannot cheaply account for a field leave it at zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStats {
    /// Lookups that returned a live entry.
    pub hits: u64,
    /// Lookups that returned nothing (including entries dropped as expired).
    pub misses: u64,
    /// Writes accepted by the cache.
    pub writes: u64,
    /// Entries dropped because they exceeded a capacity bound.
    pub evictions: u64,
    /// Entries dropped because their TTL had elapsed.
    pub expirations: u64,
    /// Entries currently retained.
    pub entries: u64,
    /// Approximate serialized size of the retained entries, in bytes.
    pub bytes: u64,
}

// ── CacheSkipReason ───────────────────────────────────────────────────────────

/// Why the agent loop declined to consult the response cache for a call.
///
/// `docs/modules/harness/cache.md` specifies a richer decision surface than the
/// bare `CacheHit`/`CacheMiss` pair; this enum carries the "no lookup happened
/// at all" half so a 0% hit rate is always explainable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheSkipReason {
    /// No [`ResponseCache`] is attached to the harness.
    NoCacheAttached,
    /// The effective [`CachePolicy`] disables response caching for this call.
    PolicyDisabled,
    /// The transcript already contains an assistant or tool turn, so the
    /// request is unique to this run and can never be re-served.
    MultiTurnTranscript,
}

impl CacheSkipReason {
    /// A short, stable, grep-friendly token for logs and events.
    pub fn as_str(self) -> &'static str {
        match self {
            CacheSkipReason::NoCacheAttached => "no_cache_attached",
            CacheSkipReason::PolicyDisabled => "policy_disabled",
            CacheSkipReason::MultiTurnTranscript => "multi_turn_transcript",
        }
    }
}

// ── ResponseCache ─────────────────────────────────────────────────────────────

/// Local response cache that lets the harness skip provider calls entirely.
///
/// Keys should be produced by [`super::cache_key`] (folded with the resolved
/// model's [`cache_identity`][tinyinference::model::ChatModel::cache_identity]
/// via [`super::scoped_cache_key`]) for consistency. Callers are responsible for
/// deciding when caching is safe (e.g., not caching side-effecting tool calls).
///
/// # Implementing
///
/// Only [`get`](Self::get) and [`put`](Self::put) are required. TTL support,
/// bulk invalidation, and statistics are optional: the defaults below keep an
/// existing third-party implementation compiling and behaving exactly as before.
#[async_trait]
pub trait ResponseCache: Send + Sync {
    /// Returns the cached [`ModelResponse`] for `key`, or `None` on a miss.
    ///
    /// An implementation that supports TTLs must treat an expired entry as a
    /// miss (and should drop it).
    async fn get(&self, key: &str) -> Result<Option<ModelResponse>>;

    /// Stores `value` under `key` with no expiry.
    async fn put(&self, key: &str, value: ModelResponse) -> Result<()>;

    /// Stores `value` under `key`, expiring it after `ttl` when supported.
    ///
    /// The default implementation ignores `ttl` and delegates to
    /// [`put`](Self::put), so a cache without expiry support keeps working;
    /// implementations that can expire entries should override this and
    /// implement `put` as `put_with_ttl(key, value, None)`.
    async fn put_with_ttl(
        &self,
        key: &str,
        value: ModelResponse,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let _ = ttl;
        self.put(key, value).await
    }

    /// Drops every entry.
    ///
    /// Needed because a poisoned entry (for example one written before a bug
    /// fix changed the key derivation) is otherwise permanent in a cache with
    /// no TTL. The default is a no-op `Ok(())` so existing implementations do
    /// not break; a cache that cannot clear should say so in its own docs.
    async fn clear(&self) -> Result<()> {
        Ok(())
    }

    /// Returns point-in-time counters for this cache.
    ///
    /// Defaults to all-zero for implementations that do not account.
    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

/// Thread-safe in-memory response cache.
///
/// Intended for unit tests and short-lived local runs. Contains no durable
/// storage: all entries are lost when the value is dropped. For a cache that
/// survives a restart see `SqliteResponseCache` (feature `sqlite`).
///
/// Entries are bounded on **two** axes so a long-lived cache attached to a busy
/// harness cannot grow without limit:
///
/// * an entry count (default [`InMemoryResponseCache::DEFAULT_CAPACITY`]), and
/// * an approximate byte budget (default
///   [`InMemoryResponseCache::DEFAULT_MAX_BYTES`]) — 1024 long-context
///   responses carrying large tool payloads is hundreds of megabytes, which an
///   entry count alone does not bound.
///
/// Whichever bound trips first evicts the least-recently-used entry. Reads and
/// writes move a key to the most-recently-used end. Per-entry TTLs are honored:
/// an expired entry is dropped on read and reported as a miss.
#[derive(Clone, Debug)]
pub struct InMemoryResponseCache {
    pub(crate) inner: Arc<Mutex<LruResponseMap>>,
}

/// One retained response plus its bookkeeping.
#[derive(Clone, Debug)]
pub(crate) struct CacheEntry {
    /// The cached response.
    pub(crate) value: ModelResponse,
    /// Monotonic recency ticket; larger is more recently used.
    pub(crate) recency: u64,
    /// Approximate serialized size in bytes, used for the byte bound.
    pub(crate) bytes: usize,
    /// Absolute expiry instant, when a TTL was supplied.
    pub(crate) expires_at: Option<std::time::Instant>,
}

/// LRU-ordered map backing [`InMemoryResponseCache`].
///
/// Recency is tracked with a monotonic ticket per entry plus a `BTreeMap`
/// ordered by that ticket. Touching a key is `O(log n)` (one `BTreeMap` remove
/// plus one insert) and eviction pops the first entry, also `O(log n)`. The
/// previous `VecDeque` layout scanned linearly, comparing up to `capacity`
/// `String`s and memmoving the deque **on every hit** — 1024 string compares
/// per cached call at the default capacity.
#[derive(Debug)]
pub(crate) struct LruResponseMap {
    /// Cached responses keyed by cache key.
    pub(crate) data: HashMap<String, CacheEntry>,
    /// Recency ticket -> key, ordered least- to most-recently-used.
    pub(crate) order: BTreeMap<u64, String>,
    /// Next recency ticket to hand out.
    pub(crate) next_recency: u64,
    /// Maximum number of entries retained before LRU eviction.
    pub(crate) capacity: usize,
    /// Maximum approximate total bytes retained before LRU eviction.
    pub(crate) max_bytes: usize,
    /// Current approximate total bytes retained.
    pub(crate) bytes: usize,
    /// Running counters exposed through [`ResponseCache::stats`].
    pub(crate) stats: CacheStats,
}

// ── PromptCacheLayout ─────────────────────────────────────────────────────────

/// A snapshot of the ordered cacheable prompt-segment prefix that the provider
/// will see and may cache in its own KV store.
///
/// The harness computes a `PromptCacheLayout` before and after each middleware
/// pass so it can detect and report accidental prefix invalidations.
///
/// # Content awareness
///
/// The layout records **both** the declared segment identities *and* a digest
/// of the material those segments carry:
///
/// * [`Self::prefix_ids`] — the ordered ids of cacheable segments,
/// * [`Self::fingerprint`] — an FNV-1a digest over each cacheable segment's
///   `(id, role)` pair *and* the request's
///   [`prompt_fingerprint`][tinyinference::model::ModelRequest::prompt_fingerprint]
///   and tool schemas, and
/// * a per-message digest chain, so "did the prompt only grow at the tail?"
///   — the provider's actual KV-prefix rule — is answerable.
///
/// Comparing ids alone reported "prefix stable" after a middleware rewrote the
/// **text** of a stable segment, which is precisely the failure this type
/// exists to catch: the ids match while the provider's cached bytes are gone.
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
    /// Deterministic content-aware fingerprint (16 lowercase hex chars).
    pub(crate) fingerprint: String,
    /// Per-message digests in transcript order, used to decide whether one
    /// layout's message stream is a pure tail-extension of another's.
    pub(crate) message_digests: Vec<String>,
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
    /// `true` if the segment identities were preserved but their **content**
    /// changed — an edit that leaves the ids matching while destroying the
    /// provider's cached bytes.
    pub content_only_change: bool,
    /// `true` if a [`CachePolicy::protect_prompt_prefix`] was in force and this
    /// change violates it. Always `false` for the policy-free
    /// [`CacheLayoutEvent::new`] constructor.
    pub violates_policy: bool,
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
///
/// **This type is deliberately excluded from [`super::cache_key`]**: it selects
/// *whether* to cache, never *what the model answers*, so folding it into the
/// key made flipping [`Self::protect_prompt_prefix`] silently invalidate every
/// existing entry.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePolicy {
    /// When `true`, the harness will look up (and write) local response cache
    /// entries via [`ResponseCache`] before calling the provider.
    pub response_cache_enabled: bool,
    /// When `true`, middleware must preserve the order **and content** of
    /// cacheable prefix segments. Violations are reported as
    /// [`CacheLayoutEvent`]s with `violates_policy: true`, and the harness
    /// additionally derives a provider `prompt_cache_key` breakpoint from the
    /// stable prefix (see [`super::apply_prompt_cache_breakpoints`]) so one
    /// logical thread routes to the same provider cache shard.
    pub protect_prompt_prefix: bool,
    /// Time-to-live for entries written under this policy, in milliseconds.
    ///
    /// `None` means "never expire", which is the historical behaviour. A TTL is
    /// the only bound on a poisoned entry in a cache that is never cleared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<u64>,
    /// Optional key namespace, folded into the cache key.
    ///
    /// Lets one shared cache serve several logically separate populations
    /// (per-tenant, per-experiment) without the risk of cross-serving, and lets
    /// a caller invalidate one population by rotating the namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl CachePolicy {
    /// A policy with local response caching enabled and no expiry.
    pub fn enabled() -> Self {
        Self {
            response_cache_enabled: true,
            ..Self::default()
        }
    }

    /// Returns this policy's TTL as a [`Duration`], when one is set.
    pub fn ttl(&self) -> Option<Duration> {
        self.ttl_ms.map(Duration::from_millis)
    }

    /// Sets the entry time-to-live.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl_ms = Some(ttl.as_millis() as u64);
        self
    }

    /// Sets the key namespace.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }
}
