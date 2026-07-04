//! Harness cache module — prompt, response, and layout caches.
//!
//! In the recursive runtime the same request can recur many times — a
//! sub-agent re-asked an identical sub-question, a graph node replayed during
//! recovery, or a deterministic test driving the loop twice. This module makes
//! that recursion cheap and deterministic: the local response cache short-
//! circuits an identical model call entirely (the agent loop emits
//! [`crate::harness::events::AgentEvent::CacheHit`] /
//! [`crate::harness::events::AgentEvent::CacheMiss`]), while the prompt-cache
//! layout tooling protects the stable prefix the *provider* itself caches.
//!
//! # Two distinct caching concerns
//!
//! ## 1. Local response cache
//! [`ResponseCache`] + [`InMemoryResponseCache`] let the harness skip provider
//! API calls entirely when it has already seen an identical request. Use
//! [`cache_key`] to produce a stable, deterministic key from a
//! [`crate::harness::model::ModelRequest`].
//!
//! ## 2. Provider prompt / KV-cache layout protection
//! [`PromptCacheLayout`] records the ordered cacheable prefix of a request.
//! [`CacheLayoutEvent`] describes mutations so middleware can signal whether it
//! preserved or invalidated the provider's KV-cache prefix.
//! [`CachePolicy`] toggles both concerns at the call-site level.
//!
mod types;

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub use types::*;

use crate::error::{Result, TinyAgentsError};
use crate::harness::model::{ModelRequest, ModelResponse};

// ── Deterministic hash ────────────────────────────────────────────────────────

/// Renders a finalized SHA-256 digest as a 64-character lowercase hex string.
fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Folds one JSON `value` into `hasher` as a self-delimiting frame: an ASCII
/// domain `tag`, then the canonical byte length (little-endian `u64`), then the
/// canonical bytes.
///
/// Canonicalizing per component keeps peak memory bounded by the single largest
/// value rather than the whole request tree, and the length prefix makes the
/// concatenation of frames unambiguous — no two distinct component sequences
/// can hash to the same byte stream.
fn fold_canonical(hasher: &mut Sha256, tag: u8, value: Value) {
    let bytes = serde_json::to_vec(&canonical_value(value)).unwrap_or_default();
    hasher.update([tag]);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
}

/// Computes a deterministic FNV-1a 64-bit hash over `data` and returns it as
/// a 16-character lowercase hex string.
///
/// FNV-1a uses a fixed, seed-free offset basis so the result is identical
/// across process restarts — unlike Rust's default `SipHash`, which is seeded
/// randomly at startup. It is used only for short local prompt-layout
/// fingerprints, not for response-cache identity.
fn fnv1a_hex(data: &[u8]) -> String {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:016x}")
}

/// Recursively sorts the keys of every JSON object so that the serialized form
/// is canonical regardless of insertion order.
fn canonical_value(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(String, Value)> = map.into_iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(
                pairs
                    .into_iter()
                    .map(|(k, val)| (k, canonical_value(val)))
                    .collect(),
            )
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(canonical_value).collect()),
        other => other,
    }
}

// ── cache_key ─────────────────────────────────────────────────────────────────

/// Produces a stable, deterministic cache key for `request`.
///
/// The key is a 64-character lowercase SHA-256 hex string built by folding the
/// request into the hasher **incrementally**, one component at a time:
/// 1. Serialize `request` once to a [`serde_json::Value`].
/// 2. Fold each conversation message as its own length-prefixed, canonicalized
///    frame (tag `M`), preceded by the message count.
/// 3. Fold each tool schema likewise (tag `T`), preceded by the tool count.
/// 4. Fold the remaining scalar/parameter fields — everything left in the
///    request object once `messages` and `tools` are removed — as one envelope
///    frame (tag `E`).
///
/// This avoids the previous approach's three simultaneous whole-transcript
/// allocations (a full `Value` tree, a full canonical rebuild, and a full byte
/// buffer). Transcripts routinely carry large tool results; canonicalizing and
/// serializing per component bounds peak memory by the single largest component
/// instead of the entire request. The envelope is taken as "whatever remains"
/// so that any future [`ModelRequest`] field automatically participates in the
/// key — no field can silently drop out and cause a false cache hit.
///
/// Distinct requests still map to distinct keys (modulo SHA-256 collision
/// resistance): per-component length prefixes make the frame stream
/// unambiguous, and every behavior-affecting field is folded exactly once.
///
/// # Panics
/// Does not panic. If serialization unexpectedly fails, the affected frame
/// folds empty bytes; the key stays well-defined.
pub fn cache_key(request: &ModelRequest) -> String {
    let mut hasher = Sha256::new();
    let mut root = serde_json::to_value(request).unwrap_or(Value::Null);

    if let Value::Object(map) = &mut root {
        // Messages: fold one at a time so a long transcript never materializes
        // a second full tree. The count frame keeps `[a, b]` distinct from a
        // single message that happens to serialize to the same concatenation.
        if let Some(Value::Array(messages)) = map.remove("messages") {
            hasher.update(b"M");
            hasher.update((messages.len() as u64).to_le_bytes());
            for message in messages {
                fold_canonical(&mut hasher, b'm', message);
            }
        }
        // Tool schemas: already name-sorted by `ToolRegistry::schemas`, so the
        // order is deterministic across calls.
        if let Some(Value::Array(tools)) = map.remove("tools") {
            hasher.update(b"T");
            hasher.update((tools.len() as u64).to_le_bytes());
            for tool in tools {
                fold_canonical(&mut hasher, b't', tool);
            }
        }
    }

    // Envelope: every remaining scalar/parameter field in one frame.
    fold_canonical(&mut hasher, b'E', root);
    hex_digest(hasher.finalize())
}

// ── InMemoryResponseCache ─────────────────────────────────────────────────────

impl InMemoryResponseCache {
    /// Default LRU capacity when constructed via [`new`](Self::new) or
    /// [`Default`].
    pub const DEFAULT_CAPACITY: usize = 1024;

    /// Creates a new, empty in-memory response cache bounded by
    /// [`DEFAULT_CAPACITY`](Self::DEFAULT_CAPACITY) entries.
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Creates a new, empty in-memory response cache retaining at most
    /// `capacity` entries (least-recently-used evicted first). A `capacity` of
    /// zero is treated as `1` so the cache always retains the last write.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(LruResponseMap {
                data: std::collections::HashMap::new(),
                order: std::collections::VecDeque::new(),
                capacity: capacity.max(1),
            })),
        }
    }
}

impl Default for InMemoryResponseCache {
    fn default() -> Self {
        Self::new()
    }
}

impl LruResponseMap {
    /// Moves `key` to the most-recently-used end of the order queue.
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos).expect("position is valid");
            self.order.push_back(k);
        }
    }
}

#[async_trait]
impl ResponseCache for InMemoryResponseCache {
    async fn get(&self, key: &str) -> Result<Option<ModelResponse>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("cache lock poisoned: {e}")))?;
        let hit = inner.data.get(key).cloned();
        if hit.is_some() {
            inner.touch(key);
        }
        Ok(hit)
    }

    async fn put(&self, key: &str, value: ModelResponse) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("cache lock poisoned: {e}")))?;
        if inner.data.insert(key.to_string(), value).is_some() {
            // Existing key: refresh its recency without changing the length.
            inner.touch(key);
        } else {
            inner.order.push_back(key.to_string());
            // Evict least-recently-used entries until within capacity.
            while inner.order.len() > inner.capacity {
                if let Some(evicted) = inner.order.pop_front() {
                    inner.data.remove(&evicted);
                }
            }
        }
        Ok(())
    }
}

// ── PromptCacheLayout ─────────────────────────────────────────────────────────

impl PromptCacheLayout {
    /// Builds a [`PromptCacheLayout`] from `request` by collecting the ids of
    /// all cacheable (stable) segments in their declared order.
    ///
    /// The fingerprint is a deterministic FNV-1a hash of the joined prefix ids
    /// so regression tests can assert prefix stability independently of the
    /// full request hash.
    pub fn from_request(request: &ModelRequest) -> Self {
        let prefix_ids: Vec<String> = request.cacheable_prefix_ids();
        let fingerprint = fnv1a_hex(prefix_ids.join(",").as_bytes());
        Self {
            prefix_ids,
            fingerprint,
        }
    }

    /// Returns the ordered ids of cacheable (stable) prefix segments.
    pub fn prefix_ids(&self) -> &[String] {
        &self.prefix_ids
    }

    /// Returns the deterministic fingerprint of the ordered prefix ids.
    ///
    /// Two layouts with identical `prefix_ids` produce the same fingerprint.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Returns `true` if `self` and `other` have the same cacheable prefix ids
    /// in the same order, meaning the provider KV-cache prefix is stable
    /// across the two requests.
    pub fn is_prefix_stable_against(&self, other: &PromptCacheLayout) -> bool {
        self.prefix_ids == other.prefix_ids
    }
}

// ── CacheLayoutEvent ──────────────────────────────────────────────────────────

impl CacheLayoutEvent {
    /// Constructs a [`CacheLayoutEvent`] by comparing `before` and `after`
    /// layouts, filling in the computed `changed_prefix` and `volatile_only`
    /// flags automatically.
    pub fn new(before: &PromptCacheLayout, after: &PromptCacheLayout) -> Self {
        Self {
            changed_prefix: !before.is_prefix_stable_against(after),
            volatile_only: after.prefix_ids().is_empty(),
            segment_ids_before: before.prefix_ids().to_vec(),
            segment_ids_after: after.prefix_ids().to_vec(),
        }
    }
}

#[cfg(test)]
mod test;
