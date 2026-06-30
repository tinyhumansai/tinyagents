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

/// Computes a deterministic SHA-256 digest over `data` and returns it as a
/// 64-character lowercase hex string.
fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
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
/// The key is a 16-character lowercase hex string derived from:
/// 1. Serializing `request` to a [`serde_json::Value`].
/// 2. Sorting all JSON object keys recursively (canonical form).
/// 3. Re-serializing to a compact JSON string.
/// 4. Applying SHA-256 to the canonical bytes.
///
/// Because every behavior-affecting field of [`ModelRequest`] participates in
/// the serialization, two requests that differ in any field produce different
/// canonical bytes and should produce different keys modulo SHA-256 collision
/// resistance.
///
/// # Panics
/// Does not panic. If serialization unexpectedly fails, returns an empty-data
/// hash.
pub fn cache_key(request: &ModelRequest) -> String {
    let value = serde_json::to_value(request).unwrap_or(Value::Null);
    let canonical = canonical_value(value);
    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    sha256_hex(&bytes)
}

// ── InMemoryResponseCache ─────────────────────────────────────────────────────

impl InMemoryResponseCache {
    /// Creates a new, empty in-memory response cache.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ResponseCache for InMemoryResponseCache {
    async fn get(&self, key: &str) -> Result<Option<ModelResponse>> {
        let data = self
            .data
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("cache lock poisoned: {e}")))?;
        Ok(data.get(key).cloned())
    }

    async fn put(&self, key: &str, value: ModelResponse) -> Result<()> {
        let mut data = self
            .data
            .lock()
            .map_err(|e| TinyAgentsError::Validation(format!("cache lock poisoned: {e}")))?;
        data.insert(key.to_string(), value);
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
