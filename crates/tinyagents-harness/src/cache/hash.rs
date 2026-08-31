//! Deterministic hashing primitives shared by the cache key derivation
//! ([`super::key`]) and the prompt-cache layout tooling ([`super::layout`]).
//!
//! Everything here is seed-free and canonical so a digest computed in one
//! process matches one computed in another — a cache that is only valid within
//! a single process lifetime is not a cache.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Renders a finalized SHA-256 digest as a 64-character lowercase hex string.
pub(super) fn hex_digest(digest: impl AsRef<[u8]>) -> String {
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
pub(super) fn fold_canonical(hasher: &mut Sha256, tag: u8, value: Value) {
    fold_bytes(
        hasher,
        tag,
        &serde_json::to_vec(&canonical_value(value)).unwrap_or_default(),
    );
}

/// Folds raw `bytes` into `hasher` as a self-delimiting `tag`-ed frame.
pub(super) fn fold_bytes(hasher: &mut Sha256, tag: u8, bytes: &[u8]) {
    hasher.update([tag]);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Computes a deterministic FNV-1a 64-bit hash over `data` and returns it as
/// a 16-character lowercase hex string.
///
/// FNV-1a uses a fixed, seed-free offset basis so the result is identical
/// across process restarts — unlike Rust's default `SipHash`, which is seeded
/// randomly at startup. It is used only for short local prompt-layout
/// fingerprints, not for response-cache identity.
pub(super) fn fnv1a_hex(data: &[u8]) -> String {
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
pub(super) fn canonical_value(v: Value) -> Value {
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
