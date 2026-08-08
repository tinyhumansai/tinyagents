//! Provider prompt / KV-cache layout protection.
//!
//! A provider's prompt cache is a **byte prefix** cache: it survives only while
//! the leading bytes of the request are unchanged, and appending to the tail is
//! the one edit that preserves it. This module models that rule directly.

use serde_json::Value;

use super::hash::fnv1a_hex;
use super::types::{CacheLayoutEvent, CachePolicy, PromptCacheLayout};
use crate::harness::model::ModelRequest;

impl PromptCacheLayout {
    /// Builds a [`PromptCacheLayout`] from `request`.
    ///
    /// Captures three things:
    ///
    /// * the ordered ids of cacheable (stable) segments,
    /// * a **content-aware** fingerprint — an FNV-1a hash over each cacheable
    ///   segment's `(id, role)` pair, the request's
    ///   [`prompt_fingerprint`][ModelRequest::prompt_fingerprint] (which
    ///   [`PromptBuilder::fingerprint`][crate::harness::prompt::PromptBuilder::fingerprint]
    ///   derives from the segments' actual messages), and the declared tool
    ///   schemas, and
    /// * a per-message digest chain used by
    ///   [`Self::is_prefix_stable_against`].
    ///
    /// The fingerprint used to hash the joined prefix **ids** only, so editing
    /// the *text* of a stable segment — or swapping a tool schema — reported
    /// "prefix stable" while the provider's KV prefix was already destroyed.
    ///
    /// # Cost
    /// One serialization pass over the transcript, comparable to
    /// [`super::cache_key`]. Call it once per middleware pass, not per message.
    pub fn from_request(request: &ModelRequest) -> Self {
        let prefix_ids: Vec<String> = request.cacheable_prefix_ids();

        // Segment identity *and* role/cacheability, so a role flip or a
        // cacheable-flag flip on an otherwise identically named segment is not
        // mistaken for "unchanged".
        let mut material = String::new();
        for segment in &request.cache_segments {
            material.push_str(&segment.id);
            material.push('\u{1}');
            material.push_str(
                match serde_json::to_value(segment.role) {
                    Ok(Value::String(role)) => role,
                    _ => String::new(),
                }
                .as_str(),
            );
            material.push('\u{1}');
            material.push(if segment.cacheable { '1' } else { '0' });
            material.push('\u{2}');
        }
        // Content of the stable prefix, when the builder computed it.
        material.push_str(request.prompt_fingerprint.as_deref().unwrap_or(""));
        material.push('\u{2}');
        // Tool declarations sit inside the stable prefix on every provider that
        // caches prompts, so a schema edit invalidates it.
        material.push_str(&serde_json::to_string(&request.tools).unwrap_or_default());

        Self {
            prefix_ids,
            fingerprint: fnv1a_hex(material.as_bytes()),
            message_digests: request
                .messages
                .iter()
                .map(|message| {
                    fnv1a_hex(serde_json::to_vec(message).unwrap_or_default().as_slice())
                })
                .collect(),
        }
    }

    /// Returns the ordered ids of cacheable (stable) prefix segments.
    pub fn prefix_ids(&self) -> &[String] {
        &self.prefix_ids
    }

    /// Returns the deterministic content-aware fingerprint of the stable
    /// prefix (16 lowercase hex characters).
    ///
    /// Two layouts with the same segment identities *and* the same segment
    /// content, tools, and roles produce the same fingerprint.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Returns `true` when the provider's KV-cache prefix survives the move
    /// from `self` to `other`.
    ///
    /// That requires **both**:
    ///
    /// 1. the same cacheable segment ids in the same order **with the same
    ///    content** (equal [`Self::fingerprint`]), and
    /// 2. one message stream being a pure tail-extension of the other — the
    ///    only edit a byte-prefix cache tolerates.
    ///
    /// Comparing ids alone (the previous behaviour) reported stability after a
    /// middleware rewrote a stable segment's text, which is the precise failure
    /// this type exists to catch.
    pub fn is_prefix_stable_against(&self, other: &PromptCacheLayout) -> bool {
        if self.prefix_ids != other.prefix_ids || self.fingerprint != other.fingerprint {
            return false;
        }
        let (shorter, longer) = if self.message_digests.len() <= other.message_digests.len() {
            (&self.message_digests, &other.message_digests)
        } else {
            (&other.message_digests, &self.message_digests)
        };
        longer.starts_with(shorter.as_slice())
    }

    /// Returns `true` when the segment identities match but the material they
    /// carry does not — the silent invalidation an id-only comparison missed.
    pub fn is_content_only_change(&self, other: &PromptCacheLayout) -> bool {
        self.prefix_ids == other.prefix_ids && !self.is_prefix_stable_against(other)
    }
}

impl CacheLayoutEvent {
    /// Constructs a [`CacheLayoutEvent`] by comparing `before` and `after`
    /// layouts, filling in the computed flags automatically.
    ///
    /// `violates_policy` is always `false` here; use
    /// [`Self::under_policy`] to evaluate the change against a
    /// [`CachePolicy`].
    pub fn new(before: &PromptCacheLayout, after: &PromptCacheLayout) -> Self {
        Self {
            changed_prefix: !before.is_prefix_stable_against(after),
            volatile_only: after.prefix_ids().is_empty(),
            content_only_change: before.is_content_only_change(after),
            violates_policy: false,
            segment_ids_before: before.prefix_ids().to_vec(),
            segment_ids_after: after.prefix_ids().to_vec(),
        }
    }

    /// Evaluates the `before` -> `after` change against `policy`.
    ///
    /// Returns `None` when the prefix survived. When it did not, the returned
    /// event carries `violates_policy: true` iff
    /// [`CachePolicy::protect_prompt_prefix`] was in force — which is what
    /// makes that flag load-bearing instead of the inert struct field it was.
    pub fn under_policy(
        policy: &CachePolicy,
        before: &PromptCacheLayout,
        after: &PromptCacheLayout,
    ) -> Option<Self> {
        let mut event = Self::new(before, after);
        if !event.changed_prefix {
            return None;
        }
        event.violates_policy = policy.protect_prompt_prefix;
        if event.violates_policy {
            tracing::warn!(
                content_only_change = event.content_only_change,
                before = ?event.segment_ids_before,
                after = ?event.segment_ids_after,
                "[cache] prompt-cache prefix invalidated while protect_prompt_prefix was set"
            );
        }
        Some(event)
    }
}
