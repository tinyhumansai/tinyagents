//! Tests added in a later pass.
//!
//! Smoke tests confirming that [`super::InMemoryResponseCache`] round-trips a
//! response, that [`super::cache_key`] is deterministic, and that
//! [`super::PromptCacheLayout`] correctly detects stable vs. changed prefixes.

use super::*;
use crate::harness::model::{ModelRequest, ModelResponse, PromptSegment, SegmentRole};

#[tokio::test]
async fn response_cache_put_get() {
    let cache = InMemoryResponseCache::new();
    assert!(cache.get("k").await.unwrap().is_none());

    let response = ModelResponse::assistant("hello");
    cache.put("k", response.clone()).await.unwrap();
    let fetched = cache.get("k").await.unwrap().expect("should be cached");
    assert_eq!(fetched.text(), "hello");
}

#[test]
fn cache_key_is_deterministic() {
    let req = ModelRequest::new(vec![]).with_model("gpt-4");
    let k1 = cache_key(&req);
    let k2 = cache_key(&req);
    assert_eq!(k1, k2, "cache_key must be deterministic");
    assert_eq!(k1.len(), 16, "cache_key should be 16 hex chars");
}

#[test]
fn cache_key_differs_for_different_requests() {
    let r1 = ModelRequest::new(vec![]).with_model("gpt-4");
    let r2 = ModelRequest::new(vec![]).with_model("claude-3");
    assert_ne!(cache_key(&r1), cache_key(&r2));
}

#[test]
fn prompt_cache_layout_stable_prefix() {
    let req = ModelRequest::new(vec![]).with_cache_segments(vec![
        PromptSegment {
            id: "sys".into(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "tail".into(),
            role: SegmentRole::Volatile,
            cacheable: false,
        },
    ]);
    let layout = PromptCacheLayout::from_request(&req);
    assert_eq!(layout.prefix_ids(), &["sys"]);
    assert_eq!(layout.fingerprint().len(), 16);

    let same = PromptCacheLayout::from_request(&req);
    assert!(layout.is_prefix_stable_against(&same));
}

#[test]
fn prompt_cache_layout_detects_changed_prefix() {
    let req_a = ModelRequest::new(vec![]).with_cache_segments(vec![PromptSegment {
        id: "sys".into(),
        role: SegmentRole::System,
        cacheable: true,
    }]);
    let req_b = ModelRequest::new(vec![]).with_cache_segments(vec![PromptSegment {
        id: "sys-v2".into(),
        role: SegmentRole::System,
        cacheable: true,
    }]);

    let before = PromptCacheLayout::from_request(&req_a);
    let after = PromptCacheLayout::from_request(&req_b);

    assert!(!before.is_prefix_stable_against(&after));

    let event = CacheLayoutEvent::new(&before, &after);
    assert!(event.changed_prefix);
    assert!(!event.volatile_only);
    assert_eq!(event.segment_ids_before, vec!["sys"]);
    assert_eq!(event.segment_ids_after, vec!["sys-v2"]);
}

#[test]
fn cache_layout_event_volatile_only_when_no_cacheable_segments() {
    let req_a = ModelRequest::new(vec![]).with_cache_segments(vec![PromptSegment {
        id: "sys".into(),
        role: SegmentRole::System,
        cacheable: true,
    }]);
    let req_b = ModelRequest::new(vec![]).with_cache_segments(vec![PromptSegment {
        id: "user".into(),
        role: SegmentRole::Volatile,
        cacheable: false,
    }]);

    let before = PromptCacheLayout::from_request(&req_a);
    let after = PromptCacheLayout::from_request(&req_b);
    let event = CacheLayoutEvent::new(&before, &after);

    assert!(event.changed_prefix);
    assert!(event.volatile_only, "all segments are volatile after the change");
}
