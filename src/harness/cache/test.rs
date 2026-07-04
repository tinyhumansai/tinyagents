//! Tests added in a later pass.
//!
//! Smoke tests confirming that [`super::InMemoryResponseCache`] round-trips a
//! response, that [`super::cache_key`] is deterministic, and that
//! [`super::PromptCacheLayout`] correctly detects stable vs. changed prefixes.

use super::*;
use crate::harness::message::Message;
use crate::harness::model::{ModelRequest, ModelResponse, PromptSegment, SegmentRole};
use crate::harness::tool::ToolSchema;
use serde_json::json;

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
    assert_eq!(k1.len(), 64, "cache_key should be a SHA-256 hex digest");
    assert!(k1.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn cache_key_differs_for_different_requests() {
    let r1 = ModelRequest::new(vec![]).with_model("gpt-4");
    let r2 = ModelRequest::new(vec![]).with_model("claude-3");
    assert_ne!(cache_key(&r1), cache_key(&r2));
}

#[test]
fn cache_key_is_deterministic_with_messages_and_tools() {
    let build = || {
        ModelRequest::new(vec![
            Message::system("you are terse"),
            Message::user("hello"),
        ])
        .with_model("gpt-4")
        .with_tools(vec![ToolSchema::new(
            "spin",
            "spin a value",
            json!({"type": "object"}),
        )])
    };
    assert_eq!(cache_key(&build()), cache_key(&build()));
}

#[test]
fn cache_key_reflects_message_content() {
    let r1 = ModelRequest::new(vec![Message::user("hello")]);
    let r2 = ModelRequest::new(vec![Message::user("goodbye")]);
    assert_ne!(
        cache_key(&r1),
        cache_key(&r2),
        "a changed message body must change the key"
    );
}

#[test]
fn cache_key_reflects_message_count() {
    let r1 = ModelRequest::new(vec![Message::user("hello")]);
    let r2 = ModelRequest::new(vec![Message::user("hello"), Message::user("again")]);
    assert_ne!(
        cache_key(&r1),
        cache_key(&r2),
        "appending a message must change the key"
    );
}

#[test]
fn cache_key_is_order_sensitive() {
    let r1 = ModelRequest::new(vec![Message::user("a"), Message::user("b")]);
    let r2 = ModelRequest::new(vec![Message::user("b"), Message::user("a")]);
    assert_ne!(
        cache_key(&r1),
        cache_key(&r2),
        "message order must change the key (length-prefixed frames)"
    );
}

#[test]
fn cache_key_reflects_tool_schemas() {
    let base = ModelRequest::new(vec![Message::user("hi")]);
    let with_tool = base.clone().with_tools(vec![ToolSchema::new(
        "spin",
        "spin a value",
        json!({"type": "object"}),
    )]);
    assert_ne!(
        cache_key(&base),
        cache_key(&with_tool),
        "adding a tool schema must change the key"
    );
}

#[test]
fn cache_key_reflects_scalar_envelope_fields() {
    let base = ModelRequest::new(vec![Message::user("hi")]);
    let mut hot = base.clone();
    hot.temperature = Some(0.9);
    assert_ne!(
        cache_key(&base),
        cache_key(&hot),
        "a scalar field change must change the key via the envelope frame"
    );
}

#[test]
fn cache_key_does_not_confuse_messages_with_tools() {
    // A message array of length 1 and a tool array of length 1 are folded under
    // distinct tags with count frames, so a request carrying one must not
    // collide with an otherwise-empty request carrying the other.
    let one_message = ModelRequest::new(vec![Message::user("x")]);
    let one_tool = ModelRequest::new(vec![]).with_tools(vec![ToolSchema::new(
        "x",
        "x",
        json!({"type": "object"}),
    )]);
    assert_ne!(cache_key(&one_message), cache_key(&one_tool));
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
    // Prompt-layout fingerprints are short local stability markers; response
    // cache identity uses the stronger `cache_key` digest.
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
    assert!(
        event.volatile_only,
        "all segments are volatile after the change"
    );
}
