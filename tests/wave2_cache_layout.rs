//! Wave 2 — prompt-cache layout protection and provider breakpoints.
//!
//! Covers CACHE-6 (layout protection was inert and compared ids only) and
//! C-BREAKPOINT (the tooling only ever *observed* a prefix; it now injects a
//! provider `prompt_cache_key`).

use tinyagents::harness::cache::{
    CacheLayoutEvent, CachePolicy, PROMPT_CACHE_KEY_OPTION, PromptCacheLayout,
    apply_prompt_cache_breakpoints, prompt_cache_key,
};
use tinyagents::harness::message::Message;
use tinyagents::harness::model::{ModelRequest, PromptSegment, SegmentRole};
use tinyagents::harness::prompt::PromptBuilder;
use tinyagents::harness::tool::{ToolFormat, ToolSchema};

fn segment(id: &str, role: SegmentRole, cacheable: bool) -> PromptSegment {
    PromptSegment {
        id: id.to_string(),
        role,
        cacheable,
    }
}

/// Builds a request through [`PromptBuilder`] so `prompt_fingerprint` — the
/// content digest the layout now consults — is actually populated.
fn built_with_system(system: &str, question: &str) -> ModelRequest {
    let mut builder = PromptBuilder::new();
    builder.push_system("sys", vec![Message::system(system)]);
    builder.build(vec![Message::user(question)])
}

fn tool(name: &str, description: &str) -> ToolSchema {
    ToolSchema {
        name: name.to_string(),
        description: description.to_string(),
        parameters: serde_json::json!({ "type": "object", "properties": {} }),
        format: ToolFormat::Json,
    }
}

// ── CACHE-6 ──────────────────────────────────────────────────────────────────

#[test]
fn editing_a_stable_segments_text_is_reported_as_a_prefix_change() {
    // This is the exact failure the module exists to catch: the segment *ids*
    // are unchanged, so an id-only comparison reported "prefix stable" while
    // the provider's KV prefix was already destroyed. The content digest comes
    // from `PromptBuilder::fingerprint`, which hashes the segments' messages
    // and was previously ignored by the layout entirely.
    let before_request = built_with_system("You are a careful assistant.", "q");
    let after_request = built_with_system("You are a RECKLESS assistant.", "q");

    let before = PromptCacheLayout::from_request(&before_request);
    let after = PromptCacheLayout::from_request(&after_request);

    assert_eq!(
        before.prefix_ids(),
        after.prefix_ids(),
        "the ids are deliberately identical — that is the trap"
    );
    assert!(
        !before.is_prefix_stable_against(&after),
        "rewriting a stable segment's TEXT must not report a stable prefix"
    );
    assert!(
        before.is_content_only_change(&after),
        "the change is content-only: same ids, different bytes"
    );
    assert_ne!(before.fingerprint(), after.fingerprint());
    assert_eq!(before.fingerprint().len(), 16);
}

#[test]
fn editing_a_tool_schema_invalidates_the_prefix() {
    // Tool declarations sit inside the stable prefix on every provider that
    // caches prompts.
    let base = |description: &str| {
        ModelRequest::new(vec![Message::user("q")])
            .with_cache_segments(vec![segment("sys", SegmentRole::System, true)])
            .with_tools(vec![tool("search", description)])
    };
    let before = PromptCacheLayout::from_request(&base("Search the web."));
    let after = PromptCacheLayout::from_request(&base("Search the web, carefully."));

    assert_eq!(before.prefix_ids(), after.prefix_ids());
    assert!(
        !before.is_prefix_stable_against(&after),
        "a tool-schema edit must invalidate the prefix"
    );
}

#[test]
fn appending_to_the_tail_keeps_the_prefix_stable() {
    // A provider prompt cache is a byte-prefix cache: appending is the one edit
    // it tolerates, and the common multi-turn case must not be flagged.
    let segments = vec![
        segment("sys", SegmentRole::System, true),
        segment("turn", SegmentRole::Volatile, false),
    ];
    let before = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![Message::user("q1")]).with_cache_segments(segments.clone()),
    );
    let after = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q2"),
        ])
        .with_cache_segments(segments),
    );

    assert!(
        before.is_prefix_stable_against(&after),
        "appending turns must not be reported as a prefix invalidation"
    );
    assert!(CacheLayoutEvent::new(&before, &after).changed_prefix == false);
}

#[test]
fn rewriting_history_mid_stream_invalidates_the_prefix() {
    // A summarizer that compacts history rewrites bytes the provider already
    // cached. Ids are unchanged, so only content awareness catches it.
    let segments = vec![segment("sys", SegmentRole::System, true)];
    let before = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![Message::user("q1"), Message::assistant("a1")])
            .with_cache_segments(segments.clone()),
    );
    let after = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![
            Message::user("[summary of earlier turns]"),
            Message::assistant("a1"),
        ])
        .with_cache_segments(segments),
    );
    assert!(!before.is_prefix_stable_against(&after));
}

#[test]
fn protect_prompt_prefix_is_load_bearing_for_the_layout_event() {
    // The flag had no reader anywhere in the crate: only struct literals and
    // one assertion. It now decides whether a detected invalidation counts as a
    // policy violation.
    let before = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![Message::user("q")]).with_cache_segments(vec![segment(
            "sys",
            SegmentRole::System,
            true,
        )]),
    );
    let after = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![Message::user("q")]).with_cache_segments(vec![segment(
            "turn",
            SegmentRole::Volatile,
            false,
        )]),
    );

    let unprotected = CacheLayoutEvent::under_policy(&CachePolicy::default(), &before, &after)
        .expect("the prefix did change");
    assert!(unprotected.changed_prefix);
    assert!(
        !unprotected.violates_policy,
        "without protection a change is reported but is not a violation"
    );

    let protected = CachePolicy {
        protect_prompt_prefix: true,
        ..CachePolicy::default()
    };
    let violation =
        CacheLayoutEvent::under_policy(&protected, &before, &after).expect("the prefix did change");
    assert!(violation.violates_policy, "the flag must be load-bearing");
    assert!(violation.volatile_only);

    // No change, no event, under either policy.
    assert!(CacheLayoutEvent::under_policy(&protected, &before, &before).is_none());
}

// ── C-BREAKPOINT ─────────────────────────────────────────────────────────────

#[test]
fn a_prompt_cache_key_is_derived_from_the_stable_prefix() {
    let request = |question: &str| built_with_system("You are a careful assistant.", question);

    let first = prompt_cache_key(&request("q1")).expect("a stable prefix exists");
    let second = prompt_cache_key(&request("q2")).expect("a stable prefix exists");
    assert_eq!(
        first, second,
        "every turn of one logical thread must route to the same provider cache shard"
    );

    let other = built_with_system("You are a different assistant.", "q1");
    assert_ne!(
        prompt_cache_key(&other).expect("a stable prefix exists"),
        first,
        "a different stable prefix must route to a different shard"
    );

    // No declared prefix, nothing to route.
    assert!(prompt_cache_key(&ModelRequest::new(vec![Message::user("q")])).is_none());
}

#[test]
fn breakpoints_are_injected_only_under_the_protection_policy() {
    let build = |policy: Option<CachePolicy>| {
        let mut request = ModelRequest::new(vec![Message::user("q")])
            .with_cache_segments(vec![segment("sys", SegmentRole::System, true)]);
        request.cache_policy = policy;
        request
    };

    // Off by default: no policy, no injection.
    let mut none = build(None);
    assert!(!apply_prompt_cache_breakpoints(&mut none));
    assert!(none.provider_options.get(PROMPT_CACHE_KEY_OPTION).is_none());

    // On: a routing key is written into `provider_options`.
    let protected = CachePolicy {
        protect_prompt_prefix: true,
        ..CachePolicy::default()
    };
    let mut on = build(Some(protected.clone()));
    assert!(apply_prompt_cache_breakpoints(&mut on));
    let injected = on
        .provider_options
        .get(PROMPT_CACHE_KEY_OPTION)
        .and_then(|v| v.as_str())
        .expect("a prompt_cache_key was injected");
    assert!(injected.starts_with("tap-"));

    // A caller who already set the key wins, matching the rest of the crate's
    // provider-options precedence.
    let mut explicit = build(Some(protected));
    explicit.provider_options = serde_json::json!({ PROMPT_CACHE_KEY_OPTION: "mine" });
    assert!(!apply_prompt_cache_breakpoints(&mut explicit));
    assert_eq!(
        explicit.provider_options[PROMPT_CACHE_KEY_OPTION],
        serde_json::json!("mine")
    );
}
