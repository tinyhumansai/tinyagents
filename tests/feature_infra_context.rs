//! Feature/integration tests for the harness context-management infrastructure
//! (`harness::summarization` + `harness::cache`).
//!
//! Covers the anti-context-rot machinery: token estimation, trimming strategies
//! (system-message preservation), the deterministic `ConcatSummarizer` with
//! provenance, context-window-aware summarization gating, and the two caching
//! concerns — the deterministic response `cache_key`, the LRU
//! `InMemoryResponseCache`, and provider prompt-cache layout protection.
//!
//! Deterministic and offline: `ConcatSummarizer` makes no LLM call and the mock
//! response cache is in-process.

use tinyagents::harness::cache::{
    CacheLayoutEvent, CachePolicy, InMemoryResponseCache, PromptCacheLayout, ResponseCache,
    cache_key,
};
use tinyagents::harness::message::Message;
use tinyagents::harness::model::{ModelRequest, ModelResponse, PromptSegment, SegmentRole};
use tinyagents::harness::summarization::{
    ConcatSummarizer, SummarizationPolicy, Summarizer, TrimStrategy, estimate_tokens, trim_messages,
};

// ── Summarization: token estimation + trimming ──────────────────────────────

#[test]
fn estimate_tokens_uses_chars_over_four_heuristic() {
    assert_eq!(estimate_tokens(""), 0);
    // Any non-empty string is at least one token.
    assert_eq!(estimate_tokens("hi"), 1);
    assert_eq!(estimate_tokens(&"x".repeat(40)), 10);
}

#[test]
fn keep_last_retains_recent_non_system_and_all_system() {
    let msgs = vec![
        Message::system("rules"),
        Message::user("m0"),
        Message::assistant("m1"),
        Message::user("m2"),
    ];
    let trimmed = trim_messages(&msgs, &TrimStrategy::KeepLast(1));
    // System message survives + the single most-recent non-system message.
    assert_eq!(trimmed.len(), 2);
    assert_eq!(trimmed[0].text(), "rules");
    assert_eq!(trimmed[1].text(), "m2");
}

#[test]
fn keep_first_and_last_drops_the_middle() {
    let msgs = vec![
        Message::user("a"),
        Message::user("b"),
        Message::user("c"),
        Message::user("d"),
    ];
    let trimmed = trim_messages(&msgs, &TrimStrategy::KeepFirstAndLast { first: 1, last: 1 });
    assert_eq!(trimmed.len(), 2);
    assert_eq!(trimmed[0].text(), "a");
    assert_eq!(trimmed[1].text(), "d");
}

#[test]
fn max_tokens_drops_from_front_and_sheds_system_only_as_last_resort() {
    // Each message is ~5 tokens (20 chars / 4). A generous budget keeps the
    // system message plus the most recent non-system messages.
    let msgs = vec![
        Message::system("s".repeat(20)),
        Message::user("a".repeat(20)),
        Message::user("b".repeat(20)),
    ];
    let trimmed = trim_messages(&msgs, &TrimStrategy::MaxTokens(10));
    // System (5) is preserved; one non-system (5) fits under 10.
    assert!(trimmed.iter().any(|m| matches!(m, Message::System(_))));
    assert_eq!(trimmed.last().unwrap().text(), "b".repeat(20));
}

// ── Summarization: ConcatSummarizer ─────────────────────────────────────────

#[tokio::test]
async fn concat_summarizer_folds_messages_with_provenance() {
    let msgs = vec![Message::user("what is 2+2"), Message::assistant("4")];
    let record = ConcatSummarizer.summarize(&msgs).await.unwrap();

    // The summary is a single system message referencing both source turns.
    assert!(matches!(record.summary, Message::System(_)));
    assert!(record.summary.text().contains("2+2"));
    assert!(record.summary.text().contains("4"));

    // Provenance records synthetic positional ids and token estimates.
    assert_eq!(record.provenance.source_ids, vec!["msg-0", "msg-1"]);
    assert!(record.provenance.original_token_estimate > 0);
    assert!(record.provenance.reason.contains("ConcatSummarizer"));
}

#[tokio::test]
async fn concat_summarizer_rejects_empty_input() {
    assert!(ConcatSummarizer.summarize(&[]).await.is_err());
}

// ── Summarization: policy gating ────────────────────────────────────────────

#[test]
fn policy_triggers_on_raw_trigger_tokens_without_a_window() {
    let policy = SummarizationPolicy {
        trigger_tokens: 5,
        keep_last: 1,
        ..Default::default()
    };
    let small = vec![Message::user("hi")]; // ~1 token
    assert!(!policy.should_summarize(&small));

    let big = vec![Message::user("x".repeat(40))]; // ~10 tokens > 5
    assert!(policy.should_summarize(&big));
    assert_eq!(policy.trigger_budget(), 5);
}

#[test]
fn context_window_policy_triggers_at_threshold_fraction() {
    // Window of 40 tokens at 0.5 → budget 20 tokens.
    let policy = SummarizationPolicy::default()
        .with_context_window(40)
        .with_threshold_fraction(0.5);
    assert_eq!(policy.trigger_budget(), 20);

    // 40 chars ≈ 10 tokens: under the 20-token budget.
    let under = vec![Message::user("a".repeat(40))];
    assert!(!policy.should_summarize(&under));

    // 120 chars ≈ 30 tokens: at/above the budget.
    let over = vec![Message::user("a".repeat(120))];
    assert!(policy.should_summarize(&over));
}

#[test]
fn policy_plan_keeps_system_and_recent_summarizes_the_rest() {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..Default::default()
    };
    let msgs = vec![
        Message::system("rules"),
        Message::user("old-0"),
        Message::user("old-1"),
        Message::user("recent"),
    ];
    let (to_summarize, to_keep) = policy.plan(&msgs);

    // The two oldest non-system messages are summarized.
    assert_eq!(to_summarize.len(), 2);
    assert_eq!(to_summarize[0].text(), "old-0");
    // System message is never summarized; the recent one is kept verbatim.
    assert!(to_keep.iter().any(|m| matches!(m, Message::System(_))));
    assert_eq!(to_keep.last().unwrap().text(), "recent");
}

// ── Cache: deterministic cache_key ──────────────────────────────────────────

fn request(prompt: &str) -> ModelRequest {
    ModelRequest::new(vec![Message::user(prompt)]).with_model("test-model")
}

#[test]
fn cache_key_is_deterministic_and_input_sensitive() {
    let k1 = cache_key(&request("hello"));
    let k2 = cache_key(&request("hello"));
    let k3 = cache_key(&request("goodbye"));

    // Identical requests hash to the same 64-char hex key.
    assert_eq!(k1, k2);
    assert_eq!(k1.len(), 64);
    assert!(k1.chars().all(|c| c.is_ascii_hexdigit()));
    // A different prompt yields a different key.
    assert_ne!(k1, k3);
}

#[test]
fn cache_key_reflects_parameter_changes() {
    let base = request("hello");
    let hotter = request("hello").with_temperature(0.9);
    // A behavior-affecting parameter change must change the key.
    assert_ne!(cache_key(&base), cache_key(&hotter));
}

// ── Cache: InMemoryResponseCache (LRU) ──────────────────────────────────────

#[tokio::test]
async fn response_cache_stores_and_returns_on_hit() {
    let cache = InMemoryResponseCache::new();
    let key = cache_key(&request("q"));
    assert!(cache.get(&key).await.unwrap().is_none());

    cache
        .put(&key, ModelResponse::assistant("cached answer"))
        .await
        .unwrap();
    let hit = cache.get(&key).await.unwrap().expect("cache hit");
    assert_eq!(hit.message.content.len(), 1);
}

#[tokio::test]
async fn response_cache_evicts_least_recently_used() {
    let cache = InMemoryResponseCache::with_capacity(2);
    cache.put("a", ModelResponse::assistant("A")).await.unwrap();
    cache.put("b", ModelResponse::assistant("B")).await.unwrap();

    // Touch "a" so it becomes most-recently-used; "b" is now the LRU victim.
    assert!(cache.get("a").await.unwrap().is_some());
    cache.put("c", ModelResponse::assistant("C")).await.unwrap();

    assert!(cache.get("b").await.unwrap().is_none()); // evicted
    assert!(cache.get("a").await.unwrap().is_some());
    assert!(cache.get("c").await.unwrap().is_some());
}

#[tokio::test]
async fn response_cache_zero_capacity_retains_last_write() {
    // A capacity of 0 is clamped to 1, so the most recent write survives.
    let cache = InMemoryResponseCache::with_capacity(0);
    cache.put("x", ModelResponse::assistant("X")).await.unwrap();
    assert!(cache.get("x").await.unwrap().is_some());
}

// ── Cache: prompt-cache layout protection ───────────────────────────────────

fn segment(id: &str, role: SegmentRole, cacheable: bool) -> PromptSegment {
    PromptSegment {
        id: id.to_string(),
        role,
        cacheable,
    }
}

#[test]
fn prompt_cache_layout_captures_stable_prefix_and_fingerprint() {
    let req = ModelRequest::new(vec![Message::user("q")]).with_cache_segments(vec![
        segment("sys", SegmentRole::System, true),
        segment("tools", SegmentRole::Tools, true),
        segment("turn", SegmentRole::Volatile, false),
    ]);
    let layout = PromptCacheLayout::from_request(&req);
    // Only cacheable segments form the protected prefix, in declared order.
    assert_eq!(
        layout.prefix_ids(),
        &["sys".to_string(), "tools".to_string()]
    );
    assert!(!layout.fingerprint().is_empty());

    // An identical prefix yields an identical fingerprint and is "stable".
    let same = PromptCacheLayout::from_request(&req);
    assert_eq!(layout.fingerprint(), same.fingerprint());
    assert!(layout.is_prefix_stable_against(&same));
}

#[test]
fn cache_layout_event_flags_prefix_invalidation() {
    let before = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![Message::user("q")]).with_cache_segments(vec![segment(
            "sys",
            SegmentRole::System,
            true,
        )]),
    );
    // A middleware pass that reordered/removed the cacheable prefix.
    let after = PromptCacheLayout::from_request(
        &ModelRequest::new(vec![Message::user("q")]).with_cache_segments(vec![segment(
            "turn",
            SegmentRole::Volatile,
            false,
        )]),
    );

    let event = CacheLayoutEvent::new(&before, &after);
    assert!(event.changed_prefix);
    // The "after" layout has no stable prefix left.
    assert!(event.volatile_only);
    assert_eq!(event.segment_ids_before, vec!["sys".to_string()]);
    assert!(event.segment_ids_after.is_empty());
}

#[test]
fn cache_policy_defaults_to_safe_off() {
    // Both caching concerns are opt-in.
    let policy = CachePolicy::default();
    assert!(!policy.response_cache_enabled);
    assert!(!policy.protect_prompt_prefix);
}
