//! Tests for the progressive-disclosure handoff cache.
//!
//! The module had none where it came from — it was covered only indirectly,
//! through the sub-agent runner's integration tests. These pin the behaviour
//! that a caller actually depends on, and in particular the cases where a
//! regression would be silent rather than loud.

use super::*;

fn cache() -> ResultHandoffCache {
    ResultHandoffCache::new()
}

/// A payload comfortably over any threshold used here.
fn big(n: usize) -> String {
    "x".repeat(n)
}

// ── The cache ─────────────────────────────────────────────────────────────────

#[test]
fn a_stored_payload_round_trips_by_id() {
    let c = cache();
    let id = c.store("gmail_list".to_string(), "payload".to_string());
    let got = c.get(&id).expect("stored payload is retrievable");
    assert_eq!(got.content, "payload");
    assert_eq!(got.tool_name, "gmail_list");
}

#[test]
fn ids_are_unique_across_stores() {
    // Ids are handed to a model and used as lookup keys. A collision would
    // silently serve one tool's payload in answer to another's query.
    let c = cache();
    let ids: Vec<String> = (0..5)
        .map(|i| c.store(format!("tool{i}"), format!("body{i}")))
        .collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "duplicate result ids: {ids:?}");
}

#[test]
fn an_unknown_id_is_none_rather_than_a_panic() {
    // The id comes back through a model, so it can be anything at all.
    assert!(cache().get("res_deadbeef").is_none());
    assert!(cache().get("").is_none());
}

#[test]
fn eviction_is_fifo_and_bounded() {
    // The cache is per-spawn and unbounded growth is a leak, but evicting the
    // WRONG end would drop the payload a model is most likely to ask about.
    let c = cache();
    let ids: Vec<String> = (0..HANDOFF_MAX_ENTRIES + 2)
        .map(|i| c.store("t".to_string(), format!("body{i}")))
        .collect();

    assert!(c.get(&ids[0]).is_none(), "oldest entry should be evicted");
    assert!(c.get(&ids[1]).is_none(), "second-oldest should be evicted");
    let newest = ids.last().expect("at least one id");
    assert!(c.get(newest).is_some(), "newest entry must survive");
}

// ── apply_handoff ─────────────────────────────────────────────────────────────

#[test]
fn a_small_result_passes_through_untouched_and_is_not_cached() {
    let c = cache();
    let out = apply_handoff(&c, "search", "task-1", "agent-1", "small".to_string(), 10);
    assert_eq!(out, "small");
    assert!(
        c.get("res_1").is_none(),
        "nothing should have been stashed for a small result"
    );
}

#[test]
fn an_oversized_result_is_stashed_and_replaced_by_a_placeholder() {
    let c = cache();
    let raw = big(4_000); // ~1000 tokens at the 4-chars/token heuristic
    let out = apply_handoff(&c, "gmail_list", "task-1", "agent-1", raw.clone(), 10);

    assert_ne!(out, raw, "the raw payload must not reach history");
    assert!(out.contains("oversized tool output"));
    assert!(out.contains("extract_from_result"));

    // The placeholder must name an id that actually resolves, or the model is
    // told to call a tool that cannot find anything.
    let id = out
        .split("result_id=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("placeholder carries a result_id");
    assert_eq!(
        c.get(id).expect("the advertised id resolves").content,
        raw,
        "full fidelity must be preserved in the cache"
    );
}

#[test]
fn the_threshold_is_honoured_in_both_directions() {
    // Same payload, two thresholds: this is the parameter that replaced the
    // env-var backdoor, so it has to actually decide the outcome.
    let raw = big(400); // ~100 tokens
    let below = apply_handoff(&cache(), "t", "task", "agent", raw.clone(), 10);
    let above = apply_handoff(&cache(), "t", "task", "agent", raw.clone(), 10_000);
    assert!(below.contains("oversized tool output"));
    assert_eq!(above, raw);
}

#[test]
fn an_error_result_passes_through_however_large() {
    // Errors are diagnostic text the agent must read directly. Stashing one
    // behind an extraction call would hide the failure it needs to react to.
    let c = cache();
    let err = format!("Error: {}", big(8_000));
    let out = apply_handoff(&c, "gmail_list", "task", "agent", err.clone(), 1);
    assert_eq!(out, err);
}

#[test]
fn an_extraction_result_is_never_re_stashed() {
    // The extractor answers a query against a stashed payload. Handing its
    // answer back through the same path would stash the answer and hand the
    // model another placeholder — a loop that never converges.
    let c = cache();
    let raw = big(8_000);
    let out = apply_handoff(&c, "extract_from_result", "task", "agent", raw.clone(), 1);
    assert_eq!(out, raw);
}

// ── The placeholder ───────────────────────────────────────────────────────────

#[test]
fn the_placeholder_reports_size_and_previews_the_head() {
    let raw = format!("HEAD-MARKER{}", big(5_000));
    let text = build_handoff_placeholder("gmail_list", "res_1", &raw);

    assert!(text.contains("res_1"));
    assert!(text.contains("gmail_list"));
    assert!(
        text.contains(&raw.len().to_string()),
        "raw byte count shown"
    );
    assert!(
        text.contains("HEAD-MARKER"),
        "the preview must come from the start of the payload"
    );
    assert!(
        text.len() < raw.len(),
        "a placeholder larger than the payload defeats the purpose"
    );
}

#[test]
fn a_short_payload_previews_whole_without_padding() {
    let text = build_handoff_placeholder("t", "res_1", "tiny");
    assert!(text.contains("tiny"));
    // The preview length is reported to the model; claiming the full budget
    // for a 4-char payload would misdescribe what it is looking at.
    assert!(text.contains("first 4 chars"));
}

#[test]
fn the_preview_is_capped_for_a_large_payload() {
    let raw = big(HANDOFF_PREVIEW_CHARS * 4);
    let text = build_handoff_placeholder("t", "res_1", &raw);
    assert!(text.contains(&format!("first {HANDOFF_PREVIEW_CHARS} chars")));
}

#[test]
fn the_preview_never_splits_a_multibyte_character() {
    // Taken by chars, not bytes — a byte-indexed cut here would panic rather
    // than merely misformat.
    let raw = "é".repeat(HANDOFF_PREVIEW_CHARS * 2);
    let text = build_handoff_placeholder("t", "res_1", &raw);
    assert!(text.is_char_boundary(text.len()));
}
