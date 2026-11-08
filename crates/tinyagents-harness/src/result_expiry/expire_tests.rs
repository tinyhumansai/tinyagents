use serde_json::json;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, ToolMessage};
use tinyinference_llm::tool::ToolCall;

use super::*;
use crate::result_expiry::{ExpiryOverrides, ToolResultExpiryPolicy};

fn on(max_age_turns: usize) -> ToolResultExpiryPolicy {
    ToolResultExpiryPolicy {
        enabled: true,
        max_age_turns,
        exempt_tools: Vec::new(),
    }
}

fn call(id: &str, tool: &str) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(id, tool, json!({}))],
        usage: None,
        origin: None,
    })
}

/// One user turn that called `tool` and got `result` back.
fn turn(n: usize, tool: &str, result: &str) -> Vec<Message> {
    let id = format!("call-{n}");
    vec![
        Message::user(format!("question {n}")),
        call(&id, tool),
        Message::tool(id, result),
    ]
}

/// Four turns, oldest first: `result-1` .. `result-4`, the last user message
/// being the one the run is about to answer.
fn four_turns() -> Vec<Message> {
    let mut messages = Vec::new();
    for n in 1..=3 {
        messages.extend(turn(n, "lookup", &format!("result-{n}")));
    }
    messages.push(Message::user("question 4"));
    messages
}

fn tool_text(messages: &[Message], id: &str) -> String {
    messages
        .iter()
        .find_map(|m| match m {
            Message::Tool(t) if t.tool_call_id == id => Some(m.text()),
            _ => None,
        })
        .expect("tool result present")
}

#[test]
fn a_result_older_than_the_limit_becomes_a_note_and_newer_ones_stay() {
    // Ages by user messages after each result: call-1 -> 3, call-2 -> 2,
    // call-3 -> 1.
    let mut messages = four_turns();
    let expired = expire_tool_results(&mut messages, &on(2));
    assert_eq!(expired, 1);
    assert_eq!(tool_text(&messages, "call-1"), "[used lookup: result-1]");
    assert_eq!(tool_text(&messages, "call-2"), "result-2");
    assert_eq!(tool_text(&messages, "call-3"), "result-3");
}

#[test]
fn the_note_keeps_the_first_120_chars_on_one_line() {
    let long = format!("first line\nsecond   line {}", "x".repeat(300));
    let mut messages = turn(1, "lookup", &long);
    messages.push(Message::user("next"));
    expire_tool_results(&mut messages, &on(0));
    let note = tool_text(&messages, "call-1");
    let flat: String = "first line second line "
        .chars()
        .chain(std::iter::repeat_n(
            'x',
            120 - "first line second line ".len(),
        ))
        .collect();
    assert_eq!(note, format!("[used lookup: {flat} \u{2026}]"));
    assert!(!note.contains('\n'));
}

#[test]
fn a_result_with_no_matching_call_is_noted_under_a_generic_name() {
    let mut messages = vec![
        Message::user("a"),
        Message::tool("orphan", "body"),
        Message::user("b"),
    ];
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 1);
    assert_eq!(tool_text(&messages, "orphan"), "[used tool: body]");
}

#[test]
fn exempt_tools_and_trusted_verbatim_results_are_never_replaced() {
    let mut messages = turn(1, "tool_search", "schemas");
    messages.extend(turn(2, "lookup", "kept"));
    if let Message::Tool(t) = &mut messages[5] {
        t.trusted_verbatim = true;
    }
    messages.extend(turn(3, "lookup", "old"));
    messages.push(Message::user("now"));
    let policy = ToolResultExpiryPolicy {
        exempt_tools: vec!["tool_search".to_string()],
        ..on(0)
    };
    // Only the ordinary result expires; the exempt and verbatim ones stay.
    assert_eq!(expire_tool_results(&mut messages, &policy), 1);
    assert_eq!(tool_text(&messages, "call-1"), "schemas");
    assert_eq!(tool_text(&messages, "call-2"), "kept");
    assert_eq!(tool_text(&messages, "call-3"), "[used lookup: old]");
}

#[test]
fn a_disabled_policy_leaves_the_transcript_unchanged() {
    let mut messages = four_turns();
    let before = messages.clone();
    let off = ToolResultExpiryPolicy {
        max_age_turns: 0,
        ..ToolResultExpiryPolicy::default()
    };
    assert!(!off.enabled);
    assert_eq!(expire_tool_results(&mut messages, &off), 0);
    assert_eq!(messages, before);
}

#[test]
fn expiry_is_idempotent() {
    let mut messages = four_turns();
    assert_eq!(expire_tool_results(&mut messages, &on(1)), 2);
    let once = messages.clone();
    assert_eq!(expire_tool_results(&mut messages, &on(1)), 0);
    assert_eq!(messages, once);
}

#[test]
fn only_text_changes_the_artifact_and_call_id_are_kept() {
    let mut messages = four_turns();
    if let Message::Tool(t) = &mut messages[2] {
        t.artifact = Some(json!({"k": 1}));
    }
    expire_tool_results(&mut messages, &on(0));
    let Message::Tool(ToolMessage {
        tool_call_id,
        content,
        artifact,
        ..
    }) = &messages[2]
    else {
        panic!("tool message");
    };
    assert_eq!(tool_call_id, "call-1");
    assert_eq!(
        content,
        &vec![ContentBlock::Text("[used lookup: result-1]".to_string())]
    );
    assert_eq!(artifact, &Some(json!({"k": 1})));
}

#[test]
fn overrides_apply_per_field_and_default_to_empty() {
    let base = ToolResultExpiryPolicy::default();
    assert!(ExpiryOverrides::default().is_empty());
    assert_eq!(base.with_overrides(&ExpiryOverrides::default()), base);
    let set = base.with_overrides(&ExpiryOverrides {
        enabled: Some(true),
        max_age_turns: Some(2),
        exempt_tools: Some(vec!["tool_search".to_string()]),
    });
    assert!(set.enabled);
    assert_eq!(set.max_age_turns, 2);
    assert_eq!(set.exempt_tools, vec!["tool_search".to_string()]);
    let off = set.with_overrides(&ExpiryOverrides {
        enabled: Some(false),
        ..ExpiryOverrides::default()
    });
    assert!(!off.enabled);
    assert_eq!(off.max_age_turns, 2);
}

#[test]
fn a_repaired_wrapper_call_is_exempt_under_the_tool_it_ran() {
    let wrapped = Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(
            "w1",
            "tool_call",
            json!({"name": "lookup", "arguments": {}}),
        )],
        usage: None,
        origin: None,
    });
    let mut messages = vec![
        Message::user("q"),
        wrapped,
        Message::tool("w1", "kept body"),
        Message::user("now"),
    ];
    let policy = ToolResultExpiryPolicy {
        exempt_tools: vec!["lookup".to_string()],
        ..on(0)
    };
    assert_eq!(expire_tool_results(&mut messages, &policy), 0);
    assert_eq!(tool_text(&messages, "w1"), "kept body");
}

#[test]
fn duplicate_call_ids_are_paired_with_their_own_call_by_occurrence() {
    let both = Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![
            ToolCall::new("dup", "ordinary", json!({})),
            ToolCall::new("dup", "lookup", json!({})),
        ],
        usage: None,
        origin: None,
    });
    let mut messages = vec![
        Message::user("q"),
        both,
        Message::tool("dup", "first body"),
        Message::tool("dup", "second body"),
        Message::user("now"),
    ];
    let policy = ToolResultExpiryPolicy {
        exempt_tools: vec!["lookup".to_string()],
        ..on(0)
    };
    assert_eq!(expire_tool_results(&mut messages, &policy), 1);
    assert_eq!(messages[2].text(), "[used ordinary: first body]");
    assert_eq!(messages[3].text(), "second body");
}

#[test]
fn an_oversized_body_shaped_like_a_note_still_expires() {
    let body = format!("[used x: {}]", "y".repeat(5000));
    let mut messages = turn(1, "lookup", &body);
    messages.push(Message::user("now"));
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 1);
    assert!(tool_text(&messages, "call-1").chars().count() < 400);
}

#[test]
fn a_note_for_a_long_tool_name_is_recognised_and_not_rewritten_again() {
    let tool = "t".repeat(120);
    let mut messages = turn(1, &tool, &"x".repeat(500));
    messages.push(Message::user("now"));
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 1);
    let first = tool_text(&messages, "call-1");
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 0);
    assert_eq!(tool_text(&messages, "call-1"), first);
}

#[test]
fn a_tool_named_like_a_wrapper_is_exempt_under_its_own_name() {
    let wrapper = Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(
            "w1",
            "tool_call",
            json!({"name": "lookup", "arguments": {}}),
        )],
        usage: None,
        origin: None,
    });
    let mut messages = vec![
        Message::user("q"),
        wrapper,
        Message::tool("w1", "kept body"),
        Message::user("now"),
    ];
    let policy = ToolResultExpiryPolicy {
        exempt_tools: vec!["tool_call".to_string()],
        ..on(0)
    };
    assert_eq!(expire_tool_results(&mut messages, &policy), 0);
}

#[test]
fn a_tools_own_note_shaped_output_is_still_replaced_by_a_note_for_that_tool() {
    let mut messages = turn(1, "lookup", "[used other: cached]");
    messages.push(Message::user("now"));
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 1);
    assert_eq!(
        tool_text(&messages, "call-1"),
        "[used lookup: [used other: cached]]"
    );
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 0);
}

#[test]
fn a_long_wrapped_target_name_is_bounded_in_the_note() {
    let long = "n".repeat(400);
    let wrapped = Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(
            "w1",
            "tool_call",
            json!({"name": long, "arguments": {}}),
        )],
        usage: None,
        origin: None,
    });
    let mut messages = vec![
        Message::user("q"),
        wrapped,
        Message::tool("w1", "short"),
        Message::user("now"),
    ];
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 1);
    assert!(tool_text(&messages, "w1").chars().count() < 100);
    assert_eq!(expire_tool_results(&mut messages, &on(0)), 0);
}
