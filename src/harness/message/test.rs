//! Unit tests for the rich internal message model.
//!
//! Covers the ergonomic [`Message`] constructors and the [`Message::text`]
//! accessor (including that it concatenates text blocks and ignores non-text
//! content), that assistant messages carry tool calls and usage, that tool
//! messages preserve their call id, and the [`MessageDelta`] default.

use super::*;
use crate::harness::tool::ToolCall;
use crate::harness::usage::Usage;
use serde_json::json;

#[test]
fn constructors_and_text_accessor() {
    assert_eq!(Message::system("sys").text(), "sys");
    assert_eq!(Message::user("hi").text(), "hi");
    assert_eq!(Message::assistant("yo").text(), "yo");
    assert_eq!(Message::tool("c-1", "result").text(), "result");
}

#[test]
fn text_concatenates_and_ignores_non_text() {
    let msg = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("a".into()),
            ContentBlock::Json(json!({"k": "v"})),
            ContentBlock::Text("b".into()),
        ],
    });
    assert_eq!(msg.text(), "ab");
}

#[test]
fn char_len_matches_text_char_count_including_multibyte() {
    // Mixed text blocks with multi-byte scalar values; non-text blocks ignored.
    let msg = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("héllo".into()),
            ContentBlock::Json(json!({"k": "v"})),
            ContentBlock::Text("🌍!".into()),
        ],
    });
    // char_len counts Unicode scalar values, matching text().chars().count()
    // without allocating the joined string.
    assert_eq!(msg.char_len(), msg.text().chars().count());
    assert_eq!(
        msg.char_len(),
        "héllo".chars().count() + "🌍!".chars().count()
    );
}

#[test]
fn assistant_holds_tool_calls_and_usage() {
    let msg = Message::Assistant(AssistantMessage {
        id: Some("m-1".into()),
        content: vec![ContentBlock::Text("calling".into())],
        tool_calls: vec![ToolCall::new("c-1", "lookup", json!({}))],
        usage: Some(Usage::new(5, 5)),
    });
    if let Message::Assistant(a) = &msg {
        assert_eq!(a.tool_calls.len(), 1);
        assert_eq!(a.usage.unwrap().total_tokens, 10);
    } else {
        panic!("expected assistant");
    }
}

#[test]
fn tool_message_carries_call_id() {
    let msg = Message::tool("c-7", "done");
    match &msg {
        Message::Tool(t) => {
            assert_eq!(t.tool_call_id, "c-7");
            assert_eq!(msg.text(), "done");
        }
        _ => panic!("expected tool message"),
    }
}

#[test]
fn message_delta_default() {
    let delta = MessageDelta::default();
    assert_eq!(delta.text, "");
    assert!(delta.tool_call.is_none());
}
