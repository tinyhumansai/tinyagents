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

#[test]
fn text_ignores_thinking_blocks() {
    let msg = Message::Assistant(AssistantMessage {
        id: None,
        content: vec![
            ContentBlock::thinking("let me reason about this"),
            ContentBlock::Text("the answer is 42".into()),
            ContentBlock::RedactedThinking {
                data: "opaque".into(),
            },
        ],
        tool_calls: Vec::new(),
        usage: None,
    });
    // Reasoning blocks must never leak into visible text.
    assert_eq!(msg.text(), "the answer is 42");
}

#[test]
fn thinking_accessors() {
    let signed = ContentBlock::Thinking {
        text: "reasoning".into(),
        signature: Some("sig-abc".into()),
    };
    assert_eq!(signed.as_thinking(), Some(("reasoning", Some("sig-abc"))));
    assert!(signed.is_reasoning());
    assert!(signed.as_text().is_none());

    let unsigned = ContentBlock::thinking("just thinking");
    assert_eq!(unsigned.as_thinking(), Some(("just thinking", None)));
    assert!(unsigned.is_reasoning());

    let redacted = ContentBlock::RedactedThinking {
        data: "opaque".into(),
    };
    assert!(redacted.is_reasoning());
    assert!(redacted.as_thinking().is_none());

    assert!(!ContentBlock::Text("hi".into()).is_reasoning());
}

#[test]
fn thinking_block_serde_round_trips() {
    let signed = ContentBlock::Thinking {
        text: "step by step".into(),
        signature: Some("sig-1".into()),
    };
    let wire = serde_json::to_value(&signed).unwrap();
    assert_eq!(
        wire,
        json!({ "thinking": { "text": "step by step", "signature": "sig-1" } })
    );
    let back: ContentBlock = serde_json::from_value(wire).unwrap();
    assert_eq!(back, signed);

    // An unsigned thinking block omits the signature key entirely.
    let unsigned = ContentBlock::thinking("no sig");
    let wire = serde_json::to_value(&unsigned).unwrap();
    assert_eq!(wire, json!({ "thinking": { "text": "no sig" } }));
    let back: ContentBlock = serde_json::from_value(wire).unwrap();
    assert_eq!(back, unsigned);

    let redacted = ContentBlock::RedactedThinking {
        data: "opaque".into(),
    };
    let wire = serde_json::to_value(&redacted).unwrap();
    assert_eq!(wire, json!({ "redacted_thinking": { "data": "opaque" } }));
    let back: ContentBlock = serde_json::from_value(wire).unwrap();
    assert_eq!(back, redacted);
}

#[test]
fn legacy_content_without_thinking_still_parses() {
    // Additive tagging: transcripts serialized before thinking blocks existed
    // (only text/json/image/provider_extension) must deserialize unchanged.
    let legacy = json!([
        { "text": "hello" },
        { "json": { "k": "v" } }
    ]);
    let blocks: Vec<ContentBlock> = serde_json::from_value(legacy).unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].as_text(), Some("hello"));
    assert!(!blocks[1].is_reasoning());
}
