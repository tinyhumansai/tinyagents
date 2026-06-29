use super::*;
use crate::chat::{ChatMessage, ChatRole};
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
fn bridges_from_chat_message() {
    let chat = ChatMessage::tool("c-7", "done");
    let msg: Message = chat.into();
    match &msg {
        Message::Tool(t) => assert_eq!(t.tool_call_id, "c-7"),
        _ => panic!("expected tool message"),
    }
}

#[test]
fn bridges_to_chat_message() {
    let chat = Message::system("rules").to_chat();
    assert_eq!(chat.role, ChatRole::System);
    assert_eq!(chat.content, "rules");

    let tool_chat = Message::tool("c-2", "out").to_chat();
    assert_eq!(tool_chat.role, ChatRole::Tool);
    assert_eq!(tool_chat.name.as_deref(), Some("c-2"));
}

#[test]
fn message_delta_default() {
    let delta = MessageDelta::default();
    assert_eq!(delta.text, "");
    assert!(delta.tool_call.is_none());
}
