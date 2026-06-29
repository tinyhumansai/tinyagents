//! Serialization behavior for the harness [`Message`] model.

use tinyagents::harness::message::{ContentBlock, Message, UserMessage};

#[test]
fn serializes_messages_with_role_tags() {
    let system = serde_json::to_value(Message::system("You are concise.")).unwrap();
    let user = serde_json::to_value(Message::user("Hello")).unwrap();

    // The `Message` enum is tagged by its snake_case variant name; each variant
    // carries an ordered list of content blocks.
    assert_eq!(system["system"]["content"][0]["text"], "You are concise.");
    assert_eq!(user["user"]["content"][0]["text"], "Hello");
}

#[test]
fn message_round_trips_through_serde() {
    let original = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("part one ".into()),
            ContentBlock::Text("part two".into()),
        ],
    });

    let json = serde_json::to_string(&original).unwrap();
    let decoded: Message = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, original);
    assert_eq!(decoded.text(), "part one part two");
}

#[test]
fn constructs_correlated_tool_message() {
    let message = Message::tool("lookup", "42");

    match &message {
        Message::Tool(tool) => assert_eq!(tool.tool_call_id, "lookup"),
        other => panic!("expected a tool message, got {other:?}"),
    }
    assert_eq!(message.text(), "42");
}
