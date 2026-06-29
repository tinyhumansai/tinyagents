use tinyagents::{ChatMessage, ChatRole, ModelRequest};

#[test]
fn serializes_chat_messages() {
    let request = ModelRequest::new(vec![
        ChatMessage::system("You are concise."),
        ChatMessage::user("Hello"),
    ])
    .temperature(0.2)
    .max_tokens(128);

    let json = serde_json::to_value(request).unwrap();

    assert_eq!(json["temperature"], 0.2);
    assert_eq!(json["max_tokens"], 128);
    assert_eq!(json["messages"][0]["role"], "system");
    assert_eq!(json["messages"][1]["content"], "Hello");
}

#[test]
fn constructs_named_tool_message() {
    let message = ChatMessage::tool("lookup", "42");

    assert_eq!(message.role, ChatRole::Tool);
    assert_eq!(message.name.as_deref(), Some("lookup"));
}
