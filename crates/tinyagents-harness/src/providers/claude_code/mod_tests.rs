use super::*;

#[test]
fn chat_model_profile_advertises_streaming_without_native_tools() {
    let workspace = tempfile::tempdir().expect("workspace");
    let project = tempfile::tempdir().expect("project");
    let provider = ClaudeCodeProvider::new(
        "claude-sonnet-4-6",
        PathBuf::from("claude"),
        workspace.path().to_path_buf(),
        project.path().to_path_buf(),
        None,
    );

    let profile = provider.profile().expect("profile");
    assert_eq!(profile.provider.as_deref(), Some("claude-code"));
    assert_eq!(profile.model.as_deref(), Some("claude-sonnet-4-6"));
    assert!(!profile.tool_calling);
    assert!(!profile.parallel_tool_calls);
    assert!(profile.streaming);
    assert!(!profile.streaming_tool_chunks);
}

#[test]
fn thread_key_uses_caller_supplied_metadata() {
    let mut first = ModelRequest::new(vec![Message::user("hello")]);
    first.metadata = serde_json::json!({"thread_id": "thread-a"});
    let mut second = ModelRequest::new(vec![Message::user("hello")]);
    second.metadata = serde_json::json!({"thread_id": "thread-b"});
    assert_ne!(
        thread_key_from_request(&first),
        thread_key_from_request(&second)
    );
    assert_eq!(thread_key_from_request(&first), "thread-a");
}

#[test]
fn thread_key_without_id_is_ephemeral() {
    let request = ModelRequest::new(vec![Message::user("same text")]);
    let first = thread_key_from_request(&request);
    let second = thread_key_from_request(&request);
    assert_ne!(first, second);
    assert!(first.starts_with("ephemeral_"));
}

#[test]
fn every_system_message_is_coalesced_in_order() {
    let messages = vec![
        ChatMessage::system("base"),
        ChatMessage::user("question"),
        ChatMessage::system("middleware addition"),
    ];
    assert_eq!(
        coalesce_system_prompt(&messages).as_deref(),
        Some("base\n\nmiddleware addition")
    );
}

#[test]
fn cache_identity_includes_project_scope() {
    let workspace = tempfile::tempdir().expect("workspace");
    let project_a = tempfile::tempdir().expect("project a");
    let project_b = tempfile::tempdir().expect("project b");
    let first = ClaudeCodeProvider::new(
        "claude-sonnet-4-6",
        PathBuf::from("claude"),
        workspace.path().to_path_buf(),
        project_a.path().to_path_buf(),
        None,
    );
    let second = ClaudeCodeProvider::new(
        "claude-sonnet-4-6",
        PathBuf::from("claude"),
        workspace.path().to_path_buf(),
        project_b.path().to_path_buf(),
        None,
    );
    assert_ne!(first.cache_identity(), second.cache_identity());
}

#[test]
fn prompt_guided_tool_response_is_exposed_to_the_harness() {
    let response = model_response_with_tools(
        ChatResponse {
            text: Some(
                "before <tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>"
                    .into(),
            ),
            usage: None,
        },
        true,
    );
    assert_eq!(response.text(), "before");
    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(response.message.tool_calls[0].name, "lookup");
}

#[test]
fn streaming_prompt_tool_markup_is_hidden_but_final_call_is_recovered() {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut scrubber = ToolCallStreamScrubber::new();
    let fragments = [
        "before ",
        "<tool_",
        "call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"needle\"}}",
        "</tool_call>",
        " after",
    ];

    for fragment in fragments {
        forward_delta(
            &sender,
            ProviderDelta::TextDelta {
                delta: fragment.into(),
            },
            Some(&mut scrubber),
        );
    }
    forward_delta(
        &sender,
        ProviderDelta::ThinkingDelta {
            delta: "reasoning remains separate".into(),
        },
        Some(&mut scrubber),
    );
    flush_tool_call_scrubber(&sender, Some(&mut scrubber));

    let mut visible = String::new();
    let mut reasoning = String::new();
    while let Ok(ModelStreamItem::MessageDelta(delta)) = receiver.try_recv() {
        visible.push_str(&delta.text);
        reasoning.push_str(&delta.reasoning);
        assert!(
            !delta.text.contains("<tool")
                && !delta.text.contains("lookup")
                && !delta.text.contains("needle"),
            "prompt protocol markup and JSON must not reach stream consumers: {delta:?}"
        );
    }
    assert_eq!(visible, "before  after");
    assert_eq!(reasoning, "reasoning remains separate");

    let response = model_response_with_tools(
        ChatResponse {
            text: Some(fragments.concat()),
            usage: None,
        },
        true,
    );
    assert_eq!(response.text(), "before  after");
    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(response.message.tool_calls[0].name, "lookup");
    assert_eq!(
        response.message.tool_calls[0].arguments,
        serde_json::json!({"query": "needle"})
    );
}

#[test]
fn request_messages_include_tool_and_schema_instructions() {
    let request = ModelRequest {
        messages: vec![Message::user("lookup")],
        tools: vec![tinyinference_llm::tool::ToolSchema::new(
            "lookup",
            "look up a value",
            serde_json::json!({"type":"object"}),
        )],
        response_format: Some(ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: serde_json::json!({"type":"object"}),
        }),
        ..Default::default()
    };
    let messages = request_messages(&request);
    let system = messages
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(system.contains("Tool Use Protocol"));
    assert!(system.contains("JSON Schema"));
}

#[test]
fn request_rendering_preserves_text_adjacent_to_typed_images() {
    use tinyinference_llm::message::{ImageRef, UserMessage};

    let request = ModelRequest::new(vec![Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("before ".into()),
            ContentBlock::Image(ImageRef {
                url: "data:image/png;base64,QUJD".into(),
                mime_type: Some("image/png".into()),
            }),
            ContentBlock::Text(" after".into()),
        ],
    })]);
    let row: serde_json::Value =
        serde_json::from_slice(&render_request_stdin(&request, true)).expect("stream-json row");
    let content = row["message"]["content"]
        .as_array()
        .expect("content blocks");

    assert_eq!(content[0]["text"], "before ");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[2]["text"], " after");
}

#[test]
fn request_rendering_keeps_private_image_marker_text_literal() {
    let request = ModelRequest::new(vec![Message::user(
        "literal [OH_IMAGE:data:image/png;base64,QUJD]",
    )]);
    let row: serde_json::Value =
        serde_json::from_slice(&render_request_stdin(&request, true)).expect("stream-json row");
    let content = row["message"]["content"]
        .as_array()
        .expect("content blocks");

    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["text"], "literal ");
    assert_eq!(content[1]["text"], "[OH_IMAGE:data:image/png;base64,QUJD]");
}
