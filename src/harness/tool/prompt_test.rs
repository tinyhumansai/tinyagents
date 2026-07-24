//! Tests for the prompt-guided tool-call protocol.

use super::*;
use crate::harness::message::{ContentBlock, Message};

fn schema(name: &str) -> ToolSchema {
    ToolSchema {
        name: name.to_string(),
        description: format!("{name} description"),
        parameters: serde_json::json!({"type": "object"}),
        format: Default::default(),
    }
}

#[test]
fn prompt_instructions_list_each_tool() {
    let text = prompt_tool_instructions(&[schema("read_file"), schema("write_file")]);
    assert!(text.contains("## Tool Use Protocol"));
    assert!(text.contains("<tool_call>"));
    assert!(text.contains("**read_file**"));
    assert!(text.contains("**write_file**"));
}

#[test]
fn prompt_instructions_append_to_system() {
    let msgs = vec![Message::system("You are helpful."), Message::user("hi")];
    let out = with_prompt_tool_instructions(&msgs, &[schema("read_file")]);
    assert_eq!(out.len(), 2);
    let Message::System(system) = &out[0] else {
        panic!("first message should stay system")
    };
    let joined: String = system
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(joined.contains("You are helpful."));
    assert!(joined.contains("Tool Use Protocol"));
}

#[test]
fn prompt_instructions_insert_system_when_absent() {
    let msgs = vec![Message::user("hi")];
    let out = with_prompt_tool_instructions(&msgs, &[schema("read_file")]);
    assert_eq!(out.len(), 2);
    assert!(matches!(out[0], Message::System(_)));
}

#[test]
fn empty_tools_leave_messages_unchanged() {
    let msgs = vec![Message::user("hi")];
    assert_eq!(with_prompt_tool_instructions(&msgs, &[]), msgs);
}

#[test]
fn prompt_results_coalesce_consecutive_tool_messages() {
    let messages = vec![
        Message::user("question"),
        Message::assistant("calling tools"),
        Message::tool("call-1", "first"),
        Message::tool("call-2", "second"),
        Message::assistant("done"),
    ];

    let out = coalesce_prompt_tool_results(&messages);

    assert_eq!(out.len(), 4);
    assert!(matches!(out[0], Message::User(_)));
    assert!(matches!(out[1], Message::Assistant(_)));
    assert!(matches!(out[2], Message::User(_)));
    assert_eq!(
        out[2].text(),
        "[Tool results]\n<tool_result>\nfirst\n</tool_result>\n<tool_result>\nsecond\n</tool_result>"
    );
    assert!(matches!(out[3], Message::Assistant(_)));
}

#[test]
fn prompt_result_coalescing_without_tools_is_identity() {
    let messages = vec![Message::system("system"), Message::user("question")];
    assert_eq!(coalesce_prompt_tool_results(&messages), messages);
}

#[test]
fn prompt_replay_renders_assistant_calls_before_results() {
    let mut assistant = Message::assistant("I will inspect both files.");
    let Message::Assistant(message) = &mut assistant else {
        unreachable!()
    };
    message.tool_calls = vec![
        ToolCall::new("call-1", "read_file", serde_json::json!({"path":"a.txt"})),
        ToolCall::new("call-2", "read_file", serde_json::json!({"path":"b.txt"})),
    ];
    let messages = vec![
        Message::user("compare them"),
        assistant,
        Message::tool("call-1", "first"),
        Message::tool("call-2", "second"),
    ];

    let out = coalesce_prompt_tool_results(&messages);

    let Message::Assistant(replayed) = &out[1] else {
        panic!("assistant call turn should remain an assistant turn")
    };
    assert!(replayed.tool_calls.is_empty());
    assert!(out[1].text().contains("I will inspect both files."));
    assert!(
        out[1].text().contains(
            r#"<tool_call>{"arguments":{"path":"a.txt"},"name":"read_file"}</tool_call>"#
        )
    );
    assert!(
        out[1].text().contains(
            r#"<tool_call>{"arguments":{"path":"b.txt"},"name":"read_file"}</tool_call>"#
        )
    );
    assert!(
        out[2]
            .text()
            .contains("<tool_result>\nfirst\n</tool_result>")
    );
    assert!(
        out[2]
            .text()
            .contains("<tool_result>\nsecond\n</tool_result>")
    );
}

#[test]
fn prompt_parser_extracts_single_tool_call() {
    let text = r#"Let me read it.
<tool_call>
{"name": "read_file", "arguments": {"path": "a.txt"}}
</tool_call>"#;
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(cleaned, "Let me read it.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read_file");
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].arguments, serde_json::json!({"path": "a.txt"}));
}

#[test]
fn prompt_parser_extracts_multiple_calls_and_keeps_prose() {
    let text = r#"a<tool_call>{"name":"one","arguments":{}}</tool_call>b<tool_call>{"name":"two","arguments":{"x":1}}</tool_call>c"#;
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(cleaned, "abc");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "one");
    assert_eq!(calls[1].name, "two");
    assert_eq!(calls[1].id, "call_2");
}

#[test]
fn prompt_parser_defaults_missing_arguments_to_empty_object() {
    let (_, calls) =
        parse_prompt_tool_calls_from_text(r#"<tool_call>{"name":"noargs"}</tool_call>"#);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, serde_json::json!({}));
}

#[test]
fn prompt_parser_drops_malformed_block() {
    let (cleaned, calls) = parse_prompt_tool_calls_from_text("<tool_call>not json</tool_call>done");
    assert!(calls.is_empty());
    assert_eq!(cleaned, "done");
}

#[test]
fn prompt_parser_keeps_unterminated_block_as_text() {
    let text = "text <tool_call>{\"name\":\"x\"}";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert_eq!(cleaned, "text <tool_call>{\"name\":\"x\"}");
}

#[test]
fn prompt_parser_returns_plain_text_verbatim() {
    let (cleaned, calls) = parse_prompt_tool_calls_from_text("just a normal answer");
    assert!(calls.is_empty());
    assert_eq!(cleaned, "just a normal answer");
}

// --- Attribute / variant-tolerant matching (Hermes / DeepSeek templates) ---

#[test]
fn prompt_parser_matches_attribute_form_open_tag() {
    // Regression for the exact-literal miss: `<tool_call id="…">` must match so a
    // native model that leaks the call as text doesn't dump raw markup.
    let text = r#"<tool_call id="call_0">{"name":"foo","arguments":{"a":1}}</tool_call>"#;
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "foo");
    assert_eq!(calls[0].arguments, serde_json::json!({"a": 1}));
    assert!(cleaned.is_empty());
    assert!(!cleaned.contains("<tool_call"));
}

#[test]
fn prompt_parser_matches_pipe_variant_open_tag() {
    let text = r#"<tool_call|>{"name":"foo","arguments":{}}</tool_call>"#;
    let (_, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "foo");
}

#[test]
fn prompt_parser_matches_deepseek_delimiters() {
    let text = "<｜tool▁call▁begin｜>{\"name\":\"foo\",\"arguments\":{}}<｜tool▁call▁end｜>";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "foo");
    assert!(cleaned.is_empty());
}

#[test]
fn prompt_parser_drops_attribute_form_no_name_body_without_leak() {
    // The reported bug: `<tool_call id="call_0">{}</tool_call>` has no `name`.
    // It must be dropped — never echoed back as assistant content.
    let text = "prefix <tool_call id=\"call_0\">\n{}\n</tool_call> suffix";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert!(!cleaned.contains("<tool_call"));
    assert!(!cleaned.contains("{}"));
    assert_eq!(cleaned, "prefix  suffix");
}

#[test]
fn prompt_parser_does_not_match_plural_tag() {
    // `<tool_calls>` must not be mistaken for an opening `<tool_call>`.
    let text = "see the <tool_calls> list";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert_eq!(cleaned, "see the <tool_calls> list");
}

#[test]
fn prompt_parser_does_not_misparse_prose_open_tag_without_close() {
    let text = "You can emit a <tool_call id=\"1\"> block to call a tool.";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert_eq!(cleaned, text);
}

// --- apply_prompt_tool_calls recovery + native-mode fallback gating ---

#[test]
fn apply_prompt_tool_calls_recovers_attribute_markup() {
    // A native model emitted the call as text with EMPTY structured tool_calls:
    // recovery yields a structured call and the raw markup does NOT survive.
    let resp = crate::harness::model::ModelResponse::assistant(
        r#"<tool_call id="call_0">{"name":"foo","arguments":{}}</tool_call>"#,
    );
    let out = apply_prompt_tool_calls(resp);
    assert_eq!(out.message.tool_calls.len(), 1);
    assert_eq!(out.message.tool_calls[0].name, "foo");
    assert!(!out.text().contains("<tool_call"));
    assert!(out.text().is_empty());
}

#[test]
fn should_recover_gating_matrix() {
    // Prompt-guided (native == false): always recover when tools were offered.
    assert!(should_recover(false, true, 0));
    assert!(should_recover(false, true, 2));
    // Native fallback: only when tools offered AND no structured calls.
    assert!(should_recover(true, true, 0));
    // Native with structured calls present: never recover (path stays as-is).
    assert!(!should_recover(true, true, 1));
    // No tools offered: nothing to recover, either mode.
    assert!(!should_recover(true, false, 0));
    assert!(!should_recover(false, false, 0));
}

#[test]
fn should_recover_skips_native_response_with_structured_calls() {
    // When a real structured call is present the guard skips recovery, so the
    // native path is left byte-for-byte unchanged.
    let calls = vec![ToolCall {
        id: "call_1".to_string(),
        name: "real".to_string(),
        arguments: serde_json::json!({}),
        invalid: None,
    }];
    assert!(!should_recover(true, true, calls.len()));
}
