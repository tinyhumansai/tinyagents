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
