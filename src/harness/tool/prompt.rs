//! Provider-neutral prompt-guided (text-mode) tool calling for models without native tool support.
//!
//! Provider adapters whose model profile has `tool_calling = false` can use these
//! helpers to embed tool specs **in the system prompt** as a small protocol and
//! parse the model's `<tool_call>…</tool_call>` blocks back into
//! [`ToolCall`]s — so the agent loop sees tool calls identically to the native
//! path, without changing the harness loop.
//!
//! The `<tool_call>{"name":…,"arguments":…}</tool_call>` convention matches the
//! long-standing OpenHuman host format so models already prompted for it behave
//! identically after the crate cutover.

use std::fmt::Write as _;

use serde_json::{Map, Value};

use crate::harness::message::{ContentBlock, Message};
use crate::harness::model::ModelResponse;
use crate::harness::tool::{ToolCall, ToolSchema};

/// Opening / closing delimiters for a text-mode tool call.
const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";

/// Build the tool-use protocol block appended to the system prompt when native
/// tool calling is unavailable. Describes the `<tool_call>` convention and lists
/// each tool's name, description, and JSON-Schema parameters.
pub fn prompt_tool_instructions(tools: &[ToolSchema]) -> String {
    let mut out = String::new();
    out.push_str("## Tool Use Protocol\n\n");
    out.push_str("To use a tool, wrap a JSON object in <tool_call></tool_call> tags:\n\n");
    out.push_str(OPEN_TAG);
    out.push('\n');
    out.push_str(r#"{"name": "tool_name", "arguments": {"param": "value"}}"#);
    out.push('\n');
    out.push_str(CLOSE_TAG);
    out.push_str("\n\n");
    out.push_str("You may emit multiple tool calls in a single response. ");
    out.push_str("After execution, results appear in <tool_result> tags. ");
    out.push_str("Continue reasoning with the results until you can give a final answer.\n\n");
    out.push_str("### Available Tools\n\n");
    for tool in tools {
        let params = serde_json::to_string(&tool.parameters).unwrap_or_else(|_| "{}".to_string());
        // Infallible: writing to a String never errors.
        let _ = writeln!(out, "**{}**: {}", tool.name, tool.description);
        let _ = writeln!(out, "Parameters: `{params}`\n");
    }
    out
}

/// Return `messages` with the tool-use protocol appended to the system prompt:
/// the instructions are added as a trailing block on the first system message, or
/// a new leading system message when the request carries none. `tools` empty →
/// `messages` is returned unchanged (cloned).
pub fn with_prompt_tool_instructions(messages: &[Message], tools: &[ToolSchema]) -> Vec<Message> {
    if tools.is_empty() {
        return messages.to_vec();
    }
    let block = prompt_tool_instructions(tools);
    let mut out = messages.to_vec();
    if let Some(Message::System(system)) = out.iter_mut().find(|m| matches!(m, Message::System(_)))
    {
        // Append as a distinct text block so the original system prompt is intact.
        system
            .content
            .push(ContentBlock::Text(format!("\n\n{block}")));
    } else {
        out.insert(0, Message::system(block));
    }
    out
}

/// Convert native tool-result messages into prompt-guided user turns.
///
/// Models without native tool calling cannot consume a provider `tool` role.
/// Consecutive results are therefore folded into one `[Tool results]` user
/// message while every non-tool message keeps its original order and type.
pub fn coalesce_prompt_tool_results(messages: &[Message]) -> Vec<Message> {
    let mut out = Vec::with_capacity(messages.len());
    let mut pending = Vec::new();

    fn flush(out: &mut Vec<Message>, pending: &mut Vec<String>) {
        if !pending.is_empty() {
            out.push(Message::user(format!(
                "[Tool results]\n{}",
                std::mem::take(pending).join("\n")
            )));
        }
    }

    for message in messages {
        if matches!(message, Message::Tool(_)) {
            pending.push(message.text());
        } else {
            flush(&mut out, &mut pending);
            out.push(message.clone());
        }
    }
    flush(&mut out, &mut pending);
    out
}

/// Extract `<tool_call>…</tool_call>` blocks from `text`, parsing each inner JSON
/// object (`{"name":…,"arguments":…}`) into a [`ToolCall`]. Returns the text with
/// the blocks removed (trimmed) plus the parsed calls, in order.
///
/// Robust to noise: a block whose inner text is not a JSON object with a string
/// `name` is dropped; a dangling `<tool_call>` with no close is left verbatim in
/// the returned text.
pub fn parse_prompt_tool_calls_from_text(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = text;

    while let Some(start) = rest.find(OPEN_TAG) {
        cleaned.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN_TAG.len()..];
        let Some(end) = after_open.find(CLOSE_TAG) else {
            // Unterminated block: keep it (and everything after) as plain text.
            cleaned.push_str(&rest[start..]);
            return (cleaned.trim().to_string(), calls);
        };
        let inner = after_open[..end].trim();
        if let Some(call) = parse_one(inner, calls.len() + 1) {
            calls.push(call);
        }
        rest = &after_open[end + CLOSE_TAG.len()..];
    }
    cleaned.push_str(rest);
    (cleaned.trim().to_string(), calls)
}

/// Extract prompt-guided `<tool_call>` blocks from a completed [`ModelResponse`]'s
/// text into `message.tool_calls`, replacing the message content with the cleaned
/// prose. No-op when the text carries no blocks — so a plain text answer is
/// untouched. Provider adapters should apply this to each completed response after
/// using [`with_prompt_tool_instructions`].
pub fn apply_prompt_tool_calls(mut response: ModelResponse) -> ModelResponse {
    let text = response.text();
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(&text);
    if calls.is_empty() {
        return response;
    }
    response.message.tool_calls.extend(calls);
    response.message.content = if cleaned.is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock::Text(cleaned)]
    };
    response
}

/// Parse a single tool-call body into a [`ToolCall`] with a synthetic 1-based id.
fn parse_one(inner: &str, index: usize) -> Option<ToolCall> {
    let value: Value = serde_json::from_str(inner).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    Some(ToolCall {
        id: format!("call_{index}"),
        name,
        arguments,
        invalid: None,
    })
}

#[cfg(test)]
mod test {
    use super::*;

    fn schema(name: &str) -> ToolSchema {
        ToolSchema {
            name: name.to_string(),
            description: format!("{name} description"),
            parameters: serde_json::json!({"type": "object"}),
            format: Default::default(),
        }
    }

    #[test]
    fn instructions_list_each_tool() {
        let text = prompt_tool_instructions(&[schema("read_file"), schema("write_file")]);
        assert!(text.contains("## Tool Use Protocol"));
        assert!(text.contains("<tool_call>"));
        assert!(text.contains("**read_file**"));
        assert!(text.contains("**write_file**"));
    }

    #[test]
    fn with_tool_instructions_appends_to_system() {
        let msgs = vec![Message::system("You are helpful."), Message::user("hi")];
        let out = with_prompt_tool_instructions(&msgs, &[schema("read_file")]);
        assert_eq!(out.len(), 2);
        let Message::System(sys) = &out[0] else {
            panic!("first message should stay system")
        };
        // Original prompt preserved + protocol appended.
        let joined: String = sys
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(joined.contains("You are helpful."));
        assert!(joined.contains("Tool Use Protocol"));
    }

    #[test]
    fn with_tool_instructions_inserts_system_when_absent() {
        let msgs = vec![Message::user("hi")];
        let out = with_prompt_tool_instructions(&msgs, &[schema("read_file")]);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Message::System(_)));
    }

    #[test]
    fn empty_tools_leaves_messages_unchanged() {
        let msgs = vec![Message::user("hi")];
        assert_eq!(with_prompt_tool_instructions(&msgs, &[]), msgs);
    }

    #[test]
    fn coalesces_consecutive_tool_results_into_one_user_turn() {
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
        assert_eq!(out[2].text(), "[Tool results]\nfirst\nsecond");
        assert!(matches!(out[3], Message::Assistant(_)));
    }

    #[test]
    fn coalescing_without_tool_results_is_identity() {
        let messages = vec![Message::system("system"), Message::user("question")];
        assert_eq!(coalesce_prompt_tool_results(&messages), messages);
    }

    #[test]
    fn parses_single_tool_call() {
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
    fn parses_multiple_calls_and_keeps_prose() {
        let text = r#"a<tool_call>{"name":"one","arguments":{}}</tool_call>b<tool_call>{"name":"two","arguments":{"x":1}}</tool_call>c"#;
        let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
        assert_eq!(cleaned, "abc");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "one");
        assert_eq!(calls[1].name, "two");
        assert_eq!(calls[1].id, "call_2");
    }

    #[test]
    fn missing_arguments_defaults_to_empty_object() {
        let (_, calls) =
            parse_prompt_tool_calls_from_text(r#"<tool_call>{"name":"noargs"}</tool_call>"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn malformed_block_is_dropped() {
        let (cleaned, calls) =
            parse_prompt_tool_calls_from_text("<tool_call>not json</tool_call>done");
        assert!(calls.is_empty());
        assert_eq!(cleaned, "done");
    }

    #[test]
    fn unterminated_block_kept_as_text() {
        let text = "text <tool_call>{\"name\":\"x\"}";
        let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
        assert!(calls.is_empty());
        assert_eq!(cleaned, "text <tool_call>{\"name\":\"x\"}");
    }

    #[test]
    fn no_blocks_returns_text_verbatim() {
        let (cleaned, calls) = parse_prompt_tool_calls_from_text("just a normal answer");
        assert!(calls.is_empty());
        assert_eq!(cleaned, "just a normal answer");
    }
}
