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
