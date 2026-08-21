//! The native dialect: the provider's own structured tool-calling channel.
//!
//! The most reliable of the three when it is available, and the only one with
//! real call ids — which is what makes a parallel multi-call turn correlatable
//! at all. It still falls back to text parsing, because a model with a
//! structured channel available sometimes narrates a `<tool_call>` tag anyway,
//! and dropping that call burns an iteration for no reason.
//!
//! Its real complexity is not parsing but **replay**: see
//! [`pair_tool_cycles`](super::pairing::pair_tool_cycles).

use serde_json::Value;

use super::pairing::pair_tool_cycles;
use super::types::{
    DialectMessage, DialectResponse, ToolCallFormat, ToolOutcome, ToolResultEntry, TranscriptEntry,
};
use super::xml::XmlDialect;
use super::ToolDialect;
use crate::harness::tool::ToolSchema;
use crate::harness::tool_calling::ParsedToolCall;

/// Call id used when an outcome carries none. Only reachable if a host hands
/// this dialect an outcome from a text-parsed call, which the fallback path can
/// produce.
const UNKNOWN_CALL_ID: &str = "unknown";

/// Provider-native structured tool calling.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeDialect;

impl ToolDialect for NativeDialect {
    fn parse_response(&self, response: &DialectResponse) -> (String, Vec<ParsedToolCall>) {
        let text = response.text.clone().unwrap_or_default();
        let calls: Vec<ParsedToolCall> = response
            .tool_calls
            .iter()
            .map(|call| ParsedToolCall {
                name: call.name.clone(),
                arguments: serde_json::from_str(&call.arguments).unwrap_or_else(|error| {
                    tracing::warn!(
                        tool = %call.name,
                        %error,
                        "failed to parse native tool call arguments as JSON; defaulting to empty object"
                    );
                    Value::Object(serde_json::Map::new())
                }),
                id: Some(call.id.clone()),
            })
            .collect();

        if !calls.is_empty() {
            tracing::debug!(
                parse_mode = "native_structured",
                parsed_tool_calls = calls.len(),
                "native dialect parsed response"
            );
            return (text, calls);
        }

        if !text.is_empty() {
            let (fallback_text, fallback_calls) = XmlDialect::parse_text(&text);
            if !fallback_calls.is_empty() {
                // An empty narrative means the whole message was the call; keep
                // the original text so the turn is not left blank.
                let display_text = if fallback_text.is_empty() {
                    text
                } else {
                    fallback_text
                };
                tracing::debug!(
                    parse_mode = "text_fallback",
                    parsed_tool_calls = fallback_calls.len(),
                    "native dialect parsed response"
                );
                return (display_text, fallback_calls);
            }
        }

        tracing::debug!(
            parse_mode = "none",
            parsed_tool_calls = 0,
            "native dialect parsed response"
        );
        (text, calls)
    }

    fn format_results(&self, results: &[ToolOutcome]) -> TranscriptEntry {
        TranscriptEntry::ToolResults(
            results
                .iter()
                .map(|result| ToolResultEntry {
                    tool_call_id: result
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| UNKNOWN_CALL_ID.to_string()),
                    content: result.output.clone(),
                })
                .collect(),
        )
    }

    fn prompt_instructions(&self, _tools: &[ToolSchema]) -> String {
        // No catalogue: the provider already has the full schemas in the
        // request. What the model still needs is the behavioural half —
        // notably that narrating an intention is not calling a tool.
        [
            "## Tool Use Protocol",
            "",
            "When a tool is needed, emit tool calls directly via the model's native tool-calling output.",
            "Do not only narrate intent (for example, avoid \"Let me check...\") without emitting the tool call.",
            "After tool results are provided, continue reasoning and then produce the final answer.",
            "",
        ]
        .join("\n")
    }

    fn to_provider_messages(&self, history: &[TranscriptEntry]) -> Vec<DialectMessage> {
        pair_tool_cycles(history)
            .into_iter()
            .flat_map(|entry| match entry {
                TranscriptEntry::Chat(chat) => vec![chat.clone()],
                TranscriptEntry::AssistantToolCalls {
                    text,
                    tool_calls,
                    reasoning_content,
                    extra_metadata,
                } => {
                    let mut payload = serde_json::json!({
                        "content": text,
                        "tool_calls": tool_calls,
                    });
                    if let Some(reasoning) = reasoning_content {
                        payload["reasoning_content"] = Value::String(reasoning.clone());
                    }
                    vec![
                        DialectMessage::assistant(payload.to_string())
                            .with_metadata(extra_metadata.clone()),
                    ]
                }
                TranscriptEntry::ToolResults(results) => results
                    .iter()
                    .map(|result| {
                        DialectMessage::tool(
                            serde_json::json!({
                                "tool_call_id": result.tool_call_id,
                                "content": result.content,
                            })
                            .to_string(),
                        )
                    })
                    .collect(),
            })
            .collect()
    }

    fn should_send_tool_specs(&self) -> bool {
        true
    }

    fn tool_call_format(&self) -> ToolCallFormat {
        ToolCallFormat::Native
    }
}
