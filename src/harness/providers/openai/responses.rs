//! OpenAI **Responses API** (`/v1/responses`) request/response translation.
//!
//! A second wire shape [`OpenAiModel`](super::OpenAiModel) can speak, selected
//! by [`with_responses_api_primary`](super::OpenAiModel::with_responses_api_primary).
//! Where Chat Completions uses `messages` / `choices`, the Responses API uses
//! `input` / `instructions` (the system prompt) / `output`. It is the wire the
//! OpenAI Codex OAuth path requires (paired with `with_extra_query_param` +
//! `with_user_agent`).
//!
//! This first port is **text-in / text-out**: system messages fold into
//! `instructions`, user/assistant/tool turns become `input` items, and the
//! terminal `output_text` (or the first `output_text` content part) becomes the
//! assistant reply. Native tool calls over `/responses` and true SSE streaming
//! are follow-ups; the harness embeds tool specs in the prompt for this path
//! (its [`profile`](super::OpenAiModel) advertises the caller's chosen
//! `tool_calling`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::message::{AssistantMessage, ContentBlock, Message};
use crate::harness::model::ModelResponse;
use crate::harness::usage::Usage;

/// The `/v1/responses` request body.
#[derive(Debug, Serialize)]
pub(super) struct ResponsesRequest {
    pub(super) model: String,
    pub(super) input: Vec<ResponsesInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) store: Option<bool>,
    /// `max_output_tokens` — the Responses-API output cap, carrying the request's
    /// `max_tokens`. Omitted for the Codex OAuth backend, which rejects it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_output_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesInput {
    pub(super) role: String,
    pub(super) content: Vec<ResponsesContentPart>,
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesContentPart {
    #[serde(rename = "type")]
    pub(super) kind: String,
    pub(super) text: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesResponse {
    #[serde(default)]
    pub(super) output: Vec<ResponsesOutput>,
    #[serde(default)]
    pub(super) output_text: Option<String>,
    #[serde(default)]
    pub(super) usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesOutput {
    #[serde(default)]
    pub(super) content: Vec<ResponsesContent>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesContent {
    #[serde(rename = "type")]
    pub(super) kind: Option<String>,
    pub(super) text: Option<String>,
}

/// Responses-API usage block (`input_tokens` / `output_tokens`).
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesUsage {
    #[serde(default)]
    pub(super) input_tokens: Option<u64>,
    #[serde(default)]
    pub(super) output_tokens: Option<u64>,
}

/// Concatenates the visible text of a message's content blocks.
fn message_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(ContentBlock::as_text)
        .collect::<Vec<_>>()
        .join("")
}

/// Normalizes a message role for the Responses API: assistant + tool turns fold
/// into `assistant` (which the API keys to `output_text`), everything else to
/// `user` (`input_text`). Mirrors the host `normalize_responses_role`.
fn normalize_role(message: &Message) -> &'static str {
    match message {
        Message::Assistant(_) | Message::Tool(_) => "assistant",
        _ => "user",
    }
}

/// Splits a provider-neutral message list into the Responses `instructions`
/// (concatenated system text) and `input` items. Empty-text turns are skipped;
/// the content-part `kind` tracks the *normalized* role (`output_text` for
/// assistant/tool, `input_text` otherwise) — the API rejects `input_text` on an
/// assistant item.
pub(super) fn build_responses_input(messages: &[Message]) -> (Option<String>, Vec<ResponsesInput>) {
    let mut instructions_parts = Vec::new();
    let mut input = Vec::new();

    for message in messages {
        let text = match message {
            Message::System(m) => {
                let t = message_text(&m.content);
                if !t.trim().is_empty() {
                    instructions_parts.push(t);
                }
                continue;
            }
            Message::User(m) => message_text(&m.content),
            Message::Assistant(m) => message_text(&m.content),
            Message::Tool(m) => message_text(&m.content),
        };
        if text.trim().is_empty() {
            continue;
        }
        let role = normalize_role(message);
        input.push(ResponsesInput {
            role: role.to_string(),
            content: vec![ResponsesContentPart {
                kind: if role == "assistant" {
                    "output_text".to_string()
                } else {
                    "input_text".to_string()
                },
                text,
            }],
        });
    }

    let instructions = (!instructions_parts.is_empty()).then(|| instructions_parts.join("\n\n"));
    (instructions, input)
}

/// Extracts the assistant text from a Responses body: the convenience
/// `output_text` field first, else the first `output_text` content part.
pub(super) fn extract_responses_text(response: &ResponsesResponse) -> Option<String> {
    if let Some(text) = response
        .output_text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        return Some(text.to_string());
    }
    for item in &response.output {
        for content in &item.content {
            if content.kind.as_deref() == Some("output_text")
                && let Some(text) = content
                    .text
                    .as_deref()
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
            {
                return Some(text.to_string());
            }
        }
    }
    None
}

/// Parses a raw `/v1/responses` JSON body into a [`ModelResponse`] (text reply).
pub(super) fn parse_responses_response(value: Value) -> ModelResponse {
    let parsed: ResponsesResponse =
        serde_json::from_value(value.clone()).unwrap_or(ResponsesResponse {
            output: Vec::new(),
            output_text: None,
            usage: None,
        });
    let text = extract_responses_text(&parsed).unwrap_or_default();
    let usage = parsed.usage.as_ref().map(|u| Usage {
        input_tokens: u.input_tokens.unwrap_or(0),
        output_tokens: u.output_tokens.unwrap_or(0),
        total_tokens: u.input_tokens.unwrap_or(0) + u.output_tokens.unwrap_or(0),
        ..Usage::default()
    });
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text)],
            tool_calls: Vec::new(),
            usage,
        },
        usage,
        finish_reason: Some("stop".to_string()),
        raw: Some(value),
        resolved_model: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::message::Message;
    use serde_json::json;

    #[test]
    fn build_input_folds_system_into_instructions_and_keys_roles() {
        let messages = vec![
            Message::system("be terse"),
            Message::system("and correct"),
            Message::user("hi"),
            Message::assistant("hello"),
            Message::user("  "), // empty → skipped
        ];
        let (instructions, input) = build_responses_input(&messages);
        assert_eq!(instructions.as_deref(), Some("be terse\n\nand correct"));
        assert_eq!(input.len(), 2);
        assert_eq!(input[0].role, "user");
        assert_eq!(input[0].content[0].kind, "input_text");
        assert_eq!(input[0].content[0].text, "hi");
        // Assistant items must use `output_text`, not `input_text`.
        assert_eq!(input[1].role, "assistant");
        assert_eq!(input[1].content[0].kind, "output_text");
        assert_eq!(input[1].content[0].text, "hello");
    }

    #[test]
    fn extract_text_prefers_output_text_then_scans_content() {
        let with_convenience = ResponsesResponse {
            output: Vec::new(),
            output_text: Some("  final  ".to_string()),
            usage: None,
        };
        assert_eq!(
            extract_responses_text(&with_convenience).as_deref(),
            Some("final")
        );

        let via_content = ResponsesResponse {
            output: vec![ResponsesOutput {
                content: vec![
                    ResponsesContent {
                        kind: Some("reasoning".into()),
                        text: Some("...".into()),
                    },
                    ResponsesContent {
                        kind: Some("output_text".into()),
                        text: Some("answer".into()),
                    },
                ],
            }],
            output_text: None,
            usage: None,
        };
        assert_eq!(
            extract_responses_text(&via_content).as_deref(),
            Some("answer")
        );

        let empty = ResponsesResponse {
            output: Vec::new(),
            output_text: None,
            usage: None,
        };
        assert_eq!(extract_responses_text(&empty), None);
    }

    #[test]
    fn parse_maps_text_and_usage_onto_model_response() {
        let body = json!({
            "output_text": "the answer",
            "usage": { "input_tokens": 12, "output_tokens": 5 }
        });
        let resp = parse_responses_response(body);
        assert_eq!(resp.text(), "the answer");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        let usage = resp.usage.expect("usage mapped");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 17);
    }

    #[test]
    fn parse_tolerates_a_body_without_output() {
        let resp = parse_responses_response(json!({ "id": "resp_1" }));
        assert_eq!(resp.text(), "");
    }
}
