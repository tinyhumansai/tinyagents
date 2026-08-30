//! OpenAI **Responses API** (`/v1/responses`) request/response translation.
//!
//! A second wire shape [`OpenAiModel`](super::OpenAiModel) can speak, selected
//! by [`with_responses_api_primary`](super::OpenAiModel::with_responses_api_primary).
//! Where Chat Completions uses `messages` / `choices`, the Responses API uses
//! `input` / `instructions` (the system prompt) / `output`. It is the wire the
//! OpenAI Codex OAuth path requires (paired with `with_extra_query_param` +
//! `with_user_agent`).
//!
//! System messages fold into `instructions`, user/assistant/tool turns become
//! `input` items, and the terminal `output_text` (or the first `output_text`
//! content part) becomes the assistant reply.
//!
//! # What this path now carries
//!
//! The request used to be `{model, input, instructions, stream, store,
//! max_output_tokens}` and **silently dropped everything else** a caller set —
//! `tools`, `tool_choice`, `response_format`, `temperature`, `top_p`, `seed`,
//! `stop_sequences`, `continuation_id`, and `provider_options`. That last one
//! made `reasoning: {effort, summary}` unreachable on the only wire format in
//! this crate that supports it. All of them are on the wire now, and the
//! response side reads reasoning items, their `encrypted_content`, and the
//! cache/reasoning usage breakdowns that were previously ignored (so every
//! cached token on this path was billed at the full input rate).
//!
//! # Remaining gaps
//!
//! Tool *declarations* are sent, but a model that calls one comes back as a
//! `function_call` output item this port does not yet decode into
//! [`ToolCall`](crate::tool::ToolCall)s — and tool *results* are
//! rendered as `user` turns carrying an explicit `[tool_result id=…]` prefix
//! rather than native `function_call_output` items. That preserves the causal
//! link the previous fold-into-assistant behaviour erased, but structural
//! tool support and true SSE streaming remain follow-ups.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{AssistantMessage, ContentBlock, Message};
use crate::model::ModelResponse;
use crate::usage::Usage;

/// The `/v1/responses` request body.
///
/// # What used to be missing
///
/// This struct carried only `{model, input, instructions, stream, store,
/// max_output_tokens}`. Everything else a caller set was **silently dropped**:
/// `tools`, `tool_choice`, `response_format`, `temperature`, `top_p`,
/// `stop_sequences`, `seed`, `previous_response_id`, and — most pointedly —
/// `provider_options`, which meant `reasoning: {effort, summary}` was
/// unreachable on the one wire format that supports it. A request that looked
/// fully configured produced an unconfigured call.
#[derive(Debug, Default, Serialize)]
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
    /// Tool declarations. The Responses API flattens the function schema onto
    /// the tool object rather than nesting it under `function` as Chat
    /// Completions does.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_choice: Option<Value>,
    /// Structured output. The Responses API nests it under `text.format`, not
    /// the Chat Completions `response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) text: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) seed: Option<i64>,
    /// Stop sequences. Serialized only when non-empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) stop: Vec<String>,
    /// The stateful-continuation handle. [`ModelRequest::continuation_id`]
    /// existed with a builder and **no reader anywhere in the crate**, so this
    /// was never sent and stateful follow-ups silently restarted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) previous_response_id: Option<String>,
    /// `reasoning: { effort, summary }`, lowered from the provider-neutral
    /// [`ReasoningConfig`][rc].
    ///
    /// [rc]: crate::model::ReasoningConfig
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning: Option<Value>,
    /// Which extra payloads to return.
    ///
    /// Load-bearing for reasoning replay: with `store: false` the server keeps
    /// no state, so reasoning items may be dropped between turns **unless** they
    /// carry `encrypted_content` — which only arrives when
    /// `include: ["reasoning.encrypted_content"]` is requested. Asking for
    /// reasoning and not asking for this is asking for reasoning that cannot be
    /// replayed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) include: Vec<String>,
    /// Provider-specific passthrough merged onto the body. Keys here win, and
    /// this is the escape hatch for anything the typed fields above cannot say.
    #[serde(flatten)]
    pub(super) extra: serde_json::Map<String, Value>,
}

/// The `include` entry that makes reasoning replayable under `store: false`.
pub(super) const INCLUDE_ENCRYPTED_REASONING: &str = "reasoning.encrypted_content";

/// Lowers a provider-neutral [`ReasoningConfig`][rc] onto the Responses
/// `reasoning` object.
///
/// Returns `None` when the config asks for nothing, so an empty config never
/// adds a field. `budget_tokens` has no Responses spelling and is dropped here
/// deliberately — it is Anthropic's knob, and inventing an OpenAI field for it
/// would be worse than ignoring it.
///
/// [rc]: crate::model::ReasoningConfig
pub(super) fn translate_reasoning(config: &crate::model::ReasoningConfig) -> Option<Value> {
    if config.is_empty() {
        return None;
    }
    let mut object = serde_json::Map::new();
    if let Some(effort) = config.effort {
        object.insert("effort".to_string(), Value::String(effort.as_str().into()));
    }
    if let Some(summary) = &config.summary {
        object.insert("summary".to_string(), Value::String(summary.clone()));
    }
    (!object.is_empty()).then_some(Value::Object(object))
}

/// Translates a tool schema onto the Responses API's flattened tool shape.
pub(super) fn translate_tool(schema: &crate::tool::ToolSchema) -> Value {
    serde_json::json!({
        "type": "function",
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters,
    })
}

/// Translates a [`ResponseFormat`][rf] onto the Responses API's `text.format`
/// nesting (Chat Completions' `response_format` has no counterpart here).
///
/// [rf]: crate::model::ResponseFormat
pub(super) fn translate_text_format(
    format: &crate::model::ResponseFormat,
    strict: bool,
) -> Option<Value> {
    use crate::model::ResponseFormat;
    let inner = match format {
        ResponseFormat::Text => return None,
        ResponseFormat::JsonObject => serde_json::json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
            serde_json::json!({
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": strict,
            })
        }
    };
    Some(serde_json::json!({ "format": inner }))
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

#[derive(Debug, Default, Deserialize)]
pub(super) struct ResponsesOutput {
    /// Item kind: `message`, `reasoning`, `function_call`, …
    #[serde(rename = "type", default)]
    pub(super) kind: Option<String>,
    #[serde(default)]
    pub(super) content: Vec<ResponsesContent>,
    /// Reasoning summary parts, on a `reasoning` item.
    #[serde(default)]
    pub(super) summary: Vec<ResponsesContent>,
    /// The opaque reasoning payload that survives `store: false`.
    ///
    /// Only present when the request asked for
    /// [`INCLUDE_ENCRYPTED_REASONING`]. Preserved so a caller can replay
    /// reasoning across turns; without it the server drops reasoning between
    /// stateless turns.
    #[serde(default)]
    pub(super) encrypted_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesContent {
    #[serde(rename = "type")]
    pub(super) kind: Option<String>,
    pub(super) text: Option<String>,
}

/// Responses-API usage block.
///
/// The details sub-objects used to be absent from this struct entirely, so
/// **every cached token on this path was billed at the full input rate** and
/// reasoning tokens were invisible.
#[derive(Debug, Default, Deserialize)]
pub(super) struct ResponsesUsage {
    #[serde(default)]
    pub(super) input_tokens: Option<u64>,
    #[serde(default)]
    pub(super) output_tokens: Option<u64>,
    #[serde(default)]
    pub(super) input_tokens_details: Option<ResponsesInputTokenDetails>,
    #[serde(default)]
    pub(super) output_tokens_details: Option<ResponsesOutputTokenDetails>,
}

/// `usage.input_tokens_details` — the cache breakdown of the input total.
#[derive(Debug, Default, Deserialize)]
pub(super) struct ResponsesInputTokenDetails {
    #[serde(default)]
    pub(super) cached_tokens: Option<u64>,
    /// Cache **writes**, under either spelling gateways use.
    #[serde(default)]
    pub(super) cache_write_tokens: Option<u64>,
    #[serde(default)]
    pub(super) cache_creation_tokens: Option<u64>,
}

/// `usage.output_tokens_details` — where OpenAI reports reasoning tokens.
///
/// Note that **Anthropic has no equivalent field**: its thinking tokens are
/// billed inside `output_tokens`, so a zero here is not evidence that no
/// reasoning happened on an Anthropic-shaped gateway.
#[derive(Debug, Default, Deserialize)]
pub(super) struct ResponsesOutputTokenDetails {
    #[serde(default)]
    pub(super) reasoning_tokens: Option<u64>,
}

/// Concatenates the visible text of a message's content blocks.
fn message_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(ContentBlock::as_text)
        .collect::<Vec<_>>()
        .join("")
}

/// Normalizes a message role for the Responses API.
///
/// Assistant turns key to `output_text`; everything else to `input_text`.
///
/// **Tool results no longer fold into `assistant`.** They used to, which erased
/// tool identity entirely: a tool result became an anonymous assistant utterance
/// with no `tool_call_id`, so the model saw an assistant asserting a fact rather
/// than the answer to a call it made. They are now rendered as `user` turns
/// carrying an explicit `[tool_result …]` prefix (see
/// [`build_responses_input`]), which preserves the causal link on a wire format
/// this text-in/text-out port cannot express structurally. A true
/// `function_call_output` item is the complete fix and rides with native tool
/// support on this path.
fn normalize_role(message: &Message) -> &'static str {
    match message {
        Message::Assistant(_) => "assistant",
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
            // Keep the call id visible so the model can tell *which* call this
            // answers. Folding it into an anonymous assistant turn lost that.
            Message::Tool(m) => {
                let body = message_text(&m.content);
                if body.trim().is_empty() {
                    String::new()
                } else {
                    format!("[tool_result id={} ]\n{body}", m.tool_call_id)
                }
            }
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
    // `output_text` is the whole answer when the server supplies it.
    if let Some(text) = response
        .output_text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        return Some(text.to_string());
    }
    for item in &response.output {
        // A `reasoning` item's parts are chain-of-thought, not the answer;
        // folding them into the visible text would leak reasoning as content.
        if item.kind.as_deref() == Some("reasoning") {
            continue;
        }
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

/// Collects the reasoning text a Responses body carries.
///
/// Reads `reasoning` items' `summary` parts, then any `content` parts on the
/// same item. Returns `None` when there is none. The parser used to read
/// **no** reasoning at all from this path.
pub(super) fn extract_responses_reasoning(response: &ResponsesResponse) -> Option<String> {
    let mut text = String::new();
    for item in &response.output {
        if item.kind.as_deref() != Some("reasoning") {
            continue;
        }
        for part in item.summary.iter().chain(item.content.iter()) {
            if let Some(fragment) = part
                .text
                .as_deref()
                .map(str::trim)
                .filter(|t| !t.is_empty())
            {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(fragment);
            }
        }
    }
    (!text.is_empty()).then_some(text)
}

/// The opaque reasoning payload to replay on the next turn, when the request
/// asked for [`INCLUDE_ENCRYPTED_REASONING`] and the server supplied it.
pub(super) fn extract_encrypted_reasoning(response: &ResponsesResponse) -> Option<String> {
    response
        .output
        .iter()
        .find_map(|item| item.encrypted_content.clone())
        .filter(|value| !value.is_empty())
}

/// Maps a Responses `usage` block onto the neutral [`Usage`], including the
/// cache and reasoning breakdowns.
pub(super) fn convert_responses_usage(wire: &ResponsesUsage) -> Usage {
    let input_details = wire.input_tokens_details.as_ref();
    let cache_read_tokens = input_details.and_then(|d| d.cached_tokens).unwrap_or(0);
    let cache_creation_tokens = input_details
        .map(|d| {
            d.cache_write_tokens
                .unwrap_or(0)
                .max(d.cache_creation_tokens.unwrap_or(0))
        })
        .unwrap_or(0);
    let input_tokens = wire.input_tokens.unwrap_or(0);
    let output_tokens = wire.output_tokens.unwrap_or(0);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        reasoning_tokens: wire
            .output_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .unwrap_or(0),
    }
}

/// Parses a raw `/v1/responses` JSON body into a [`ModelResponse`].
///
/// Reasoning items surface as a leading
/// [`ContentBlock::Thinking`] block, consistent with the Chat Completions path;
/// the encrypted payload, when present, rides on the block's `signature` so it
/// can be replayed on a later turn.
pub(super) fn parse_responses_response(value: Value) -> ModelResponse {
    let parsed: ResponsesResponse =
        serde_json::from_value(value.clone()).unwrap_or_else(|_| ResponsesResponse {
            output: Vec::new(),
            output_text: None,
            usage: None,
        });
    let text = extract_responses_text(&parsed).unwrap_or_default();
    let usage = parsed.usage.as_ref().map(convert_responses_usage);

    let mut content = Vec::new();
    if let Some(reasoning) = extract_responses_reasoning(&parsed) {
        content.push(ContentBlock::Thinking {
            text: reasoning,
            signature: extract_encrypted_reasoning(&parsed),
        });
    }
    content.push(ContentBlock::Text(text));

    ModelResponse {
        message: AssistantMessage {
            id: None,
            content,
            tool_calls: Vec::new(),
            usage,
        },
        usage,
        finish_reason: Some("stop".to_string()),
        raw: Some(value),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
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
                ..ResponsesOutput::default()
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
