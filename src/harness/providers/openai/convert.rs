//! Request/response conversion between the provider-neutral harness
//! types and the OpenAI wire format (`translate_message`,
//! `parse_response`, usage conversion, reasoning-text extraction).
//!
//! Split out of `openai/mod.rs`; see that module's doc comment for the
//! full provider overview.

use super::*;

/// Translates one harness [`Message`] into an OpenAI wire message.
///
/// User messages are rendered as OpenAI content-parts when they carry non-text
/// blocks (for example images), so image inputs are actually sent rather than
/// silently dropped. Blocks that have no faithful OpenAI representation return a
/// [`TinyAgentsError::Validation`] instead of being discarded.
pub(super) fn translate_message(message: &Message) -> Result<ChatMessageWire> {
    let wire = match message {
        Message::System(_) => ChatMessageWire {
            role: "system".to_string(),
            content: Some(MessageContentWire::Text(message.text())),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        Message::User(user) => ChatMessageWire {
            role: "user".to_string(),
            content: Some(translate_user_content(&user.content)?),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        Message::Assistant(assistant) => {
            let text = message.text();
            // OpenAI accepts a null content for tool-call-only assistant turns.
            let content = if text.is_empty() && !assistant.tool_calls.is_empty() {
                None
            } else {
                Some(MessageContentWire::Text(text))
            };
            let tool_calls = assistant
                .tool_calls
                .iter()
                .map(|call| {
                    Ok(ToolCallWire {
                        id: call.id.clone(),
                        kind: "function".to_string(),
                        function: FunctionCallWire {
                            name: call.name.clone(),
                            // OpenAI expects arguments as a JSON string.
                            arguments: serde_json::to_string(&call.arguments)?,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ChatMessageWire {
                role: "assistant".to_string(),
                content,
                tool_calls,
                tool_call_id: None,
            }
        }
        Message::Tool(tool) => ChatMessageWire {
            role: "tool".to_string(),
            content: Some(MessageContentWire::Text(message.text())),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool.tool_call_id.clone()),
        },
    };
    Ok(wire)
}

/// Renders user-message content blocks into OpenAI message content.
///
/// Text-only content collapses to a plain string (preserving the historical wire
/// shape). When an image block is present, content is emitted as OpenAI
/// content-parts so the image is actually sent. JSON blocks are serialized into
/// text parts. A [`ContentBlock::ProviderExtension`] has no faithful OpenAI
/// representation, so it fails closed with a validation error rather than being
/// silently dropped.
pub(super) fn translate_user_content(blocks: &[ContentBlock]) -> Result<MessageContentWire> {
    let has_image = blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Image(_)));

    if !has_image {
        // No image: render as a single string, but still fail closed on blocks
        // that cannot be represented.
        let mut text = String::new();
        for block in blocks {
            match block {
                ContentBlock::Text(t) => text.push_str(t),
                ContentBlock::Json(value) => text.push_str(&value.to_string()),
                ContentBlock::Image(_) => unreachable!("guarded by has_image"),
                // OpenAI-compatible requests have no representation for
                // reasoning blocks; they are dropped rather than failing the
                // request (matching the assistant path, which serializes via
                // `Message::text` and drops them naturally).
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
                ContentBlock::ProviderExtension(_) => {
                    return Err(unrepresentable_block_error());
                }
            }
        }
        return Ok(MessageContentWire::Text(text));
    }

    let mut parts = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            ContentBlock::Text(t) => parts.push(ContentPartWire::Text { text: t.clone() }),
            ContentBlock::Json(value) => parts.push(ContentPartWire::Text {
                text: value.to_string(),
            }),
            ContentBlock::Image(image) => parts.push(ContentPartWire::ImageUrl {
                image_url: ImageUrlWire {
                    url: image.url.clone(),
                },
            }),
            // See the string-rendering arm above: reasoning blocks have no
            // OpenAI representation and are dropped, not failed.
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::ProviderExtension(_) => {
                return Err(unrepresentable_block_error());
            }
        }
    }
    Ok(MessageContentWire::Parts(parts))
}

/// Error returned when a content block cannot be represented in an OpenAI
/// request. Failing closed keeps the block from being silently dropped.
pub(super) fn unrepresentable_block_error() -> TinyAgentsError {
    TinyAgentsError::Validation(
        "OpenAI request cannot represent a provider-extension content block; \
         remove it or target the originating provider"
            .to_string(),
    )
}

/// Translates a [`ToolChoice`] into the OpenAI `tool_choice` JSON value.
pub(super) fn translate_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

/// Translates a [`ResponseFormat`] into the OpenAI `response_format` JSON value.
///
/// Returns `None` for [`ResponseFormat::Text`] so the field is omitted entirely.
pub(super) fn translate_response_format(format: &ResponseFormat) -> Option<Value> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(json!({ "type": "json_object" })),
        // OpenAI supports native structured output, so `Auto` maps to a JSON
        // schema request directly. (The agent loop normally resolves `Auto`
        // before reaching the provider; this keeps direct calls correct too.)
        ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
            Some(json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": true,
                }
            }))
        }
    }
}

/// Parses an OpenAI response body (already decoded into a [`Value`]) into a
/// provider-neutral [`ModelResponse`].
///
/// The first choice is used. The raw JSON is preserved in
/// [`ModelResponse::raw`].
///
/// # Errors
///
/// Returns [`TinyAgentsError::Serialization`] if the value does not match the
/// expected response shape, or [`TinyAgentsError::Model`] when no choices are
/// present.
/// Test-only shorthand for [`parse_chat_response`] with inline extraction off.
/// The production paths call [`parse_chat_response`] directly with the model's
/// configured [`ReasoningTagExtraction`].
#[cfg(test)]
pub(super) fn parse_response(value: Value) -> Result<ModelResponse> {
    parse_chat_response(value, None)
}

/// Like [`parse_response`], but also normalizes reasoning into a leading
/// [`ContentBlock::Thinking`] block. Side-channel reasoning
/// (`reasoning_content` / `reasoning`) is always extracted; inline
/// `<think>…</think>` tags in the visible content are extracted only when
/// `reasoning_tags` is `Some`. When both are present, side-channel reasoning
/// leads and inline reasoning follows, joined by the configured separator.
pub(super) fn parse_chat_response(
    value: Value,
    reasoning_tags: Option<&ReasoningTagExtraction>,
) -> Result<ModelResponse> {
    let parsed: ChatCompletionResponse = serde_json::from_value(value.clone())?;

    let choice = parsed.choices.into_iter().next().ok_or_else(|| {
        TinyAgentsError::Model("openai response contained no choices".to_string())
    })?;

    let mut content = Vec::new();

    // Side-channel reasoning first, normalized the same way as the stream path.
    let mut reasoning = String::new();
    for value in [choice.message.reasoning_content, choice.message.reasoning]
        .into_iter()
        .flatten()
    {
        if let Some(fragment) = reasoning_value_text(value) {
            reasoning.push_str(&fragment);
        }
    }

    // Inline `<think>` extraction on the visible content, when enabled.
    let visible = match (
        choice.message.content.filter(|t| !t.is_empty()),
        reasoning_tags,
    ) {
        (Some(text), Some(config)) => {
            let (visible, inline) = extract_reasoning(config, &text);
            if !inline.is_empty() {
                if !reasoning.is_empty() {
                    reasoning.push_str(config.separator());
                }
                reasoning.push_str(&inline);
            }
            visible
        }
        (Some(text), None) => text,
        (None, _) => String::new(),
    };

    if !reasoning.is_empty() {
        content.push(ContentBlock::Thinking {
            text: reasoning,
            signature: None,
        });
    }
    if !visible.is_empty() {
        content.push(ContentBlock::Text(visible));
    }

    let tool_calls = choice
        .message
        .tool_calls
        .into_iter()
        .map(|call| {
            Ok(ToolCall {
                id: call.id.clone(),
                name: call.function.name.clone(),
                // Tool arguments arrive as a JSON string. Invalid JSON is a
                // provider/model error, not an empty/default argument payload.
                arguments: parse_tool_arguments(
                    "openai response",
                    &call.id,
                    &call.function.name,
                    &call.function.arguments,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let usage = parsed.usage.map(convert_usage);

    let message = AssistantMessage {
        id: parsed.id,
        content,
        tool_calls,
        usage,
    };

    Ok(ModelResponse {
        message,
        usage,
        finish_reason: choice.finish_reason,
        raw: Some(value),
        resolved_model: None,
    })
}

/// Returns the effective call id for a streamed tool-call slot: the
/// provider-assigned id when present, or a stable `tool-{slot}` fallback keyed to
/// the slot's position so delta ids and the final call id always agree.
pub(super) fn tool_call_id(slot: usize, id: &str) -> String {
    if id.is_empty() {
        format!("tool-{slot}")
    } else {
        id.to_string()
    }
}

pub(super) fn parse_tool_arguments(
    context: &str,
    call_id: &str,
    name: &str,
    raw: &str,
) -> Result<Value> {
    // Some OpenAI-compatible backends emit an empty arguments string for a
    // zero-argument tool call. That is a well-formed "no arguments" payload, not
    // malformed JSON, so map it to an empty object instead of failing the call.
    if raw.trim().is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    match serde_json::from_str(raw) {
        Ok(value) => Ok(value),
        Err(err) => {
            // Some OpenAI-compatible gateways fail to strip a model's
            // chat-template tool-call delimiters (e.g. a trailing `<tool_call|>`)
            // before placing the call in `function.arguments`, turning
            // otherwise-valid JSON into an unparseable blob. Seen in the wild
            // leaking a trailing `<tool_call|>` into a Composio `composio_execute`
            // call, which then never parses — orphaning the sub-agent's reduce.
            // Attempt one conservative repair before failing. `recover_tool_arguments`
            // runs only after `raw` has already failed to parse, so well-formed
            // arguments are never rewritten.
            if let Some(value) = recover_tool_arguments(raw) {
                return Ok(value);
            }
            // Still unparseable after repair: fail fast with a clear,
            // non-retryable error. Surfacing the failure keeps the malformed
            // call from becoming a never-resolving tool call that stalls the
            // agent loop (and the parent's join/reduce) indefinitely.
            Err(TinyAgentsError::Model(format!(
                "{context} contained invalid JSON arguments for tool call `{call_id}` (`{name}`): {err}; raw arguments: {raw:?}"
            )))
        }
    }
}

/// Chat-template tool-call delimiters that some OpenAI-compatible gateways fail
/// to strip before placing a call in `function.arguments`. Different model
/// families (Hermes/Qwen/Kimi/…) wrap tool calls in their own markers; a
/// leaked one turns valid argument JSON unparseable.
const TOOL_CALL_TEMPLATE_MARKERS: &[&str] = &[
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call|>",
    "<|tool_sep|>",
    "<tool_call|>",
    "</tool_call>",
    "<tool_call>",
];

/// Attempts to recover a usable arguments object from a tool-call arguments
/// string that failed to parse as JSON.
///
/// Two conservative strategies, tried in order; the caller only invokes this
/// *after* the raw string has already failed `serde_json::from_str`, so this can
/// never rewrite arguments that were already valid:
///
/// 1. Strip leaked chat-template tool-call delimiters (see
///    [`TOOL_CALL_TEMPLATE_MARKERS`]) and re-parse — recovers a valid call whose
///    only corruption is a leaked marker (e.g. `{"a":1}<tool_call|>`).
/// 2. Take the first complete JSON *object* from the front of the (marker-stripped)
///    string — recovers a valid leading call followed by trailing template noise
///    or a second concatenated fragment (e.g. `{"a":1}<tool_call|>{"b":2}`).
///
/// Restricting strategy 2 to a leading `Value::Object` keeps it from accepting a
/// bare number/string scraped out of surrounding noise as if it were the call's
/// arguments. Returns `None` when neither strategy yields valid object-shaped JSON,
/// so the caller still fails fast on genuinely malformed input.
fn recover_tool_arguments(raw: &str) -> Option<Value> {
    let stripped = strip_tool_call_markers(raw);
    let candidate = stripped.as_deref().unwrap_or(raw);

    // Strategy 1: the marker-stripped string parses cleanly on its own.
    if stripped.is_some()
        && let Ok(value) = serde_json::from_str::<Value>(candidate)
    {
        return Some(value);
    }

    // Strategy 2: recover the first complete JSON value if it is an object.
    let mut values =
        serde_json::Deserializer::from_str(candidate.trim_start()).into_iter::<Value>();
    match values.next() {
        Some(Ok(value @ Value::Object(_))) => Some(value),
        _ => None,
    }
}

/// Removes any [`TOOL_CALL_TEMPLATE_MARKERS`] found in `raw` and trims the
/// result. Returns `Some(cleaned)` only when a marker was actually present and
/// the trimmed result is non-empty; otherwise `None` (nothing to strip).
fn strip_tool_call_markers(raw: &str) -> Option<String> {
    let mut cleaned = raw.to_string();
    let mut changed = false;
    for &marker in TOOL_CALL_TEMPLATE_MARKERS {
        if cleaned.contains(marker) {
            cleaned = cleaned.replace(marker, "");
            changed = true;
        }
    }
    if !changed {
        return None;
    }
    let trimmed = cleaned.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Converts an OpenAI [`UsageWire`] into the harness-neutral [`Usage`].
pub(super) fn convert_usage(wire: UsageWire) -> Usage {
    // OpenAI-compatible endpoints sometimes omit `total_tokens` entirely
    // (deserializes to `0` via `#[serde(default)]`); fall back to
    // `prompt + completion` so `total_tokens` is never a misleading zero for
    // a call that clearly consumed tokens.
    let total_tokens = if wire.total_tokens > 0 {
        wire.total_tokens
    } else {
        wire.prompt_tokens + wire.completion_tokens
    };
    Usage {
        input_tokens: wire.prompt_tokens,
        output_tokens: wire.completion_tokens,
        total_tokens,
        cache_read_tokens: wire
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0),
        reasoning_tokens: wire
            .completion_tokens_details
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0),
        ..Usage::default()
    }
}

/// Normalizes provider-specific reasoning/thinking payloads into text.
///
/// OpenAI-compatible gateways do not agree on this field: some stream a plain
/// `reasoning_content` string, others use `reasoning`, and a few wrap text in
/// an object/array. Preserve renderable text when obvious and ignore opaque
/// shapes rather than failing an otherwise valid completion.
pub(super) fn reasoning_value_text(value: Value) -> Option<String> {
    match value {
        Value::String(text) => (!text.is_empty()).then_some(text),
        Value::Object(map) => ["text", "content", "summary"]
            .into_iter()
            .find_map(|key| map.get(key).and_then(Value::as_str))
            .filter(|text| !text.is_empty())
            .map(str::to_string),
        Value::Array(values) => {
            let text = values
                .into_iter()
                .filter_map(reasoning_value_text)
                .collect::<String>();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

/// Extracts the reasoning/thinking text from a streamed delta, accepting the
/// common OpenAI-compatible aliases.
pub(super) fn delta_reasoning_text(delta: &mut ChunkDeltaWire) -> String {
    let mut text = String::new();
    for value in [delta.reasoning_content.take(), delta.reasoning.take()]
        .into_iter()
        .flatten()
    {
        if let Some(fragment) = reasoning_value_text(value) {
            text.push_str(&fragment);
        }
    }
    text
}
