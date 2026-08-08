//! Request/response conversion between the provider-neutral harness
//! types and the OpenAI wire format (`translate_message`,
//! `parse_response`, usage conversion, reasoning-text extraction).
//!
//! Split out of `openai/mod.rs`; see that module's doc comment for the
//! full provider overview.

use super::*;

use std::sync::atomic::{AtomicU64, Ordering};

/// Process-global counter handing every decoded response a distinct **epoch**.
///
/// Synthetic tool-call ids used to be `tool-{slot}`, keyed only to the call's
/// position in its own response. Any runtime that omits `id` (several Ollama
/// builds do) therefore emitted `tool-0` on *every* assistant turn, so one run's
/// transcript contained several distinct calls all declaring the same id — an
/// unresolvable pairing for the agent loop. Prefixing with a monotonic epoch
/// makes the id unique for the life of the process while staying stable within
/// the response that minted it (the epoch is drawn once, at the top of decoding,
/// and reused for every slot and every streamed delta of that response).
static SYNTHETIC_ID_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Draws the next synthetic-id epoch. Call **once** per decoded response (unary
/// parse, or accumulator construction on the streaming path) and thread the
/// value through every `tool_call_id` / `tool_call_from_wire` call for that
/// response.
pub(super) fn next_synthetic_id_epoch() -> u64 {
    SYNTHETIC_ID_EPOCH.fetch_add(1, Ordering::Relaxed)
}

/// Prefix for ids this crate synthesizes on the **native/provider** boundary.
///
/// Deliberately distinct from the prompt-guided text protocol's `call_{index}`
/// ids (`crate::harness::tool::prompt`) so a transcript that mixes both — a
/// model that degraded from native to prompt-guided mid-run — can never produce
/// two different calls carrying the same id.
const SYNTHETIC_ID_PREFIX: &str = "tacall";

/// Characters a tool-call id may contain before the provider boundary rewrites
/// it. Mirrors LangChain's `_TOOL_CALL_ID_PATTERN` (`^[a-zA-Z0-9_-]+$`).
fn is_conforming_tool_call_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Rewrites a non-conforming provider tool-call id into the conforming
/// alphabet, **deterministically**.
///
/// Some gateways emit ids that other providers reject on the way back
/// (`functions.write_todos:0` is the canonical example). Rewriting at the
/// provider boundary keeps the id and its paired tool result consistent, because
/// both are derived from the same [`ToolCall`]. Determinism is the whole point:
/// the same wire id must always map to the same rewritten id, or a replayed
/// transcript would stop pairing.
///
/// Offending bytes become `_`, and a short hash of the original is appended so
/// two distinct ids that sanitize to the same string stay distinguishable.
fn normalize_tool_call_id(id: &str) -> String {
    if is_conforming_tool_call_id(id) {
        return id.to_string();
    }
    let sanitized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // FNV-1a over the original bytes: tiny, dependency-free, and stable across
    // processes and crate versions (unlike `DefaultHasher`, whose output is not
    // guaranteed stable).
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let normalized = format!("{sanitized}_{hash:x}");
    tracing::debug!(
        target: "tinyagents::providers::openai",
        normalized = %normalized,
        "[openai] rewrote a non-conforming provider tool-call id"
    );
    normalized
}

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

/// The degraded `response_format` wire form for a [`ResponseFormat::JsonObject`]
/// request against a server that rejects `{"type":"json_object"}` (LM Studio,
/// and likely other local OpenAI-compatible runtimes).
///
/// Maps to a permissive `json_schema` that constrains output to *some* JSON
/// object without pinning its shape: an empty object schema with `strict:false`,
/// so the model is free to choose keys. Verified accepted by LM Studio; a strict
/// or fully-specified schema would over-constrain the free-form "just JSON"
/// intent that [`ResponseFormat::JsonObject`] expresses.
pub(super) fn degraded_json_object_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "json_object",
            "schema": { "type": "object" },
            "strict": false,
        }
    })
}

/// Prepares a caller-supplied JSON Schema for OpenAI **strict** structured
/// output.
///
/// OpenAI's strict mode is not "the same schema, validated harder": it rejects
/// any object that does not carry `additionalProperties: false` and list *every*
/// declared property in `required`. Sending a caller's raw schema with
/// `strict: true` therefore 400s on schemas that are perfectly valid JSON
/// Schema — including this crate's own documented example.
///
/// # Wave-2 dependency
///
/// The recursive sanitizer (force-populate `required`, set
/// `additionalProperties: false` at every object level) is being built as a
/// callable function in `crate::harness::tool::schema`. **This is its call
/// site**: when that function lands, replace the body below with a call to it.
/// Until then this is the identity transform, and correctness rests on the
/// `strict` default — [`OpenAiModel::with_strict_json_schema`][swj] — which is
/// `false` for local runtimes and can be turned off anywhere, plus the
/// automatic 400-driven degrade to `strict: false`.
///
/// [swj]: super::OpenAiModel::with_strict_json_schema
fn prepare_strict_schema(schema: &Value) -> Value {
    // TODO(wave-2): call `crate::harness::tool::schema::harden_for_strict` here
    // once that agent's function lands; see the doc comment above.
    schema.clone()
}

/// Translates a [`ResponseFormat`] into the OpenAI `response_format` JSON value.
///
/// Returns `None` for [`ResponseFormat::Text`] so the field is omitted entirely.
///
/// `strict` selects OpenAI strict structured output for the schema forms. It is
/// **not** hardcoded: hosted OpenAI defaults it on, local runtimes default it
/// off (they reject the key outright, and their schema support is looser), and
/// a 400 implicating the schema degrades it for a single retry. When `strict` is
/// on the schema first goes through [`prepare_strict_schema`].
pub(super) fn translate_response_format(format: &ResponseFormat, strict: bool) -> Option<Value> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(json!({ "type": "json_object" })),
        // OpenAI supports native structured output, so `Auto` maps to a JSON
        // schema request directly. (The agent loop normally resolves `Auto`
        // before reaching the provider; this keeps direct calls correct too.)
        ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
            let schema = if strict {
                prepare_strict_schema(schema)
            } else {
                schema.clone()
            };
            Some(json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": strict,
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
    parse_chat_response(value, None, CacheTokenAccounting::default())
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
    accounting: CacheTokenAccounting,
) -> Result<ModelResponse> {
    let parsed: ChatCompletionResponse = serde_json::from_value(value.clone())?;
    // One epoch for the whole response, so every synthesized id in it shares a
    // prefix and no later response can reuse it.
    let epoch = next_synthetic_id_epoch();

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
        .enumerate()
        .map(|(index, call)| {
            // Local servers routinely omit `id`; synthesize the same
            // run-unique fallback the streaming path uses so the agent loop
            // can still correlate the tool result back to this call. An empty
            // id is treated as absent.
            tool_call_from_wire(
                "openai response",
                epoch,
                index,
                &call.id,
                &call.function.name,
                &call.function.arguments,
            )
        })
        .collect::<Vec<_>>();

    let usage = parsed
        .usage
        .map(|wire| convert_usage_with(wire, accounting));

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
        continue_turn: None,
    })
}

/// Returns the effective call id for a tool-call slot.
///
/// * A provider-assigned id is kept, after [`normalize_tool_call_id`] rewrites
///   any character the conforming alphabet (`[A-Za-z0-9_-]`) rejects.
/// * An absent id is synthesized as `tacall-{epoch}-{slot}`. `epoch` comes from
///   [`next_synthetic_id_epoch`] and is drawn **once per decoded response**, so
///   the id is stable across the streamed deltas and the terminal response of
///   one call while never repeating on a later turn — the defect the old
///   `tool-{slot}` form had, which made every id-less Ollama turn emit `tool-0`.
pub(super) fn tool_call_id(epoch: u64, slot: usize, id: &str) -> String {
    if id.is_empty() {
        format!("{SYNTHETIC_ID_PREFIX}-{epoch}-{slot}")
    } else {
        normalize_tool_call_id(id)
    }
}

/// Builds a provider-neutral [`ToolCall`] from the wire fields, tolerating the
/// defects small local models produce.
///
/// `epoch` + `slot` identify the call: `slot` is its position in the response
/// and `epoch` the response's [`next_synthetic_id_epoch`] draw, which together
/// synthesize a run-unique id when the provider omits one (Ollama did so until
/// v0.12.11). When the arguments cannot be parsed even after repair, the call is
/// marked [`ToolCall::invalid`] with the raw arguments preserved rather than
/// failing the whole model call: the agent loop feeds the error back to the
/// model as a tool result so it can retry (mirroring LangChain and the AI SDK),
/// and — because the call still resolves — a malformed argument blob can never
/// become a never-resolving tool call that stalls the loop.
pub(super) fn tool_call_from_wire(
    context: &str,
    epoch: u64,
    slot: usize,
    id: &str,
    name: &str,
    raw: &str,
) -> ToolCall {
    let call_id = tool_call_id(epoch, slot, id);
    match parse_tool_arguments(raw) {
        Ok(arguments) => ToolCall {
            id: call_id,
            name: name.to_string(),
            arguments,
            invalid: None,
        },
        Err(detail) => {
            let reason = format!(
                "{context} contained invalid JSON arguments for tool call `{call_id}` (`{name}`): {detail}; raw arguments: {raw:?}"
            );
            ToolCall::invalid(call_id, name, raw, reason)
        }
    }
}

/// Parses a tool-call arguments string into JSON, returning `Err(detail)` with a
/// short human-readable reason when the string cannot be recovered.
///
/// This never fails the model call itself; the caller
/// ([`tool_call_from_wire`]) turns an `Err` into an [`ToolCall::invalid`] call.
pub(super) fn parse_tool_arguments(raw: &str) -> std::result::Result<Value, String> {
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
            Err(err.to_string())
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
/// Three conservative strategies, tried in order; the caller only invokes this
/// *after* the raw string has already failed `serde_json::from_str`, so this can
/// never rewrite arguments that were already valid:
///
/// 1. Strip leaked chat-template tool-call delimiters (see
///    [`TOOL_CALL_TEMPLATE_MARKERS`]) and re-parse — recovers a valid call whose
///    only corruption is a leaked marker (e.g. `{"a":1}<tool_call|>`).
/// 2. Take the first complete JSON *object* from the front of the (marker-stripped)
///    string — recovers a valid leading call followed by trailing template noise
///    or a second concatenated fragment (e.g. `{"a":1}<tool_call|>{"b":2}`).
/// 3. Repair the relaxed / malformed JSON small local models emit — unquoted
///    object keys (`{tool:…}`), redundant wrapping braces (`{{…}}`, escalating
///    as the model retries a bounced call), and leaked quote tokens
///    (`[<|">discord<|">]`). Conservative and object-only; see [`super::relaxed_json`].
///
/// Restricting strategy 2 to a leading `Value::Object` keeps it from accepting a
/// bare number/string scraped out of surrounding noise as if it were the call's
/// arguments; strategy 3 likewise accepts only a strictly-parseable object.
/// Returns `None` when no strategy yields valid object-shaped JSON, so the caller
/// still fails fast on genuinely malformed input.
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
    if let Some(Ok(value @ Value::Object(_))) = values.next() {
        return Some(value);
    }

    // Strategy 3: repair relaxed/malformed JSON (unquoted keys, redundant
    // wrapping braces, leaked quote tokens). Without it these calls never parse,
    // are marked `invalid`, and the model loops adding braces until the step
    // budget is exhausted.
    super::relaxed_json::recover_relaxed_object(candidate)
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

/// Whether a provider's reported input-token count already contains the tokens
/// it served from (or wrote into) its prompt cache.
///
/// This is a real cross-provider divergence, not a detail:
///
/// * **OpenAI** reports `prompt_tokens` as the *total* input, with
///   `prompt_tokens_details.cached_tokens` a breakdown of it. Subtracting cache
///   reads to price the uncached remainder is correct.
/// * **Anthropic** reports `input_tokens` *excluding* cache reads and cache
///   writes — the true input total is `input + cache_read + cache_creation`.
///
/// An OpenAI-compatible gateway fronting an Anthropic model can pass either
/// convention through, and guessing wrong silently under-bills (OpenAI
/// semantics assumed over Anthropic data) or double-counts. Select the right one
/// with [`OpenAiModel::with_cache_token_accounting`][cta].
///
/// [cta]: super::OpenAiModel::with_cache_token_accounting
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheTokenAccounting {
    /// OpenAI semantics: the reported input count already contains cache
    /// read/creation tokens. The default.
    #[default]
    IncludedInInput,
    /// Anthropic semantics: cache read/creation tokens are reported *outside*
    /// the input count, so the true input total is recomputed as
    /// `input + cache_read + cache_creation`.
    ExcludedFromInput,
}

/// Converts an OpenAI [`UsageWire`] into [`Usage`] under an explicit
/// [`CacheTokenAccounting`] convention.
///
/// Maps both cache directions, unlike the original which only read
/// `cached_tokens`: `cache_creation_tokens` is a first-class, summed and priced
/// field on [`Usage`] that no provider ever populated, so cache **writes** were
/// invisible to every cost report in the crate.
pub(super) fn convert_usage_with(wire: UsageWire, accounting: CacheTokenAccounting) -> Usage {
    let prompt_details = wire.prompt_tokens_details.unwrap_or_default();
    let cache_read_tokens = prompt_details.cached_tokens;
    let cache_creation_tokens = prompt_details.cache_creation_tokens();

    let input_tokens = match accounting {
        CacheTokenAccounting::IncludedInInput => wire.prompt_tokens,
        CacheTokenAccounting::ExcludedFromInput => wire
            .prompt_tokens
            .saturating_add(cache_read_tokens)
            .saturating_add(cache_creation_tokens),
    };

    // OpenAI-compatible endpoints sometimes omit `total_tokens` entirely
    // (deserializes to `0` via `#[serde(default)]`); fall back to
    // `prompt + completion` so `total_tokens` is never a misleading zero for
    // a call that clearly consumed tokens. Under Anthropic semantics the
    // recomputed `input_tokens` is what the total must be built from.
    let total_tokens =
        if wire.total_tokens > 0 && accounting == CacheTokenAccounting::IncludedInInput {
            wire.total_tokens
        } else {
            input_tokens + wire.completion_tokens
        };

    Usage {
        input_tokens,
        output_tokens: wire.completion_tokens,
        total_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        reasoning_tokens: wire
            .completion_tokens_details
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0),
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
