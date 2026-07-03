//! Wire types for the OpenAI Chat Completions API.
//!
//! These structs mirror the JSON shapes accepted and returned by
//! `POST {base_url}/chat/completions`. They are deliberately kept separate from
//! the provider-neutral harness types in [`crate::harness::model`]: the mapping
//! between the two lives in the sibling [`super`] module so the wire format
//! never leaks into core harness code.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Request shapes (serialized and sent to OpenAI)
// ---------------------------------------------------------------------------

/// Top-level request body for `POST /chat/completions`.
#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletionRequest {
    /// Target model id (for example `gpt-4.1-mini`).
    pub model: String,
    /// Ordered conversation messages.
    pub messages: Vec<ChatMessageWire>,
    /// Tool (function) declarations exposed to the model. Omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolWire>,
    /// Tool-choice policy. Omitted when no tools are declared.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Structured-output request. Omitted for free-form text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// Sampling temperature. Omitted when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling probability mass. Omitted when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Maximum number of output tokens. Omitted when unset. Used for classic
    /// Chat Completions models; the o-series uses `max_completion_tokens`
    /// instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Maximum number of output tokens for reasoning (o-series) models, which
    /// reject `max_tokens`. Omitted when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Stop sequences that terminate generation. Omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Deterministic generation seed. Omitted when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Request Server-Sent-Events streaming. Omitted (false) for unary calls.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Streaming options (for example `{"include_usage": true}`). Omitted for
    /// unary calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    /// Provider-specific top-level request fields. Flattened after normalized
    /// fields so OpenAI-compatible local providers can receive their own typed
    /// controls without new wire structs.
    #[serde(flatten, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

// ---------------------------------------------------------------------------
// Streaming response shapes (deserialized from SSE `data:` chunks)
// ---------------------------------------------------------------------------

/// One streamed chunk from `POST /chat/completions` with `stream: true`.
///
/// Each Server-Sent-Events `data:` line carries one of these JSON objects. The
/// terminal `data: [DONE]` sentinel is handled by the transport, not parsed
/// into this type.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatCompletionChunk {
    /// Provider response/message id (repeated on every chunk).
    #[serde(default)]
    pub id: Option<String>,
    /// Per-choice incremental deltas; the first choice is used.
    #[serde(default)]
    pub choices: Vec<ChunkChoiceWire>,
    /// Cumulative usage, sent on the final chunk when `include_usage` is set.
    #[serde(default)]
    pub usage: Option<UsageWire>,
}

/// A single streamed choice carrying an incremental [`ChunkDeltaWire`].
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChunkChoiceWire {
    /// The incremental delta for this choice.
    #[serde(default)]
    pub delta: ChunkDeltaWire,
    /// Finish reason, present only on the terminal content chunk.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The incremental `delta` object inside a [`ChunkChoiceWire`].
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChunkDeltaWire {
    /// Incremental text fragment, when present.
    #[serde(default)]
    pub content: Option<String>,
    /// Incremental tool-call fragments, correlated by `index`.
    #[serde(default)]
    pub tool_calls: Vec<ToolCallChunkWire>,
}

/// One incremental tool-call fragment in a streamed delta.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ToolCallChunkWire {
    /// Stable slot index used to correlate fragments across chunks.
    ///
    /// `None` when the provider omits `index` entirely (some OpenAI-compatible
    /// backends do); fragments are then correlated by id/name instead of all
    /// collapsing onto slot 0.
    #[serde(default)]
    pub index: Option<u32>,
    /// Provider-assigned call id, sent on the first fragment for the slot.
    #[serde(default)]
    pub id: Option<String>,
    /// Incremental function name/arguments fragment.
    #[serde(default)]
    pub function: Option<FunctionChunkWire>,
}

/// The incremental `function` payload of a [`ToolCallChunkWire`].
#[derive(Clone, Debug, Default, Deserialize)]
pub struct FunctionChunkWire {
    /// Function name, sent on the first fragment for the slot.
    #[serde(default)]
    pub name: Option<String>,
    /// Incremental stringified-JSON arguments fragment.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// A single message in the request `messages` array.
#[derive(Clone, Debug, Serialize)]
pub struct ChatMessageWire {
    /// Role: `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    /// Textual content. `None` (serialized as absent) for assistant messages
    /// that only carry tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls requested by an assistant message. Omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallWire>,
    /// For `tool` messages, the id of the call this message answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A function-tool declaration in the request `tools` array.
#[derive(Clone, Debug, Serialize)]
pub struct ToolWire {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function name, description, and JSON-schema parameters.
    pub function: FunctionSchemaWire,
}

/// The `function` payload of a [`ToolWire`].
#[derive(Clone, Debug, Serialize)]
pub struct FunctionSchemaWire {
    /// Function (tool) name.
    pub name: String,
    /// Human/model readable description.
    pub description: String,
    /// JSON Schema describing the function arguments.
    pub parameters: Value,
}

// ---------------------------------------------------------------------------
// Shared shapes (appear in both request and response)
// ---------------------------------------------------------------------------

/// A tool call: present in assistant request messages (echoing prior calls) and
/// in response messages (newly requested calls).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallWire {
    /// Provider-assigned call id.
    pub id: String,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The function name and stringified-JSON arguments.
    pub function: FunctionCallWire,
}

/// The `function` payload of a [`ToolCallWire`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallWire {
    /// Function name.
    pub name: String,
    /// Arguments encoded as a JSON **string** (OpenAI sends stringified JSON).
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Response shapes (deserialized from OpenAI)
// ---------------------------------------------------------------------------

/// Top-level response body from `POST /chat/completions`.
#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionResponse {
    /// Provider response/message id.
    #[serde(default)]
    pub id: Option<String>,
    /// Candidate completions; the first is used.
    #[serde(default)]
    pub choices: Vec<ChoiceWire>,
    /// Token usage, when reported.
    #[serde(default)]
    pub usage: Option<UsageWire>,
}

/// A single completion candidate.
#[derive(Clone, Debug, Deserialize)]
pub struct ChoiceWire {
    /// The assistant message produced for this choice.
    pub message: ResponseMessageWire,
    /// Why generation stopped (for example `stop`, `tool_calls`, `length`).
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The assistant message inside a [`ChoiceWire`].
#[derive(Clone, Debug, Deserialize)]
pub struct ResponseMessageWire {
    /// Textual content, when present.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls requested by the model.
    #[serde(default)]
    pub tool_calls: Vec<ToolCallWire>,
}

/// Token usage reported by OpenAI.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct UsageWire {
    /// Prompt/input tokens.
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Completion/output tokens.
    #[serde(default)]
    pub completion_tokens: u64,
    /// Total tokens reported by the provider.
    #[serde(default)]
    pub total_tokens: u64,
    /// Optional input-token breakdown (carries cached tokens).
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetailsWire>,
}

/// The `prompt_tokens_details` breakdown of a [`UsageWire`].
#[derive(Clone, Debug, Default, Deserialize)]
pub struct PromptTokensDetailsWire {
    /// Input tokens served from OpenAI's prompt cache.
    #[serde(default)]
    pub cached_tokens: u64,
}

// ---------------------------------------------------------------------------
// Model-listing shapes (`GET {base_url}/models`)
// ---------------------------------------------------------------------------

/// One model advertised by a provider's list-models endpoint
/// (`GET {base_url}/models`).
///
/// Returned by [`OpenAiModel::list_models`][super::OpenAiModel::list_models].
/// The same shape is served by every OpenAI-compatible provider (Ollama,
/// Together, Groq, OpenRouter, …); fields other than `id` are best-effort and
/// may be absent depending on the provider.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelListing {
    /// The model id (usable as [`OpenAiModel::with_model`][super::OpenAiModel::with_model]).
    pub id: String,
    /// Unix-epoch creation time, when the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    /// Owning organization/account, when the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
}

/// The `{ "object": "list", "data": [...] }` envelope returned by the
/// list-models endpoint.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ModelListWire {
    /// The advertised models.
    #[serde(default)]
    pub data: Vec<ModelListing>,
}
