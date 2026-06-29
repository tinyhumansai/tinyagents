//! Harness model layer types.
//!
//! These are the rich, harness-internal request/response shapes, distinct from
//! the simple top-level [`crate::model`] types. They carry tool declarations,
//! tool-choice policy, structured-output formats, and prompt-cache layout
//! metadata.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::message::{AssistantMessage, Message};
use crate::harness::tool::{ToolDelta, ToolSchema};
use crate::harness::usage::Usage;

/// Policy controlling whether and how the model may call tools.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    #[default]
    Auto,
    /// The model must not call any tool.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call the named tool.
    Tool(String),
}

/// The requested output format for a model response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Free-form text.
    Text,
    /// Any JSON object.
    JsonObject,
    /// JSON constrained to a named schema.
    JsonSchema {
        /// Schema name advertised to the provider.
        name: String,
        /// JSON Schema document.
        schema: Value,
    },
}

/// The role a [`PromptSegment`] plays in the assembled prompt. Earlier roles
/// form the stable, cacheable prefix; [`SegmentRole::Volatile`] marks the tail
/// that changes every turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentRole {
    /// System prompt.
    System,
    /// Tool declarations.
    Tools,
    /// Stable instructions.
    Instructions,
    /// Conversation history.
    History,
    /// Volatile, per-turn content that must stay out of stable prefixes.
    Volatile,
}

/// A labeled segment of the prompt used to reason about provider prompt/KV
/// cache stability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptSegment {
    /// Stable identifier for the segment.
    pub id: String,
    /// The role the segment plays in the prompt.
    pub role: SegmentRole,
    /// Whether this segment is part of the cacheable stable prefix.
    pub cacheable: bool,
}

/// A provider-neutral chat model request.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Tool declarations exposed for this call.
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    /// Tool-choice policy.
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Requested response format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Model id or registry alias override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Per-call timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Free-form request metadata.
    #[serde(default)]
    pub metadata: Value,
    /// Tags propagated to events and traces.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Declared prompt cache segments for KV-cache stability.
    #[serde(default)]
    pub cache_segments: Vec<PromptSegment>,
    /// Fingerprint of the stable prompt prefix, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_fingerprint: Option<String>,
}

/// A provider-neutral chat model response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The assistant message produced by the model.
    pub message: AssistantMessage,
    /// Token usage, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Provider finish reason (for example `stop`, `tool_calls`, `length`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Raw provider metadata preserved for callers who need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

/// An incremental streamed chunk of a model response.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelDelta {
    /// Id of the model call this delta belongs to.
    pub call_id: String,
    /// Incremental text content.
    #[serde(default)]
    pub content: String,
    /// Incremental tool-call fragment, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolDelta>,
}

/// A provider-neutral chat model.
///
/// Generic over the application `State`. Distinct from the simple top-level
/// [`crate::model::ChatModel`].
#[async_trait]
pub trait ChatModel<State: Send + Sync>: Send + Sync {
    /// Invokes the model and returns a complete response.
    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse>;

    /// Streams the model response. The default implementation calls
    /// [`ChatModel::invoke`] and yields a single delta with the full text.
    async fn stream(&self, state: &State, request: ModelRequest) -> Result<Vec<ModelDelta>> {
        let response = self.invoke(state, request).await?;
        Ok(vec![ModelDelta {
            call_id: response.message.id.clone().unwrap_or_default(),
            content: response.text(),
            tool_call: None,
        }])
    }
}

/// A name-keyed registry of chat models with an optional default selection.
pub struct ModelRegistry<State: Send + Sync> {
    pub(crate) models: HashMap<String, Arc<dyn ChatModel<State>>>,
    pub(crate) default: Option<String>,
}

use crate::Result;
