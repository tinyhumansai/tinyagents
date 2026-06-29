//! Rich internal message model for the harness.
//!
//! These typed values are the data that moves between recursion levels — parent
//! to sub-agent, node to sub-graph, model to REPL — so they are deliberately
//! structured rather than stringly typed.
//!
//! Raw strings appear only at API boundaries; internally the harness works
//! with structured [`Message`] values made of typed [`ContentBlock`]s.
//! Ergonomic constructors ([`Message::system`], [`Message::user`], …) and a
//! [`Message::text`] accessor keep the public surface easy to use.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::tool::ToolCall;
use crate::harness::usage::Usage;

/// A typed unit of message content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text(String),
    /// Structured JSON content.
    Json(Value),
    /// A reference to an image input.
    Image(ImageRef),
    /// An opaque provider-specific block preserved verbatim.
    ProviderExtension(Value),
}

/// A reference to an image, either by URL or inline base64 data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageRef {
    /// URL or data URI of the image.
    pub url: String,
    /// Optional MIME type (for example `image/png`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// A system/developer instruction message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemMessage {
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
}

/// A user/human input message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
}

/// An assistant/model output message, possibly carrying tool calls and usage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// Optional provider message id for continuation/resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
    /// Tool calls requested by the model.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Token usage reported for this message, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A tool result message correlated to a prior tool call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolMessage {
    /// Id of the tool call this message answers.
    pub tool_call_id: String,
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
}

/// A structured conversation message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Message {
    /// System/developer instructions.
    System(SystemMessage),
    /// User/human input.
    User(UserMessage),
    /// Assistant/model output.
    Assistant(AssistantMessage),
    /// Tool result.
    Tool(ToolMessage),
}

/// An incremental message update used for streaming model output.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Incremental text fragment.
    #[serde(default)]
    pub text: String,
    /// Incremental tool-call fragment, when the provider streams tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<crate::harness::tool::ToolDelta>,
}
