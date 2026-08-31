//! The transcript vocabulary a [`ToolDialect`](super::ToolDialect) speaks.
//!
//! # Why this is not [`tinyinference::message::Message`]
//!
//! The harness's own message model is richer than what a dialect needs, and
//! richer than what most hosts persist. A host that has been storing agent
//! transcripts for years has a *durable* record shape — a role string, a
//! content string, some passthrough metadata — and the exact bytes it puts on
//! the provider wire are part of its contract with those providers. Lifting
//! that host's dialect logic into this crate through
//! [`Message`](tinyinference::message::Message) would mean a lossy round-trip
//! through a model built for a different purpose, and the loss would land
//! precisely on the fields providers reject requests over: `reasoning_content`,
//! per-call `extra_content`, the tool-call/tool-result pairing.
//!
//! So these types are deliberately thin. They are the *lowest common
//! denominator of a chat transcript* — enough to render a dialect, and no
//! opinion about anything else. A host maps its own records onto them in a
//! handful of `From` impls and gets byte-identical output back.
//!
//! For hosts driving the crate's own agent loop, the equivalent surface is
//! [`crate::tool::prompt`], which speaks
//! [`Message`](tinyinference::message::Message) instead. The two are parallel
//! on purpose; see [the module docs](super) for which one to reach for.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who produced a transcript message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DialectRole {
    /// The system/developer instruction turn.
    System,
    /// A human (or host-synthesized) input turn.
    User,
    /// A model output turn.
    Assistant,
    /// A tool-result turn, in the provider's native `tool` role.
    Tool,
}

impl DialectRole {
    /// The wire spelling of the role: `system` / `user` / `assistant` / `tool`.
    pub fn as_str(self) -> &'static str {
        match self {
            DialectRole::System => "system",
            DialectRole::User => "user",
            DialectRole::Assistant => "assistant",
            DialectRole::Tool => "tool",
        }
    }
}

/// One flat chat message, as a dialect emits it toward a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DialectMessage {
    /// Which turn this is.
    pub role: DialectRole,
    /// The message body. For dialects that pack structure into the body (the
    /// native dialect's assistant turns, for instance) this is a JSON string
    /// the host's provider adapter parses back out.
    pub content: String,
    /// Host passthrough metadata carried verbatim from the transcript record.
    /// The dialect never reads it; it only makes sure it survives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_metadata: Option<Value>,
}

impl DialectMessage {
    /// A `system` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(DialectRole::System, content)
    }

    /// A `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(DialectRole::User, content)
    }

    /// An `assistant` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(DialectRole::Assistant, content)
    }

    /// A `tool` message.
    pub fn tool(content: impl Into<String>) -> Self {
        Self::new(DialectRole::Tool, content)
    }

    /// A message in an explicit role, with no metadata.
    pub fn new(role: DialectRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            extra_metadata: None,
        }
    }

    /// Attaches host passthrough metadata.
    pub fn with_metadata(mut self, metadata: Option<Value>) -> Self {
        self.extra_metadata = metadata;
        self
    }
}

/// A structured tool call exactly as a provider reported it.
///
/// `arguments` stays a **string**, not a [`Value`]: providers stream it as one,
/// and re-serializing a parsed value would reorder keys and lose the exact
/// bytes some providers checksum. Parsing happens once, in
/// [`ToolDialect::parse_response`](super::ToolDialect::parse_response), and the
/// raw form is what gets replayed on the next turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeToolCall {
    /// Provider-assigned call id, correlated with the matching tool result.
    pub id: String,
    /// The tool the model asked for.
    pub name: String,
    /// Raw JSON arguments, verbatim from the provider.
    pub arguments: String,
    /// Provider-specific passthrough for this call, echoed back verbatim on the
    /// next assistant turn. Google Gemini's required
    /// `extra_content.google.thought_signature` rides here; every provider that
    /// does not emit one leaves it `None`, so their history stays byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<Value>,
}

/// What a model returned for one iteration, in the shape a dialect reads.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DialectResponse {
    /// Visible text, if any. May be empty when the model only called tools.
    pub text: Option<String>,
    /// Structured tool calls, if the provider reported any natively.
    pub tool_calls: Vec<NativeToolCall>,
}

impl DialectResponse {
    /// Response text, or `""` when the model returned none.
    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}

/// The outcome of executing one tool call, ready to be rendered back into the
/// transcript.
///
/// Distinct from [`crate::tool::ToolResult`] because the two answer
/// different questions. `ToolResult` is what a [`Tool`](crate::tool::Tool)
/// *produced* — with timings, raw payload, and a mandatory call id. This is what
/// the dialect must *say* about it, where the call id is genuinely optional:
/// text dialects correlate results by tool name, and only native tool calling
/// has an id to correlate by.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    /// The tool that ran.
    pub name: String,
    /// Its output, already rendered to text.
    pub output: String,
    /// Whether the call succeeded. Text dialects surface this as a `status`
    /// attribute; the native dialect leaves the model to read the body.
    pub success: bool,
    /// The provider call id this answers, when the call had one.
    pub tool_call_id: Option<String>,
}

impl ToolOutcome {
    /// A successful outcome.
    pub fn ok(name: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            output: output.into(),
            success: true,
            tool_call_id: None,
        }
    }

    /// A failed outcome.
    pub fn failed(name: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            output: output.into(),
            success: false,
            tool_call_id: None,
        }
    }

    /// Correlates the outcome with a provider call id.
    pub fn with_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
}

/// One tool result as it is persisted in a transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultEntry {
    /// The call id this result answers.
    pub tool_call_id: String,
    /// The rendered output.
    pub content: String,
}

/// One durable transcript record.
///
/// This is the unit a dialect reads when replaying history onto the wire. The
/// three variants are the only shapes a tool-calling transcript needs: ordinary
/// chat, an assistant turn that asked for tools, and the results that answer it.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptEntry {
    /// An ordinary chat turn.
    Chat(DialectMessage),
    /// An assistant turn carrying structured tool calls.
    AssistantToolCalls {
        /// Visible text alongside the calls, if any.
        text: Option<String>,
        /// The calls the model made.
        tool_calls: Vec<NativeToolCall>,
        /// The model's thinking output, when the provider surfaced it.
        ///
        /// Replayed verbatim because thinking-mode APIs reject an assistant
        /// turn that carries `tool_calls` without it (DeepSeek returns a 400).
        reasoning_content: Option<String>,
        /// Host passthrough metadata.
        extra_metadata: Option<Value>,
    },
    /// The results answering the immediately preceding tool calls.
    ToolResults(Vec<ToolResultEntry>),
}

/// How the model is told to spell a tool call.
///
/// Drives the tool catalogue rendering in the system prompt, so it has to agree
/// with the dialect that will parse the model's answer — which is why the
/// dialect itself reports it, via
/// [`ToolDialect::tool_call_format`](super::ToolDialect::tool_call_format),
/// rather than the host configuring the two independently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// `tool_name[arg1|arg2|…]` — compact, positional.
    #[default]
    PFormat,
    /// JSON object inside a `<tool_call>` tag, with full schemas in the prompt.
    Json,
    /// The provider supplies structured calls; the catalogue is informational.
    Native,
}
