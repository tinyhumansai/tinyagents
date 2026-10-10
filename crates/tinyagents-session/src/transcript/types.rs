//! Public transcript domain types: per-message usage, the `_meta` header,
//! the model-context and display projections, and thread usage summaries.

use serde::{Deserialize, Serialize};
use tinytools_agent::dialect::{
    ContentPart, NativeToolCall, encode_assistant_envelope, encode_tool_envelope, join_image_parts,
    parse_assistant_envelope, parse_tool_envelope, split_image_parts,
};

/// A provider-neutral tool call as recorded in a durable transcript.
///
/// Arguments deliberately remain their original string.  Parsing them into a
/// provider or inference representation here would lose malformed-but-valid
/// streamed payloads and provider extension data needed by a later turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<serde_json::Value>,
}

/// A durable media location. Inline bytes are deliberately absent: hosts resolve
/// these references only on ephemeral provider requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptMediaRef {
    /// A path relative to the agent's acting workspace, or a host-approved path.
    Path { path: String },
    /// An external location. Hosts apply fetch policy before resolving it.
    Url { url: String },
}

impl TranscriptMediaRef {
    fn location(&self) -> &str {
        match self {
            Self::Path { path } => path,
            Self::Url { url } => url,
        }
    }
}

/// One ordered part of a user row that mixes text and media.
///
/// Serialized as `{"type":"text","text":..}` / `{"type":"image","url":..}`,
/// which is also the on-disk `parts` shape of a typed `user_parts` line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptPart {
    /// Literal text, verbatim.
    Text { text: String },
    /// An image reference (a `data:` URI, an `http(s)` URL or a path).
    Image { url: String },
    /// Audio whose original bytes remain at the referenced location.
    Audio {
        source: TranscriptMediaRef,
        mime_type: String,
    },
    /// Video whose original bytes remain at the referenced location.
    Video {
        source: TranscriptMediaRef,
        mime_type: String,
    },
    /// A document whose original bytes remain at the referenced location.
    Document {
        source: TranscriptMediaRef,
        mime_type: String,
    },
}

/// The legacy string a row was lifted from ([`TranscriptMessage::normalized`]),
/// remembered so [`TranscriptMessage::legacy_content`] can hand back the exact
/// original bytes (key order, `null` content, embedded reasoning) while the row
/// is still semantically what that string parses to.
///
/// Equal to every other value: it is a memo, not part of the row's identity.
/// It is serialized (only when present) so the original bytes survive a serde
/// round trip.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LegacyText(Option<String>);

impl LegacyText {
    fn is_none(&self) -> bool {
        self.0.is_none()
    }
}

impl PartialEq for LegacyText {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// The host-visible marker prefix an image part renders as in display text.
const DISPLAY_IMAGE_PREFIX: &str = "[IMAGE:";

/// A durable, provider-neutral message record.
///
/// This intentionally is not an inference message: transcript persistence is
/// an on-disk compatibility boundary, and callers adapt it to their runtime
/// message dialect at the host boundary.
///
/// # Typed rows
///
/// `content` is always the row's **plain text**. Structure that a flat
/// `{role, content}` row used to smuggle inside `content` as a string
/// (a native tool-call envelope, a tool-result envelope, inline image
/// markers) lives in typed fields instead:
///
/// - an `assistant` row that made native tool calls carries them in
///   [`tool_calls`](Self::tool_calls), with the visible prose in `content`;
/// - a `tool` row answering a native call carries that call's id in
///   [`tool_call_id`](Self::tool_call_id), with the output in `content`;
/// - a `user` row that mixes text and images carries the ordered
///   [`parts`](Self::parts), with the concatenated text in `content`.
///
/// Rows read from older transcripts are normalized into this shape by the
/// readers ([`Self::normalized`]); [`Self::legacy_content`] rebuilds the old
/// string for the few places that must keep writing it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(into = "TranscriptMessageWire")]
pub struct TranscriptMessage {
    #[serde(default)]
    pub id: Option<String>,
    pub role: String,
    pub content: String,
    /// Native tool calls an `assistant` row made, in order. Empty otherwise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<TranscriptToolCall>,
    /// The call a `tool` row answers (native tool calling), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// The ordered text and image parts of a `user` row that carries images.
    /// `None` for a text-only row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<TranscriptPart>>,
    /// The legacy string this row was lifted from, if any; see [`LegacyText`].
    /// Serialized (only while it still describes the row; see
    /// [`TranscriptMessageWire`]) so a store that round-trips the row through
    /// serde keeps the original bytes of a non-canonical envelope.
    #[serde(default)]
    pub legacy: LegacyText,
    #[serde(default)]
    pub extra_metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub cache_breakpoints: Vec<usize>,
    /// Usage and provider provenance already associated with this durable row.
    /// The JSONL codec writes this as first-class line fields; keeping it on
    /// the neutral record makes read → append replay lossless without using a
    /// host metadata namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_usage: Option<TurnUsage>,
    /// Stable request/turn correlation recorded with this row, if the host has
    /// one. This is deliberately opaque to the session crate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Whether `request_id` is authoritative even when it is absent. Readers
    /// set this for replayed rows so a resumed write does not attribute an
    /// earlier request-less row to the current turn.
    #[serde(default)]
    pub preserve_request_id: bool,
    /// Whether this row is display-only because a streamed answer stopped
    /// before completion.
    #[serde(default)]
    pub interrupted: bool,
    /// Tool-execution failure display data associated with this row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_failure: Option<ToolFailure>,
}

impl TranscriptMessage {
    /// Builds a bare message with `role` and `content` set and every other
    /// field at its default (no usage, no id, not interrupted).
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: None,
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
            legacy: LegacyText::default(),
            extra_metadata: None,
            cache_breakpoints: Vec::new(),
            turn_usage: None,
            request_id: None,
            preserve_request_id: false,
            interrupted: false,
            tool_failure: None,
        }
    }

    /// Shorthand for [`TranscriptMessage::new`] with `role = "system"`.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Shorthand for [`TranscriptMessage::new`] with `role = "user"`.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Shorthand for [`TranscriptMessage::new`] with `role = "assistant"`.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// Shorthand for [`TranscriptMessage::new`] with `role = "tool"`.
    pub fn tool(content: impl Into<String>) -> Self {
        Self::new("tool", content)
    }

    /// Sets the row's id (a tool row's id is the call id it answers).
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// An `assistant` row that made native tool calls; `text` is the visible
    /// prose (possibly empty).
    pub fn assistant_with_calls(
        text: impl Into<String>,
        tool_calls: Vec<TranscriptToolCall>,
    ) -> Self {
        let mut row = Self::assistant(text);
        row.tool_calls = tool_calls;
        row
    }

    /// A `tool` row carrying the output of the native call `tool_call_id`.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        let mut row = Self::tool(content);
        row.tool_call_id = Some(tool_call_id.into());
        row
    }

    /// A `user` row from ordered text/media parts. `content` becomes the
    /// concatenated text parts. A parts list with only text is just text.
    pub fn user_with_parts(parts: Vec<TranscriptPart>) -> Self {
        let text = parts_text(&parts);
        let mut row = Self::user(text);
        if parts
            .iter()
            .any(|part| !matches!(part, TranscriptPart::Text { .. }))
        {
            row.parts = Some(parts);
        }
        row
    }

    /// Builds a normalized row from a legacy `{role, content}` pair whose
    /// `content` may carry a string-encoded envelope or image marker.
    pub fn from_legacy(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(role, content).normalized()
    }

    /// Whether the row carries any typed structure.
    #[must_use]
    pub fn is_typed(&self) -> bool {
        !self.tool_calls.is_empty() || self.tool_call_id.is_some() || self.parts.is_some()
    }

    /// Lifts a legacy string-encoded structure out of `content` into the typed
    /// fields, leniently: an assistant native envelope (any JSON object with a
    /// non-empty `tool_calls` array), a tool-result envelope (`tool_call_id`),
    /// or a user row with `[OH_IMAGE:<url>]` markers. A row that already
    /// carries typed fields, or whose `content` is none of those, is returned
    /// unchanged. An envelope's own `reasoning_content` is not lifted (the host
    /// keeps reasoning in `extra_metadata`).
    #[must_use]
    pub fn normalized(mut self) -> Self {
        if self.is_typed() {
            return self;
        }
        let original = self.content.clone();
        match self.role.as_str() {
            "assistant" => {
                if let Some(envelope) = parse_assistant_envelope(&self.content) {
                    self.content = envelope.content;
                    self.tool_calls = envelope
                        .tool_calls
                        .into_iter()
                        .map(TranscriptToolCall::from)
                        .collect();
                }
            }
            "tool" => {
                if let Some((tool_call_id, content)) = parse_tool_envelope(&self.content) {
                    self.content = content;
                    self.tool_call_id = Some(tool_call_id);
                }
            }
            "user" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&self.content)
                    && let Some(parts) = value.get("_tinyagents_media_parts")
                    && let Ok(parts) = serde_json::from_value::<Vec<TranscriptPart>>(parts.clone())
                    && parts.iter().any(|part| {
                        matches!(
                            part,
                            TranscriptPart::Audio { .. }
                                | TranscriptPart::Video { .. }
                                | TranscriptPart::Document { .. }
                        )
                    })
                    && encode_media_parts(&parts) == self.content
                {
                    self.content = parts_text(&parts);
                    self.parts = Some(parts);
                    self.legacy = LegacyText(Some(original));
                    return self;
                }
                let parts = split_image_parts(&self.content);
                if parts
                    .iter()
                    .any(|part| matches!(part, ContentPart::Image(_)))
                {
                    let parts: Vec<TranscriptPart> = parts
                        .into_iter()
                        .map(|part| match part {
                            ContentPart::Text(text) => TranscriptPart::Text { text },
                            ContentPart::Image(url) => TranscriptPart::Image { url },
                        })
                        .collect();
                    self.content = parts_text(&parts);
                    self.parts = Some(parts);
                }
            }
            _ => {}
        }
        if self.is_typed() {
            self.legacy = LegacyText(Some(original));
        }
        self
    }

    /// The string a flat `{role, content}` row would have held for this row:
    /// the native envelope for an assistant row with calls (`content` as its
    /// text), the tool-result envelope for a tool row with a call id, and
    /// `[OH_IMAGE:<url>]` markers for image-only media rows. Audio/video/document
    /// rows use a `_tinyagents_media_parts` JSON compatibility envelope; old
    /// binaries cannot interpret those new media types. Everything else is
    /// `content` as is.
    ///
    /// For the compatibility adapters that must keep writing the old string
    /// (host files shared with older binaries, journals). A reader of the
    /// result gets the typed row back through [`Self::normalized`].
    #[must_use]
    pub fn legacy_content(&self) -> String {
        // A row lifted from a legacy string hands that string back untouched
        // for as long as it still means the same thing.
        if let Some(raw) = self.legacy.0.as_deref() {
            let mut lifted = Self::new(self.role.clone(), raw);
            lifted = lifted.normalized();
            if lifted.same_structure_as(self) {
                return raw.to_string();
            }
        }
        if self.role == "assistant" && !self.tool_calls.is_empty() {
            let calls: Vec<NativeToolCall> =
                self.tool_calls.iter().cloned().map(Into::into).collect();
            return encode_assistant_envelope(Some(&self.content), &calls, None);
        }
        if self.role == "tool"
            && let Some(id) = self.tool_call_id.as_deref()
        {
            return encode_tool_envelope(id, &self.content);
        }
        if let Some(parts) = self.parts.as_deref() {
            if parts.iter().any(|part| {
                matches!(
                    part,
                    TranscriptPart::Audio { .. }
                        | TranscriptPart::Video { .. }
                        | TranscriptPart::Document { .. }
                )
            }) {
                return encode_media_parts(parts);
            }
            let parts: Vec<ContentPart> = parts
                .iter()
                .map(|part| match part {
                    TranscriptPart::Text { text } => ContentPart::Text(text.clone()),
                    TranscriptPart::Image { url } => ContentPart::Image(url.clone()),
                    TranscriptPart::Audio { .. }
                    | TranscriptPart::Video { .. }
                    | TranscriptPart::Document { .. } => {
                        unreachable!("new media uses the typed compatibility envelope")
                    }
                })
                .collect();
            return join_image_parts(&parts);
        }
        self.content.clone()
    }

    /// The row's text as a person reads it: `content`, except that a user row's
    /// media parts render in place as `[IMAGE:<url>]`, `[AUDIO:<path>]`, etc.
    #[must_use]
    pub fn display_content(&self) -> String {
        let Some(parts) = self.parts.as_deref() else {
            return self.content.clone();
        };
        let mut out = String::new();
        for part in parts {
            match part {
                TranscriptPart::Text { text } => out.push_str(text),
                TranscriptPart::Image { url } => {
                    out.push_str(DISPLAY_IMAGE_PREFIX);
                    out.push_str(url);
                    out.push(']');
                }
                TranscriptPart::Audio { source, .. }
                | TranscriptPart::Video { source, .. }
                | TranscriptPart::Document { source, .. } => {
                    let label = match part {
                        TranscriptPart::Audio { .. } => "AUDIO",
                        TranscriptPart::Video { .. } => "VIDEO",
                        _ => "DOCUMENT",
                    };
                    out.push_str(&format!("[{label}:{}]", source.location()));
                }
            }
        }
        out
    }

    /// Whether two rows are the same model-visible row: role, text, id and the
    /// typed structure. Usage, metadata and correlation ids are excluded, as
    /// they are enriched between the in-memory and the persisted row.
    #[must_use]
    pub fn same_row_as(&self, other: &Self) -> bool {
        // A legacy string row and the typed row lifted from it are the same
        // row; compare them in one normal form.
        let left = self.clone().normalized();
        let right = other.clone().normalized();
        // Two rows lifted from different legacy strings are different stored
        // rows even when the lifted fields agree (embedded reasoning or
        // provider keys differ), so a redaction is never taken for a no-op.
        let same_stored_text = match (left.live_legacy(), right.live_legacy()) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        };
        left.id == right.id && left.same_structure_as(&right) && same_stored_text
    }

    /// The legacy string this row was lifted from, while the row still means
    /// what that string parses to.
    fn live_legacy(&self) -> Option<&str> {
        let raw = self.legacy.0.as_deref()?;
        let lifted = Self::new(self.role.clone(), raw).normalized();
        lifted.same_structure_as(self).then_some(raw)
    }

    /// The row as it was *before* normalization lifted it: `content` holds the
    /// original legacy string and the typed fields are empty. `None` when the
    /// row was not lifted from a legacy string, or has been edited since (its
    /// structure no longer matches what that string parses to).
    ///
    /// The JSONL writer stores such a row from this form, so a non-canonical
    /// envelope (embedded reasoning, extra keys) survives a read and rewrite
    /// byte for byte instead of being re-encoded from the lifted fields.
    pub(crate) fn unlifted(&self) -> Option<Self> {
        let raw = self.live_legacy()?;
        let mut row = self.clone();
        row.content = raw.to_string();
        row.tool_calls = Vec::new();
        row.tool_call_id = None;
        row.parts = None;
        row.legacy = LegacyText::default();
        Some(row)
    }

    /// [`Self::same_row_as`] without the id.
    fn same_structure_as(&self, other: &Self) -> bool {
        self.role == other.role
            && self.content == other.content
            && self.tool_calls == other.tool_calls
            && self.tool_call_id == other.tool_call_id
            && self.parts == other.parts
    }
}

/// The serialized form of a [`TranscriptMessage`].
///
/// Identical to the row's own fields, except that the `legacy` memo is written
/// only while the row still means what that string parses to. An edit (a
/// redaction, a replacement) clears it, so the superseded text is never
/// serialized alongside the new content.
#[derive(Serialize)]
struct TranscriptMessageWire {
    id: Option<String>,
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<TranscriptToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parts: Option<Vec<TranscriptPart>>,
    #[serde(skip_serializing_if = "LegacyText::is_none")]
    legacy: LegacyText,
    extra_metadata: Option<serde_json::Value>,
    cache_breakpoints: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_usage: Option<TurnUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    preserve_request_id: bool,
    interrupted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_failure: Option<ToolFailure>,
}

impl From<TranscriptMessage> for TranscriptMessageWire {
    fn from(row: TranscriptMessage) -> Self {
        let legacy = if row.unlifted().is_some() {
            row.legacy
        } else {
            LegacyText::default()
        };
        Self {
            id: row.id,
            role: row.role,
            content: row.content,
            tool_calls: row.tool_calls,
            tool_call_id: row.tool_call_id,
            parts: row.parts,
            legacy,
            extra_metadata: row.extra_metadata,
            cache_breakpoints: row.cache_breakpoints,
            turn_usage: row.turn_usage,
            request_id: row.request_id,
            preserve_request_id: row.preserve_request_id,
            interrupted: row.interrupted,
            tool_failure: row.tool_failure,
        }
    }
}

/// The concatenated text parts.
fn encode_media_parts(parts: &[TranscriptPart]) -> String {
    serde_json::json!({"_tinyagents_media_parts": parts}).to_string()
}

fn parts_text(parts: &[TranscriptPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            TranscriptPart::Text { text } => Some(text.as_str()),
            TranscriptPart::Image { .. }
            | TranscriptPart::Audio { .. }
            | TranscriptPart::Video { .. }
            | TranscriptPart::Document { .. } => None,
        })
        .collect()
}

impl From<NativeToolCall> for TranscriptToolCall {
    fn from(call: NativeToolCall) -> Self {
        Self {
            id: call.id,
            name: call.name,
            arguments: call.arguments,
            extra_content: call.extra_content,
        }
    }
}

impl From<TranscriptToolCall> for NativeToolCall {
    fn from(call: TranscriptToolCall) -> Self {
        Self {
            id: call.id,
            name: call.name,
            arguments: call.arguments,
            extra_content: call.extra_content,
        }
    }
}

/// Provider-neutral failure status for a durable tool-result row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolFailure {
    #[serde(default)]
    pub failed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Per-message usage figures attributed to the last assistant turn.
///
/// `input`, `output` and `cached_input` are the turn's *spend*: they sum every
/// provider call the turn made, so a turn of 70 tool rounds reports roughly 70
/// times its context. How full the context window was is a different figure,
/// the size of one request, and lives in `last_call_input` /
/// `last_call_output`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MessageUsage {
    pub input: u64,
    pub output: u64,
    pub cached_input: u64,
    #[serde(default)]
    pub context_window: u64,
    pub cost_usd: f64,
    /// Input tokens of the turn's final provider call: the context the model
    /// held when it last answered. `0` when the writer predates the field.
    #[serde(default)]
    pub last_call_input: u64,
    /// Output tokens of the turn's final provider call. `0` when the writer
    /// predates the field.
    #[serde(default)]
    pub last_call_output: u64,
    /// Where `cost_usd` came from. `None` on records written before the field
    /// existed: their cost may be a host's guessed rate, so a reader should
    /// not present it as spend without re-pricing it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_source: Option<UsageCostSource>,
}

/// Provenance of a recorded cost, least certain last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageCostSource {
    /// Every call's cost is what the provider reported billing.
    Charged,
    /// At least one call was priced from published list rates.
    Estimated,
    /// At least one call had no known cost; `cost_usd` is incomplete.
    Unknown,
}

/// Usage + provenance for one provider response, attached to the last
/// assistant message in a turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TurnUsage {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    pub usage: MessageUsage,
    /// RFC-3339 timestamp of the response.
    #[serde(default)]
    pub ts: String,
    /// Raw reasoning/thinking content returned by thinking models. This is
    /// persisted as metadata so the later transcript view can show the model's
    /// thoughts without depending on the live stream still being open.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Native tool calls emitted in this provider response, if any. Text-mode
    /// calls remain present in `content` as the raw markup the model emitted.
    #[serde(default)]
    pub tool_calls: Vec<TranscriptToolCall>,
    /// One-based engine iteration for this provider response.
    #[serde(default)]
    pub iteration: u32,
}

/// Schema version stamped on the `_meta` header line. Bumped when the JSONL
/// record shape changes in a way future readers may need to branch on. `0`
/// (absent) denotes pre-append-only files written before this field existed.
pub const TRANSCRIPT_SCHEMA_VERSION: u32 = 1;

/// Metadata header for a session transcript file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptMeta {
    pub agent_name: String,
    /// Canonical registry id for the agent that produced this transcript.
    /// `agent_name` may be per-thread renamed for file names; this remains the
    /// stable archetype id when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Coarse runtime kind (`root`, `subagent`, `extractor`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<String>,
    pub dispatcher: String,
    /// Provider label used for the most recent recorded response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model id used for the most recent recorded response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub created: String,
    pub updated: String,
    pub turn_count: usize,
    /// Number of leading model messages frozen as the session prompt in this
    /// generation. `None` on older transcripts that predate this boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_message_count: Option<usize>,
    /// Cumulative input tokens across all provider calls this session.
    pub input_tokens: u64,
    /// Cumulative output tokens across all provider calls this session.
    pub output_tokens: u64,
    /// Cumulative input tokens served from the KV cache.
    pub cached_input_tokens: u64,
    /// Cumulative amount charged in USD.
    pub charged_amount_usd: f64,
    /// Caller-owned logical thread identifier. Hosts may forward it to a
    /// compatible inference endpoint for request grouping or cache affinity.
    /// `None` for sessions that are not thread-scoped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Sub-agent task id, when this transcript belongs to a spawned worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Durable identity of the session this transcript holds, as produced by
    /// [`session_stem`](crate::transcript::session_stem). Before this existed
    /// a transcript's only identity was its filename, so nothing could tell
    /// two files of one conversation apart from two unrelated ones. `None` on
    /// transcripts written before session identity landed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The session this one succeeded when a compaction sealed it. Compaction
    /// never rewrites history in place: it opens the next generation and
    /// points back here, so the whole conversation stays recoverable by
    /// walking the chain even though the model only sees the head.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
}

/// A parsed session transcript: metadata + exact message array.
#[derive(Debug, Clone)]
pub struct SessionTranscript {
    pub meta: TranscriptMeta,
    pub messages: Vec<TranscriptMessage>,
    /// The model-visible tool declarations most recently recorded for this
    /// session (the last `{"kind":"tools"}` record), exactly as they were
    /// sent. `None` for transcripts written before tool recording existed or
    /// by a writer that records none. Opaque JSON here: the runtime owns its
    /// shape (a list of tool specs).
    pub tools: Option<serde_json::Value>,
}

// ── Display read types ───────────────────────────────────────────────

/// One message in a display projection, carrying the turn-boundary + partial
/// flags the model-context [`SessionTranscript`] discards.
#[derive(Debug, Clone)]
pub struct DisplayMessage {
    pub message: TranscriptMessage,
    /// `true` when this is an interrupted partial answer (display only).
    pub interrupted: bool,
    /// Turn boundary marker (`request_id`), when stamped.
    pub request_id: Option<String>,
    pub iteration: Option<u32>,
    pub ts: Option<String>,
    /// Usage/provenance for assistant messages that carried it.
    pub turn_usage: Option<TurnUsage>,
    /// Raw reasoning/thinking captured for this line, when present. Mirrors the
    /// line's `reasoning_content` directly so it survives even on lines without
    /// full turn-usage provenance (e.g. an interrupted partial, which carries no
    /// provider/model/usage). Prefer this over digging into [`Self::turn_usage`]
    /// for display: it is populated from `turn_usage.reasoning_content` too.
    pub reasoning_content: Option<String>,
    /// `true` when this is a **failed** tool-result line (`ToolResult::is_error`
    /// at execution time). The display projection renders an error tool row
    /// instead of success. Always `false` for non-tool lines and legacy files.
    pub failure: bool,
    /// Optional short reason for a failed tool call (present only with
    /// `failure: true`).
    pub failure_detail: Option<String>,
    /// Who delivered this line out of band, when it was written by
    /// [`append_background_message`](super::append_background_message)
    /// rather than by a live turn. `None` for every turn-written line.
    pub background: Option<BackgroundOrigin>,
}

/// A compaction marker in a display projection.
#[derive(Debug, Clone)]
pub struct CompactionMarker {
    /// The reduced message set this compaction installed as the new context.
    pub replacement: Vec<DisplayMessage>,
    pub ts: Option<String>,
    pub request_id: Option<String>,
}

/// One record in a display projection, in file order.
#[derive(Debug, Clone)]
pub enum DisplayRecord {
    Message(Box<DisplayMessage>),
    Compaction(CompactionMarker),
}

/// A display projection of a transcript: **all** records, including
/// pre-compaction history, compaction markers, and interrupted partials.
#[derive(Debug, Clone)]
pub struct DisplaySessionTranscript {
    pub meta: TranscriptMeta,
    pub records: Vec<DisplayRecord>,
}

/// Aggregated token/cost usage for a chat thread, summed across **all** of the
/// thread's root session transcripts (a thread reopened across days/restarts
/// produces several files). `last_turn_*`, `model`, and `updated` come from the
/// newest transcript so the UI can render a context-window gauge for the most
/// recent turn. Returns `None` when no transcript exists yet (a brand-new
/// thread with no completed turns).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThreadUsageSummary {
    /// Orchestrator (parent) token totals — the root transcript(s) only. Root
    /// transcripts never include sub-agent calls (those go to a separate
    /// observer + their own `__` transcript files); see [`Self::subagents`].
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cost_usd: f64,
    pub turn_count: usize,
    /// Input/output tokens of the most recent assistant turn (context gauge).
    pub last_turn_input_tokens: u64,
    pub last_turn_output_tokens: u64,
    /// Model that served the most recent turn, if recorded.
    pub model: Option<String>,
    /// RFC-3339 `updated` of the newest transcript.
    pub updated: String,
    /// Per-archetype sub-agent spend, reconstructed from the thread's `__`
    /// sub-agent transcripts (grouped by `agent_name`).
    pub subagents: Vec<SubagentArchetypeUsage>,
}

/// One sub-agent archetype's summed spend within a thread (e.g. all `coder`
/// runs). `model` is the model that served one of its runs, used to price it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubagentArchetypeUsage {
    pub agent_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    /// How many sub-agent runs of this archetype contributed.
    pub runs: usize,
    pub model: Option<String>,
}

// ── Background appends ───────────────────────────────────────────────

/// Where an out-of-band transcript line came from, recorded on the line as an
/// additive top-level `background` field.
///
/// Written by [`append_background_message`](super::append_background_message).
/// A reader that predates the field keeps the line as an ordinary message (the
/// field lands in the line's unknown-field catch-all), so old binaries resume
/// the delivered message exactly like a new one does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackgroundOrigin {
    /// Caller-chosen key that makes the append idempotent within one
    /// generation (for a scheduled job, its run id).
    pub idempotency_key: String,
    /// Opaque caller provenance, e.g.
    /// `{"kind":"cron","job_id":"…","run_id":"…"}`. Never interpreted here.
    #[serde(default)]
    pub provenance: serde_json::Value,
}

/// Options for [`append_background_message`](super::append_background_message).
#[derive(Debug, Clone, PartialEq)]
pub struct BackgroundAppend {
    /// Idempotency key; must be non-empty. See [`BackgroundOrigin::idempotency_key`].
    pub idempotency_key: String,
    /// Opaque provenance stored on the line. See [`BackgroundOrigin::provenance`].
    pub provenance: serde_json::Value,
    /// The head generation the caller expects to append to. `None` appends to
    /// whatever the head is when the append runs.
    pub expected_generation: Option<u32>,
}

impl BackgroundAppend {
    /// Options with no expected generation.
    pub fn new(idempotency_key: impl Into<String>, provenance: serde_json::Value) -> Self {
        Self {
            idempotency_key: idempotency_key.into(),
            provenance,
            expected_generation: None,
        }
    }

    /// Requires the session head to still be `generation` when the append runs.
    #[must_use]
    pub fn expecting_generation(mut self, generation: u32) -> Self {
        self.expected_generation = Some(generation);
        self
    }
}

/// What [`append_background_message`](super::append_background_message) did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundAppendOutcome {
    /// The message was appended to head generation `generation`.
    Appended { generation: u32 },
    /// Head generation `generation` already holds a line with this
    /// idempotency key; nothing was written.
    Duplicate { generation: u32 },
    /// The head is `head`, not the caller's `expected` generation; nothing was
    /// written.
    StaleGeneration { expected: u32, head: u32 },
    /// The session has no transcript yet; nothing was written (and no
    /// transcript was created).
    NoSession,
}
