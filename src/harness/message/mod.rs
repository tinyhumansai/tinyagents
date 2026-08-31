//! Rich internal message model.
//!
//! [`Message`] is the common currency that flows through every level of the
//! recursive runtime: the same typed value is what a parent agent sends into a
//! sub-agent, what a sub-graph node consumes, and what a REPL step inspects as a
//! runtime *value* rather than raw prompt text. Keeping the model structured
//! (typed [`ContentBlock`]s rather than strings) is what lets those recursive
//! hand-offs stay inspectable and lossless.
//!
//! See [`types`] for definitions. This module provides ergonomic constructors
//! and a [`Message::text`] accessor.

mod tokens;
mod types;

pub use tokens::*;
pub use types::*;

/// Approximate token-estimation weight of a single image content block, in
/// "characters" (the token estimator divides char weight by 4, so this is
/// ~1024 tokens per image).
///
/// Vision models tokenize an image into a roughly fixed count that is
/// independent of the encoded byte length, so this is a flat conservative
/// estimate rather than the (potentially huge, e.g. a base64 `data:` URI)
/// [`ImageRef::url`] length, which would wildly over-count.
const IMAGE_CHAR_WEIGHT: usize = 4 * 1024;

impl ContentBlock {
    /// Returns the text of this block if it is a [`ContentBlock::Text`].
    ///
    /// Reasoning blocks ([`ContentBlock::Thinking`] /
    /// [`ContentBlock::RedactedThinking`]) are intentionally *not* treated as
    /// text, so they never leak into visible assistant output via
    /// [`concat_text`] / [`Message::text`].
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Creates a [`ContentBlock::Thinking`] block with no signature.
    pub fn thinking(text: impl Into<String>) -> Self {
        ContentBlock::Thinking {
            text: text.into(),
            signature: None,
        }
    }

    /// Returns the reasoning text and optional signature if this is a
    /// [`ContentBlock::Thinking`] block.
    pub fn as_thinking(&self) -> Option<(&str, Option<&str>)> {
        match self {
            ContentBlock::Thinking { text, signature } => {
                Some((text.as_str(), signature.as_deref()))
            }
            _ => None,
        }
    }

    /// Returns `true` if this is a reasoning block ([`ContentBlock::Thinking`]
    /// or [`ContentBlock::RedactedThinking`]).
    pub fn is_reasoning(&self) -> bool {
        matches!(
            self,
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. }
        )
    }

    /// Approximate character weight of this block for token estimation.
    ///
    /// Unlike [`as_text`](Self::as_text) — which returns only visible
    /// [`Text`](Self::Text) so reasoning never leaks into assistant output —
    /// this accounts for *every* block that occupies model context: text,
    /// structured JSON, reasoning ([`Thinking`](Self::Thinking) /
    /// [`RedactedThinking`](Self::RedactedThinking)), provider extensions, and
    /// a flat [`IMAGE_CHAR_WEIGHT`] per image. It is used only by token
    /// budgeting / compaction gating, never by the visible-text accessors, so a
    /// transcript dominated by images, large tool-result JSON, or model
    /// reasoning no longer under-counts to near-zero and silently defeats
    /// summarization.
    pub fn estimated_char_weight(&self) -> usize {
        match self {
            ContentBlock::Text(text) => text.chars().count(),
            ContentBlock::Json(value) => value.to_string().chars().count(),
            ContentBlock::Image(_) => IMAGE_CHAR_WEIGHT,
            ContentBlock::Thinking { text, .. } => text.chars().count(),
            ContentBlock::RedactedThinking { data } => data.chars().count(),
            ContentBlock::ProviderExtension(value) => value.to_string().chars().count(),
            // A zero-width marker weighs nothing. It must not contribute here:
            // this figure gates compaction, and inflating it would trigger
            // summarization earlier for a block that carries no information.
            ContentBlock::CacheBreakpoint => 0,
        }
    }
}

/// Concatenates the text of all [`ContentBlock::Text`] blocks in `content`.
fn concat_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(ContentBlock::as_text)
        .collect::<Vec<_>>()
        .join("")
}

impl Message {
    /// Creates a system message from text.
    pub fn system(content: impl Into<String>) -> Self {
        Message::System(SystemMessage {
            content: vec![ContentBlock::Text(content.into())],
        })
    }

    /// Creates a user message from text.
    pub fn user(content: impl Into<String>) -> Self {
        Message::User(UserMessage {
            content: vec![ContentBlock::Text(content.into())],
        })
    }

    /// Creates an assistant message from text, with no tool calls or usage.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant(AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(content.into())],
            tool_calls: Vec::new(),
            usage: None,
        })
    }

    /// Creates a tool result message for the given tool call id.
    ///
    /// Leaves [`ToolMessage::trusted_verbatim`] unset. Use
    /// [`Message::tool_from_result`] to fold a real
    /// [`crate::harness::tool::ToolResult`], which carries the flag across.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool(ToolMessage {
            tool_call_id: tool_call_id.into(),
            content: vec![ContentBlock::Text(content.into())],
            trusted_verbatim: false,
            artifact: None,
        })
    }

    /// Folds a [`crate::harness::tool::ToolResult`] into the transcript message
    /// that answers its call.
    ///
    /// Preferred over [`Message::tool`] on the agent-loop path: a `ToolResult`
    /// carries structured metadata the message shape would otherwise drop, and
    /// [`ToolMessage::trusted_verbatim`] is the part a host must not lose — it
    /// is what tells the host this content may not be reshaped.
    ///
    /// [`ToolResult::raw`][crate::harness::tool::ToolResult::raw] is carried
    /// across into [`ToolMessage::artifact`], so a tool can return a small
    /// model-facing summary in `content` and still leave the full structured
    /// payload reachable from `run.messages` (LangChain's
    /// `response_format="content_and_artifact"`). The artifact is host-side
    /// only — provider conversion serialises [`Message::text`], never the
    /// artifact.
    pub fn tool_from_result(result: &crate::harness::tool::ToolResult) -> Self {
        Message::Tool(ToolMessage {
            tool_call_id: result.call_id.clone(),
            content: vec![ContentBlock::Text(result.content.clone())],
            trusted_verbatim: result.is_trusted_verbatim(),
            artifact: result.raw.clone(),
        })
    }

    /// Returns the structured artifact carried by a tool message, if any.
    ///
    /// `None` for every non-tool message and for tool messages whose producing
    /// [`ToolResult`][crate::harness::tool::ToolResult] set no `raw` payload.
    /// See [`ToolMessage::artifact`].
    pub fn artifact(&self) -> Option<&serde_json::Value> {
        match self {
            Message::Tool(m) => m.artifact.as_ref(),
            _ => None,
        }
    }

    /// Returns the concatenated text of all text content blocks.
    pub fn text(&self) -> String {
        match self {
            Message::System(m) => concat_text(&m.content),
            Message::User(m) => concat_text(&m.content),
            Message::Assistant(m) => concat_text(&m.content),
            Message::Tool(m) => concat_text(&m.content),
        }
    }

    /// Returns the total number of Unicode scalar values across all text content
    /// blocks, without allocating the concatenated string.
    ///
    /// Equivalent to `self.text().chars().count()` but avoids the intermediate
    /// `String` allocation, which matters on hot paths such as token estimation
    /// over a whole transcript.
    pub fn char_len(&self) -> usize {
        let content = match self {
            Message::System(m) => &m.content,
            Message::User(m) => &m.content,
            Message::Assistant(m) => &m.content,
            Message::Tool(m) => &m.content,
        };
        content
            .iter()
            .filter_map(ContentBlock::as_text)
            .map(|t| t.chars().count())
            .sum()
    }

    /// Approximate character weight of the message across *all* content blocks
    /// (text, JSON, images, reasoning, provider extensions) **plus the
    /// structural payload that lives outside `content`**: an assistant
    /// message's [`tool_calls`][AssistantMessage::tool_calls] and a tool
    /// message's [`tool_call_id`][ToolMessage::tool_call_id].
    ///
    /// Distinct from [`char_len`](Self::char_len), which counts only visible
    /// text: a transcript dominated by images, large tool-result JSON, or model
    /// reasoning under-counts badly under `char_len`, so compaction/trim would
    /// silently never trigger even as the real context window overflows. See
    /// [`ContentBlock::estimated_char_weight`].
    ///
    /// # Why tool calls must be counted here
    ///
    /// An assistant turn that *only* calls tools carries **empty `content`**:
    /// the tool name and its argument JSON — often the largest part of the turn
    /// — live in `tool_calls`. Counting `content` alone estimated such a
    /// message at zero, so a 50-turn tool-driven run whose assistant messages
    /// each carry a 2 KB argument blob estimated to near-nothing and never
    /// tripped [`SummarizationPolicy::should_summarize`][crate::harness::summarization::SummarizationPolicy::should_summarize],
    /// letting the window overflow uncompacted — exactly the failure this
    /// estimator exists to prevent.
    ///
    /// Mirrors LangChain's `count_tokens_approximately`, which adds
    /// `repr(tool_calls)` for AI messages and the `tool_call_id` for tool
    /// messages. The role label and per-message overhead are *not* added here;
    /// they belong to the message-level counters in
    /// [`crate::harness::message::count_tokens_approximately`].
    ///
    /// The [`ToolMessage::artifact`] payload is deliberately **not** counted: it
    /// never reaches the provider, so it occupies no context window.
    pub fn estimated_char_weight(&self) -> usize {
        let content = match self {
            Message::System(m) => &m.content,
            Message::User(m) => &m.content,
            Message::Assistant(m) => &m.content,
            Message::Tool(m) => &m.content,
        };
        let content_weight: usize = content
            .iter()
            .map(ContentBlock::estimated_char_weight)
            .sum();

        let structural_weight = match self {
            Message::Assistant(m) => tool_calls_char_weight(&m.tool_calls),
            Message::Tool(m) => m.tool_call_id.chars().count(),
            _ => 0,
        };

        content_weight + structural_weight
    }
}

/// Approximate character weight of an assistant message's tool-call array.
///
/// Serialises the calls to JSON (the closest analogue of LangChain's
/// `repr(tool_calls)`) and counts characters. Falls back to a per-call estimate
/// from the name plus the raw argument value when serialisation fails, so the
/// weight is never silently zero.
fn tool_calls_char_weight(tool_calls: &[crate::harness::tool::ToolCall]) -> usize {
    if tool_calls.is_empty() {
        return 0;
    }
    match serde_json::to_string(tool_calls) {
        Ok(rendered) => rendered.chars().count(),
        Err(_) => tool_calls
            .iter()
            .map(|call| {
                call.name.chars().count()
                    + call.id.chars().count()
                    + call.arguments.to_string().chars().count()
            })
            .sum(),
    }
}

#[cfg(test)]
mod test;
