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

mod types;

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
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool(ToolMessage {
            tool_call_id: tool_call_id.into(),
            content: vec![ContentBlock::Text(content.into())],
        })
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
    /// (text, JSON, images, reasoning, provider extensions), for token
    /// estimation and context-window gating.
    ///
    /// Distinct from [`char_len`](Self::char_len), which counts only visible
    /// text: a transcript dominated by images, large tool-result JSON, or model
    /// reasoning under-counts badly under `char_len`, so compaction/trim would
    /// silently never trigger even as the real context window overflows. See
    /// [`ContentBlock::estimated_char_weight`].
    pub fn estimated_char_weight(&self) -> usize {
        let content = match self {
            Message::System(m) => &m.content,
            Message::User(m) => &m.content,
            Message::Assistant(m) => &m.content,
            Message::Tool(m) => &m.content,
        };
        content
            .iter()
            .map(ContentBlock::estimated_char_weight)
            .sum()
    }
}

#[cfg(test)]
mod test;
