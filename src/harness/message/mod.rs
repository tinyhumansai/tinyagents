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

impl ContentBlock {
    /// Returns the text of this block if it is a [`ContentBlock::Text`].
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text(text) => Some(text),
            _ => None,
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
}

#[cfg(test)]
mod test;
