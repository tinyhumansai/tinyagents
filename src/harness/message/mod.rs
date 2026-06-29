//! Rich internal message model.
//!
//! See [`types`] for definitions. This module provides ergonomic constructors,
//! a [`Message::text`] accessor, and a bridge to and from the simple top-level
//! [`crate::chat::ChatMessage`].

mod types;

use crate::chat::{ChatMessage, ChatRole};

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

    /// Bridges this message back to a simple top-level [`ChatMessage`].
    ///
    /// Tool messages carry the tool call id in the `name` field so the simple
    /// type can still round-trip the correlation id.
    pub fn to_chat(&self) -> ChatMessage {
        match self {
            Message::System(_) => ChatMessage::system(self.text()),
            Message::User(_) => ChatMessage::user(self.text()),
            Message::Assistant(_) => ChatMessage::assistant(self.text()),
            Message::Tool(m) => ChatMessage::tool(m.tool_call_id.clone(), self.text()),
        }
    }
}

impl From<ChatMessage> for Message {
    fn from(msg: ChatMessage) -> Self {
        match msg.role {
            ChatRole::System => Message::system(msg.content),
            ChatRole::User => Message::user(msg.content),
            ChatRole::Assistant => Message::assistant(msg.content),
            ChatRole::Tool => Message::tool(msg.name.unwrap_or_default(), msg.content),
        }
    }
}

#[cfg(test)]
mod test;
