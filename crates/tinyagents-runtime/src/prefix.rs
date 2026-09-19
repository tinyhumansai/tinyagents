use tinyinference_llm::message::Message;

/// Immutable messages which remain at the front of a session's history.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefixSnapshot {
    messages: Vec<Message>,
}

impl PrefixSnapshot {
    /// Captures the prefix once, before turn history starts growing.
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Returns the captured messages in their original order.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}
