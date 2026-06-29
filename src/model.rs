use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Result, chat::ChatMessage};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl ModelRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub message: ChatMessage,
}

impl ModelResponse {
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            message: ChatMessage::assistant(content),
        }
    }
}

#[async_trait]
pub trait ChatModel<State>: Send + Sync {
    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse>;
}
