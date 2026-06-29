//! Harness model layer.
//!
//! See [`types`] for definitions. This module provides builder methods on
//! [`ModelRequest`], accessors on [`ModelResponse`], and the [`ModelRegistry`]
//! logic.

mod types;

use std::sync::Arc;

use serde_json::Value;

use crate::harness::message::{AssistantMessage, ContentBlock, Message};
use crate::harness::tool::{ToolCall, ToolSchema};
use crate::harness::usage::Usage;

pub use types::*;

impl ResponseFormat {
    /// Constructs a [`ResponseFormat::JsonSchema`] format.
    pub fn json_schema(name: impl Into<String>, schema: Value) -> Self {
        ResponseFormat::JsonSchema {
            name: name.into(),
            schema,
        }
    }
}

impl ModelRequest {
    /// Creates a request from a list of messages.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }

    /// Sets the tool declarations exposed for this call.
    pub fn with_tools(mut self, tools: Vec<ToolSchema>) -> Self {
        self.tools = tools;
        self
    }

    /// Sets the tool-choice policy.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Sets the response format.
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Sets the model id or registry alias.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the per-call timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Adds a tag to the request.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets the declared prompt cache segments.
    pub fn with_cache_segments(mut self, segments: Vec<PromptSegment>) -> Self {
        self.cache_segments = segments;
        self
    }

    /// Returns the ids of cacheable segments in declaration order, describing
    /// the stable prompt prefix middleware should preserve.
    pub fn cacheable_prefix_ids(&self) -> Vec<String> {
        self.cache_segments
            .iter()
            .filter(|s| s.cacheable)
            .map(|s| s.id.clone())
            .collect()
    }
}

impl ModelResponse {
    /// Creates a response wrapping a plain assistant text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            message: AssistantMessage {
                id: None,
                content: vec![ContentBlock::Text(content.into())],
                tool_calls: Vec::new(),
                usage: None,
            },
            usage: None,
            finish_reason: None,
            raw: None,
        }
    }

    /// Attaches usage to the response (and mirrors it onto the message).
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.message.usage = Some(usage);
        self.usage = Some(usage);
        self
    }

    /// Sets the provider finish reason.
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Returns the tool calls requested by the model, if any.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.message.tool_calls
    }

    /// Returns the concatenated text of the assistant message.
    pub fn text(&self) -> String {
        Message::Assistant(self.message.clone()).text()
    }
}

impl<State: Send + Sync> ModelRegistry<State> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
            default: None,
        }
    }

    /// Registers a model under `name`. The first registered model becomes the
    /// default unless one is already set.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
    ) -> &mut Self {
        let name = name.into();
        if self.default.is_none() {
            self.default = Some(name.clone());
        }
        self.models.insert(name, model);
        self
    }

    /// Sets the default model name.
    pub fn set_default(&mut self, name: impl Into<String>) -> &mut Self {
        self.default = Some(name.into());
        self
    }

    /// Looks up a model by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn ChatModel<State>>> {
        self.models.get(name).cloned()
    }

    /// Returns the default model, if one is configured.
    pub fn default_model(&self) -> Option<Arc<dyn ChatModel<State>>> {
        self.default.as_deref().and_then(|name| self.get(name))
    }

    /// Returns the configured default model name.
    pub fn default_name(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// Returns registered model names in sorted order.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.models.keys().cloned().collect();
        names.sort();
        names
    }
}

impl<State: Send + Sync> Default for ModelRegistry<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
