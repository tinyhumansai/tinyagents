//! Tool layer for the harness.
//!
//! See [`types`] for definitions. This module provides constructors and the
//! [`ToolRegistry`] logic for registering and looking up tools by name.

mod types;

use std::sync::Arc;

use serde_json::Value;

pub use types::*;

impl ToolSchema {
    /// Creates a tool schema.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

impl ToolCall {
    /// Creates a tool call with the given id, name, and arguments.
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

impl ToolResult {
    /// Creates a successful textual tool result.
    pub fn text(
        call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            content: content.into(),
            raw: None,
            error: None,
            elapsed_ms: 0,
        }
    }

    /// Creates an error tool result, preserving the call id for repair.
    pub fn error(
        call_id: impl Into<String>,
        name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let message = message.into();
        Self {
            call_id: call_id.into(),
            name: name.into(),
            content: message.clone(),
            raw: None,
            error: Some(message),
            elapsed_ms: 0,
        }
    }

    /// Returns `true` when the tool reported an error.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

impl<State> ToolRegistry<State> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            tools: std::collections::HashMap::new(),
        }
    }

    /// Registers a tool under its [`Tool::name`], replacing any existing tool
    /// with the same name.
    pub fn register(&mut self, tool: Arc<dyn Tool<State>>) -> &mut Self {
        self.tools.insert(tool.name().to_owned(), tool);
        self
    }

    /// Looks up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool<State>>> {
        self.tools.get(name).cloned()
    }

    /// Returns the registered tool names in sorted order.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the schemas of all registered tools, sorted by name.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        let mut schemas: Vec<ToolSchema> = self.tools.values().map(|t| t.schema()).collect();
        schemas.sort_by(|a, b| a.name.cmp(&b.name));
        schemas
    }
}

impl<State> Default for ToolRegistry<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
