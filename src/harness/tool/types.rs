//! Tool layer types used by the harness.
//!
//! Here a [`ToolCall`] carries a required `id` so results can be correlated
//! back to the originating call, matching provider tool-call semantics.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

/// A model-visible declaration of a tool: its name, description, and a
/// JSON-schema-compatible parameter shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Canonical tool name (ASCII `snake_case` by convention).
    pub name: String,
    /// Human/model readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the model-visible input arguments.
    pub parameters: Value,
}

/// A request from the model to invoke a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id, required for result correlation.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments supplied by the model, as raw JSON.
    #[serde(default)]
    pub arguments: Value,
}

/// The outcome of executing a [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Id of the [`ToolCall`] this result answers.
    pub call_id: String,
    /// Name of the tool that produced the result.
    pub name: String,
    /// Model-facing textual content.
    pub content: String,
    /// Optional structured value for application code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    /// Error message when the tool failed; `None` on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock execution time in milliseconds.
    #[serde(default)]
    pub elapsed_ms: u64,
}

/// An incremental progress update emitted while a tool runs (streaming).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDelta {
    /// Id of the [`ToolCall`] this delta belongs to.
    pub call_id: String,
    /// Incremental content fragment.
    pub content: String,
}

/// A tool the harness can invoke during an agent loop.
///
/// Generic over the application `State` so tools can read shared context
/// without exposing it to model-visible schemas.
#[async_trait]
pub trait Tool<State>: Send + Sync {
    /// Canonical tool name.
    fn name(&self) -> &str;

    /// Human/model readable description.
    fn description(&self) -> &str;

    /// Returns the model-visible schema for this tool.
    fn schema(&self) -> ToolSchema;

    /// Executes the tool against application state and a validated call.
    async fn call(&self, state: &State, call: ToolCall) -> Result<ToolResult>;
}

/// A name-keyed registry of tools available to the harness.
pub struct ToolRegistry<State> {
    pub(crate) tools: HashMap<String, Arc<dyn Tool<State>>>,
}
