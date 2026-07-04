//! Tool layer for the harness.
//!
//! In the recursive architecture the [`Tool`] trait is the universal call
//! boundary that makes recursion uniform: a tool can be a plain function, but it
//! can equally be an *entire other agent* —
//! [`crate::harness::subagent::SubAgentTool`] implements [`Tool`], so "a model
//! calling a model" is just "a model calling a tool". Everything the agent loop
//! can invoke flows through this layer and its [`ToolRegistry`].
//!
//! See [`types`] for definitions. This module provides constructors and the
//! [`ToolRegistry`] logic for registering and looking up tools by name.

mod types;

use std::sync::Arc;

use serde_json::Value;

use crate::error::{Result, TinyAgentsError};

pub use types::*;

impl ToolSchema {
    /// Creates a tool schema.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            format: ToolFormat::Json,
        }
    }

    /// Sets the preferred model-visible tool-call format.
    pub fn with_format(mut self, format: ToolFormat) -> Self {
        self.format = format;
        self
    }

    /// Validates a model-supplied tool call against this tool's schema.
    ///
    /// The harness supports the JSON Schema subset used for model-visible tool
    /// declarations: `type`, object `properties`, `required`,
    /// `additionalProperties: false`, array `items`, and `enum`. Unknown schema
    /// keywords are ignored so providers can still receive richer schemas while
    /// the local execution boundary fails closed for the structural constraints
    /// it understands.
    pub fn validate_call(&self, call: &ToolCall) -> Result<()> {
        if call.name != self.name {
            return Err(TinyAgentsError::Validation(format!(
                "tool call `{}` does not match schema `{}`",
                call.name, self.name
            )));
        }
        validate_schema_value(
            &self.parameters,
            &call.arguments,
            &format!("tool `{}` arguments", self.name),
        )
    }
}

impl ToolFormat {
    /// Returns `true` for the default JSON/function-call format.
    pub fn is_json(&self) -> bool {
        matches!(self, ToolFormat::Json)
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

impl ToolPolicy {
    /// A classified, side-effect-free read-only policy.
    ///
    /// This is the recommended baseline for pure tools (computation, lookups
    /// against in-memory state) that never touch the filesystem, network, or
    /// money. Being *classified*, it passes strict policy enforcement.
    pub fn read_only() -> Self {
        Self {
            classified: true,
            side_effects: ToolSideEffects {
                read_only: true,
                ..ToolSideEffects::default()
            },
            runtime: ToolRuntime {
                idempotent: true,
                cancelable: true,
                ..ToolRuntime::default()
            },
            access: ToolAccess {
                background_safe: true,
                ..ToolAccess::default()
            },
        }
    }

    /// A classified policy with no side effects declared yet, ready for the
    /// builder methods below.
    pub fn classified() -> Self {
        Self {
            classified: true,
            ..Self::default()
        }
    }

    /// Sets the declared side effects.
    pub fn with_side_effects(mut self, side_effects: ToolSideEffects) -> Self {
        self.classified = true;
        self.side_effects = side_effects;
        self
    }

    /// Sets the declared runtime requirements.
    pub fn with_runtime(mut self, runtime: ToolRuntime) -> Self {
        self.classified = true;
        self.runtime = runtime;
        self
    }

    /// Sets the declared access requirements.
    pub fn with_access(mut self, access: ToolAccess) -> Self {
        self.classified = true;
        self.access = access;
        self
    }

    /// Marks the tool as requiring explicit human approval before each call.
    pub fn requiring_approval(mut self) -> Self {
        self.classified = true;
        self.access.approval_required = true;
        self
    }

    /// Returns `true` when the policy declares any side effect beyond read-only.
    pub fn has_side_effects(&self) -> bool {
        let s = &self.side_effects;
        s.writes_files
            || s.network
            || s.installs_dependencies
            || s.destructive
            || s.external_service
            || s.payment
    }
}

impl<State: Send + Sync> ToolRegistry<State> {
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

    /// Returns a snapshot of every registered tool's [`ToolPolicy`], keyed by
    /// tool name. This is the projection policy-enforcement middleware and audit
    /// logs consume.
    pub fn policies(&self) -> std::collections::HashMap<String, ToolPolicy> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.clone(), tool.policy()))
            .collect()
    }
}

impl<State: Send + Sync> Default for ToolRegistry<State> {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_schema_value(schema: &Value, value: &Value, path: &str) -> Result<()> {
    if schema.is_null() || schema.as_object().is_some_and(|o| o.is_empty()) {
        return Ok(());
    }

    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|allowed| allowed == value)
    {
        return Err(TinyAgentsError::Validation(format!(
            "{path} must be one of the declared enum values"
        )));
    }

    if let Some(type_spec) = schema.get("type") {
        validate_type_spec(type_spec, value, path)?;
    }

    // Enforce `required` independently of `properties`. A schema may declare
    // required fields without listing per-field property schemas; nesting this
    // check under `properties` would let such schemas fail open, silently
    // accepting calls that omit required arguments.
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        let object = value.as_object().ok_or_else(|| {
            TinyAgentsError::Validation(format!("{path} must be an object with declared fields"))
        })?;
        for field in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(field) {
                return Err(TinyAgentsError::Validation(format!(
                    "{path}.{field} is required"
                )));
            }
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        let object = value.as_object().ok_or_else(|| {
            TinyAgentsError::Validation(format!("{path} must be an object with declared fields"))
        })?;

        if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
            for field in object.keys() {
                if !properties.contains_key(field) {
                    return Err(TinyAgentsError::Validation(format!(
                        "{path}.{field} is not allowed"
                    )));
                }
            }
        }

        for (field, field_schema) in properties {
            if let Some(field_value) = object.get(field) {
                validate_schema_value(field_schema, field_value, &format!("{path}.{field}"))?;
            }
        }
    }

    if let Some(items_schema) = schema.get("items")
        && let Some(items) = value.as_array()
    {
        for (index, item) in items.iter().enumerate() {
            validate_schema_value(items_schema, item, &format!("{path}[{index}]"))?;
        }
    }

    Ok(())
}

fn validate_type_spec(type_spec: &Value, value: &Value, path: &str) -> Result<()> {
    if let Some(kind) = type_spec.as_str() {
        if json_value_matches_type(value, kind) {
            return Ok(());
        }
        return Err(TinyAgentsError::Validation(format!(
            "{path} must be {kind}, got {}",
            json_value_kind(value)
        )));
    }

    if let Some(kinds) = type_spec.as_array() {
        let allowed: Vec<&str> = kinds.iter().filter_map(Value::as_str).collect();
        if allowed
            .iter()
            .any(|kind| json_value_matches_type(value, kind))
        {
            return Ok(());
        }
        return Err(TinyAgentsError::Validation(format!(
            "{path} must be one of {}, got {}",
            allowed.join(", "),
            json_value_kind(value)
        )));
    }

    Ok(())
}

fn json_value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        _ => true,
    }
}

fn json_value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.as_i64().is_some() || n.as_u64().is_some() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod test;
