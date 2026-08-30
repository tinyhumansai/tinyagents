//! Tool layer for the harness.
//!
//! In the recursive architecture the [`Tool`] trait is the universal call
//! boundary that makes recursion uniform: a tool can be a plain function, but it
//! can equally be an *entire other agent* —
//! [`crate::subagent::SubAgentTool`] implements [`Tool`], so "a model
//! calling a model" is just "a model calling a tool". Everything the agent loop
//! can invoke flows through this layer and its [`ToolRegistry`].
//!
//! See [`types`] for definitions. This module provides constructors and the
//! [`ToolRegistry`] logic for registering and looking up tools by name.

mod error_policy;
pub mod injected;
mod prompt;
mod schema;
mod schema_prepare;
pub mod select;
mod timeout;
mod types;

use std::sync::Arc;

use serde_json::Value;

use crate::error::{Result, TinyAgentsError};

pub use error_policy::{ToolErrorPolicy, is_control_flow_error};
// Rendering a tool call for a human is not harness-specific, and two copies of
// the prefix list is how one of them silently stops stripping a prefix the
// other does. The definitions live in `tinytools` so a host that never links
// this crate still renders a tool name the same way.
pub use injected::{project_injected_arguments, strip_injected_arguments};
pub use prompt::*;
pub use schema::*;
pub use schema_prepare::*;
pub use select::*;
pub use timeout::*;
pub use tinytools::{
    ContextDetailOptions, context_detail_from_args, context_detail_from_args_with,
    humanize_tool_name,
};
pub use types::*;

impl ToolTimeout {
    /// Returns `true` for the default inherited timeout behavior.
    pub fn is_inherit(&self) -> bool {
        matches!(self, ToolTimeout::Inherit)
    }
}

impl ToolDisplay {
    /// Returns `true` when no display metadata is set.
    pub fn is_empty(&self) -> bool {
        self.label.is_none() && self.detail.is_none()
    }

    /// Creates display metadata with optional label and detail fields.
    pub fn new(label: Option<impl Into<String>>, detail: Option<impl Into<String>>) -> Self {
        Self {
            label: label.map(Into::into),
            detail: detail.map(Into::into),
        }
    }

    /// Creates display metadata with only a label.
    pub fn label(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            detail: None,
        }
    }

    /// Sets the static display detail.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
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
            display: ToolDisplay::default(),
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

    /// Sets human-facing presentation metadata for timeline/audit use.
    pub fn with_display(mut self, display: ToolDisplay) -> Self {
        self.display = display;
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

    /// Returns the **model-facing** schemas of all registered tools, sorted by
    /// name.
    ///
    /// Each schema has its tool's
    /// [`injected_arguments`][Tool::injected_arguments] projected out — removed
    /// from `properties` and from `required` alike — so a host-supplied
    /// argument is never advertised to the model and never demanded of it. See
    /// [`crate::tool::injected`] for the matching execution-time rule.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        let mut schemas: Vec<ToolSchema> = self
            .tools
            .values()
            .map(|t| project_injected_arguments(t.schema(), t.injected_arguments()))
            .collect();
        schemas.sort_by(|a, b| a.name.cmp(&b.name));
        schemas
    }

    /// Returns the **declared** schemas, including any injected arguments.
    ///
    /// This is the introspection view — registry listings, audit logs, docs —
    /// not the model-facing one. Never put this on the wire; use
    /// [`Self::schemas`].
    pub fn declared_schemas(&self) -> Vec<ToolSchema> {
        let mut schemas: Vec<ToolSchema> = self.tools.values().map(|t| t.schema()).collect();
        schemas.sort_by(|a, b| a.name.cmp(&b.name));
        schemas
    }

    /// Returns each registered tool's injected-argument names, keyed by tool
    /// name. Tools declaring none are omitted.
    pub fn injected_arguments(&self) -> std::collections::HashMap<String, Vec<String>> {
        self.tools
            .iter()
            .filter_map(|(name, tool)| {
                let injected = tool.injected_arguments();
                if injected.is_empty() {
                    return None;
                }
                Some((
                    name.clone(),
                    injected.iter().map(|key| (*key).to_string()).collect(),
                ))
            })
            .collect()
    }

    /// Returns each registered tool's [`ToolErrorPolicy`], keyed by tool name.
    pub fn error_policies(&self) -> std::collections::HashMap<String, ToolErrorPolicy> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.clone(), tool.error_policy()))
            .collect()
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

#[cfg(test)]
mod schema_test;
#[cfg(test)]
mod test;

#[cfg(test)]
mod timeout_test;
