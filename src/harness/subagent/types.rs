//! Type definitions for first-class sub-agents.
//!
//! A [`SubAgent`] wraps an [`AgentHarness`] so it can be invoked as a *child
//! run*: a fully independent agent loop that runs one level deeper in the
//! recursion tree than its caller. [`SubAgentTool`] adapts a sub-agent into a
//! [`Tool`] so a parent agent can call another agent the same way it calls any
//! other tool — the key agent-calling-agent compositional pattern.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::harness::subagent` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs`.

use std::sync::Arc;

use serde_json::Value;

use crate::harness::runtime::AgentHarness;

/// The argument key a [`SubAgentTool`] reads the child input from.
///
/// When a parent model calls the tool, the harness passes the model-supplied
/// JSON arguments. The tool reads the string field named by this constant as
/// the child run's user prompt; if the arguments are a bare JSON string the
/// whole string is used instead.
pub const SUBAGENT_INPUT_FIELD: &str = "input";

/// A reusable, named child agent built on top of an [`AgentHarness`].
///
/// A `SubAgent` bundles:
/// - an `Arc<AgentHarness<State, Ctx>>` that drives the child agent loop,
/// - a stable `name` and `description` (used for tool schemas and observability),
/// - an optional `system_prompt` prepended to every child run as a system
///   message (the "fixed prompt template").
///
/// Invoking a sub-agent always produces a *child run* one level deeper than the
/// caller's depth. The harness's [`crate::harness::limits::RunLimits::max_depth`]
/// cap bounds how deep nested sub-agents may recurse; an invocation whose child
/// depth would exceed the cap fails with
/// [`crate::error::RustAgentsError::SubAgentDepth`].
///
/// `SubAgent` is cheap to clone-share via `Arc`; wrap it in an `Arc` to expose
/// the same child agent through several [`SubAgentTool`]s.
pub struct SubAgent<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The harness that drives the child agent loop.
    pub(crate) harness: Arc<AgentHarness<State, Ctx>>,
    /// Stable identifier for the sub-agent (used as the default tool name).
    pub(crate) name: String,
    /// Human/model readable description of what the sub-agent does.
    pub(crate) description: String,
    /// Optional system prompt prepended to every child run.
    pub(crate) system_prompt: Option<String>,
}

/// A [`Tool`] adapter that exposes a [`SubAgent`] to a parent agent.
///
/// [`Tool`]: crate::harness::tool::Tool
///
/// When the parent model calls this tool, [`SubAgentTool`] runs the wrapped
/// sub-agent as a child run at the configured `parent_depth` and returns the
/// child's final assistant text as the [`crate::harness::tool::ToolResult`]
/// content. This makes an entire agent composable as a single tool call.
///
/// Because the [`Tool`] trait gives `call` no access to the live parent
/// [`crate::harness::context::RunContext`], the depth the child runs at is fixed
/// at construction (`parent_depth`, default `0`). Nesting deeper sub-agents is
/// expressed by constructing the inner tool with a larger `parent_depth`. For
/// fully context-threaded invocation (reading the live parent depth) call
/// [`SubAgent::invoke_in_parent`] directly instead of going through the tool.
pub struct SubAgentTool<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The wrapped child agent.
    pub(crate) subagent: Arc<SubAgent<State, Ctx>>,
    /// Tool name exposed to the model (defaults to the sub-agent name).
    pub(crate) tool_name: String,
    /// The caller depth this tool invokes the child at; the child runs at
    /// `parent_depth + 1`.
    pub(crate) parent_depth: usize,
    /// JSON Schema describing the tool's model-visible arguments.
    pub(crate) parameters: Value,
}
