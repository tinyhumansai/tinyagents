//! First-class sub-agents with recursion-depth tracking.
//!
//! This module provides the agent-calling-agent compositional primitive:
//!
//! - [`SubAgent`] wraps an [`AgentHarness`] and runs it as a *child run* one
//!   level deeper in the recursion tree than its caller.
//! - [`SubAgentTool`] adapts a [`SubAgent`] into a [`Tool`] so a parent agent
//!   can invoke another agent exactly like any other tool.
//!
//! # Depth tracking
//!
//! Every run carries a `depth` in its [`RunConfig`] (top-level runs are depth
//! `0`). When a sub-agent is invoked at `parent_depth`, its child run is created
//! at `parent_depth + 1`. The depth cap is
//! [`RunLimits::max_depth`][crate::harness::limits::RunLimits::max_depth]
//! (default [`RunLimits::DEFAULT_MAX_DEPTH`][crate::harness::limits::RunLimits::DEFAULT_MAX_DEPTH],
//! i.e. `8`), read from the child harness's [`RunPolicy`][crate::harness::runtime::RunPolicy].
//! If the child depth would exceed the cap, the invocation fails fast with
//! [`RustAgentsError::SubAgentDepth`] *before* any model call — a deterministic,
//! cheap guard against unbounded recursion.
//!
//! # Observability
//!
//! Each invocation emits [`AgentEvent::SubAgentStarted`] and
//! [`AgentEvent::SubAgentCompleted`] (carrying the sub-agent name and child
//! depth). When invoked with a shared [`EventSink`] — via
//! [`SubAgent::invoke_with_events`] or [`SubAgent::invoke_in_parent`] — the child
//! run's own events also flow onto the parent sink, so a parent observer sees
//! the full nested run tree.
//!
//! # Layout
//!
//! - [`types`] holds the public type definitions.
//! - This file holds the impls (constructors, the invoke methods, and the
//!   [`Tool`] adapter).
//! - `test.rs` holds focused tests.

mod types;

pub use types::*;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::{Result, RustAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, EventSink};
use crate::harness::message::Message;
use crate::harness::middleware::AgentRun;
use crate::harness::runtime::AgentHarness;
use crate::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};

impl<State: Send + Sync, Ctx: Send + Sync> SubAgent<State, Ctx> {
    /// Creates a sub-agent wrapping `harness` with a stable `name` and
    /// `description`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        harness: Arc<AgentHarness<State, Ctx>>,
    ) -> Self {
        Self {
            harness,
            name: name.into(),
            description: description.into(),
            system_prompt: None,
        }
    }

    /// Sets a fixed system prompt prepended to every child run as a leading
    /// system message. Returns `self` for chaining.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Returns the sub-agent's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the sub-agent's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the wrapped harness.
    pub fn harness(&self) -> &Arc<AgentHarness<State, Ctx>> {
        &self.harness
    }

    /// Builds the child run's seed messages from the optional system prompt and
    /// the caller `input`.
    fn seed_messages(&self, input: String) -> Vec<Message> {
        let mut messages = Vec::with_capacity(2);
        if let Some(prompt) = &self.system_prompt {
            messages.push(Message::system(prompt.clone()));
        }
        messages.push(Message::user(input));
        messages
    }

    /// Builds the child [`RunConfig`] for an invocation at `parent_depth`,
    /// enforcing the depth cap.
    ///
    /// Returns [`RustAgentsError::SubAgentDepth`] when the child depth
    /// (`parent_depth + 1`) would exceed the harness policy's `max_depth`.
    fn child_config(&self, parent_depth: usize) -> Result<RunConfig> {
        let max_depth = self.harness.policy().limits.max_depth;
        let child_depth = parent_depth + 1;
        if child_depth > max_depth {
            return Err(RustAgentsError::SubAgentDepth(max_depth));
        }
        Ok(RunConfig::new(format!("{}-d{child_depth}", self.name))
            .with_depth(child_depth)
            .with_max_depth(max_depth))
    }

    /// Runs the sub-agent as a child run at `parent_depth`, returning the
    /// child's [`AgentRun`].
    ///
    /// The child run is created at `parent_depth + 1`. `ctx_data` seeds the
    /// child [`RunContext`]. Sub-agent lifecycle events are emitted on the
    /// child's own (fresh) event sink; use [`Self::invoke_with_events`] to fan
    /// them out to a shared parent sink.
    ///
    /// # Errors
    ///
    /// Returns [`RustAgentsError::SubAgentDepth`] if the child depth would
    /// exceed the configured `max_depth`, or any error surfaced by the child
    /// agent loop.
    pub async fn invoke(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent_depth: usize,
        input: impl Into<String>,
    ) -> Result<AgentRun> {
        let config = self.child_config(parent_depth)?;
        let ctx = RunContext::new(config, ctx_data);
        self.run_child(state, ctx, input.into()).await
    }

    /// Like [`Self::invoke`] but routes the child run's events (and the
    /// sub-agent lifecycle events) onto the shared `events` sink so a parent
    /// observer sees the nested run.
    pub async fn invoke_with_events(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent_depth: usize,
        input: impl Into<String>,
        events: &EventSink,
    ) -> Result<AgentRun> {
        let config = self.child_config(parent_depth)?;
        let ctx = RunContext::new(config, ctx_data).with_events(events.clone());
        self.run_child(state, ctx, input.into()).await
    }

    /// Runs the sub-agent as a child of the live `parent` context.
    ///
    /// This is the fully context-threaded entry point: the child depth is
    /// derived from `parent.depth()` and the child inherits the parent's event
    /// sink so all nested events share one stream. `ctx_data` seeds the child
    /// context's user data.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::invoke`].
    pub async fn invoke_in_parent(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent: &RunContext<Ctx>,
        input: impl Into<String>,
    ) -> Result<AgentRun> {
        self.invoke_with_events(state, ctx_data, parent.depth(), input, &parent.events)
            .await
    }

    /// Shared driver: emits the sub-agent lifecycle events around the child
    /// agent loop.
    async fn run_child(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: String,
    ) -> Result<AgentRun> {
        let depth = ctx.depth();
        let messages = self.seed_messages(input);
        // Clone the sink (it shares listeners and the offset counter with the
        // context) so the completion event can be emitted after `ctx` is moved
        // into the child agent loop.
        let events = ctx.events.clone();

        events.emit(AgentEvent::SubAgentStarted {
            name: self.name.clone(),
            depth,
        });

        let run = self.harness.invoke_in_context(state, ctx, messages).await?;

        events.emit(AgentEvent::SubAgentCompleted {
            name: self.name.clone(),
            depth,
        });

        Ok(run)
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> SubAgentTool<State, Ctx> {
    /// Default JSON Schema for a sub-agent tool: an object with one required
    /// string field named [`SUBAGENT_INPUT_FIELD`].
    fn default_parameters() -> Value {
        json!({
            "type": "object",
            "properties": {
                SUBAGENT_INPUT_FIELD: {
                    "type": "string",
                    "description": "The task or question to delegate to the sub-agent."
                }
            },
            "required": [SUBAGENT_INPUT_FIELD]
        })
    }

    /// Wraps `subagent` as a tool invoked at `parent_depth = 0` (child runs at
    /// depth `1`). The tool name defaults to the sub-agent name.
    pub fn new(subagent: Arc<SubAgent<State, Ctx>>) -> Self {
        let tool_name = subagent.name().to_owned();
        Self {
            subagent,
            tool_name,
            parent_depth: 0,
            parameters: Self::default_parameters(),
        }
    }

    /// Overrides the model-visible tool name.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    /// Sets the caller depth this tool invokes the child at; the child runs at
    /// `parent_depth + 1`. Use this to express deeper nesting through the tool
    /// path (where the live parent depth is not available).
    pub fn with_parent_depth(mut self, parent_depth: usize) -> Self {
        self.parent_depth = parent_depth;
        self
    }

    /// Overrides the model-visible JSON Schema for the tool arguments.
    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// Extracts the child input string from model-supplied `arguments`.
    ///
    /// Accepts either an object carrying a string [`SUBAGENT_INPUT_FIELD`] field
    /// or a bare JSON string; anything else yields the empty string.
    fn extract_input(arguments: &Value) -> String {
        match arguments {
            Value::String(s) => s.clone(),
            Value::Object(map) => map
                .get(SUBAGENT_INPUT_FIELD)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            _ => String::new(),
        }
    }
}

#[async_trait]
impl<State, Ctx> Tool<State> for SubAgentTool<State, Ctx>
where
    State: Send + Sync,
    Ctx: Send + Sync + Default,
{
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        self.subagent.description()
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.tool_name.clone(),
            self.subagent.description().to_owned(),
            self.parameters.clone(),
        )
    }

    async fn call(&self, state: &State, call: ToolCall) -> Result<ToolResult> {
        let input = Self::extract_input(&call.arguments);
        let run = self
            .subagent
            .invoke(state, Ctx::default(), self.parent_depth, input)
            .await?;
        let text = run.text().unwrap_or_default();
        Ok(ToolResult::text(call.id, &self.tool_name, text))
    }
}

#[cfg(test)]
mod test;
