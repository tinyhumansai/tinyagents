//! First-class sub-agents with recursion-depth tracking.
//!
//! This is the harness's flagship recursion surface: it lets one agent run
//! another agent as a child of itself, so a language model orchestrating tools
//! is, transparently, a language model orchestrating *other models*. It is the
//! concrete "agents calling agents" mechanism behind the crate's
//! recursive-language-model framing — the in-harness analogue of the graph-side
//! [`crate::graph::subgraph`] recursion.
//!
//! This module provides the agent-calling-agent compositional primitive:
//!
//! - [`SubAgent`] wraps an [`AgentHarness`] and runs it as a *child run* one
//!   level deeper in the recursion tree than its caller.
//! - [`SubAgentTool`] adapts a [`SubAgent`] into a [`Tool`] so a parent agent
//!   can invoke another agent exactly like any other tool.
//! - [`SubAgentSession`] keeps a single [`SubAgent`] alive across multiple
//!   turns, *reusing* the same harness while accumulating the conversation
//!   transcript — the post-completion, human-in-the-loop reuse primitive.
//!
//! # Reuse vs. steering
//!
//! There are two ways an orchestrator keeps a sub-agent "in play" across human
//! input:
//!
//! - **Reuse** ([`SubAgentSession`]): the child run *completes*, the
//!   orchestrator obtains human input, then calls the **same** sub-agent again
//!   carrying the prior transcript. Nothing is killed or restarted.
//! - **Steering** ([`crate::harness::steering`]): an orchestrator/human injects
//!   commands into a **still-running** agent at safe checkpoints.
//!
//! `SubAgentSession` implements the first. The flow is:
//!
//! 1. `session.send(state, ctx, vec![Message::user("…")])` — runs the sub-agent
//!    over the retained transcript and folds its reply back in.
//! 2. Inspect the returned [`AgentRun`]; obtain human input out-of-band.
//! 3. `session.send(state, ctx, vec![Message::user(human_reply)])` — the same
//!    sub-agent answers with the full prior context still in the transcript.
//!
//! Every send after the first emits [`AgentEvent::SubAgentReused`] so the reuse
//! is visible alongside the per-send
//! [`SubAgentStarted`][AgentEvent::SubAgentStarted]/[`SubAgentCompleted`][AgentEvent::SubAgentCompleted]
//! bracket.
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
//! [`TinyAgentsError::SubAgentDepth`] *before* any model call — a deterministic,
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

use crate::error::{Result, TinyAgentsError};
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
    /// Returns [`TinyAgentsError::SubAgentDepth`] when the child depth
    /// (`parent_depth + 1`) would exceed the harness policy's `max_depth`.
    fn child_config(&self, parent_depth: usize) -> Result<RunConfig> {
        let max_depth = self.harness.policy().limits.max_depth;
        let child_depth = parent_depth + 1;
        if child_depth > max_depth {
            return Err(TinyAgentsError::SubAgentDepth(max_depth));
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
    /// Returns [`TinyAgentsError::SubAgentDepth`] if the child depth would
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

impl<State: Send + Sync, Ctx: Send + Sync> SubAgentSession<State, Ctx> {
    /// Creates a session that reuses `subagent` across turns.
    ///
    /// The child runs at depth `1` by default (caller `parent_depth = 0`); use
    /// [`Self::with_parent_depth`] to express deeper nesting.
    pub fn new(subagent: Arc<SubAgent<State, Ctx>>) -> Self {
        Self {
            subagent,
            transcript: Vec::new(),
            turn: 0,
            parent_depth: 0,
            events: EventSink::new(),
            seeded: false,
        }
    }

    /// Creates a session from an owned [`SubAgent`], wrapping it in an `Arc`.
    pub fn from_subagent(subagent: SubAgent<State, Ctx>) -> Self {
        Self::new(Arc::new(subagent))
    }

    /// Routes the reuse lifecycle and the child run's own events onto `events`
    /// so an external observer (or testkit recorder) sees every send. Returns
    /// `self` for chaining.
    pub fn with_events(mut self, events: EventSink) -> Self {
        self.events = events;
        self
    }

    /// Sets the caller depth the child runs at; the child run is created at
    /// `parent_depth + 1`. Returns `self` for chaining.
    pub fn with_parent_depth(mut self, parent_depth: usize) -> Self {
        self.parent_depth = parent_depth;
        self
    }

    /// Returns the reused sub-agent. The same `Arc` is shared across every
    /// send, so this is how callers confirm the harness was never rebuilt.
    pub fn subagent(&self) -> &Arc<SubAgent<State, Ctx>> {
        &self.subagent
    }

    /// Returns the accumulated conversation transcript carried across sends.
    pub fn transcript(&self) -> &[Message] {
        &self.transcript
    }

    /// Returns the number of completed sends (turns).
    pub fn turns(&self) -> usize {
        self.turn
    }

    /// Clears the retained transcript and turn counter, so the next [`Self::send`]
    /// starts a fresh conversation (re-seeding the fixed system prompt). The
    /// underlying [`SubAgent`]/harness is left untouched and still reused.
    pub fn reset(&mut self) {
        self.transcript.clear();
        self.turn = 0;
        self.seeded = false;
    }

    /// Builds the child [`RunConfig`] for the current turn, enforcing the depth
    /// cap exactly as [`SubAgent::invoke`] does.
    fn child_config(&self) -> Result<RunConfig> {
        let max_depth = self.subagent.harness.policy().limits.max_depth;
        let child_depth = self.parent_depth + 1;
        if child_depth > max_depth {
            return Err(TinyAgentsError::SubAgentDepth(max_depth));
        }
        Ok(RunConfig::new(format!(
            "{}-t{}-d{child_depth}",
            self.subagent.name, self.turn
        ))
        .with_depth(child_depth)
        .with_max_depth(max_depth))
    }

    /// Runs the reused sub-agent for one turn over the FULL accumulated
    /// transcript, then folds the produced assistant/tool messages back into
    /// the transcript so the next send continues with full context.
    ///
    /// `input` (typically a single [`Message::user`] carrying human input) is
    /// appended to the retained transcript before the run. On the first send
    /// the sub-agent's fixed [`SubAgent::with_system_prompt`] is prepended once.
    /// The same underlying harness is reused on every call — nothing is
    /// reconstructed.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::SubAgentDepth`] if the child depth would
    /// exceed the configured `max_depth`, or any error surfaced by the child
    /// agent loop.
    pub async fn send(
        &mut self,
        state: &State,
        ctx_data: Ctx,
        input: Vec<Message>,
    ) -> Result<AgentRun> {
        // Seed the fixed system prompt once, on the first send.
        if !self.seeded {
            if let Some(prompt) = &self.subagent.system_prompt {
                self.transcript.push(Message::system(prompt.clone()));
            }
            self.seeded = true;
        }

        // Append the new (e.g. human/user) input to the retained transcript.
        self.transcript.extend(input);

        let config = self.child_config()?;
        let depth = config.depth;
        let ctx = RunContext::new(config, ctx_data).with_events(self.events.clone());

        // Clone the sink so we can emit the completion event after `ctx` is
        // moved into the child agent loop.
        let events = self.events.clone();
        events.emit(AgentEvent::SubAgentStarted {
            name: self.subagent.name.clone(),
            depth,
        });
        if self.turn > 0 {
            events.emit(AgentEvent::SubAgentReused {
                name: self.subagent.name.clone(),
                turn: self.turn,
            });
        }

        // REUSE the same underlying harness/SubAgent (no reconstruction),
        // running it over the full accumulated transcript.
        let run = self
            .subagent
            .harness
            .invoke_in_context(state, ctx, self.transcript.clone())
            .await?;

        events.emit(AgentEvent::SubAgentCompleted {
            name: self.subagent.name.clone(),
            depth,
        });

        // Carry the produced assistant/tool messages forward. `run.messages` is
        // the working transcript the loop ended with (everything we passed plus
        // the new assistant/tool messages), so the next send continues with the
        // full context.
        self.transcript = run.messages.clone();
        self.turn += 1;

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

    fn limit_result_for_parent(
        call_id: String,
        tool_name: &str,
        error: &TinyAgentsError,
    ) -> Option<ToolResult> {
        let limit_kind = match error {
            TinyAgentsError::LimitExceeded(_) => "configured run limit",
            TinyAgentsError::Timeout(_) => "wall-clock deadline",
            TinyAgentsError::SubAgentDepth(_) => "recursion depth limit",
            _ => return None,
        };

        Some(ToolResult::error(
            call_id,
            tool_name,
            format!(
                "Sub-agent `{tool_name}` stopped before completing because it hit its {limit_kind}: {error}. The parent orchestrator should treat this as a delegated-agent limit signal, not a completed answer."
            ),
        ))
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
        let call_id = call.id;
        let run = match self
            .subagent
            .invoke(state, Ctx::default(), self.parent_depth, input)
            .await
        {
            Ok(run) => run,
            Err(error) => {
                if let Some(result) =
                    Self::limit_result_for_parent(call_id, &self.tool_name, &error)
                {
                    return Ok(result);
                }
                return Err(error);
            }
        };
        let text = run.text().unwrap_or_default();
        Ok(ToolResult::text(call_id, &self.tool_name, text))
    }
}

#[cfg(test)]
mod test;
