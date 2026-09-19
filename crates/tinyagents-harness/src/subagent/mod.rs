//! First-class sub-agents with recursion-depth tracking.
//!
//! This is the harness's flagship recursion surface: it lets one agent run
//! another agent as a child of itself, so a language model orchestrating tools
//! is, transparently, a language model orchestrating *other models*. It is the
//! concrete "agents calling agents" mechanism behind the crate's
//! recursive-language-model framing — the in-harness analogue of the graph-side
//! `subgraph` recursion (in `tinyagents-graph`).
//!
//! This module provides the agent-calling-agent compositional primitive:
//!
//! - [`SubAgent`] wraps an [`AgentHarness`] and runs it as a *child run* one
//!   level deeper in the recursion tree than its caller.
//! - [`SubAgentTool`] adapts a [`SubAgent`] into a typed
//!   [`ToolDispatch`] so a parent agent can invoke another agent with its live
//!   run context.
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
//! - **Steering** ([`crate::steering`]): an orchestrator/human injects
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
//! [`RunLimits::max_depth`][crate::limits::RunLimits::max_depth]
//! (default [`RunLimits::DEFAULT_MAX_DEPTH`][crate::limits::RunLimits::DEFAULT_MAX_DEPTH],
//! i.e. `8`), read from the child harness's [`RunPolicy`][crate::runtime::RunPolicy].
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
//! - `types` holds the public type definitions.
//! - This file holds the impls (constructors, the invoke methods, and the
//!   typed-parent dispatcher).
//! - `test.rs` holds focused tests.

mod types;

pub use types::*;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::context::{RunConfig, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::{AgentEvent, EventSink};
use crate::ids::{ThreadId, next_seq};
use crate::middleware::AgentRun;
use crate::runtime::AgentHarness;
use crate::tool::ToolDispatch;
use tinyinference_llm::message::Message;

impl<State: Send + Sync, Ctx: Send + Sync + 'static> SubAgent<State, Ctx> {
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
    /// enforcing the depth cap and deriving an isolated child thread from a
    /// parent thread when one is available.
    ///
    /// `parent` is `Some((parent_run_id, ordinal))` for every entry point that
    /// has a live parent [`RunContext`] to derive from — `ordinal` comes from
    /// [`RunContext::next_child_ordinal`], a counter scoped to that one
    /// context instance (not process-global). The child run id is then a pure
    /// function of the parent's run id and that ordinal
    /// (`{name}-d{depth}-{parent_run_id}-{ordinal}`), so two processes
    /// replaying the identical sequence of calls against the identical parent
    /// run derive the identical child run ids (M-2) — unlike the historical
    /// `ids::next_seq()` suffix, which restarts at a different value every
    /// process and made replayed journals of nested runs diverge across
    /// processes.
    ///
    /// `parent` is `None` only for the standalone entry points
    /// ([`Self::invoke`]/[`Self::invoke_with_events`]) that are not called
    /// with a live parent context at all; those fall back to
    /// [`crate::ids::next_seq`] since there is no parent run to derive
    /// determinism from.
    ///
    /// Returns [`TinyAgentsError::SubAgentDepth`] when the child depth
    /// (`parent_depth + 1`) would exceed the harness policy's `max_depth`.
    fn child_config(
        &self,
        parent_depth: usize,
        thread_id: Option<&ThreadId>,
        max_turn_output_tokens: Option<u32>,
        parent: Option<(&str, u64)>,
    ) -> Result<RunConfig> {
        let max_depth = self.harness.policy().limits.max_depth;
        let child_depth = RunConfig::checked_child_depth(parent_depth, max_depth)?;
        let child_run_id = match parent {
            Some((parent_run_id, ordinal)) => {
                format!("{}-d{child_depth}-{parent_run_id}-{ordinal}", self.name)
            }
            // No parent context to derive determinism from: suffix a
            // process-unique sequence so each invocation still gets its own
            // run id (a bare `{name}-d{depth}` was reused across invocations,
            // which interleaved journals and status stores keyed by run id).
            None => format!("{}-d{child_depth}-{}", self.name, next_seq()),
        };
        let mut config = RunConfig::new(child_run_id.clone())
            .with_depth(child_depth)
            .with_max_depth(max_depth);
        config.thread_id = thread_id.map(|parent| child_thread_id(parent, &child_run_id));
        config.max_turn_output_tokens = max_turn_output_tokens;
        Ok(config)
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
        let config = self.child_config(parent_depth, None, None, None)?;
        let ctx = RunContext::new(config, ctx_data);
        self.run_child(state, ctx, input.into(), false).await
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
        let config = self.child_config(parent_depth, None, None, None)?;
        let ctx = RunContext::new(config, ctx_data).with_events(events.clone());
        self.run_child(state, ctx, input.into(), false).await
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
        // This is the generic explicit-model entry point, intentionally kept
        // available to borrowed `State` callers. A hosted parent must enter
        // through `invoke_hosted_in_parent` below, where the `State: 'static`
        // bound makes the invocation authority type-safe. Falling through to
        // the explicit loop here would discard the parent's definition and
        // approval authority, so reject it before constructing a child.
        if parent.host_authority.is_some() {
            return Err(TinyAgentsError::Validation(
                "hosted parent delegation requires invoke_hosted_in_parent".into(),
            ));
        }
        // The child harness may tighten the tree cap, but it may never widen
        // the explicit parent lineage cap. `RunContext::child` is the one
        // place that copies the live recursive capabilities and creates the
        // isolated counters/control slot for this invocation.
        let config = self.child_config(
            parent.depth(),
            parent.thread_id(),
            parent.config.max_turn_output_tokens,
            Some((parent.run_id().as_str(), parent.next_child_ordinal())),
        )?;
        let ctx = parent.child(config, ctx_data)?;
        self.run_child(state, ctx, input.into(), parent.streaming)
            .await
    }

    /// Shared driver: emits the sub-agent lifecycle events around the child
    /// agent loop.
    ///
    /// When `streaming` is `true` the child runs through the streaming loop
    /// path, so its per-token model/reasoning deltas are emitted onto the
    /// shared [`EventSink`] and reach the parent's stream (stamped with the
    /// child's own `run_id` and `depth`). When `false` the child runs the
    /// unary path, leaving the parent's event stream unchanged.
    async fn run_child(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: String,
        streaming: bool,
    ) -> Result<AgentRun> {
        let depth = ctx.depth();
        let messages = self.seed_messages(input.clone());
        // Clone the sink (it shares listeners and the offset counter with the
        // context) so the completion event can be emitted after `ctx` is moved
        // into the child agent loop.
        let events = ctx.events.clone();

        events.emit(AgentEvent::SubAgentStarted {
            name: self.name.clone(),
            depth,
        });

        let run = if streaming {
            self.harness
                .invoke_streaming_in_context(state, ctx, messages)
                .await?
        } else {
            self.harness.invoke_in_context(state, ctx, messages).await?
        };

        events.emit(AgentEvent::SubAgentCompleted {
            name: self.name.clone(),
            depth,
        });

        Ok(run)
    }
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> SubAgent<State, Ctx> {
    /// Runs this child under the exact host authority installed on `parent`.
    ///
    /// Unlike [`Self::invoke_in_parent`], this path resolves the parent's
    /// delegate allowlist and re-enters the child through the parent's shared
    /// invocation bundle. A context that is not hosted fails closed rather
    /// than silently acquiring an unrelated harness configuration.
    pub async fn invoke_hosted_in_parent(
        &self,
        state: &State,
        ctx_data: Ctx,
        parent: &RunContext<Ctx>,
        input: impl Into<String>,
    ) -> Result<AgentRun> {
        if parent.host_authority.is_none() {
            return Err(TinyAgentsError::Validation(
                "hosted subagent invocation requires parent host authority".into(),
            ));
        }
        let config = self.child_config(
            parent.depth(),
            parent.thread_id(),
            parent.config.max_turn_output_tokens,
            Some((parent.run_id().as_str(), parent.next_child_ordinal())),
        )?;
        let child = parent.child(config, ctx_data)?;
        self.run_hosted_child(state, child, input.into(), parent.streaming)
            .await
    }

    /// Hosted recursive driver. Kept separate from the generic explicit-model
    /// path so a borrowed `State` never has to interact with live host
    /// authority stored on a `RunContext`.
    async fn run_hosted_child(
        &self,
        state: &State,
        ctx: RunContext<Ctx>,
        input: String,
        streaming: bool,
    ) -> Result<AgentRun> {
        let depth = ctx.depth();
        let messages = self.seed_messages(input.clone());
        let binding = crate::runtime::host_invocation_binding::<State, Ctx>(&ctx)?;
        let Some(binding) = binding else {
            return self.run_child(state, ctx, input, streaming).await;
        };
        let parent_agent = ctx.host_agent_id.as_deref().ok_or_else(|| {
            TinyAgentsError::Validation(
                "hosted parent delegation is missing its parent agent identity".into(),
            )
        })?;
        let delegates = binding
            .host
            .definitions
            .delegates_for(parent_agent)
            .await
            .map_err(|error| {
                TinyAgentsError::Validation(format!(
                    "delegate authorization lookup failed: {error}"
                ))
            })?;
        if !delegates.iter().any(|delegate| delegate == &self.name) {
            return Err(TinyAgentsError::Validation(format!(
                "agent `{parent_agent}` is not authorized to delegate to `{}`",
                self.name
            )));
        }
        let runtime = binding.runtime.clone().ok_or_else(|| {
            TinyAgentsError::Validation(
                "hosted subagent invocation is missing its parent runtime overlay".into(),
            )
        })?;

        let events = ctx.events.clone();
        events.emit(AgentEvent::SubAgentStarted {
            name: self.name.clone(),
            depth,
        });
        let invocation = crate::runtime::AgentInvocation::from_shared_host(
            binding.host.clone(),
            crate::runtime::AgentTurnRequest::new(self.name.clone(), messages),
            ctx,
            Some(runtime),
        );
        // A hosted parent always re-enters through this exact capability
        // bundle. The child harness supplies durable mechanics only; it cannot
        // select an alternate host authority.
        let run = if streaming {
            self.harness
                .invoke_agent_streaming_with_capabilities(invocation, state)
                .await?
        } else {
            self.harness
                .invoke_agent_with_capabilities(invocation, state)
                .await?
        };
        events.emit(AgentEvent::SubAgentCompleted {
            name: self.name.clone(),
            depth,
        });
        Ok(run)
    }
}

/// Derives an isolated child thread id from the parent thread and the child's
/// run id. The run id already carries a process-unique sequence, so the thread
/// id inherits its uniqueness.
fn child_thread_id(parent: &ThreadId, child_run_id: &str) -> ThreadId {
    ThreadId::new(format!("{}-subagent-{child_run_id}", parent.as_str()))
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
        let child_depth = RunConfig::checked_child_depth(self.parent_depth, max_depth)?;
        // As in `SubAgent::child_config`, suffix a process-unique sequence so
        // two sessions reusing the same sub-agent never share run ids for the
        // same turn number.
        Ok(RunConfig::new(format!(
            "{}-t{}-d{child_depth}-{}",
            self.subagent.name,
            self.turn,
            next_seq()
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
        let depth = config.depth();
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

impl<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> SubAgentTool<State, Ctx> {
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

    /// Wraps `subagent` as a typed-parent tool.
    ///
    /// `child_data` is required: it makes application-data inheritance explicit
    /// for every recursive invocation.
    pub fn new(subagent: Arc<SubAgent<State, Ctx>>, child_data: ChildDataPolicy<Ctx>) -> Self {
        let tool_name = subagent.name().to_owned();
        Self {
            subagent,
            tool_name,
            child_data,
            parameters: Self::default_parameters(),
            declaration: std::sync::OnceLock::new(),
        }
    }

    /// Overrides the model-visible tool name.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
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

    /// Invokes this sub-agent from the actual parent [`RunContext`].
    ///
    /// This is the agent-native recursive-tool boundary.  It is intentionally
    /// separate from `tinytools::Tool`: TinyTools only receives the narrow
    /// workspace/thread/output vocabulary it needs for ordinary tools, while
    /// a child agent must inherit the parent run's live lineage, cancellation,
    /// stores, events, workspace, steering, and streaming state.  The harness
    /// tool dispatcher registers this typed entry point explicitly; it does
    /// not downcast a generic tool or recover a parent from a global map.
    pub async fn invoke_in_parent_context(
        &self,
        state: &State,
        args: Value,
        options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> Result<tinytools::ToolResult> {
        let input = Self::extract_input(&args);
        let config = match self.subagent.child_config(
            parent.depth(),
            parent.thread_id(),
            parent.config.max_turn_output_tokens,
            Some((parent.run_id().as_str(), parent.next_child_ordinal())),
        ) {
            Ok(config) => config,
            Err(error) => {
                return Ok(tinytools::ToolResult::error(format!(
                    "Sub-agent `{}` stopped before completing because it hit its recursion depth limit: {error}. The parent orchestrator should treat this as a delegated-agent limit signal, not a completed answer.",
                    self.tool_name
                )));
            }
        };
        let child_data = self.child_data.child_data(&parent.data);
        let child = match parent.child(config, child_data) {
            Ok(child) => child,
            Err(TinyAgentsError::SubAgentDepth(_)) => {
                return Ok(tinytools::ToolResult::error(format!(
                    "Sub-agent `{}` stopped before completing because it hit its recursion depth limit. The parent orchestrator should treat this as a delegated-agent limit signal, not a completed answer.",
                    self.tool_name
                )));
            }
            Err(error) => return Err(error),
        };
        let run = match self
            .subagent
            .run_hosted_child(state, child, input, parent.streaming)
            .await
        {
            Ok(run) => run,
            Err(error) => {
                if matches!(
                    error,
                    TinyAgentsError::LimitExceeded(_)
                        | TinyAgentsError::Timeout(_)
                        | TinyAgentsError::CallTimeout(_)
                        | TinyAgentsError::SubAgentDepth(_)
                ) {
                    return Ok(tinytools::ToolResult::error(format!(
                        "Sub-agent `{}` stopped before completing because it hit a configured run limit: {error}. The parent orchestrator should treat this as a delegated-agent limit signal, not a completed answer.",
                        self.tool_name
                    )));
                }
                return Err(error);
            }
        };
        let text = run.text().unwrap_or_default();
        let result = tinytools::ToolResult::success(text);
        Ok(if options.prefer_markdown {
            result.with_markdown(run.text().unwrap_or_default())
        } else {
            result
        })
    }
}

#[async_trait]
impl<State, Ctx> ToolDispatch<State, Ctx> for SubAgentTool<State, Ctx>
where
    State: Send + Sync + 'static,
    Ctx: Send + Sync + 'static,
{
    fn tool(&self) -> Arc<dyn tinytools::Tool> {
        // Built once and cached (M-4): `tool()` is called several times per
        // admitted call and once per tool per run for `schemas()`, and a
        // fresh `Arc<SubAgentToolDeclaration>` with a cloned `parameters`
        // `Value` on every call is unnecessary allocation for a declaration
        // that never changes after registration.
        Arc::clone(self.declaration.get_or_init(|| {
            Arc::new(SubAgentToolDeclaration {
                name: self.tool_name.clone(),
                description: self.subagent.description().to_owned(),
                parameters: self.parameters.clone(),
            })
        }))
    }

    fn output_origin(&self) -> crate::host::ContentOrigin {
        crate::host::ContentOrigin::Agent
    }

    async fn execute(
        &self,
        state: &State,
        arguments: Value,
        options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<tinytools::ToolResult> {
        self.invoke_in_parent_context(state, arguments, options, parent)
            .await
            .map_err(anyhow::Error::from)
    }
}

/// Pure canonical declaration for a typed-parent sub-agent dispatcher.
///
/// Calling it through `tinytools::Tool` directly is refused because that trait
/// intentionally lacks the parent `RunContext`; register the enclosing
/// [`SubAgentTool`] with [`crate::tool::ToolRegistry::register_dispatch`].
struct SubAgentToolDeclaration {
    name: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl tinytools::Tool for SubAgentToolDeclaration {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }

    async fn execute(&self, _args: Value) -> anyhow::Result<tinytools::ToolResult> {
        anyhow::bail!(
            "sub-agent `{}` requires typed-parent dispatch; register SubAgentTool with ToolRegistry::register_dispatch",
            self.name
        )
    }
}

#[cfg(test)]
mod test;
