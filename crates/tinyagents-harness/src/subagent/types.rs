//! Type definitions for first-class sub-agents.
//!
//! A [`SubAgent`] wraps an [`AgentHarness`] so it can be invoked as a *child
//! run*: a fully independent agent loop that runs one level deeper in the
//! recursion tree than its caller. [`SubAgentTool`] adapts a sub-agent into a
//! typed [`crate::tool::ToolDispatch`] so a parent agent can call another agent
//! through its live run context — the key agent-calling-agent compositional
//! pattern.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::subagent` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs`.

use std::sync::Arc;

use serde_json::Value;

use crate::events::EventSink;
use crate::runtime::AgentHarness;
use tinyinference_llm::message::Message;

/// The argument key a [`SubAgentTool`] reads the child input from.
///
/// When a parent model calls the tool, the harness passes the model-supplied
/// JSON arguments. The tool reads the string field named by this constant as
/// the child run's user prompt; if the arguments are a bare JSON string the
/// whole string is used instead.
pub const SUBAGENT_INPUT_FIELD: &str = "input";

/// Typed policy that constructs a child's user data from its parent data.
///
/// Recursive capabilities are inherited by [`RunContext::child`](crate::context::RunContext::child); this policy
/// makes the separate application-data decision explicit instead of silently
/// substituting `Default` or reaching for task-local state.
#[derive(Clone)]
pub struct ChildDataPolicy<Ctx> {
    pub(crate) transform: Arc<dyn Fn(&Ctx) -> Ctx + Send + Sync>,
}

impl<Ctx> ChildDataPolicy<Ctx> {
    /// Creates a policy from a pure parent-to-child transformation.
    pub fn new(transform: impl Fn(&Ctx) -> Ctx + Send + Sync + 'static) -> Self {
        Self {
            transform: Arc::new(transform),
        }
    }

    /// Produces one child's data value from its parent.
    pub fn child_data(&self, parent: &Ctx) -> Ctx {
        (self.transform)(parent)
    }
}

/// A reusable, named child agent built on top of an [`AgentHarness`].
///
/// A `SubAgent` bundles:
/// - an `Arc<AgentHarness<State, Ctx>>` that drives the child agent loop,
/// - a stable `name` and `description` (used for tool schemas and observability),
/// - an optional `system_prompt` prepended to every child run as a system
///   message (the "fixed prompt template").
///
/// Invoking a sub-agent always produces a *child run* one level deeper than the
/// caller's depth. The harness's [`crate::limits::RunLimits::max_depth`]
/// cap bounds how deep nested sub-agents may recurse; an invocation whose child
/// depth would exceed the cap fails with
/// [`crate::error::TinyAgentsError::SubAgentDepth`].
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

/// A persistent, *reusable* conversation with a single [`SubAgent`].
///
/// Where [`SubAgentTool`] runs a fresh, stateless child run per tool call, a
/// `SubAgentSession` keeps the **same** underlying [`SubAgent`] (and therefore
/// the same [`AgentHarness`]) alive across multiple turns and retains the full
/// conversation transcript between them. This is *post-completion reuse*: the
/// child run finishes normally, the orchestrator inspects/awaits human input,
/// then calls the same sub-agent again — distinct from *steering*, which
/// interrupts a still-running agent.
///
/// # Human-in-the-loop reuse flow
///
/// 1. `send` the first input (e.g. a user question). The session appends it to
///    the retained transcript, runs the sub-agent over the full transcript, and
///    folds the resulting assistant (and any tool) messages back in.
/// 2. Inspect the returned [`AgentRun`](crate::middleware::AgentRun) and obtain human input out-of-band.
/// 3. Wrap that human input as a [`Message::user`] and `send` it again. Because
///    the prior turn's messages are still in the transcript, the sub-agent
///    answers *with full context* — without being killed and restarted.
///
/// Each send after the first emits [`AgentEvent::SubAgentReused`][reused]
/// (alongside the usual [`SubAgentStarted`][started]/[`SubAgentCompleted`][completed]
/// bracket) so reuse is observable in the event stream.
///
/// [reused]: crate::events::AgentEvent::SubAgentReused
/// [started]: crate::events::AgentEvent::SubAgentStarted
/// [completed]: crate::events::AgentEvent::SubAgentCompleted
pub struct SubAgentSession<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The reused child agent. The same `Arc` is shared across every send, so
    /// the underlying harness is never reconstructed.
    pub(crate) subagent: Arc<SubAgent<State, Ctx>>,
    /// The accumulating conversation transcript carried across sends.
    pub(crate) transcript: Vec<Message>,
    /// Number of completed sends (turns) so far.
    pub(crate) turn: usize,
    /// Caller depth the child runs at; the child run is created at
    /// `parent_depth + 1` (default `0`, so the child runs at depth `1`).
    pub(crate) parent_depth: usize,
    /// Event sink the reuse lifecycle (and the child run's own events) are
    /// emitted on. Defaults to a fresh, unsubscribed sink.
    pub(crate) events: EventSink,
    /// Whether the fixed system prompt has been seeded into the transcript yet
    /// (it is prepended once, on the first send).
    pub(crate) seeded: bool,
}

/// A typed-parent dispatcher that exposes a [`SubAgent`] to a parent agent —
/// the surface that turns "agents calling agents" into an ordinary tool call.
///
/// When the parent model calls this tool, [`SubAgentTool`] runs the wrapped
/// sub-agent as a child run and returns the child's final assistant text as the
/// [`tinytools::ToolResult`]
/// content. This makes an entire agent composable as a single tool call, so a
/// model orchestrating tools is, transparently, a model orchestrating models.
///
/// Register this with [`crate::tool::ToolRegistry::register_dispatch`]. The
/// agent loop calls it with the live parent [`crate::context::RunContext`], so
/// its depth, cancellation, events, stores, workspace, steering, and streaming
/// state are inherited by the child. [`ChildDataPolicy`] makes the separate
/// application-data decision explicit.
pub struct SubAgentTool<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The wrapped child agent.
    pub(crate) subagent: Arc<SubAgent<State, Ctx>>,
    /// Tool name exposed to the model (defaults to the sub-agent name).
    pub(crate) tool_name: String,
    /// Explicit parent-to-child application-data policy.
    pub(crate) child_data: ChildDataPolicy<Ctx>,
    /// JSON Schema describing the tool's model-visible arguments.
    pub(crate) parameters: Value,
    /// Cached [`ToolDispatch::tool`] declaration, built once from
    /// `tool_name`/`parameters` on first access rather than allocated fresh
    /// (a new `Arc` with cloned schema `Value`) on every call — `tool()` is
    /// invoked several times per admitted call plus once per tool per run for
    /// `schemas()` (M-4). Safe to cache lazily: the `with_tool_name`/
    /// `with_parameters` builders consume `self` and are only meant to run
    /// before the tool is registered, never after.
    pub(crate) declaration: std::sync::OnceLock<Arc<dyn tinytools::Tool>>,
}
