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

use crate::harness::events::EventSink;
use crate::harness::message::Message;
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
/// 2. Inspect the returned [`AgentRun`] and obtain human input out-of-band.
/// 3. Wrap that human input as a [`Message::user`] and `send` it again. Because
///    the prior turn's messages are still in the transcript, the sub-agent
///    answers *with full context* — without being killed and restarted.
///
/// Each send after the first emits [`AgentEvent::SubAgentReused`][reused]
/// (alongside the usual [`SubAgentStarted`][started]/[`SubAgentCompleted`][completed]
/// bracket) so reuse is observable in the event stream.
///
/// [reused]: crate::harness::events::AgentEvent::SubAgentReused
/// [started]: crate::harness::events::AgentEvent::SubAgentStarted
/// [completed]: crate::harness::events::AgentEvent::SubAgentCompleted
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
