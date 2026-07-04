//! Run configuration and runtime context.
//!
//! [`RunContext`] is the unit of recursion in the runtime: every nested layer —
//! a sub-agent, a sub-graph, a REPL-driven sub-call — runs inside its own
//! context, and [`RunConfig::depth`]/[`RunConfig::max_depth`] plus
//! [`RunConfig::child`] track and bound how deep that recursion may go while a
//! shared [`CancellationToken`] and event sink let signals and observability
//! flow across the whole tree.
//!
//! This module owns the authoritative run-context contract that downstream
//! middleware, the agent loop, and graph nodes code against:
//!
//! - [`RunConfig`] is the declarative, serializable description of a run.
//! - [`RunContext`] is the live handle bundling that config with the run's
//!   stores, event sink, limit tracker, and arbitrary user data.
//!
//! See [`types`] for the field-level definitions.
//!
//! # Example
//!
//! ```
//! use tinyagents::harness::context::{RunConfig, RunContext};
//! use tinyagents::harness::events::AgentEvent;
//!
//! let config = RunConfig::new("run-1").with_max_model_calls(2);
//! let mut ctx: RunContext = RunContext::new(config, ());
//!
//! ctx.emit(AgentEvent::RunStarted {
//!     run_id: ctx.run_id().clone(),
//!     thread_id: None,
//! });
//! ctx.record_model_call().expect("within limit");
//! assert_eq!(ctx.limits.model_calls(), 1);
//! ```

mod types;

pub use types::*;

use crate::error::Result;
use crate::harness::cancel::CancellationToken;
use crate::harness::events::{AgentEvent, EventRecord, EventSink};
use crate::harness::ids::{RunId, ThreadId};
use crate::harness::limits::{LimitTracker, RunLimits};
use crate::harness::store::StoreRegistry;

// ── RunConfig ─────────────────────────────────────────────────────────────────

impl RunConfig {
    /// Creates a run configuration with sensible defaults.
    ///
    /// Defaults: no thread, no tags, `null` metadata, no timeout,
    /// `max_model_calls = 25`, and `max_tool_calls = 50`. These mirror the
    /// crate-wide [`RunLimits`] defaults.
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: RunId::new(run_id),
            thread_id: None,
            tags: Vec::new(),
            metadata: serde_json::Value::Null,
            timeout_ms: None,
            max_model_calls: 25,
            max_tool_calls: 50,
            max_turn_output_tokens: None,
            depth: 0,
            max_depth: RunLimits::default().max_depth,
        }
    }

    /// Associates this run with a conversation thread.
    pub fn with_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(ThreadId::new(thread_id));
        self
    }

    /// Appends a classification tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Replaces the metadata blob.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets a wall-clock timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Sets the maximum number of model calls permitted for this run.
    pub fn with_max_model_calls(mut self, n: usize) -> Self {
        self.max_model_calls = n;
        self
    }

    /// Sets the maximum number of tool invocations permitted for this run.
    pub fn with_max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = n;
        self
    }

    /// Sets the maximum output tokens requested for each model turn.
    pub fn with_max_turn_output_tokens(mut self, n: u32) -> Self {
        self.max_turn_output_tokens = Some(n);
        self
    }

    /// Sets this run's depth in the sub-agent / recursion tree.
    ///
    /// Top-level runs are depth `0`; child runs spawned by a
    /// [`crate::harness::subagent::SubAgent`] carry the parent depth plus one.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Sets the maximum sub-agent / recursion depth permitted for this run tree.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Derives a child run's depth from its parent, enforcing the recursion cap.
    ///
    /// The single source of truth for the sub-agent depth guard: returns
    /// `parent_depth + 1`, or [`crate::error::TinyAgentsError::SubAgentDepth`]
    /// carrying `max_depth` when the child would exceed the cap. Every recursion
    /// surface — [`crate::harness::subagent::SubAgent`], its reuse-session tool,
    /// and the REPL sub-run builtin — funnels its `depth + 1` check through here
    /// so the fail-closed guard cannot drift out of sync between them.
    pub fn checked_child_depth(parent_depth: usize, max_depth: usize) -> Result<usize> {
        let child_depth = parent_depth + 1;
        if child_depth > max_depth {
            return Err(crate::error::TinyAgentsError::SubAgentDepth(max_depth));
        }
        Ok(child_depth)
    }

    /// Builds the [`RunConfig`] for a child run one level deeper than this one.
    ///
    /// The returned config keeps this config's `max_depth` and thread, sets
    /// `depth = self.depth + 1`, and uses `child_run_id` as the run identity.
    /// It does **not** copy tags or metadata, which are run-specific.
    pub fn child(&self, child_run_id: impl Into<String>) -> Self {
        let mut config = Self::new(child_run_id);
        config.depth = self.depth + 1;
        config.max_depth = self.max_depth;
        config.thread_id = self.thread_id.clone();
        config.max_turn_output_tokens = self.max_turn_output_tokens;
        config
    }

    /// Builds the [`RunLimits`] policy implied by this config.
    ///
    /// Carries `max_model_calls`, `max_tool_calls`, the timeout (as the
    /// wall-clock cap), and `max_depth` across; other limit fields use their
    /// defaults.
    fn to_run_limits(&self) -> RunLimits {
        RunLimits::default()
            .with_max_model_calls(self.max_model_calls)
            .with_max_tool_calls(self.max_tool_calls)
            .with_max_wall_clock_ms(self.timeout_ms)
            .with_max_depth(self.max_depth)
    }
}

// ── RunContext ────────────────────────────────────────────────────────────────

impl<Ctx> RunContext<Ctx> {
    /// Builds a live run context from `config` and user `data`.
    ///
    /// A default [`StoreRegistry`] and [`EventSink`] are created, and a
    /// [`LimitTracker`] is derived from the config's limits (model-call cap,
    /// tool-call cap, and timeout). Use [`Self::with_stores`] /
    /// [`Self::with_events`] to inject shared instances instead.
    pub fn new(config: RunConfig, data: Ctx) -> Self {
        let limits = LimitTracker::new(config.to_run_limits());
        // Seed the owned sink's event-id prefix from the run id so this run's
        // event ids are stable and collision-free across process restarts
        // (a durable journal replayed after a restart re-mints the same ids).
        let events = EventSink::with_stream_id(config.run_id.as_str());
        Self {
            config,
            data,
            stores: StoreRegistry::new(),
            events,
            limits,
            steering: None,
            cancellation: CancellationToken::new(),
            control: std::sync::Arc::new(std::sync::Mutex::new(None)),
            workspace: None,
            streaming: false,
        }
    }

    /// Attaches an isolated workspace descriptor that is threaded into every
    /// [`ToolExecutionContext`][crate::harness::tool::ToolExecutionContext] this
    /// run creates, so tools read their allowed root from context. To prepare
    /// and tear down the environment via a
    /// [`WorkspaceIsolation`][crate::harness::workspace::WorkspaceIsolation]
    /// provider (emitting the workspace lifecycle events), use
    /// [`crate::harness::workspace::prepare_workspace`] to obtain the descriptor
    /// first.
    pub fn with_workspace(
        mut self,
        workspace: crate::harness::workspace::WorkspaceDescriptor,
    ) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// Requests a [`MiddlewareControl`] outcome. The agent loop drains and acts
    /// on it at its next safe checkpoint (after the current model response).
    ///
    /// When an undrained request is already pending, the higher-precedence one
    /// wins (see [`MiddlewareControl::precedence`]); ties keep the earlier
    /// request. This gives competing middleware layers a deterministic outcome
    /// instead of last-writer-wins — e.g. a pause request is never downgraded to
    /// a stop by a later, weaker request.
    pub fn request_control(&self, control: MiddlewareControl) {
        if let Ok(mut guard) = self.control.lock() {
            let replace = match guard.as_ref() {
                Some(existing) => control.precedence() > existing.precedence(),
                None => true,
            };
            if replace {
                *guard = Some(control);
            }
        }
    }

    /// Takes any pending [`MiddlewareControl`] request, clearing it.
    pub fn take_control(&self) -> Option<MiddlewareControl> {
        self.control.lock().ok().and_then(|mut guard| guard.take())
    }

    /// Attaches a [`CancellationToken`] so an orchestrator can request that this
    /// run stop cooperatively at its next safe checkpoint.
    ///
    /// The agent loop polls [`CancellationToken::is_cancelled`] before each
    /// model call and before each tool call (alongside steering), and the
    /// streaming pipeline races [`CancellationToken::cancelled`] against the
    /// provider stream. When the token is cancelled the run ends with
    /// [`crate::error::TinyAgentsError::Cancelled`]. Without this, the run
    /// carries a fresh token that is never cancelled.
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Replaces the store registry with a (possibly shared) `stores`.
    pub fn with_stores(mut self, stores: StoreRegistry) -> Self {
        self.stores = stores;
        self
    }

    /// Replaces the event sink with a (possibly shared) `events`.
    pub fn with_events(mut self, events: EventSink) -> Self {
        self.events = events;
        self
    }

    /// Attaches a [`crate::harness::steering::SteeringHandle`] so an
    /// orchestrator can steer this run at safe checkpoints.
    ///
    /// The agent loop drains the handle before each model call via
    /// [`crate::harness::steering::apply_pending_steering`]. Without this the
    /// run accepts no steering.
    pub fn with_steering(mut self, steering: crate::harness::steering::SteeringHandle) -> Self {
        self.steering = Some(steering);
        self
    }

    /// Emits `event` on this run's event sink, returning the recorded entry.
    pub fn emit(&self, event: AgentEvent) -> EventRecord {
        self.events.emit(event)
    }

    /// Returns this run's identifier.
    pub fn run_id(&self) -> &RunId {
        &self.config.run_id
    }

    /// Returns this run's thread id, if it is threaded.
    pub fn thread_id(&self) -> Option<&ThreadId> {
        self.config.thread_id.as_ref()
    }

    /// Returns this run's depth in the sub-agent / recursion tree.
    pub fn depth(&self) -> usize {
        self.config.depth
    }

    /// Returns the maximum sub-agent / recursion depth permitted for this run
    /// tree.
    pub fn max_depth(&self) -> usize {
        self.config.max_depth
    }

    /// Records one model call against the run's limits.
    ///
    /// Returns an error if the configured model-call cap is exceeded.
    pub fn record_model_call(&mut self) -> Result<()> {
        self.limits.record_model_call()
    }

    /// Records one tool call against the run's limits.
    ///
    /// Returns an error if the configured tool-call cap is exceeded.
    pub fn record_tool_call(&mut self) -> Result<()> {
        self.limits.record_tool_call()
    }

    /// Checks whether the run has exceeded its wall-clock deadline.
    ///
    /// Returns `Ok(())` when no timeout is configured or it has not elapsed.
    pub fn check_deadline(&mut self) -> Result<()> {
        self.limits.check_wall_clock()
    }

    /// Returns the wall-clock budget still remaining before the run's deadline.
    ///
    /// Delegates to [`crate::harness::limits::LimitTracker::remaining_wall_clock`].
    /// Returns `None` when the run has no configured timeout (so the caller
    /// should not bound work by time); otherwise the remaining budget,
    /// saturating at zero once the deadline has elapsed. The agent loop uses
    /// this to bound each individual model call.
    pub fn remaining_wall_clock(&self) -> Option<std::time::Duration> {
        self.limits.remaining_wall_clock()
    }
}

#[cfg(test)]
mod test;
