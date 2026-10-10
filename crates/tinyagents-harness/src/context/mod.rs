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
//! See `types` for the field-level definitions.
//!
//! # Example
//!
//! ```
//! use tinyagents_harness::context::{RunConfig, RunContext};
//! use tinyagents_harness::events::AgentEvent;
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

mod stats;
mod types;

pub use stats::*;
pub use types::*;

use crate::cancel::CancellationToken;
use crate::error::Result;
use crate::events::{AgentEvent, EventRecord, EventSink};
use crate::ids::{RunId, ThreadId};
use crate::limits::{LimitTracker, RunLimits};
use crate::store::StoreRegistry;

#[derive(serde::Deserialize)]
struct RunConfigWire {
    run_id: RunId,
    #[serde(default)]
    thread_id: Option<ThreadId>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    metadata: serde_json::Value,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    max_model_calls: Option<usize>,
    #[serde(default)]
    max_tool_calls: Option<usize>,
    #[serde(default)]
    max_turn_output_tokens: Option<u32>,
    // Legacy wire fields, accepted only while reading old persisted configs.
    #[serde(default)]
    depth: Option<usize>,
    #[serde(default)]
    max_depth: Option<usize>,
    #[serde(default)]
    lineage: Option<RunLineage>,
}

impl<'de> serde::Deserialize<'de> for RunConfig {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let wire = RunConfigWire::deserialize(deserializer)?;
        let lineage = wire.lineage.unwrap_or_else(|| RunLineage {
            root_run_id: wire.run_id.clone(),
            parent_run_id: None,
            depth: wire.depth.unwrap_or(0),
            max_depth: wire.max_depth.unwrap_or(RunLimits::DEFAULT_MAX_DEPTH),
        });
        Ok(Self {
            run_id: wire.run_id,
            thread_id: wire.thread_id,
            tags: wire.tags,
            metadata: wire.metadata,
            timeout_ms: wire.timeout_ms,
            max_model_calls: wire.max_model_calls,
            max_tool_calls: wire.max_tool_calls,
            max_turn_output_tokens: wire.max_turn_output_tokens,
            lineage,
        })
    }
}

/// Mints the next process-unique [`RunContext::instance_id`].
fn next_context_instance_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ── RunConfig ─────────────────────────────────────────────────────────────────

impl RunConfig {
    /// Creates a run configuration with sensible defaults.
    ///
    /// Defaults: no thread, no tags, `null` metadata, no timeout, and **unset**
    /// call caps (`max_model_calls`/`max_tool_calls` are `None`), which resolve
    /// to the crate-wide [`RunLimits`] defaults of 25 and 50. Leaving them unset
    /// is what lets a harness-wide `RunPolicy` raise them; see
    /// [`RunConfig::max_model_calls`].
    pub fn new(run_id: impl Into<String>) -> Self {
        let run_id = RunId::new(run_id);
        Self {
            lineage: RunLineage {
                root_run_id: run_id.clone(),
                parent_run_id: None,
                depth: 0,
                max_depth: RunLimits::default().max_depth,
            },
            run_id,
            thread_id: None,
            tags: Vec::new(),
            metadata: serde_json::Value::Null,
            timeout_ms: None,
            max_model_calls: None,
            max_tool_calls: None,
            max_turn_output_tokens: None,
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

    /// Sets the maximum number of model calls permitted for this run,
    /// **explicitly**.
    ///
    /// An explicitly-set cap is a ceiling: the agent loop reconciles it with the
    /// harness [`RunPolicy`][crate::runtime::RunPolicy] by taking the
    /// stricter of the two, so a policy default can only ever tighten it.
    pub fn with_max_model_calls(mut self, n: usize) -> Self {
        self.max_model_calls = Some(n);
        self
    }

    /// Sets the maximum number of tool invocations permitted for this run,
    /// **explicitly**. Same ceiling semantics as
    /// [`RunConfig::with_max_model_calls`].
    pub fn with_max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = Some(n);
        self
    }

    /// The model-call cap actually applied to this run: the explicitly-set
    /// value, or the crate-default [`RunLimits`] cap when unset.
    pub fn effective_max_model_calls(&self) -> usize {
        self.max_model_calls
            .unwrap_or_else(|| RunLimits::default().max_model_calls)
    }

    /// The tool-call cap actually applied to this run: the explicitly-set
    /// value, or the crate-default [`RunLimits`] cap when unset.
    pub fn effective_max_tool_calls(&self) -> usize {
        self.max_tool_calls
            .unwrap_or_else(|| RunLimits::default().max_tool_calls)
    }

    /// Sets the maximum output tokens requested for each model turn.
    pub fn with_max_turn_output_tokens(mut self, n: u32) -> Self {
        self.max_turn_output_tokens = Some(n);
        self
    }

    /// Sets this run's depth in the sub-agent / recursion tree.
    ///
    /// Top-level runs are depth `0`; child runs spawned by a
    /// Subagents in `tinyagents-orchestration` carry the parent depth plus one.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.lineage.depth = depth;
        self
    }

    /// Sets the maximum sub-agent / recursion depth permitted for this run tree.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.lineage.max_depth = max_depth;
        self
    }

    /// This run's recursive depth.
    pub fn depth(&self) -> usize {
        self.lineage.depth
    }

    /// The inclusive recursion-depth cap for this run tree.
    pub fn max_depth(&self) -> usize {
        self.lineage.max_depth
    }

    /// Derives a child run's depth from its parent, enforcing the recursion cap.
    ///
    /// The single source of truth for the sub-agent depth guard: returns
    /// `parent_depth + 1`, or [`crate::error::TinyAgentsError::SubAgentDepth`]
    /// carrying `max_depth` when the child would exceed the cap. Every recursion
    /// surface in `tinyagents-orchestration`, its reuse-session tool,
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
    /// The returned config keeps the stricter of this config's and the child
    /// config's `max_depth`, inherits the thread, sets `depth = self.depth +
    /// 1`, and uses `child_run_id` as the run identity.
    /// It does **not** copy tags or metadata, which are run-specific.
    pub fn child(&self, mut config: Self) -> Result<Self> {
        let max_depth = self.max_depth().min(config.max_depth());
        let child_depth = Self::checked_child_depth(self.depth(), max_depth)?;
        config.lineage = RunLineage {
            root_run_id: self.lineage.root_run_id.clone(),
            parent_run_id: Some(self.run_id.clone()),
            depth: child_depth,
            max_depth,
        };
        if config.thread_id.is_none() {
            config.thread_id = self.thread_id.clone();
        }
        if config.max_turn_output_tokens.is_none() {
            config.max_turn_output_tokens = self.max_turn_output_tokens;
        }
        Ok(config)
    }

    /// Builds the [`RunLimits`] policy implied by this config.
    ///
    /// Carries `max_model_calls`, `max_tool_calls`, the timeout (as the
    /// wall-clock cap), and `max_depth` across; other limit fields use their
    /// defaults.
    fn to_run_limits(&self) -> RunLimits {
        RunLimits::default()
            .with_max_model_calls(self.effective_max_model_calls())
            .with_max_tool_calls(self.effective_max_tool_calls())
            .with_max_wall_clock_ms(self.timeout_ms)
            .with_max_depth(self.max_depth())
    }
}

// ── RunContext ────────────────────────────────────────────────────────────────

impl<Ctx> RunContext<Ctx> {
    /// Whether this context carries a host-owned invocation authority.
    ///
    /// Orchestration layers use this to fail closed when a hosted parent is
    /// sent through an explicit-model child entry point.
    pub fn is_hosted(&self) -> bool {
        self.host_authority.is_some()
    }
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
            instance_id: next_context_instance_id(),
            lifecycle: std::sync::Arc::new(()),
            config,
            data,
            frozen_system_prefix_len: None,
            model_profile: None,
            stores: StoreRegistry::new(),
            namespaced_store: None,
            state_view: None,
            events,
            limits,
            steering: None,
            run_queue: None,
            cancellation: CancellationToken::new(),
            control: std::sync::Arc::new(std::sync::Mutex::new(None)),
            repeat_noted: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            state_updates: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            tool_state_updates: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            terminate_votes: Vec::new(),
            truncated_call_positions: std::collections::HashSet::new(),
            truncated_repeat_positions: std::collections::HashSet::new(),
            batch_admissions: 0,
            workspace: None,
            on_error_dispatched: false,
            streaming: false,
            host_agent_id: None,
            host_authority: None,
            terminal_observer: None,
            turns: crate::agent_loop::TurnTracker::default(),
            halted_by_guard: None,
            last_limit: std::sync::Mutex::new(None),
            active_model_call: None,
            provider_started: false,
            call_provider_started: false,
            model_call_failed: false,
            call_streamed: false,
            prefix_epoch: 0,
            discarded_usage: Vec::new(),
            deferred_results: None,
            approved_calls: std::collections::HashSet::new(),
            refusal_metadata: std::collections::HashMap::new(),
            child_ordinal: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            tool_effect_ledger: None,
            tool_effect_ledger_failure: crate::tool::LedgerFailure::default(),
            nested_serial: std::sync::Arc::default(),
            compaction_sink: None,
        }
    }

    /// Pin the count of session-owned leading System tiers for this run.
    /// Later System compaction summaries remain in history rather than being
    /// promoted into the provider's reusable prompt prefix.
    #[must_use]
    pub fn with_frozen_system_prefix_len(mut self, count: usize) -> Self {
        self.frozen_system_prefix_len = Some(count);
        self
    }

    /// Attaches the resolutions for the deferred tool calls this run resumes
    /// (A2). The agent loop applies them to the unanswered tool calls on the
    /// transcript's last assistant row before making its next model call.
    /// Prefer [`crate::runtime::AgentHarness::resume_deferred`], which does
    /// this for you.
    #[must_use]
    pub fn with_deferred_results(mut self, results: crate::tool::DeferredToolResults) -> Self {
        self.deferred_results = Some(results);
        self
    }

    /// Takes the pending deferred-call resolutions, if any (A2).
    pub(crate) fn take_deferred_results(&mut self) -> Option<crate::tool::DeferredToolResults> {
        self.deferred_results.take()
    }

    /// Whether a human approved the tool call `call_id` on resume (A2).
    ///
    /// The agent loop consults this to skip its own deferral checks for an
    /// approved call; an approval gate implemented as a `before_tool`
    /// middleware should consult it too so it does not re-defer a call the
    /// human already decided on.
    pub fn is_call_approved(&self, call_id: &str) -> bool {
        self.approved_calls.contains(call_id)
    }

    /// Asks the loop to stamp `metadata` on the error result that answers
    /// `call_id` when a `before_tool` hook refuses the call (by returning
    /// `ToolFailed`/`ModelRetry`). The metadata is in place before any
    /// `after_tool` hook sees the result, whatever its position in the stack.
    /// Unused when the call is not refused.
    pub fn set_refusal_metadata(
        &mut self,
        call_id: impl Into<String>,
        metadata: serde_json::Value,
    ) {
        self.refusal_metadata.insert(call_id.into(), metadata);
    }

    /// Takes the metadata queued for the refused call `call_id`.
    pub(crate) fn take_refusal_metadata(&mut self, call_id: &str) -> Option<serde_json::Value> {
        self.refusal_metadata.remove(call_id)
    }

    /// Marks `call_id` as approved for this run (A2).
    pub(crate) fn mark_call_approved(&mut self, call_id: impl Into<String>) {
        self.approved_calls.insert(call_id.into());
    }

    /// Returns the next value from this context's own child-ordinal counter
    /// (starting at `0`), advancing it.
    ///
    /// The counter is per-context, not process-global: a freshly constructed
    /// context (including a child context, which never inherits its parent's
    /// counter) always starts at `0`. Callers that spawn deterministically
    /// named children — `tinyagents_orchestration::SubAgent`, for one — use this
    /// instead of a process-wide sequence so two processes calling the same
    /// parent context's child spawner in the same order derive identical
    /// ordinals, and therefore identical child run ids (M-2).
    pub fn next_child_ordinal(&self) -> u64 {
        self.child_ordinal
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Builds an isolated child context from this live parent context,
    /// propagating the parent's host authority.
    ///
    /// A child gets a new run id, lineage record, [`LimitTracker`], control
    /// slot, and instance id.  It deliberately shares the capabilities that
    /// describe one recursive operation: cancellation, events, stores,
    /// workspace policy, steering, streaming mode, thread identity, output
    /// cap, and depth cap. Metadata is shallow-merged automatically: any key
    /// set on `child_config.metadata` overlays the parent's metadata object
    /// (see `shallow_merge_metadata`), so callers only need to pass the
    /// child-specific keys.
    ///
    /// This keeps the child's `Ctx` type identical to the parent's, which is
    /// what makes propagating [`Self::host_authority`] sound: the type-erased
    /// authority installed by a hosted invocation is keyed to the exact
    /// `(State, Ctx)` pair it was constructed for, and this method is the only
    /// place that carries it forward. A recursive call that needs a
    /// *different* `Ctx` type must go through [`Self::child_with_data`]
    /// instead, which never propagates host authority.
    pub fn child(&self, child_config: RunConfig, data: Ctx) -> Result<RunContext<Ctx>> {
        let mut child = self.child_without_authority(child_config, data)?;
        child.host_authority = self.host_authority.clone();
        Ok(child)
    }

    /// Builds an isolated child context whose user data type may differ from
    /// this context's, deliberately *not* propagating host authority.
    ///
    /// Use this whenever the child's `Ctx` differs from the parent's (for
    /// example, a differently-typed sub-harness). Because [`RunContext`] does
    /// not track its `State` type parameter at all, and the erased host
    /// authority is keyed to a specific `(State, Ctx)` pair, there is no sound
    /// way to check at this boundary whether the parent's authority would
    /// still apply to the child's types. Rather than guess, the child simply
    /// starts unhosted; a caller that legitimately needs to delegate hosted
    /// authority across a `Ctx` change must do so explicitly through the
    /// hosted subagent entry points, which re-derive authority from the live
    /// host capability bundle rather than reinterpreting the parent's.
    pub fn child_with_data<ChildCtx>(
        &self,
        child_config: RunConfig,
        data: ChildCtx,
    ) -> Result<RunContext<ChildCtx>> {
        self.child_without_authority(child_config, data)
    }

    fn child_without_authority<ChildCtx>(
        &self,
        child_config: RunConfig,
        data: ChildCtx,
    ) -> Result<RunContext<ChildCtx>> {
        let mut config = self.config.child(child_config)?;
        config.metadata = shallow_merge_metadata(&self.config.metadata, config.metadata);
        let child_run_id = config.run_id.clone();
        // Derive a per-child handle (not a bare clone): it shares the parent's
        // queue/policy but only drains commands addressed to *this* child's
        // run id, `SteeringTarget::Root`-addressed commands stay with the
        // parent, and its pause/checkpoint state is its own (I-5).
        let steering = self
            .steering
            .as_ref()
            .map(|handle| handle.for_child(child_run_id));
        let mut child = RunContext::new(config, data)
            .with_stores(self.stores.clone())
            .with_optional_namespaced_store(self.namespaced_store.clone())
            .with_optional_state_view(self.state_view.clone())
            .with_events(self.events.clone())
            // A linked child token: a parent cancel cascades down, but
            // cancelling this child leaves the parent and siblings running.
            .with_cancellation(self.cancellation.child_token())
            .with_optional_steering(steering)
            .with_optional_workspace(self.workspace.clone())
            .with_streaming(self.streaming);
        child.host_agent_id = self.host_agent_id.clone();
        child.tool_effect_ledger = self.tool_effect_ledger.clone();
        child.tool_effect_ledger_failure = self.tool_effect_ledger_failure;
        child.compaction_sink = self.compaction_sink.clone();
        Ok(child)
    }

    /// Installs runtime-owned terminal bookkeeping for this invocation.
    ///
    /// This is crate-private because public callers must not couple their
    /// behavior to future cancellation mechanics.
    pub(crate) fn set_terminal_observer(&mut self, observer: TerminalObserver) {
        self.terminal_observer = Some(observer);
    }

    /// Returns this run's recursive ancestry.
    pub fn lineage(&self) -> &RunLineage {
        &self.config.lineage
    }

    /// Returns this context's process-unique instance id.
    ///
    /// Two concurrent runs can carry the same [`RunConfig::run_id`] (it is a
    /// caller-supplied label), so keep per-run bookkeeping keyed on this
    /// instead when it is shared across runs.
    pub fn instance_id(&self) -> u64 {
        self.instance_id
    }

    /// Records that the middleware stack already dispatched `on_error` for the
    /// error currently unwinding this run.
    pub(crate) fn mark_on_error_dispatched(&mut self) {
        self.on_error_dispatched = true;
    }

    /// Takes (and clears) the flag set by [`Self::mark_on_error_dispatched`].
    ///
    /// `true` means the stack already delivered `on_error` for this failure, so
    /// the driver must not dispatch it again.
    pub(crate) fn take_on_error_dispatched(&mut self) -> bool {
        std::mem::take(&mut self.on_error_dispatched)
    }

    /// Attaches a host-owned workspace descriptor that is threaded into every
    /// [`ToolExecutionContext`][crate::tool::ToolExecutionContext] this
    /// run creates, so tools read their allowed root from context. Preparation,
    /// cleanup, and sandbox policy belong to the host; around-agent middleware
    /// can set this before calling the inner run and clean it up afterward.
    pub fn with_workspace(mut self, workspace: tinytools::WorkspaceDescriptor) -> Self {
        self.workspace = Some(workspace);
        self
    }

    fn with_optional_workspace(
        mut self,
        workspace: Option<tinytools::WorkspaceDescriptor>,
    ) -> Self {
        self.workspace = workspace;
        self
    }

    fn with_optional_steering(mut self, steering: Option<crate::steering::SteeringHandle>) -> Self {
        self.steering = steering;
        self
    }

    /// Records the usage of a provider response a wrap middleware is about to
    /// discard and re-request. The provider billed it, so the agent loop adds
    /// it to the run's usage (and the host budget) when it accounts for the
    /// call that replaces it.
    pub fn record_discarded_usage(&mut self, usage: tinyinference_llm::usage::Usage) {
        self.discarded_usage.push(usage);
    }

    /// Declares that a middleware rewrote the prompt prefix on purpose
    /// (compaction, truncation): the next provider cache read is expected to
    /// be cold, so it is not a cache miss.
    pub fn mark_prompt_prefix_changed(&mut self) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        self.prefix_epoch = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The value [`Self::mark_prompt_prefix_changed`] last set (`0` if never).
    pub fn prompt_prefix_epoch(&self) -> u64 {
        self.prefix_epoch
    }

    pub(crate) fn take_discarded_usage(&mut self) -> Vec<tinyinference_llm::usage::Usage> {
        std::mem::take(&mut self.discarded_usage)
    }

    fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
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
    ///
    /// [`MiddlewareControl::Continue`] is never installed: it carries no
    /// instruction, so requesting it is a no-op regardless of what (if
    /// anything) is already pending.
    pub fn request_control(&self, control: MiddlewareControl) {
        if matches!(control, MiddlewareControl::Continue) {
            return;
        }
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

    /// Records that a middleware just noted a repeat on a tool result (an
    /// identical call re-issued, an identical reply). The agent loop reads it
    /// once before its next model call ([`Self::take_repeat_noted`]); a
    /// model running without reasoning that repeats itself is handed
    /// reasoning back.
    pub fn note_repeat(&self) {
        self.repeat_noted
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Asks for reasoning on the next model call while the loop's reasoning
    /// fallback has switched it off. Same flag as [`Self::note_repeat`]: a
    /// middleware about to issue a call where thinking is worth a dead call's
    /// bounded cost (the finish check, which has to ask what the request
    /// implied) uses this. With the fallback disabled reasoning is never off,
    /// so the request is a no-op by construction; it never raises the effort
    /// above what the request already asks for.
    pub fn request_reasoning(&self) {
        self.repeat_noted
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a repeat was noted (or reasoning requested) since the last
    /// take; clears it.
    pub fn take_repeat_noted(&self) -> bool {
        self.repeat_noted
            .swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    /// Queues a [`StateUpdate`] for the host to apply.
    ///
    /// The agent loop only ever holds `state: &State` (a shared reference), so
    /// [`MiddlewareControl::UpdateState`] cannot be applied in place; the loop
    /// pushes it here instead of discarding it. Called by
    /// [`crate::agent_loop`]'s control-checkpoint handling; a host drains the
    /// queue with [`Self::take_state_updates`] and applies each update against
    /// its own `&mut State` between runs (or between turns, via its own
    /// checkpoint).
    pub fn push_state_update(&self, update: StateUpdate) {
        if let Ok(mut guard) = self.state_updates.lock() {
            guard.push(update);
        }
    }

    /// Drains every [`StateUpdate`] queued so far, in request order.
    pub fn take_state_updates(&self) -> Vec<StateUpdate> {
        self.state_updates
            .lock()
            .ok()
            .map(|mut guard| std::mem::take(&mut *guard))
            .unwrap_or_default()
    }

    /// Queues a raw JSON state update a tool requested via
    /// [`tinytools::ToolControl::state_update`][tc].
    ///
    /// A canonical tool has no access to the harness's typed `State`, so its
    /// state update travels as `serde_json::Value` rather than a
    /// [`StateUpdate`] closure. Kept as a separate queue (not merged into
    /// [`Self::push_state_update`]) so a host can tell a middleware-originated
    /// typed update from a tool-originated JSON one without downcasting.
    ///
    /// [tc]: tinytools::ToolControl::state_update
    pub fn push_tool_state_update(&self, update: serde_json::Value) {
        if let Ok(mut guard) = self.tool_state_updates.lock() {
            guard.push(update);
        }
    }

    /// Drains every raw JSON tool state update queued so far, in request
    /// order. See [`Self::push_tool_state_update`].
    pub fn take_tool_state_updates(&self) -> Vec<serde_json::Value> {
        self.tool_state_updates
            .lock()
            .ok()
            .map(|mut guard| std::mem::take(&mut *guard))
            .unwrap_or_default()
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

    /// Attaches the hierarchical store every tool in this run receives as
    /// [`ToolExecutionContext::store`][crate::tool::ToolExecutionContext::store]
    /// (B1). See [`RunContext::namespaced_store`].
    #[must_use]
    pub fn with_namespaced_store(
        mut self,
        store: std::sync::Arc<dyn crate::store::namespaced::NamespacedStore>,
    ) -> Self {
        self.namespaced_store = Some(store);
        self
    }

    fn with_optional_namespaced_store(
        mut self,
        store: Option<std::sync::Arc<dyn crate::store::namespaced::NamespacedStore>>,
    ) -> Self {
        self.namespaced_store = store;
        self
    }

    /// Attaches a read-only snapshot of the application state every tool in
    /// this run can recover with
    /// [`ToolExecutionContext::state::<S>()`][crate::tool::ToolExecutionContext::state]
    /// (B1). See [`RunContext::state_view`] for why this is an owned `Arc`
    /// the host supplies rather than the loop's own `&State`.
    #[must_use]
    pub fn with_state_view<S: std::any::Any + Send + Sync>(
        mut self,
        state: std::sync::Arc<S>,
    ) -> Self {
        self.state_view = Some(state);
        self
    }

    fn with_optional_state_view(
        mut self,
        state: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Self {
        self.state_view = state;
        self
    }

    /// Replaces the event sink with a (possibly shared) `events`.
    pub fn with_events(mut self, events: EventSink) -> Self {
        self.events = events;
        self
    }

    /// Attaches a [`crate::steering::SteeringHandle`] so an
    /// orchestrator can steer this run at safe checkpoints.
    ///
    /// The agent loop drains the handle before each model call via
    /// [`crate::steering::apply_pending_steering`]. Without this the
    /// run accepts no steering.
    ///
    /// Binds the handle to this run's id as the **root** of its steering tree
    /// (see [`crate::steering::SteeringTarget::Root`]); a child run created
    /// from this context via [`Self::child`]/[`Self::child_with_data`] gets a
    /// derived handle scoped to its own id instead of sharing this binding
    /// (I-5).
    pub fn with_steering(mut self, steering: crate::steering::SteeringHandle) -> Self {
        let root_run_id = self.lineage().root_run_id.clone();
        self.steering = Some(steering.bind_root(root_run_id));
        self
    }

    /// Attaches a [`crate::run_queue::RunQueueHandle`] so messages pushed
    /// from outside the run reach the transcript at the loop's safe turn
    /// boundaries (A4). See [`RunContext::run_queue`] for the drain points
    /// and [`crate::runtime::RunPolicy::queue_mode`] for how many items each
    /// boundary takes. Without this the loop consumes no queued messages.
    pub fn with_run_queue(mut self, queue: crate::run_queue::RunQueueHandle) -> Self {
        self.run_queue = Some(queue);
        self
    }

    /// Attaches a durable tool-effect ledger (B5), so the agent loop writes a
    /// `started` row before each tool call executes and a `completed`/
    /// `failed` row after it settles. `None` (the default) disables all
    /// ledger writes.
    ///
    /// See [`crate::tool::ToolEffectLedger`] and
    /// [`RunContext::with_tool_effect_ledger_failure`] for how a `started`
    /// write failure is handled.
    #[must_use]
    pub fn with_tool_effect_ledger(
        mut self,
        ledger: std::sync::Arc<dyn crate::tool::ToolEffectLedger>,
    ) -> Self {
        self.tool_effect_ledger = Some(ledger);
        self
    }

    /// Sets how the agent loop reacts when [`crate::tool::ToolEffectLedger::started`]
    /// itself fails. Defaults to [`crate::tool::LedgerFailure::Abort`].
    #[must_use]
    pub fn with_tool_effect_ledger_failure(mut self, failure: crate::tool::LedgerFailure) -> Self {
        self.tool_effect_ledger_failure = failure;
        self
    }

    /// Attaches a durable [`crate::summarization::CompactionSink`] so every
    /// [`crate::summarization::CompactionRecord`] a compaction produces on
    /// this run is persisted (typically into a session's entry tree),
    /// instead of only living in
    /// [`crate::middleware::ContextCompressionMiddleware::records`]'s
    /// in-process buffer. `None` (the default) disables persistence.
    #[must_use]
    pub fn with_compaction_sink(
        mut self,
        sink: std::sync::Arc<dyn crate::summarization::CompactionSink>,
    ) -> Self {
        self.compaction_sink = Some(sink);
        self
    }

    /// Emits `event` on this run's event sink, returning the recorded entry.
    pub fn emit(&self, event: AgentEvent) -> EventRecord {
        // Remember which cap tripped last, so a `LimitExceeded` failure can be
        // classified with its kind (the error itself carries only text).
        if let AgentEvent::LimitReached { kind } = &event {
            *self
                .last_limit
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(*kind);
        }
        self.events.emit(event)
    }

    /// Forgets the cached limit kind. The loop calls this at each turn's limit
    /// checks so a kind emitted earlier cannot be attributed to a later,
    /// unrelated `LimitExceeded` that emitted no event of its own.
    pub(crate) fn clear_last_limit(&self) {
        self.last_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
    }

    /// The most recent `LimitReached` kind, without consuming it.
    pub fn peek_last_limit(&self) -> Option<crate::events::LimitKind> {
        *self
            .last_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn take_last_limit(&self) -> Option<crate::events::LimitKind> {
        self.last_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Records that a provider call was dispatched. Called by the model base
    /// call of each loop driver immediately before it reaches the provider.
    pub fn mark_provider_started(&mut self) {
        self.provider_started = true;
        self.call_provider_started = true;
    }

    /// Records that a summarizer (a compaction, outside the model-call layer)
    /// dispatched a provider call. Sets only the run-wide flag: the per-call
    /// flag still describes the main model call.
    pub(crate) fn mark_summarizer_dispatched(&mut self) {
        tracing::debug!("[tinyagents::run] summarizer dispatched a provider call");
        self.provider_started = true;
    }

    #[doc(hidden)]
    pub fn mark_model_call_failed(&mut self) {
        self.model_call_failed = true;
    }

    /// Whether a model call surfaced an error (see `model_call_failed`).
    pub fn model_call_failed(&self) -> bool {
        self.model_call_failed
    }

    /// Marks the start of a model call: clears the per-call provider flag.
    pub fn begin_model_call(&mut self) {
        self.call_provider_started = false;
        self.model_call_failed = false;
    }

    /// Whether the current (or most recent) model call reached the provider.
    pub fn call_provider_started(&self) -> bool {
        self.call_provider_started
    }

    /// Whether any provider call has been dispatched during this run.
    pub fn provider_started(&self) -> bool {
        self.provider_started
    }

    /// Takes the repeat-progress guard's halt summary, if one was latched.
    pub fn take_halted_by_guard(&mut self) -> Option<String> {
        self.halted_by_guard.take()
    }

    /// Resets lifecycle tracking for a run whose input already contains seed messages.
    pub fn reset_turn_tracker(&mut self, seed_len: usize) {
        self.turns = crate::agent_loop::TurnTracker::new(seed_len);
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
        self.config.depth()
    }

    /// Races `fut` against this run's cooperative cancellation and, when
    /// `deadline` is `Some`, a wall-clock timeout — the one home for the
    /// `tokio::select! { biased; _ = cancelled() => .., _ = timeout(remaining,
    /// fut) => .. }` pattern that used to be copied at every host/provider I-O
    /// boundary in the agent loop (R-1).
    ///
    /// `timeout_message` is only invoked when the timeout branch actually
    /// fires, so callers can build a call-specific message (which fields it
    /// names, which deadline it blames) without paying for the `format!` on
    /// the hot, non-timeout path. `fut`'s own error type must convert from
    /// [`TinyAgentsError`] so `Cancelled`/`Timeout` can be returned through
    /// the same `Result` the callee already returns.
    pub(crate) async fn bounded<T, E>(
        &self,
        deadline: Option<std::time::Duration>,
        fut: impl std::future::Future<Output = std::result::Result<T, E>>,
        timeout_message: impl FnOnce() -> String,
    ) -> std::result::Result<T, E>
    where
        E: From<crate::error::TinyAgentsError>,
    {
        match deadline {
            Some(remaining) => tokio::select! {
                biased;
                _ = self.cancellation.cancelled() => {
                    Err(crate::error::TinyAgentsError::Cancelled.into())
                }
                result = tokio::time::timeout(remaining, fut) => match result {
                    Ok(inner) => inner,
                    Err(_) => Err(crate::error::TinyAgentsError::Timeout(timeout_message()).into()),
                },
            },
            None => tokio::select! {
                biased;
                _ = self.cancellation.cancelled() => {
                    Err(crate::error::TinyAgentsError::Cancelled.into())
                }
                result = fut => result,
            },
        }
    }

    /// Returns the maximum sub-agent / recursion depth permitted for this run
    /// tree.
    pub fn max_depth(&self) -> usize {
        self.config.max_depth()
    }

    /// Records one model call against the run's limits.
    ///
    /// Returns an error if the configured model-call cap is exceeded.
    pub fn record_model_call(&mut self) -> Result<()> {
        self.last_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        self.limits.record_model_call()
    }

    /// Records one tool call against the run's limits.
    ///
    /// Returns an error if the configured tool-call cap is exceeded.
    pub fn record_tool_call(&mut self) -> Result<()> {
        self.last_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
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
    /// Delegates to [`crate::limits::LimitTracker::remaining_wall_clock`].
    /// Returns `None` when the run has no configured timeout (so the caller
    /// should not bound work by time); otherwise the remaining budget,
    /// saturating at zero once the deadline has elapsed. The agent loop uses
    /// this to bound each individual model call.
    pub fn remaining_wall_clock(&self) -> Option<std::time::Duration> {
        self.limits.remaining_wall_clock()
    }
}

/// Applies the child metadata semantics used by [`RunContext::child`]: an
/// object-valued child metadata shallow-merges onto (and overrides) the
/// parent's object, a non-null non-object child metadata replaces the
/// parent's wholesale, and a `null` child metadata inherits the parent's
/// unchanged.
fn shallow_merge_metadata(
    parent: &serde_json::Value,
    child: serde_json::Value,
) -> serde_json::Value {
    if child.is_null() {
        return parent.clone();
    }
    match (parent, child) {
        (serde_json::Value::Object(parent), serde_json::Value::Object(child)) => {
            let mut merged = parent.clone();
            merged.extend(child);
            serde_json::Value::Object(merged)
        }
        (_, child) => child,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
