//! Type definitions for the harness observability and events layer.
//!
//! These types are the vocabulary for observing a recursive run tree: the
//! [`AgentEvent`] enum names every lifecycle transition (including the
//! sub-agent boundaries that mark one level of recursion), [`EventRecord`]
//! gives each event a replayable offset, and [`HarnessRunStatus`] threads the
//! `root_run_id` / `parent_run_id` lineage that ties a child run back to its
//! parent.
//!
//! All structs, enums, and traits in this module form the public surface of
//! `crate::harness::events`. Implementations, free functions, and tests live in
//! the sibling `mod.rs` and `test.rs` files.

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::harness::cost::CostTotals;
use crate::harness::ids::{
    CallId, ComponentId, EventId, ExecutionStatus, HarnessPhase, RunId, ThreadId,
};
use crate::harness::message::MessageDelta;
use crate::harness::usage::{Usage, UsageTotals};

// ---------------------------------------------------------------------------
// AgentEvent
// ---------------------------------------------------------------------------

/// A typed lifecycle event emitted by the harness during a run.
///
/// Every significant state transition — model calls, tool invocations,
/// middleware hooks, routing decisions, and run boundaries — is represented as
/// a distinct enum variant so downstream listeners receive structured data
/// rather than opaque strings.
///
/// Serialized with `"kind"` as a tag field so JSON consumers can dispatch on
/// the event type without inspecting nested fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AgentEvent {
    /// A new harness run has been initiated.
    RunStarted {
        /// Unique identifier assigned to this run.
        run_id: RunId,
        /// Thread the run belongs to, when provided by the caller.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thread_id: Option<ThreadId>,
    },

    /// A model call is about to be dispatched to a provider.
    ModelStarted {
        /// Identifier for this model call, correlates deltas and completion.
        call_id: CallId,
        /// Registry name or provider model id selected for this call.
        model: String,
    },

    /// An incremental chunk of model output arrived during streaming.
    ModelDelta {
        /// The run that produced this delta. Attributed explicitly so a UI can
        /// route a delta to its run/thread lineage without depending on which
        /// sink it arrived on — sinks are shared across a recursive run tree.
        run_id: RunId,
        /// Identifier for the model call that produced this delta.
        call_id: CallId,
        /// Incremental text and/or tool-call fragment.
        delta: MessageDelta,
    },

    /// A model call completed successfully.
    ModelCompleted {
        /// Identifier for the model call that completed.
        call_id: CallId,
        /// Wall-clock time the model call *started*, in Unix-epoch
        /// milliseconds. Captured by the agent loop when it dispatches the
        /// call (alongside [`AgentEvent::ModelStarted`]) so exporters can
        /// render a real duration instead of a zero-width point. `None` for
        /// events serialized before this field existed (`#[serde(default)]`
        /// keeps old journals deserializable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        started_at_ms: Option<u64>,
        /// Token usage reported by the provider, when available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// The request messages sent to the model, captured only when
        /// [`PayloadCapture::model_io`][crate::harness::runtime::PayloadCapture::model_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the prompt in a generation's Input panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        /// The model completion (assistant message), captured only when
        /// [`PayloadCapture::model_io`][crate::harness::runtime::PayloadCapture::model_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the completion in a generation's Output panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
    },

    /// A tool-selection middleware filtered the model-visible tool set before a
    /// model call. Makes exposure decisions auditable: a UI or log can see
    /// which tools were withheld from the model and by which policy.
    ToolsFiltered {
        /// Name of the middleware/policy that made the decision.
        by: String,
        /// Tools removed from the model-visible set, in their original order.
        excluded: Vec<String>,
        /// Number of tools left exposed to the model.
        remaining: usize,
    },

    /// A tool invocation has been dispatched.
    ToolStarted {
        /// Identifier for this tool call, correlates with completion.
        call_id: CallId,
        /// Name of the tool being invoked.
        tool_name: String,
    },

    /// A tool invocation returned.
    ToolCompleted {
        /// Identifier for the tool call that completed.
        call_id: CallId,
        /// Name of the tool that was invoked.
        tool_name: String,
        /// Wall-clock time the tool call *started*, in Unix-epoch
        /// milliseconds. Captured by the agent loop when it dispatches the
        /// call (alongside [`AgentEvent::ToolStarted`]) so exporters can
        /// render a real duration instead of a zero-width point. `None` for
        /// events serialized before this field existed (`#[serde(default)]`
        /// keeps old journals deserializable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        started_at_ms: Option<u64>,
        /// The arguments the tool was invoked with, captured only when
        /// [`PayloadCapture::tool_io`][crate::harness::runtime::PayloadCapture::tool_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the call in a tool observation's Input panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        /// The tool result content, captured only when
        /// [`PayloadCapture::tool_io`][crate::harness::runtime::PayloadCapture::tool_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the result in a tool observation's Output panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
        /// Wall-clock duration of the call in milliseconds (completion minus
        /// [`started_at_ms`]). Present regardless of payload capture, so an
        /// exporter renders a real duration without a side-channel. `None` for
        /// events serialized before this field existed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        /// Size, in bytes, of the tool's textual result content. Present even in
        /// payload-free mode (unlike [`output`]), so an exporter can show result
        /// size without capturing the body. `None` for older events.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_bytes: Option<u64>,
        /// Failure message when the tool call failed; `None` on success. Lets an
        /// exporter render success/failure and a reason from the journalled
        /// event itself rather than a live outcome side-channel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    /// The model called a tool that is not registered, and the run's
    /// [`UnknownToolPolicy`][crate::harness::runtime::UnknownToolPolicy]
    /// recovered from it instead of aborting.
    ///
    /// This is distinct from a tool that ran and returned an error: no tool was
    /// executed. The original requested name and arguments are preserved so the
    /// event stream can drive repair/analysis.
    UnknownToolCall {
        /// Identifier of the offending tool call.
        call_id: CallId,
        /// The tool name the model requested (which is not registered).
        requested_name: String,
        /// The raw arguments the model supplied for the call, preserved
        /// verbatim so repair middleware or analysis can re-target or replay
        /// the intended invocation.
        arguments: serde_json::Value,
        /// How the run recovered (for example `"tool_error"` or
        /// `"rewrite:other_tool"`).
        recovery: String,
    },

    /// A per-agent isolated workspace/sandbox was prepared.
    WorkspacePrepared {
        /// Audit identity of the policy that produced the environment.
        policy_id: String,
        /// The allowed root, rendered as a string.
        root: String,
    },

    /// A tool attempted to touch a path outside its allowed workspace roots and
    /// was blocked.
    WorkspaceViolation {
        /// The offending path, rendered as a string.
        path: String,
    },

    /// A previously prepared isolated workspace/sandbox was cleaned up (or the
    /// cleanup failed, when `error` is set).
    WorkspaceCleanup {
        /// Audit identity of the policy whose environment was cleaned up.
        policy_id: String,
        /// Cleanup error, when cleanup failed.
        error: Option<String>,
    },

    /// The agent loop honored a
    /// [`MiddlewareControl`][crate::harness::context::MiddlewareControl] request
    /// at a safe checkpoint (for example an early-exit stop or a pause). Recorded
    /// so control decisions are auditable and replayable from the journal.
    ControlApplied {
        /// Stable label of the control outcome (see
        /// [`MiddlewareControl::kind`][crate::harness::context::MiddlewareControl::kind]).
        control: String,
        /// Human-readable detail (the final text, or the interrupt node/message).
        detail: String,
    },

    /// Agent or graph-node state was mutated.
    ///
    /// Emitted after state transitions so downstream subscribers can
    /// invalidate cached views.
    StateUpdate,

    /// A middleware hook started executing.
    MiddlewareStarted {
        /// Registered name of the middleware.
        name: String,
    },

    /// A middleware hook finished executing.
    MiddlewareCompleted {
        /// Registered name of the middleware.
        name: String,
    },

    /// A response-cache lookup served the model call from the local
    /// [`crate::harness::cache::ResponseCache`]; the provider was **not**
    /// invoked.
    CacheHit {
        /// Identifier for the model call this cache hit satisfies.
        call_id: CallId,
        /// The stable cache key (see [`crate::harness::cache::cache_key`]).
        key: String,
    },

    /// A response-cache lookup missed, so the provider is being invoked and the
    /// result will be stored under `key`.
    CacheMiss {
        /// Identifier for the model call that triggered the lookup.
        call_id: CallId,
        /// The stable cache key (see [`crate::harness::cache::cache_key`]).
        key: String,
    },

    /// A failed call has been scheduled for retry.
    RetryScheduled {
        /// Identifier for the call that will be retried.
        call_id: CallId,
        /// 1-based attempt number of the upcoming retry.
        attempt: usize,
    },

    /// A rate-limit gate (token bucket) blocked a model call until capacity was
    /// available. Emitted by
    /// [`RateLimitMiddleware`][crate::harness::middleware::RateLimitMiddleware]
    /// once per gated call, after the tokens were finally acquired.
    RateLimitWaited {
        /// Actual wall-clock time the call was held back, in milliseconds
        /// (measured with the middleware's injectable clock).
        waited_ms: u64,
    },

    /// A model fallback middleware swapped the request from one model to
    /// another after the primary failed. Emitted by
    /// [`ModelFallbackMiddleware`][crate::harness::middleware::ModelFallbackMiddleware].
    FallbackSelected {
        /// The model that failed (the previous selection).
        from: String,
        /// The fallback model now being tried.
        to: String,
    },

    /// An explicit per-request model override was skipped during resolution —
    /// the requested model is unregistered, lacks a required capability, or is
    /// provider-retired — and resolution fell through to a lower-priority
    /// candidate (documented fail-closed behavior). Emitted by the agent loop
    /// so the silent fall-through is observable.
    ModelOverrideSkipped {
        /// The model name the request explicitly asked for.
        requested: String,
        /// The model that was actually resolved instead.
        resolved: String,
    },

    /// A sub-agent child run is about to be invoked from a parent run.
    SubAgentStarted {
        /// Name of the sub-agent being invoked.
        name: String,
        /// Depth of the child run in the recursion tree (parent depth + 1).
        depth: usize,
    },

    /// A sub-agent child run finished.
    SubAgentCompleted {
        /// Name of the sub-agent that completed.
        name: String,
        /// Depth of the child run in the recursion tree.
        depth: usize,
    },

    /// An existing sub-agent was *reused* for a follow-up turn rather than
    /// reconstructed, carrying the prior conversation context forward.
    ///
    /// Emitted by [`crate::harness::subagent::SubAgentSession`] on every send
    /// after the first (i.e. `turn >= 1`), so post-completion reuse — the
    /// orchestrator → sub-agent → human input → *same* sub-agent pattern — is
    /// visible in the event stream and distinguishable from a fresh
    /// [`AgentEvent::SubAgentStarted`].
    SubAgentReused {
        /// Name of the reused sub-agent.
        name: String,
        /// Zero-based index of the turn being started for this reuse (the
        /// second send is `turn == 1`).
        turn: usize,
    },

    /// A steering command was delivered to a running agent at a safe
    /// checkpoint and either applied or rejected by the run's steering policy.
    ///
    /// Emitted by the agent loop for every drained
    /// [`crate::harness::steering::SteeringCommand`] so that orchestrator and
    /// human steering is fully observable in the event stream and never an
    /// untracked side channel.
    Steered {
        /// Stable name of the steered command kind (e.g. `"inject_message"`,
        /// `"cancel"`); see
        /// [`crate::harness::steering::SteeringCommandKind::as_str`].
        command_kind: String,
        /// `true` when the run's policy permitted the command and it was
        /// applied; `false` when the policy rejected it.
        accepted: bool,
    },

    /// The transcript was compressed/summarized because it neared the model's
    /// context window. Emitted by
    /// [`ContextCompressionMiddleware`][crate::harness::middleware::ContextCompressionMiddleware]
    /// only when it actually compresses; below-threshold requests pass through
    /// without emitting this event.
    Compressed {
        /// Estimated total tokens of the transcript before compression.
        from_tokens: u64,
        /// Estimated total tokens of the transcript after compression.
        to_tokens: u64,
    },

    /// A graph routing decision produced a named route.
    RouteSelected {
        /// The route name chosen by the router.
        route: String,
    },

    /// Token usage for a completed model call was folded into the run totals.
    ///
    /// Emitted by the agent loop immediately after a model response that
    /// carried provider-reported usage, so usage-mode stream consumers and
    /// durable journals see per-call token counts without inspecting the
    /// [`AgentEvent::ModelCompleted`] payload.
    UsageRecorded {
        /// The usage reported for the model call just completed.
        usage: Usage,
    },

    /// Estimated cost for a completed model call was folded into the run
    /// totals.
    ///
    /// Defined for future emit: cost is computed once a pricing table is wired
    /// into the loop. Carries the cost delta attributed to the call.
    CostRecorded {
        /// The cost attributed to the model call just completed.
        cost: CostTotals,
    },

    /// A budget crossed its configured warning threshold but has not been
    /// exceeded, so the run continues.
    ///
    /// Emitted by
    /// [`BudgetMiddleware`][crate::harness::middleware::BudgetMiddleware] after a
    /// spend pushes cumulative usage/cost past `warn_fraction` of a limit.
    BudgetWarning {
        /// Human-readable description of which budget threshold was crossed.
        reason: String,
    },

    /// Budget preflight reserved an estimate of the upcoming model call's input
    /// tokens against the run budget, before dispatching the call. Lets a budget
    /// bound a call *before* it overshoots, rather than only detecting the
    /// overshoot afterward.
    BudgetReserved {
        /// Estimated input tokens reserved for the upcoming call.
        estimated_input_tokens: u64,
    },

    /// Budget reconciled a prior reservation against the provider-reported usage
    /// after the call returned, so the difference between the estimate and the
    /// actual is auditable.
    BudgetReconciled {
        /// Tokens that had been reserved (estimated) for the call.
        estimated_input_tokens: u64,
        /// Input tokens the provider actually reported.
        actual_input_tokens: u64,
    },

    /// A budget limit was reached. When emitted from budget preflight the run is
    /// about to be blocked with
    /// [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded];
    /// when emitted post-spend it flags that the accumulated totals now exceed a
    /// limit.
    BudgetExceeded {
        /// Human-readable description of which budget limit was hit.
        reason: String,
        /// Whether this occurrence blocked a model call (preflight) rather than
        /// being detected after a spend.
        blocked: bool,
    },

    /// A configured run limit (cap) tripped and the run is about to fail.
    ///
    /// Emitted by the agent loop just before returning
    /// [`crate::error::TinyAgentsError::LimitExceeded`] /
    /// [`crate::error::TinyAgentsError::Timeout`] so observers can distinguish
    /// a limit-driven stop from other failures.
    LimitReached {
        /// Which cap was reached. Serialized as `limit_kind` to avoid colliding
        /// with the enum's `"kind"` serde tag.
        #[serde(rename = "limit_kind")]
        kind: LimitKind,
    },

    /// Conversation/working memory was loaded for the run.
    ///
    /// Defined for future emit when memory wiring lands; carries no payload so
    /// that loading remains observable without exposing memory contents.
    MemoryLoaded,

    /// Conversation/working memory was persisted for the run.
    ///
    /// Defined for future emit when memory wiring lands.
    MemorySaved,

    /// A long-running tool reported incremental progress before completing.
    ///
    /// Defined for future emit: a tool that streams progress can surface it
    /// here so UIs render activity between [`AgentEvent::ToolStarted`] and
    /// [`AgentEvent::ToolCompleted`].
    ToolProgress {
        /// Identifier for the in-flight tool call.
        call_id: CallId,
        /// Human-readable progress message.
        message: String,
    },

    /// A middleware hook reported a failure.
    ///
    /// Defined for future emit alongside [`AgentEvent::MiddlewareStarted`] /
    /// [`AgentEvent::MiddlewareCompleted`] so a failing hook is observable.
    MiddlewareFailed {
        /// Registered name of the middleware that failed.
        name: String,
        /// Human-readable error description.
        error: String,
    },

    /// A streaming model call's chunk stream was closed (gracefully or by
    /// cancellation).
    ///
    /// Defined for future emit so stream consumers can detect end-of-stream
    /// without correlating against [`AgentEvent::ModelCompleted`].
    StreamClosed,

    /// A harness run finished successfully.
    RunCompleted {
        /// Identifier for the run that completed.
        run_id: RunId,
    },

    /// A harness run ended with an unrecoverable error.
    RunFailed {
        /// Identifier for the run that failed.
        run_id: RunId,
        /// Human-readable error description.
        error: String,
    },
}

/// Names the kind of run limit that tripped in an [`AgentEvent::LimitReached`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    /// The maximum number of model calls per run was reached.
    ModelCalls,
    /// The maximum number of tool calls per run was reached.
    ToolCalls,
    /// The run's wall-clock deadline elapsed.
    WallClock,
}

impl LimitKind {
    /// Returns a stable, snake_case string naming the limit kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            LimitKind::ModelCalls => "model_calls",
            LimitKind::ToolCalls => "tool_calls",
            LimitKind::WallClock => "wall_clock",
        }
    }
}

impl AgentEvent {
    /// Returns a stable, dot-separated string that names the kind of event.
    ///
    /// The returned string is a static literal, suitable for logging,
    /// filtering, and serde-independent routing. Examples: `"run.started"`,
    /// `"model.delta"`, `"tool.completed"`.
    pub fn kind(&self) -> &'static str {
        match self {
            AgentEvent::RunStarted { .. } => "run.started",
            AgentEvent::ModelStarted { .. } => "model.started",
            AgentEvent::ModelDelta { .. } => "model.delta",
            AgentEvent::ModelCompleted { .. } => "model.completed",
            AgentEvent::ControlApplied { .. } => "control.applied",
            AgentEvent::ToolsFiltered { .. } => "tool.filtered",
            AgentEvent::ToolStarted { .. } => "tool.started",
            AgentEvent::ToolCompleted { .. } => "tool.completed",
            AgentEvent::UnknownToolCall { .. } => "tool.unknown",
            AgentEvent::BudgetWarning { .. } => "budget.warning",
            AgentEvent::BudgetReserved { .. } => "budget.reserved",
            AgentEvent::BudgetReconciled { .. } => "budget.reconciled",
            AgentEvent::BudgetExceeded { .. } => "budget.exceeded",
            AgentEvent::WorkspacePrepared { .. } => "workspace.prepared",
            AgentEvent::WorkspaceViolation { .. } => "workspace.violation",
            AgentEvent::WorkspaceCleanup { .. } => "workspace.cleanup",
            AgentEvent::StateUpdate => "state.update",
            AgentEvent::MiddlewareStarted { .. } => "middleware.started",
            AgentEvent::MiddlewareCompleted { .. } => "middleware.completed",
            AgentEvent::CacheHit { .. } => "cache.hit",
            AgentEvent::CacheMiss { .. } => "cache.miss",
            AgentEvent::RetryScheduled { .. } => "retry.scheduled",
            AgentEvent::RateLimitWaited { .. } => "rate_limit.waited",
            AgentEvent::FallbackSelected { .. } => "model.fallback_selected",
            AgentEvent::ModelOverrideSkipped { .. } => "model.override_skipped",
            AgentEvent::SubAgentStarted { .. } => "subagent.started",
            AgentEvent::SubAgentCompleted { .. } => "subagent.completed",
            AgentEvent::SubAgentReused { .. } => "subagent.reused",
            AgentEvent::Steered { .. } => "agent.steered",
            AgentEvent::Compressed { .. } => "context.compressed",
            AgentEvent::RouteSelected { .. } => "route.selected",
            AgentEvent::UsageRecorded { .. } => "usage.recorded",
            AgentEvent::CostRecorded { .. } => "cost.recorded",
            AgentEvent::LimitReached { .. } => "limit.reached",
            AgentEvent::MemoryLoaded => "memory.loaded",
            AgentEvent::MemorySaved => "memory.saved",
            AgentEvent::ToolProgress { .. } => "tool.progress",
            AgentEvent::MiddlewareFailed { .. } => "middleware.failed",
            AgentEvent::StreamClosed => "stream.closed",
            AgentEvent::RunCompleted { .. } => "run.completed",
            AgentEvent::RunFailed { .. } => "run.failed",
        }
    }
}

// ---------------------------------------------------------------------------
// EventRecord
// ---------------------------------------------------------------------------

/// A timestamped, offset-keyed wrapper around an [`AgentEvent`].
///
/// The monotonic `offset` allows late subscribers to request replay from a
/// known position in the event stream without loading the full history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    /// Stable, unique identifier for this event.
    pub id: EventId,
    /// Monotonically increasing position in the stream (starts at 0).
    pub offset: u64,
    /// The typed event payload.
    pub event: AgentEvent,
}

// ---------------------------------------------------------------------------
// EventListener trait
// ---------------------------------------------------------------------------

/// An observer that receives typed event records from an [`EventSink`].
///
/// Implementations must be **Send + Sync** and are expected to be
/// low-latency. Any heavy processing (I/O, serialization, network) should be
/// deferred to a background task or channel so that the calling harness step
/// is not delayed.
pub trait EventListener: Send + Sync {
    /// Called synchronously by [`EventSink::emit`] for every emitted event.
    ///
    /// The provided `record` is borrowed; clone it if the listener needs to
    /// retain it beyond the call.
    fn on_event(&self, record: &EventRecord);
}

// ---------------------------------------------------------------------------
// EventSink
// ---------------------------------------------------------------------------

/// Shared, cloneable event fan-out bus.
///
/// All clones share the same underlying listener list and monotonic offset
/// counter via an `Arc<Mutex<…>>`. Any clone can subscribe new listeners or
/// emit events. The `emit` method assigns a monotonic [`EventId`] and offset
/// and enqueues the record under one critical section, then a single draining
/// emitter delivers queued records to listeners in offset order — so listeners
/// never observe offset `n + 1` before offset `n`, even under concurrent
/// emits.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use tinyagents::harness::events::{AgentEvent, EventSink, RecordingListener};
/// use tinyagents::harness::ids::RunId;
///
/// let sink = EventSink::new();
/// let recorder = Arc::new(RecordingListener::new());
/// sink.subscribe(recorder.clone());
///
/// sink.emit(AgentEvent::RunStarted { run_id: RunId::new("r1"), thread_id: None });
/// assert_eq!(recorder.events().len(), 1);
/// ```
#[derive(Clone)]
pub struct EventSink {
    pub(crate) inner: Arc<Mutex<EventSinkInner>>,
}

/// Interior state shared among all clones of an [`EventSink`].
pub(crate) struct EventSinkInner {
    /// Stream-scoping prefix for emitted [`EventId`]s. Combined with the
    /// per-emit `offset` to form ids of the form `{stream_id}-evt-{offset}`,
    /// so ids stay unique across sinks and, when the prefix is a stable
    /// run/thread id, across process restarts (see [`EventSink::with_stream_id`]).
    pub(crate) stream_id: String,
    /// Next offset to assign; incremented atomically on each `emit`.
    pub(crate) next_offset: u64,
    /// Registered listeners, notified in insertion order.
    pub(crate) listeners: Vec<Arc<dyn EventListener>>,
    /// Records assigned an offset but not yet delivered to listeners, in
    /// offset order. Each entry carries the listener snapshot taken when the
    /// offset was assigned so late subscribers never see earlier offsets.
    pub(crate) pending: std::collections::VecDeque<(EventRecord, Vec<Arc<dyn EventListener>>)>,
    /// `true` while some emitter is draining `pending`. Guarantees a single
    /// drainer at a time, which is what makes listener delivery globally
    /// ordered by offset (and keeps re-entrant emits from listeners safe:
    /// they enqueue and return, and the active drainer delivers them).
    pub(crate) dispatching: bool,
}

// ---------------------------------------------------------------------------
// RecordingListener
// ---------------------------------------------------------------------------

/// An [`EventListener`] that collects every received [`EventRecord`] into an
/// in-memory buffer for later inspection.
///
/// Useful in tests, debugging sessions, and in-process dashboards. Thread-safe
/// via an internal `Arc<Mutex<…>>`.
pub struct RecordingListener {
    pub(crate) records: Arc<Mutex<Vec<EventRecord>>>,
}

// ---------------------------------------------------------------------------
// EventJournal
// ---------------------------------------------------------------------------

/// An append-only in-memory journal of [`EventRecord`]s.
///
/// Supports both live append from an active run and offset-based replay for
/// late subscribers. The journal does not fan out to listeners; callers that
/// need live delivery should use an [`EventSink`] alongside the journal.
///
/// Records are stored in offset order: the journal's buffer is populated by an
/// internal listener on the sink's ordered dispatch path, so
/// [`EventJournal::replay_from`] always returns a contiguous, offset-ordered
/// prefix of the stream — never the completion order of racing appends.
pub struct EventJournal {
    pub(crate) records: Arc<Mutex<Vec<EventRecord>>>,
    /// Internal sink used to assign monotonic ids and offsets.
    pub(crate) sink: EventSink,
}

/// Internal [`EventListener`] that copies each dispatched record into an
/// [`EventJournal`]'s buffer. Because sink dispatch is globally ordered by
/// offset, the buffer stays in offset order regardless of append concurrency.
pub(crate) struct JournalRecorder {
    pub(crate) records: Arc<Mutex<Vec<EventRecord>>>,
}

// ---------------------------------------------------------------------------
// HarnessRunStatus
// ---------------------------------------------------------------------------

/// A compact, readable snapshot of an active or completed harness run.
///
/// Status records intentionally omit full prompt text, raw tool outputs, and
/// provider payloads. Only counters, ids, phase markers, timing, error
/// summaries, and cumulative usage/cost are stored so dashboards and
/// supervisors can read current state cheaply.
///
/// A graph node that invokes a child harness can correlate runs via
/// `parent_run_id` and `root_run_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HarnessRunStatus {
    /// Unique identifier for this run.
    pub run_id: RunId,

    /// Parent run id when this run was invoked from a graph node or another
    /// harness. `None` for top-level runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<RunId>,

    /// Root ancestor run, equal to `run_id` for top-level runs.
    pub root_run_id: RunId,

    /// Conversation thread this run belongs to, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<ThreadId>,

    /// The component (model, tool, middleware, or graph node) that owns this
    /// run.
    pub component: ComponentId,

    /// Coarse lifecycle status.
    pub status: ExecutionStatus,

    /// Active harness operation within the current status.
    pub current_phase: HarnessPhase,

    /// Number of model calls that have completed within this run.
    pub model_calls: usize,

    /// Number of tool invocations that have completed within this run.
    pub tool_calls: usize,

    /// The in-flight model call id, if a model call is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_model_call: Option<CallId>,

    /// Tool call ids currently executing (may be concurrent).
    #[serde(default)]
    pub active_tool_calls: Vec<CallId>,

    /// Id of the most recent event recorded for this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_event_id: Option<EventId>,

    /// Cumulative token usage across all model calls in this run.
    pub usage: UsageTotals,

    /// Cumulative estimated cost across all model calls in this run.
    pub cost: CostTotals,

    /// Wall-clock time when the run started.
    #[serde(with = "serde_system_time")]
    pub started_at: SystemTime,

    /// Wall-clock time of the most recent status mutation.
    #[serde(with = "serde_system_time")]
    pub updated_at: SystemTime,

    /// Wall-clock time when the run ended (`None` while still active).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_system_time_opt"
    )]
    pub ended_at: Option<SystemTime>,

    /// Human-readable error when the run failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Arbitrary caller-supplied key/value metadata.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Serde helpers for SystemTime
// ---------------------------------------------------------------------------

/// Serialize/deserialize [`SystemTime`] as Unix epoch seconds (`u64`).
mod serde_system_time {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let secs = t
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        s.serialize_u64(secs)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

/// Serialize/deserialize `Option<SystemTime>` as an optional Unix epoch
/// seconds value (`u64 | null`).
mod serde_system_time_opt {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &Option<SystemTime>, s: S) -> Result<S::Ok, S::Error> {
        match t {
            Some(t) => {
                let secs = t
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_secs();
                s.serialize_some(&secs)
            }
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<SystemTime>, D::Error> {
        let secs = Option::<u64>::deserialize(d)?;
        Ok(secs.map(|s| UNIX_EPOCH + Duration::from_secs(s)))
    }
}
