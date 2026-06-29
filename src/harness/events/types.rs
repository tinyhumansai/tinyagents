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
        /// Identifier for the model call that produced this delta.
        call_id: CallId,
        /// Incremental text and/or tool-call fragment.
        delta: MessageDelta,
    },

    /// A model call completed successfully.
    ModelCompleted {
        /// Identifier for the model call that completed.
        call_id: CallId,
        /// Token usage reported by the provider, when available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
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
    /// each time it must wait before acquiring tokens.
    RateLimitWaited {
        /// Approximate time the call was held back, in milliseconds.
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
            AgentEvent::ToolStarted { .. } => "tool.started",
            AgentEvent::ToolCompleted { .. } => "tool.completed",
            AgentEvent::StateUpdate => "state.update",
            AgentEvent::MiddlewareStarted { .. } => "middleware.started",
            AgentEvent::MiddlewareCompleted { .. } => "middleware.completed",
            AgentEvent::CacheHit { .. } => "cache.hit",
            AgentEvent::CacheMiss { .. } => "cache.miss",
            AgentEvent::RetryScheduled { .. } => "retry.scheduled",
            AgentEvent::RateLimitWaited { .. } => "rate_limit.waited",
            AgentEvent::FallbackSelected { .. } => "model.fallback_selected",
            AgentEvent::SubAgentStarted { .. } => "subagent.started",
            AgentEvent::SubAgentCompleted { .. } => "subagent.completed",
            AgentEvent::SubAgentReused { .. } => "subagent.reused",
            AgentEvent::Steered { .. } => "agent.steered",
            AgentEvent::Compressed { .. } => "context.compressed",
            AgentEvent::RouteSelected { .. } => "route.selected",
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
/// before calling each registered listener in registration order.
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
    /// Next offset to assign; incremented atomically on each `emit`.
    pub(crate) next_offset: u64,
    /// Registered listeners, notified in insertion order.
    pub(crate) listeners: Vec<Arc<dyn EventListener>>,
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
pub struct EventJournal {
    pub(crate) records: Arc<Mutex<Vec<EventRecord>>>,
    /// Internal sink used to assign monotonic ids and offsets.
    pub(crate) sink: EventSink,
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
