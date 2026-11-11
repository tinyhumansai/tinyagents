//! Harness runtime modules — the execution layer of the recursive runtime.
//!
//! The harness is the surface where a single model call becomes a recursive
//! system: it runs the agent loop (model ⇄ tools), while the orchestration crate
//! can wrap a whole harness agent as a typed tool. Parent/child run lineage, depth limits,
//! usage/cost roll-up, [`steering`], and [`cancel`]lation all flow through here,
//! making nested runs first-class, observable, and policy-checked.
//!
//! The harness is intentionally split by feature. Each submodule owns one
//! substantial part of model/tool orchestration so the implementation can grow
//! without creating one large runtime file.
//!
//! # Cargo features
//!
//! - `sqlite` — retained as a no-op for compatibility; durable stores are the `Driver*` types over `tinystoragedrivers`.
//! - `multimodal` — image/audio/binary content resolution (see the `multimodal`
//!   module when the feature is enabled),
//!   pulling in `reqwest` and `flate2`.
//! - `claude-code` — the Claude Code CLI and Claude Agent SDK provider
//!   adapters under [`providers`].
//! - `langfuse` — the Langfuse observability exporter under [`observability`],
//!   pulling in `reqwest`.
//! - `tracing` — a no-op compatibility alias; tracing instrumentation is
//!   always compiled in.
//!
//! `claude-code` and `langfuse` are part of `default` so existing consumers
//! see no change; disable default features to opt out of either.
//!
//! Persisting a thread's transcript across runs lives in the separate
//! `tinyagents-session` crate: a host reads history into a run's `input` and
//! appends the run's messages back afterward.
//!
//! # Vendor re-exports
//!
//! The harness pins exact versions of the `tinyinference-llm`, `tinytools`,
//! and `tinytools-agent` vendor crates and exposes their public types
//! (e.g. `ChatMessage`, tool schemas) across its own API. Downstream crates
//! must reach those types through [`tinyinference_llm`], [`tinytools`], and
//! [`tinytools_agent`] re-exported here rather than depending on the vendor
//! crates directly, or the compiler will see two distinct copies of the same
//! type.

pub mod agent_error;
pub mod agent_loop;
pub mod artifacts;
pub mod cache;
pub mod cancel;
mod canonical_json;
pub mod catalog;
pub mod close;
pub mod config;
pub use config::{
    CompactionSettings, CompactionStrategy, ContextConfig, DEFAULT_KEEP_RECENT_TOOL_RESULTS,
};
pub mod context;
pub mod cost;
pub mod dialect;
pub mod driver_ids;
pub mod error;
pub mod events;
mod finish_reason;
pub mod host;
pub mod ids;
pub mod limits;
#[cfg(feature = "media")]
pub mod media;
pub mod middleware;
#[cfg(feature = "multimodal")]
pub mod multimodal;
pub mod no_progress;
pub mod observability;
pub mod oneshot;
pub mod progress;
pub mod prompt;
pub mod providers;
pub mod reasoning_replay;
pub mod result_expiry;
pub mod retry;
pub mod run_queue;
pub mod runtime;
pub mod steering;
pub mod storage;
pub mod store;
pub mod structured;
pub mod summarization;
pub mod terminal;
pub mod testkit;
pub mod thread_reasoning;
pub mod title;

pub mod token_estimation;
pub mod tool;
pub mod turn_deadline;
pub mod turn_outcome;
pub mod turn_policy;
pub mod turn_run_error;
pub mod turn_run_finalize;
pub mod turn_runner;
pub mod turn_stop;
/// Shared URL admission rules for outbound requests, including SSRF checks
/// and DNS address pinning.
pub mod url_guard;
pub mod workspace;

#[cfg(feature = "media")]
pub use tinyinference_image;
/// Re-exported vendor crates. Downstream consumers should reach these
/// dependencies' types through these re-exports (e.g.
/// `tinyagents_harness::tinyinference_llm::ChatMessage`) rather than adding
/// their own `tinyinference-llm` / `tinytools` / `tinytools-agent`
/// dependency, since the harness pins exact vendor versions and a second,
/// independent dependency would produce a duplicate, incompatible copy of
/// the same types.
pub use tinyinference_llm;
#[cfg(feature = "media")]
pub use tinyinference_video;
pub use tinytools;
pub use tinytools_agent;

pub use cancel::CancellationToken;
pub use cost::CostTotals;
pub use error::{Result, TinyAgentsError};
pub use ids::*;
pub use no_progress::{
    ArgumentChurnDetector, CallGate, ClassifiedFailure, ClassifiedFailureTracker,
    DEFAULT_IDENTICAL_HALT_THRESHOLD, DEFAULT_REPEAT_CALL_THRESHOLD,
    DEFAULT_REPEAT_OUTPUT_THRESHOLD, NoProgress, NoProgressTracker, OutcomeFingerprinter,
    PingPongDetector, PostCompactionGuard, RepeatEscalation, RepeatMonitor, RepeatProgressConfig,
    StreamTextStallDetector, SuccessfulRepeat, SuccessfulRepeatTracker, ToolAttempt,
    VolatileSpanNormalizer,
};
pub use observability::replay::{
    RunEventsPage, list_active_runs, read_run_events_page, read_run_status,
};
pub use observability::{
    AgentCallLatency, AgentLatencyMetrics, AgentObservation, FanOutSink, HarnessEventJournal,
    HarnessStatusStore, InMemoryEventJournal, InMemoryStatusStore, JournalSink, JsonlSink,
    ObservedChatModel, PhysicalDispatchGuard, PhysicalDispatchMetadata, PhysicalDispatchObserver,
    PhysicalDispatchOutcome, PhysicalDispatchStatus, PhysicalDispatchTerminal,
    PhysicalDispatchUsage, RedactingSink, StoreEventJournal,
};
pub use observability::{DriverStatusStore, mint_run_id, reap_orphaned_runs};
#[cfg(feature = "langfuse")]
pub use observability::{
    LangfuseAuth, LangfuseClient, LangfuseScore, LangfuseScoreValue, LangfuseTraceConfig,
};
pub use progress::AgentProgress;
pub use run_queue::{
    ForwardEvent, ForwardEventSink, ForwarderCleanup, QueueLane, QueueMode, QueueStatus,
    QueuedMessage, RunQueue, RunQueueHandle, SteeringForwarderGuard,
};
pub use steering::{
    RecentRequestIds, RequestIdError, SteeringCommand, SteeringCommandKind, SteeringHandle,
    SteeringOutcome, SteeringPolicy,
};
pub use storage::{collections, open_storage, storage_schema};
pub use terminal::{TerminalClass, TerminalOutcome, TerminalReason, TimeoutPhase};
pub use tool::ToolRegistry;
pub use turn_stop::{TurnFailureClass, TurnStop, TurnStopKind};
