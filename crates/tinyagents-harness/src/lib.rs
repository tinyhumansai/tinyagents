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
//! - `sqlite` — durable, file-backed stores (`rusqlite`).
//! - `builtin-tools` — the bundled [`tools`] (currently the time tool). The
//!   old name `tools` is kept as a deprecated alias.
//! - `multimodal` — image/audio/binary content resolution ([`multimodal`]),
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
//! # Host utilities not on the agent loop path (M-10)
//!
//! [`handoff`] is exported for a host to build on, but [`agent_loop`] does
//! not call into it on its own — it is opt-in plumbing, not implicit loop
//! behavior. (A
//! [`run_queue::RunQueue`] *is* drained by the loop once attached via
//! [`context::RunContext::with_run_queue`]; see that module's docs.)
//!
//! - [`handoff`] is a progressive-disclosure cache for oversized tool
//!   results; a host calls [`handoff::apply_handoff`] itself before
//!   appending a tool result to history, and registers an extraction tool
//!   that reads the same [`handoff::ResultHandoffCache`].
//!
//! Persisting a thread's transcript across runs lives in the separate
//! `tinyagents-session` crate: a host reads history into a run's `input` and
//! appends the run's messages back afterward.
//!
//! Wiring any of these directly into the loop is deliberately future work
//! rather than default behavior, so a host that does not need one pays
//! nothing for it.
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

pub mod agent_loop;
pub mod artifacts;
mod blocking;
pub mod cache;
pub mod cancel;
pub mod capability;
pub mod config;
pub mod context;
pub mod cost;
pub mod error;
pub mod events;
pub mod handoff;
pub mod host;
pub mod ids;
pub mod limits;
#[cfg(feature = "media")]
pub mod media;
pub mod middleware;
pub mod model_registry;
#[cfg(feature = "multimodal")]
pub mod multimodal;
pub mod no_progress;
pub mod observability;
pub mod prompt;
pub mod providers;
pub(crate) mod relaxed_json;
pub mod retriever;
pub mod retry;
pub mod run_queue;
pub mod runtime;
pub mod steering;
pub mod store;
pub mod stream;
pub mod structured;
pub mod summarization;
pub mod testkit;
pub mod token_estimation;
pub mod tool;
#[cfg(feature = "builtin-tools")]
pub mod tools;
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
pub use capability::{
    Capability, CapabilityToolSet, LOAD_CAPABILITY_TOOL_NAME, LoadCapabilityTool,
    ModelRequestDefaults,
};
pub use cost::CostTotals;
pub use error::{Result, TinyAgentsError};
pub use ids::*;
pub use model_registry::{ModelRegistry, ModelSelection, ResolvedModelBinding};
pub use no_progress::{
    DEFAULT_IDENTICAL_HALT_THRESHOLD, DEFAULT_REPEAT_CALL_THRESHOLD,
    DEFAULT_REPEAT_OUTPUT_THRESHOLD, NoProgress, NoProgressTracker, SuccessfulRepeat,
    SuccessfulRepeatTracker, ToolAttempt,
};
pub use observability::{
    AgentCallLatency, AgentLatencyMetrics, AgentObservation, FanOutSink, HarnessEventJournal,
    HarnessStatusStore, InMemoryEventJournal, InMemoryStatusStore, JournalSink, JsonlSink,
    RedactingSink, StoreEventJournal,
};
#[cfg(feature = "langfuse")]
pub use observability::{
    LangfuseAuth, LangfuseClient, LangfuseScore, LangfuseScoreValue, LangfuseTraceConfig,
};
pub use run_queue::{QueueLane, QueueMode, QueueStatus, RunQueue, RunQueueHandle};
pub use steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringOutcome, SteeringPolicy,
};
pub use tool::ToolRegistry;
