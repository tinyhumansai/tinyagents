//! Type definitions for the harness run-context module.
//!
//! These types carry the recursion bookkeeping (depth / max-depth) and the
//! shared signals (cancellation, steering, events) that let a parent run and
//! its nested sub-runs behave as one coordinated tree.
//!
//! [`RunConfig`] is the serializable, declarative description of a run (its
//! identity, limits, and metadata). [`RunContext`] is the live, in-process
//! handle threaded through model calls, tool calls, middleware, and graph
//! nodes; it bundles the config with the dependencies a run needs (stores,
//! events, and limit tracking) plus arbitrary user data.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::harness::context` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs`.

use serde::{Deserialize, Serialize};

use crate::harness::cancel::CancellationToken;
use crate::harness::events::EventSink;
use crate::harness::ids::{RunId, ThreadId};
use crate::harness::limits::LimitTracker;
use crate::harness::steering::SteeringHandle;
use crate::harness::store::StoreRegistry;

/// Declarative, serializable configuration for a single harness run.
///
/// `RunConfig` captures everything that defines a run independent of live
/// runtime state: its identity, the thread it belongs to, classification tags,
/// free-form metadata, and the hard limits applied to it. Because it is
/// `Serialize`/`Deserialize`, a run can be described, stored, and replayed.
///
/// Construct one with [`RunConfig::new`] and refine it with the `with_*`
/// builder methods.
///
/// # Example
///
/// ```
/// use tinyagents::harness::context::RunConfig;
///
/// let config = RunConfig::new("run-1")
///     .with_thread("thread-7")
///     .with_tag("nightly")
///     .with_max_model_calls(10);
/// assert_eq!(config.run_id.as_str(), "run-1");
/// assert_eq!(config.max_model_calls, 10);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    /// Unique identifier for this run.
    pub run_id: RunId,
    /// Conversation thread this run belongs to, when threaded.
    pub thread_id: Option<ThreadId>,
    /// Free-form classification tags (for example `"nightly"`, `"eval"`).
    pub tags: Vec<String>,
    /// Arbitrary caller-supplied metadata. Defaults to JSON `null`.
    pub metadata: serde_json::Value,
    /// Wall-clock timeout in milliseconds. `None` means no deadline.
    pub timeout_ms: Option<u64>,
    /// Maximum number of model calls permitted for this run.
    pub max_model_calls: usize,
    /// Maximum number of tool invocations permitted for this run.
    pub max_tool_calls: usize,
    /// Maximum output tokens requested for each model turn in this run.
    ///
    /// When set, the agent loop applies this as an upper bound to
    /// [`crate::harness::model::ModelRequest::max_tokens`] before dispatching a
    /// model call. Child sub-agent runs inherit the same cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turn_output_tokens: Option<u32>,
    /// Current depth of this run in the sub-agent / recursion tree.
    ///
    /// A top-level run is depth `0`. When a [`crate::harness::subagent::SubAgent`]
    /// invokes a child harness, the child run's `depth` is the parent's depth
    /// plus one. Defaults to `0` and is `#[serde(default)]` so configs written
    /// before this field existed still deserialize.
    #[serde(default)]
    pub depth: usize,
    /// Maximum sub-agent / recursion depth permitted for the run tree rooted at
    /// this run. Carried into [`crate::harness::limits::RunLimits`] so the agent
    /// loop and sub-agent guard share one cap. Defaults to
    /// [`crate::harness::limits::RunLimits::DEFAULT_MAX_DEPTH`].
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
}

/// Serde default for [`RunConfig::max_depth`]: the crate-wide depth cap.
fn default_max_depth() -> usize {
    crate::harness::limits::RunLimits::DEFAULT_MAX_DEPTH
}

/// A structured control outcome a middleware (or any step) can request on the
/// [`RunContext`] to steer the agent loop from outside its `Result<()>` return
/// channel.
///
/// This is the harness-native complement to the graph
/// [`Command`][crate::graph::command::Command]/[`Interrupt`][crate::graph::command::Interrupt]
/// vocabulary: the agent loop drains any requested control at its safe
/// checkpoints (after each model response) and acts on it, so behaviors like
/// "stop after an early-exit tool" or "pause on budget" no longer need a
/// bespoke side channel. Requests are visible via
/// [`RunContext::take_control`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MiddlewareControl {
    /// Stop the loop now and use this text as the final assistant response.
    StopWithFinal(String),
    /// Pause the run at the next safe checkpoint, surfacing
    /// [`crate::error::TinyAgentsError::Interrupted`] with this node/message so a
    /// caller can persist a checkpoint and resume later.
    Interrupt {
        /// Logical node/label the interrupt is attributed to.
        node: String,
        /// Human-readable reason surfaced with the interrupt.
        message: String,
    },
}

/// Live, in-process handle threaded through every step of a harness run.
///
/// A `RunContext` bundles the declarative [`RunConfig`] with the runtime
/// dependencies a run needs:
/// - `stores`: the [`StoreRegistry`] for long-term persistence.
/// - `events`: the [`EventSink`] for observability fan-out.
/// - `limits`: the [`LimitTracker`] enforcing the run's caps.
///
/// The generic `Ctx` parameter carries arbitrary user data (dependencies,
/// shared services, accumulated state). It defaults to `()` for runs that need
/// no extra data.
///
/// Unlike [`RunConfig`], `RunContext` is **not** serializable: it owns live
/// counters, listener lists, and user handles.
pub struct RunContext<Ctx = ()> {
    /// The declarative configuration this context was built from.
    pub config: RunConfig,
    /// Arbitrary user-supplied run data.
    pub data: Ctx,
    /// Registry of named long-term stores.
    pub stores: StoreRegistry,
    /// Event fan-out bus for observability.
    pub events: EventSink,
    /// Live limit tracker derived from `config`.
    pub limits: LimitTracker,
    /// Optional steering channel an orchestrator (parent agent, human UI,
    /// graph supervisor, or test) uses to guide this run at safe checkpoints.
    ///
    /// `None` means the run accepts no steering. Attach one with
    /// [`RunContext::with_steering`]; the agent loop drains it before each
    /// model call via
    /// [`crate::harness::steering::apply_pending_steering`].
    pub steering: Option<SteeringHandle>,
    /// Cooperative cancellation token for this run.
    ///
    /// Defaults to a fresh, never-cancelled [`CancellationToken`], so a run is
    /// only cancellable if a caller installs a shared token via
    /// [`RunContext::with_cancellation`]. The agent loop polls
    /// [`CancellationToken::is_cancelled`] at the same safe checkpoints used for
    /// steering — before each model call and before each tool call — and the
    /// streaming pipeline races [`CancellationToken::cancelled`] against the
    /// provider stream. On observing cancellation the run ends with
    /// [`crate::error::TinyAgentsError::Cancelled`].
    pub cancellation: CancellationToken,
    /// A one-shot control request a middleware or step can set to steer the
    /// loop (stop with a final response, or interrupt). Drained by the agent
    /// loop at its safe checkpoints via [`RunContext::take_control`].
    pub control: std::sync::Arc<std::sync::Mutex<Option<MiddlewareControl>>>,
}
