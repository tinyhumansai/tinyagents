//! Type definitions for the harness run-context module.
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

use crate::harness::events::EventSink;
use crate::harness::ids::{RunId, ThreadId};
use crate::harness::limits::LimitTracker;
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
/// use rustagents::harness::context::RunConfig;
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
}
