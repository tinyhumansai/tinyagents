//! Type definitions for the harness runtime facade.
//!
//! [`RunPolicy`] is the declarative bundle of cross-cutting policy applied to
//! every run a harness drives (limits, retry, fallback, and a default response
//! format). [`AgentHarness`] is the high-level facade that wires a model
//! registry, a tool registry, a middleware stack, and a policy into a single
//! ergonomic entry point for the agent loop.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::harness::runtime` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs`.

use crate::harness::limits::RunLimits;
use crate::harness::middleware::MiddlewareStack;
use crate::harness::model::{ModelRegistry, ResponseFormat};
use crate::harness::retry::{FallbackPolicy, RetryPolicy};
use crate::harness::tool::ToolRegistry;

/// Declarative, run-scoped policy shared by every invocation of an
/// [`AgentHarness`].
///
/// A `RunPolicy` carries the four cross-cutting concerns the agent loop needs
/// to bound and steer a run:
///
/// - `limits`: hard caps (model calls, tool calls, wall-clock) enforced
///   fail-closed by the loop.
/// - `retry`: exponential-backoff retry policy applied to each model call.
/// - `fallback`: optional ordered chain of model names to try when the current
///   model exhausts its retries.
/// - `default_response_format`: when set, attached to every [`crate::harness::model::ModelRequest`]
///   the loop builds; a [`ResponseFormat::JsonSchema`] also drives structured
///   output extraction on the final response.
///
/// [`RunPolicy::default`] yields the crate-default limits and retry policy, no
/// fallback chain, and no response format.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunPolicy {
    /// Hard run limits enforced fail-closed by the agent loop.
    pub limits: RunLimits,
    /// Retry policy applied to each model call.
    pub retry: RetryPolicy,
    /// Optional ordered model fallback chain.
    pub fallback: Option<FallbackPolicy>,
    /// Response format attached to every model request when set.
    pub default_response_format: Option<ResponseFormat>,
}

/// High-level facade that composes model selection, tool execution, middleware,
/// and run policy behind one builder and runs the default agent loop.
///
/// `AgentHarness` is generic over the application `State` (shared, read-only
/// data threaded into every model and tool call) and the run-context data type
/// `Ctx` (defaults to `()`), which is moved into the [`crate::harness::context::RunContext`]
/// for the duration of a run.
///
/// The registries, middleware stack, and policy are kept crate-private; build a
/// harness with [`AgentHarness::new`] and the `register_*` / `push_middleware` /
/// `with_policy` builder methods, and read them back through the accessor
/// methods. The agent loop itself is implemented in
/// [`crate::harness::agent_loop`].
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use rustagents::harness::providers::MockModel;
/// use rustagents::harness::runtime::AgentHarness;
///
/// let mut harness: AgentHarness<()> = AgentHarness::new();
/// harness.register_model("mock", Arc::new(MockModel::constant("hello")));
/// assert_eq!(harness.models().default_name(), Some("mock"));
/// ```
pub struct AgentHarness<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// Name-keyed registry of chat models with an optional default.
    pub(crate) models: ModelRegistry<State>,
    /// Name-keyed registry of tools exposed to the model.
    pub(crate) tools: ToolRegistry<State>,
    /// Ordered middleware stack wrapping agent, model, and tool execution.
    pub(crate) middleware: MiddlewareStack<State, Ctx>,
    /// Cross-cutting run policy (limits, retry, fallback, response format).
    pub(crate) policy: RunPolicy,
}
