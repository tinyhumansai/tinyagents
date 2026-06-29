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

use std::sync::Arc;

use crate::harness::cache::{CachePolicy, ResponseCache};
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
/// fallback chain, no response format, and a [`CachePolicy`] whose response
/// caching is enabled — caching only takes effect once a [`ResponseCache`] is
/// actually attached via [`AgentHarness::with_response_cache`], so the default
/// is safe even without a cache.
#[derive(Clone, Debug, PartialEq)]
pub struct RunPolicy {
    /// Hard run limits enforced fail-closed by the agent loop.
    pub limits: RunLimits,
    /// Retry policy applied to each model call.
    pub retry: RetryPolicy,
    /// Optional ordered model fallback chain.
    pub fallback: Option<FallbackPolicy>,
    /// Response format attached to every model request when set.
    pub default_response_format: Option<ResponseFormat>,
    /// Default caching policy for the run.
    ///
    /// The loop consults [`CachePolicy::response_cache_enabled`] only when a
    /// [`ResponseCache`] is attached to the harness *and* the per-call
    /// [`crate::harness::model::ModelRequest::cache_policy`] does not override
    /// it. A request-level `cache_policy` always wins over this default.
    pub cache: CachePolicy,
}

impl Default for RunPolicy {
    fn default() -> Self {
        Self {
            limits: RunLimits::default(),
            retry: RetryPolicy::default(),
            fallback: None,
            default_response_format: None,
            // Caching defaults ON, but is gated by an attached `ResponseCache`,
            // so a harness with no cache never caches regardless of this flag.
            cache: CachePolicy {
                response_cache_enabled: true,
                protect_prompt_prefix: false,
            },
        }
    }
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
/// use tinyagents::harness::providers::MockModel;
/// use tinyagents::harness::runtime::AgentHarness;
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
    /// Optional local response cache shared across runs of this harness.
    ///
    /// When set, the agent loop consults it before each provider call (subject
    /// to the effective [`CachePolicy`]) and stores successful responses back
    /// into it. Because it is owned by the harness rather than a single run, a
    /// repeated identical request can be served from an earlier run's result.
    pub(crate) response_cache: Option<Arc<dyn ResponseCache>>,
}
