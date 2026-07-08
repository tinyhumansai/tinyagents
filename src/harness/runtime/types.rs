//! Type definitions for the harness runtime facade.
//!
//! [`AgentHarness`] is the re-entrant runtime that the whole recursive
//! architecture stands inside: parent agents, nested sub-agents, subgraph
//! nodes, and model-authored blueprints all execute against the same composed
//! registries, middleware, and policy, so recursion reuses one runtime instead
//! of forking new ones. [`RunPolicy`] is the cross-cutting policy that runtime
//! enforces on every (parent or nested) run.
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
/// How the agent loop reacts when the model calls a tool that is not
/// registered.
///
/// The default is [`UnknownToolPolicy::Fail`], preserving the historical
/// fail-fast behavior. The recoverable variants let a run keep going so the
/// model can correct itself — each recovery still consumes a tool-call budget
/// slot, so [`RunLimits::max_tool_calls`] bounds any unknown-tool loop.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum UnknownToolPolicy {
    /// Abort the run with
    /// [`TinyAgentsError::ToolNotFound`][crate::error::TinyAgentsError::ToolNotFound]
    /// (the default, historical behavior).
    #[default]
    Fail,
    /// Inject a tool-error result (naming the originally requested tool and
    /// listing the registered tools) back into the transcript and continue the
    /// loop, letting the model retry with a valid tool.
    ReturnToolError,
    /// Rewrite an unknown call to a fixed compatibility tool name and retry the
    /// lookup once. If the rewrite target is also unregistered, fall back to
    /// [`UnknownToolPolicy::ReturnToolError`] behavior.
    Rewrite {
        /// The registered tool an unknown call is rewritten to.
        tool_name: String,
    },
}

/// How the agent loop reacts when the model calls a *registered* tool with
/// arguments that fail schema validation.
///
/// The default is [`InvalidArgsPolicy::Fail`], preserving the historical
/// fail-fast behavior where a missing `required` field, wrong type, or bad
/// `enum` aborts the whole turn. The recoverable variant lets a run keep going
/// so the model can self-correct — the recovery still consumes a tool-call
/// budget slot, so [`RunLimits::max_tool_calls`] bounds any invalid-args loop.
/// Mirrors [`UnknownToolPolicy`] for the schema-validation seam.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InvalidArgsPolicy {
    /// Abort the run with
    /// [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation]
    /// (the default, historical behavior).
    #[default]
    Fail,
    /// Inject a tool-error result (carrying the validation detail and the
    /// tool's expected parameter schema) back into the transcript and continue
    /// the loop, letting the model retry with corrected arguments.
    ReturnToolError,
}

/// Controls whether the agent loop captures model and tool **payloads**
/// (prompt/completion text, tool arguments/results) onto the
/// [`AgentEvent::ModelCompleted`][crate::harness::events::AgentEvent::ModelCompleted]
/// and [`AgentEvent::ToolCompleted`][crate::harness::events::AgentEvent::ToolCompleted]
/// events it emits.
///
/// The observability layer is **payload-free by default**: events carry only
/// ids, counters, and usage so privacy-sensitive deployments never journal or
/// export prompt text or tool I/O. Opt in per-family here to surface the request
/// messages + completion (`model_io`) and tool arguments + result (`tool_io`) so
/// downstream exporters — notably the Langfuse exporter — can populate the
/// Input/Output panels of a generation or tool observation.
///
/// Captured payloads flow through the same event pipeline as everything else, so
/// a [`RedactingSink`][crate::harness::observability::RedactingSink] configured
/// with secret substrings still masks them before they reach a journal or
/// exporter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadCapture {
    /// Capture the request messages and the model completion onto every
    /// [`AgentEvent::ModelCompleted`][crate::harness::events::AgentEvent::ModelCompleted].
    pub model_io: bool,
    /// Capture the tool arguments and the tool result onto every
    /// [`AgentEvent::ToolCompleted`][crate::harness::events::AgentEvent::ToolCompleted].
    pub tool_io: bool,
}

impl PayloadCapture {
    /// A capture policy with both model and tool I/O enabled.
    ///
    /// Prefer this only when the observability pipeline is trusted (or a
    /// [`RedactingSink`][crate::harness::observability::RedactingSink] is in
    /// place), since it journals and can export full prompt/completion text and
    /// tool arguments/results.
    pub const fn all() -> Self {
        Self {
            model_io: true,
            tool_io: true,
        }
    }

    /// `true` when neither model nor tool payloads are captured (the default).
    pub const fn is_disabled(&self) -> bool {
        !self.model_io && !self.tool_io
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunPolicy {
    /// Hard run limits enforced fail-closed by the agent loop.
    pub limits: RunLimits,
    /// How the loop reacts to a model call for an unregistered tool.
    pub unknown_tool: UnknownToolPolicy,
    /// How the loop reacts when a registered tool's arguments fail schema
    /// validation.
    pub invalid_args: InvalidArgsPolicy,
    /// Retry policy applied to each model call.
    pub retry: RetryPolicy,
    /// Optional ordered model fallback chain.
    pub fallback: Option<FallbackPolicy>,
    /// Response format attached to every model request when set.
    pub default_response_format: Option<ResponseFormat>,
    /// Whether the loop captures model/tool payloads onto completion events.
    ///
    /// Defaults to [`PayloadCapture::default`] (payload-free), preserving the
    /// privacy-preserving behavior where events carry only ids and usage.
    pub capture: PayloadCapture,
    /// Default caching policy for the run.
    ///
    /// The loop consults [`CachePolicy::response_cache_enabled`] only when a
    /// [`ResponseCache`] is attached to the harness *and* the per-call
    /// [`crate::harness::model::ModelRequest::cache_policy`] does not override
    /// it. A request-level `cache_policy` always wins over this default.
    pub cache: CachePolicy,
    /// When `true`, an empty provider completion in the finalization branch (no
    /// text, no tool calls, and no structured output) fails the run with
    /// [`crate::error::TinyAgentsError::EmptyResponse`] instead of terminating
    /// with a blank final answer.
    ///
    /// Defaults to `false` to preserve the historical behavior for callers who
    /// rely on empty finals; opt in to turn a silent blank success into a typed
    /// error the caller can re-prompt on.
    pub error_on_empty_response: bool,
}

impl Default for RunPolicy {
    fn default() -> Self {
        Self {
            limits: RunLimits::default(),
            unknown_tool: UnknownToolPolicy::default(),
            invalid_args: InvalidArgsPolicy::default(),
            retry: RetryPolicy::default(),
            fallback: None,
            default_response_format: None,
            capture: PayloadCapture::default(),
            // Caching defaults ON, but is gated by an attached `ResponseCache`,
            // so a harness with no cache never caches regardless of this flag.
            cache: CachePolicy {
                response_cache_enabled: true,
                protect_prompt_prefix: false,
            },
            // Opt-in: preserve the historical blank-final behavior by default.
            error_on_empty_response: false,
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
