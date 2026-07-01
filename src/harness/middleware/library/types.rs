//! Public types for the built-in middleware library.
//!
//! This module holds the type definitions for the ready-to-use middleware that
//! ship with the harness. They build on the two extension surfaces defined in
//! [`crate::harness::middleware`]:
//!
//! - the lifecycle [`Middleware`][crate::harness::middleware::Middleware] trait
//!   (`before_*`/`after_*`/`on_*` hooks), used by the policy/guard middleware
//!   here ([`ToolAllowlistMiddleware`], [`DynamicToolSelectionMiddleware`],
//!   [`HumanApprovalMiddleware`], [`StructuredOutputValidatorMiddleware`],
//!   [`DynamicPromptMiddleware`], [`RedactionMiddleware`],
//!   [`TracingMiddleware`]);
//! - the around-call [`ModelMiddleware`][crate::harness::middleware::ModelMiddleware]
//!   wrap trait, used by the resilience middleware here
//!   ([`RetryMiddleware`], [`TimeoutMiddleware`], [`ModelFallbackMiddleware`],
//!   [`RateLimitMiddleware`]).
//!
//! Behavioral code (constructors and trait impls) lives in the sibling
//! `mod.rs`; tests live in `test.rs`. Every public item is re-exported through
//! `crate::harness::middleware` so callers import from one place.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::harness::context::RunConfig;
use crate::harness::model::ResponseFormat;
use crate::harness::retry::RateLimiter;
use crate::harness::tool::{ToolCall, ToolSchema};

// ── RetryMiddleware ───────────────────────────────────────────────────────────

/// Around-model wrap middleware that retries the wrapped model call on
/// retryable errors.
///
/// Implements [`ModelMiddleware`][crate::harness::middleware::ModelMiddleware]:
/// it calls `next` (the rest of the onion and the real model call) and, when
/// that fails with a [retryable][crate::harness::retry::is_retryable] error and
/// the configured [`RetryPolicy`][crate::harness::retry::RetryPolicy] still
/// permits another attempt, retries. Each scheduled retry emits an
/// [`AgentEvent::RetryScheduled`][crate::harness::events::AgentEvent::RetryScheduled]
/// with a [`CallId`][crate::harness::ids::CallId] derived from the run id.
///
/// # Sleeping
///
/// Like the agent loop's own retry path, this middleware *computes* the backoff
/// from the policy but does **not** sleep, keeping the loop fast and tests
/// deterministic. A production integration may sleep for
/// [`RetryMiddleware::backoff_for_attempt`] before each retry.
///
/// # Failure mode
///
/// Non-retryable errors, or retryable errors once attempts are exhausted,
/// propagate unchanged.
pub struct RetryMiddleware {
    pub(crate) label: &'static str,
    pub(crate) policy: crate::harness::retry::RetryPolicy,
}

// ── TimeoutMiddleware ─────────────────────────────────────────────────────────

/// Around-model wrap middleware that bounds the wrapped model call with a
/// wall-clock timeout.
///
/// Implements [`ModelMiddleware`][crate::harness::middleware::ModelMiddleware]:
/// it races `next` against [`tokio::time::timeout`]; if the deadline elapses the
/// in-flight future is dropped (cancelling the underlying provider call) and a
/// [`TinyAgentsError::Timeout`][crate::error::TinyAgentsError::Timeout] is
/// returned.
///
/// # Failure mode
///
/// On elapse returns `Timeout`; otherwise propagates the wrapped call's result
/// unchanged.
pub struct TimeoutMiddleware {
    pub(crate) label: &'static str,
    pub(crate) timeout: Duration,
}

// ── ModelFallbackMiddleware ───────────────────────────────────────────────────

/// Around-model wrap middleware that retries the wrapped call against a chain of
/// fallback model names when the primary call fails.
///
/// Implements [`ModelMiddleware`][crate::harness::middleware::ModelMiddleware]:
/// it calls `next` with the request as-is; on error it sets
/// [`ModelRequest::model`][crate::harness::model::ModelRequest::model] to each
/// fallback name in order, emitting
/// [`AgentEvent::FallbackSelected`][crate::harness::events::AgentEvent::FallbackSelected]
/// before each attempt, and returns the first success.
///
/// The wrapped base call must honor `request.model` (re-resolve the model from
/// the request) for the swap to take effect; the harness exposes its own
/// registry-backed fallback for the default base, so this middleware is most
/// useful when a custom base resolves per-request models.
///
/// # Failure mode
///
/// If every fallback fails, the last error is returned.
pub struct ModelFallbackMiddleware {
    pub(crate) label: &'static str,
    pub(crate) fallbacks: Vec<String>,
}

// ── RateLimitMiddleware ───────────────────────────────────────────────────────

/// What a [`RateLimitMiddleware`] does when the token bucket has insufficient
/// capacity for a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateLimitBehavior {
    /// Fail the call immediately with
    /// [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded].
    Error,
    /// Wait (polling at the configured interval) until the bucket refills enough
    /// to admit the call, emitting
    /// [`AgentEvent::RateLimitWaited`][crate::harness::events::AgentEvent::RateLimitWaited]
    /// for each wait.
    Wait,
}

/// A clock closure returning the current instant, injected for deterministic
/// tests.
pub type NowFn = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Around-model wrap middleware that gates model calls through a shared
/// token-bucket [`RateLimiter`].
///
/// Implements [`ModelMiddleware`][crate::harness::middleware::ModelMiddleware]:
/// before calling `next` it attempts to acquire `tokens` from the limiter at the
/// current (injectable) clock. On success the call proceeds; on insufficient
/// capacity it follows the configured [`RateLimitBehavior`].
///
/// # Testability
///
/// The clock is injectable via [`RateLimitMiddleware::with_clock`] and the
/// poll interval (for [`RateLimitBehavior::Wait`]) is configurable, so tests can
/// drive the limiter deterministically without real sleeping (use a zero
/// interval and an advancing clock).
pub struct RateLimitMiddleware {
    pub(crate) label: &'static str,
    pub(crate) limiter: Arc<RateLimiter>,
    pub(crate) tokens: u64,
    pub(crate) behavior: RateLimitBehavior,
    pub(crate) poll_interval: Duration,
    pub(crate) now: NowFn,
}

// ── ToolAllowlistMiddleware ───────────────────────────────────────────────────

/// Lifecycle middleware that rejects tool calls whose name is not on an
/// allowlist.
///
/// Implements [`Middleware`][crate::harness::middleware::Middleware]'s
/// `before_tool` hook: if the [`ToolCall::name`] is not in the allowed set it
/// returns [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation]
/// before the tool runs.
pub struct ToolAllowlistMiddleware {
    pub(crate) label: &'static str,
    pub(crate) allowed: std::collections::HashSet<String>,
}

// ── BudgetMiddleware ──────────────────────────────────────────────────────────

/// Token and money budget limits enforced by a [`BudgetMiddleware`].
///
/// Every field is optional; an unset limit is not enforced. `warn_fraction`
/// (for example `0.9`) emits an
/// [`AgentEvent::BudgetWarning`][crate::harness::events::AgentEvent::BudgetWarning]
/// once cumulative usage/cost crosses that fraction of any set limit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BudgetLimits {
    /// Maximum cumulative input tokens.
    pub max_input_tokens: Option<u64>,
    /// Maximum cumulative output tokens.
    pub max_output_tokens: Option<u64>,
    /// Maximum cumulative total (effective) tokens.
    pub max_total_tokens: Option<u64>,
    /// Maximum cumulative reasoning tokens.
    pub max_reasoning_tokens: Option<u64>,
    /// Maximum cumulative estimated cost (pricing-table currency).
    pub max_cost: Option<f64>,
    /// Fraction of any limit (0.0–1.0) at which a warning is emitted.
    pub warn_fraction: Option<f64>,
}

/// Shared, accumulating budget spend.
///
/// Cloning a [`BudgetTracker`] shares the same underlying accumulator, so
/// handing the same tracker to a parent harness and every sub-agent harness
/// makes a single budget roll up across an entire recursive run tree.
#[derive(Clone, Debug, Default)]
pub struct BudgetTracker {
    pub(crate) inner: Arc<Mutex<BudgetSpend>>,
}

/// A point-in-time snapshot of accumulated budget spend.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BudgetSpend {
    /// Accumulated token usage.
    pub usage: crate::harness::usage::UsageTotals,
    /// Accumulated estimated cost.
    pub cost: crate::harness::cost::CostTotals,
    /// Whether a warning has already been emitted (warn-once).
    pub warned: bool,
}

/// Around-nothing lifecycle middleware that enforces a token/money
/// [`BudgetLimits`] across a run (or a shared run tree).
///
/// - `before_model` (preflight): if the accumulated spend already meets or
///   exceeds any limit, emits
///   [`AgentEvent::BudgetExceeded`][crate::harness::events::AgentEvent::BudgetExceeded]
///   `{ blocked: true }` and fails the call with
///   [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded],
///   so a recursive run stops once a root budget is exhausted.
/// - `after_model` (spend + reconcile): folds the response usage into the
///   tracker, prices it via the configured per-model [`ModelPricing`] table
///   (wiring cost into the loop, emitting
///   [`AgentEvent::UsageRecorded`][crate::harness::events::AgentEvent::UsageRecorded]
///   and [`AgentEvent::CostRecorded`][crate::harness::events::AgentEvent::CostRecorded]),
///   and emits warn/exceeded events as thresholds are crossed.
///
/// Reservation and refund semantics are intentionally out of scope here; this
/// middleware enforces post-hoc spend against limits with a fail-closed
/// preflight.
pub struct BudgetMiddleware {
    pub(crate) label: &'static str,
    pub(crate) limits: BudgetLimits,
    pub(crate) tracker: BudgetTracker,
    pub(crate) pricing: std::collections::HashMap<String, crate::registry::catalog::ModelPricing>,
}

// ── ToolPolicyMiddleware ──────────────────────────────────────────────────────

/// Lifecycle middleware that enforces per-tool [`ToolPolicy`] metadata, both at
/// model-visible exposure time (`before_model`) and at execution time
/// (`before_tool`).
///
/// Unlike [`ToolAllowlistMiddleware`] (name lists) and
/// [`DynamicToolSelectionMiddleware`] (schema-only predicates), this middleware
/// reads the structured [`ToolPolicy`] each tool advertises via
/// [`Tool::policy`][crate::harness::tool::Tool::policy]. Build it from a
/// registry snapshot with
/// [`ToolRegistry::policies`][crate::harness::tool::ToolRegistry::policies].
///
/// # Fail-closed behavior
///
/// - With [`require_classification`](Self::require_classification) set (the
///   default via [`strict`](Self::strict)), a tool whose policy is *unclassified*
///   (`ToolPolicy::classified == false`) — or has no snapshot entry at all — is
///   hidden from the model and rejected if called.
/// - Tools declaring any side effect in the [`deny`](Self::deny) mask are hidden
///   and rejected.
/// - When [`require_background_safe`](Self::require_background_safe) is set, tools
///   that are not `access.background_safe` are hidden and rejected.
///
/// Rejections at `before_tool` surface as
/// [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation].
pub struct ToolPolicyMiddleware {
    pub(crate) label: &'static str,
    pub(crate) policies: std::collections::HashMap<String, crate::harness::tool::ToolPolicy>,
    pub(crate) require_classification: bool,
    pub(crate) require_background_safe: bool,
    pub(crate) deny: crate::harness::tool::ToolSideEffects,
    /// When `true`, a tool whose runtime declares
    /// [`SandboxMode::Required`][crate::harness::tool::SandboxMode::Required] is
    /// blocked unless the run carries a workspace whose sandbox is `Required`.
    pub(crate) require_sandbox: bool,
    /// When `true`, a tool declaring `approval_required` is blocked unless its
    /// name is in [`approved`](Self::approved).
    pub(crate) require_approval: bool,
    /// Tools pre-approved for calls when [`require_approval`](Self::require_approval)
    /// is set.
    pub(crate) approved: std::collections::HashSet<String>,
    /// When `true`, a tool result larger than its declared `max_result_bytes` is
    /// truncated and flagged, enforcing the declared payload cap.
    pub(crate) enforce_result_bytes: bool,
}

// ── DynamicToolSelectionMiddleware ────────────────────────────────────────────

/// A predicate deciding whether a [`ToolSchema`] should be exposed to the model.
pub type ToolPredicate = Arc<dyn Fn(&ToolSchema) -> bool + Send + Sync>;

/// Lifecycle middleware that filters the tools exposed to the model on each
/// call.
///
/// Implements [`Middleware`][crate::harness::middleware::Middleware]'s
/// `before_model` hook: it retains only the
/// [`ModelRequest::tools`][crate::harness::model::ModelRequest::tools] for which
/// the configured [`ToolPredicate`] returns `true`, implementing dynamic tool
/// exposure (for example narrowing the toolset based on state or run tags).
///
/// This only changes what the model *sees*; the harness's tool registry is
/// untouched, so [`ToolAllowlistMiddleware`] should still guard execution if a
/// model calls a hidden tool.
pub struct DynamicToolSelectionMiddleware {
    pub(crate) label: &'static str,
    pub(crate) predicate: ToolPredicate,
}

// ── ContextualToolSelectionMiddleware ─────────────────────────────────────────

/// Run/agent/model context a [`ContextualToolPredicate`] can inspect when
/// deciding whether to expose a tool.
///
/// This closes the gap left by [`DynamicToolSelectionMiddleware`], whose
/// predicate sees only the [`ToolSchema`] and therefore cannot vary exposure by
/// recursion depth, run tags (security tier / background vs interactive), or the
/// model being called.
#[derive(Clone, Debug)]
pub struct ToolSelectionContext {
    /// The current run id.
    pub run_id: String,
    /// Recursion depth (0 for a top-level run; deeper for sub-agents).
    pub depth: usize,
    /// Run tags (for example a security tier or `background` marker).
    pub tags: Vec<String>,
    /// The model this request will be sent to, when set on the request.
    pub requested_model: Option<String>,
}

/// A predicate deciding whether a [`ToolSchema`] should be exposed, given run
/// context.
pub type ContextualToolPredicate =
    Arc<dyn Fn(&ToolSchema, &ToolSelectionContext) -> bool + Send + Sync>;

/// Lifecycle middleware that filters model-visible tools using a predicate that
/// receives both the [`ToolSchema`] and the live [`ToolSelectionContext`].
///
/// Build one directly from a context-aware predicate with [`new`](Self::new),
/// or from explicit allow/deny lists with
/// [`from_lists`](Self::from_lists) (deny wins; when an allow-list is present a
/// tool must appear in it — fail-closed for unknown tools).
pub struct ContextualToolSelectionMiddleware {
    pub(crate) label: &'static str,
    pub(crate) predicate: ContextualToolPredicate,
}

// ── HumanApprovalMiddleware ───────────────────────────────────────────────────

/// A callback consulted to approve or reject a flagged [`ToolCall`].
///
/// Returns `true` to allow the call to proceed, `false` to reject it (the
/// middleware then raises an interrupt).
pub type ApprovalFn = Arc<dyn Fn(&ToolCall) -> bool + Send + Sync>;

/// Lifecycle middleware implementing a simple human-in-the-loop gate for
/// sensitive tools.
///
/// Implements [`Middleware`][crate::harness::middleware::Middleware]'s
/// `before_tool` hook: when a tool's name is flagged as requiring approval, it
/// consults the optional [`ApprovalFn`]. If no callback is configured, or the
/// callback returns `false`, it raises
/// [`TinyAgentsError::Interrupted`][crate::error::TinyAgentsError::Interrupted]
/// (node `"tool"`) so the run pauses for human input.
///
/// # HITL hookup
///
/// This is the harness-native signal; the full graph interrupt/resume path is a
/// separate concern. A caller can supply an [`ApprovalFn`] that consults a UI,
/// queue, or policy store synchronously, or treat the `Interrupted` error as the
/// point at which to persist a checkpoint and surface an approval request.
pub struct HumanApprovalMiddleware {
    pub(crate) label: &'static str,
    pub(crate) flagged: std::collections::HashSet<String>,
    pub(crate) approve: Option<ApprovalFn>,
}

// ── StructuredOutputValidatorMiddleware ───────────────────────────────────────

/// Lifecycle middleware that validates a model response against an expected
/// structured-output format.
///
/// Implements [`Middleware`][crate::harness::middleware::Middleware]'s
/// `after_model` hook. Because the hook does not see the original request, the
/// expected [`ResponseFormat`] is supplied at construction:
///
/// - [`ResponseFormat::Text`] — no validation.
/// - [`ResponseFormat::JsonObject`] — the response text must parse as JSON.
/// - [`ResponseFormat::JsonSchema`] / [`ResponseFormat::Auto`] — extracted via a
///   provider-schema [`StructuredExtractor`][crate::harness::structured::StructuredExtractor].
///
/// On failure it returns
/// [`TinyAgentsError::StructuredOutput`][crate::error::TinyAgentsError::StructuredOutput].
pub struct StructuredOutputValidatorMiddleware {
    pub(crate) label: &'static str,
    pub(crate) format: ResponseFormat,
}

// ── DynamicPromptMiddleware ───────────────────────────────────────────────────

/// A closure deriving an optional system prompt from application state and the
/// run's [`RunConfig`].
pub type PromptFn<State> = Arc<dyn Fn(&State, &RunConfig) -> Option<String> + Send + Sync>;

/// Lifecycle middleware that injects a derived system message before each model
/// call.
///
/// Implements [`Middleware`][crate::harness::middleware::Middleware]'s
/// `before_model` hook: it calls the configured [`PromptFn`] with the shared
/// `&State` and the run's [`RunConfig`]; when it returns `Some(text)` a
/// [`Message::system`] is inserted at the front of
/// [`ModelRequest::messages`][crate::harness::model::ModelRequest::messages].
///
/// Generic over `State`/`Ctx` because the closure reads application state.
pub struct DynamicPromptMiddleware<State, Ctx = ()> {
    pub(crate) label: &'static str,
    pub(crate) prompt: PromptFn<State>,
    pub(crate) _marker: PhantomData<fn(Ctx)>,
}

// ── RedactionMiddleware ───────────────────────────────────────────────────────

/// Lifecycle middleware that redacts configured secret/PII substrings from text
/// before it leaves the harness.
///
/// Implements [`Middleware`][crate::harness::middleware::Middleware]'s
/// `after_model` and `after_tool` hooks: every configured pattern found in the
/// model response text or tool result content is replaced with the mask string.
///
/// Patterns are literal substrings (no regex dependency); supply pre-built
/// patterns for the secrets you need to scrub. The number of redactions is
/// tracked and available via [`RedactionMiddleware::redactions`].
pub struct RedactionMiddleware {
    pub(crate) label: &'static str,
    pub(crate) patterns: Vec<String>,
    pub(crate) mask: String,
    pub(crate) redactions: Mutex<usize>,
}

// ── TracingMiddleware ─────────────────────────────────────────────────────────

/// A structured begin/end record captured by [`TracingMiddleware`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseTrace {
    /// The lifecycle phase, for example `"agent"`, `"model"`, or `"tool"`.
    pub phase: &'static str,
    /// Whether this record marks the start or the end of the phase.
    pub boundary: TraceBoundary,
}

/// Whether a [`PhaseTrace`] marks the beginning or end of a phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceBoundary {
    /// The phase is starting (a `before_*`/`on_*` hook fired).
    Begin,
    /// The phase is ending (an `after_*` hook fired).
    End,
}

/// Lifecycle middleware that records structured begin/end traces and per-phase
/// counts for an entire run.
///
/// Implements every lifecycle [`Middleware`][crate::harness::middleware::Middleware]
/// hook: each one appends a [`PhaseTrace`] to an internal recorder and bumps a
/// per-phase counter. The recorder is retrievable via
/// [`TracingMiddleware::records`] and counts via [`TracingMiddleware::counts`],
/// giving tests and dashboards a structured timeline without parsing the event
/// stream. (The surrounding [`MiddlewareStack`][crate::harness::middleware::MiddlewareStack]
/// also emits `MiddlewareStarted`/`MiddlewareCompleted` events around each hook.)
pub struct TracingMiddleware {
    pub(crate) label: &'static str,
    pub(crate) records: Mutex<Vec<PhaseTrace>>,
    pub(crate) counts: Mutex<TraceCounts>,
}

/// Per-phase begin counts captured by [`TracingMiddleware`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceCounts {
    /// Number of `before_agent` begins.
    pub agent: usize,
    /// Number of `before_model` begins.
    pub model: usize,
    /// Number of `before_tool` begins.
    pub tool: usize,
    /// Number of streamed model/tool deltas observed.
    pub delta: usize,
    /// Number of `on_error` invocations.
    pub error: usize,
}
