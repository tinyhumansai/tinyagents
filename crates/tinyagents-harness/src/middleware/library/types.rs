//! Public types for the built-in middleware library.
//!
//! This module holds the type definitions for the ready-to-use middleware that
//! ship with the harness. They build on extension surfaces defined in
//! [`crate::middleware`]:
//!
//! - the lifecycle [`Middleware`][crate::middleware::Middleware] trait
//!   (`before_*`/`after_*`/`on_*` hooks), used by the policy/guard middleware
//!   here ([`ToolAllowlistMiddleware`], [`DynamicToolSelectionMiddleware`],
//!   [`HumanApprovalMiddleware`], [`StructuredOutputValidatorMiddleware`],
//!   [`DynamicPromptMiddleware`], [`RedactionMiddleware`],
//!   [`TracingMiddleware`]);
//! - the around-call [`ModelMiddleware`][crate::middleware::ModelMiddleware]
//!   wrap trait, used by the resilience middleware here
//!   ([`RetryMiddleware`], [`TimeoutMiddleware`], [`ModelFallbackMiddleware`],
//!   [`RateLimitMiddleware`]).
//!
//! Behavioral code (constructors and trait impls) lives in the sibling
//! `mod.rs`; tests live in `test.rs`. Every public item is re-exported through
//! `crate::middleware` so callers import from one place.

use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::context::RunConfig;
use crate::retry::RateLimiter;
use tinyinference_llm::model::ResponseFormat;
use tinyinference_llm::tool::{ToolCall, ToolSchema};
use tinytools::{ToolPolicy, ToolSideEffects};

// ── RetryMiddleware ───────────────────────────────────────────────────────────

/// Around-model wrap middleware that retries the wrapped model call on
/// retryable errors.
///
/// Implements [`ModelMiddleware`][crate::middleware::ModelMiddleware]:
/// it calls `next` (the rest of the onion and the real model call) and, when
/// that fails with a [retryable][crate::retry::is_retryable] error and
/// the configured [`RetryPolicy`][crate::retry::RetryPolicy] still
/// permits another attempt, retries. Each scheduled retry emits an
/// [`AgentEvent::RetryScheduled`][crate::events::AgentEvent::RetryScheduled]
/// with the same [`CallId`][crate::ids::CallId] the agent loop is using for
/// the in-flight call (mirrored onto
/// [`RunContext::active_model_call`][crate::context::RunContext::active_model_call]),
/// falling back to a run-scoped id only when this middleware runs outside the
/// agent loop.
///
/// # An alternative to `RunPolicy::retry`, not a companion
///
/// This middleware and the loop's own [`RunPolicy::retry`][crate::runtime::RunPolicy::retry]
/// are two implementations of the same idea. Registering both does not
/// compose them: [`crate::middleware::MiddlewareStack::has_retry_override`]
/// tells the loop's base call to skip its own retry loop whenever any
/// `ModelMiddleware` reports [`ModelMiddleware::overrides_retry`][crate::middleware::ModelMiddleware::overrides_retry]
/// (this middleware always does), so only this middleware's `RetryPolicy`
/// governs the attempt count — `RunPolicy::retry` is ignored for the base
/// call while it is registered. Without that guard the two layers would
/// multiply attempts (`mw.max_attempts x policy.retry.max_attempts x
/// |fallback|` provider calls for one logical failure); see I-7. Prefer
/// `RunPolicy::retry` for the common case (it also drives the fallback
/// chain) and reach for this middleware only when retry needs to run at a
/// specific point in the wrap onion (e.g. after a guardrail middleware has
/// already inspected the request). Full unification into one retry engine is
/// tracked as a later phase.
///
/// # Sleeping
///
/// This middleware sleeps for the policy's computed backoff between attempts
/// only when the policy opts in via
/// [`RetryPolicy::with_backoff_sleep`][crate::retry::RetryPolicy::with_backoff_sleep];
/// otherwise it retries back-to-back, keeping tests fast and deterministic.
///
/// # Failure mode
///
/// Non-retryable errors, or retryable errors once attempts are exhausted,
/// propagate unchanged.
pub struct RetryMiddleware {
    pub(crate) label: &'static str,
    pub(crate) policy: crate::retry::RetryPolicy,
}

// ── TimeoutMiddleware ─────────────────────────────────────────────────────────

/// Around-model wrap middleware that bounds the wrapped model call with a
/// wall-clock timeout.
///
/// Implements [`ModelMiddleware`][crate::middleware::ModelMiddleware]:
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
/// Implements [`ModelMiddleware`][crate::middleware::ModelMiddleware]:
/// it calls `next` with the request as-is; on error it sets
/// [`ModelRequest::model`][tinyinference_llm::model::ModelRequest::model] to each
/// fallback name in order, emitting
/// [`AgentEvent::FallbackSelected`][crate::events::AgentEvent::FallbackSelected]
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
    /// Fail immediately: a temporarily empty bucket that can refill returns
    /// [`TinyAgentsError::RateLimited`][crate::error::TinyAgentsError::RateLimited],
    /// while a bucket that can never admit the requested tokens returns
    /// [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded].
    Error,
    /// Wait (polling at the configured interval) until the bucket refills enough
    /// to admit the call, then emit a single
    /// [`AgentEvent::RateLimitWaited`][crate::events::AgentEvent::RateLimitWaited]
    /// carrying the actual wall-clock time waited. When waiting can never
    /// succeed — the limiter's refill rate is zero (or negative), or the call
    /// requests more tokens than the bucket capacity — the call fails fast
    /// with [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded]
    /// instead of polling forever.
    Wait,
}

/// A clock closure returning the current instant, injected for deterministic
/// tests.
pub type NowFn = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Around-model wrap middleware that gates model calls through a shared
/// token-bucket [`RateLimiter`].
///
/// Implements [`ModelMiddleware`][crate::middleware::ModelMiddleware]:
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
/// Implements [`Middleware`][crate::middleware::Middleware]'s
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
/// [`AgentEvent::BudgetWarning`][crate::events::AgentEvent::BudgetWarning]
/// once cumulative usage/cost crosses that fraction of any set limit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BudgetLimits {
    /// Maximum cumulative input tokens.
    pub max_input_tokens: Option<u64>,
    /// Maximum cumulative cache-read (cached) input tokens.
    pub max_cached_input_tokens: Option<u64>,
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
    pub usage: tinyinference_llm::usage::UsageTotals,
    /// Accumulated estimated cost.
    pub cost: crate::cost::CostTotals,
    /// Whether a warning has already been emitted (warn-once).
    pub warned: bool,
    /// Sum of input tokens preflight-reserved by calls that have not yet
    /// reconciled in `after_model`. Shared trackers (handed to concurrent
    /// sub-agent runs) can have more than one outstanding reservation at
    /// once, so this is a running total, not a single call's estimate;
    /// each in-flight call's own reservation is tracked separately, per run, by
    /// its [`BudgetMiddleware`] and released from this total when it reconciles
    /// (or is abandoned).
    pub reserved_input_total: u64,
}

/// Around-nothing lifecycle middleware that enforces a token/money
/// [`BudgetLimits`] across a run (or a shared run tree).
///
/// - `before_model` (preflight): if the accumulated spend already meets or
///   exceeds any limit, emits
///   [`AgentEvent::BudgetExceeded`][crate::events::AgentEvent::BudgetExceeded]
///   `{ blocked: true }` and fails the call with
///   [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded],
///   so a recursive run stops once a root budget is exhausted.
/// - `after_model` (spend + reconcile): folds the response usage into the
///   tracker, prices it via the configured per-model [`ModelPricing`](crate::cost::ModelPricing) table
///   (wiring cost into the loop, emitting
///   [`AgentEvent::UsageRecorded`][crate::events::AgentEvent::UsageRecorded]
///   and [`AgentEvent::CostRecorded`][crate::events::AgentEvent::CostRecorded]),
///   and emits warn/exceeded events as thresholds are crossed.
///
/// Reservation and refund semantics are intentionally out of scope here; this
/// middleware enforces post-hoc spend against limits with a fail-closed
/// preflight.
pub struct BudgetMiddleware {
    pub(crate) label: &'static str,
    pub(crate) limits: BudgetLimits,
    pub(crate) tracker: BudgetTracker,
    pub(crate) pricing: std::collections::HashMap<String, crate::cost::ModelPricing>,
    /// Outstanding preflight reservations (input tokens) awaiting
    /// reconciliation in `after_model`, keyed by
    /// [`RunContext::instance_id`][crate::context::RunContext::instance_id].
    ///
    /// One middleware instance serves every run on the harness it is registered
    /// with (`invoke` takes `&self`), so a single scalar here would be
    /// overwritten whenever two runs interleave their model calls — each would
    /// then release the other's amount and permanently skew
    /// [`BudgetTracker::reserved_input_total`][crate::middleware::BudgetSpend::reserved_input_total].
    /// Keying by the run context's process-unique instance id (not its
    /// caller-supplied `run_id`, which concurrent runs may share) keeps each
    /// run releasing exactly what it reserved.
    pub(crate) pending_reservations: std::sync::Mutex<std::collections::HashMap<u64, u64>>,
}

// ── ToolPolicyMiddleware ──────────────────────────────────────────────────────

/// Lifecycle middleware that enforces per-tool [`ToolPolicy`] metadata, both at
/// model-visible exposure time (`before_model`) and at execution time
/// (`before_tool`).
///
/// Unlike [`ToolAllowlistMiddleware`] (name lists) and
/// [`DynamicToolSelectionMiddleware`] (schema-only predicates), this middleware
/// reads the structured [`ToolPolicy`] each tool advertises via
/// [`Tool::policy`](tinytools::Tool::policy). Build it from a
/// registry snapshot with
/// [`ToolRegistry::policies`][crate::tool::ToolRegistry::policies].
///
/// # Fail-closed behavior
///
/// - With [`require_classification`](Self::require_classification) set (the
///   default via [`strict`](Self::strict)), a tool whose policy is *unclassified*
///   (`ToolPolicy::classified == false`) — or has no snapshot entry at all — is
///   hidden from the model and rejected if called.
/// - Tools declaring any side effect in the `deny` mask are hidden
///   and rejected.
/// - When [`require_background_safe`](Self::require_background_safe) is set, tools
///   that are not `access.background_safe` are hidden and rejected.
///
/// Rejections at `before_tool` surface as
/// [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation].
///
/// # Relationship to `crate::tool::toolset`
///
/// This middleware's policy classification (side-effect/background-safe/
/// approval enforcement from the vendor `tinytools` declaration) is a
/// different axis from [`crate::tool::toolset::FilteredToolSet`] (an
/// arbitrary per-tool predicate) and
/// [`crate::tool::toolset::ApprovalRequiredToolSet`] (which only *sets* the
/// approval flag this middleware enforces) — kept as its own implementation
/// rather than rebased onto either, since neither adaptor reads
/// [`ToolPolicy`] as a whole.
pub struct ToolPolicyMiddleware {
    pub(crate) label: &'static str,
    pub(crate) policies: std::collections::HashMap<String, ToolPolicy>,
    pub(crate) require_classification: bool,
    pub(crate) require_background_safe: bool,
    pub(crate) deny: ToolSideEffects,
    /// When `true`, a tool whose runtime declares
    /// [`SandboxMode::Required`][crate::tool::SandboxMode::Required] is
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
    /// When `true`, the reserved intrinsic discovery-bridge name
    /// (`tool_search`) is exempted from classification/side-effect
    /// checks whenever `policies` has no entry for them.
    ///
    /// Off by default, including under [`Self::strict`]: exempting *every*
    /// policy-less tool sharing this magic name is only safe when the
    /// caller can guarantee `policies` is the *complete* registry snapshot
    /// (so "no entry" reliably means "not a registered tool, must be the
    /// intrinsic bridge"). An incomplete or stale snapshot could otherwise let
    /// a real, side-effecting host tool that happens to be registered under
    /// this reserved name bypass `strict()`'s fail-closed checks.
    /// Enable explicitly with
    /// [`exempt_discovery_bridge`](Self::exempt_discovery_bridge) when using
    /// the harness's own `tool_search` bridge together with a
    /// policy snapshot you trust to be complete.
    pub(crate) exempt_discovery_bridge: bool,
}

// ── DynamicToolSelectionMiddleware ────────────────────────────────────────────

/// A predicate deciding whether a [`ToolSchema`] should be exposed to the model.
pub type ToolPredicate = Arc<dyn Fn(&ToolSchema) -> bool + Send + Sync>;

/// Lifecycle middleware that filters the tools exposed to the model on each
/// call.
///
/// Implements [`Middleware`][crate::middleware::Middleware]'s
/// `before_model` hook: it retains only the
/// [`ModelRequest::tools`][tinyinference_llm::model::ModelRequest::tools] for which
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
///
/// # Relationship to `crate::tool::toolset`
///
/// [`DynamicToolSelectionMiddleware`] and
/// [`crate::tool::toolset::PreparedToolSet`] both operate on a bare
/// [`ToolSchema`] predicate; this middleware additionally reads
/// [`ToolSelectionContext`] (depth, tags, the requested model), which a
/// `ToolSet::tools`'s own `ctx: &RunContext<Ctx>` argument can already carry
/// through `Ctx` — kept as its own predicate type rather than folded into
/// [`crate::tool::toolset::PreparedToolSet::filtering`] to avoid coupling
/// every `Ctx` to this specific context shape.
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

/// What an approval callback decided about one flagged [`ToolCall`] (A2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalOutcome {
    /// Run the call now.
    Allow,
    /// Do not run it; the model sees the message as a tool-error result and
    /// the run continues (no interrupt, no deferral).
    Deny(String),
    /// Hand the call back to the host: the loop finishes the batch's other
    /// calls and exits with `AgentRun::deferred` listing this call under
    /// `approvals` (or resolves it through a registered
    /// [`DeferredToolHandler`][crate::tool::DeferredToolHandler]). On resume
    /// the middleware sees the approval through
    /// [`RunContext::is_call_approved`][crate::context::RunContext::is_call_approved]
    /// and lets the call through.
    Defer,
}

/// A richer approval callback returning an [`ApprovalOutcome`] instead of a
/// bare `bool`; see [`HumanApprovalMiddleware::with_approval_outcome`].
pub type ApprovalOutcomeFn = Arc<dyn Fn(&ToolCall) -> ApprovalOutcome + Send + Sync>;

/// Lifecycle middleware implementing a simple human-in-the-loop gate for
/// sensitive tools.
///
/// Implements [`Middleware`][crate::middleware::Middleware]'s
/// `before_tool` hook: when a tool's name is flagged as requiring approval, it
/// consults the optional [`ApprovalFn`]. If no callback is configured, or the
/// callback returns `false`, it raises
/// [`TinyAgentsError::Interrupted`][crate::error::TinyAgentsError::Interrupted]
/// (node `"tool"`) so the run pauses for human input.
///
/// An [`ApprovalOutcomeFn`] (see [`Self::with_approval_outcome`]) replaces
/// the bare `bool` with [`ApprovalOutcome::{Allow, Deny, Defer}`]: `Deny`
/// answers the model with a tool-error result instead of interrupting, and
/// `Defer` turns the call into a resumable deferred request (A2).
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
    /// Takes precedence over `approve` when set (A2).
    pub(crate) outcome: Option<ApprovalOutcomeFn>,
}

// ── StructuredOutputValidatorMiddleware ───────────────────────────────────────

/// Lifecycle middleware that validates a model response against an expected
/// structured-output format.
///
/// Implements [`Middleware`][crate::middleware::Middleware]'s
/// `after_model` hook. Because the hook does not see the original request, the
/// expected [`ResponseFormat`] is supplied at construction:
///
/// - [`ResponseFormat::Text`] — no validation.
/// - [`ResponseFormat::JsonObject`] — the response text must parse as JSON.
/// - [`ResponseFormat::JsonSchema`] / [`ResponseFormat::Auto`] — extracted via a
///   provider-schema [`StructuredExtractor`][crate::structured::StructuredExtractor].
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
/// Implements [`Middleware`][crate::middleware::Middleware]'s
/// `before_model` hook: it calls the configured [`PromptFn`] with the shared
/// `&State` and the run's [`RunConfig`]; when it returns `Some(text)` a
/// [`Message::system`](tinyinference_llm::message::Message::system) is inserted at the front of
/// [`ModelRequest::messages`][tinyinference_llm::model::ModelRequest::messages].
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
/// Implements [`Middleware`][crate::middleware::Middleware]'s
/// `after_model`, `before_tool`, and `after_tool` hooks: every configured
/// pattern found in model response text/JSON blocks, model-authored tool-call
/// arguments, raw provider/tool payloads, tool result content, or tool error
/// messages is replaced with the mask string.
///
/// Patterns are literal substrings (no regex dependency); supply pre-built
/// patterns for the secrets you need to scrub. The number of redactions is
/// tracked and available via [`RedactionMiddleware::redactions`].
///
/// Redaction is idempotent and never self-matching: text is scanned in a
/// single pass over the original input (so a pattern cannot match inside mask
/// text produced by an earlier replacement), and existing occurrences of the
/// full mask string are treated as opaque (so re-running over already-redacted
/// text is a no-op, even when a pattern is a substring of the mask). A pattern
/// exactly equal to the mask is therefore never matched.
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

/// Default cap on the number of [`PhaseTrace`] entries a [`TracingMiddleware`]
/// retains before evicting the oldest. Long-running agent loops otherwise grow
/// this recorder without bound.
pub const DEFAULT_TRACE_RECORD_CAP: usize = 1024;

/// Lifecycle middleware that records structured begin/end traces and per-phase
/// counts for an entire run.
///
/// Implements every lifecycle [`Middleware`][crate::middleware::Middleware]
/// hook: each one appends a [`PhaseTrace`] to an internal recorder and bumps a
/// per-phase counter. The recorder is retrievable via
/// [`TracingMiddleware::records`] and counts via [`TracingMiddleware::counts`],
/// giving tests and dashboards a structured timeline without parsing the event
/// stream. (The surrounding [`MiddlewareStack`][crate::middleware::MiddlewareStack]
/// also emits `MiddlewareStarted`/`MiddlewareCompleted` events around each hook.)
///
/// The recorder is a bounded ring buffer (default cap
/// [`DEFAULT_TRACE_RECORD_CAP`], configurable via
/// [`TracingMiddleware::with_max_records`]): once full, the oldest trace is
/// dropped to make room for the newest, so an unbounded run cannot grow this
/// middleware's memory footprint forever.
pub struct TracingMiddleware {
    pub(crate) label: &'static str,
    pub(crate) records: Mutex<VecDeque<PhaseTrace>>,
    pub(crate) counts: Mutex<TraceCounts>,
    pub(crate) max_records: usize,
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

// ── PlanModeMiddleware ─────────────────────────────────────────────────────────

/// A per-run mode gating side-effecting tools, toggled by the host without
/// restarting the run.
///
/// `Build` (the default) leaves tool exposure and execution unrestricted.
/// `Plan` hides every side-effecting tool from the model
/// ([`PlanModeMiddleware`]'s `before_model`, the same auditable
/// `AgentEvent::ToolsFiltered` mechanism [`ContextualToolSelectionMiddleware`]
/// uses) and denies it at execution (`before_tool`, the same
/// [`ToolPolicy`]/[`ToolSideEffects`] classification
/// [`ToolPolicyMiddleware::deny_side_effects`] enforces) — except for tools on
/// the middleware's allowlist, which stay available in either mode so a host
/// can keep read-only tools and a handful of plan-mode-specific tools (e.g.
/// `plan_exit`, `request_plan_review`, `todo`) reachable while planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunMode {
    /// Every registered tool may be exposed and executed. The default.
    #[default]
    Build,
    /// Only allowlisted and side-effect-free tools may be exposed or executed.
    Plan,
}

/// A shared, host-settable handle to the current [`RunMode`] for a run.
///
/// Cloning shares the same underlying state: a host can hold one clone to
/// flip modes (e.g. from a UI toggle or a `plan_exit` tool call) while
/// [`PlanModeMiddleware`] holds another to read it. [`Self::set`] takes
/// effect on the very next tool exposure or execution check — no run restart
/// is required, so a host can switch modes mid-run, between turns.
#[derive(Debug, Clone)]
pub struct RunModeHandle(pub(crate) Arc<std::sync::atomic::AtomicU8>);

impl RunModeHandle {
    /// Creates a handle starting in `mode`.
    pub fn new(mode: RunMode) -> Self {
        Self(Arc::new(std::sync::atomic::AtomicU8::new(Self::encode(
            mode,
        ))))
    }

    /// Returns the current mode.
    pub fn get(&self) -> RunMode {
        Self::decode(self.0.load(std::sync::atomic::Ordering::SeqCst))
    }

    /// Sets the current mode.
    pub fn set(&self, mode: RunMode) {
        self.0
            .store(Self::encode(mode), std::sync::atomic::Ordering::SeqCst);
    }

    fn encode(mode: RunMode) -> u8 {
        match mode {
            RunMode::Build => 0,
            RunMode::Plan => 1,
        }
    }

    fn decode(value: u8) -> RunMode {
        match value {
            1 => RunMode::Plan,
            _ => RunMode::Build,
        }
    }
}

impl Default for RunModeHandle {
    fn default() -> Self {
        Self::new(RunMode::default())
    }
}

/// Lifecycle middleware that enforces [`RunMode::Plan`] by hiding and denying
/// side-effecting tools, driven by a live [`RunModeHandle`].
///
/// Build with [`plan_mode_middleware`] or [`PlanModeMiddleware::new`], then
/// widen the plan-mode allowlist with [`PlanModeMiddleware::allow`]. A tool is
/// treated as side-effecting when its [`ToolPolicy::side_effects`] declares
/// any of `writes_files`, `network`, `installs_dependencies`, `destructive`,
/// `external_service`, or `payment` — or when `policies` has no entry for it
/// at all (fail-closed: an unclassified tool is assumed capable of side
/// effects until proven otherwise).
pub struct PlanModeMiddleware {
    pub(crate) label: &'static str,
    pub(crate) mode: RunModeHandle,
    pub(crate) policies: std::collections::HashMap<String, ToolPolicy>,
    pub(crate) allow: HashSet<String>,
}

/// Creates a [`PlanModeMiddleware`] driven by `mode`, classifying tools from
/// `policies` (typically [`ToolRegistry::policies`][crate::tool::ToolRegistry::policies]).
///
/// Equivalent to [`PlanModeMiddleware::new`]; provided as a free function so a
/// host can wire plan mode into a [`MiddlewareStack`][crate::middleware::MiddlewareStack]
/// with one call.
pub fn plan_mode_middleware(
    mode: RunModeHandle,
    policies: std::collections::HashMap<String, ToolPolicy>,
) -> PlanModeMiddleware {
    PlanModeMiddleware::new(mode, policies)
}
