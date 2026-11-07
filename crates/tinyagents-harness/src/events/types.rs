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
//! `crate::events`. Implementations, free functions, and tests live in
//! the sibling `mod.rs` and `test.rs` files.

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::GenerationEnrichment;
use crate::cost::CostTotals;
use crate::ids::{CallId, ComponentId, EventId, ExecutionStatus, HarnessPhase, RunId, ThreadId};
use tinyinference_llm::message::MessageDelta;
use tinyinference_llm::usage::{Usage, UsageTotals};

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
#[non_exhaustive]
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
        /// The run that produced this delta. Attributed explicitly so a UI can
        /// route a delta to its run/thread lineage without depending on which
        /// sink it arrived on — sinks are shared across a recursive run tree.
        run_id: RunId,
        /// Identifier for the model call that produced this delta.
        call_id: CallId,
        /// Incremental text and/or tool-call fragment.
        delta: MessageDelta,
    },

    /// A model call completed successfully.
    ModelCompleted {
        /// Identifier for the model call that completed.
        call_id: CallId,
        /// Wall-clock time the model call *started*, in Unix-epoch
        /// milliseconds. Captured by the agent loop when it dispatches the
        /// call (alongside [`AgentEvent::ModelStarted`]) so exporters can
        /// render a real duration instead of a zero-width point. `None` for
        /// events serialized before this field existed (`#[serde(default)]`
        /// keeps old journals deserializable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        started_at_ms: Option<u64>,
        /// Token usage reported by the provider, when available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// The request messages sent to the model, captured only when
        /// [`PayloadCapture::model_io`][crate::runtime::PayloadCapture::model_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the prompt in a generation's Input panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        /// The model completion (assistant message), captured only when
        /// [`PayloadCapture::model_io`][crate::runtime::PayloadCapture::model_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the completion in a generation's Output panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
    },

    /// Content-free provider, pricing, and estimate metadata for one call.
    ///
    /// Hosts emit this after the facts are known. It may precede
    /// [`AgentEvent::ModelCompleted`] in durable observations; the trace
    /// collector retains it only while that exact call is still open.
    GenerationEnriched {
        /// Call identity, scoped by the enclosing observation's run.
        call_id: CallId,
        /// Validated host metadata for this model generation.
        enrichment: GenerationEnrichment,
    },

    /// The agent loop fixed the run's **pre-middleware** tool surface: how
    /// many schemas were assembled from the registry (direct tools plus the
    /// bridge tools when any tool is deferred), how many are deferred behind
    /// `tool_search`, and what that base set costs in bytes. Emitted once per
    /// run, before the first model call and before `before_agent`/
    /// `before_model` middleware runs.
    ///
    /// This is a fixed run-start baseline, not a live per-request wire
    /// metric: exposure-narrowing middleware
    /// (`ToolPolicyMiddleware::before_model`, dynamic/contextual tool
    /// selection) can still shrink `request.tools` on any given turn, and a
    /// structured-output tool-call fallback can still grow it. Track this
    /// event for the ceiling the run started with, not for what a specific
    /// request actually sent.
    ToolsAdvertised {
        /// Count of `Direct`-exposure tool schemas assembled before per-turn
        /// middleware runs. Does **not** include the intrinsic `tool_search`
        /// bridge schema added to the wire set when `deferred > 0` — it is
        /// implied by `deferred` being nonzero, not double-counted here.
        direct: usize,
        /// Tools reachable only through `tool_search` (called by their own name).
        deferred: usize,
        /// Compact-JSON size of the actual pre-middleware wire schema set
        /// (the `direct` schemas plus the bridge schema when
        /// `deferred > 0`), not of whatever a specific request's
        /// `before_model` pass narrows or grows it to.
        schema_bytes: usize,
    },

    /// The model searched the deferred-tool catalogue through the intrinsic
    /// `tool_search` bridge.
    ToolSearched {
        /// Identifier of the `tool_search` call.
        call_id: CallId,
        /// The model's query, verbatim — but only when
        /// [`RunPolicy::capture`][crate::runtime::RunPolicy::capture]`.tool_io`
        /// is enabled (default `false`, payload-free); empty string
        /// otherwise. Same privacy class and gate as a normal successful
        /// tool call's arguments.
        query: String,
        /// Number of deferred tools returned.
        matched: usize,
        /// Which ranker's answer was served: `"bm25"`, or the host ranker's
        /// [`tinytools::ToolRanker::kind`]. Empty when the query was rejected
        /// before ranking.
        #[serde(default)]
        ranker: String,
        /// The best hit's calibrated confidence, when the ranker gave one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_confidence: Option<f64>,
        /// Why the host ranker was not served, when one was active.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fallback: Option<String>,
        /// The BM25 ranking, when the policy asked to compare it against the
        /// served one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shadow_matched: Option<Vec<String>>,
        /// Wall time of the ranking, in milliseconds.
        #[serde(default)]
        latency_ms: u64,
    },

    /// The model called a deferred tool by its own name (it was reachable
    /// only through `tool_search`, never on the request's `tools`). Emitted at
    /// admission, before the `ToolStarted` for the same call.
    DeferredToolCall {
        /// Identifier of the model's tool call.
        call_id: CallId,
        /// The deferred tool that was called.
        tool_name: String,
    },

    /// A tool call was deferred out of the loop (A2): it needs a human
    /// approval or host-side execution before it can be answered. The loop
    /// finishes the batch's other calls and exits with
    /// `AgentRun::deferred`, or resolves it inline through a registered
    /// `DeferredToolHandler`. Terminal partner of a `ToolStarted` when the
    /// tool itself raised the deferral mid-execution.
    ToolDeferred {
        /// Identifier of the deferred call.
        call_id: CallId,
        /// Why it was deferred (`approval_required`, `call_deferred`,
        /// `external`, or a middleware-supplied reason).
        reason: String,
    },

    /// A previously deferred call was approved on resume and is about to
    /// execute (with the model's or the approver's edited arguments).
    ToolApproved {
        /// Identifier of the approved call.
        call_id: CallId,
    },

    /// A previously deferred call was denied on resume; no tool runs and the
    /// model sees `message` as a tool-error result.
    ToolDenied {
        /// Identifier of the denied call.
        call_id: CallId,
        /// The denial message handed to the model.
        message: String,
    },

    /// A tool-selection middleware filtered the model-visible tool set before a
    /// model call. Makes exposure decisions auditable: a UI or log can see
    /// which tools were withheld from the model and by which policy.
    ToolsFiltered {
        /// Name of the middleware/policy that made the decision.
        by: String,
        /// Tools removed from the model-visible set, in their original order.
        excluded: Vec<String>,
        /// Number of tools left exposed to the model.
        remaining: usize,
        /// Per-tool reason a [`crate::tool::toolset::ToolSet`] adaptor
        /// changed or withheld a tool this turn, keyed by the tool's
        /// original name.
        ///
        /// Additive (`docs/sdk-gaps/tools.md` §9's "explainable exposure
        /// decisions"): `#[serde(default)]` keeps events recorded before
        /// this field existed deserializable, and a middleware that only
        /// reports `excluded` (no explanations) leaves this empty rather
        /// than failing to construct the event.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        explanations: Vec<(String, crate::tool::ToolExposureExplanation)>,
    },

    /// A tool invocation has been dispatched.
    ToolStarted {
        /// Identifier for this tool call, correlates with completion.
        call_id: CallId,
        /// Name of the tool being invoked.
        tool_name: String,
        /// The arguments the tool is being invoked with, captured only when
        /// [`PayloadCapture::tool_io`][crate::runtime::PayloadCapture::tool_io]
        /// is enabled. `None` in the default payload-free mode, and for
        /// events serialized before this field existed. Populated so a host
        /// or UI can render the call's arguments as soon as it starts,
        /// instead of waiting for [`AgentEvent::ToolCompleted`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        /// The call whose tool made this one through
        /// `ToolExecutionContext::call_tool` — the **immediate** parent, which
        /// is itself nested when this call is two or more levels deep (id
        /// `p1/1/1` has parent `p1/1`). `None` for a call the model issued.
        /// A nested call's `call_id` is `<parent call id>/<n>`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_call_id: Option<CallId>,
    },

    /// A tool invocation returned.
    ToolCompleted {
        /// Identifier for the tool call that completed.
        call_id: CallId,
        /// Name of the tool that was invoked.
        tool_name: String,
        /// Wall-clock time the tool call *started*, in Unix-epoch
        /// milliseconds. Captured by the agent loop when it dispatches the
        /// call (alongside [`AgentEvent::ToolStarted`]) so exporters can
        /// render a real duration instead of a zero-width point. `None` for
        /// events serialized before this field existed (`#[serde(default)]`
        /// keeps old journals deserializable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        started_at_ms: Option<u64>,
        /// The arguments the tool was invoked with, captured only when
        /// [`PayloadCapture::tool_io`][crate::runtime::PayloadCapture::tool_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the call in a tool observation's Input panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        /// The tool result content, captured only when
        /// [`PayloadCapture::tool_io`][crate::runtime::PayloadCapture::tool_io]
        /// is enabled. `None` in the default payload-free mode. Populated so an
        /// exporter can render the result in a tool observation's Output panel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
        /// Wall-clock duration of the call in milliseconds (completion minus
        /// `started_at_ms`). Present regardless of payload capture, so an
        /// exporter renders a real duration without a side-channel. `None` for
        /// events serialized before this field existed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        /// Size, in bytes, of the serialized arguments the tool was invoked
        /// with. Present even in payload-free mode (unlike `input`), so a host
        /// can attribute argument cost without capturing the body. `None` for
        /// older events.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input_bytes: Option<u64>,
        /// Size, in bytes, of the tool's textual result content. Present even in
        /// payload-free mode (unlike `output`), so an exporter can show result
        /// size without capturing the body. `None` for older events.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_bytes: Option<u64>,
        /// Failure message when the tool call failed; `None` on success. Lets an
        /// exporter render success/failure and a reason from the journalled
        /// event itself rather than a live outcome side-channel.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// Host-only metadata the tool attached to its result
        /// (`tinytools::ToolResult::metadata`, B2). Carried here and on
        /// [`crate::middleware::AgentRun::tool_metadata`] for events,
        /// persistence, and telemetry; **never** rendered into the transcript
        /// the model sees. Present regardless of payload capture: it is the
        /// tool's deliberate host-facing channel, not captured I/O.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
        /// The call whose tool made this one through
        /// `ToolExecutionContext::call_tool` — the **immediate** parent, which
        /// is itself nested when this call is two or more levels deep (id
        /// `p1/1/1` has parent `p1/1`). `None` for a call the model issued.
        /// A nested call's `call_id` is `<parent call id>/<n>`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_call_id: Option<CallId>,
    },

    /// A tool invocation failed and the run is propagating the error rather
    /// than turning it into a tool result.
    ///
    /// This is the terminal partner of [`AgentEvent::ToolStarted`] on the error
    /// path. Without it, every `?` that escapes the tool-dispatch path leaves a
    /// `ToolStarted` with no matching terminal event, and any exporter pairing
    /// started/completed by `call_id` silently drops the failed call — the
    /// spans that matter most. The middleware stack already maintains this
    /// invariant deliberately (`run_stack_hook!` emits `MiddlewareCompleted`
    /// *before* inspecting the result, "the onion's balance invariant"); this
    /// variant extends the same guarantee to tools.
    ///
    /// Distinct from [`AgentEvent::ToolCompleted`] with `error: Some(_)`, which
    /// means the tool ran, failed, and the failure was fed back to the model as
    /// a tool result. `ToolFailed` means the run itself is aborting.
    ToolFailed {
        /// Identifier for the tool call that failed; pairs with the
        /// [`AgentEvent::ToolStarted`] of the same id.
        call_id: CallId,
        /// Name of the tool that was invoked.
        tool_name: String,
        /// Wall-clock time the tool call *started*, in Unix-epoch milliseconds,
        /// mirroring [`AgentEvent::ToolCompleted::started_at_ms`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        started_at_ms: Option<u64>,
        /// Wall-clock duration until the failure, in milliseconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        /// Human-readable failure description.
        error: String,
        /// The call whose tool made this one through
        /// `ToolExecutionContext::call_tool` — the **immediate** parent, which
        /// is itself nested when this call is two or more levels deep (id
        /// `p1/1/1` has parent `p1/1`). `None` for a call the model issued.
        /// A nested call's `call_id` is `<parent call id>/<n>`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_call_id: Option<CallId>,
    },

    /// A resumed run reconciled an unresolved tool-effect-ledger row left
    /// behind by an interrupted prior attempt (B5).
    ///
    /// Emitted by
    /// [`crate::runtime::AgentHarness::reconcile_tool_effects`] for each
    /// `started`-but-never-settled effect belonging to the last assistant
    /// tool-call turn, once it has decided what to do per the tool's
    /// [`tinytools::ToolReplay`] declaration.
    ToolEffectReconciled {
        /// Identifier of the reconciled tool call.
        call_id: CallId,
        /// What the reconciliation did: `"re_execute"` when the call was left
        /// pending for the loop to run again (`ToolReplay::Safe`), or
        /// `"interrupted"` when a synthesized tool-error result was appended
        /// instead (`ToolReplay::Never`).
        action: String,
    },

    /// A model call failed and the run is propagating the error.
    ///
    /// The terminal partner of [`AgentEvent::ModelStarted`] on the error path;
    /// see [`AgentEvent::ToolFailed`] for the rationale. Emitted once the retry
    /// ladder is exhausted — an *individual* failed attempt that will be retried
    /// is already covered by [`AgentEvent::RetryScheduled`].
    ModelFailed {
        /// Identifier for the model call that failed; pairs with the
        /// [`AgentEvent::ModelStarted`] of the same id.
        call_id: CallId,
        /// Registry name or provider model id the call was dispatched to.
        model: String,
        /// Wall-clock time the model call *started*, in Unix-epoch milliseconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        started_at_ms: Option<u64>,
        /// Number of attempts made before giving up (`1` when the call was not
        /// retried). Lets an exporter distinguish "failed once" from "failed
        /// after exhausting the retry policy".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempts: Option<usize>,
        /// Human-readable failure description.
        error: String,
    },

    /// A sub-agent child run failed.
    ///
    /// The terminal partner of [`AgentEvent::SubAgentStarted`] /
    /// [`AgentEvent::SubAgentReused`] on the error path; see
    /// [`AgentEvent::ToolFailed`] for the rationale. Without it a failing branch
    /// of the recursion tree simply stops appearing in the stream, and a
    /// consumer cannot tell a crashed child from one still running.
    SubAgentFailed {
        /// Name of the sub-agent whose run failed.
        name: String,
        /// Depth of the child run in the recursion tree.
        depth: usize,
        /// Human-readable failure description.
        error: String,
    },

    /// The model called a tool that is not registered, and the run's
    /// [`UnknownToolPolicy`][crate::runtime::UnknownToolPolicy]
    /// recovered from it instead of aborting.
    ///
    /// This is distinct from a tool that ran and returned an error: no tool was
    /// executed. The original requested name and arguments are preserved so the
    /// event stream can drive repair/analysis.
    UnknownToolCall {
        /// Identifier of the offending tool call.
        call_id: CallId,
        /// The tool name the model requested (which is not registered).
        requested_name: String,
        /// The raw arguments the model supplied for the call, preserved
        /// verbatim so repair middleware or analysis can re-target or replay
        /// the intended invocation.
        arguments: serde_json::Value,
        /// How the run recovered (for example `"tool_error"` or
        /// `"rewrite:other_tool"`).
        recovery: String,
    },

    /// The model called a *registered* tool with arguments that failed schema
    /// validation, and the run's
    /// [`InvalidArgsPolicy`][crate::runtime::InvalidArgsPolicy]
    /// recovered from it instead of aborting.
    ///
    /// Distinct from a tool that ran and returned an error: no tool was
    /// executed. The tool name, arguments, and validation error are preserved
    /// so the event stream can drive repair/analysis.
    InvalidToolArgs {
        /// Identifier of the offending tool call.
        call_id: CallId,
        /// The registered tool whose schema the supplied arguments violated.
        tool_name: String,
        /// The raw arguments the model supplied, preserved verbatim so repair
        /// middleware or analysis can re-target or replay the invocation.
        arguments: serde_json::Value,
        /// The schema-validation error detail.
        error: String,
        /// How the run recovered (currently always `"tool_error"`).
        recovery: String,
    },

    /// A per-agent isolated workspace/sandbox was prepared.
    WorkspacePrepared {
        /// Audit identity of the policy that produced the environment.
        policy_id: String,
        /// The allowed root, rendered as a string.
        root: String,
    },

    /// A tool attempted to touch a path outside its allowed workspace roots and
    /// was blocked.
    WorkspaceViolation {
        /// The offending path, rendered as a string.
        path: String,
    },

    /// A previously prepared isolated workspace/sandbox was cleaned up (or the
    /// cleanup failed, when `error` is set).
    WorkspaceCleanup {
        /// Audit identity of the policy whose environment was cleaned up.
        policy_id: String,
        /// Cleanup error, when cleanup failed.
        error: Option<String>,
    },

    /// The agent loop honored a
    /// [`MiddlewareControl`][crate::context::MiddlewareControl] request
    /// at a safe checkpoint (for example an early-exit stop or a pause). Recorded
    /// so control decisions are auditable and replayable from the journal.
    ControlApplied {
        /// Stable label of the control outcome (see
        /// [`MiddlewareControl::kind`][crate::context::MiddlewareControl::kind]).
        control: String,
        /// Human-readable detail (the final text, or the interrupt node/message).
        detail: String,
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
        /// The tool call this layer wraps, set only by the tool-wrap onion.
        /// Wrapped calls of one batch run concurrently, so their events
        /// interleave; pair a `Started` with its `Completed` and attribute
        /// both to a call by this id, never by event order.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<CallId>,
    },

    /// A middleware hook finished executing.
    MiddlewareCompleted {
        /// Registered name of the middleware.
        name: String,
        /// The tool call this layer wrapped; see
        /// [`AgentEvent::MiddlewareStarted::call_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<CallId>,
    },

    /// A response-cache lookup served the model call from the local
    /// [`crate::cache::ResponseCache`]; the provider was **not**
    /// invoked.
    CacheHit {
        /// Identifier for the model call this cache hit satisfies.
        call_id: CallId,
        /// The stable cache key (see [`crate::cache::cache_key`]).
        key: String,
    },

    /// A model response was served from a cache without an exposed cache key.
    ///
    /// This records the cache fact for adapters that do not own the harness
    /// response-cache key. The provider was not invoked for this response.
    /// Adapters must omit cached provider charges from [`AgentEvent::UsageRecorded`];
    /// this event has no call linkage for that run-level aggregate.
    ModelResponseCached {
        /// Identifier for the model call satisfied by the cached response.
        call_id: CallId,
    },

    /// A response-cache lookup missed, so the provider is being invoked and the
    /// result will be stored under `key`.
    CacheMiss {
        /// Identifier for the model call that triggered the lookup.
        call_id: CallId,
        /// The stable cache key (see [`crate::cache::cache_key`]).
        key: String,
    },

    /// A model call's provider prompt cache read back far less than the
    /// previous call of the same conversation had put in it, so the
    /// difference was re-billed as fresh input. Reported by
    /// [`PromptCacheGuardMiddleware`][crate::middleware::PromptCacheGuardMiddleware]
    /// from `usage.cache_read_tokens`; shortfalls within the noise floor
    /// (see [`crate::cache::DEFAULT_CACHE_MISS_NOISE_FLOOR_TOKENS`]) are not
    /// reported. Distinct from [`Self::CacheMiss`], which is a *response*-cache
    /// lookup miss.
    PromptCacheMiss {
        /// Identifier for the model call whose usage showed the miss.
        call_id: CallId,
        /// Prompt tokens that should have been cache reads.
        expected_cached_tokens: u64,
        /// Tokens the provider actually read from cache.
        cached_tokens: u64,
        /// `expected_cached_tokens - cached_tokens`, re-billed as input.
        wasted_input_tokens: u64,
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
    /// [`RateLimitMiddleware`][crate::middleware::RateLimitMiddleware]
    /// once per gated call, after the tokens were finally acquired.
    RateLimitWaited {
        /// Actual wall-clock time the call was held back, in milliseconds
        /// (measured with the middleware's injectable clock).
        waited_ms: u64,
    },

    /// A model fallback middleware swapped the request from one model to
    /// another after the primary failed. Emitted by
    /// [`ModelFallbackMiddleware`][crate::middleware::ModelFallbackMiddleware].
    FallbackSelected {
        /// The model that failed (the previous selection).
        from: String,
        /// The fallback model now being tried.
        to: String,
    },

    /// An explicit per-request model override was skipped during resolution —
    /// the requested model is unregistered, lacks a required capability, or is
    /// provider-retired — and resolution fell through to a lower-priority
    /// candidate (documented fail-closed behavior). Emitted by the agent loop
    /// so the silent fall-through is observable.
    ModelOverrideSkipped {
        /// The model name the request explicitly asked for.
        requested: String,
        /// The model that was actually resolved instead.
        resolved: String,
    },

    /// A runtime fallback candidate was skipped because it failed the request's
    /// capability/lifecycle gate — it lacks a required capability or is
    /// provider-retired — so the fallback chain advanced to the next candidate
    /// instead. Initial resolution gates the primary selection the same way;
    /// this makes the equivalent gate on the fallback path observable (issue
    /// #4641), so a primary failure can never silently fall back to a model that
    /// cannot satisfy the request.
    FallbackSkipped {
        /// The fallback model name that was skipped.
        model: String,
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

    /// Links a sub-agent lifecycle span to the child's exact run identity.
    /// Kept additive so existing consumers can continue matching the legacy
    /// name/depth lifecycle events without source changes.
    SubAgentRunStarted {
        /// Name of the sub-agent being invoked.
        name: String,
        /// Depth of the child run in the recursion tree.
        depth: usize,
        /// Exact identity used by observations emitted from the child run.
        run_id: RunId,
    },

    /// Marks the successful completion of the linked child run.
    SubAgentRunCompleted {
        /// Exact identity used by observations emitted from the child run.
        run_id: RunId,
    },

    /// An existing sub-agent was *reused* for a follow-up turn rather than
    /// reconstructed, carrying the prior conversation context forward.
    ///
    /// Emitted by the host session lifecycle on every continuation
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
    /// [`crate::steering::SteeringCommand`] so that orchestrator and
    /// human steering is fully observable in the event stream and never an
    /// untracked side channel.
    Steered {
        /// Stable name of the steered command kind (e.g. `"inject_message"`,
        /// `"cancel"`); see
        /// [`crate::steering::SteeringCommandKind::as_str`].
        command_kind: String,
        /// `true` when the run's policy permitted the command and it was
        /// applied; `false` when the policy rejected it.
        accepted: bool,
    },

    /// The transcript was compressed/summarized because it neared the model's
    /// context window. Emitted by
    /// [`ContextCompressionMiddleware`][crate::middleware::ContextCompressionMiddleware]
    /// only when it actually compresses; below-threshold requests pass through
    /// without emitting this event.
    Compressed {
        /// Estimated total tokens of the transcript before compression.
        from_tokens: u64,
        /// Estimated total tokens of the transcript after compression.
        to_tokens: u64,
    },

    /// A durable, rule-driven compaction ran and produced a
    /// [`crate::summarization::CompactionRecord`].
    ///
    /// Distinguished from [`Self::Compressed`] (the older, simpler
    /// event `ContextCompressionMiddleware`'s original `before_model` path
    /// emits) by carrying [`crate::summarization::CompactionReason`] and by
    /// always being emitted for a compaction produced through
    /// `crate::summarization::compaction` — including the
    /// overflow → compact → retry recovery path, which has no other event of
    /// its own. Both events fire for the same compaction on the `before_model`
    /// path; a listener that only cares about *whether* the transcript shrank
    /// can ignore `reason` and treat this exactly like `Compressed`.
    Compacted {
        /// Why this compaction ran.
        reason: crate::summarization::CompactionReason,
        /// Estimated total tokens of the transcript before compaction.
        tokens_before: u64,
        /// Estimated total tokens of the transcript after compaction.
        tokens_after: u64,
        /// Provider usage of the summarization call(s), when the summarizer
        /// made a model call and the provider reported it. The summarizer
        /// runs outside the run's own model calls, so this is the only place
        /// its spend reaches the event stream.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<tinyinference_llm::usage::Usage>,
        /// Wall-clock milliseconds the summarization took, when it ran.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        latency_ms: Option<u64>,
    },

    /// The final turn's structured-output extraction failed schema
    /// validation, or a registered
    /// [`crate::structured::OutputValidator`] rejected the value with
    /// [`crate::error::TinyAgentsError::ModelRetry`], and the loop is
    /// re-asking the model instead of failing the run (A3's
    /// output-validation retry loop; see
    /// [`crate::runtime::RunPolicy::output_retry`]).
    OutputRetry {
        /// The 1-based retry attempt this event reports (1 is the first
        /// re-ask after the original extraction failed).
        attempt: u8,
        /// The extraction/validation error handed back to the model as the
        /// repair prompt.
        error: String,
    },

    /// The agent loop appended one or more messages from a
    /// [`crate::run_queue::RunQueue`] lane to the working transcript at a
    /// safe turn boundary (A4): `Steer` after a tool batch or at a natural
    /// finish, `Followup` at a natural finish. Emitted once per boundary
    /// with the number of messages applied; `Collect` items never produce
    /// this event because they are not applied to the transcript. Payload
    /// text is carried only under the capture policy (see `messages`).
    QueuedMessageApplied {
        /// Which lane the messages came from.
        lane: crate::run_queue::QueueLane,
        /// How many messages were appended at this boundary (`1` under
        /// [`QueueMode::OneAtATime`][crate::run_queue::QueueMode::OneAtATime]).
        count: usize,
        /// Transcript index of the first applied message; the applied messages
        /// occupy `first_index..first_index + count`.
        #[serde(default)]
        first_index: usize,
        /// The applied messages, serialized, captured per message: `tool`
        /// messages when
        /// [`PayloadCapture::tool_io`][crate::runtime::PayloadCapture::tool_io]
        /// is enabled, all others when
        /// [`PayloadCapture::model_io`][crate::runtime::PayloadCapture::model_io]
        /// is. One slot per applied message (`null` for an uncaptured one) so
        /// slots line up with `first_index..first_index + count`; empty when
        /// nothing in the batch is captured, including the default payload-free
        /// mode.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        messages: Vec<serde_json::Value>,
    },

    /// A model turn began: the loop is about to dispatch the model call
    /// numbered `turn`. A turn is one model call plus the tool batch it
    /// requested; `turn` is 1-based and counts model-call attempts, so a
    /// recovery retry of an unusable reply opens a new turn. Paired with
    /// [`AgentEvent::TurnCompleted`].
    TurnStarted {
        /// 1-based turn number within the run.
        turn: u32,
    },

    /// A model turn ended: its tool batch (if any) has been folded into the
    /// transcript, or the turn produced the final answer, or the run ended
    /// mid-turn. Always follows a [`AgentEvent::TurnStarted`] with the same
    /// `turn`.
    TurnCompleted {
        /// 1-based turn number within the run.
        turn: u32,
        /// How many tool-result messages the turn added to the transcript.
        tool_result_count: usize,
        /// The call ids those tool results answer, in transcript order.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_call_ids: Vec<CallId>,
    },

    /// A message was appended to the run's working transcript (assistant
    /// reply, tool result, nudge, steering injection, queued message, ...).
    /// Emitted in transcript order at turn boundaries and run exit, so a
    /// consumer can mirror the transcript from events alone. The seed input
    /// messages are not announced.
    MessageAppended {
        /// Message role: `system`, `user`, `assistant`, `tool` or `custom`.
        role: String,
        /// Position of the message in the working transcript.
        index: usize,
        /// For `tool` messages, the call id the result answers.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<CallId>,
        /// The serialized message, captured only when the capture policy
        /// allows it (`model_io`, or `tool_io` for `tool` messages). `None` in
        /// the default payload-free mode.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<serde_json::Value>,
    },

    /// The message at `index` (and any announced after it) was removed from the
    /// working transcript — for example an unusable assistant reply dropped
    /// before a retry or a recovery nudge. Emitted highest index first, so
    /// applying them in order to a mirror is a sequence of pops. Always
    /// precedes the [`AgentEvent::MessageAppended`] of whatever replaces it.
    MessageRetracted {
        /// Position the removed message held in the working transcript.
        index: usize,
    },

    /// The working transcript was rewritten in place (not by appending or
    /// popping): a tool-set change folded into, or inserted before, the
    /// leading system message. The transcript now holds `len` messages; a
    /// mirror should treat its copy as stale and resynchronise. Later
    /// [`AgentEvent::MessageAppended`] indices count from this new length.
    TranscriptRewritten {
        /// Message count after the rewrite.
        len: usize,
        /// Why it was rewritten (a stable snake_case label).
        reason: String,
    },

    /// A graph routing decision produced a named route.
    RouteSelected {
        /// The route name chosen by the router.
        route: String,
    },

    /// Token usage for a completed model call was folded into the run totals.
    ///
    /// Emitted by the agent loop immediately after a model response that
    /// carried provider-reported usage, so usage-mode stream consumers and
    /// durable journals see per-call token counts without inspecting the
    /// [`AgentEvent::ModelCompleted`] payload.
    UsageRecorded {
        /// The usage reported for the model call just completed.
        usage: Usage,
    },

    /// Estimated cost for a completed model call was folded into the run
    /// totals.
    ///
    /// Defined for future emit: cost is computed once a pricing table is wired
    /// into the loop. Carries the cost delta attributed to the call.
    CostRecorded {
        /// The cost attributed to the model call just completed.
        cost: CostTotals,
    },

    /// A budget crossed its configured warning threshold but has not been
    /// exceeded, so the run continues.
    ///
    /// Emitted by
    /// [`BudgetMiddleware`][crate::middleware::BudgetMiddleware] after a
    /// spend pushes cumulative usage/cost past `warn_fraction` of a limit.
    BudgetWarning {
        /// Human-readable description of which budget threshold was crossed.
        reason: String,
    },

    /// Budget preflight reserved an estimate of the upcoming model call's input
    /// tokens against the run budget, before dispatching the call. Lets a budget
    /// bound a call *before* it overshoots, rather than only detecting the
    /// overshoot afterward.
    BudgetReserved {
        /// Estimated input tokens reserved for the upcoming call.
        estimated_input_tokens: u64,
    },

    /// Budget reconciled a prior reservation against the provider-reported usage
    /// after the call returned, so the difference between the estimate and the
    /// actual is auditable.
    BudgetReconciled {
        /// Tokens that had been reserved (estimated) for the call.
        estimated_input_tokens: u64,
        /// Input tokens the provider actually reported.
        actual_input_tokens: u64,
    },

    /// A budget limit was reached. When emitted from budget preflight the run is
    /// about to be blocked with
    /// [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded];
    /// when emitted post-spend it flags that the accumulated totals now exceed a
    /// limit.
    BudgetExceeded {
        /// Human-readable description of which budget limit was hit.
        reason: String,
        /// Whether this occurrence blocked a model call (preflight) rather than
        /// being detected after a spend.
        blocked: bool,
    },

    /// A configured run limit (cap) tripped and the run is about to fail.
    ///
    /// Emitted by the agent loop just before returning
    /// [`crate::error::TinyAgentsError::LimitExceeded`] /
    /// [`crate::error::TinyAgentsError::Timeout`] so observers can distinguish
    /// a limit-driven stop from other failures.
    LimitReached {
        /// Which cap was reached. Serialized as `limit_kind` to avoid colliding
        /// with the enum's `"kind"` serde tag.
        #[serde(rename = "limit_kind")]
        kind: LimitKind,
    },

    /// Conversation/working memory was loaded for the run.
    ///
    /// Defined for future emit when memory wiring lands; carries no payload so
    /// that loading remains observable without exposing memory contents.
    MemoryLoaded,

    /// Conversation/working memory was persisted for the run.
    ///
    /// Defined for future emit when memory wiring lands.
    MemorySaved,

    /// Legacy message-only progress shape. **The agent loop does not emit this
    /// variant**; it emits [`AgentEvent::ToolProgressDetail`]. It is kept so
    /// existing enum literals compile, and shares the `tool.progress` wire kind.
    ///
    /// The ordering and flooding guarantees below describe the emitted
    /// [`AgentEvent::ToolProgressDetail`]. Ordering guarantee: every progress event for a call falls between that call's
    /// [`AgentEvent::ToolStarted`] and its terminal
    /// [`AgentEvent::ToolCompleted`] / [`AgentEvent::ToolFailed`]; an update the
    /// tool reports after the call has returned is dropped, never emitted late.
    /// Calls in one concurrent batch interleave their progress freely. A tool
    /// that floods is coalesced (see `crate::tool::ToolProgressLimits`), so the
    /// stream is a faithful but possibly thinned view of what the tool reported.
    ToolProgress {
        /// Identifier for the in-flight tool call.
        call_id: CallId,
        /// Human-readable progress message.
        message: String,
    },

    /// A running tool reported incremental progress before completing, with
    /// optional fraction and partial output. This is the variant the agent loop
    /// emits (through [`tinytools::ToolRunContext::report_progress`]).
    ///
    /// The gate clamps `fraction` to `0.0..=1.0` (and drops NaN) before
    /// emitting. The legacy [`AgentEvent::ToolProgress`] shape stays unchanged
    /// so downstream enum literals continue to compile.
    ToolProgressDetail {
        /// Identifier for the in-flight tool call.
        call_id: CallId,
        /// Human-readable progress message; empty when the update carried only
        /// a fraction or partial output.
        #[serde(default)]
        message: String,
        /// Completion in `0.0..=1.0`, when the tool reported one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fraction: Option<f32>,
        /// Partial output reported so far, passed through verbatim.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partial: Option<serde_json::Value>,
    },

    /// An application-defined event a tool (or any holder of the run's
    /// [`EventSink`][crate::events::EventSink]) emitted through
    /// [`ToolExecutionContext::custom`][crate::tool::ToolExecutionContext::custom]
    /// (B1). The harness attaches no meaning to `payload`; it exists so a
    /// tool can report structured progress — a download percentage, an
    /// intermediate finding, a UI hint — on the same ordered stream as the
    /// loop's own events, without the harness growing a variant per use.
    Custom {
        /// The tool call the event was emitted from, when it came from a
        /// tool; `None` when emitted outside a call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<CallId>,
        /// Application-defined payload, passed through verbatim.
        payload: serde_json::Value,
    },

    /// A middleware hook reported a failure.
    ///
    /// Emitted by the lifecycle-hook driver ([`crate::middleware`]'s
    /// `run_stack_hook!` macro) immediately after
    /// [`AgentEvent::MiddlewareCompleted`] when a hook returns `Err`, so a
    /// failing middleware is observable alongside
    /// [`AgentEvent::MiddlewareStarted`] / [`AgentEvent::MiddlewareCompleted`]
    /// instead of only surfacing as the run's terminal error.
    MiddlewareFailed {
        /// Registered name of the middleware that failed.
        name: String,
        /// Human-readable error description.
        error: String,
    },

    /// A cross-provider handoff transform rewrote part of the outgoing
    /// transcript immediately before a model call, because it carried
    /// assistant content from a different provider/api/model than the one
    /// about to receive it (a mid-session model switch, an explicit
    /// per-request override, or a fallback to a different provider). Emitted
    /// only when at least one message changed — same-origin runs (the
    /// common case) never emit this.
    ///
    /// See the harness's cross-provider handoff transform for the exact
    /// rules (redacted/signed thinking, tool-call id normalization, image
    /// downgrade).
    HandoffTransformApplied {
        /// Number of messages rewritten by the transform for this call.
        changes: usize,
    },

    /// A streaming model call's chunk stream was closed (gracefully or by
    /// cancellation).
    ///
    /// Defined for future emit so stream consumers can detect end-of-stream
    /// without correlating against [`AgentEvent::ModelCompleted`].
    StreamClosed,

    /// A harness run finished: it returned a result to the caller. A run
    /// that stopped on a `StopWithPartial` cap also completes; its `outcome`
    /// says so (`LimitReached`).
    RunCompleted {
        /// Identifier for the run that completed.
        run_id: RunId,
        /// How the run ended, structured. `None` for events produced before
        /// this field existed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<crate::terminal::TerminalOutcome>,
    },

    /// A harness run ended with an unrecoverable error.
    RunFailed {
        /// Identifier for the run that failed.
        run_id: RunId,
        /// Human-readable error description.
        error: String,
        /// Why the run failed, structured. `outcome.message` mirrors `error`.
        /// `None` for events produced before this field existed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<crate::terminal::TerminalOutcome>,
    },

    /// The host durably committed the result of a run.
    ///
    /// This follows a terminal event and allows exporters to extend an
    /// already-closed run span to include host persistence, without changing
    /// its terminal status or inventing an outcome.
    RunFinalized {
        /// Identity of the run whose host-side persistence finished.
        run_id: RunId,
    },

    /// A completed turn was stopped early by a typed harness stop cause.
    ///
    /// This may be emitted after `RunCompleted`; the run identity and lineage
    /// come from the enclosing observation record. `turn` is the 1-based
    /// model-attempt ordinal within that run.
    TurnStopped {
        /// 1-based model-attempt ordinal for the stopped turn.
        turn: u32,
        /// Content-free reason that stopped the turn.
        stop: crate::turn_stop::TurnStop,
    },
}

/// Names the kind of run limit that tripped in an [`AgentEvent::LimitReached`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    /// The maximum number of model calls per run was reached.
    ModelCalls,
    /// The maximum number of tool calls per run was reached.
    ToolCalls,
    /// The run's wall-clock deadline elapsed.
    WallClock,
}

impl LimitKind {
    /// Returns a stable, snake_case string naming the limit kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            LimitKind::ModelCalls => "model_calls",
            LimitKind::ToolCalls => "tool_calls",
            LimitKind::WallClock => "wall_clock",
        }
    }
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
            AgentEvent::GenerationEnriched { .. } => "generation.enriched",
            AgentEvent::ControlApplied { .. } => "control.applied",
            AgentEvent::ToolsAdvertised { .. } => "tool.advertised",
            AgentEvent::ToolSearched { .. } => "tool.searched",
            AgentEvent::DeferredToolCall { .. } => "tool.deferred_call",
            AgentEvent::ToolDeferred { .. } => "tool.deferred",
            AgentEvent::ToolApproved { .. } => "tool.approved",
            AgentEvent::ToolDenied { .. } => "tool.denied",
            AgentEvent::ToolsFiltered { .. } => "tool.filtered",
            AgentEvent::ToolStarted { .. } => "tool.started",
            AgentEvent::ToolCompleted { .. } => "tool.completed",
            AgentEvent::ToolFailed { .. } => "tool.failed",
            AgentEvent::ToolEffectReconciled { .. } => "tool.effect_reconciled",
            AgentEvent::ModelFailed { .. } => "model.failed",
            AgentEvent::SubAgentFailed { .. } => "subagent.failed",
            AgentEvent::UnknownToolCall { .. } => "tool.unknown",
            AgentEvent::InvalidToolArgs { .. } => "tool.invalid_args",
            AgentEvent::BudgetWarning { .. } => "budget.warning",
            AgentEvent::BudgetReserved { .. } => "budget.reserved",
            AgentEvent::BudgetReconciled { .. } => "budget.reconciled",
            AgentEvent::BudgetExceeded { .. } => "budget.exceeded",
            AgentEvent::WorkspacePrepared { .. } => "workspace.prepared",
            AgentEvent::WorkspaceViolation { .. } => "workspace.violation",
            AgentEvent::WorkspaceCleanup { .. } => "workspace.cleanup",
            AgentEvent::StateUpdate => "state.update",
            AgentEvent::MiddlewareStarted { .. } => "middleware.started",
            AgentEvent::MiddlewareCompleted { .. } => "middleware.completed",
            AgentEvent::CacheHit { .. } => "cache.hit",
            AgentEvent::ModelResponseCached { .. } => "model.response_cached",
            AgentEvent::CacheMiss { .. } => "cache.miss",
            AgentEvent::PromptCacheMiss { .. } => "cache.prompt_miss",
            AgentEvent::RetryScheduled { .. } => "retry.scheduled",
            AgentEvent::RateLimitWaited { .. } => "rate_limit.waited",
            AgentEvent::FallbackSelected { .. } => "model.fallback_selected",
            AgentEvent::ModelOverrideSkipped { .. } => "model.override_skipped",
            AgentEvent::FallbackSkipped { .. } => "model.fallback_skipped",
            AgentEvent::SubAgentStarted { .. } => "subagent.started",
            AgentEvent::SubAgentCompleted { .. } => "subagent.completed",
            AgentEvent::SubAgentRunStarted { .. } => "subagent.run.started",
            AgentEvent::SubAgentRunCompleted { .. } => "subagent.run.completed",
            AgentEvent::SubAgentReused { .. } => "subagent.reused",
            AgentEvent::Steered { .. } => "agent.steered",
            AgentEvent::Compressed { .. } => "context.compressed",
            AgentEvent::Compacted { .. } => "context.compacted",
            AgentEvent::OutputRetry { .. } => "output.retry",
            AgentEvent::QueuedMessageApplied { .. } => "queue.applied",
            AgentEvent::TurnStarted { .. } => "turn.started",
            AgentEvent::TurnCompleted { .. } => "turn.completed",
            AgentEvent::MessageAppended { .. } => "message.appended",
            AgentEvent::MessageRetracted { .. } => "message.retracted",
            AgentEvent::TranscriptRewritten { .. } => "transcript.rewritten",
            AgentEvent::RouteSelected { .. } => "route.selected",
            AgentEvent::UsageRecorded { .. } => "usage.recorded",
            AgentEvent::CostRecorded { .. } => "cost.recorded",
            AgentEvent::LimitReached { .. } => "limit.reached",
            AgentEvent::MemoryLoaded => "memory.loaded",
            AgentEvent::MemorySaved => "memory.saved",
            AgentEvent::ToolProgress { .. } | AgentEvent::ToolProgressDetail { .. } => {
                "tool.progress"
            }
            AgentEvent::Custom { .. } => "custom",
            AgentEvent::MiddlewareFailed { .. } => "middleware.failed",
            AgentEvent::HandoffTransformApplied { .. } => "handoff.transform_applied",
            AgentEvent::StreamClosed => "stream.closed",
            AgentEvent::RunCompleted { .. } => "run.completed",
            AgentEvent::RunFailed { .. } => "run.failed",
            AgentEvent::RunFinalized { .. } => "run.finalized",
            AgentEvent::TurnStopped { .. } => "turn.stopped",
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
/// and enqueues the record under one critical section, then a single draining
/// emitter delivers queued records to listeners in offset order — so listeners
/// never observe offset `n + 1` before offset `n`, even under concurrent
/// emits.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use tinyagents_harness::events::{AgentEvent, EventSink, RecordingListener};
/// use tinyagents_harness::ids::RunId;
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

type ListenerSnapshot = Arc<Vec<Arc<dyn EventListener>>>;
type PendingEvent = (EventRecord, ListenerSnapshot);

/// Interior state shared among all clones of an [`EventSink`].
pub(crate) struct EventSinkInner {
    /// Stream-scoping prefix for emitted [`EventId`]s. Combined with the
    /// per-emit `offset` to form ids of the form `{stream_id}-evt-{offset}`,
    /// so ids stay unique across sinks and, when the prefix is a stable
    /// run/thread id, across process restarts (see [`EventSink::with_stream_id`]).
    pub(crate) stream_id: String,
    /// Next offset to assign; incremented atomically on each `emit`.
    pub(crate) next_offset: u64,
    /// Registered listeners, notified in insertion order. Stored behind an
    /// `Arc` so each emitted event takes a cheap immutable snapshot without
    /// allocating and cloning the full listener vector.
    pub(crate) listeners: ListenerSnapshot,
    /// Records assigned an offset but not yet delivered to listeners, in
    /// offset order. Each entry carries the listener snapshot taken when the
    /// offset was assigned so late subscribers never see earlier offsets.
    pub(crate) pending: std::collections::VecDeque<PendingEvent>,
    /// `true` while some emitter is draining `pending`. Guarantees a single
    /// drainer at a time, which is what makes listener delivery globally
    /// ordered by offset (and keeps re-entrant emits from listeners safe:
    /// they enqueue and return, and the active drainer delivers them).
    pub(crate) dispatching: bool,
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
///
/// Records are stored in offset order: the journal's buffer is populated by an
/// internal listener on the sink's ordered dispatch path, so
/// [`EventJournal::replay_from`] always returns a contiguous, offset-ordered
/// prefix of the stream — never the completion order of racing appends.
pub struct EventJournal {
    pub(crate) records: Arc<Mutex<Vec<EventRecord>>>,
    /// Internal sink used to assign monotonic ids and offsets.
    pub(crate) sink: EventSink,
}

/// Internal [`EventListener`] that copies each dispatched record into an
/// [`EventJournal`]'s buffer. Because sink dispatch is globally ordered by
/// offset, the buffer stays in offset order regardless of append concurrency.
pub(crate) struct JournalRecorder {
    pub(crate) records: Arc<Mutex<Vec<EventRecord>>>,
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
