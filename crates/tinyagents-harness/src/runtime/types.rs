//! Type definitions for the harness runtime facade.
//!
//! [`AgentHarness`] is the re-entrant runtime that the whole recursive
//! architecture stands inside: parent agents, nested sub-agents, and subgraph
//! nodes all execute against the same composed
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
//! `crate::runtime` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs`.

pub use crate::config::ToolDispatcher;
use std::collections::HashSet;
use std::sync::Arc;

use crate::cache::ResponseCache;
use crate::host::HostCapabilities;
use crate::limits::RunLimits;
use crate::middleware::MiddlewareStack;
use crate::model_registry::ModelRegistry;
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::run_queue::QueueMode;
use crate::tool::{ToolRegistry, ToolTimeoutSettings};
use tinyinference_llm::cache::CachePolicy;
use tinyinference_llm::model::ResponseFormat;

/// Model and identity selected by one live host-driven invocation.
///
/// This is held only in that invocation's non-serializable
/// [`RunContext`](crate::context::RunContext). It is never stored on the
/// reusable harness, keyed by a run id, or written to graph/checkpoint state.
pub(crate) struct HostInvocationBinding<State: Send + Sync, Ctx: Send + Sync> {
    /// The capability bundle that prepared this exact invocation. This is
    /// per-run rather than read from the harness so a recursively invoked
    /// child cannot substitute its own installed (or missing) host policy.
    pub(crate) host: Arc<HostCapabilities<State>>,
    pub(crate) agent_id: String,
    /// Definition-selected pin passed to the host resolver at every provider
    /// call. It is advisory; the host remains the routing authority.
    pub(crate) model_pin: Option<String>,
    pub(crate) role: Option<String>,
    /// Canonical names the resolved definition authorizes for this exact run.
    ///
    /// `None` means the definition declared no tools at all (an empty or
    /// absent list) — [`crate::agent_loop`]'s `resolve_tool_allowlist` treats
    /// that as fail-closed (deny every tool) by default, controlled by
    /// [`HostCapabilities::fail_closed_tool_allowlist`]. `Some(set)` is
    /// always the declared set, checked by plain membership: an empty
    /// `HashSet` is never stored here (a declared-but-empty list is
    /// collapsed to `None` at construction, so "nothing declared" and
    /// "declared empty" share one fail-closed code path instead of an empty
    /// set silently meaning "unrestricted", as it used to (I-9)).
    pub(crate) allowed_tools: Option<HashSet<String>>,
    /// Per-turn ordered, nonblocking projection to the optional progress sink.
    pub(crate) progress: Option<super::agent::ProgressSender>,
    /// The exact invocation-local runtime inherited by authorized children.
    pub(crate) runtime: Option<Arc<InvocationRuntime<State, Ctx>>>,
}

impl<State: Send + Sync, Ctx: Send + Sync> Clone for HostInvocationBinding<State, Ctx> {
    fn clone(&self) -> Self {
        Self {
            host: self.host.clone(),
            agent_id: self.agent_id.clone(),
            model_pin: self.model_pin.clone(),
            role: self.role.clone(),
            allowed_tools: self.allowed_tools.clone(),
            progress: self.progress.clone(),
            runtime: self.runtime.clone(),
        }
    }
}

/// How the agent loop reacts when the model calls a tool that is not
/// registered.
///
/// The default is [`UnknownToolPolicy::ReturnToolError`]: a hallucinated tool
/// name is a routine model mistake, not a harness fault, so the run keeps going
/// and the model gets told which tools actually exist. Each recovery still
/// consumes a tool-call budget slot, so [`RunLimits::max_tool_calls`] bounds any
/// unknown-tool loop.
///
/// # Why the default flipped
///
/// `Fail` used to be the default, which made the crate inconsistent with
/// itself: an **unparseable** arguments blob has always recovered
/// unconditionally (see `agent_loop/tools.rs`), so `{city:` survived while a
/// merely unknown tool name killed the whole run. It also diverged from
/// LangGraph, whose `ToolNode` answers an unknown name with a synthetic
/// `status="error"` message listing the valid tools. `Fail` remains available
/// for callers that genuinely want a hard stop.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum UnknownToolPolicy {
    /// Abort the run with
    /// [`TinyAgentsError::ToolNotFound`][crate::error::TinyAgentsError::ToolNotFound].
    Fail,
    /// Inject a tool-error result (naming the originally requested tool and
    /// listing the registered tools) back into the transcript and continue the
    /// loop, letting the model retry with a valid tool. The default.
    #[default]
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
/// The default is [`InvalidArgsPolicy::ReturnToolError`]: a missing `required`
/// field, wrong type, or bad `enum` is model output the model can fix, so the
/// validation detail plus the expected schema go back into the transcript
/// instead of aborting the turn. The recovery consumes a tool-call budget slot,
/// so [`RunLimits::max_tool_calls`] bounds any invalid-args loop. Mirrors
/// [`UnknownToolPolicy`] for the schema-validation seam — including why the
/// default flipped away from `Fail`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InvalidArgsPolicy {
    /// Abort the run with
    /// [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation].
    Fail,
    /// Inject a tool-error result (carrying the validation detail and the
    /// tool's expected parameter schema) back into the transcript and continue
    /// the loop, letting the model retry with corrected arguments. The default.
    #[default]
    ReturnToolError,
    /// First normalize common provider-shape defects, then apply
    /// [`Self::ReturnToolError`] if the resulting arguments still fail schema
    /// validation. Normalization decodes valid JSON emitted as a string
    /// (including a markdown-fenced string), preserving decoded values for
    /// precise validation, and coerces an undecodable/non-string value to `{}`
    /// only when an object-capable tool schema declares no required fields.
    NormalizeThenReturnToolError,
}

/// Controls whether the agent loop captures model and tool **payloads**
/// (prompt/completion text, tool arguments/results) onto the
/// [`AgentEvent::ModelCompleted`][crate::events::AgentEvent::ModelCompleted]
/// and [`AgentEvent::ToolCompleted`][crate::events::AgentEvent::ToolCompleted]
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
/// a [`RedactingSink`][crate::observability::RedactingSink] configured
/// with secret substrings still masks them before they reach a journal or
/// exporter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadCapture {
    /// Capture the request messages and the model completion onto every
    /// [`AgentEvent::ModelCompleted`][crate::events::AgentEvent::ModelCompleted].
    pub model_io: bool,
    /// Capture the tool arguments and the tool result onto every
    /// [`AgentEvent::ToolCompleted`][crate::events::AgentEvent::ToolCompleted].
    pub tool_io: bool,
}

impl PayloadCapture {
    /// A capture policy with both model and tool I/O enabled.
    ///
    /// Prefer this only when the observability pipeline is trusted (or a
    /// [`RedactingSink`][crate::observability::RedactingSink] is in
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
/// - `default_response_format`: when set, attached to every [`tinyinference_llm::model::ModelRequest`]
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
    /// [`tinyinference_llm::model::ModelRequest::cache_policy`] does not override
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
    /// How tools are spoken to the model: through the provider's native
    /// channel, or through one of the text protocols owned by
    /// `tinytools-agent`.
    ///
    /// [`ToolDispatcher::Auto`] (the default) sends tool schemas on the wire
    /// and lets the provider adapter decide — the OpenAI-compatible adapter
    /// switches to the JSON-in-tag protocol by itself for a profile without
    /// native tool calling. [`ToolDispatcher::Xml`], [`ToolDispatcher::Pformat`],
    /// [`ToolDispatcher::Python`] and [`ToolDispatcher::Typescript`] force a
    /// text protocol regardless of provider: the schemas are rendered into the
    /// system prompt, nothing goes on the wire as `tools`, and the answer is
    /// parsed here. P-Format is the cheapest on tokens and the most demanding
    /// on the model, and the code dialects sit between it and JSON, which is why
    /// it is opt-in only.
    ///
    /// Under a forced text dialect the answer is always read through every
    /// text grammar — parsing text *is* the protocol. Under a native dialect
    /// the same read is the fallback for a model that narrated a call as
    /// text, gated by [`RunPolicy::text_dialect_recovery`].
    pub tool_dialect: ToolDispatcher,
    /// Under a text dialect, whether the host already rendered the protocol
    /// block and the tool catalogue into its own system prompt.
    ///
    /// `false` (the default): the loop folds `tinytools-agent`'s protocol
    /// instructions and a catalogue of the advertised tools into the system
    /// prompt right before dispatch, so a host that only sets
    /// [`RunPolicy::tool_dialect`] gets a complete text-protocol prompt.
    /// `true`: the loop still strips the schemas off the wire and still binds
    /// the positional registry that parses calls back, but appends nothing —
    /// a host that composes its prompt from the same dialect (so the model
    /// sees one catalogue, in the place the host chose, inside the cacheable
    /// prefix) sets this to avoid shipping every signature twice.
    pub host_renders_tool_catalogue: bool,
    /// Maximum consecutive re-prompts when a model signals a tool call it did
    /// not make: `finish_reason == "tool_calls"` with no structured call and
    /// no text-recoverable one.
    ///
    /// Some routers rewrite finish reasons, and some models emit the
    /// intention without the call. Treating that as the final answer ends the
    /// turn on an empty promise; re-prompting once with "issue the actual
    /// tool call now" recovers it far more often than not. Each re-prompt is a
    /// model call and counts against `limits.max_model_calls`. Defaults to
    /// `3`; `0` disables it.
    pub dropped_tool_call_nudges: u32,
    /// Number of automatic retries when a model call returns a *truncated
    /// empty* completion — `finish_reason == "length"` with no visible text, no
    /// tool calls, and no structured output.
    ///
    /// This is the failure mode of local reasoning models (for example
    /// `qwen3` via Ollama) that intermittently spend the entire token budget on
    /// the hidden reasoning channel and emit nothing usable. Because such a
    /// response is useless to *every* caller and the failure is stochastic,
    /// retrying — with a doubled token budget when the request set one (capped
    /// at 4x the original), or a plain retry when it did not — is strictly
    /// better than surfacing a blank success.
    ///
    /// The retry runs *before* [`Self::error_on_empty_response`]; only once
    /// these retries are exhausted does that guard (if enabled) apply.
    ///
    /// Defaults to `1` (one retry, two attempts total). Set to `0` to disable
    /// for exact-replay callers that must not re-issue a call.
    pub truncated_empty_retries: u32,
    /// Automatic retries for a completion with no visible text, tool calls, or
    /// structured output when the provider did not report length truncation.
    /// Reasoning-only `stop` responses are one example: the model spent tokens
    /// but gave the caller nothing it can use. Each retry reissues the same
    /// request without increasing its output-token cap and counts against the
    /// run's model-call limit. The unusable assistant row is not retained.
    ///
    /// Defaults to `0` because some callers deliberately accept blank finals
    /// and because another provider call may be billable. Hosts that require a
    /// visible reply can opt in to one bounded retry.
    pub empty_response_retries: u32,
    /// How [`tinytools::ToolExposure::Deferred`] tools are surfaced: never in
    /// the request's `tools` array, but findable through the intrinsic
    /// `tool_search` / `tool_call` bridge. See
    /// [`crate::tool::discover::ToolDiscoveryPolicy`].
    pub discovery: crate::tool::discover::ToolDiscoveryPolicy,
    /// Optional projection applied to every advertised tool schema before it
    /// is sent (ref resolution, provider keyword stripping, byte budgets). See
    /// [`crate::tool::SchemaPreparation`].
    ///
    /// `None` (the default) sends declarations verbatim, as the loop always
    /// has. A host that registers third-party schemas — MCP servers, plugins —
    /// should set one; a host that authors every schema by hand rarely needs
    /// to. Admission still validates arguments against the *declared* schema,
    /// which is never looser than the projected one.
    pub tool_schemas: Option<crate::tool::SchemaPreparation>,
    /// Whether the loop parses `<tool_call>`-style text-dialect markup out of
    /// an assistant's visible text under a native tool dialect (see
    /// [`RunPolicy::tool_dialect`]). A forced text dialect
    /// ([`ToolDispatcher::Xml`] / [`ToolDispatcher::Pformat`] /
    /// [`ToolDispatcher::Python`] / [`ToolDispatcher::Typescript`], or
    /// [`ToolDispatcher::Auto`] falling back to Xml for a model without
    /// native tool calling) always parses the answer regardless of this
    /// policy, since the model can only answer in text.
    ///
    /// Defaults to [`TextDialectRecovery::Auto`], which only attempts
    /// recovery when the resolved model's
    /// [`ModelProfile::tool_calling`][tinyinference_llm::model::ModelProfile::tool_calling]
    /// is not reported (a model that *does* report native tool calling and
    /// still answered in prose was not making a tool call — it was
    /// explaining, quoting, or documenting the format, and executing that
    /// text as a real call would silently strip visible text the caller
    /// asked to see). See [`TextDialectRecovery`].
    pub text_dialect_recovery: TextDialectRecovery,
    /// Bounds the output-validation retry loop (A3): how many times the loop
    /// re-asks the model after the final turn's structured extraction fails
    /// schema validation, or a registered
    /// [`crate::structured::OutputValidator`] rejects an otherwise
    /// schema-valid value with
    /// [`crate::error::TinyAgentsError::ModelRetry`]. See
    /// [`OutputRetryPolicy`].
    pub output_retry: OutputRetryPolicy,
    /// What the loop does when one turn's tool calls include both a
    /// structured-output "schema" call ([`crate::structured::StructuredStrategy::ToolCall`]'s
    /// synthetic tool) and one or more genuine function-tool calls (A6).
    /// Defaults to [`EndStrategy::Graceful`].
    pub end_strategy: EndStrategy,
    /// Forces the [`crate::structured::StructuredStrategy::Prompted`] or
    /// [`crate::structured::StructuredStrategy::ToolCallUnion`] mode for a
    /// `ResponseFormat::Auto` structured-output request, bypassing
    /// [`crate::structured::StructuredStrategy::for_profile`]'s
    /// provider-capability heuristic (A6).
    ///
    /// `None` (the default) preserves the existing `Auto` resolution
    /// (`ProviderSchema` or `ToolCall`, chosen from the resolved model's
    /// profile). Only consulted for `ResponseFormat::Auto`; an explicit
    /// `ResponseFormat::JsonSchema` always uses provider-native mode
    /// regardless of this field.
    pub structured_strategy_override: Option<StructuredStrategyOverride>,
    /// How many queued messages the loop takes from a
    /// [`crate::run_queue::RunQueue`] lane at each safe boundary (A4):
    /// [`QueueMode::All`] (the default) applies every pending item at once,
    /// [`QueueMode::OneAtATime`] applies the oldest and leaves the rest for
    /// the next boundary. Only consulted when the run's
    /// [`RunContext`][crate::context::RunContext] carries a queue.
    pub queue_mode: QueueMode,
    /// Which engine [`AgentHarness::invoke`][super::AgentHarness::invoke] (and
    /// friends) drives the loop with (A5).
    ///
    /// Defaults to [`LoopExecution::Direct`]: the built-in
    /// [`crate::agent_loop`] body, unchanged. Setting
    /// [`LoopExecution::Graph`] selects an [`AgentHarness::loop_driver`
    /// ][super::AgentHarness::loop_driver] instead — install one with
    /// [`AgentHarness::with_loop_driver`][super::AgentHarness::with_loop_driver]
    /// (`tinyagents-graph`'s `GraphLoopDriver` is the intended implementor;
    /// see `tinyagents_graph::agent_loop`). Selecting `Graph` with no driver
    /// installed fails the run with
    /// [`crate::error::TinyAgentsError::Validation`] rather than silently
    /// falling back to `Direct`.
    pub execution: LoopExecution,
}

/// See [`RunPolicy::execution`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopExecution {
    /// Drive the run with the built-in [`crate::agent_loop`] body
    /// (`run_loop`/`run_loop_body`). The default; behavior-identical to
    /// every release before A5.
    #[default]
    Direct,
    /// Drive the run with the installed
    /// [`AgentHarness::loop_driver`][super::AgentHarness::loop_driver]
    /// instead (a compiled-graph rendition of the loop, in the common case).
    Graph,
}

/// See [`RunPolicy::structured_strategy_override`].
#[derive(Clone, Debug, PartialEq)]
pub enum StructuredStrategyOverride {
    /// Force [`crate::structured::StructuredStrategy::Prompted`]: inject the
    /// schema into the system prompt instead of using a provider schema API
    /// or a forced tool call.
    Prompted {
        /// Custom instructions template; `None` uses
        /// [`crate::structured::default_prompted_template`].
        template: Option<String>,
    },
    /// Force [`crate::structured::StructuredStrategy::ToolCallUnion`]: offer
    /// one synthetic tool per `(name, schema)` variant instead of the single
    /// schema from the `ResponseFormat`.
    ToolCallUnion {
        /// The union's variants, in the order their tools are advertised.
        variants: Vec<(String, serde_json::Value)>,
    },
}

/// Resolves the "output tool + function tools in one turn" ambiguity (A6),
/// mirroring Pydantic AI's `end_strategy`.
///
/// The ambiguity: the model can, in a single turn, both answer (via the
/// structured-output schema call) *and* ask to run further tools. Each
/// strategy answers "what happens to those tool calls, and does the run end
/// this turn?" differently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EndStrategy {
    /// Run the accompanying function-tool calls (so their side effects still
    /// happen and their results are not silently dropped), then finish the
    /// run with the structured output already recorded. The default: it
    /// never discards a tool call the model asked for, but also never spends
    /// an extra model call once the model has already answered.
    #[default]
    Graceful,
    /// Finish the run immediately on the first output-tool call. The
    /// accompanying function-tool calls are **not** executed; their
    /// `tool_calls` entries are closed with a synthetic "run stopped before
    /// this tool call was executed" result so the transcript stays
    /// replayable. Use when the structured answer must win even if it means
    /// dropping tool calls the model also happened to request.
    Early,
    /// Ignore the output-tool call this turn (do not record it, do not
    /// finish): run the function-tool calls and give the model another turn,
    /// exactly as if the output tool had not been called. The run only
    /// finishes once a turn produces the output tool with **no** accompanying
    /// function-tool calls. Use when function tools must always be allowed to
    /// run to completion before an answer is accepted.
    Exhaustive,
}

/// Policy for the output-validation retry loop (A3), mirroring Pydantic AI's
/// `retries={'output': N}`.
///
/// On the agent loop's final turn, a structured-extraction failure or a
/// registered [`crate::structured::OutputValidator`] rejection no longer
/// immediately fails the run: the error is pushed back to the model as a
/// repair prompt (built from [`Self::message_template`]) and the loop asks
/// again, up to [`Self::max_attempts`] times total for the run. Each retry
/// still counts against [`RunLimits::max_model_calls`] like any other model
/// call — this policy only bounds how many of those calls may be spent on
/// output repair specifically.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputRetryPolicy {
    /// How many times the loop may re-ask the model after an output
    /// validation failure. `0` disables the retry loop entirely — the first
    /// failure fails the run, exactly as before A3.
    pub max_attempts: u8,
    /// The repair-prompt template pushed to the model as a
    /// [`tinyinference_llm::message::Message::user`] turn. `{error}` is
    /// replaced with the extraction/validation error text; a template
    /// without that placeholder still works (the error is simply omitted)
    /// but loses the specific reason.
    pub message_template: String,
}

impl Default for OutputRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            message_template: "{error}\n\nFix the errors and try again.".to_string(),
        }
    }
}

/// Policy for recovering `<tool_call>`-style text-dialect tool calls from an
/// assistant's visible text.
///
/// Some providers/models emit tool calls as XML-ish markup inside ordinary
/// text instead of (or in addition to failing to populate) the provider's
/// native tool-call channel. Recovering that markup lets such a model still
/// drive tools through the same loop as a model with native tool calling.
///
/// Left unconditional, this is a real correctness hazard: any assistant text
/// that merely *quotes* `<tool_call>` markup — explaining the format to a
/// user, echoing a worked example, or showing it in a fenced code block —
/// gets executed as a real tool call, with the visible text silently
/// stripped and replaced. [`TextDialectRecovery::Auto`] (the default) closes
/// the common case of that hazard by skipping recovery for any model whose
/// resolved profile reports native tool calling; recovery inside fenced code
/// blocks is always skipped regardless of this policy, since a model
/// demonstrating the syntax in a code fence is manifestly not making a call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextDialectRecovery {
    /// Never parse text-dialect tool calls.
    Off,
    /// Always attempt recovery when the provider returned no native tool
    /// calls, regardless of the resolved model's advertised capabilities.
    On,
    /// Attempt recovery only when the resolved model's profile does not
    /// report native tool calling (or the profile is unknown). This is the
    /// default.
    #[default]
    Auto,
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
                ..CachePolicy::default()
            },
            // Opt-in: preserve the historical blank-final behavior by default.
            error_on_empty_response: false,
            tool_dialect: ToolDispatcher::Auto,
            host_renders_tool_catalogue: false,
            dropped_tool_call_nudges: 3,
            // On by default: a truncated-empty completion is useless to every
            // caller, so one stochastic-failure retry is strictly better than a
            // blank final.
            truncated_empty_retries: 1,
            empty_response_retries: 0,
            text_dialect_recovery: TextDialectRecovery::default(),
            discovery: crate::tool::discover::ToolDiscoveryPolicy::default(),
            tool_schemas: None,
            output_retry: OutputRetryPolicy::default(),
            end_strategy: EndStrategy::default(),
            structured_strategy_override: None,
            queue_mode: QueueMode::default(),
            execution: LoopExecution::default(),
        }
    }
}

/// High-level facade that composes model selection, tool execution, middleware,
/// and run policy behind one builder and runs the default agent loop.
///
/// `AgentHarness` is generic over the application `State` (shared, read-only
/// data threaded into every model and tool call) and the run-context data type
/// `Ctx` (defaults to `()`), which is moved into the [`crate::context::RunContext`]
/// for the duration of a run.
///
/// The registries, middleware stack, and policy are kept crate-private; build a
/// harness with [`AgentHarness::new`] and the `register_*` / `push_middleware` /
/// `with_policy` builder methods, and read them back through the accessor
/// methods. The agent loop itself is implemented in
/// [`crate::agent_loop`].
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use tinyinference_llm::providers::MockModel;
/// use tinyagents_harness::runtime::AgentHarness;
///
/// let mut harness: AgentHarness<()> = AgentHarness::new();
/// harness.register_model("mock", Arc::new(MockModel::constant("hello")));
/// assert_eq!(harness.models().default_name(), Some("mock"));
/// ```
pub struct AgentHarness<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// Name-keyed registry of chat models with an optional default.
    pub(crate) models: ModelRegistry<State>,
    /// Name-keyed registry of tools exposed to the model.
    pub(crate) tools: ToolRegistry<State, Ctx>,
    /// Ordered middleware stack wrapping agent, model, and tool execution.
    pub(crate) middleware: MiddlewareStack<State, Ctx>,
    /// Cross-cutting run policy (limits, retry, fallback, response format).
    pub(crate) policy: RunPolicy,
    /// Optional per-tool timeout resolver shared across harness runs.
    pub(crate) tool_timeouts: Option<ToolTimeoutSettings>,
    /// Optional local response cache shared across runs of this harness.
    ///
    /// When set, the agent loop consults it before each provider call (subject
    /// to the effective [`CachePolicy`]) and stores successful responses back
    /// into it. Because it is owned by the harness rather than a single run, a
    /// repeated identical request can be served from an earlier run's result.
    pub(crate) response_cache: Option<Arc<dyn ResponseCache>>,
    /// Optional validator consulted after the final turn's structured
    /// extraction succeeds, driving the output-validation retry loop (A3).
    /// See [`crate::structured::OutputValidator`] and
    /// [`AgentHarness::with_output_validator`].
    pub(crate) output_validator: Option<Arc<dyn crate::structured::OutputValidator<State, Ctx>>>,
    /// Optional inline resolver for deferred tool calls (A2). When set, a
    /// batch that defers calls is resolved through it and the loop keeps
    /// going instead of exiting with `AgentRun::deferred`. See
    /// [`crate::tool::DeferredToolHandler`] and
    /// [`AgentHarness::with_deferred_tool_handler`].
    pub(crate) deferred_tool_handler: Option<Arc<dyn crate::tool::DeferredToolHandler>>,
    /// Optional composable [`crate::tool::toolset::ToolSet`] chain
    /// (gap B3) consulted for the model-visible tool catalogue and, when a
    /// call is not owned by [`Self::tools`], for dispatch.
    ///
    /// `None` (the default) preserves every existing harness's behavior
    /// unchanged: the loop resolves tools from [`Self::tools`] alone, exactly
    /// as before this field existed. Set with
    /// [`AgentHarness::with_toolset`]. See that method's doc comment for
    /// exactly which turn behavior this changes.
    pub(crate) toolset: Option<Arc<dyn crate::tool::toolset::ToolSet<State, Ctx>>>,
    /// Capability bundles installed via [`AgentHarness::with_capability`]
    /// (gap G3), in installation order. Kept so each new `with_capability`
    /// call can rebuild [`Self::toolset`]'s
    /// [`crate::capability::CapabilityToolSet`] layer from the complete,
    /// still-accumulating list rather than nesting one per call.
    pub(crate) capabilities: Vec<crate::capability::Capability<State, Ctx>>,
    /// The toolset chain that was installed (via [`AgentHarness::with_toolset`],
    /// or `None`) before the first [`AgentHarness::with_capability`] call.
    /// Captured once so every later `with_capability` rebuild of
    /// [`Self::toolset`] keeps composing with it, instead of losing it to
    /// the first capability's rebuild.
    pub(crate) capability_base_toolset: Option<Arc<dyn crate::tool::toolset::ToolSet<State, Ctx>>>,
    /// Alternate loop engine selected when [`RunPolicy::execution`] is
    /// [`LoopExecution::Graph`] (A5). See
    /// [`crate::agent_loop::phases::LoopDriver`] and
    /// [`AgentHarness::with_loop_driver`].
    pub(crate) loop_driver: Option<Arc<dyn crate::agent_loop::phases::LoopDriver<State, Ctx>>>,
}

/// The non-serializable mechanics selected for one hosted invocation.
///
/// A durable [`AgentHarness`] remains the public entry point and retains only
/// process-safe dependencies. Hosts whose model, tool, or middleware surface
/// is selected per turn build an `InvocationRuntime` and attach it to the
/// [`AgentInvocation`](super::AgentInvocation). It is never installed on the
/// durable harness or serialized into graph/checkpoint state.
pub struct InvocationRuntime<State: Send + Sync, Ctx: Send + Sync = ()> {
    harness: Arc<AgentHarness<State, Ctx>>,
}

impl<State: Send + Sync, Ctx: Send + Sync> InvocationRuntime<State, Ctx> {
    /// Freezes an already assembled, invocation-local runtime.
    pub fn new(harness: AgentHarness<State, Ctx>) -> Self {
        Self {
            harness: Arc::new(harness),
        }
    }

    pub(crate) fn harness(&self) -> &AgentHarness<State, Ctx> {
        &self.harness
    }
}
