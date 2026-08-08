//! Harness model layer types.
//!
//! These provider-neutral shapes are the common currency of the recursive
//! harness: the same [`ModelRequest`] / [`ModelResponse`] / [`ModelStream`]
//! types describe a call whether it originates from a top-level agent, a nested
//! sub-agent, or a graph node, so model-calls-model recursion is expressed in
//! one uniform vocabulary regardless of depth or provider.
//!
//! These are the rich, harness-internal request/response shapes. They carry
//! tool declarations, tool-choice policy, structured-output formats, capability
//! profiles ([`ModelProfile`]/[`CapabilitySet`]), model-resolution inputs, and
//! prompt-cache layout metadata.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::cache::CachePolicy;
use crate::harness::message::{AssistantMessage, Message, MessageDelta};
use crate::harness::tool::{ToolDelta, ToolSchema};
use crate::harness::usage::Usage;

/// Policy controlling whether and how the model may call tools.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    #[default]
    Auto,
    /// The model must not call any tool.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call the named tool.
    Tool(String),
}

/// The requested output format for a model response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Free-form text.
    Text,
    /// Any JSON object.
    JsonObject,
    /// JSON constrained to a named schema.
    JsonSchema {
        /// Schema name advertised to the provider.
        name: String,
        /// JSON Schema document.
        schema: Value,
    },
    /// Request structured JSON output but let the harness pick the extraction
    /// strategy from the resolved model's [`ModelProfile`].
    ///
    /// The harness resolves this into either provider-native schema mode
    /// ([`StructuredStrategy::ProviderSchema`]) when the model advertises
    /// native structured output, or a tool-call fallback
    /// ([`StructuredStrategy::ToolCall`]) otherwise. When no profile is
    /// available it falls back to provider-native schema mode.
    ///
    /// [`StructuredStrategy::ProviderSchema`]: crate::harness::structured::StructuredStrategy::ProviderSchema
    /// [`StructuredStrategy::ToolCall`]: crate::harness::structured::StructuredStrategy::ToolCall
    Auto {
        /// Schema name advertised to the provider or used as the fallback tool
        /// name.
        name: String,
        /// JSON Schema document describing the desired structure.
        schema: Value,
    },
}

/// How much effort a reasoning model should spend before answering.
///
/// Provider-neutral by design. There was no reasoning knob at all before: the
/// only route was raw [`ModelRequest::provider_options`], which is
/// provider-shaped by definition — an OpenAI `reasoning_effort` string, an
/// OpenAI Responses `reasoning: {effort, summary}` object, and an Anthropic
/// `thinking: {type, budget_tokens}` object are three incompatible spellings of
/// one idea, and a caller that hardcodes one cannot switch providers. Worse, on
/// the OpenAI **Responses** path `provider_options` was dropped from the wire
/// body entirely, so the one format that supports `reasoning` could not receive
/// it.
///
/// Anthropic's surface is the argument for an enum over a raw dict: it accepts
/// both a `thinking` object and a `reasoning_effort` literal, has a documented
/// precedence chain between them, and rejects `budget_tokens` outright on newer
/// models. A neutral enum lets each adapter lower to whatever its provider
/// currently accepts without every caller tracking that churn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// The smallest amount of reasoning the provider offers.
    Minimal,
    /// Below-default effort.
    Low,
    /// The provider's default effort.
    #[default]
    Medium,
    /// Above-default effort; slower and more expensive.
    High,
    /// Reasoning explicitly disabled.
    ///
    /// Distinct from leaving [`ReasoningConfig::effort`] as `None`, which means
    /// "no opinion — use the provider default". On OpenAI this additionally
    /// lifts the gpt-5 restriction that pins `temperature` when reasoning is on.
    None,
}

impl ReasoningEffort {
    /// The wire token OpenAI's `reasoning_effort` / `reasoning.effort` fields
    /// expect.
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::None => "none",
        }
    }
}

/// Provider-neutral reasoning/thinking configuration for one call.
///
/// Adapters lower this to whatever their provider accepts:
///
/// | Target | Lowered to |
/// |---|---|
/// | OpenAI Chat Completions | `reasoning_effort: "<effort>"` |
/// | OpenAI Responses | `reasoning: { effort, summary }` |
/// | Anthropic-shaped gateways | `thinking: { type, budget_tokens }` |
///
/// [`ModelRequest::provider_options`] remains the escape hatch and **wins on
/// key conflicts**, so a caller who needs a shape this struct cannot express is
/// never blocked by it.
///
/// Require it of a resolved model with
/// [`CapabilitySet::reasoning_effort`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// How hard to think. `None` leaves the provider default alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// An explicit thinking-token budget, for providers that take one
    /// (Anthropic's `thinking.budget_tokens`). Ignored by providers that only
    /// accept a coarse effort level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Requested reasoning-summary verbosity (`"auto"`, `"concise"`,
    /// `"detailed"`), for the OpenAI Responses `reasoning.summary` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl ReasoningConfig {
    /// A config requesting the given effort and nothing else.
    pub fn effort(effort: ReasoningEffort) -> Self {
        Self {
            effort: Some(effort),
            ..Self::default()
        }
    }

    /// Whether this config asks for anything at all.
    pub fn is_empty(&self) -> bool {
        self.effort.is_none() && self.budget_tokens.is_none() && self.summary.is_none()
    }
}

/// Lifecycle status of a model, used by [`ModelProfile`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Generally available and supported.
    #[default]
    Stable,
    /// Preview/beta; behavior may change.
    Preview,
    /// Slated for removal; callers should migrate.
    Deprecated,
    /// No longer served by the provider.
    Retired,
}

/// The input/output modalities a model supports.
///
/// [`Default`] enables text in and text out only; all media modalities are
/// disabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modalities {
    /// Accepts text input.
    pub text_in: bool,
    /// Produces text output.
    pub text_out: bool,
    /// Accepts image input (vision).
    pub image_in: bool,
    /// Produces image output.
    pub image_out: bool,
    /// Accepts audio input.
    pub audio_in: bool,
    /// Produces audio output.
    pub audio_out: bool,
}

impl Default for Modalities {
    fn default() -> Self {
        Self {
            text_in: true,
            text_out: true,
            image_in: false,
            image_out: false,
            audio_in: false,
            audio_out: false,
        }
    }
}

/// A capability profile describing what a model can do.
///
/// Profiles let the harness reject impossible requests before a network call,
/// choose native structured output versus tool-based structured output, decide
/// whether tool-call chunks can stream, and select fallbacks that satisfy
/// required capabilities. Profiles are not a pricing table; prices live in the
/// cost feature.
///
/// [`Default`] is conservative: text-only modalities and every optional
/// capability disabled. Providers should override with what they actually
/// support.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Provider family identifier (for example `openai`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Provider model id this profile describes, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Human-readable display name, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Lifecycle status.
    #[serde(default)]
    pub status: ModelStatus,
    /// Release date in ISO-8601 (`YYYY-MM-DD`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,
    /// Supported input/output modalities.
    #[serde(default)]
    pub modalities: Modalities,
    /// Supports tool/function calling.
    #[serde(default)]
    pub tool_calling: bool,
    /// Supports multiple tool calls in a single response.
    #[serde(default)]
    pub parallel_tool_calls: bool,
    /// Supports streaming responses.
    #[serde(default)]
    pub streaming: bool,
    /// Streams tool-call fragments incrementally (versus reconstructing them
    /// from a final response).
    #[serde(default)]
    pub streaming_tool_chunks: bool,
    /// Supports provider-native structured output (constrained JSON).
    #[serde(default)]
    pub native_structured_output: bool,
    /// Honors a JSON Schema in the response-format request.
    #[serde(default)]
    pub json_schema: bool,
    /// Emits reasoning/thinking output.
    #[serde(default)]
    pub reasoning: bool,
    /// Accepts a **configurable** reasoning effort
    /// ([`ReasoningConfig`]), not merely emitting reasoning output.
    ///
    /// Separate from [`Self::reasoning`] because the two really do come apart:
    /// a distilled deepseek-r1 on Ollama emits `<think>` blocks (so `reasoning`
    /// is true) while accepting no effort knob at all, and a caller that needs
    /// to *dial* reasoning must be able to require the knob.
    #[serde(default)]
    pub reasoning_effort: bool,
    /// Maximum input (context) tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    /// Maximum output tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
}

/// A set of required capabilities used to validate a request against a
/// [`ModelProfile`] and to filter candidate models during resolution.
///
/// Every boolean field is a *requirement*: `true` means the capability must be
/// present, `false` means "don't care". The token fields require a minimum
/// advertised capacity. [`Default`] requires nothing, so it is satisfied by any
/// profile.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    /// Requires tool/function calling.
    #[serde(default)]
    pub tool_calling: bool,
    /// Requires parallel tool calls.
    #[serde(default)]
    pub parallel_tool_calls: bool,
    /// Requires streaming responses.
    #[serde(default)]
    pub streaming: bool,
    /// Requires incremental tool-call streaming.
    #[serde(default)]
    pub streaming_tool_chunks: bool,
    /// Requires provider-native structured output.
    #[serde(default)]
    pub native_structured_output: bool,
    /// Requires JSON Schema support.
    #[serde(default)]
    pub json_schema: bool,
    /// Requires reasoning output.
    #[serde(default)]
    pub reasoning: bool,
    /// Requires a configurable reasoning effort ([`ReasoningConfig`]).
    #[serde(default)]
    pub reasoning_effort: bool,
    /// Requires image input (vision).
    #[serde(default)]
    pub image_in: bool,
    /// Requires image output.
    #[serde(default)]
    pub image_out: bool,
    /// Requires audio input.
    #[serde(default)]
    pub audio_in: bool,
    /// Requires audio output.
    #[serde(default)]
    pub audio_out: bool,
    /// Requires at least this many input (context) tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_input_tokens: Option<u64>,
    /// Requires at least this many output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_output_tokens: Option<u64>,
}

/// The role a [`PromptSegment`] plays in the assembled prompt. Earlier roles
/// form the stable, cacheable prefix; [`SegmentRole::Volatile`] marks the tail
/// that changes every turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentRole {
    /// System prompt.
    System,
    /// Tool declarations.
    Tools,
    /// Stable instructions.
    Instructions,
    /// Conversation history.
    History,
    /// Volatile, per-turn content that must stay out of stable prefixes.
    Volatile,
}

/// A labeled segment of the prompt used to reason about provider prompt/KV
/// cache stability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptSegment {
    /// Stable identifier for the segment.
    pub id: String,
    /// The role the segment plays in the prompt.
    pub role: SegmentRole,
    /// Whether this segment is part of the cacheable stable prefix.
    pub cacheable: bool,
}

/// A model candidate supplied by an agent, request, state, or orchestrator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelHint {
    /// Registry alias or provider model id to try.
    pub model: String,
    /// Higher priority hints are tried first.
    #[serde(default)]
    pub priority: i32,
    /// Optional human-readable reason for observability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Where the final model choice came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelResolutionSource {
    /// Explicit request-level model override.
    RequestOverride,
    /// Reused from prior run or state.
    StateReuse,
    /// Chosen from request/agent/orchestrator hints.
    Hint,
    /// Default declared by the agent configuration.
    AgentDefault,
    /// Registry-level default.
    RegistryDefault,
}

/// Durable record of the model selected for a call or run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModel {
    /// Registry name used to obtain the executable model.
    pub name: String,
    /// User/agent requested value, when different from the final name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested: Option<String>,
    /// Source that selected this model.
    pub source: ModelResolutionSource,
}

/// Input policy for resolving a model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSelection {
    /// Explicit model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested: Option<String>,
    /// Optional previous model to reuse from state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<ResolvedModel>,
    /// Whether previous state should be considered before hints/defaults.
    #[serde(default)]
    pub reuse_previous: bool,
    /// Ordered model hints. Higher priority wins; ties preserve insertion order.
    #[serde(default)]
    pub hints: Vec<ModelHint>,
    /// Agent-level default model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_default: Option<String>,
    /// Capabilities the selected model must satisfy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_capabilities: Option<CapabilitySet>,
    /// When `false` (the default), resolution skips models whose profile
    /// reports [`ModelStatus::Retired`][crate::harness::model::ModelStatus],
    /// so a provider-retired model is never selected via override, reuse,
    /// hint, or default. Set `true` to opt back into retired models (e.g. for
    /// replaying historical runs). Deprecated models are still selectable.
    #[serde(default)]
    pub allow_retired: bool,
}

/// A provider-neutral chat model request.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Tool declarations exposed for this call.
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    /// Tool-choice policy.
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Requested response format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Model id or registry alias override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Ordered model resolution hints.
    #[serde(default)]
    pub model_hints: Vec<ModelHint>,
    /// Reuse the model recorded in state when available.
    #[serde(default)]
    pub reuse_previous_model: bool,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling probability mass, when supported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Stop sequences that should terminate generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Deterministic generation seed, when supported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Per-call timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Free-form request metadata.
    #[serde(default)]
    pub metadata: Value,
    /// Tags propagated to events and traces.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Declared prompt cache segments for KV-cache stability.
    #[serde(default)]
    pub cache_segments: Vec<PromptSegment>,
    /// Fingerprint of the stable prompt prefix, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_fingerprint: Option<String>,
    /// Capabilities the resolved model must satisfy. Used to validate the
    /// request before a provider call and to filter resolution candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_capabilities: Option<CapabilitySet>,
    /// Provider-specific options passed through untouched (for example OpenAI
    /// Responses API knobs, Anthropic thinking config, Ollama local `options`,
    /// or provider-specific controls such as `hotness`). Defaults to JSON null.
    #[serde(default)]
    pub provider_options: Value,
    /// Optional caching policy for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_policy: Option<CachePolicy>,
    /// Optional provider continuation/response id for stateful follow-ups.
    ///
    /// Read by the OpenAI Responses adapter, which sends it as
    /// `previous_response_id`. It previously had **no reader anywhere in the
    /// crate** — only a builder — so a caller could set it and nothing would
    /// happen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation_id: Option<String>,
    /// Provider-neutral reasoning/thinking configuration. See
    /// [`ReasoningConfig`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

/// A provider-neutral chat model response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The assistant message produced by the model.
    pub message: AssistantMessage,
    /// Token usage, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Provider finish reason (for example `stop`, `tool_calls`, `length`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Raw provider metadata preserved for callers who need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    /// Model selected by the harness/registry for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_model: Option<ResolvedModel>,
    /// When set, this response does **not** end the turn.
    ///
    /// A response carrying tool calls always continues the turn — the loop runs
    /// them and asks for another reply. A response *without* tool calls is
    /// otherwise the turn's final answer. This field lets a model adapter say
    /// "not done" anyway: the loop appends the carried string as the next user
    /// turn and requests another reply.
    ///
    /// `None` (the default) is the historical behaviour, so every existing
    /// caller is unaffected.
    ///
    /// **Why it exists.** Text-mode protocols encode tool calls as prose, and a
    /// model that wants to say something *before* acting has no tool call to
    /// ride. Without this the loop ends its turn on the first sentence, so such
    /// a protocol cannot narrate — it must either stay silent until the work is
    /// done, or fake a tool call purely to keep the floor.
    ///
    /// The carried string is the nudge to hand back (e.g. `"(next)"`). It is a
    /// string rather than a `bool` because most chat APIs reject two assistant
    /// turns in a row, so *something* must be appended; letting the adapter
    /// choose keeps the protocol's vocabulary out of the harness.
    ///
    /// A continuing turn stays bounded by
    /// [`RunLimits::max_model_calls`](crate::harness::runtime::RunLimits) — it
    /// costs one model call per continue, like any other loop iteration — so
    /// this needs no cap of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continue_turn: Option<String>,
 served_from_cache: false,
    /// `true` when this response was served from a local
    /// [`ResponseCache`][crate::harness::cache::ResponseCache] rather than
    /// produced by a provider call.
    ///
    /// # Why accounting needs this
    ///
    /// A cached response retains the `usage` the provider originally reported.
    /// Replaying it verbatim re-bills those tokens on every hit: the run's
    /// usage totals inflate and a cost-budget middleware prices spend that
    /// never happened, which can abort a run over money nobody paid. LangChain
    /// zeroes usage on the cache-hit path for exactly this reason. This flag
    /// lets the accounting sites tell a replay from a real call without
    /// destroying the `usage` a caller may legitimately want to inspect.
    ///
    /// `#[serde(default)]` plus a skip-when-false so entries written before the
    /// flag existed still deserialize, and so a serialized response is
    /// byte-identical to the historical shape when it is a real call.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub served_from_cache: bool,
}

/// An incremental streamed chunk of a model response.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelDelta {
    /// Id of the model call this delta belongs to.
    pub call_id: String,
    /// Incremental text content.
    #[serde(default)]
    pub content: String,
    /// Incremental reasoning/thinking content, kept separate from visible text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
    /// Incremental tool-call fragment, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolDelta>,
}

/// Normalized provider failure details.
///
/// Provider adapters use this shape for HTTP failures, stream error events, and
/// terminal stream failures so callers can reason about errors without parsing
/// provider-specific JSON. The original provider payload can still be retained
/// in [`ProviderError::raw`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderError {
    /// Provider family identifier, for example `openai` or `ollama`.
    pub provider: String,
    /// Provider model id, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Transport status code, when the failure came from HTTP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Provider error code or type, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: String,
    /// Whether retrying the same request may succeed.
    #[serde(default)]
    pub retryable: bool,
 retry_after_ms: None,
    /// Server-supplied wait before retrying, in milliseconds, parsed from the
    /// HTTP `Retry-After` response header.
    ///
    /// A `429`/`503` that names how long the client must wait is authoritative:
    /// retrying sooner burns an attempt for certain.
    /// [`retry_after_hint`][crate::harness::retry::retry_after_hint] reads this
    /// field **first**, falling back to parsing the error message text only
    /// when it is `None` — until this field existed, a provider that sent the
    /// header but did not echo it into the JSON body was simply not honored.
    ///
    /// Both header forms are normalised here: delta-seconds (`Retry-After: 30`)
    /// and the HTTP-date form (`Retry-After: Wed, 21 Oct 2015 07:28:00 GMT`),
    /// the latter converted to a delay relative to receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    /// Raw provider payload, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

/// A single item produced by a real, asynchronous model stream.
///
/// A well-behaved stream begins with [`ModelStreamItem::Started`], emits zero or
/// more [`ModelStreamItem::MessageDelta`] / [`ModelStreamItem::ToolCallDelta`] /
/// [`ModelStreamItem::UsageDelta`] items as the provider produces output, and
/// terminates with exactly one of [`ModelStreamItem::Completed`] (carrying the
/// fully merged response) or [`ModelStreamItem::Failed`] (carrying an error
/// message). Providers that can build an authoritative final response — such as
/// the OpenAI adapter — emit it via [`ModelStreamItem::Completed`] so the
/// merged response preserves tool-call names and ids that individual deltas may
/// omit.
///
/// Use [`StreamAccumulator`] (or the [`collect_model_stream`] helper) to fold a
/// stream of these items back into a [`ModelResponse`].
/// # Serialization
///
/// This enum is **adjacently tagged** (`{"type": …, "content": …}`) rather than
/// internally tagged. Several variants wrap a non-struct payload
/// ([`ModelStreamItem::Failed`] wraps a `String`); an internally tagged enum
/// cannot serialize those (serde errors, or silently corrupts scalar JSON into
/// `{}`), so adjacent tagging is required for every variant to round-trip.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum ModelStreamItem {
    /// The stream has opened; no content has arrived yet.
    Started,
    /// An incremental message fragment (text and/or a tool-call fragment).
    MessageDelta(MessageDelta),
    /// An incremental tool-call argument fragment correlated by call id.
    ToolCallDelta(ToolDelta),
    /// A usage update. Providers may send cumulative usage; the accumulator
    /// keeps the most recent value.
    UsageDelta(Usage),
    /// Terminal success: the fully merged response.
    Completed(ModelResponse),
    /// Terminal failure with a human-readable error message.
    Failed(String),
    /// Terminal failure with normalized provider details.
    ProviderFailed(ProviderError),
}

/// A pinned, boxed, `Send` stream of [`ModelStreamItem`]s.
///
/// This is the return type of [`ChatModel::stream`]. It is runtime-agnostic: the
/// caller's executor drives it, and dropping it cancels consumption without any
/// dependency on a specific async runtime.
pub type ModelStream = Pin<Box<dyn Stream<Item = ModelStreamItem> + Send>>;

/// A provider-neutral chat model.
///
/// Generic over the application `State`.
#[async_trait]
pub trait ChatModel<State: Send + Sync>: Send + Sync {
    /// Returns the model's capability [`ModelProfile`], when known.
    ///
    /// The default returns `None`; providers that know their capabilities
    /// should override this. The harness consults the profile to choose a
    /// structured-output strategy for [`ResponseFormat::Auto`] and to validate
    /// [`ModelRequest::required_capabilities`].
    fn profile(&self) -> Option<&ModelProfile> {
        None
    }

    /// Returns a stable string identifying *which* model, at *which* endpoint,
    /// under *which* credential this handle talks to — folded into the response
    /// cache key by
    /// [`scoped_cache_key`][crate::harness::cache::scoped_cache_key].
    ///
    /// # Why the request is not enough
    ///
    /// [`ModelRequest::model`] is an optional *hint* that the agent loop does
    /// not even set on the requests it builds; the real model is chosen by
    /// [`ModelRegistry::resolve_request`] afterwards, and the endpoint and
    /// credential live inside this trait object and never appear in the request
    /// at all. A cache keyed on the request alone therefore has no provider or
    /// model identity in it, and one shared cache serves a hosted harness's
    /// answer to a local one. LangChain avoids this by looking up on
    /// `(prompt, llm_string)`, where `llm_string` serializes the whole model
    /// object.
    ///
    /// # Contract
    ///
    /// * Return `None` (the default) to decline; the key then folds a fixed
    ///   `anonymous-model` marker, which still separates identifying models
    ///   from each other but cannot separate two anonymous ones.
    /// * The value must be **stable across process restarts** — a cache that is
    ///   only valid within one process lifetime is not a cache.
    /// * The value must **never contain a raw credential**. It is folded into
    ///   keys that end up in logs, events, and durable cache files. Use
    ///   [`credential_fingerprint`][crate::harness::cache::credential_fingerprint]
    ///   (or the whole-identity helper
    ///   [`model_cache_identity`][crate::harness::cache::model_cache_identity]).
    fn cache_identity(&self) -> Option<String> {
        None
    }

    /// Invokes the model and returns a complete response.
    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse>;

    /// Streams the model response as a real asynchronous [`ModelStream`].
    ///
    /// The default implementation calls [`ChatModel::invoke`] and replays the
    /// complete response as three items: [`ModelStreamItem::Started`], a single
    /// [`ModelStreamItem::MessageDelta`] carrying the full text, and a terminal
    /// [`ModelStreamItem::Completed`] carrying the response. Providers that talk
    /// to a streaming endpoint (for example the OpenAI adapter) override this to
    /// emit incremental deltas as bytes arrive.
    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        let response = self.invoke(state, request).await?;
        let delta = MessageDelta {
            text: response.text(),
            reasoning: String::new(),
            tool_call: None,
        };
        let items = vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(delta),
            ModelStreamItem::Completed(response),
        ];
        Ok(Box::pin(futures::stream::iter(items)))
    }
}

/// A name-keyed registry of chat models with an optional default selection.
pub struct ModelRegistry<State: Send + Sync> {
    pub(crate) models: HashMap<String, Arc<dyn ChatModel<State>>>,
    pub(crate) default: Option<String>,
}

/// Executable model plus durable resolution metadata.
pub struct ResolvedModelBinding<State: Send + Sync> {
    /// Durable selected-model record.
    pub resolved: ResolvedModel,
    /// Executable model handle.
    pub model: Arc<dyn ChatModel<State>>,
}

use crate::Result;
