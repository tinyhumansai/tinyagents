//! Harness model layer.
//!
//! The model layer is the innermost rung of the recursive ladder: every level
//! of the RLM-style harness — a top-level agent, a sub-agent exposed as a tool,
//! or a node inside a subgraph — ultimately bottoms out in a [`ChatModel`] call
//! routed through this provider-neutral request/response shape. Because the
//! shapes are uniform, "a model calling a model" is the same typed surface at
//! every depth, with the [`ModelRegistry`] resolving *which* model answers each
//! nested call.
//!
//! See [`types`] for definitions. This module provides builder methods on
//! [`ModelRequest`], accessors on [`ModelResponse`], the [`ModelRegistry`]
//! resolution logic, and the [`StreamAccumulator`] that folds a real
//! [`ModelStream`] back into a single [`ModelResponse`].

mod types;

use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;

use crate::Result;
use crate::harness::message::{AssistantMessage, ContentBlock, Message};
use crate::harness::tool::{ToolCall, ToolSchema};
use crate::harness::usage::Usage;

pub use types::*;

/// How a context-window pattern is matched against a model id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextPatternMatch {
    /// Pattern may appear anywhere in the lowercased model id.
    Substring,
    /// Pattern must be a complete segment delimited by common provider/id
    /// separators. This avoids false positives for short model ids such as
    /// `o1` and `o3`.
    Segment,
}

/// Generic context-window hints for common provider model families.
///
/// These are deliberately provider-neutral fallbacks, not a pricing catalog.
/// Hosts should prefer authoritative provider/catalog metadata when available
/// and use this only when a raw model id needs a conservative pre-dispatch
/// budget.
///
/// Order matters: lookup returns the first matching entry, so more-specific
/// substrings such as `gpt-4.1` and `gpt-4-turbo` must stay before broader
/// patterns such as `gpt-4` that would otherwise shadow them.
const MODEL_CONTEXT_PATTERNS: &[(&str, ContextPatternMatch, u64)] = &[
    ("claude-haiku-4.5", ContextPatternMatch::Substring, 200_000),
    ("claude-haiku-4", ContextPatternMatch::Substring, 200_000),
    ("claude-haiku", ContextPatternMatch::Substring, 200_000),
    ("claude-sonnet-4", ContextPatternMatch::Substring, 200_000),
    ("claude-opus-4", ContextPatternMatch::Substring, 200_000),
    ("claude-3-5-sonnet", ContextPatternMatch::Substring, 200_000),
    ("claude-3-5-haiku", ContextPatternMatch::Substring, 200_000),
    ("claude-3-opus", ContextPatternMatch::Substring, 200_000),
    ("gpt-4.1", ContextPatternMatch::Substring, 1_047_576),
    ("gpt-4o", ContextPatternMatch::Substring, 128_000),
    ("gpt-4-turbo", ContextPatternMatch::Substring, 128_000),
    ("gpt-4", ContextPatternMatch::Substring, 128_000),
    ("gpt-3.5", ContextPatternMatch::Substring, 16_385),
    ("o1", ContextPatternMatch::Segment, 200_000),
    ("o3", ContextPatternMatch::Segment, 200_000),
    ("deepseek", ContextPatternMatch::Substring, 128_000),
    ("gemma3", ContextPatternMatch::Substring, 8_192),
    ("gemma", ContextPatternMatch::Substring, 8_192),
    ("llama-3", ContextPatternMatch::Substring, 128_000),
    ("llama3", ContextPatternMatch::Substring, 128_000),
];

fn matches_context_pattern(lower: &str, pattern: &str, mode: ContextPatternMatch) -> bool {
    match mode {
        ContextPatternMatch::Substring => lower.contains(pattern),
        ContextPatternMatch::Segment => {
            let model_name = lower.rsplit(['/', ':']).next().unwrap_or(lower);
            model_name
                .split(['-', '_', '.'])
                .next()
                .is_some_and(|segment| segment == pattern)
        }
    }
}

/// Returns a generic context-window hint for a raw provider model id.
///
/// Returns `None` for unknown ids rather than guessing. Hosts with product tier
/// aliases, local runtime profiles, or authoritative provider catalogs should
/// check those first and use this helper as a last generic fallback.
pub fn context_window_for_model_id(model: &str) -> Option<u64> {
    let normalized = model.trim();
    if normalized.is_empty() {
        return None;
    }

    let lower = normalized.to_ascii_lowercase();
    MODEL_CONTEXT_PATTERNS
        .iter()
        .find_map(|(pattern, mode, window)| {
            matches_context_pattern(&lower, pattern, *mode).then_some(*window)
        })
}

impl std::fmt::Display for ProviderError {
    /// Renders the same human-readable shape real provider adapters used to
    /// build by hand before flattening it into a plain
    /// [`crate::error::TinyAgentsError::Model`] string. Preserving this as a
    /// `Display` impl lets [`crate::error::TinyAgentsError::Provider`] keep
    /// the identical wording while also keeping the structured fields
    /// (`status`, `code`, `retryable`) intact for callers — like
    /// [`crate::harness::retry::is_retryable`] — that need to reason about
    /// the failure rather than just print it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} returned{}{}: {}",
            self.provider,
            self.status
                .map(|status| format!(" HTTP {status}"))
                .unwrap_or_default(),
            self.code
                .as_deref()
                .map(|code| format!(" ({code})"))
                .unwrap_or_default(),
            self.message
        )
    }
}

impl ResponseFormat {
    /// Constructs a [`ResponseFormat::JsonSchema`] format.
    pub fn json_schema(name: impl Into<String>, schema: Value) -> Self {
        ResponseFormat::JsonSchema {
            name: name.into(),
            schema,
        }
    }

    /// Constructs a [`ResponseFormat::Auto`] format that lets the harness pick a
    /// structured-output strategy from the resolved model profile.
    pub fn auto(name: impl Into<String>, schema: Value) -> Self {
        ResponseFormat::Auto {
            name: name.into(),
            schema,
        }
    }
}

impl ModelProfile {
    /// Returns `true` when this profile satisfies every requirement in `set`.
    ///
    /// Boolean requirements must be matched by an equal-or-stronger capability
    /// on the profile. A token requirement is satisfied only when the profile
    /// advertises a known capacity at least as large; an unknown
    /// (`None`) capacity fails a token requirement, to stay conservative.
    pub fn satisfies(&self, set: &CapabilitySet) -> bool {
        let bool_ok = (!set.tool_calling || self.tool_calling)
            && (!set.parallel_tool_calls || self.parallel_tool_calls)
            && (!set.streaming || self.streaming)
            && (!set.streaming_tool_chunks || self.streaming_tool_chunks)
            && (!set.native_structured_output || self.native_structured_output)
            && (!set.json_schema || self.json_schema)
            && (!set.reasoning || self.reasoning)
            && (!set.image_in || self.modalities.image_in)
            && (!set.image_out || self.modalities.image_out)
            && (!set.audio_in || self.modalities.audio_in)
            && (!set.audio_out || self.modalities.audio_out);
        if !bool_ok {
            return false;
        }
        if let Some(min) = set.min_input_tokens
            && self.max_input_tokens.is_none_or(|cap| cap < min)
        {
            return false;
        }
        if let Some(min) = set.min_output_tokens
            && self.max_output_tokens.is_none_or(|cap| cap < min)
        {
            return false;
        }
        true
    }

    /// Returns `true` when the model is usable for new calls: any status except
    /// [`ModelStatus::Retired`].
    ///
    /// Resolution and fallback can use this to reject models a provider no
    /// longer serves without a bespoke lookup table.
    pub fn is_usable(&self) -> bool {
        self.status != ModelStatus::Retired
    }

    /// Returns `true` when the model is deprecated (slated for removal) or
    /// already retired — callers should prefer a successor.
    pub fn is_deprecated(&self) -> bool {
        matches!(self.status, ModelStatus::Deprecated | ModelStatus::Retired)
    }

    /// Builds a runtime [`ModelProfile`] from an offline
    /// [`ModelCatalogEntry`][crate::registry::catalog::ModelCatalogEntry].
    ///
    /// This is the bridge between the two capability models: the checked-in
    /// catalog (facts: pricing, context windows, deprecation, capability flags)
    /// and the runtime profile used by resolution/fallback. A model that carries
    /// no hardcoded profile can be hydrated from the catalog so capability
    /// gating and lifecycle checks still apply. A published `deprecation_date`
    /// maps to [`ModelStatus::Deprecated`] (retirement is not inferred from a
    /// date, to avoid depending on the wall clock).
    pub fn from_catalog_entry(entry: &crate::registry::catalog::ModelCatalogEntry) -> Self {
        let caps = &entry.capabilities;
        Self {
            provider: Some(entry.provider.clone()),
            model: Some(entry.model_id.clone()),
            display_name: None,
            status: if entry.deprecation_date.is_some() {
                ModelStatus::Deprecated
            } else {
                ModelStatus::Stable
            },
            release_date: None,
            modalities: Modalities {
                text_in: true,
                text_out: true,
                image_in: caps.vision,
                image_out: false,
                audio_in: caps.audio_input,
                audio_out: caps.audio_output,
            },
            tool_calling: caps.tool_calling,
            parallel_tool_calls: caps.parallel_tool_calling,
            streaming: caps.streaming,
            streaming_tool_chunks: caps.streaming && caps.tool_calling,
            native_structured_output: caps.json_schema,
            json_schema: caps.json_schema,
            reasoning: caps.reasoning,
            max_input_tokens: entry.max_input_tokens,
            max_output_tokens: entry.max_output_tokens,
        }
    }

    /// A permissive profile that advertises every capability and broad
    /// modalities. Useful for mocks and tests.
    pub fn permissive() -> Self {
        Self {
            modalities: Modalities {
                text_in: true,
                text_out: true,
                image_in: true,
                image_out: true,
                audio_in: true,
                audio_out: true,
            },
            tool_calling: true,
            parallel_tool_calls: true,
            streaming: true,
            streaming_tool_chunks: true,
            native_structured_output: true,
            json_schema: true,
            reasoning: true,
            ..Self::default()
        }
    }
}

impl ModelRequest {
    /// Creates a request from a list of messages.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }

    /// Sets the tool declarations exposed for this call.
    pub fn with_tools(mut self, tools: Vec<ToolSchema>) -> Self {
        self.tools = tools;
        self
    }

    /// Sets the tool-choice policy.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Sets the response format.
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Sets the model id or registry alias.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Adds a model resolution hint.
    pub fn with_model_hint(mut self, hint: ModelHint) -> Self {
        self.model_hints.push(hint);
        self
    }

    /// Enables or disables previous model reuse from run/agent state.
    pub fn with_reuse_previous_model(mut self, reuse: bool) -> Self {
        self.reuse_previous_model = reuse;
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets nucleus sampling probability mass.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the maximum output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets stop sequences that should terminate generation.
    pub fn with_stop_sequences(
        mut self,
        stop_sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.stop_sequences = stop_sequences.into_iter().map(Into::into).collect();
        self
    }

    /// Sets a deterministic generation seed.
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the per-call timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Adds a tag to the request.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets the declared prompt cache segments.
    pub fn with_cache_segments(mut self, segments: Vec<PromptSegment>) -> Self {
        self.cache_segments = segments;
        self
    }

    /// Sets the capabilities the resolved model must satisfy.
    pub fn with_required_capabilities(mut self, capabilities: CapabilitySet) -> Self {
        self.required_capabilities = Some(capabilities);
        self
    }

    /// Sets the provider-specific pass-through options.
    pub fn with_provider_options(mut self, options: Value) -> Self {
        self.provider_options = options;
        self
    }

    /// Adds one provider-specific pass-through option.
    ///
    /// If `provider_options` is not already an object, it is replaced with a new
    /// object containing this option.
    pub fn with_provider_option(mut self, key: impl Into<String>, value: Value) -> Self {
        let options = self
            .provider_options
            .as_object_mut()
            .map(std::mem::take)
            .unwrap_or_default();
        let mut options = options;
        options.insert(key.into(), value);
        self.provider_options = Value::Object(options);
        self
    }

    /// Sets the caching policy for this call.
    pub fn with_cache_policy(mut self, policy: crate::harness::cache::CachePolicy) -> Self {
        self.cache_policy = Some(policy);
        self
    }

    /// Sets the provider continuation id for stateful follow-ups.
    pub fn with_continuation_id(mut self, id: impl Into<String>) -> Self {
        self.continuation_id = Some(id.into());
        self
    }

    /// Returns the ids of cacheable segments in declaration order, describing
    /// the stable prompt prefix middleware should preserve.
    pub fn cacheable_prefix_ids(&self) -> Vec<String> {
        self.cache_segments
            .iter()
            .filter(|s| s.cacheable)
            .map(|s| s.id.clone())
            .collect()
    }
}

impl ModelResponse {
    /// Creates a response wrapping a plain assistant text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            message: AssistantMessage {
                id: None,
                content: vec![ContentBlock::Text(content.into())],
                tool_calls: Vec::new(),
                usage: None,
            },
            usage: None,
            finish_reason: None,
            raw: None,
            resolved_model: None,
        }
    }

    /// Attaches usage to the response (and mirrors it onto the message).
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.message.usage = Some(usage);
        self.usage = Some(usage);
        self
    }

    /// Sets the provider finish reason.
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Attaches model resolution metadata.
    pub fn with_resolved_model(mut self, resolved: ResolvedModel) -> Self {
        self.resolved_model = Some(resolved);
        self
    }

    /// Returns the tool calls requested by the model, if any.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.message.tool_calls
    }

    /// Returns the concatenated text of the assistant message.
    pub fn text(&self) -> String {
        Message::Assistant(self.message.clone()).text()
    }
}

impl<State: Send + Sync> ModelRegistry<State> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
            default: None,
        }
    }

    /// Registers a model under `name`. The first registered model becomes the
    /// default unless one is already set.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
    ) -> &mut Self {
        let name = name.into();
        if self.default.is_none() {
            self.default = Some(name.clone());
        }
        self.models.insert(name, model);
        self
    }

    /// Sets the default model name.
    pub fn set_default(&mut self, name: impl Into<String>) -> &mut Self {
        self.default = Some(name.into());
        self
    }

    /// Looks up a model by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn ChatModel<State>>> {
        self.models.get(name).cloned()
    }

    /// Returns the default model, if one is configured.
    pub fn default_model(&self) -> Option<Arc<dyn ChatModel<State>>> {
        self.default.as_deref().and_then(|name| self.get(name))
    }

    /// Returns the configured default model name.
    pub fn default_name(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// Resolves a model using request override, previous state, hints,
    /// agent default, and finally registry default.
    pub fn resolve(&self, selection: ModelSelection) -> Option<ResolvedModelBinding<State>> {
        let required = selection.required_capabilities.as_ref();
        let allow_retired = selection.allow_retired;
        if let Some(requested) = selection.requested
            && let Some(model) = self.get(&requested)
            && model_eligible(model.as_ref(), required, allow_retired)
        {
            return Some(ResolvedModelBinding {
                resolved: ResolvedModel {
                    name: requested.clone(),
                    requested: Some(requested),
                    source: ModelResolutionSource::RequestOverride,
                },
                model,
            });
        }

        if selection.reuse_previous
            && let Some(previous) = selection.previous
            && let Some(model) = self.get(&previous.name)
            && model_eligible(model.as_ref(), required, allow_retired)
        {
            return Some(ResolvedModelBinding {
                resolved: ResolvedModel {
                    name: previous.name,
                    requested: previous.requested,
                    source: ModelResolutionSource::StateReuse,
                },
                model,
            });
        }

        let mut hints: Vec<(usize, ModelHint)> = selection.hints.into_iter().enumerate().collect();
        hints.sort_by(|(left_index, left), (right_index, right)| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left_index.cmp(right_index))
        });

        for (_, hint) in hints {
            if let Some(model) = self.get(&hint.model)
                && model_eligible(model.as_ref(), required, allow_retired)
            {
                return Some(ResolvedModelBinding {
                    resolved: ResolvedModel {
                        name: hint.model.clone(),
                        requested: Some(hint.model),
                        source: ModelResolutionSource::Hint,
                    },
                    model,
                });
            }
        }

        if let Some(agent_default) = selection.agent_default
            && let Some(model) = self.get(&agent_default)
            && model_eligible(model.as_ref(), required, allow_retired)
        {
            return Some(ResolvedModelBinding {
                resolved: ResolvedModel {
                    name: agent_default.clone(),
                    requested: Some(agent_default),
                    source: ModelResolutionSource::AgentDefault,
                },
                model,
            });
        }

        let name = self.default_name()?.to_string();
        self.default_model()
            .filter(|model| model_eligible(model.as_ref(), required, allow_retired))
            .map(|model| ResolvedModelBinding {
                resolved: ResolvedModel {
                    name,
                    requested: None,
                    source: ModelResolutionSource::RegistryDefault,
                },
                model,
            })
    }

    /// Resolves a model for one request with optional agent and previous-state
    /// context.
    pub fn resolve_request(
        &self,
        request: &ModelRequest,
        agent_default: Option<&str>,
        previous: Option<ResolvedModel>,
    ) -> Option<ResolvedModelBinding<State>> {
        self.resolve(ModelSelection {
            requested: request.model.clone(),
            previous,
            reuse_previous: request.reuse_previous_model,
            hints: request.model_hints.clone(),
            agent_default: agent_default.map(ToOwned::to_owned),
            required_capabilities: request.required_capabilities.clone(),
            allow_retired: false,
        })
    }

    /// Returns registered model names in sorted order.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.models.keys().cloned().collect();
        names.sort();
        names
    }
}

fn model_satisfies<State: Send + Sync>(
    model: &dyn ChatModel<State>,
    required: Option<&CapabilitySet>,
) -> bool {
    match required {
        None => true,
        Some(required) if required == &CapabilitySet::default() => true,
        Some(required) => model
            .profile()
            .is_some_and(|profile| profile.satisfies(required)),
    }
}

/// A model is eligible for resolution when it satisfies the required
/// capabilities *and* is not lifecycle-excluded. Unless `allow_retired` is set,
/// a model whose profile reports [`ModelStatus::Retired`] is rejected so a
/// provider-retired model is never selected. A model with no profile carries no
/// lifecycle facts, so it is treated as usable (consistent with capability
/// gating, which only rejects a model when a profile is present and fails).
fn model_eligible<State: Send + Sync>(
    model: &dyn ChatModel<State>,
    required: Option<&CapabilitySet>,
    allow_retired: bool,
) -> bool {
    if !model_satisfies(model, required) {
        return false;
    }
    if allow_retired {
        return true;
    }
    model.profile().is_none_or(ModelProfile::is_usable)
}

impl<State: Send + Sync> Default for ModelRegistry<State> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamAccumulator
// ---------------------------------------------------------------------------

/// Deterministically folds a sequence of [`ModelStreamItem`]s into a final
/// [`ModelResponse`].
///
/// The accumulator implements the chunk-merge rules from the streaming spec:
///
/// - text fragments are concatenated in arrival order;
/// - tool-call argument fragments are correlated by call id and concatenated,
///   preserving first-seen order;
/// - usage updates overwrite the running value (providers commonly report
///   cumulative usage, so the last value wins);
/// - a terminal [`ModelStreamItem::Completed`] is treated as authoritative: its
///   response is returned as-is (only back-filling usage when absent), because
///   providers build it with full tool-call names and ids that individual
///   deltas may not carry;
/// - a terminal [`ModelStreamItem::Failed`] or
///   [`ModelStreamItem::ProviderFailed`] turns [`StreamAccumulator::finish`]
///   into an error.
///
/// When no `Completed` item is seen, [`StreamAccumulator::finish`] reconstructs
/// a best-effort response from the accumulated text, tool-call fragments, and
/// usage.
#[derive(Clone, Debug, Default)]
pub struct StreamAccumulator {
    /// Concatenated text fragments.
    text: String,
    /// Accumulated reasoning/thinking fragments (side channel; not merged into
    /// the final message text).
    reasoning: String,
    /// Per-call-id accumulated tool-call argument fragments, in first-seen
    /// order: `(call_id, arguments, tool_name)`. The name is the first non-empty
    /// `ToolDelta::tool_name` seen for the call (providers surface it on the
    /// call-opening delta); `None` until one arrives.
    tool_chunks: Vec<(String, String, Option<String>)>,
    /// Most recent usage value seen.
    usage: Option<Usage>,
    /// Authoritative final response, when a `Completed` item was seen.
    completed: Option<ModelResponse>,
    /// Terminal error message, when an unstructured `Failed` item was seen.
    failed: Option<String>,
    /// Terminal structured provider failure, when a `ProviderFailed` item was
    /// seen. Kept as the full struct (not stringified) so `finish()` can return
    /// [`crate::error::TinyAgentsError::Provider`] and preserve the
    /// status/code/`retryable` classification the retry layer needs.
    failed_provider: Option<ProviderError>,
}

impl StreamAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one stream item into the running state.
    pub fn push(&mut self, item: &ModelStreamItem) {
        match item {
            ModelStreamItem::Started => {}
            ModelStreamItem::MessageDelta(delta) => {
                self.text.push_str(&delta.text);
                self.reasoning.push_str(&delta.reasoning);
                if let Some(tool_call) = &delta.tool_call {
                    self.push_tool_chunk(
                        &tool_call.call_id,
                        &tool_call.content,
                        tool_call.tool_name.as_deref(),
                    );
                }
            }
            ModelStreamItem::ToolCallDelta(delta) => {
                self.push_tool_chunk(&delta.call_id, &delta.content, delta.tool_name.as_deref());
            }
            ModelStreamItem::UsageDelta(usage) => {
                self.usage = Some(*usage);
            }
            ModelStreamItem::Completed(response) => {
                self.completed = Some(response.clone());
            }
            ModelStreamItem::Failed(message) => {
                self.failed = Some(message.clone());
            }
            ModelStreamItem::ProviderFailed(error) => {
                // Retain the structured error rather than stringifying it, so
                // `finish()` can surface `TinyAgentsError::Provider` and the retry
                // layer sees the real status/code/`retryable` (a permanent 401 /
                // `insufficient_quota` / 400 must not be retried as transient).
                self.failed_provider = Some(error.clone());
            }
        }
    }

    /// Appends a tool-call argument fragment for `call_id`, preserving
    /// first-seen ordering across calls. Records the first non-empty `tool_name`
    /// seen for the call (call-opening deltas carry it; argument fragments do
    /// not) so the reconstructed [`ToolCall`] is named.
    fn push_tool_chunk(&mut self, call_id: &str, content: &str, tool_name: Option<&str>) {
        if let Some(entry) = self.tool_chunks.iter_mut().find(|(id, ..)| id == call_id) {
            entry.1.push_str(content);
            if entry.2.is_none()
                && let Some(name) = tool_name.filter(|n| !n.is_empty())
            {
                entry.2 = Some(name.to_string());
            }
        } else {
            self.tool_chunks.push((
                call_id.to_string(),
                content.to_string(),
                tool_name.filter(|n| !n.is_empty()).map(str::to_string),
            ));
        }
    }

    /// Returns `true` when a terminal item (`Completed`, `Failed`, or
    /// `ProviderFailed`) has been folded in.
    pub fn is_terminal(&self) -> bool {
        self.completed.is_some() || self.failed.is_some() || self.failed_provider.is_some()
    }

    /// Returns the accumulated reasoning/thinking text streamed so far (the
    /// side channel kept out of the final message text).
    pub fn reasoning(&self) -> &str {
        &self.reasoning
    }

    /// Consumes the accumulator and returns the merged response.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::TinyAgentsError::Provider`] when a
    /// [`ModelStreamItem::ProviderFailed`] item was folded in — preserving the
    /// structured status/code/`retryable` so a permanent failure is not retried
    /// as transient — or [`crate::error::TinyAgentsError::Model`] when an
    /// unstructured [`ModelStreamItem::Failed`] item was folded in.
    pub fn finish(self) -> Result<ModelResponse> {
        if let Some(error) = self.failed_provider {
            return Err(crate::error::TinyAgentsError::Provider(Box::new(error)));
        }

        if let Some(message) = self.failed {
            return Err(crate::error::TinyAgentsError::Model(message));
        }

        if let Some(mut response) = self.completed {
            // Reconcile the response and message usage with any streamed
            // `UsageDelta`, preferring an already-present value and never
            // overwriting a known usage with `None` (which previously clobbered a
            // message-level usage the completed response carried).
            let merged = response.usage.or(response.message.usage).or(self.usage);
            response.usage = merged;
            response.message.usage = merged;
            return Ok(response);
        }

        // No authoritative response: reconstruct from accumulated deltas.
        //
        // Reasoning streamed on the side channel is preserved as a leading
        // `Thinking` block rather than dropped, so the reconstructed message
        // carries the model's thinking for persistence and provider replay.
        // (Signatures are provider-signed only on the Anthropic Messages path,
        // wired separately; the OpenAI-compatible path leaves them `None`.)
        let mut content = Vec::new();
        if !self.reasoning.is_empty() {
            content.push(ContentBlock::Thinking {
                text: self.reasoning,
                signature: None,
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text(self.text));
        }
        let tool_calls = self
            .tool_chunks
            .into_iter()
            .map(|(id, args, name)| ToolCall {
                name: name.unwrap_or_default(),
                arguments: serde_json::from_str(&args).unwrap_or(Value::Null),
                id,
            })
            .collect();
        let message = AssistantMessage {
            id: None,
            content,
            tool_calls,
            usage: self.usage,
        };
        Ok(ModelResponse {
            message,
            usage: self.usage,
            finish_reason: None,
            raw: None,
            resolved_model: None,
        })
    }
}

/// Drives a [`ModelStream`] to completion and folds it into a [`ModelResponse`].
///
/// This is a convenience wrapper over [`StreamAccumulator`] for callers that do
/// not need to observe individual items.
///
/// # Errors
///
/// Returns an error when the stream terminates with [`ModelStreamItem::Failed`].
pub async fn collect_model_stream(mut stream: ModelStream) -> Result<ModelResponse> {
    let mut accumulator = StreamAccumulator::new();
    while let Some(item) = stream.next().await {
        accumulator.push(&item);
    }
    accumulator.finish()
}

#[cfg(test)]
mod test;
