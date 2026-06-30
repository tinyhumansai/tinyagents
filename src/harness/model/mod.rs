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
        if let Some(requested) = selection.requested
            && let Some(model) = self.get(&requested)
            && model_satisfies(model.as_ref(), required)
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
            && model_satisfies(model.as_ref(), required)
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
                && model_satisfies(model.as_ref(), required)
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
            && model_satisfies(model.as_ref(), required)
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
            .filter(|model| model_satisfies(model.as_ref(), required))
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
    /// Per-call-id accumulated tool-call argument fragments, in first-seen
    /// order.
    tool_chunks: Vec<(String, String)>,
    /// Most recent usage value seen.
    usage: Option<Usage>,
    /// Authoritative final response, when a `Completed` item was seen.
    completed: Option<ModelResponse>,
    /// Terminal error message, when a failure item was seen.
    failed: Option<String>,
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
                if let Some(tool_call) = &delta.tool_call {
                    self.push_tool_chunk(&tool_call.call_id, &tool_call.content);
                }
            }
            ModelStreamItem::ToolCallDelta(delta) => {
                self.push_tool_chunk(&delta.call_id, &delta.content);
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
                self.failed = Some(format!(
                    "{} provider error{}: {}",
                    error.provider,
                    error
                        .code
                        .as_deref()
                        .map(|code| format!(" ({code})"))
                        .unwrap_or_default(),
                    error.message
                ));
            }
        }
    }

    /// Appends a tool-call argument fragment for `call_id`, preserving
    /// first-seen ordering across calls.
    fn push_tool_chunk(&mut self, call_id: &str, content: &str) {
        if let Some(entry) = self.tool_chunks.iter_mut().find(|(id, _)| id == call_id) {
            entry.1.push_str(content);
        } else {
            self.tool_chunks
                .push((call_id.to_string(), content.to_string()));
        }
    }

    /// Returns `true` when a terminal item (`Completed` or `Failed`) has been
    /// folded in.
    pub fn is_terminal(&self) -> bool {
        self.completed.is_some() || self.failed.is_some()
    }

    /// Consumes the accumulator and returns the merged response.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::TinyAgentsError::Model`] when a
    /// [`ModelStreamItem::Failed`] item was folded in.
    pub fn finish(self) -> Result<ModelResponse> {
        if let Some(message) = self.failed {
            return Err(crate::error::TinyAgentsError::Model(message));
        }

        if let Some(mut response) = self.completed {
            if response.usage.is_none() {
                response.usage = self.usage;
                response.message.usage = self.usage;
            }
            return Ok(response);
        }

        // No authoritative response: reconstruct from accumulated deltas.
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(ContentBlock::Text(self.text));
        }
        let tool_calls = self
            .tool_chunks
            .into_iter()
            .map(|(id, args)| ToolCall {
                name: String::new(),
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
