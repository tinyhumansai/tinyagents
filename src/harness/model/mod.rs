//! Harness model layer.
//!
//! See [`types`] for definitions. This module provides builder methods on
//! [`ModelRequest`], accessors on [`ModelResponse`], and the [`ModelRegistry`]
//! logic.

mod types;

use std::sync::Arc;

use serde_json::Value;

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

    /// Sets the maximum output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
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
        if let Some(requested) = selection.requested
            && let Some(model) = self.get(&requested)
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
            if let Some(model) = self.get(&hint.model) {
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
        self.default_model().map(|model| ResolvedModelBinding {
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
        })
    }

    /// Returns registered model names in sorted order.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.models.keys().cloned().collect();
        names.sort();
        names
    }
}

impl<State: Send + Sync> Default for ModelRegistry<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
