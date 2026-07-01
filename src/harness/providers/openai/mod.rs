//! Real OpenAI Chat Completions provider (feature `openai`).
//!
//! This is one of the concrete leaves the recursive runtime bottoms out in: a
//! single [`OpenAiModel`] backs hosted OpenAI *and* every OpenAI-compatible
//! endpoint (Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together,
//! Mistral) via the preset constructors below, so the sub-agent / sub-graph
//! layers above never need to know which provider answered.
//!
//! [`OpenAiModel`] implements [`ChatModel`] against the hosted OpenAI Chat
//! Completions endpoint (`POST {base_url}/chat/completions`). It translates the
//! provider-neutral [`ModelRequest`] into OpenAI's JSON wire format (see
//! [`types`]), performs the HTTP call with `reqwest`, and maps the response back
//! into a [`ModelResponse`] with a fully-populated [`AssistantMessage`],
//! [`ToolCall`]s, [`Usage`], and finish reason.
//!
//! The wire (de)serialization shapes live in [`types`]; this module owns only
//! the translation logic and the HTTP transport, keeping OpenAI-specific JSON
//! out of the rest of the harness.
//!
//! # Example
//!
//! ```no_run
//! use tinyagents::harness::providers::openai::OpenAiModel;
//!
//! # fn main() -> tinyagents::Result<()> {
//! // Reads OPENAI_API_KEY (and optional OPENAI_MODEL / OPENAI_BASE_URL).
//! let model = OpenAiModel::from_env()?;
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

mod types;

pub use types::*;

use std::collections::VecDeque;
use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{Map, Value, json};

use crate::error::{Result, TinyAgentsError};
use crate::harness::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use crate::harness::model::{
    ChatModel, Modalities, ModelProfile, ModelRequest, ModelResponse, ModelStatus, ModelStream,
    ModelStreamItem, ProviderError, ResponseFormat, ToolChoice,
};
use crate::harness::tool::{ToolCall, ToolDelta};
use crate::harness::usage::Usage;

use super::ProviderSpec;

/// Default model id used when neither the request nor the builder override it.
const DEFAULT_MODEL: &str = "gpt-4.1-mini";
/// Default OpenAI API base URL.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// A [`ChatModel`] backed by the hosted OpenAI Chat Completions API.
///
/// Construct one with [`OpenAiModel::new`] (plus the `with_*` builders) or
/// [`OpenAiModel::from_env`]. The model holds a reusable [`reqwest::Client`] so
/// repeated calls share a connection pool.
pub struct OpenAiModel {
    /// Shared HTTP client.
    client: reqwest::Client,
    /// API key sent as a `Bearer` token.
    api_key: String,
    /// Default model id used when a request does not override it.
    model: String,
    /// Provider family identifier used in profiles and normalized errors.
    provider: String,
    /// API base URL (no trailing slash); `/chat/completions` is appended.
    base_url: String,
    /// Capability profile derived from the default model id.
    profile: ModelProfile,
}

/// Derives a static [`ModelProfile`] for an OpenAI(-compatible) model id.
///
/// All targets support tool calling, streaming (including tool-call chunks),
/// and JSON Schema response formats. Modern OpenAI-family models additionally
/// advertise native structured output and (for the o-series) reasoning output.
fn derive_profile(provider: &str, model: &str) -> ModelProfile {
    let lower = model.to_ascii_lowercase();
    let native_structured = lower.contains("gpt-4o")
        || lower.contains("gpt-4.1")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4");
    let reasoning = lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4");
    ModelProfile {
        provider: Some(provider.to_string()),
        model: Some(model.to_string()),
        status: ModelStatus::Stable,
        modalities: Modalities {
            image_in: true,
            ..Modalities::default()
        },
        tool_calling: true,
        parallel_tool_calls: true,
        streaming: true,
        streaming_tool_chunks: true,
        native_structured_output: native_structured,
        json_schema: true,
        reasoning,
        ..ModelProfile::default()
    }
}

impl OpenAiModel {
    /// Creates a model with the given API key, the default model
    /// (`gpt-4.1-mini`), and the default base URL (`https://api.openai.com/v1`).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            provider: "openai".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            profile: derive_profile("openai", DEFAULT_MODEL),
        }
    }

    /// Overrides the default model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.profile = derive_profile(&self.provider, &self.model);
        self
    }

    /// Overrides the provider family id used in profiles and normalized errors.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self.profile = derive_profile(&self.provider, &self.model);
        self
    }

    /// Overrides the API base URL. A trailing slash is trimmed so the joined
    /// endpoint is always `{base_url}/chat/completions`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Builds a model from environment variables.
    ///
    /// Reads `OPENAI_API_KEY` (required), `OPENAI_MODEL` (optional, defaults to
    /// `gpt-4.1-mini`), and `OPENAI_BASE_URL` (optional, defaults to
    /// `https://api.openai.com/v1`).
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Validation`] when `OPENAI_API_KEY` is missing
    /// or empty.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| {
                TinyAgentsError::Validation(
                    "OPENAI_API_KEY is not set; export it or add it to a .env file".to_string(),
                )
            })?;

        let mut model = Self::new(api_key);
        if let Ok(name) = std::env::var("OPENAI_MODEL")
            && !name.trim().is_empty()
        {
            model = model.with_model(name);
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL")
            && !url.trim().is_empty()
        {
            model = model.with_base_url(url);
        }
        Ok(model)
    }

    /// Builds an OpenAI-compatible model from a provider spec and explicit API
    /// key.
    pub fn from_spec(spec: ProviderSpec, api_key: impl Into<String>) -> Result<Self> {
        if spec.model.trim().is_empty() {
            return Err(TinyAgentsError::Validation(
                "provider spec model must not be empty".to_string(),
            ));
        }
        if spec.base_url.trim().is_empty() {
            return Err(TinyAgentsError::Validation(
                "provider spec base_url must not be empty".to_string(),
            ));
        }
        Ok(Self::compatible_provider(
            spec.provider,
            api_key,
            spec.base_url,
            spec.model,
        ))
    }

    /// Builds an OpenAI-compatible model from a provider spec, reading the API
    /// key from the spec's environment variable when required.
    pub fn from_spec_env(spec: ProviderSpec) -> Result<Self> {
        let api_key = if spec.requires_api_key {
            let env = spec.api_key_env.as_deref().ok_or_else(|| {
                TinyAgentsError::Validation(format!(
                    "{} requires an api_key_env in ProviderSpec",
                    spec.provider
                ))
            })?;
            std::env::var(env)
                .ok()
                .filter(|k| !k.trim().is_empty())
                .ok_or_else(|| {
                    TinyAgentsError::Validation(format!(
                        "{env} is not set; export it or provide an explicit API key"
                    ))
                })?
        } else {
            "local".to_string()
        };
        Self::from_spec(spec, api_key)
    }

    /// Lists the models the provider advertises via `GET {base_url}/models`.
    ///
    /// This is a provider/account-level capability (independent of which model
    /// this handle is bound to), so it uses the same credentials and base URL as
    /// chat calls. Every OpenAI-compatible endpoint (Ollama, Together, Groq,
    /// OpenRouter, …) serves the same shape, so this doubles as runtime model
    /// discovery for local/self-hosted providers. Returned ids can be fed to
    /// [`with_model`](Self::with_model).
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Model`] on transport failure or a non-2xx
    /// status, and [`TinyAgentsError::Serialization`] when the body cannot be
    /// decoded.
    pub async fn list_models(&self) -> Result<Vec<ModelListing>> {
        let url = format!("{}/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| {
                let error =
                    self.provider_error(format!("request to {url} failed: {e}"), None, None, None);
                TinyAgentsError::Model(self.provider_failure_message(&error))
            })?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("openai response body read failed: {e}"))
        })?;

        if !status.is_success() {
            let error = self.parse_error_body(status.as_u16(), &text);
            return Err(TinyAgentsError::Model(
                self.provider_failure_message(&error),
            ));
        }

        let listing: ModelListWire = serde_json::from_str(&text)?;
        Ok(listing.data)
    }

    // -----------------------------------------------------------------------
    // OpenAI-compatible provider presets
    //
    // DeepSeek, Groq, xAI, OpenRouter, Together, Mistral, Ollama, and
    // Anthropic's compatibility endpoint all accept the same Chat Completions
    // wire format, so the same [`OpenAiModel`] talks to all of them — only the
    // base URL and default model differ. Each preset is a thin wrapper over
    // [`OpenAiModel::new`] + [`with_base_url`][Self::with_base_url] +
    // [`with_model`][Self::with_model]; override the model with `with_model`.
    // -----------------------------------------------------------------------

    /// Points at an arbitrary OpenAI-compatible endpoint with an explicit base
    /// URL and default model.
    ///
    /// Use this for any provider that implements the Chat Completions API but is
    /// not covered by a named preset below.
    pub fn compatible(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new(api_key).with_base_url(base_url).with_model(model)
    }

    /// Points at an arbitrary OpenAI-compatible endpoint with an explicit
    /// provider id, base URL, and default model.
    pub fn compatible_provider(
        provider: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new(api_key)
            .with_provider(provider)
            .with_base_url(base_url)
            .with_model(model)
    }

    /// DeepSeek (`https://api.deepseek.com/v1`), default model `deepseek-chat`.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "deepseek",
            api_key,
            "https://api.deepseek.com/v1",
            "deepseek-chat",
        )
    }

    /// Anthropic's OpenAI-compatible endpoint (`https://api.anthropic.com/v1`),
    /// default model `claude-3-5-sonnet-latest`.
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "anthropic",
            api_key,
            "https://api.anthropic.com/v1",
            "claude-3-5-sonnet-latest",
        )
    }

    /// Groq (`https://api.groq.com/openai/v1`), default model
    /// `llama-3.3-70b-versatile`.
    pub fn groq(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "groq",
            api_key,
            "https://api.groq.com/openai/v1",
            "llama-3.3-70b-versatile",
        )
    }

    /// xAI (`https://api.x.ai/v1`), default model `grok-2-latest`.
    pub fn xai(api_key: impl Into<String>) -> Self {
        Self::compatible_provider("xai", api_key, "https://api.x.ai/v1", "grok-2-latest")
    }

    /// OpenRouter (`https://openrouter.ai/api/v1`), default model
    /// `openai/gpt-4o-mini`.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "openrouter",
            api_key,
            "https://openrouter.ai/api/v1",
            "openai/gpt-4o-mini",
        )
    }

    /// Together AI (`https://api.together.xyz/v1`), default model
    /// `meta-llama/Llama-3.3-70B-Instruct-Turbo`.
    pub fn together(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "together",
            api_key,
            "https://api.together.xyz/v1",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )
    }

    /// Mistral (`https://api.mistral.ai/v1`), default model
    /// `mistral-small-latest`.
    pub fn mistral(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "mistral",
            api_key,
            "https://api.mistral.ai/v1",
            "mistral-small-latest",
        )
    }

    /// A local Ollama server (`http://localhost:11434/v1`), default model
    /// `llama3.2`. Ollama ignores the API key, so a placeholder is used.
    pub fn ollama() -> Self {
        Self::compatible_provider("ollama", "ollama", "http://localhost:11434/v1", "llama3.2")
    }

    /// Returns the default model id this instance will request.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the configured provider family id.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Returns the configured API base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Translates a provider-neutral [`ModelRequest`] into the OpenAI wire
    /// request body. The per-request `model` override wins over the instance
    /// default.
    fn translate_request(&self, request: &ModelRequest) -> Result<ChatCompletionRequest> {
        let messages = request
            .messages
            .iter()
            .map(translate_message)
            .collect::<Result<Vec<_>>>()?;

        let tools: Vec<ToolWire> = request
            .tools
            .iter()
            .map(|schema| ToolWire {
                kind: "function".to_string(),
                function: FunctionSchemaWire {
                    name: schema.name.clone(),
                    description: schema.description.clone(),
                    parameters: schema.parameters.clone(),
                },
            })
            .collect();

        // tool_choice is only meaningful when tools are declared.
        let tool_choice = if tools.is_empty() {
            None
        } else {
            Some(translate_tool_choice(&request.tool_choice))
        };

        let response_format = request
            .response_format
            .as_ref()
            .and_then(translate_response_format);

        Ok(ChatCompletionRequest {
            model: request.model.clone().unwrap_or_else(|| self.model.clone()),
            messages,
            tools,
            tool_choice,
            response_format,
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            stop: request.stop_sequences.clone(),
            seed: request.seed,
            stream: false,
            stream_options: None,
            extra: provider_extra_options(&request.provider_options)?,
        })
    }

    fn provider_error(
        &self,
        message: impl Into<String>,
        status: Option<u16>,
        code: Option<String>,
        raw: Option<Value>,
    ) -> ProviderError {
        let retryable = status.is_some_and(|s| s == 408 || s == 409 || s == 429 || s >= 500);
        ProviderError {
            provider: self.provider.clone(),
            model: Some(self.model.clone()),
            status,
            code,
            message: message.into(),
            retryable,
            raw,
        }
    }

    fn provider_failure_message(&self, error: &ProviderError) -> String {
        format!(
            "{} returned{}{}: {}",
            error.provider,
            error
                .status
                .map(|status| format!(" HTTP {status}"))
                .unwrap_or_default(),
            error
                .code
                .as_deref()
                .map(|code| format!(" ({code})"))
                .unwrap_or_default(),
            error.message
        )
    }

    fn parse_error_body(&self, status: u16, text: &str) -> ProviderError {
        let raw = serde_json::from_str::<Value>(text).ok();
        let error_obj = raw.as_ref().and_then(|value| value.get("error"));
        let message = error_obj
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .or_else(|| {
                raw.as_ref()
                    .and_then(|value| value.get("message"))
                    .and_then(Value::as_str)
            })
            .filter(|message| !message.trim().is_empty())
            .unwrap_or(text)
            .to_string();
        let code = error_obj
            .and_then(|error| error.get("code").or_else(|| error.get("type")))
            .and_then(Value::as_str)
            .map(str::to_string);
        self.provider_error(message, Some(status), code, raw)
    }
}

/// Returns provider-specific top-level fields to flatten into the request body.
///
/// Core OpenAI-compatible fields are intentionally reserved so normalized
/// TinyAgents fields remain the source of truth. Callers that need local-model
/// controls should use distinct provider fields such as Ollama's `options`.
fn provider_extra_options(options: &Value) -> Result<Map<String, Value>> {
    if options.is_null() {
        return Ok(Map::new());
    }
    let Some(object) = options.as_object() else {
        return Err(TinyAgentsError::Validation(
            "provider_options for OpenAI-compatible providers must be a JSON object".to_string(),
        ));
    };

    const RESERVED: &[&str] = &[
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "seed",
        "stream",
        "stream_options",
    ];

    Ok(object
        .iter()
        .filter(|(key, _)| !RESERVED.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

/// Translates one harness [`Message`] into an OpenAI wire message.
fn translate_message(message: &Message) -> Result<ChatMessageWire> {
    let wire = match message {
        Message::System(_) => ChatMessageWire {
            role: "system".to_string(),
            content: Some(message.text()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        Message::User(_) => ChatMessageWire {
            role: "user".to_string(),
            content: Some(message.text()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        Message::Assistant(assistant) => {
            let text = message.text();
            // OpenAI accepts a null content for tool-call-only assistant turns.
            let content = if text.is_empty() && !assistant.tool_calls.is_empty() {
                None
            } else {
                Some(text)
            };
            let tool_calls = assistant
                .tool_calls
                .iter()
                .map(|call| {
                    Ok(ToolCallWire {
                        id: call.id.clone(),
                        kind: "function".to_string(),
                        function: FunctionCallWire {
                            name: call.name.clone(),
                            // OpenAI expects arguments as a JSON string.
                            arguments: serde_json::to_string(&call.arguments)?,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ChatMessageWire {
                role: "assistant".to_string(),
                content,
                tool_calls,
                tool_call_id: None,
            }
        }
        Message::Tool(tool) => ChatMessageWire {
            role: "tool".to_string(),
            content: Some(message.text()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool.tool_call_id.clone()),
        },
    };
    Ok(wire)
}

/// Translates a [`ToolChoice`] into the OpenAI `tool_choice` JSON value.
fn translate_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

/// Translates a [`ResponseFormat`] into the OpenAI `response_format` JSON value.
///
/// Returns `None` for [`ResponseFormat::Text`] so the field is omitted entirely.
fn translate_response_format(format: &ResponseFormat) -> Option<Value> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(json!({ "type": "json_object" })),
        // OpenAI supports native structured output, so `Auto` maps to a JSON
        // schema request directly. (The agent loop normally resolves `Auto`
        // before reaching the provider; this keeps direct calls correct too.)
        ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
            Some(json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": true,
                }
            }))
        }
    }
}

/// Parses an OpenAI response body (already decoded into a [`Value`]) into a
/// provider-neutral [`ModelResponse`].
///
/// The first choice is used. The raw JSON is preserved in
/// [`ModelResponse::raw`].
///
/// # Errors
///
/// Returns [`TinyAgentsError::Serialization`] if the value does not match the
/// expected response shape, or [`TinyAgentsError::Model`] when no choices are
/// present.
fn parse_response(value: Value) -> Result<ModelResponse> {
    let parsed: ChatCompletionResponse = serde_json::from_value(value.clone())?;

    let choice = parsed.choices.into_iter().next().ok_or_else(|| {
        TinyAgentsError::Model("openai response contained no choices".to_string())
    })?;

    let mut content = Vec::new();
    if let Some(text) = choice.message.content.filter(|t| !t.is_empty()) {
        content.push(ContentBlock::Text(text));
    }

    let tool_calls = choice
        .message
        .tool_calls
        .into_iter()
        .map(|call| {
            Ok(ToolCall {
                id: call.id.clone(),
                name: call.function.name.clone(),
                // Tool arguments arrive as a JSON string. Invalid JSON is a
                // provider/model error, not an empty/default argument payload.
                arguments: parse_tool_arguments(
                    "openai response",
                    &call.id,
                    &call.function.name,
                    &call.function.arguments,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let usage = parsed.usage.map(convert_usage);

    let message = AssistantMessage {
        id: parsed.id,
        content,
        tool_calls,
        usage,
    };

    Ok(ModelResponse {
        message,
        usage,
        finish_reason: choice.finish_reason,
        raw: Some(value),
        resolved_model: None,
    })
}

fn parse_tool_arguments(context: &str, call_id: &str, name: &str, raw: &str) -> Result<Value> {
    serde_json::from_str(raw).map_err(|err| {
        TinyAgentsError::Model(format!(
            "{context} contained invalid JSON arguments for tool call `{call_id}` (`{name}`): {err}; raw arguments: {raw:?}"
        ))
    })
}

/// Converts an OpenAI [`UsageWire`] into the harness-neutral [`Usage`].
fn convert_usage(wire: UsageWire) -> Usage {
    Usage {
        input_tokens: wire.prompt_tokens,
        output_tokens: wire.completion_tokens,
        total_tokens: wire.total_tokens,
        cache_read_tokens: wire
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0),
        ..Usage::default()
    }
}

// ---------------------------------------------------------------------------
// Streaming (SSE) machinery
// ---------------------------------------------------------------------------

/// In-progress reconstruction of a single tool call across streamed fragments.
#[derive(Clone, Debug, Default)]
struct ToolCallBuild {
    /// Provider-assigned call id (filled from the first fragment carrying it).
    id: String,
    /// Function name (filled from the first fragment carrying it).
    name: String,
    /// Concatenated stringified-JSON argument fragments.
    args: String,
}

/// Provider-side accumulator that rebuilds the authoritative [`ModelResponse`]
/// from streamed chunks. Distinct from the generic
/// [`StreamAccumulator`][crate::harness::model::StreamAccumulator]: it tracks
/// tool-call names and ids (which the neutral deltas omit) so the terminal
/// [`ModelStreamItem::Completed`] carries a faithful response.
#[derive(Clone, Debug, Default)]
struct OpenAiStreamAcc {
    id: Option<String>,
    text: String,
    tool_calls: Vec<ToolCallBuild>,
    usage: Option<Usage>,
    finish_reason: Option<String>,
}

impl OpenAiStreamAcc {
    /// Folds one parsed chunk into the accumulator and pushes the corresponding
    /// neutral [`ModelStreamItem`]s onto `pending`.
    fn ingest(&mut self, chunk: ChatCompletionChunk, pending: &mut VecDeque<ModelStreamItem>) {
        if let Some(id) = chunk.id
            && self.id.is_none()
        {
            self.id = Some(id);
        }
        if let Some(usage_wire) = chunk.usage {
            let usage = convert_usage(usage_wire);
            self.usage = Some(usage);
            pending.push_back(ModelStreamItem::UsageDelta(usage));
        }
        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
            if let Some(content) = choice.delta.content.filter(|c| !c.is_empty()) {
                self.text.push_str(&content);
                pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                    text: content,
                    reasoning: String::new(),
                    tool_call: None,
                }));
            }
            for fragment in choice.delta.tool_calls {
                let idx = fragment.index as usize;
                while self.tool_calls.len() <= idx {
                    self.tool_calls.push(ToolCallBuild::default());
                }
                let slot = &mut self.tool_calls[idx];
                if let Some(id) = fragment.id.filter(|id| !id.is_empty()) {
                    slot.id = id;
                }
                if let Some(function) = fragment.function {
                    if let Some(name) = function.name.filter(|n| !n.is_empty()) {
                        slot.name = name;
                    }
                    if let Some(args) = function.arguments.filter(|a| !a.is_empty()) {
                        slot.args.push_str(&args);
                        let call_id = if slot.id.is_empty() {
                            format!("tool-{idx}")
                        } else {
                            slot.id.clone()
                        };
                        pending.push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                            call_id,
                            content: args,
                        }));
                    }
                }
            }
        }
    }

    /// Consumes the accumulator into the final, merged [`ModelResponse`].
    fn into_response(self) -> Result<ModelResponse> {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(ContentBlock::Text(self.text));
        }
        let tool_calls = self
            .tool_calls
            .into_iter()
            .filter(|b| !b.name.is_empty() || !b.args.is_empty())
            .enumerate()
            .map(|(idx, b)| {
                let id = if b.id.is_empty() {
                    format!("tool-{idx}")
                } else {
                    b.id.clone()
                };
                Ok(ToolCall {
                    id: id.clone(),
                    name: b.name.clone(),
                    arguments: parse_tool_arguments("openai stream", &id, &b.name, &b.args)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let message = AssistantMessage {
            id: self.id,
            content,
            tool_calls,
            usage: self.usage,
        };
        Ok(ModelResponse {
            message,
            usage: self.usage,
            finish_reason: self.finish_reason,
            raw: None,
            resolved_model: None,
        })
    }
}

/// Mutable driver state threaded through [`futures::stream::unfold`] while
/// parsing the SSE byte stream into [`ModelStreamItem`]s.
struct SseState {
    /// Raw response byte chunks (errors already mapped onto the crate error).
    bytes: Pin<Box<dyn Stream<Item = Result<Vec<u8>>> + Send>>,
    /// Bytes received but not yet split into complete lines.
    buf: String,
    /// Parsed items waiting to be yielded, in order.
    pending: VecDeque<ModelStreamItem>,
    /// Provider-side response reconstruction.
    acc: OpenAiStreamAcc,
    /// Provider family id used in normalized stream failures.
    provider: String,
    /// Provider model id used in normalized stream failures.
    model: String,
    /// Whether the leading [`ModelStreamItem::Started`] has been emitted.
    started: bool,
    /// Whether the byte stream ended or `[DONE]` was seen.
    finished: bool,
    /// Whether the terminal [`ModelStreamItem::Completed`]/[`ModelStreamItem::Failed`]
    /// has been emitted.
    terminal_emitted: bool,
}

impl SseState {
    /// Splits buffered bytes into complete lines and folds each SSE `data:`
    /// payload into the accumulator. The trailing partial line (if any) is kept
    /// for the next chunk.
    fn drain_lines(&mut self) {
        while let Some(pos) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some(rest) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = rest.trim();
            if payload == "[DONE]" {
                self.finished = true;
                continue;
            }
            // Ignore keepalives / unparseable lines rather than failing the run.
            if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(payload) {
                let mut pending = std::mem::take(&mut self.pending);
                self.acc.ingest(chunk, &mut pending);
                self.pending = pending;
            }
        }
    }
}

/// Advances the SSE [`SseState`] by one item for [`futures::stream::unfold`].
async fn sse_next(mut state: SseState) -> Option<(ModelStreamItem, SseState)> {
    loop {
        if let Some(item) = state.pending.pop_front() {
            return Some((item, state));
        }
        if !state.started {
            state.started = true;
            return Some((ModelStreamItem::Started, state));
        }
        if state.finished {
            if state.terminal_emitted {
                return None;
            }
            state.terminal_emitted = true;
            return match std::mem::take(&mut state.acc).into_response() {
                Ok(response) => Some((ModelStreamItem::Completed(response), state)),
                Err(error) => {
                    let provider_error = ProviderError {
                        provider: state.provider.clone(),
                        model: Some(state.model.clone()),
                        code: Some("invalid_tool_arguments".to_string()),
                        message: error.to_string(),
                        retryable: false,
                        ..ProviderError::default()
                    };
                    Some((ModelStreamItem::ProviderFailed(provider_error), state))
                }
            };
        }
        match state.bytes.next().await {
            Some(Ok(chunk)) => {
                state.buf.push_str(&String::from_utf8_lossy(&chunk));
                state.drain_lines();
            }
            Some(Err(error)) => {
                state.finished = true;
                state.terminal_emitted = true;
                let provider_error = ProviderError {
                    provider: state.provider.clone(),
                    model: Some(state.model.clone()),
                    message: error.to_string(),
                    retryable: true,
                    ..ProviderError::default()
                };
                return Some((ModelStreamItem::ProviderFailed(provider_error), state));
            }
            None => {
                state.finished = true;
            }
        }
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for OpenAiModel {
    /// Returns the capability profile derived from the configured model id.
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    /// Invokes the OpenAI Chat Completions endpoint and maps the response into a
    /// [`ModelResponse`].
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Model`] on transport failure or a non-2xx
    /// status (the message includes the status code and response body), and
    /// [`TinyAgentsError::Serialization`] when the response cannot be decoded.
    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let body = self.translate_request(&request)?;
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let error =
                    self.provider_error(format!("request to {url} failed: {e}"), None, None, None);
                TinyAgentsError::Model(self.provider_failure_message(&error))
            })?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("openai response body read failed: {e}"))
        })?;

        if !status.is_success() {
            let error = self.parse_error_body(status.as_u16(), &text);
            return Err(TinyAgentsError::Model(
                self.provider_failure_message(&error),
            ));
        }

        let value: Value = serde_json::from_str(&text)?;
        parse_response(value)
    }

    /// Streams the OpenAI Chat Completions response as a real [`ModelStream`].
    ///
    /// Sends the request with `stream: true` (and `stream_options.include_usage`
    /// so a usage chunk is delivered), reads the Server-Sent-Events body
    /// incrementally with [`reqwest::Response::bytes_stream`], and parses each
    /// `data:` line into [`ModelStreamItem`]s: a leading
    /// [`ModelStreamItem::Started`], a [`ModelStreamItem::MessageDelta`] per text
    /// fragment, a [`ModelStreamItem::ToolCallDelta`] per tool-call argument
    /// fragment, a [`ModelStreamItem::UsageDelta`] when usage arrives, and a
    /// terminal [`ModelStreamItem::Completed`] carrying the fully merged
    /// response (with reassembled tool-call names and ids). Transport errors
    /// surface as a terminal [`ModelStreamItem::ProviderFailed`].
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Model`] when the initial request fails or the
    /// endpoint returns a non-2xx status (per-chunk transport errors are
    /// surfaced as [`ModelStreamItem::ProviderFailed`] inside the stream
    /// instead).
    async fn stream(&self, _state: &State, request: ModelRequest) -> Result<ModelStream> {
        let mut body = self.translate_request(&request)?;
        body.stream = true;
        body.stream_options = Some(json!({ "include_usage": true }));
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let error = self.provider_error(
                    format!("stream request to {url} failed: {e}"),
                    None,
                    None,
                    None,
                );
                TinyAgentsError::Model(self.provider_failure_message(&error))
            })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let error = self.parse_error_body(status.as_u16(), &text);
            return Err(TinyAgentsError::Model(
                self.provider_failure_message(&error),
            ));
        }

        // Map each raw byte chunk onto an owned `Vec<u8>` so the boxed stream's
        // item type is nameable without depending on the `bytes` crate.
        let bytes = response.bytes_stream().map(|chunk| {
            chunk
                .map(|b| b.to_vec())
                .map_err(|e| TinyAgentsError::Model(format!("stream chunk failed: {e}")))
        });

        let state = SseState {
            bytes: Box::pin(bytes),
            buf: String::new(),
            pending: VecDeque::new(),
            acc: OpenAiStreamAcc::default(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            started: false,
            finished: false,
            terminal_emitted: false,
        };

        Ok(Box::pin(futures::stream::unfold(state, sse_next)))
    }
}

#[cfg(test)]
mod test;
