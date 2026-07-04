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
use std::time::Duration;

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
/// Sane default TCP connect timeout applied to every call. Bounds connection
/// establishment without capping the (potentially long) response body, so it is
/// safe for streaming too.
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 30;
/// Default overall timeout applied to unary calls when the request does not set
/// [`ModelRequest::timeout_ms`]. Streaming calls get no overall cap by default.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 600;

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

/// Returns `true` for OpenAI o-series reasoning models (`o1`/`o3`/`o4`), which
/// reject `max_tokens` and require `max_completion_tokens` instead.
fn is_reasoning_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4")
}

/// Derives a static [`ModelProfile`] for an OpenAI(-compatible) model id.
///
/// All targets support tool calling, streaming (including tool-call chunks),
/// and JSON Schema response formats. Modern OpenAI-family models additionally
/// advertise native structured output and (for the o-series) reasoning output.
fn derive_profile(provider: &str, model: &str) -> ModelProfile {
    let lower = model.to_ascii_lowercase();
    let reasoning = is_reasoning_model(model);
    let native_structured = lower.contains("gpt-4o") || lower.contains("gpt-4.1") || reasoning;
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
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
                .build()
                .expect("default reqwest client builds"),
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
            return Err(TinyAgentsError::Provider(Box::new(error)));
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

        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        // The o-series reasoning models reject `max_tokens` and require
        // `max_completion_tokens`; classic Chat Completions models use
        // `max_tokens`. Route the request's cap to whichever field the target
        // model accepts.
        let (max_tokens, max_completion_tokens) = if is_reasoning_model(&model) {
            (None, request.max_tokens)
        } else {
            (request.max_tokens, None)
        };

        Ok(ChatCompletionRequest {
            model,
            messages,
            tools,
            tool_choice,
            response_format,
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens,
            max_completion_tokens,
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

/// Resolves the per-request timeout to apply to an outbound HTTP call.
///
/// An explicit [`ModelRequest::timeout_ms`] always wins. Otherwise a unary call
/// falls back to [`DEFAULT_REQUEST_TIMEOUT_SECS`], while a streaming call gets no
/// overall cap (a total-request timeout would truncate a legitimately
/// long-running stream mid-flight).
fn request_timeout(timeout_ms: Option<u64>, streaming: bool) -> Option<Duration> {
    match timeout_ms {
        Some(ms) => Some(Duration::from_millis(ms)),
        None if streaming => None,
        None => Some(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)),
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
        "max_completion_tokens",
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
///
/// User messages are rendered as OpenAI content-parts when they carry non-text
/// blocks (for example images), so image inputs are actually sent rather than
/// silently dropped. Blocks that have no faithful OpenAI representation return a
/// [`TinyAgentsError::Validation`] instead of being discarded.
fn translate_message(message: &Message) -> Result<ChatMessageWire> {
    let wire = match message {
        Message::System(_) => ChatMessageWire {
            role: "system".to_string(),
            content: Some(MessageContentWire::Text(message.text())),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        Message::User(user) => ChatMessageWire {
            role: "user".to_string(),
            content: Some(translate_user_content(&user.content)?),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        Message::Assistant(assistant) => {
            let text = message.text();
            // OpenAI accepts a null content for tool-call-only assistant turns.
            let content = if text.is_empty() && !assistant.tool_calls.is_empty() {
                None
            } else {
                Some(MessageContentWire::Text(text))
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
            content: Some(MessageContentWire::Text(message.text())),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool.tool_call_id.clone()),
        },
    };
    Ok(wire)
}

/// Renders user-message content blocks into OpenAI message content.
///
/// Text-only content collapses to a plain string (preserving the historical wire
/// shape). When an image block is present, content is emitted as OpenAI
/// content-parts so the image is actually sent. JSON blocks are serialized into
/// text parts. A [`ContentBlock::ProviderExtension`] has no faithful OpenAI
/// representation, so it fails closed with a validation error rather than being
/// silently dropped.
fn translate_user_content(blocks: &[ContentBlock]) -> Result<MessageContentWire> {
    let has_image = blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Image(_)));

    if !has_image {
        // No image: render as a single string, but still fail closed on blocks
        // that cannot be represented.
        let mut text = String::new();
        for block in blocks {
            match block {
                ContentBlock::Text(t) => text.push_str(t),
                ContentBlock::Json(value) => text.push_str(&value.to_string()),
                ContentBlock::Image(_) => unreachable!("guarded by has_image"),
                ContentBlock::ProviderExtension(_) => {
                    return Err(unrepresentable_block_error());
                }
            }
        }
        return Ok(MessageContentWire::Text(text));
    }

    let mut parts = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            ContentBlock::Text(t) => parts.push(ContentPartWire::Text { text: t.clone() }),
            ContentBlock::Json(value) => parts.push(ContentPartWire::Text {
                text: value.to_string(),
            }),
            ContentBlock::Image(image) => parts.push(ContentPartWire::ImageUrl {
                image_url: ImageUrlWire {
                    url: image.url.clone(),
                },
            }),
            ContentBlock::ProviderExtension(_) => {
                return Err(unrepresentable_block_error());
            }
        }
    }
    Ok(MessageContentWire::Parts(parts))
}

/// Error returned when a content block cannot be represented in an OpenAI
/// request. Failing closed keeps the block from being silently dropped.
fn unrepresentable_block_error() -> TinyAgentsError {
    TinyAgentsError::Validation(
        "OpenAI request cannot represent a provider-extension content block; \
         remove it or target the originating provider"
            .to_string(),
    )
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

/// Returns the effective call id for a streamed tool-call slot: the
/// provider-assigned id when present, or a stable `tool-{slot}` fallback keyed to
/// the slot's position so delta ids and the final call id always agree.
fn tool_call_id(slot: usize, id: &str) -> String {
    if id.is_empty() {
        format!("tool-{slot}")
    } else {
        id.to_string()
    }
}

fn parse_tool_arguments(context: &str, call_id: &str, name: &str, raw: &str) -> Result<Value> {
    // Some OpenAI-compatible backends emit an empty arguments string for a
    // zero-argument tool call. That is a well-formed "no arguments" payload, not
    // malformed JSON, so map it to an empty object instead of failing the call.
    if raw.trim().is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    serde_json::from_str(raw).map_err(|err| {
        TinyAgentsError::Model(format!(
            "{context} contained invalid JSON arguments for tool call `{call_id}` (`{name}`): {err}; raw arguments: {raw:?}"
        ))
    })
}

/// Converts an OpenAI [`UsageWire`] into the harness-neutral [`Usage`].
fn convert_usage(wire: UsageWire) -> Usage {
    // OpenAI-compatible endpoints sometimes omit `total_tokens` entirely
    // (deserializes to `0` via `#[serde(default)]`); fall back to
    // `prompt + completion` so `total_tokens` is never a misleading zero for
    // a call that clearly consumed tokens.
    let total_tokens = if wire.total_tokens > 0 {
        wire.total_tokens
    } else {
        wire.prompt_tokens + wire.completion_tokens
    };
    Usage {
        input_tokens: wire.prompt_tokens,
        output_tokens: wire.completion_tokens,
        total_tokens,
        cache_read_tokens: wire
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0),
        reasoning_tokens: wire
            .completion_tokens_details
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0),
        ..Usage::default()
    }
}

/// Normalizes provider-specific reasoning/thinking payloads into text.
///
/// OpenAI-compatible gateways do not agree on this field: some stream a plain
/// `reasoning_content` string, others use `reasoning`, and a few wrap text in
/// an object/array. Preserve renderable text when obvious and ignore opaque
/// shapes rather than failing an otherwise valid completion.
fn reasoning_value_text(value: Value) -> Option<String> {
    match value {
        Value::String(text) => (!text.is_empty()).then_some(text),
        Value::Object(map) => ["text", "content", "summary"]
            .into_iter()
            .find_map(|key| map.get(key).and_then(Value::as_str))
            .filter(|text| !text.is_empty())
            .map(str::to_string),
        Value::Array(values) => {
            let text = values
                .into_iter()
                .filter_map(reasoning_value_text)
                .collect::<String>();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

/// Extracts the reasoning/thinking text from a streamed delta, accepting the
/// common OpenAI-compatible aliases.
fn delta_reasoning_text(delta: &mut ChunkDeltaWire) -> String {
    let mut text = String::new();
    for value in [delta.reasoning_content.take(), delta.reasoning.take()]
        .into_iter()
        .flatten()
    {
        if let Some(fragment) = reasoning_value_text(value) {
            text.push_str(&fragment);
        }
    }
    text
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
        for mut choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
            let reasoning = delta_reasoning_text(&mut choice.delta);
            if !reasoning.is_empty() {
                pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                    text: String::new(),
                    reasoning,
                    tool_call: None,
                }));
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
                let idx = self.resolve_slot(&fragment);
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
                        let call_id = tool_call_id(idx, &slot.id);
                        pending.push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                            call_id,
                            content: args,
                        }));
                    }
                }
            }
        }
    }

    /// Resolves the accumulator slot a streamed tool-call fragment belongs to.
    ///
    /// OpenAI itself always sends a stable `index`; some OpenAI-compatible
    /// backends omit it. When `index` is present it selects the slot directly
    /// (growing the vector as needed). When it is absent, fragments are
    /// correlated by `id`: a fragment carrying a new id opens a new slot, one
    /// carrying a known id reuses that slot, and an id-less continuation fragment
    /// (arguments only) appends to the most recent slot — so parallel calls no
    /// longer all collapse onto slot 0.
    fn resolve_slot(&mut self, fragment: &ToolCallChunkWire) -> usize {
        if let Some(index) = fragment.index {
            let idx = index as usize;
            while self.tool_calls.len() <= idx {
                self.tool_calls.push(ToolCallBuild::default());
            }
            return idx;
        }
        if let Some(id) = fragment.id.as_deref().filter(|id| !id.is_empty()) {
            if let Some(pos) = self.tool_calls.iter().position(|slot| slot.id == id) {
                return pos;
            }
            self.tool_calls.push(ToolCallBuild::default());
            return self.tool_calls.len() - 1;
        }
        if self.tool_calls.is_empty() {
            self.tool_calls.push(ToolCallBuild::default());
        }
        self.tool_calls.len() - 1
    }

    /// Consumes the accumulator into the final, merged [`ModelResponse`].
    fn into_response(self) -> Result<ModelResponse> {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(ContentBlock::Text(self.text));
        }
        // Enumerate over the full slot vector *before* filtering so the synthetic
        // fallback id (`tool-{idx}`) matches the one streamed in `ToolCallDelta`
        // items — filtering first would renumber the slots and desynchronize the
        // delta ids from the final call ids.
        let tool_calls = self
            .tool_calls
            .into_iter()
            .enumerate()
            .filter(|(_, b)| !b.name.is_empty() || !b.args.is_empty())
            .map(|(idx, b)| {
                let id = tool_call_id(idx, &b.id);
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
    /// Raw bytes received but not yet split into complete lines. Kept as bytes
    /// (not a `String`) so a multi-byte UTF-8 character split across two network
    /// chunks is reassembled before decoding, instead of being corrupted into
    /// replacement characters by a premature lossy decode.
    buf: Vec<u8>,
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
    /// Splits buffered bytes into complete newline-terminated lines and folds
    /// each SSE `data:` payload into the accumulator. The trailing partial line
    /// (if any) is kept in `buf` for the next chunk, so a `data:` line split
    /// across chunk boundaries — including one that splits a multi-byte UTF-8
    /// character — is only decoded once it is complete.
    fn drain_lines(&mut self) {
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            // A complete line (bounded by the ASCII `\n`) is whole UTF-8, so a
            // lossy decode here can no longer straddle a chunk boundary.
            let line = String::from_utf8_lossy(&line_bytes).into_owned();
            self.process_line(&line);
        }
    }

    /// Folds any bytes still buffered after the byte stream ends into a final
    /// line. Providers that terminate the last SSE event without a trailing
    /// newline would otherwise leave the final `data:` payload unprocessed.
    fn drain_remaining(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let line = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        self.process_line(&line);
    }

    /// Parses one SSE line and folds any resulting chunk into the accumulator.
    fn process_line(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        let Some(rest) = line.strip_prefix("data:") else {
            return;
        };
        let payload = rest.trim();
        if payload == "[DONE]" {
            self.finished = true;
            return;
        }
        // Ignore keepalives / unparseable lines rather than failing the run.
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        // Some providers stream a mid-stream `{"error": ...}` payload instead of
        // a chunk. This also deserializes cleanly as an all-defaults
        // `ChatCompletionChunk`, so it must be detected first and surfaced as a
        // terminal failure rather than folded in as an empty chunk and swallowed.
        if let Some(error) = value.get("error") {
            self.pending
                .push_back(ModelStreamItem::ProviderFailed(self.stream_error(error)));
            self.finished = true;
            self.terminal_emitted = true;
            return;
        }
        if let Ok(chunk) = serde_json::from_value::<ChatCompletionChunk>(value) {
            let mut pending = std::mem::take(&mut self.pending);
            self.acc.ingest(chunk, &mut pending);
            self.pending = pending;
        }
    }

    /// Builds a normalized [`ProviderError`] from a streamed `error` payload.
    fn stream_error(&self, error: &Value) -> ProviderError {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .filter(|message| !message.trim().is_empty())
            .unwrap_or("provider reported a stream error")
            .to_string();
        let code = error
            .get("code")
            .or_else(|| error.get("type"))
            .and_then(Value::as_str)
            .map(str::to_string);
        ProviderError {
            provider: self.provider.clone(),
            model: Some(self.model.clone()),
            code,
            message,
            retryable: false,
            raw: Some(error.clone()),
            ..ProviderError::default()
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
                state.buf.extend_from_slice(&chunk);
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
                // Drain any final `data:` line the provider sent without a
                // trailing newline before terminating.
                state.drain_remaining();
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

        let mut builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body);
        if let Some(timeout) = request_timeout(request.timeout_ms, false) {
            builder = builder.timeout(timeout);
        }
        let response = builder.send().await.map_err(|e| {
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
            return Err(TinyAgentsError::Provider(Box::new(error)));
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

        let mut builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body);
        if let Some(timeout) = request_timeout(request.timeout_ms, true) {
            builder = builder.timeout(timeout);
        }
        let response = builder.send().await.map_err(|e| {
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
            return Err(TinyAgentsError::Provider(Box::new(error)));
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
            buf: Vec::new(),
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
