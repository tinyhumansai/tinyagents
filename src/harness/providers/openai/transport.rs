//! HTTP transport: `OpenAiModel` construction, provider presets, request
//! building, and the `ChatModel` impl (`invoke`/`stream`).
//!
//! Split out of `openai/mod.rs`; see that module's doc comment for the
//! full provider overview.

use super::*;

/// How the provider expects the API credential to be sent on each request.
///
/// OpenAI-compatible endpoints diverge on auth: hosted OpenAI and most gateways
/// use `Bearer`, some providers use a bare `x-api-key`, Anthropic's compat
/// endpoint pairs `x-api-key` with an `anthropic-version` header, and a few need
/// an arbitrary custom header carrying the raw credential. Defaults to
/// [`AuthStyle::Bearer`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AuthStyle {
    /// No authentication header (e.g. a local Ollama server).
    None,
    /// `Authorization: Bearer <key>` — hosted OpenAI and most gateways.
    #[default]
    Bearer,
    /// `x-api-key: <key>` — used by some OpenAI-compatible providers.
    XApiKey,
    /// `x-api-key: <key>` + `anthropic-version: 2023-06-01` (Anthropic compat).
    Anthropic,
    /// A custom header name carrying the raw credential.
    Custom(String),
}

/// A [`ChatModel`] backed by the hosted OpenAI Chat Completions API.
///
/// Construct one with [`OpenAiModel::new`] (plus the `with_*` builders) or
/// [`OpenAiModel::from_env`]. The model holds a reusable [`reqwest::Client`] so
/// repeated calls share a connection pool.
pub struct OpenAiModel {
    /// Shared HTTP client.
    client: reqwest::Client,
    /// API credential; how it is sent is governed by [`Self::auth`].
    api_key: String,
    /// How `api_key` is attached to each request (default [`AuthStyle::Bearer`]).
    auth: AuthStyle,
    /// Extra static headers attached to every request (e.g. provider
    /// attribution headers). Applied after the auth header.
    extra_headers: Vec<(String, String)>,
    /// Default model id used when a request does not override it.
    model: String,
    /// Provider family identifier used in profiles and normalized errors.
    provider: String,
    /// API base URL (no trailing slash); `/chat/completions` is appended.
    base_url: String,
    /// Capability profile derived from the default model id.
    profile: ModelProfile,
}

/// The auth headers `(name, value)` for a given [`AuthStyle`] + credential.
///
/// Pure (no request), so the header mapping is unit-testable without a network
/// round-trip. Applied by [`OpenAiModel::authorized`].
pub(super) fn auth_headers(auth: &AuthStyle, api_key: &str) -> Vec<(String, String)> {
    match auth {
        AuthStyle::None => Vec::new(),
        AuthStyle::Bearer => vec![("Authorization".to_string(), format!("Bearer {api_key}"))],
        AuthStyle::XApiKey => vec![("x-api-key".to_string(), api_key.to_string())],
        AuthStyle::Anthropic => vec![
            ("x-api-key".to_string(), api_key.to_string()),
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ],
        AuthStyle::Custom(name) => vec![(name.clone(), api_key.to_string())],
    }
}

/// Returns `true` for OpenAI o-series reasoning models (`o1`/`o3`/`o4`), which
/// reject `max_tokens` and require `max_completion_tokens` instead.
pub(super) fn is_reasoning_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4")
}

/// Derives a static [`ModelProfile`] for an OpenAI(-compatible) model id.
///
/// All targets support tool calling, streaming (including tool-call chunks),
/// and JSON Schema response formats. Modern OpenAI-family models additionally
/// advertise native structured output and (for the o-series) reasoning output.
///
/// The context window is populated from the provider-neutral
/// [`context_window_for_model_id`][crate::harness::model::context_window_for_model_id]
/// hint table when the id is recognized (`None` otherwise), so downstream
/// context-window-aware trimming/compaction
/// ([`SummarizationPolicy::from_profile`][crate::harness::summarization::SummarizationPolicy::from_profile])
/// engages on a real window instead of silently falling back to a fixed
/// threshold.
pub(super) fn derive_profile(provider: &str, model: &str) -> ModelProfile {
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
        max_input_tokens: crate::harness::model::context_window_for_model_id(model),
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
            auth: AuthStyle::Bearer,
            extra_headers: Vec::new(),
            model: DEFAULT_MODEL.to_string(),
            provider: "openai".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            profile: derive_profile("openai", DEFAULT_MODEL),
        }
    }

    /// Overrides how the API credential is sent (default [`AuthStyle::Bearer`]).
    ///
    /// Use this for OpenAI-compatible endpoints that authenticate with
    /// `x-api-key`, the Anthropic header pair, or a custom header instead of a
    /// bearer token.
    pub fn with_auth_style(mut self, auth: AuthStyle) -> Self {
        self.auth = auth;
        self
    }

    /// Attaches a static header to every request (repeatable). Applied after the
    /// auth header — e.g. provider attribution headers.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
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
            .send_checked(self.authorized(self.client.get(&url)), "request", &url)
            .await?;

        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("openai response body read failed: {e}"))
        })?;

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
    pub(super) fn translate_request(
        &self,
        request: &ModelRequest,
    ) -> Result<ChatCompletionRequest> {
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

    /// Attaches the provider's credential (per [`Self::auth`]) plus any static
    /// [`Self::extra_headers`] to an outbound request.
    ///
    /// The single place auth is applied, shared by the chat and model-listing
    /// calls. The header mapping itself lives in the pure [`auth_headers`] helper
    /// so it is unit-testable without a network round-trip.
    fn authorized(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (name, value) in auth_headers(&self.auth, &self.api_key) {
            builder = builder.header(name, value);
        }
        for (name, value) in &self.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
    }

    /// Sends `builder` and returns the checked (2xx) [`reqwest::Response`].
    ///
    /// The shared transport tail for every OpenAI call: a send/transport failure
    /// is mapped to a [`TinyAgentsError::Model`] describing `what` (e.g.
    /// `"request"`, `"stream request"`) against `url`, and any non-2xx status is
    /// decoded through [`Self::parse_error_body`] into a structured
    /// [`TinyAgentsError::Provider`]. On success the raw response is handed back
    /// so the caller can read it as text (unary/list) or stream its body.
    async fn send_checked(
        &self,
        builder: reqwest::RequestBuilder,
        what: &str,
        url: &str,
    ) -> Result<reqwest::Response> {
        let response = builder.send().await.map_err(|e| {
            let error =
                self.provider_error(format!("{what} to {url} failed: {e}"), None, None, None);
            TinyAgentsError::Model(self.provider_failure_message(&error))
        })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let error = self.parse_error_body(status.as_u16(), &text);
            return Err(TinyAgentsError::Provider(Box::new(error)));
        }
        Ok(response)
    }

    /// Issues an authenticated `POST {base_url}/chat/completions` with `body`,
    /// applying the resolved per-request timeout, and returns the checked
    /// response.
    ///
    /// Shared by the unary ([`Self::invoke`]) and streaming ([`Self::stream`])
    /// paths so URL construction, auth, timeout selection, and transport/status
    /// handling live in exactly one place.
    async fn post_json(
        &self,
        body: &ChatCompletionRequest,
        timeout_ms: Option<u64>,
        streaming: bool,
        what: &str,
    ) -> Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut builder = self.authorized(self.client.post(&url)).json(body);
        if let Some(timeout) = request_timeout(timeout_ms, streaming) {
            builder = builder.timeout(timeout);
        }
        self.send_checked(builder, what, &url).await
    }

    fn provider_error(
        &self,
        message: impl Into<String>,
        status: Option<u16>,
        code: Option<String>,
        raw: Option<Value>,
    ) -> ProviderError {
        let message = message.into();
        let retryable =
            crate::harness::retry::classify_provider_failure(status, code.as_deref(), &message)
                .is_retryable();
        ProviderError {
            provider: self.provider.clone(),
            model: Some(self.model.clone()),
            status,
            code,
            message,
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

    pub(super) fn parse_error_body(&self, status: u16, text: &str) -> ProviderError {
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
pub(super) fn request_timeout(timeout_ms: Option<u64>, streaming: bool) -> Option<Duration> {
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
pub(super) fn provider_extra_options(options: &Value) -> Result<Map<String, Value>> {
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

        let response = self
            .post_json(&body, request.timeout_ms, false, "request")
            .await?;

        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("openai response body read failed: {e}"))
        })?;

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

        let response = self
            .post_json(&body, request.timeout_ms, true, "stream request")
            .await?;

        // Forward each raw chunk as the `bytes::Bytes` buffer reqwest already
        // produced (a cheap refcount clone, no per-chunk copy); only the error
        // type is mapped onto the crate error. `SseState` is crate-internal,
        // so `bytes` never leaks into the public API.
        let bytes = response.bytes_stream().map(|chunk| {
            chunk.map_err(|e| TinyAgentsError::Model(format!("stream chunk failed: {e}")))
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
