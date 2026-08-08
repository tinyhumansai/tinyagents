//! HTTP transport: `OpenAiModel` construction, provider presets, request
//! building, and the `ChatModel` impl (`invoke`/`stream`).
//!
//! Split out of `openai/mod.rs`; see that module's doc comment for the
//! full provider overview.

use super::responses;
use super::*;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::harness::model::StreamAccumulator;

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
    /// Whether the client was supplied by the caller and owns default deadlines.
    caller_owned_client: bool,
    /// API credential; how it is sent is governed by [`Self::auth`].
    api_key: String,
    /// How `api_key` is attached to each request (default [`AuthStyle::Bearer`]).
    auth: AuthStyle,
    /// Extra static headers attached to every request (e.g. provider
    /// attribution headers). Applied after the auth header.
    extra_headers: Vec<(String, String)>,
    /// Model-id glob patterns (`*` wildcard) whose targets reject a `temperature`
    /// parameter; matching requests omit `temperature` entirely. Empty by default.
    temperature_unsupported: Vec<String>,
    /// When set, overrides the request's sampling temperature for every call
    /// (unless the model is in [`Self::temperature_unsupported`]).
    temperature_override: Option<f64>,
    /// When `true`, system messages are folded into the first user message and
    /// the `system` role is dropped — for OpenAI-compatible endpoints that reject
    /// a `system` role. `false` by default (system messages pass through).
    merge_system_into_user: bool,
    /// Whether the endpoint accepts a **named** `tool_choice`
    /// (`{"type":"function","function":{"name":…}}`). `true` by default. When
    /// `false`, a [`ToolChoice::Tool`] request is degraded to
    /// `tool_choice:"required"` with the `tools` array filtered to the named tool
    /// — some local runtimes (LM Studio, llama.cpp server) 400 on the object form.
    /// See [`Self::with_named_tool_choice`].
    ///
    /// An [`AtomicBool`] (like [`Self::stream_required`]) because a 400 that
    /// implicates this shape latches it `false` in [`Self::post_chat_with_degrade`]
    /// so later calls on the same shared `&self` skip the doomed baseline
    /// request instead of re-discovering the rejection every time.
    named_tool_choice_supported: AtomicBool,
    /// Whether the endpoint accepts `response_format:{"type":"json_object"}`.
    /// `true` by default. When `false`, a [`ResponseFormat::JsonObject`] request is
    /// degraded to a permissive `json_schema` wire form — some local runtimes 400
    /// on `json_object`. See [`Self::with_json_object_format`].
    ///
    /// An [`AtomicBool`] for the same reason as
    /// [`Self::named_tool_choice_supported`]: the 400-driven discovery is latched
    /// on the instance instead of thrown away after the retry.
    json_object_format_supported: AtomicBool,
    /// Default model id used when a request does not override it.
    model: String,
    /// Provider family identifier used in profiles and normalized errors.
    provider: String,
    /// API base URL (no trailing slash); `/chat/completions` is appended.
    base_url: String,
    /// Capability profile derived from the default model id, optionally adjusted
    /// by [`Self::with_native_tool_calling`] / [`Self::with_vision`].
    profile: ModelProfile,
    /// Preserve conservative local-runtime tool/vision capability overrides
    /// when callers subsequently replace the default model id.
    local_capabilities_locked: bool,
    /// Provider-specific options baked onto every request (e.g. a local model's
    /// `{"options": {"num_ctx": 8192}}`). Merged under each request's own
    /// `provider_options`, which win on key conflicts. `Value::Null` by default
    /// (no baked options). See [`Self::with_default_provider_options`].
    default_provider_options: Value,
    /// When `true`, calls go to the OpenAI **Responses API** (`/v1/responses`)
    /// instead of Chat Completions. See [`Self::with_responses_api_primary`].
    responses_api_primary: bool,
    /// When `true` (Responses path only), omit `max_output_tokens` from the wire
    /// body — the OpenAI Codex OAuth backend rejects it. See
    /// [`Self::with_responses_omit_max_output_tokens`].
    responses_omit_max_output_tokens: bool,
    /// Static query parameters appended to every request URL (e.g. the Codex
    /// `client_version`). See [`Self::with_extra_query_param`].
    extra_query_params: Vec<(String, String)>,
    /// A `User-Agent` header override (e.g. the Codex CLI UA). `None` uses
    /// reqwest's default. See [`Self::with_user_agent`].
    user_agent: Option<String>,
    /// Inline `<think>…</think>` reasoning-tag extraction config. `Some` moves
    /// inline chain-of-thought onto the reasoning channel; `None` passes content
    /// through untouched. Until [`Self::with_reasoning_tag_extraction`] is
    /// called (`reasoning_tags_overridden == false`) this default only takes
    /// effect for non-hosted base URLs — see
    /// [`Self::effective_reasoning_tags`].
    reasoning_tags: Option<ReasoningTagExtraction>,
    /// Whether [`Self::with_reasoning_tag_extraction`] was called explicitly.
    /// When `false`, extraction auto-enables only for OpenAI-compatible
    /// endpoints (`base_url != DEFAULT_BASE_URL`): hosted OpenAI never emits
    /// inline `<think>` reasoning, and unconditional extraction would silently
    /// strip legitimate content that mentions a literal `<think>` tag.
    reasoning_tags_overridden: bool,
    /// When set, the unary [`ChatModel::invoke`] path sends `stream: true` on
    /// the wire and folds the server-sent-events stream into a single
    /// [`ModelResponse`] internally — for providers that reject a
    /// `stream: false` body (`{"detail":"Stream must be set to true"}` and
    /// friends). Starts `false`; seeded by
    /// [`with_requires_streaming`](Self::with_requires_streaming) and
    /// **latched at run time** the first time the provider rejects a
    /// non-streaming request.
    ///
    /// It is an [`AtomicBool`] rather than a plain `bool` because
    /// [`ChatModel::invoke`] takes `&self` (models are shared behind
    /// `Arc<dyn ChatModel>`): without interior mutability the discovery could
    /// not be remembered, so every single call would keep paying a
    /// guaranteed-400 round trip before falling back. See
    /// [`Self::latch_stream_required`].
    stream_required: AtomicBool,
    /// Whether a JSON-Schema `response_format` is sent with OpenAI **strict**
    /// structured output (`"strict": true`).
    ///
    /// Used to be hardcoded `true` for every endpoint and every schema. Strict
    /// mode is far narrower than JSON Schema — it requires
    /// `additionalProperties: false` on every object and *every* property listed
    /// in `required` — so a caller's ordinary schema 400s. Local runtimes reject
    /// the key outright. Defaults `true` for hosted OpenAI (unchanged wire
    /// shape), `false` for local runtimes, overridable with
    /// [`Self::with_strict_json_schema`], and latched `false` by a 400 that
    /// implicates the schema (see [`degrade_for_400`]).
    json_schema_strict: AtomicBool,
    /// Whether the endpoint accepts native `tools` on the wire.
    ///
    /// This is the **transport** half of the tool decision; the *advertised*
    /// half is [`ModelProfile::tool_calling`]. They used to be the same value,
    /// hard-disabled for every local runtime, which forced prompt-guided tools
    /// on every local call and excluded every local model from any
    /// `CapabilitySet { tool_calling: true }` resolution. Now it is discovered:
    /// seeded optimistically (or from [`Self::probe_local_profile`]) and latched
    /// `false` by a 400 implicating `tools`, exactly like
    /// [`Self::stream_required`]. See [`Self::with_native_tools_on_wire`].
    native_tools_on_wire: AtomicBool,
    /// How the provider counts cache tokens against its input total. See
    /// [`CacheTokenAccounting`].
    cache_accounting: CacheTokenAccounting,
    /// Whether this instance points at a local runtime (Ollama, LM Studio,
    /// llama.cpp server, vLLM, …). Local runtimes get the degrade knobs
    /// pre-set, a native-API escape hatch for `num_ctx`, a `model not found`
    /// error rewrite with real remediation, and are the only targets
    /// [`Self::probe_local_profile`] will probe.
    local_runtime: Option<LocalRuntimeKind>,
    /// Optional `keep_alive` residency hint baked onto every local request. See
    /// [`Self::with_keep_alive`].
    keep_alive: Option<String>,
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

/// Case-insensitive glob match supporting the `*` wildcard (the only metacharacter
/// used by model-id patterns). `"o1*"` matches `"o1-mini"`; `"*turbo"` matches
/// `"gpt-4-turbo"`; a pattern with no `*` matches exactly.
pub(super) fn glob_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.to_ascii_lowercase();
    let value = value.to_ascii_lowercase();
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == value;
    }
    let mut cursor = 0usize;
    for (idx, segment) in segments.iter().enumerate() {
        if segment.is_empty() {
            continue;
        }
        if idx == 0 {
            // A non-empty leading segment must be a prefix.
            if !value[cursor..].starts_with(segment) {
                return false;
            }
            cursor += segment.len();
        } else if idx == segments.len() - 1 {
            // A non-empty trailing segment must be a suffix.
            return value[cursor..].ends_with(segment);
        } else {
            match value[cursor..].find(segment) {
                Some(offset) => cursor += offset + segment.len(),
                None => return false,
            }
        }
    }
    // Trailing `*` (empty last segment) matches the remainder.
    true
}

/// The `temperature` to send for `model`: `None` (omitted) when the model matches
/// a temperature-unsupported pattern, else the override if set, else the request's
/// temperature. Pure, so the policy is unit-testable without a request.
pub(super) fn effective_temperature(
    model: &str,
    request_temperature: Option<f64>,
    temperature_override: Option<f64>,
    temperature_unsupported: &[String],
) -> Option<f64> {
    if temperature_unsupported.iter().any(|p| glob_match(p, model)) {
        return None;
    }
    temperature_override.or(request_temperature)
}

/// Concatenates the text of a message's content blocks (non-text blocks — images,
/// json, thinking — are ignored). Used to gather system-prompt text for merging.
fn content_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Folds all `system` messages into the first `user` message and drops the
/// `system` role, for endpoints that reject a `system` role. The concatenated
/// system text is prefixed to the first user message (`"{system}\n\n{user}"` —
/// matching the common host behavior); image/other user content blocks are
/// preserved. When there is no user message, the system text is promoted to a
/// user turn. Pure, so the transform is unit-testable.
pub(super) fn merge_system_into_user(messages: &[Message]) -> Vec<Message> {
    use crate::harness::message::UserMessage;

    let system_text = messages
        .iter()
        .filter_map(|m| match m {
            Message::System(s) => {
                let t = content_text(&s.content);
                (!t.is_empty()).then_some(t)
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    if system_text.is_empty() {
        // Nothing to fold; drop any (empty) system messages if present, else pass
        // through untouched.
        if messages.iter().any(|m| matches!(m, Message::System(_))) {
            return messages
                .iter()
                .filter(|m| !matches!(m, Message::System(_)))
                .cloned()
                .collect();
        }
        return messages.to_vec();
    }

    let mut merged: Vec<Message> = Vec::with_capacity(messages.len());
    let mut folded = false;
    for msg in messages {
        match msg {
            Message::System(_) => {} // dropped
            Message::User(user) if !folded => {
                folded = true;
                let mut content = Vec::with_capacity(user.content.len() + 1);
                match user.content.split_first() {
                    Some((ContentBlock::Text(first), rest)) => {
                        content.push(ContentBlock::Text(format!("{system_text}\n\n{first}")));
                        content.extend(rest.iter().cloned());
                    }
                    _ => {
                        content.push(ContentBlock::Text(format!("{system_text}\n\n")));
                        content.extend(user.content.iter().cloned());
                    }
                }
                merged.push(Message::User(UserMessage { content }));
            }
            other => merged.push(other.clone()),
        }
    }

    if !folded {
        // No user message to fold into — promote the system text to a user turn.
        merged.insert(0, Message::user(system_text));
    }

    merged
}

/// Returns `true` for OpenAI reasoning models, which reject `max_tokens` and
/// require `max_completion_tokens` instead.
///
/// Covers the o-series (`o1`/`o3`/`o4`) **and the gpt-5 family**. gpt-5 was
/// missing entirely: its cap was routed to `max_tokens`, which OpenAI rejects
/// outright (`Unsupported parameter: 'max_tokens' … use
/// 'max_completion_tokens'`), and it was additionally profiled as
/// `reasoning: false` / `native_structured_output: false`, so a
/// `CapabilitySet { reasoning: true }` filtered it out and structured output
/// picked the tool-call fallback over native schema mode.
pub(super) fn is_reasoning_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || is_gpt5_family(&lower)
}

/// Returns `true` for the gpt-5 family, tolerating the gateway-prefixed ids
/// routers use (`openai/gpt-5-mini` on OpenRouter).
///
/// Takes an already-lowercased id; callers inside this module always have one.
pub(super) fn is_gpt5_family(lower: &str) -> bool {
    lower.starts_with("gpt-5") || lower.starts_with("gpt5") || lower.contains("/gpt-5")
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
    derive_profile_for(provider, model, false)
}

/// [`derive_profile`], with `local` selecting the local-runtime policy.
///
/// The only difference is [`ModelProfile::max_input_tokens`], and it matters a
/// great deal. The generic hint table matches **bare substrings**, so
/// `llama3.2:3b` served by Ollama resolved through `("llama3", Substring,
/// 128_000)` and advertised a 128 000-token window against a server whose
/// default `num_ctx` is 2048. Compaction fires at `window * threshold`, so it
/// never fired and the server truncated the front of the prompt silently.
///
/// A local profile therefore reports `None` — "unknown" — which this crate
/// already supports end to end and which is strictly better than a wrong
/// number. LangChain reaches the same conclusion by shipping no ChatOllama
/// profile at all and hard-failing its summarization middleware with a message
/// telling you to pass absolute token counts. The real answer is to ask the
/// server: [`OpenAiModel::probe_local_profile`].
pub(super) fn derive_profile_for(provider: &str, model: &str, local: bool) -> ModelProfile {
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
        // Every OpenAI reasoning model accepts `reasoning_effort`; models that
        // merely *emit* reasoning (a deepseek-r1 distill leaking `<think>`
        // through a local runtime) do not, which is why this is a separate flag.
        reasoning_effort: reasoning && !local,
        max_input_tokens: if local {
            None
        } else {
            crate::harness::model::context_window_for_model_id(model)
        },
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
            caller_owned_client: false,
            api_key: api_key.into(),
            auth: AuthStyle::Bearer,
            extra_headers: Vec::new(),
            temperature_unsupported: Vec::new(),
            temperature_override: None,
            merge_system_into_user: false,
            named_tool_choice_supported: AtomicBool::new(true),
            json_object_format_supported: AtomicBool::new(true),
            model: DEFAULT_MODEL.to_string(),
            provider: "openai".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            profile: derive_profile("openai", DEFAULT_MODEL),
            local_capabilities_locked: false,
            default_provider_options: Value::Null,
            responses_api_primary: false,
            responses_omit_max_output_tokens: false,
            extra_query_params: Vec::new(),
            user_agent: None,
            // Inline `<think>` extraction defaults ON, but until overridden it
            // only takes effect for non-hosted base URLs (see
            // `effective_reasoning_tags`): unhandled leakage is the common
            // local-model failure, while hosted OpenAI never emits inline
            // `<think>` and must not strip literal mentions of the tag.
            reasoning_tags: Some(ReasoningTagExtraction::default()),
            reasoning_tags_overridden: false,
            stream_required: AtomicBool::new(false),
            json_schema_strict: AtomicBool::new(true),
            native_tools_on_wire: AtomicBool::new(true),
            cache_accounting: CacheTokenAccounting::default(),
            local_runtime: None,
            keep_alive: None,
        }
    }

    /// When the provider requires `stream: true` for every request, routes
    /// unary [`invoke`](ChatModel::invoke) through the streaming path internally.
    ///
    /// Optional: the transport also discovers this on its own (see
    /// [`Self::latch_stream_required`]). Set it explicitly to skip the one
    /// exploratory request that discovery costs.
    pub fn with_requires_streaming(self, enabled: bool) -> Self {
        self.stream_required.store(enabled, Ordering::Relaxed);
        self
    }

    /// Returns whether this instance currently requires streaming for all
    /// unary calls — either declared up front via
    /// [`with_requires_streaming`](Self::with_requires_streaming) or learned
    /// from a provider rejection.
    pub fn requires_streaming(&self) -> bool {
        self.stream_required.load(Ordering::Relaxed)
    }

    /// Remembers that this endpoint only accepts `stream: true`.
    ///
    /// Called once the provider has told us so with an HTTP 400/422 (see
    /// [`is_stream_required_error`]). Without this latch the transport
    /// re-discovers the constraint on **every** unary call, so a streaming-only
    /// proxy costs one guaranteed-400 round trip per agent turn — the shape
    /// behind the 511 events / 2 users on openhuman#5165. Logs on the
    /// transition only, so the discovery is visible exactly once per process.
    pub(super) fn latch_stream_required(&self) {
        if !self.stream_required.swap(true, Ordering::Relaxed) {
            tracing::info!(
                provider = %self.provider,
                model = %self.model,
                "[openai] provider requires stream:true; latching for subsequent calls"
            );
        }
    }

    /// Remembers a newly-discovered request-shape rejection so later calls skip
    /// straight to the degraded body instead of re-paying a guaranteed 400.
    ///
    /// Mirrors [`Self::latch_stream_required`]: `degrade` is the union
    /// [`degrade_for_400`] just computed, so this stores `false` into exactly
    /// the knob(s) that flipped on this call. Logs on each transition only.
    pub(super) fn latch_degrade(&self, degrade: Degrade) {
        if degrade.named_tool_choice
            && self
                .named_tool_choice_supported
                .swap(false, Ordering::Relaxed)
        {
            tracing::info!(
                provider = %self.provider,
                model = %self.model,
                "[openai] provider rejects named tool_choice; latching degraded shape for subsequent calls"
            );
        }
        if degrade.json_object
            && self
                .json_object_format_supported
                .swap(false, Ordering::Relaxed)
        {
            tracing::info!(
                provider = %self.provider,
                model = %self.model,
                "[openai] provider rejects response_format:json_object; latching degraded shape for subsequent calls"
            );
        }
        if degrade.json_schema_strict && self.json_schema_strict.swap(false, Ordering::Relaxed) {
            tracing::info!(
                provider = %self.provider,
                model = %self.model,
                "[openai] provider rejects strict json_schema; latching strict:false for subsequent calls"
            );
        }
        if degrade.native_tools && self.native_tools_on_wire.swap(false, Ordering::Relaxed) {
            tracing::info!(
                provider = %self.provider,
                model = %self.model,
                "[openai] provider rejects native tools; latching prompt-guided tools for subsequent calls"
            );
        }
    }

    /// Routes calls to the OpenAI **Responses API** (`/v1/responses`) instead of
    /// Chat Completions. Required for the OpenAI Codex OAuth backend; pair with
    /// [`with_extra_query_param`](Self::with_extra_query_param) +
    /// [`with_user_agent`](Self::with_user_agent) for Codex.
    pub fn with_responses_api_primary(mut self) -> Self {
        self.responses_api_primary = true;
        self
    }

    /// Omits `max_output_tokens` from Responses requests (the Codex OAuth backend
    /// rejects it). No effect on the Chat Completions path.
    pub fn with_responses_omit_max_output_tokens(mut self) -> Self {
        self.responses_omit_max_output_tokens = true;
        self
    }

    /// Appends a static query parameter to every request URL (repeatable) — e.g.
    /// the Codex `client_version`.
    pub fn with_extra_query_param(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.extra_query_params.push((name.into(), value.into()));
        self
    }

    /// Overrides the `User-Agent` header sent on every request (e.g. the Codex
    /// CLI user agent).
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Reuses a caller-configured HTTP client, including its connection pool,
    /// proxy settings, and request deadlines.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self.caller_owned_client = true;
        self
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

    /// Sets model-id glob patterns (`*` wildcard, case-insensitive) whose targets
    /// reject a `temperature` parameter. A request whose model matches any pattern
    /// omits `temperature` from the wire body (e.g. OpenAI o-series / some hosted
    /// reasoning models that 400 on an explicit temperature).
    pub fn with_temperature_unsupported_models(
        mut self,
        patterns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.temperature_unsupported = patterns.into_iter().map(Into::into).collect();
        self
    }

    /// Overrides the sampling temperature for every request (unless the model is
    /// temperature-unsupported). Useful for endpoints that require a fixed
    /// temperature regardless of the caller's request.
    pub fn with_temperature_override(mut self, temperature: Option<f64>) -> Self {
        self.temperature_override = temperature;
        self
    }

    /// Folds system messages into the first user message and drops the `system`
    /// role, for OpenAI-compatible endpoints that reject a `system` role.
    pub fn with_merge_system_into_user(mut self) -> Self {
        self.merge_system_into_user = true;
        self
    }

    /// Declares whether the endpoint accepts a **named** `tool_choice`
    /// (`{"type":"function","function":{"name":…}}`). `true` by default.
    ///
    /// Pass `false` for local OpenAI-compatible runtimes (LM Studio, llama.cpp
    /// server, and others) that only accept the string forms `none`/`auto`/
    /// `required` and 400 on the object form. A [`ToolChoice::Tool`] request is
    /// then degraded to `tool_choice:"required"` with the wire `tools` array
    /// filtered down to just the named tool, preserving the "must call *this*
    /// tool" semantics. Independent of this flag, a 400 whose body implicates
    /// `tool_choice` triggers the same degraded retry automatically — and, once
    /// discovered, is latched on the instance so later calls skip the doomed
    /// baseline request instead of re-paying the 400 every time.
    pub fn with_named_tool_choice(self, supported: bool) -> Self {
        self.named_tool_choice_supported
            .store(supported, Ordering::Relaxed);
        self
    }

    /// Declares whether the endpoint accepts
    /// `response_format:{"type":"json_object"}`. `true` by default.
    ///
    /// Pass `false` for local OpenAI-compatible runtimes that only accept
    /// `json_schema`/`text` and 400 on `json_object`. A
    /// [`ResponseFormat::JsonObject`] request is then degraded to a permissive
    /// `json_schema` wire form (an empty object schema with `strict:false`).
    /// Independent of this flag, a 400 whose body implicates `response_format`
    /// triggers the same degraded retry automatically — and, once discovered,
    /// is latched on the instance so later calls skip the doomed baseline
    /// request instead of re-paying the 400 every time.
    pub fn with_json_object_format(self, supported: bool) -> Self {
        self.json_object_format_supported
            .store(supported, Ordering::Relaxed);
        self
    }

    /// Configures inline `<think>…</think>` reasoning-tag extraction for
    /// OpenAI-compatible reasoning models that embed chain-of-thought in the
    /// visible `content` string (qwen3, deepseek-r1 distills via Ollama `/v1`,
    /// LM Studio, llama.cpp) rather than on the `reasoning_content` /
    /// `reasoning` side-channel.
    ///
    /// Until this method is called, extraction defaults ON with the plain
    /// `think` tag for **non-hosted base URLs only** (hosted OpenAI never emits
    /// inline `<think>` reasoning, and extraction there would silently strip
    /// legitimate content that mentions a literal tag). Pass `None` to disable
    /// it everywhere (content passes through verbatim), or `Some(config)` to
    /// force it on — including for the hosted base URL — and customize the tag
    /// name, separator, or DeepSeek-R1 template mode via
    /// [`ReasoningTagExtraction`]. Extracted reasoning surfaces as a leading
    /// [`ContentBlock::Thinking`](crate::harness::message::ContentBlock::Thinking)
    /// block on both the unary and streamed paths, consistent with the
    /// side-channel normalization.
    pub fn with_reasoning_tag_extraction(mut self, config: Option<ReasoningTagExtraction>) -> Self {
        self.reasoning_tags = config;
        self.reasoning_tags_overridden = true;
        self
    }

    /// The reasoning-tag extraction config in effect for a call: the explicit
    /// [`Self::with_reasoning_tag_extraction`] override when one was given,
    /// otherwise the built-in `think` default gated to OpenAI-compatible
    /// endpoints (`base_url != DEFAULT_BASE_URL`).
    pub(super) fn effective_reasoning_tags(&self) -> Option<&ReasoningTagExtraction> {
        if !self.reasoning_tags_overridden && self.base_url == DEFAULT_BASE_URL {
            return None;
        }
        self.reasoning_tags.as_ref()
    }

    /// Overrides whether this model advertises **native** tool calling on its
    /// [`profile`](ChatModel::profile). Many self-hosted / local OpenAI-compatible
    /// runtimes (Ollama and others) reject the OpenAI `tools` parameter with an
    /// HTTP 400; passing `false` lets a harness detect that and embed tool specs
    /// in the prompt instead of sending native `tools`. Disabling native tools
    /// also clears `parallel_tool_calls` and `streaming_tool_chunks`, which are
    /// meaningless without them.
    ///
    /// This mutates the derived profile, so apply it **after**
    /// [`with_model`](Self::with_model) / [`with_provider`](Self::with_provider)
    /// (which re-derive the profile).
    pub fn with_native_tool_calling(mut self, enabled: bool) -> Self {
        self.profile.tool_calling = enabled;
        if !enabled {
            self.profile.parallel_tool_calls = false;
            self.profile.streaming_tool_chunks = false;
        }
        self
    }

    /// Overrides whether this model advertises image input (vision) on its
    /// [`profile`](ChatModel::profile). The OpenAI wire preset defaults to `true`;
    /// pass `false` for text-only local/self-hosted models so a harness does not
    /// route image content to an endpoint that cannot accept it.
    ///
    /// This mutates the derived profile, so apply it **after**
    /// [`with_model`](Self::with_model) / [`with_provider`](Self::with_provider).
    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.profile.modalities.image_in = enabled;
        self
    }

    /// Bakes provider-specific options onto every request (e.g. a local model's
    /// `{"options": {"num_ctx": 8192}}`). These are merged **under** each
    /// request's own [`ModelRequest::provider_options`], so a per-call option of
    /// the same key wins. Reserved OpenAI fields are still stripped downstream by
    /// [`provider_extra_options`]. Passing `Value::Null` clears the baked options.
    pub fn with_default_provider_options(mut self, options: Value) -> Self {
        self.default_provider_options = options;
        self
    }

    /// Overrides the default model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.rederive_profile();
        self
    }

    /// Overrides the provider family id used in profiles and normalized errors.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self.rederive_profile();
        self
    }

    /// Re-derives [`Self::profile`] after the model id or provider id changed,
    /// preserving the local-runtime capability overrides when they are locked.
    ///
    /// `max_input_tokens` is part of what is preserved. It used not to be:
    /// [`Self::with_model`] kept the locked tool/vision overrides but
    /// **re-derived** the window from the model id, so the wrong hosted-sized
    /// window survived every model swap on a local handle.
    fn rederive_profile(&mut self) {
        let locked = self.local_capabilities_locked.then_some((
            self.profile.tool_calling,
            self.profile.parallel_tool_calls,
            self.profile.streaming_tool_chunks,
            self.profile.modalities.image_in,
            self.profile.max_input_tokens,
        ));
        self.profile =
            derive_profile_for(&self.provider, &self.model, self.local_runtime.is_some());
        if let Some((
            tool_calling,
            parallel_tool_calls,
            streaming_tool_chunks,
            image_in,
            max_input_tokens,
        )) = locked
        {
            self.profile.tool_calling = tool_calling;
            self.profile.parallel_tool_calls = parallel_tool_calls;
            self.profile.streaming_tool_chunks = streaming_tool_chunks;
            self.profile.modalities.image_in = image_in;
            self.profile.max_input_tokens = max_input_tokens;
        }
    }

    /// Declares whether a JSON-Schema `response_format` is sent with OpenAI
    /// **strict** structured output.
    ///
    /// `true` on hosted OpenAI, `false` on every local runtime (they reject the
    /// key, and LangChain pops it for Ollama for exactly that reason). Strict
    /// mode demands `additionalProperties: false` on every object and every
    /// property listed in `required`, so a caller's ordinary schema 400s under
    /// it; independent of this flag, a 400 implicating the schema degrades to
    /// `strict: false` for a single retry and latches.
    pub fn with_strict_json_schema(self, strict: bool) -> Self {
        self.json_schema_strict.store(strict, Ordering::Relaxed);
        self
    }

    /// Declares whether the endpoint accepts native `tools` on the wire.
    ///
    /// Distinct from [`with_native_tool_calling`](Self::with_native_tool_calling),
    /// which changes what the *profile advertises* (and therefore what
    /// [`CapabilitySet`][cs] resolution will accept). This one changes only
    /// which branch the transport takes. Pass `false` for a server known to 400
    /// on `tools`; otherwise leave it alone — a 400 implicating `tools` flips it
    /// automatically and latches, so the discovery is paid once per process.
    ///
    /// [cs]: crate::harness::model::CapabilitySet
    pub fn with_native_tools_on_wire(self, enabled: bool) -> Self {
        self.native_tools_on_wire.store(enabled, Ordering::Relaxed);
        self
    }

    /// Whether native `tools` go on the wire for the next call: the model must
    /// advertise tool calling **and** the transport must not have latched a
    /// rejection.
    pub fn native_tools_enabled(&self) -> bool {
        self.profile.tool_calling && self.native_tools_on_wire.load(Ordering::Relaxed)
    }

    /// Declares how the provider counts cache tokens against its input total.
    ///
    /// Defaults to OpenAI semantics ([`CacheTokenAccounting::IncludedInInput`]).
    /// Set [`CacheTokenAccounting::ExcludedFromInput`] for a gateway that
    /// forwards Anthropic's convention, where `input_tokens` omits cache reads
    /// and writes — assuming OpenAI semantics over Anthropic data silently
    /// under-bills.
    pub fn with_cache_token_accounting(mut self, accounting: CacheTokenAccounting) -> Self {
        self.cache_accounting = accounting;
        self
    }

    /// Bakes a `keep_alive` residency hint onto this local handle.
    ///
    /// Ollama unloads an idle model after 5 minutes by default, so the next turn
    /// pays a cold multi-gigabyte load inside the 600 s unary deadline. The
    /// value is Ollama's own format (`"30m"`, `"-1"` for forever, `"0"` to
    /// unload immediately) and is delivered by [`Self::warm_up`], which speaks
    /// the native API — `keep_alive` has no OpenAI-wire spelling.
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Requests an explicit context window for the loaded local model.
    ///
    /// Shorthand for the documented `{"options": {"num_ctx": n}}` escape hatch,
    /// merged into [`Self::default_provider_options`], **and** — crucially —
    /// the value [`Self::warm_up`] delivers over the native API, which is the
    /// only path a server actually reads it on. It also updates the advertised
    /// [`ModelProfile::max_input_tokens`], because a window you asked for and a
    /// window you advertise must be the same number or compaction is gated on
    /// fiction.
    pub fn with_local_num_ctx(mut self, num_ctx: u32) -> Self {
        let merged = merge_provider_options(
            &self.default_provider_options,
            &json!({ "options": { "num_ctx": num_ctx } }),
        );
        self.default_provider_options = merged;
        self.profile.max_input_tokens = Some(u64::from(num_ctx));
        self
    }

    /// The local runtime this handle points at, or `None` for a hosted endpoint.
    pub fn local_runtime_kind(&self) -> Option<LocalRuntimeKind> {
        self.local_runtime
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
        let api_key = api_key.into();
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
        // Local runtimes need the same treatment whichever one it is: no
        // Authorization header, a base URL normalised to the `/v1` root, and
        // the request-shape degradations these servers require. Ollama and LM
        // Studio differ only in their default port.
        if let Some(kind) = local_runtime_kind(&spec.kind) {
            let auth = if spec.requires_api_key {
                AuthStyle::Bearer
            } else {
                AuthStyle::None
            };
            return Ok(Self::local_runtime(
                kind,
                &spec.provider,
                normalize_local_v1_base_url(spec.base_url, kind.default_root())?,
                api_key,
                spec.model,
            )
            .with_auth_style(auth));
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
            .send_checked(self.list_models_request(&url), "request", &url)
            .await?;

        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("openai response body read failed: {e}"))
        })?;

        let listing: ModelListWire = serde_json::from_str(&text)?;
        Ok(listing.data)
    }

    // -----------------------------------------------------------------------
    // Local-runtime probing and warm-up (C10 / C11 / C12 / C14)
    // -----------------------------------------------------------------------

    /// Asks a **live local server** what the loaded model can actually do.
    ///
    /// Returns the raw [`LocalProbe`] without touching this handle; use
    /// [`Self::apply_local_probe`] to fold it into the profile, or
    /// [`Self::probed`] to do both in one step.
    ///
    /// # Why this exists
    ///
    /// [`list_models`](Self::list_models) has existed for a while and is called
    /// from nowhere in this crate, and [`ModelListing`] discards everything but
    /// `id`/`created`/`owned_by` — so the two facts that matter most about a
    /// local model were unreachable. This reaches them:
    ///
    /// | Runtime | Endpoint | Yields |
    /// |---|---|---|
    /// | Ollama | `POST {root}/api/show` | `model_info.*.context_length`, `capabilities: [tools, vision, thinking]` |
    /// | LM Studio | `GET {root}/api/v0/models` | `max_context_length`, `loaded_context_length`, `type` |
    /// | others | — | nothing; returns an empty probe rather than an error |
    ///
    /// This is the root fix for three separate defects: the invented context
    /// window, the unconditionally-disabled native tools, and the "only two
    /// runtimes get local treatment" gap.
    ///
    /// # Opt-in on purpose
    ///
    /// It is **never** called during construction. It costs a network round
    /// trip, and a constructor that blocks on one is unusable where this crate
    /// is embedded. Call it once at startup (or lazily on first use) and cache
    /// the result on the handle.
    ///
    /// # Errors
    ///
    /// [`TinyAgentsError::Validation`] when this handle is not a local runtime,
    /// and [`TinyAgentsError::Model`] on transport failure or an undecodable
    /// body. A **non-2xx** status is not an error: it yields an empty probe, so
    /// an older server without the endpoint degrades to "learned nothing"
    /// rather than failing the caller's startup.
    pub async fn probe_local_profile(&self) -> Result<LocalProbe> {
        let Some(kind) = self.local_runtime else {
            return Err(TinyAgentsError::Validation(format!(
                "probe_local_profile is only meaningful for a local runtime; \
                 `{}` at {} is not one",
                self.provider, self.base_url
            )));
        };
        let root = kind.native_root(&self.base_url);
        let (endpoint, builder) = match kind {
            LocalRuntimeKind::Ollama => {
                let url = format!("{root}/api/show");
                let builder = self
                    .authorized(self.client.post(&url))
                    .json(&ollama_show_body(&self.model));
                (url, builder)
            }
            LocalRuntimeKind::LmStudio => {
                let url = format!("{root}/api/v0/models");
                let builder = self.authorized(self.client.get(&url));
                (url, builder)
            }
            // llama.cpp-server and vLLM expose no richer metadata endpoint this
            // crate can rely on. They still get every other piece of local
            // treatment; they simply learn nothing here.
            LocalRuntimeKind::LlamaCpp | LocalRuntimeKind::Vllm => {
                tracing::debug!(
                    provider = %self.provider,
                    kind = kind.as_str(),
                    "[openai] no probe endpoint for this local runtime; returning an empty probe"
                );
                return Ok(LocalProbe::default());
            }
        };

        tracing::debug!(
            provider = %self.provider,
            model = %self.model,
            endpoint = %endpoint,
            "[openai] probing local runtime capabilities"
        );
        let response = builder
            .timeout(PROBE_TIMEOUT)
            .send()
            .await
            .map_err(|e| probe_error(&endpoint, e))?;
        if !response.status().is_success() {
            tracing::debug!(
                provider = %self.provider,
                endpoint = %endpoint,
                status = response.status().as_u16(),
                "[openai] local probe endpoint unavailable; continuing without a probe"
            );
            return Ok(LocalProbe::default());
        }
        let body: Value = response
            .json()
            .await
            .map_err(|e| probe_error(&endpoint, e))?;

        let probe = match kind {
            LocalRuntimeKind::Ollama => probe_from_ollama_show(&body),
            LocalRuntimeKind::LmStudio => probe_from_lm_studio_models(&body, &self.model),
            _ => LocalProbe::default(),
        };
        tracing::info!(
            provider = %self.provider,
            model = %self.model,
            max_input_tokens = ?probe.effective_context_window(),
            tool_calling = ?probe.tool_calling,
            vision = ?probe.vision,
            "[openai] local probe complete"
        );
        Ok(probe)
    }

    /// Folds a [`LocalProbe`] into this handle's profile and transport knobs.
    ///
    /// Only fields the probe actually learned are written; a `None` leaves the
    /// current value alone, so a server that reports nothing cannot erase a
    /// caller's explicit configuration. Pure and synchronous, so the
    /// probe-to-profile policy is unit-testable without a live server.
    pub fn apply_local_probe(mut self, probe: &LocalProbe) -> Self {
        if let Some(window) = probe.effective_context_window() {
            self.profile.max_input_tokens = Some(window);
        }
        if let Some(tool_calling) = probe.tool_calling {
            self.profile.tool_calling = tool_calling;
            self.profile.parallel_tool_calls = tool_calling;
            self.profile.streaming_tool_chunks = tool_calling;
            self.native_tools_on_wire
                .store(tool_calling, Ordering::Relaxed);
        }
        if let Some(vision) = probe.vision {
            self.profile.modalities.image_in = vision;
        }
        if let Some(reasoning) = probe.reasoning {
            self.profile.reasoning = reasoning;
        }
        self
    }

    /// [`probe_local_profile`](Self::probe_local_profile) +
    /// [`apply_local_probe`](Self::apply_local_probe) in one await, for the
    /// common startup shape:
    ///
    /// ```no_run
    /// # use tinyagents::harness::providers::openai::OpenAiModel;
    /// # async fn f() -> tinyagents::Result<()> {
    /// let model = OpenAiModel::ollama().with_model("llama3.2:3b").probed().await?;
    /// # Ok(()) }
    /// ```
    pub async fn probed(self) -> Result<Self> {
        let probe = self.probe_local_profile().await?;
        Ok(self.apply_local_probe(&probe))
    }

    /// Verifies the configured model is actually served, with a remediation
    /// naming the fix.
    ///
    /// The embeddings adapter has always done this ("Run `ollama pull {model}`
    /// or choose an installed embedding model"); the chat path surfaced an
    /// opaque 404 through [`Self::parse_error_body`]. LangChain gates the same
    /// check behind an opt-in flag, and so does this: it costs a round trip, so
    /// it is a method you call rather than something construction does to you.
    ///
    /// # Errors
    ///
    /// [`TinyAgentsError::Validation`] when the model is not among the served
    /// ids, listing what *is* available. Transport failures surface unchanged
    /// from [`Self::list_models`].
    pub async fn validate_model(&self) -> Result<()> {
        let listed = self.list_models().await?;
        if listed.is_empty() {
            // An empty listing is "this server does not enumerate", not
            // "nothing is served". Refusing here would be a false negative.
            tracing::debug!(
                provider = %self.provider,
                "[openai] model listing is empty; skipping model validation"
            );
            return Ok(());
        }
        if listed.iter().any(|entry| entry.id == self.model) {
            return Ok(());
        }
        let mut available: Vec<&str> = listed.iter().map(|entry| entry.id.as_str()).collect();
        available.sort_unstable();
        let remediation = match self.local_runtime {
            Some(LocalRuntimeKind::Ollama) => {
                format!(" Run `ollama pull {}` to install it.", self.model)
            }
            _ => String::new(),
        };
        Err(TinyAgentsError::Validation(format!(
            "{} at {} does not serve model `{}`.{} Available: {}",
            self.provider,
            self.base_url,
            self.model,
            remediation,
            available.join(", ")
        )))
    }

    /// Loads the model with this handle's local options and holds it resident.
    ///
    /// This is the **only** path on which `num_ctx` and `keep_alive` reach the
    /// server. Both are `/api/chat` fields; the chat adapter speaks
    /// `POST {base_url}/chat/completions`, where Ollama's compatibility layer
    /// simply ignores them — so
    /// [`with_default_provider_options`](Self::with_default_provider_options)
    /// documenting `{"options": {"num_ctx": 8192}}` as the local escape hatch
    /// described a field that, on that path, went nowhere. (The existing tests
    /// asserted only that the request JSON *contained* it, never that a server
    /// honoured it, which is how that survived.)
    ///
    /// Call it once before the first real turn. It doubles as the warm-up that
    /// keeps Ollama from unloading after its 5-minute idle default and charging
    /// the next turn a cold multi-gigabyte load inside the 600 s unary deadline.
    ///
    /// **Scope, stated plainly:** this configures the loaded *runner*, which
    /// Ollama then reuses for subsequent `/v1` requests that do not demand
    /// conflicting options. That is a property of the server's runner reuse, not
    /// a guarantee of the OpenAI wire format. A full native `/api/chat` chat
    /// adapter is the complete fix and remains a follow-up.
    ///
    /// A no-op (returning `Ok(())`) for a runtime with no native API.
    ///
    /// # Errors
    ///
    /// [`TinyAgentsError::Model`] on transport failure. A non-2xx status is
    /// logged and swallowed: a warm-up that the server declined must not fail
    /// the caller's startup.
    pub async fn warm_up(&self) -> Result<()> {
        let Some(kind) = self.local_runtime.filter(|k| k.has_native_api()) else {
            return Ok(());
        };
        let url = format!("{}/api/chat", kind.native_root(&self.base_url));
        let body = ollama_load_body(
            &self.model,
            local_options_object(&self.default_provider_options),
            self.keep_alive.as_deref(),
        );
        tracing::debug!(
            provider = %self.provider,
            model = %self.model,
            url = %url,
            num_ctx = ?body.pointer("/options/num_ctx"),
            keep_alive = ?self.keep_alive,
            "[openai] warming up local runtime over the native API"
        );
        let response = self
            .authorized(self.client.post(&url))
            .json(&body)
            .timeout(
                self.effective_request_timeout(None, false)
                    .unwrap_or(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)),
            )
            .send()
            .await
            .map_err(|e| {
                TinyAgentsError::Model(format!("[openai] warm-up of {url} failed: {e}"))
            })?;
        if !response.status().is_success() {
            tracing::warn!(
                provider = %self.provider,
                url = %url,
                status = response.status().as_u16(),
                "[openai] local warm-up declined; continuing without it"
            );
        }
        Ok(())
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
    /// `llama3.2`.
    pub fn ollama() -> Self {
        Self::ollama_at("http://localhost:11434", "llama3.2")
            .expect("the built-in Ollama URL is valid")
    }

    /// An Ollama server exposed through its OpenAI-compatible HTTP API.
    pub fn ollama_at(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let kind = LocalRuntimeKind::Ollama;
        Ok(Self::local_runtime(
            kind,
            "ollama",
            normalize_local_v1_base_url(base_url.into(), kind.default_root())?,
            "",
            model,
        ))
    }

    /// A llama.cpp `llama-server` exposed through its OpenAI-compatible HTTP
    /// API (default root `http://localhost:8080`).
    ///
    /// llama.cpp-server used to fall through to the hosted `Compatible` path,
    /// which gave it Bearer auth, a full hosted capability profile, no `/v1`
    /// normalisation, and none of the request-shape degrade knobs — despite
    /// being the runtime the `json_object` and named-`tool_choice` degrades were
    /// written for.
    pub fn llama_cpp(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let kind = LocalRuntimeKind::LlamaCpp;
        Ok(Self::local_runtime(
            kind,
            kind.as_str(),
            normalize_local_v1_base_url(base_url.into(), kind.default_root())?,
            "",
            model,
        ))
    }

    /// A vLLM OpenAI-compatible server (default root `http://localhost:8000`).
    ///
    /// Same rationale as [`Self::llama_cpp`]: a self-hosted server is a local
    /// runtime whatever its name, and gets the local treatment.
    pub fn vllm(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        let kind = LocalRuntimeKind::Vllm;
        let api_key = api_key.into();
        let auth = if api_key.trim().is_empty() {
            AuthStyle::None
        } else {
            AuthStyle::Bearer
        };
        Ok(Self::local_runtime(
            kind,
            kind.as_str(),
            normalize_local_v1_base_url(base_url.into(), kind.default_root())?,
            api_key,
            model,
        )
        .with_auth_style(auth))
    }

    /// An LM Studio server exposed through its OpenAI-compatible HTTP API.
    ///
    /// Authentication is disabled when `api_key` is empty and uses a bearer
    /// token otherwise, matching LM Studio's optional API-token mode.
    pub fn lm_studio(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        let api_key = api_key.into();
        let auth = if api_key.trim().is_empty() {
            AuthStyle::None
        } else {
            AuthStyle::Bearer
        };
        let kind = LocalRuntimeKind::LmStudio;
        Ok(Self::local_runtime(
            kind,
            kind.as_str(),
            normalize_local_v1_base_url(base_url.into(), kind.default_root())?,
            api_key,
            model,
        )
        .with_auth_style(auth))
    }

    /// The shared local-runtime preset.
    ///
    /// Three things changed here, each fixing a defect the old preset baked in:
    ///
    /// * **`tool_calling` stays `true`.** It used to be hard-disabled for every
    ///   local runtime unconditionally, which (a) forced the prompt-guided
    ///   branch on every call — injecting the protocol block plus every tool's
    ///   serialized JSON Schema into the system prompt, against a real 2048-token
    ///   window, which then truncated from the front and dropped the very prompt
    ///   carrying the protocol — and (b) cleared `parallel_tool_calls` and
    ///   `streaming_tool_chunks`, so any `CapabilitySet { tool_calling: true }`
    ///   excluded **every** local model from resolution. LangChain's ChatOllama
    ///   sends tools natively with no capability check at all. The pessimism is
    ///   replaced by discovery: [`Self::probe_local_profile`] asks the server,
    ///   and a 400 implicating `tools` latches the transport onto the
    ///   prompt-guided branch for the rest of the process.
    /// * **The window is `None`, not a guess.** See [`derive_profile_for`].
    /// * **The degrade knobs are pre-set.** `named_tool_choice` and
    ///   `json_object` are the two shapes the module doc and README name as
    ///   "local servers reject this"; auto-degrade recovered from them, but only
    ///   after one wasted 400 per process — precisely the cost the
    ///   `stream_required` latch exists to eliminate. `strict` JSON Schema is
    ///   off for the same reason.
    fn local_runtime(
        kind: LocalRuntimeKind,
        provider: &str,
        base_url: String,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let mut model = Self::compatible_provider(provider, api_key, base_url, model)
            .with_auth_style(AuthStyle::None)
            .with_vision(false)
            .with_named_tool_choice(false)
            .with_json_object_format(false)
            .with_strict_json_schema(false);
        model.local_runtime = Some(kind);
        // Re-derive **before** locking. The lock preserves the profile values it
        // finds, and at this point they are still the hosted ones the
        // `compatible_provider` call above produced — locking first would
        // preserve exactly the invented context window this preset exists to
        // remove (`llama3.2` → 128 000 from the generic hint table).
        model.rederive_profile();
        model.profile.modalities.image_in = false;
        model.local_capabilities_locked = true;
        model
    }

    #[cfg(test)]
    pub(super) fn auth_config(&self) -> (&str, &AuthStyle) {
        (&self.api_key, &self.auth)
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

    /// The baseline request-shape degradations to apply for this instance,
    /// derived from its capability knobs. A `true` field means "degrade this
    /// shape on the wire".
    pub(super) fn baseline_degrade(&self) -> Degrade {
        Degrade {
            named_tool_choice: !self.named_tool_choice_supported.load(Ordering::Relaxed),
            json_object: !self.json_object_format_supported.load(Ordering::Relaxed),
            json_schema_strict: !self.json_schema_strict.load(Ordering::Relaxed),
            native_tools: !self.native_tools_enabled(),
        }
    }

    /// Translates a provider-neutral [`ModelRequest`] into the OpenAI wire
    /// request body, applying this instance's baseline request-shape
    /// degradations (see [`Self::baseline_degrade`]).
    ///
    /// Test-only: production paths go through [`Self::build_chat_body`] /
    /// [`Self::post_chat_with_degrade`], which thread an explicit [`Degrade`].
    #[cfg(test)]
    pub(super) fn translate_request(
        &self,
        request: &ModelRequest,
    ) -> Result<ChatCompletionRequest> {
        self.translate_request_with(request, self.baseline_degrade())
    }

    /// Translates a provider-neutral [`ModelRequest`] into the OpenAI wire
    /// request body, applying the given request-shape `degrade`. The per-request
    /// `model` override wins over the instance default.
    pub(super) fn translate_request_with(
        &self,
        request: &ModelRequest,
        degrade: Degrade,
    ) -> Result<ChatCompletionRequest> {
        // Prompt-guided tools: a model without native tool calling that is still
        // handed tools gets the tool protocol embedded in its system prompt and no
        // native `tools` on the wire (many local runtimes 400 on `tools`). The
        // model's `<tool_call>` blocks are parsed back in [`Self::invoke`]/stream.
        // Native tools go on the wire unless the profile says the model has none
        // *or* this attempt is degrading them away. `degrade.native_tools`
        // carries both the instance latch (via `baseline_degrade`) and a
        // freshly-discovered 400, so one condition covers both.
        let native_tools = !degrade.native_tools;
        let prompt_guided_tools = !native_tools && !request.tools.is_empty();
        let prompt_messages;
        let instructed_messages;
        let coalesced_messages: &[Message] = if prompt_guided_tools {
            prompt_messages = crate::harness::tool::coalesce_prompt_tool_results(&request.messages);
            instructed_messages = crate::harness::tool::with_prompt_tool_instructions(
                &prompt_messages,
                &request.tools,
            );
            &instructed_messages
        } else {
            &request.messages
        };

        // A model with no native tool calling is rendered through its own chat
        // template by the serving runtime, and templates like Qwen 3's abort the
        // request outright when they cannot locate a user query. Guarantee one
        // (openhuman#5291). Scoped to the no-native-tools profile — that is the
        // set of models served through a Jinja template with this guard — and
        // applied after the coalescing above so folded tool results are already
        // in their final user-turn shape when we look for a real query.
        let user_normalized_messages;
        let base_messages: &[Message] = if native_tools {
            coalesced_messages
        } else {
            user_normalized_messages =
                crate::harness::tool::ensure_resolvable_user_turn(coalesced_messages);
            &user_normalized_messages
        };

        // Optionally fold system messages into the first user turn (for endpoints
        // that reject a `system` role) before wire translation.
        let merged_messages;
        let source_messages: &[Message] = if self.merge_system_into_user {
            merged_messages = merge_system_into_user(base_messages);
            &merged_messages
        } else {
            base_messages
        };
        let messages = source_messages
            .iter()
            .map(translate_message)
            .collect::<Result<Vec<_>>>()?;

        // Project the declarations through the shared preparation seam rather
        // than shipping `schema.parameters` verbatim: a tool whose `parameters`
        // is `Value::Null` (the type permits it) otherwise serialises as
        // `"parameters": null`, which every provider 400s on, and local `$ref`s
        // are resolved for routes that cannot follow them.
        //
        // Deliberately **not** `.with_strict()`. The strict sanitizer forces
        // every declared property into `required`, which changes the contract
        // the model is given — an optional argument becomes mandatory. That is
        // only correct when the wire actually carries `strict: true`, and this
        // adapter does not set it on the tool object (it is a `response_format`
        // concern here). Applying it anyway would silently make every optional
        // tool argument required.
        let preparation = crate::harness::tool::SchemaPreparation::openai();
        let mut tools: Vec<ToolWire> = if prompt_guided_tools {
            Vec::new()
        } else {
            crate::harness::tool::prepare_tool_schemas(&request.tools, &preparation)
                .into_iter()
                .map(|schema| ToolWire {
                    kind: "function".to_string(),
                    function: FunctionSchemaWire {
                        name: schema.name,
                        description: schema.description,
                        parameters: schema.parameters,
                    },
                })
                .collect()
        };

        // tool_choice is only meaningful when tools are declared.
        let tool_choice = if tools.is_empty() {
            None
        } else if let (true, ToolChoice::Tool(name)) =
            (degrade.named_tool_choice, &request.tool_choice)
        {
            // The endpoint rejects a named `tool_choice` object. Preserve the
            // "must call *this* tool" semantics by sending `"required"` and, when
            // the named tool is actually declared, filtering the wire `tools`
            // down to just it so the model has no other tool to pick. If the named
            // tool is absent, leave `tools` intact (mirrors the un-degraded path,
            // which would also send an unmatched name) and still send "required".
            if tools.iter().any(|t| t.function.name == *name) {
                tools.retain(|t| t.function.name == *name);
            }
            Some(json!("required"))
        } else {
            Some(translate_tool_choice(&request.tool_choice))
        };

        let response_format = request.response_format.as_ref().and_then(|format| {
            if degrade.json_object && matches!(format, ResponseFormat::JsonObject) {
                // The endpoint rejects `{"type":"json_object"}`; use a permissive
                // `json_schema` that still guarantees a JSON object.
                Some(degraded_json_object_format())
            } else {
                translate_response_format(format, !degrade.json_schema_strict)
            }
        });

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

        let temperature = effective_temperature(
            &model,
            request.temperature,
            self.temperature_override,
            &self.temperature_unsupported,
        );

        // Provider options are the escape hatch and win on key conflicts, so a
        // caller who spelled `reasoning_effort` there suppresses the typed
        // field entirely — emitting both would put the key on the wire twice.
        let merged_provider_options =
            merge_provider_options(&self.default_provider_options, &request.provider_options);
        let reasoning_effort = if merged_provider_options
            .get("reasoning_effort")
            .is_some_and(|v| !v.is_null())
        {
            None
        } else {
            request
                .reasoning
                .as_ref()
                .and_then(|config| config.effort)
                .map(|effort| effort.as_str().to_string())
        };

        Ok(ChatCompletionRequest {
            model,
            messages,
            tools,
            tool_choice,
            response_format,
            temperature,
            top_p: request.top_p,
            max_tokens,
            max_completion_tokens,
            stop: request.stop_sequences.clone(),
            seed: request.seed,
            reasoning_effort,
            stream: false,
            stream_options: None,
            extra: provider_extra_options(&merged_provider_options)?,
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
        if let Some(user_agent) = &self.user_agent {
            builder = builder.header(reqwest::header::USER_AGENT, user_agent.as_str());
        }
        if !self.extra_query_params.is_empty() {
            builder = builder.query(&self.extra_query_params);
        }
        builder
    }

    /// Builds the authorized GET request for [`Self::list_models`], applying
    /// the same deadline policy as every other outbound call.
    ///
    /// Every other outbound path (`post_json`, `send_responses`) resolves a
    /// deadline through [`Self::effective_request_timeout`]; `list_models` used
    /// not to, so a reachable-but-wedged endpoint (an Ollama/LM Studio server
    /// that accepts the TCP connect and then never responds — exactly the
    /// runtime model discovery this method exists for) hung the call forever.
    /// Factored out (rather than inlined in `list_models`) so the timeout
    /// policy is unit-testable via [`reqwest::RequestBuilder::build`] without a
    /// network round trip.
    pub(super) fn list_models_request(&self, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self.authorized(self.client.get(url));
        if let Some(timeout) = self.effective_request_timeout(None, false) {
            builder = builder.timeout(timeout);
        }
        builder
    }

    /// The `/responses` endpoint URL — a sibling of `/chat/completions` under the
    /// same base URL, tolerating a base that already ends in `/responses` or a
    /// `…/v1` chat base.
    fn responses_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/responses") {
            base.to_string()
        } else {
            format!("{base}/responses")
        }
    }

    /// Builds the `/v1/responses` request body from a provider-neutral request.
    ///
    /// Carries the **whole** request. The previous version built only
    /// `{model, input, instructions, stream, store, max_output_tokens}` and
    /// dropped tools, tool choice, response format, sampling, stop sequences,
    /// seed, the continuation id, and `provider_options` — which is how
    /// `reasoning` became unreachable on the one wire format that supports it.
    fn translate_responses_request(&self, request: &ModelRequest) -> responses::ResponsesRequest {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let (instructions, input) = responses::build_responses_input(&request.messages);
        let max_output_tokens = if self.responses_omit_max_output_tokens {
            None
        } else {
            request.max_tokens
        };
        let strict = self.json_schema_strict.load(Ordering::Relaxed);

        let extra = provider_extra_options(&merge_provider_options(
            &self.default_provider_options,
            &request.provider_options,
        ))
        .unwrap_or_default();
        // Same precedence rule as the Chat Completions path: a `reasoning` key
        // in `provider_options` is the escape hatch and wins, so the typed
        // field stands down rather than emitting the key twice.
        let reasoning = if extra.contains_key("reasoning") {
            None
        } else {
            request
                .reasoning
                .as_ref()
                .and_then(responses::translate_reasoning)
        };
        // Reasoning replay is only possible under `store: false` when the
        // reasoning item carries `encrypted_content`, and that only arrives when
        // it is explicitly requested. Asking for reasoning without asking for
        // this yields reasoning that cannot survive to the next turn.
        let include = if reasoning.is_some() {
            vec![responses::INCLUDE_ENCRYPTED_REASONING.to_string()]
        } else {
            Vec::new()
        };

        let tools = if self.native_tools_enabled() {
            request
                .tools
                .iter()
                .map(responses::translate_tool)
                .collect()
        } else {
            Vec::new()
        };
        let tool_choice = (!tools.is_empty()).then(|| translate_tool_choice(&request.tool_choice));

        responses::ResponsesRequest {
            model,
            input,
            instructions,
            stream: None,
            store: Some(false),
            max_output_tokens,
            tools,
            tool_choice,
            text: request
                .response_format
                .as_ref()
                .and_then(|format| responses::translate_text_format(format, strict)),
            temperature: effective_temperature(
                &self.model,
                request.temperature,
                self.temperature_override,
                &self.temperature_unsupported,
            ),
            top_p: request.top_p,
            seed: request.seed,
            stop: request.stop_sequences.clone(),
            previous_response_id: request.continuation_id.clone(),
            reasoning,
            include,
            // `provider_options` is the escape hatch and wins on key conflicts,
            // exactly as `merge_provider_options` already arranges for the Chat
            // Completions path. It was dropped entirely here.
            extra,
        }
    }

    /// Issues a `POST` to the Responses endpoint and maps the body onto a
    /// [`ModelResponse`]. On a 400 whose body implicates `max_output_tokens`, the
    /// request is retried once without that field (some `/responses` backends
    /// reject the cap outright).
    async fn invoke_responses(&self, request: &ModelRequest) -> Result<ModelResponse> {
        let url = self.responses_url();
        let body = self.translate_responses_request(request);
        let response = match self.send_responses(&body, request.timeout_ms, &url).await {
            Ok(r) => r,
            Err(TinyAgentsError::Provider(err))
                if err.status == Some(400)
                    && body.max_output_tokens.is_some()
                    && err.message.contains("max_output_tokens") =>
            {
                // Retry once without the cap.
                let retry = responses::ResponsesRequest {
                    max_output_tokens: None,
                    ..body
                };
                self.send_responses(&retry, request.timeout_ms, &url)
                    .await?
            }
            Err(e) => return Err(e),
        };
        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("openai responses body read failed: {e}"))
        })?;
        let value: Value = serde_json::from_str(&text)?;
        Ok(responses::parse_responses_response(value))
    }

    /// Shared `POST {responses_url}` with auth, query params, and timeout, mapped
    /// through the same checked-transport tail as chat calls.
    async fn send_responses(
        &self,
        body: &responses::ResponsesRequest,
        timeout_ms: Option<u64>,
        url: &str,
    ) -> Result<reqwest::Response> {
        let mut builder = self.authorized(self.client.post(url)).json(body);
        if let Some(timeout) = self.effective_request_timeout(timeout_ms, false) {
            builder = builder.timeout(timeout);
        }
        self.send_checked(builder, "responses request", url).await
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
            // Read `Retry-After` *before* the body is consumed: a 429/503 that
            // names how long to wait is authoritative, and retrying sooner
            // burns an attempt for certain.
            let retry_after_ms = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(parse_retry_after_header_ms);
            let text = response.text().await.unwrap_or_default();
            let mut error = self.parse_error_body(status.as_u16(), &text);
            if let Some(ms) = retry_after_ms {
                tracing::debug!(
                    provider = %self.provider,
                    status = status.as_u16(),
                    retry_after_ms = ms,
                    "[openai] honoring a server-supplied Retry-After header"
                );
                error.retry_after_ms = Some(ms);
            }
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
        if let Some(timeout) = self.effective_request_timeout(timeout_ms, streaming) {
            builder = builder.timeout(timeout);
        }
        self.send_checked(builder, what, &url).await
    }

    pub(super) fn effective_request_timeout(
        &self,
        timeout_ms: Option<u64>,
        streaming: bool,
    ) -> Option<Duration> {
        if self.caller_owned_client && timeout_ms.is_none() {
            None
        } else {
            request_timeout(timeout_ms, streaming)
        }
    }

    /// Builds the chat-completions wire body for `request` under the given
    /// `degrade`, setting the streaming fields when `streaming` is `true`.
    fn build_chat_body(
        &self,
        request: &ModelRequest,
        degrade: Degrade,
        streaming: bool,
    ) -> Result<ChatCompletionRequest> {
        let mut body = self.translate_request_with(request, degrade)?;
        if streaming {
            body.stream = true;
            body.stream_options = Some(json!({ "include_usage": true }));
        }
        Ok(body)
    }

    /// Posts a chat-completions request with automatic single-shot degraded
    /// retry for local-server request-shape rejections.
    ///
    /// The first attempt applies this instance's baseline degradations. If it
    /// fails with an HTTP 400 whose body implicates a named `tool_choice` or a
    /// `json_object` `response_format` that the request actually used — and that
    /// shape was not already degraded — the request is rebuilt with that shape
    /// degraded and sent exactly once more. Any other error (or a request that
    /// used neither shape) surfaces unchanged. Shared by [`Self::invoke`] and
    /// [`Self::stream`], so the retry covers the streaming path too.
    async fn post_chat_with_degrade(
        &self,
        request: &ModelRequest,
        streaming: bool,
        what: &str,
    ) -> Result<reqwest::Response> {
        let baseline = self.baseline_degrade();
        let body = self.build_chat_body(request, baseline, streaming)?;
        match self
            .post_json(&body, request.timeout_ms, streaming, what)
            .await
        {
            Ok(response) => Ok(response),
            Err(TinyAgentsError::Provider(err))
                if err.status == Some(400)
                    && let Some(degrade) = degrade_for_400(&err.message, request, baseline) =>
            {
                self.latch_degrade(degrade);
                let retry = self.build_chat_body(request, degrade, streaming)?;
                self.post_json(&retry, request.timeout_ms, streaming, what)
                    .await
            }
            Err(e) => Err(e),
        }
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

    /// Decodes a non-2xx body into a structured [`ProviderError`].
    ///
    /// Two classifications are applied on top of the raw decode:
    ///
    /// * **Local missing-model 404s** are rewritten with a remediation naming
    ///   the fix (`ollama pull …`), matching what the embeddings adapter has
    ///   always done. The chat path used to surface the server's bare
    ///   `{"error":"model 'x' not found"}`.
    /// * **Context overflow** is stamped with a stable
    ///   [`CONTEXT_OVERFLOW_CODE`], so a caller can compact and retry on a code
    ///   instead of string-matching a provider message. A typed
    ///   `TinyAgentsError` variant would live in `src/error.rs`; promoting the
    ///   code to one is a follow-up.
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
        let mut code = error_obj
            .and_then(|error| error.get("code").or_else(|| error.get("type")))
            .and_then(Value::as_str)
            .map(str::to_string);

        let message = match self.local_runtime.and_then(|kind| {
            missing_model_remediation(kind, status, text, &self.model, &self.base_url)
        }) {
            Some(remediated) => {
                tracing::info!(
                    provider = %self.provider,
                    model = %self.model,
                    "[openai] rewrote a local missing-model 404 with remediation"
                );
                remediated
            }
            None => message,
        };

        if is_context_overflow(status, &message) {
            tracing::warn!(
                provider = %self.provider,
                model = %self.model,
                max_input_tokens = ?self.profile.max_input_tokens,
                "[openai] classified failure as a context overflow"
            );
            code = Some(CONTEXT_OVERFLOW_CODE.to_string());
        }

        self.provider_error(message, Some(status), code, raw)
    }
}

/// Parses an HTTP `Retry-After` header value into a delay in milliseconds.
///
/// [RFC 9110 §10.2.3] defines two forms and both appear in the wild:
///
/// * **delta-seconds** — `Retry-After: 30`. Used by OpenAI, Anthropic, and most
///   gateways on a 429.
/// * **HTTP-date** — `Retry-After: Wed, 21 Oct 2015 07:28:00 GMT`. Used by some
///   CDNs and proxies fronting a provider, and by 503 maintenance responses.
///   Converted to a delay relative to *now*; a date already in the past yields
///   `0` (retry immediately), never a negative or wrapped value.
///
/// Returns `None` for an unparseable value rather than guessing, so the caller
/// falls through to the existing exponential backoff.
///
/// [RFC 9110 §10.2.3]: https://www.rfc-editor.org/rfc/rfc9110#field.retry-after
pub(super) fn parse_retry_after_header_ms(raw: &str) -> Option<u64> {
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    // delta-seconds. Fractional seconds are not in the grammar but some
    // providers send them, so accept a float too.
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(seconds.saturating_mul(1000));
    }
    if let Ok(seconds) = value.parse::<f64>()
        && seconds.is_finite()
        && seconds >= 0.0
    {
        return Some((seconds * 1000.0) as u64);
    }
    // HTTP-date. `parse_from_rfc2822` covers the IMF-fixdate form (including
    // the obsolete `GMT` zone token); the explicit format is the fallback for
    // servers that omit the day-of-week comma.
    let parsed = chrono::DateTime::parse_from_rfc2822(value)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT")
                .map(|naive| naive.and_utc())
        })
        .ok()?;
    let delta = parsed.signed_duration_since(chrono::Utc::now());
    Some(delta.num_milliseconds().max(0) as u64)
}

/// Maps a [`ProviderKind`] onto the [`LocalRuntimeKind`] it denotes, or `None`
/// for a hosted provider.
///
/// The single place that decides "is this kind a local runtime?", so a new local
/// provider is one arm here rather than a condition to keep in sync across the
/// transport. It used to recognise only Ollama and LM Studio, leaving
/// llama.cpp-server and vLLM to be constructed as [`ProviderKind::Compatible`]
/// — `requires_api_key: true`, Bearer auth, a full hosted profile, no `/v1`
/// normalisation, and no degrade knobs.
pub(super) fn local_runtime_kind(
    kind: &crate::harness::providers::ProviderKind,
) -> Option<LocalRuntimeKind> {
    use crate::harness::providers::ProviderKind;
    match kind {
        ProviderKind::Ollama => Some(LocalRuntimeKind::Ollama),
        ProviderKind::LmStudio => Some(LocalRuntimeKind::LmStudio),
        ProviderKind::LlamaCpp => Some(LocalRuntimeKind::LlamaCpp),
        ProviderKind::Vllm => Some(LocalRuntimeKind::Vllm),
        _ => None,
    }
}

fn normalize_local_v1_base_url(raw: String, default_root: &str) -> Result<String> {
    let trimmed = raw.trim().trim_end_matches('/');
    let root = if trimmed.is_empty() {
        default_root.to_owned()
    } else if trimmed.contains("://") {
        trimmed.to_owned()
    } else {
        format!("http://{trimmed}")
    };
    let mut url = reqwest::Url::parse(&root).map_err(|error| {
        TinyAgentsError::Validation(format!("invalid local runtime URL `{root}`: {error}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(TinyAgentsError::Validation(format!(
            "local runtime URL must use http or https, got `{}`",
            url.scheme()
        )));
    }
    let mut segments: Vec<&str> = url
        .path()
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    if segments.ends_with(&["chat", "completions"]) {
        segments.truncate(segments.len() - 2);
    } else if segments.last() == Some(&"models") {
        segments.pop();
    }
    if segments.last() != Some(&"v1") {
        segments.push("v1");
    }
    url.set_path(&format!("/{}", segments.join("/")));
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

/// Request-shape degradations to apply when building an OpenAI wire body.
///
/// Each field, when `true`, replaces a request shape that some local
/// OpenAI-compatible servers reject with an equivalent one they accept. The
/// baseline comes from the instance's capability knobs
/// ([`OpenAiModel::baseline_degrade`]) from the instance's capability knobs; a
/// 400 whose error body implicates one of these shapes turns the corresponding
/// field on for a single degraded retry (see [`degrade_for_400`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Degrade {
    /// Degrade a named `tool_choice` to `"required"` + a filtered `tools` array.
    pub named_tool_choice: bool,
    /// Degrade `response_format:{"type":"json_object"}` to a permissive
    /// `json_schema`.
    pub json_object: bool,
    /// Degrade a `json_schema` `response_format` from `strict: true` to
    /// `strict: false`.
    ///
    /// A `JsonSchema` request had **no** degradation path at all: only
    /// `JsonObject` was considered, so a 400 on a strict-schema request was
    /// terminal even though the retry that fixes it is one boolean away.
    pub json_schema_strict: bool,
    /// Drop native `tools` from the wire and embed the tool protocol in the
    /// prompt instead.
    ///
    /// The third latch, and the one that replaces "every local runtime is
    /// assumed to have no tool support, forever". Seeded from the probe (or
    /// optimistically `false` = "send tools"), flipped by a 400 implicating
    /// `tools`, and retried once prompt-guided. Follows the `stream_required`
    /// latch pattern exactly.
    pub native_tools: bool,
}

/// Statuses an OpenAI-compatible proxy uses to reject a non-streaming request.
///
/// `400` is what the reference implementation returns. `422` is included because
/// the wording seen in the wild arrives in a FastAPI `{"detail": …}` envelope,
/// and FastAPI's own rejection status is `422` — a proxy that validates `stream`
/// as a request field rather than raising an explicit `HTTPException(400)` lands
/// there. The stream-specific wording still has to match, so widening the status
/// set does not widen the false-positive surface meaningfully.
const STREAM_REQUIRED_STATUSES: [u16; 2] = [400, 422];

/// Recognises "this endpoint only accepts `stream: true`" from a provider error.
///
/// Some OpenAI-compatible proxies refuse unary calls outright. The wording is
/// not standardised, so match a family of phrasings case-insensitively rather
/// than one literal string:
///
/// - `{"detail":"Stream must be set to true"}` — the openhuman#5165 report
/// - `stream must be true` / `'stream' must be set to true`
/// - `streaming is required` / `only streaming is supported`
/// - `stream=true is required`
///
/// Both the decoded `message` **and** the raw body are searched, because
/// [`OpenAiModel::parse_error_body`] only promotes `error.message` / `message`
/// to `ProviderError::message`. A proxy that nests the reason under any other
/// key (`detail`, `errors[0].reason`, …) alongside a generic top-level
/// `message` would otherwise be invisible to the check.
///
/// The `stream` mention is required in addition to the "must be true" phrasing
/// so unrelated 400s ("`max_output_tokens` must be true"-shaped nonsense,
/// `tool_choice` rejections) never route a call into the streaming path.
///
/// Pure, so the detection policy is unit-testable without a network call.
pub(super) fn is_stream_required_error(error: &ProviderError) -> bool {
    if !error
        .status
        .is_some_and(|status| STREAM_REQUIRED_STATUSES.contains(&status))
    {
        return false;
    }
    if mentions_stream_required(&error.message) {
        return true;
    }
    error
        .raw
        .as_ref()
        .is_some_and(|raw| mentions_stream_required(&raw.to_string()))
}

/// The wording half of [`is_stream_required_error`], split out so both the
/// decoded message and the raw body run through identical rules.
fn mentions_stream_required(haystack: &str) -> bool {
    let lower = haystack.to_ascii_lowercase();
    if !lower.contains("stream") {
        return false;
    }
    const PHRASES: [&str; 7] = [
        "must be set to true",
        "must be true",
        "streaming is required",
        "requires streaming",
        "only streaming",
        "stream=true",
        "stream: true is required",
    ];
    PHRASES.iter().any(|phrase| lower.contains(phrase))
}

/// Computes the additional degradation to apply after an HTTP 400, or `None`
/// when the failure is not an auto-degradable request-shape rejection.
///
/// Returns `Some(degrade)` only when the 400 error body implicates a shape the
/// request actually used *and* that shape was not already degraded on the
/// original attempt — so a degraded retry is issued at most once and only when
/// it could plausibly help. The returned [`Degrade`] is the union of `already`
/// and the newly implicated shape, so the retry keeps any baseline degradations.
///
/// Pure, so the 400-detection policy is unit-testable without a network call.
pub(super) fn degrade_for_400(
    message: &str,
    request: &ModelRequest,
    already: Degrade,
) -> Option<Degrade> {
    let lower = message.to_ascii_lowercase();
    let mut degrade = already;

    if !already.named_tool_choice
        && lower.contains("tool_choice")
        && matches!(request.tool_choice, ToolChoice::Tool(_))
    {
        degrade.named_tool_choice = true;
    }
    if !already.json_object
        && lower.contains("response_format")
        && matches!(request.response_format, Some(ResponseFormat::JsonObject))
    {
        degrade.json_object = true;
    }
    if !already.json_schema_strict
        && (lower.contains("response_format")
            || lower.contains("json_schema")
            || lower.contains("strict")
            || lower.contains("additionalproperties"))
        && matches!(
            request.response_format,
            Some(ResponseFormat::JsonSchema { .. } | ResponseFormat::Auto { .. })
        )
    {
        degrade.json_schema_strict = true;
    }
    if !already.native_tools && !request.tools.is_empty() && mentions_tools_unsupported(message) {
        degrade.native_tools = true;
    }

    (degrade != already).then_some(degrade)
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

/// Resolves the `timeout_ms` [`invoke_with_streaming`] should carry when a
/// unary [`ChatModel::invoke`] call degrades onto the streaming wire path.
///
/// An explicit `timeout_ms` always wins (unchanged). Otherwise the caller's
/// mode is unary, not the wire mode — so unlike [`request_timeout`] with
/// `streaming: true`, an unset `timeout_ms` here must **not** resolve to `None`:
/// `stream()` folds `None` into no overall deadline, which would let a provider
/// that returns `200 text/event-stream` and then stops sending bytes hang the
/// unary caller forever. Skipped for caller-owned clients (`with_client`),
/// which manage their own client-level timeout policy and opt out of
/// [`OpenAiModel::effective_request_timeout`] the same way.
pub(super) fn unary_fold_timeout_ms(
    timeout_ms: Option<u64>,
    caller_owned_client: bool,
) -> Option<u64> {
    match timeout_ms {
        Some(ms) => Some(ms),
        None if caller_owned_client => None,
        None => Some(DEFAULT_REQUEST_TIMEOUT_SECS * 1_000),
    }
}

/// Merges baked `defaults` under a request's own `overrides` provider options.
///
/// Keys present in `overrides` win over `defaults`. A `Null` on either side
/// contributes nothing; when neither side is an object the result is
/// `Value::Null`, so [`provider_extra_options`] short-circuits to an empty map.
///
/// A **non-null, non-object** `overrides` is invalid caller input and is returned
/// untouched (never merged) so [`provider_extra_options`] still rejects it with
/// its clear validation error instead of the merge silently dropping it. Pure, so
/// the merge policy is unit-testable without a request.
pub(super) fn merge_provider_options(defaults: &Value, overrides: &Value) -> Value {
    if !overrides.is_null() && !overrides.is_object() {
        return overrides.clone();
    }
    match (defaults.as_object(), overrides.as_object()) {
        (None, None) => Value::Null,
        (Some(base), None) => Value::Object(base.clone()),
        (None, Some(over)) => Value::Object(over.clone()),
        (Some(base), Some(over)) => {
            let mut merged = base.clone();
            for (key, value) in over {
                merged.insert(key.clone(), value.clone());
            }
            Value::Object(merged)
        }
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
    /// When the provider requires streaming — declared via
    /// [`OpenAiModel::with_requires_streaming`], or discovered from a rejection
    /// matching [`is_stream_required_error`] — the call is degraded to the
    /// streaming path internally and the SSE stream is folded into a single
    /// response. A discovered constraint is latched on the instance, so only the
    /// first call pays the rejected round trip.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Model`] on transport failure or a non-2xx
    /// status (the message includes the status code and response body), and
    /// [`TinyAgentsError::Serialization`] when the response cannot be decoded.
    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        if self.responses_api_primary {
            return self.invoke_responses(&request).await;
        }

        // Short-circuit: providers known to require `stream: true` on every
        // call skip the non-streaming attempt entirely to avoid a guaranteed
        // 400. "Known" covers both an explicit `with_requires_streaming(true)`
        // and a constraint latched from an earlier rejection on this instance.
        if self.requires_streaming() {
            tracing::debug!(
                provider = %self.provider,
                model = %self.model,
                "[openai] stream:true required; folding stream into unary response"
            );
            return invoke_with_streaming(self, request).await;
        }

        let response = match self
            .post_chat_with_degrade(&request, false, "request")
            .await
        {
            Ok(response) => response,
            Err(TinyAgentsError::Provider(err)) if is_stream_required_error(&err) => {
                // The provider rejects `stream: false`. Remember it so the next
                // call goes straight to streaming instead of re-paying this
                // failed round trip, then fall back to the streaming path and
                // fold the SSE stream into a response.
                self.latch_stream_required();
                tracing::info!(
                    provider = %err.provider,
                    model = %err.model.as_deref().unwrap_or("?"),
                    status = ?err.status,
                    "[openai] provider rejected non-streaming request; \
                     falling back to streaming path"
                );
                return invoke_with_streaming(self, request).await;
            }
            Err(e) => return Err(e),
        };

        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("openai response body read failed: {e}"))
        })?;

        let value: Value = serde_json::from_str(&text)?;
        let response = parse_chat_response(
            value,
            self.effective_reasoning_tags(),
            self.cache_accounting,
        )?;
        // Recover the model's `<tool_call>` blocks into `message.tool_calls` for
        // prompt-guided models, and as a fallback for native models that emitted
        // the call as text despite being flagged native (empty structured
        // `tool_calls`). Native responses that already carry structured calls are
        // returned unchanged.
        if crate::harness::tool::should_recover(
            self.native_tools_enabled(),
            !request.tools.is_empty(),
            response.message.tool_calls.len(),
        ) {
            return Ok(crate::harness::tool::apply_prompt_tool_calls(response));
        }
        Ok(response)
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
        // The Responses path is text-in/text-out in this port: do the unary call
        // and surface it as a single terminal `Completed` (a leading `Started` +
        // one `MessageDelta` carrying the text so a UI still renders it). True
        // Responses SSE is a follow-up.
        if self.responses_api_primary {
            let response = self.invoke_responses(&request).await?;
            let delta = crate::harness::message::MessageDelta {
                text: response.text(),
                reasoning: String::new(),
                tool_call: None,
            };
            let items = vec![
                ModelStreamItem::Started,
                ModelStreamItem::MessageDelta(delta),
                ModelStreamItem::Completed(response),
            ];
            return Ok(Box::pin(futures::stream::iter(items)));
        }
        let response = self
            .post_chat_with_degrade(&request, true, "stream request")
            .await?;

        // Leniency: some OpenAI-compatible endpoints (and test mocks) ignore
        // `stream: true` and return a single non-streaming JSON body. Detect a
        // non-`text/event-stream` content-type and parse it as one completion,
        // surfaced as `Started` + one `MessageDelta` + `Completed` — matching the
        // tolerant behaviour of hand-rolled OpenAI-compatible clients so a
        // non-streaming upstream still drives a streaming turn.
        let is_event_stream = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
            .unwrap_or(false);
        if !is_event_stream {
            let text = response.text().await.map_err(|e| {
                TinyAgentsError::Model(format!("openai non-stream stream-body read failed: {e}"))
            })?;
            let value: Value = serde_json::from_str(&text)?;
            let mut parsed = parse_chat_response(
                value,
                self.effective_reasoning_tags(),
                self.cache_accounting,
            )?;
            if crate::harness::tool::should_recover(
                self.native_tools_enabled(),
                !request.tools.is_empty(),
                parsed.message.tool_calls.len(),
            ) {
                parsed = crate::harness::tool::apply_prompt_tool_calls(parsed);
            }
            let delta = crate::harness::message::MessageDelta {
                text: parsed.text(),
                reasoning: String::new(),
                tool_call: None,
            };
            let items = vec![
                ModelStreamItem::Started,
                ModelStreamItem::MessageDelta(delta),
                ModelStreamItem::Completed(parsed),
            ];
            return Ok(Box::pin(futures::stream::iter(items)));
        }

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
            acc: OpenAiStreamAcc::new(
                self.effective_reasoning_tags().cloned(),
                self.cache_accounting,
            ),
            provider: self.provider.clone(),
            model: self.model.clone(),
            started: false,
            finished: false,
            terminal_emitted: false,
        };

        let stream = futures::stream::unfold(state, sse_next);
        // Recover `<tool_call>` blocks into `message.tool_calls`: always for
        // prompt-guided models, and as a fallback for native models whose terminal
        // response carried no structured `tool_calls` (some OpenAI-compatible
        // routes leak the call as `<tool_call>` text instead). Two surfaces are
        // cleaned so nothing leaks:
        //   * the terminal `Completed` response (the aggregated answer), and
        //   * each streamed `MessageDelta`, via `ToolCallStreamScrubber`, so a
        //     consumer rendering live text never sees raw markup mid-stream.
        // With no tools offered the stream is forwarded untouched. When native
        // structured calls came back the terminal response passes through
        // unchanged (`should_recover` is false); delta text is re-chunked but its
        // concatenation is byte-identical, since a structured-call response has no
        // `<tool_call>` markup for the scrubber to remove.
        if request.tools.is_empty() {
            return Ok(Box::pin(stream));
        }
        let native = self.native_tools_enabled();
        let mut scrubber = crate::harness::tool::ToolCallStreamScrubber::new();
        Ok(Box::pin(stream.flat_map(move |item| {
            futures::stream::iter(clean_stream_item(item, &mut scrubber, native))
        })))
    }
}

/// Applies text-mode tool-call cleanup to one streamed item, in the
/// recovery-eligible path (tools were offered).
///
/// * A [`MessageDelta`](ModelStreamItem::MessageDelta) has its visible text run
///   through `scrubber` so raw `<tool_call>` markup never reaches a consumer
///   rendering live deltas; a delta the scrubber emptied entirely (and that
///   carries no reasoning or tool-call chunk) is dropped.
/// * The terminal [`Completed`](ModelStreamItem::Completed) response first emits
///   any text the scrubber was holding as a final delta (so delta-rebuilt text
///   matches the cleaned response), then recovers `<tool_call>` blocks per
///   [`should_recover`](crate::harness::tool::should_recover) — a native response
///   with structured calls passes through unchanged.
///
/// Returns the item(s) to forward; stateful across a stream via `scrubber`.
pub(super) fn clean_stream_item(
    item: ModelStreamItem,
    scrubber: &mut crate::harness::tool::ToolCallStreamScrubber,
    native: bool,
) -> Vec<ModelStreamItem> {
    match item {
        ModelStreamItem::MessageDelta(mut delta) => {
            if !delta.text.is_empty() {
                delta.text = scrubber.feed(&delta.text);
            }
            if delta.text.is_empty() && delta.reasoning.is_empty() && delta.tool_call.is_none() {
                Vec::new()
            } else {
                vec![ModelStreamItem::MessageDelta(delta)]
            }
        }
        ModelStreamItem::Completed(response) => {
            let mut out = Vec::new();
            let tail = scrubber.flush();
            if !tail.is_empty() {
                out.push(ModelStreamItem::MessageDelta(
                    crate::harness::message::MessageDelta::text(tail),
                ));
            }
            let recovered = if crate::harness::tool::should_recover(
                native,
                true,
                response.message.tool_calls.len(),
            ) {
                crate::harness::tool::apply_prompt_tool_calls(response)
            } else {
                response
            };
            out.push(ModelStreamItem::Completed(recovered));
            out
        }
        other => vec![other],
    }
}

/// Folds a streaming OpenAI call into a single [`ModelResponse`].
///
/// Issues the request with `stream: true`, reads the SSE body, and folds
/// every [`ModelStreamItem`] through [`StreamAccumulator`].
///
/// The state parameter is ignored — [`OpenAiModel`] does not use it for
/// streaming: it is passed only to satisfy the trait method signature.
/// This is safe because `OpenAiModel` ignores the state argument in its
/// `ChatModel` impl (the stream method never reads it).
async fn invoke_with_streaming(
    model: &OpenAiModel,
    mut request: ModelRequest,
) -> Result<ModelResponse> {
    // `invoke` is a unary call to its caller and is documented (and, below,
    // tested) to be capped at `DEFAULT_REQUEST_TIMEOUT_SECS` when the caller did
    // not set an explicit `timeout_ms`. Folding onto the streaming wire path
    // must not silently drop that cap — see `unary_fold_timeout_ms`.
    request.timeout_ms = unary_fold_timeout_ms(request.timeout_ms, model.caller_owned_client);

    let mut stream = model.stream(&(), request).await?;
    let mut acc = StreamAccumulator::new();

    while let Some(item) = stream.next().await {
        acc.push(&item);
    }

    acc.finish()
}
