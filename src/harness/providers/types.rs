//! Providers module types.
//!
//! All public and internal types for the `providers` module live here.
//! Implementations and trait-impls are in `mod.rs`.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::model::ModelResponse;

// ---------------------------------------------------------------------------
// Provider selection types
// ---------------------------------------------------------------------------

/// Common chat-model providers that can be selected through a uniform factory.
///
/// This enum mirrors LangChain's pragmatic provider registry: it covers popular
/// providers directly while leaving [`ProviderKind::Compatible`] for any
/// endpoint that implements the OpenAI Chat Completions wire protocol.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// Hosted OpenAI API.
    OpenAi,
    /// Anthropic via its OpenAI-compatible Chat Completions endpoint.
    Anthropic,
    /// Local Ollama server exposing `/v1/chat/completions`.
    Ollama,
    /// DeepSeek OpenAI-compatible endpoint.
    DeepSeek,
    /// Groq OpenAI-compatible endpoint.
    Groq,
    /// xAI OpenAI-compatible endpoint.
    Xai,
    /// OpenRouter OpenAI-compatible endpoint.
    OpenRouter,
    /// Together AI OpenAI-compatible endpoint.
    Together,
    /// Mistral OpenAI-compatible endpoint.
    Mistral,
    /// Any user-supplied OpenAI-compatible endpoint.
    Compatible,
}

impl ProviderKind {
    /// Stable provider identifier used in profiles, errors, and registry names.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderKind::OpenAi => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Ollama => "ollama",
            ProviderKind::DeepSeek => "deepseek",
            ProviderKind::Groq => "groq",
            ProviderKind::Xai => "xai",
            ProviderKind::OpenRouter => "openrouter",
            ProviderKind::Together => "together",
            ProviderKind::Mistral => "mistral",
            ProviderKind::Compatible => "compatible",
        }
    }

    /// Best-effort inference from a LangChain-style model string.
    ///
    /// Supports explicit prefixes like `openai:gpt-4.1-mini` as well as common
    /// bare model prefixes. Inference is intentionally conservative; pass an
    /// explicit [`ProviderSpec`] when ambiguity matters.
    pub fn infer(model: &str) -> Option<Self> {
        let lower = model.to_ascii_lowercase();
        if let Some((prefix, _)) = lower.split_once(':') {
            return match prefix {
                "openai" => Some(ProviderKind::OpenAi),
                "anthropic" => Some(ProviderKind::Anthropic),
                "ollama" => Some(ProviderKind::Ollama),
                "deepseek" => Some(ProviderKind::DeepSeek),
                "groq" => Some(ProviderKind::Groq),
                "xai" => Some(ProviderKind::Xai),
                "openrouter" => Some(ProviderKind::OpenRouter),
                "together" => Some(ProviderKind::Together),
                "mistral" | "mistralai" => Some(ProviderKind::Mistral),
                _ => None,
            };
        }
        if lower.starts_with("gpt-")
            || lower.starts_with("o1")
            || lower.starts_with("o3")
            || lower.starts_with("o4")
        {
            Some(ProviderKind::OpenAi)
        } else if lower.starts_with("claude") {
            Some(ProviderKind::Anthropic)
        } else if lower.starts_with("deepseek") {
            Some(ProviderKind::DeepSeek)
        } else if lower.starts_with("grok") {
            Some(ProviderKind::Xai)
        } else if lower.starts_with("mistral") || lower.starts_with("mixtral") {
            Some(ProviderKind::Mistral)
        } else {
            None
        }
    }
}

/// Provider configuration used to construct a chat model adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSpec {
    /// Provider family.
    pub kind: ProviderKind,
    /// Provider id written to profiles and normalized errors.
    pub provider: String,
    /// Default provider model id.
    pub model: String,
    /// API base URL without a trailing slash.
    pub base_url: String,
    /// Environment variable containing the API key, when one is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Whether the provider requires a real API key.
    #[serde(default)]
    pub requires_api_key: bool,
}

impl ProviderSpec {
    /// Returns the default provider spec for a known provider.
    pub fn for_kind(kind: ProviderKind) -> Self {
        match kind {
            ProviderKind::OpenAi => Self::new(
                kind,
                "gpt-4.1-mini",
                "https://api.openai.com/v1",
                Some("OPENAI_API_KEY"),
                true,
            ),
            ProviderKind::Anthropic => Self::new(
                kind,
                "claude-3-5-sonnet-latest",
                "https://api.anthropic.com/v1",
                Some("ANTHROPIC_API_KEY"),
                true,
            ),
            ProviderKind::Ollama => {
                Self::new(kind, "llama3.2", "http://localhost:11434/v1", None, false)
            }
            ProviderKind::DeepSeek => Self::new(
                kind,
                "deepseek-chat",
                "https://api.deepseek.com/v1",
                Some("DEEPSEEK_API_KEY"),
                true,
            ),
            ProviderKind::Groq => Self::new(
                kind,
                "llama-3.3-70b-versatile",
                "https://api.groq.com/openai/v1",
                Some("GROQ_API_KEY"),
                true,
            ),
            ProviderKind::Xai => Self::new(
                kind,
                "grok-2-latest",
                "https://api.x.ai/v1",
                Some("XAI_API_KEY"),
                true,
            ),
            ProviderKind::OpenRouter => Self::new(
                kind,
                "openai/gpt-4o-mini",
                "https://openrouter.ai/api/v1",
                Some("OPENROUTER_API_KEY"),
                true,
            ),
            ProviderKind::Together => Self::new(
                kind,
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "https://api.together.xyz/v1",
                Some("TOGETHER_API_KEY"),
                true,
            ),
            ProviderKind::Mistral => Self::new(
                kind,
                "mistral-small-latest",
                "https://api.mistral.ai/v1",
                Some("MISTRAL_API_KEY"),
                true,
            ),
            ProviderKind::Compatible => Self::new(kind, "", "", None, true),
        }
    }

    fn new(
        kind: ProviderKind,
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: Option<&str>,
        requires_api_key: bool,
    ) -> Self {
        let provider = kind.as_str().to_string();
        Self {
            kind,
            provider,
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key_env: api_key_env.map(str::to_string),
            requires_api_key,
        }
    }

    /// Overrides the default model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Overrides the provider id.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Overrides the API-key environment variable.
    pub fn with_api_key_env(mut self, env: impl Into<String>) -> Self {
        self.api_key_env = Some(env.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Internal behavior enum
// ---------------------------------------------------------------------------

/// The scripted behavior that drives a [`MockModel`] invocation.
///
/// This is an internal type — callers interact with [`MockModel`]'s named
/// constructors instead.
pub(crate) enum MockBehavior {
    /// Echoes the text of the last [`Message::User`][crate::harness::message::Message]
    /// in the request back as the assistant reply.
    Echo,

    /// Always returns a fixed assistant text string, regardless of input.
    Constant(String),

    /// Returns responses from a pre-loaded vector in order, cycling back to
    /// the start when all responses have been consumed. See
    /// [`MockModel::with_responses`] for details.
    Scripted(Vec<ModelResponse>),

    /// Returns a single tool-call request for the named tool.  The
    /// `AssistantMessage` carries the call in its `tool_calls` field and the
    /// `finish_reason` is `"tool_calls"`.
    ToolCall {
        /// Name of the tool the model is requesting.
        name: String,
        /// JSON arguments to supply to the tool.
        arguments: Value,
    },
}

// ---------------------------------------------------------------------------
// Internal mutable state (behind a Mutex for Send + Sync)
// ---------------------------------------------------------------------------

/// Mutable runtime state for [`MockModel`], protected by a `Mutex`.
#[derive(Default)]
pub(crate) struct MockInner {
    /// Total number of [`ChatModel::invoke`][crate::harness::model::ChatModel]
    /// calls made so far (not counting `stream` calls that delegate to invoke).
    pub(crate) call_count: u64,
    /// Next index into the scripted response list (used by [`MockBehavior::Scripted`]).
    pub(crate) scripted_index: usize,
}

// ---------------------------------------------------------------------------
// MockModel
// ---------------------------------------------------------------------------

/// A deterministic, in-process chat model for tests and harness development.
///
/// `MockModel` implements [`ChatModel<State>`][crate::harness::model::ChatModel]
/// generically for *any* `State: Send + Sync`.  It never makes network calls
/// and has no external dependencies.
///
/// # Constructors
///
/// | Constructor | Behaviour |
/// |---|---|
/// | [`MockModel::echo`] | Echoes the last user message text back. |
/// | [`MockModel::constant`] | Always returns the same fixed string. |
/// | [`MockModel::with_responses`] | Returns scripted [`ModelResponse`]s in order, cycling when exhausted. |
/// | [`MockModel::with_tool_call`] | Always returns one tool-call request. |
///
/// # Streaming
///
/// The [`ChatModel::stream`][crate::harness::model::ChatModel] override
/// internally calls [`ChatModel::invoke`] and replays the response as a real
/// [`ModelStream`][crate::harness::model::ModelStream]: a
/// [`Started`][crate::harness::model::ModelStreamItem::Started] item, one or two
/// [`MessageDelta`][crate::harness::model::ModelStreamItem::MessageDelta] items
/// (text split into two equal-sized halves by Unicode scalar value), and a
/// terminal [`Completed`][crate::harness::model::ModelStreamItem::Completed]
/// item carrying the full response. This lets downstream streaming consumers be
/// exercised without any real streaming infrastructure. When the response
/// contains no text (e.g. a tool-call response), a single empty text delta is
/// emitted before completion.
///
/// # Usage estimates
///
/// Every response carries a deterministic [`Usage`][crate::harness::usage::Usage]
/// derived from character counts:
/// - `input_tokens` ≈ total characters in all request messages ÷ 4
/// - `output_tokens` ≈ total characters in the response text ÷ 4 (minimum 1)
///
/// This gives cost-accounting code realistic non-zero values to work with.
///
/// # Placement of real providers
///
/// Real network-backed providers live in sub-modules alongside this one. The
/// OpenAI (and OpenAI-compatible) adapter is always compiled; providers with a
/// different wire protocol would be gated behind their own Cargo feature:
///
/// ```text
/// pub mod openai;                          // always compiled
/// // #[cfg(feature = "anthropic")] pub mod anthropic;
/// // #[cfg(feature = "ollama")]   pub mod ollama;
/// ```
///
/// Add the feature flag to `Cargo.toml` and implement
/// [`ChatModel`][crate::harness::model::ChatModel] in the corresponding module.
/// No changes to `mod.rs` or `harness/mod.rs` are needed beyond enabling the
/// `pub mod` declaration.
pub struct MockModel {
    pub(crate) behavior: MockBehavior,
    pub(crate) inner: Mutex<MockInner>,
}
