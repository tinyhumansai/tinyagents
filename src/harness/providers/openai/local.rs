//! Local OpenAI-compatible runtimes: identification, capability **probing**,
//! and the native-API escape hatches the OpenAI wire format cannot express.
//!
//! # Why this module exists
//!
//! A local runtime is not "hosted OpenAI at a different URL". It differs in
//! three ways the transport used to paper over with hard-coded guesses:
//!
//! 1. **Its context window is tiny and not derivable from the model id.**
//!    `derive_profile` filled `max_input_tokens` from the generic hint table,
//!    which matches bare substrings — so `llama3.2:3b` on Ollama claimed
//!    128 000 tokens while Ollama's real default `num_ctx` is **2048**, roughly
//!    a 60× overstatement. Compaction fires at `window * threshold`, so it never
//!    fired and the server silently truncated the front of the prompt.
//!    LangChain refuses to guess here (ChatOllama ships no profile at all and
//!    its summarization middleware hard-fails asking for absolute counts), and
//!    an invented window is strictly worse than the `None` this crate already
//!    supports. See [`LocalProbe::max_input_tokens`].
//! 2. **Whether it accepts native `tools` is a property of the loaded model,
//!    not of "being local".** The transport hard-disabled native tools for every
//!    local runtime unconditionally, which forced the prompt-guided branch —
//!    injecting the protocol block *plus* every tool's JSON Schema into the
//!    system prompt, against that real 2048-token window, which then truncated
//!    from the front and dropped the very prompt carrying the protocol.
//!    Ollama reports this directly in `/api/show`'s `capabilities` array.
//! 3. **Some knobs have no OpenAI-wire spelling at all.** `num_ctx` and
//!    `keep_alive` are `/api/chat` fields; `POST /v1/chat/completions` drops
//!    them on the floor. See [`LocalRuntimeKind::native_root`].
//!
//! Probing is **opt-in** and never runs during construction: it costs a network
//! round trip, and a constructor that blocks on one is unusable in the contexts
//! this crate is embedded in.

use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{Result, TinyAgentsError};

/// A local OpenAI-compatible model server.
///
/// The single place that answers "is this endpoint a local runtime, and which
/// one?". Adding a runtime is one variant plus its arms here — not a condition
/// to keep in sync across the transport.
///
/// Before this existed only Ollama and LM Studio were recognised;
/// llama.cpp-server and vLLM fell through to the hosted `Compatible` path and
/// got Bearer auth, `tool_calling: true`, `image_in: true`, no `/v1`
/// normalisation, and none of the request-shape degrade knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocalRuntimeKind {
    /// Ollama. Serves an OpenAI-compatible surface under `/v1` **and** its own
    /// native API under `/api` — the only place `num_ctx` and `keep_alive` are
    /// readable.
    Ollama,
    /// LM Studio. OpenAI-compatible under `/v1`, with a richer model listing
    /// under `/api/v0/models` (context length, load state, quantisation).
    LmStudio,
    /// `llama-server` from llama.cpp. OpenAI-compatible only.
    LlamaCpp,
    /// vLLM's OpenAI-compatible server.
    Vllm,
}

impl LocalRuntimeKind {
    /// Stable identifier used in provider ids, log lines, and errors.
    pub fn as_str(self) -> &'static str {
        match self {
            LocalRuntimeKind::Ollama => "ollama",
            LocalRuntimeKind::LmStudio => "lm_studio",
            LocalRuntimeKind::LlamaCpp => "llama_cpp",
            LocalRuntimeKind::Vllm => "vllm",
        }
    }

    /// The server root assumed when a spec carries a blank `base_url`.
    pub fn default_root(self) -> &'static str {
        match self {
            LocalRuntimeKind::Ollama => "http://localhost:11434",
            LocalRuntimeKind::LmStudio => "http://localhost:1234",
            LocalRuntimeKind::LlamaCpp => "http://localhost:8080",
            LocalRuntimeKind::Vllm => "http://localhost:8000",
        }
    }

    /// Strips the OpenAI-compatibility suffix off `base_url`, yielding the
    /// server root the runtime's **native** API hangs off.
    ///
    /// `http://localhost:11434/v1` → `http://localhost:11434`, so `/api/show`,
    /// `/api/chat` and `/api/v0/models` can be reached. Idempotent for a base
    /// that already is the root.
    pub fn native_root(self, base_url: &str) -> String {
        base_url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .trim_end_matches('/')
            .to_string()
    }

    /// Whether this runtime speaks a native (non-OpenAI) API this crate knows
    /// how to use. Only Ollama does today.
    pub fn has_native_api(self) -> bool {
        matches!(self, LocalRuntimeKind::Ollama)
    }
}

/// What a probe of a live local server learned about the loaded model.
///
/// Every field is [`Option`] on purpose: a runtime that does not report a fact
/// leaves it `None` and the caller keeps whatever it already had, rather than
/// having a guess written over it. This is the shape LangChain's
/// `libs/model-profiles` keys on (`max_input_tokens` is a first-class key
/// there) — with the pointed difference that no `ollama` profile file exists in
/// that repo, which is the evidence that a static catalogue is the wrong answer
/// for local models and runtime probing is the state of the art.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalProbe {
    /// The model's real context window in tokens, when the server reports one.
    ///
    /// For Ollama this is the `*.context_length` entry of `/api/show`'s
    /// `model_info` — the **architecture's** trained window. Note the runtime
    /// still loads with `num_ctx` (default 2048) unless told otherwise, so a
    /// caller that wants the full window must also request it; see
    /// [`OpenAiModel::with_local_num_ctx`][wlnc].
    ///
    /// [wlnc]: super::OpenAiModel::with_local_num_ctx
    pub max_input_tokens: Option<u64>,
    /// Whether the loaded model advertises native tool calling.
    pub tool_calling: Option<bool>,
    /// Whether the loaded model advertises image input.
    pub vision: Option<bool>,
    /// Whether the loaded model advertises a reasoning/thinking channel.
    pub reasoning: Option<bool>,
    /// The `num_ctx` the runtime says it actually loaded the model with, when
    /// it reports one. This — not [`Self::max_input_tokens`] — is the number
    /// that bounds a live request.
    pub loaded_num_ctx: Option<u64>,
}

impl LocalProbe {
    /// Whether the probe learned anything at all.
    pub fn is_empty(&self) -> bool {
        *self == LocalProbe::default()
    }

    /// The context window to advertise: the loaded `num_ctx` when known (it is
    /// the real ceiling for a live request), else the architecture window, else
    /// `None`.
    ///
    /// Deliberately **not** "the bigger of the two". Overstating the window is
    /// the LOCAL-1 defect: compaction is gated on it, so a window larger than
    /// the server will honour means compaction never fires and the server
    /// truncates the prompt from the front instead — losing the system prompt
    /// silently.
    pub fn effective_context_window(&self) -> Option<u64> {
        self.loaded_num_ctx.or(self.max_input_tokens)
    }
}

// ---------------------------------------------------------------------------
// Ollama `/api/show`
// ---------------------------------------------------------------------------

/// The subset of Ollama's `POST /api/show` body this crate reads.
#[derive(Debug, Default, Deserialize)]
struct OllamaShowResponse {
    /// Architecture metadata. Keys are namespaced by architecture
    /// (`llama.context_length`, `qwen3.context_length`, …), so the reader scans
    /// for a `*.context_length` suffix rather than guessing the prefix.
    #[serde(default)]
    model_info: serde_json::Map<String, Value>,
    /// Capability tags: `completion`, `tools`, `vision`, `thinking`, `insert`.
    #[serde(default)]
    capabilities: Vec<String>,
}

/// Extracts the architecture context length from an Ollama `model_info` map.
///
/// Keys are `{architecture}.context_length`, so match on the suffix. Returns
/// the smallest candidate when several match, staying conservative for the same
/// reason [`LocalProbe::effective_context_window`] does.
fn context_length_from_model_info(model_info: &serde_json::Map<String, Value>) -> Option<u64> {
    model_info
        .iter()
        .filter(|(key, _)| key.ends_with(".context_length") || key.as_str() == "context_length")
        .filter_map(|(_, value)| value.as_u64())
        .filter(|value| *value > 0)
        .min()
}

/// Turns an Ollama `/api/show` body into a [`LocalProbe`].
///
/// Pure, so the whole mapping is unit-testable without a live Ollama.
pub(super) fn probe_from_ollama_show(body: &Value) -> LocalProbe {
    let parsed: OllamaShowResponse =
        serde_json::from_value(body.clone()).unwrap_or_else(|_| OllamaShowResponse::default());
    let has = |tag: &str| {
        parsed
            .capabilities
            .iter()
            .any(|c| c.eq_ignore_ascii_case(tag))
    };
    // An empty `capabilities` array means "this server did not tell us",
    // not "this model can do nothing" — leave those `None` so the caller keeps
    // whatever it already had.
    let capabilities_reported = !parsed.capabilities.is_empty();
    LocalProbe {
        max_input_tokens: context_length_from_model_info(&parsed.model_info),
        tool_calling: capabilities_reported.then(|| has("tools")),
        vision: capabilities_reported.then(|| has("vision")),
        reasoning: capabilities_reported.then(|| has("thinking")),
        loaded_num_ctx: None,
    }
}

// ---------------------------------------------------------------------------
// LM Studio `/api/v0/models`
// ---------------------------------------------------------------------------

/// One entry of LM Studio's richer `GET /api/v0/models` listing.
#[derive(Debug, Deserialize)]
struct LmStudioModel {
    #[serde(default)]
    id: String,
    /// The model's context length. LM Studio reports the trained window here.
    #[serde(default)]
    max_context_length: Option<u64>,
    /// The context the model is currently **loaded** with, when loaded.
    #[serde(default)]
    loaded_context_length: Option<u64>,
    /// `llm`, `vlm` (vision), or `embeddings`.
    #[serde(default)]
    r#type: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct LmStudioModelList {
    #[serde(default)]
    data: Vec<LmStudioModel>,
}

/// Turns an LM Studio `/api/v0/models` body into a [`LocalProbe`] for `model`.
///
/// Pure, so the mapping is unit-testable without a live LM Studio.
pub(super) fn probe_from_lm_studio_models(body: &Value, model: &str) -> LocalProbe {
    let parsed: LmStudioModelList =
        serde_json::from_value(body.clone()).unwrap_or_else(|_| LmStudioModelList::default());
    let Some(entry) = parsed.data.iter().find(|m| m.id == model) else {
        return LocalProbe::default();
    };
    LocalProbe {
        max_input_tokens: entry.max_context_length.filter(|v| *v > 0),
        // LM Studio does not report tool support in this listing; leave it
        // untouched rather than inventing an answer.
        tool_calling: None,
        vision: entry
            .r#type
            .as_deref()
            .map(|t| t.eq_ignore_ascii_case("vlm")),
        reasoning: None,
        loaded_num_ctx: entry.loaded_context_length.filter(|v| *v > 0),
    }
}

// ---------------------------------------------------------------------------
// Request bodies for the native escape hatches
// ---------------------------------------------------------------------------

/// The `POST /api/show` body: which model to describe.
pub(super) fn ollama_show_body(model: &str) -> Value {
    json!({ "model": model })
}

/// The `POST /api/chat` preflight body that loads `model` with explicit
/// `options` and residency.
///
/// # Why a preflight rather than a request field
///
/// `num_ctx` and `keep_alive` are **`/api/chat` fields**. The chat adapter
/// speaks `POST {base_url}/chat/completions`, and Ollama's OpenAI-compatibility
/// layer does not read them — so
/// [`with_default_provider_options`][super::OpenAiModel::with_default_provider_options]
/// documenting `{"options": {"num_ctx": 8192}}` as "the local escape hatch"
/// was, on this path, a field that went nowhere. (The crate's tests asserted
/// only that the request JSON *contained* it, never that a server honoured it,
/// which is exactly why that went unnoticed.)
///
/// An `/api/chat` call with an empty `messages` array is Ollama's documented
/// **load** request: it loads the model with the given `options` and holds it
/// resident for `keep_alive`. Issuing it once before the first real turn gets
/// `num_ctx` where the OpenAI wire cannot, and doubles as the warm-up that
/// keeps Ollama from unloading after its 5-minute default and charging the next
/// turn a cold multi-gigabyte load inside the 600 s unary deadline.
///
/// **Caveat, stated plainly:** this configures the *loaded runner*. Ollama
/// reuses an already-loaded runner for a subsequent `/v1` request that does not
/// demand conflicting options, which is the case here — but it is a property of
/// the server's runner reuse, not a guarantee of the OpenAI wire format. A full
/// native `/api/chat` chat adapter remains the complete fix and is called out as
/// a follow-up.
pub(super) fn ollama_load_body(
    model: &str,
    options: Option<&Value>,
    keep_alive: Option<&str>,
) -> Value {
    let mut body = json!({ "model": model, "messages": [] });
    if let Some(options) = options.filter(|o| o.is_object()) {
        body["options"] = options.clone();
    }
    if let Some(keep_alive) = keep_alive {
        body["keep_alive"] = json!(keep_alive);
    }
    body
}

/// Extracts an `options` object out of merged provider options, if present.
///
/// The escape hatch's documented shape is `{"options": {"num_ctx": 8192}}`, so
/// this is what the preflight forwards natively.
pub(super) fn local_options_object(provider_options: &Value) -> Option<&Value> {
    provider_options.get("options").filter(|v| v.is_object())
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/// Rewrites a local runtime's opaque 404 into a message naming the fix.
///
/// The embeddings adapter has done this for a while — "Run `ollama pull
/// {model}` or choose an installed embedding model" — while the chat path
/// surfaced whatever the server said, typically a bare
/// `{"error":"model 'x' not found"}`. Returns `None` when the failure is not a
/// missing-model 404, so the original message survives untouched.
pub(super) fn missing_model_remediation(
    kind: LocalRuntimeKind,
    status: u16,
    body: &str,
    model: &str,
    base_url: &str,
) -> Option<String> {
    if status != 404 {
        return None;
    }
    let lower = body.to_ascii_lowercase();
    if !(lower.contains("model")
        && (lower.contains("not found") || lower.contains("does not exist")))
    {
        return None;
    }
    Some(match kind {
        LocalRuntimeKind::Ollama => format!(
            "Ollama model `{model}` is not installed at {base_url}. \
             Run `ollama pull {model}`, or call `list_models()` to see what is installed"
        ),
        LocalRuntimeKind::LmStudio => format!(
            "LM Studio at {base_url} is not serving a model called `{model}`. \
             Load it in LM Studio, or call `list_models()` to see what is loaded — \
             the id is whichever GGUF the server has open, so there is no default to guess"
        ),
        _ => format!(
            "{} at {base_url} is not serving a model called `{model}`. \
             Call `list_models()` to see what is available",
            kind.as_str()
        ),
    })
}

/// Recognises "the prompt did not fit in this model's context window" from a
/// provider error.
///
/// Hosted providers raise an explicit 400 for this; local servers usually
/// truncate the front of the prompt silently instead, which is why this must be
/// paired with a *real* context window from [`LocalProbe`] rather than relied on
/// alone. When it does fire, the classification is stable so a caller can act on
/// it (compact and retry) instead of string-matching a provider message.
///
/// Surfaced as a [`ProviderError::code`][pe] of
/// [`CONTEXT_OVERFLOW_CODE`], because a typed
/// `TinyAgentsError::ContextOverflow` variant would have to be added in
/// `src/error.rs` — outside this module's ownership. Promoting the code to a
/// typed variant is a follow-up.
///
/// [pe]: crate::harness::model::ProviderError::code
pub(super) fn is_context_overflow(status: u16, message: &str) -> bool {
    if !matches!(status, 400 | 413 | 422 | 500) {
        return false;
    }
    let lower = message.to_ascii_lowercase();
    const PHRASES: [&str; 7] = [
        "context length",
        "context window",
        "maximum context",
        "too many tokens",
        "reduce the length of the messages",
        "prompt is too long",
        "exceeds the maximum",
    ];
    PHRASES.iter().any(|phrase| lower.contains(phrase))
}

/// The [`ProviderError::code`][pe] stamped on a recognised context overflow.
///
/// [pe]: crate::harness::model::ProviderError::code
pub const CONTEXT_OVERFLOW_CODE: &str = "context_overflow";

/// Recognises "this endpoint rejects the `tools` parameter" from a 400 body.
///
/// Drives the [`Degrade::native_tools`][d] latch, which is the auto-degrade half
/// of C11: a local server that cannot do native tools tells us so once, and
/// every later call goes straight to the prompt-guided branch — instead of the
/// old behaviour, which assumed *every* local server was in that state forever.
///
/// [d]: super::transport::Degrade
pub(super) fn mentions_tools_unsupported(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    if !(lower.contains("tool") || lower.contains("function")) {
        return false;
    }
    // `tool_choice` rejections are a *different* degrade with its own latch;
    // matching them here would flip the wrong knob.
    if lower.contains("tool_choice") && !lower.contains("tools") {
        return false;
    }
    const PHRASES: [&str; 8] = [
        "does not support tools",
        "does not support function",
        "not supported",
        "unsupported",
        "unknown parameter",
        "unrecognized",
        "invalid parameter",
        "no tool support",
    ];
    PHRASES.iter().any(|phrase| lower.contains(phrase))
}

/// Deadline for a probe request. Probing is a convenience, never the point of
/// the call, so it fails fast rather than blocking a turn behind a wedged
/// server — the same failure the `list_models` deadline was added for.
pub(super) const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Maps a probe transport failure onto the crate error with a grep-friendly
/// message naming the endpoint.
pub(super) fn probe_error(endpoint: &str, detail: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Model(format!(
        "[openai] local probe of {endpoint} failed: {detail}"
    ))
}

/// Convenience alias so callers of the probe read naturally.
pub type ProbeResult = Result<LocalProbe>;

#[cfg(test)]
#[path = "local_test.rs"]
mod local_test;
