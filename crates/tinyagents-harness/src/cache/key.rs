//! Response-cache key derivation and provider prompt-cache breakpoints.
//!
//! # Why the key is a two-tuple, not just the prompt
//!
//! A cache keyed on the prompt alone is not a cache of an *answer*, it is a
//! cache of a *question*: the same question asked of a hosted frontier model
//! and of a local 3B model has two entirely different answers, and one shared
//! [`ResponseCache`][super::ResponseCache] would serve either to the other.
//!
//! LangChain looks up on `(prompt, llm_string)` — never the prompt alone — and
//! `llm_string` serializes the whole model object (class path, model name,
//! params). This module mirrors that: [`cache_key`] hashes the request, and
//! [`scoped_cache_key`] folds in the *resolved* model's
//! [`cache_identity`][tinyinference::model::ChatModel::cache_identity] plus the
//! streaming mode and the policy namespace. The composition is deliberate — the
//! request half can be computed before model resolution and reused, while the
//! identity half is only knowable after it.

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::hash::{fold_bytes, fold_canonical, hex_digest};
use tinyinference::model::ModelRequest;

/// Produces a stable, deterministic **request-half** cache key for `request`.
///
/// The key is a 64-character lowercase SHA-256 hex string built by folding the
/// request into the hasher **incrementally**, one component at a time:
/// 1. Fold each conversation message as its own length-prefixed, canonicalized
///    frame (tag `m`), preceded by the message count.
/// 2. Fold each tool schema likewise (tag `t`), preceded by the tool count.
/// 3. Fold an **explicit allowlist projection** of the remaining
///    behaviour-affecting parameters as one envelope frame (tag `E`).
///
/// Folding per component bounds peak memory by the single largest component
/// instead of the entire request; transcripts routinely carry large tool
/// results.
///
/// # Why an allowlist, not "everything that is left"
///
/// The envelope used to be "whatever remains once `messages` and `tools` are
/// removed", which folded in fields that cannot affect what the model says:
/// [`tags`][ModelRequest::tags] (documented as propagated to events and
/// traces), [`timeout_ms`][ModelRequest::timeout_ms],
/// [`metadata`][ModelRequest::metadata] (free-form — a caller putting a run id
/// there, its natural use, got a permanent 0% hit rate with no diagnostic),
/// [`cache_policy`][ModelRequest::cache_policy] (which selects *whether* to
/// cache), [`prompt_fingerprint`][ModelRequest::prompt_fingerprint] (derived
/// from the messages already folded above), and
/// [`cache_segments`][ModelRequest::cache_segments] (pure annotation).
/// LangChain likewise keys on a deliberate projection and strips run-specific
/// message ids before hashing.
///
/// The projection destructures [`ModelRequest`] **exhaustively** (no `..`), so
/// adding a field to the request is a compile error here until someone decides
/// whether it belongs in the key. That preserves the old "no field can silently
/// drop out" guarantee without the over-inclusion.
///
/// # This is only half the key
///
/// It carries no provider or model identity — the request's `model` field is an
/// optional *hint*, and the endpoint and credentials live inside the
/// `Arc<dyn ChatModel>`. Always compose with [`scoped_cache_key`] once the model
/// has actually been resolved.
///
/// # Panics
/// Does not panic. If serialization unexpectedly fails, the affected frame
/// folds empty bytes; the key stays well-defined.
pub fn cache_key(request: &ModelRequest) -> String {
    let mut hasher = Sha256::new();

    // Messages: fold one at a time so a long transcript never materializes a
    // second full tree. The count frame keeps `[a, b]` distinct from a single
    // message that happens to serialize to the same concatenation.
    //
    // Each message is serialized individually. The previous implementation
    // serialized the whole request and then `map.remove("messages")`-ed it
    // inside an `if let Some(Value::Array(..))`: `remove` ran unconditionally,
    // so a `messages` value that was not a JSON array would have been dropped
    // *without being hashed* — the exact silent-drop false hit the doc comment
    // promised could not happen. Serializing per message removes the shape
    // assumption entirely.
    hasher.update(b"M");
    hasher.update((request.messages.len() as u64).to_le_bytes());
    for message in &request.messages {
        fold_canonical(
            &mut hasher,
            b'm',
            serde_json::to_value(message).unwrap_or(Value::Null),
        );
    }

    // Tool schemas: already name-sorted by `ToolRegistry::schemas`, so the
    // order is deterministic across calls.
    hasher.update(b"T");
    hasher.update((request.tools.len() as u64).to_le_bytes());
    for tool in &request.tools {
        fold_canonical(
            &mut hasher,
            b't',
            serde_json::to_value(tool).unwrap_or(Value::Null),
        );
    }

    fold_canonical(&mut hasher, b'E', cache_key_envelope(request));
    hex_digest(hasher.finalize())
}

/// Builds the allowlist projection of the behaviour-affecting request
/// parameters folded into [`cache_key`]'s envelope frame.
///
/// The exhaustive destructure below is load-bearing: it makes a new
/// [`ModelRequest`] field a compile error rather than a silent omission.
fn cache_key_envelope(request: &ModelRequest) -> Value {
    let ModelRequest {
        // Folded as their own frames by `cache_key`.
        messages: _,
        tools: _,
        // ── Included: these change what the model produces ──────────────────
        tool_choice,
        response_format,
        model,
        model_hints,
        reuse_previous_model,
        temperature,
        top_p,
        max_tokens,
        stop_sequences,
        seed,
        required_capabilities,
        provider_options,
        continuation_id,
        reasoning,
        // ── Excluded: cannot affect the answer ──────────────────────────────
        // Transport deadline only.
        timeout_ms: _,
        // Free-form caller annotation; the natural place to put a run id.
        metadata: _,
        // Documented as propagated to events and traces.
        tags: _,
        // Pure annotation describing prompt structure.
        cache_segments: _,
        // Derived from `messages`/`tools`, both already folded above.
        prompt_fingerprint: _,
        // Selects *whether* to cache, never what is answered.
        cache_policy: _,
    } = request;

    serde_json::json!({
        "tool_choice": tool_choice,
        "response_format": response_format,
        "model": model,
        "model_hints": model_hints,
        "reuse_previous_model": reuse_previous_model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stop_sequences": stop_sequences,
        "seed": seed,
        "required_capabilities": required_capabilities,
        "provider_options": provider_options,
        "continuation_id": continuation_id,
        "reasoning": reasoning,
    })
}

/// Folds the *resolved* model's identity, the streaming mode, and an optional
/// policy namespace into a request-half key from [`cache_key`].
///
/// `identity` is the value returned by
/// [`ChatModel::cache_identity`][tinyinference::model::ChatModel::cache_identity]
/// on the model that is actually about to be (or was) called. `None` means the
/// model declines to identify itself; the key then folds a fixed
/// `"anonymous-model"` marker so a mixed registry of identifying and
/// non-identifying models still cannot cross-serve identifying ones.
///
/// `streaming` is folded because it is a parameter of the call, not a field of
/// the request: a warm streaming run must not be served an entry written by a
/// unary run unless the caller has opted into that sharing.
pub fn scoped_cache_key(
    request_key: &str,
    identity: Option<&str>,
    streaming: bool,
    namespace: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    fold_bytes(&mut hasher, b'R', request_key.as_bytes());
    fold_bytes(
        &mut hasher,
        b'I',
        identity.unwrap_or("anonymous-model").as_bytes(),
    );
    fold_bytes(&mut hasher, b'S', if streaming { b"1" } else { b"0" });
    fold_bytes(&mut hasher, b'N', namespace.unwrap_or("").as_bytes());
    hex_digest(hasher.finalize())
}

/// A non-reversible, log-safe fingerprint of a credential.
///
/// **Never** put a raw API key in a cache key, a log line, or an event: keys
/// leak through crash dumps, exported traces, and durable cache files. This
/// returns the first 16 hex characters of a domain-separated SHA-256 of
/// `secret`, which is enough to distinguish two credentials without carrying
/// either. An empty secret maps to the fixed token `"no-credential"` so a
/// keyless local runtime does not hash the empty string into something that
/// looks like a real fingerprint.
pub fn credential_fingerprint(secret: &str) -> String {
    if secret.is_empty() {
        return "no-credential".to_string();
    }
    let mut hasher = Sha256::new();
    hasher.update(b"tinyagents.credential.v1\0");
    hasher.update(secret.as_bytes());
    hex_digest(hasher.finalize())[..16].to_string()
}

/// Builds the canonical
/// [`cache_identity`][tinyinference::model::ChatModel::cache_identity] string
/// for a provider-backed model.
///
/// The identity names everything that can make two models answer the same
/// prompt differently while sharing one cache: the provider family, the model
/// id, the API base URL, an optional organization/project scope, and a
/// *fingerprint* of the credential (two API keys can address two different
/// fine-tunes or two different tenants behind the same base URL).
///
/// # Never log or store the raw credential
/// `credential` is passed through [`credential_fingerprint`] before it reaches
/// the digest, so neither the identity string nor any key derived from it can
/// carry the secret.
pub fn model_cache_identity(
    provider: &str,
    model: &str,
    api_base: &str,
    scope: Option<&str>,
    credential: &str,
) -> String {
    format!(
        "{provider}|{model}|{api_base}|{}|{}",
        scope.unwrap_or(""),
        credential_fingerprint(credential)
    )
}

// ── Provider prompt-cache breakpoints ────────────────────────────────────────

/// The provider option key carrying the routing hint for a prompt cache shard.
///
/// Named after OpenAI's `prompt_cache_key`; adapters that speak a different
/// dialect read it from `provider_options` and lower it themselves.
pub const PROMPT_CACHE_KEY_OPTION: &str = "prompt_cache_key";

/// Derives a stable routing key for the request's cacheable prompt prefix, or
/// `None` when the request declares no stable prefix.
///
/// A provider prompt cache is sharded: two requests that share a byte prefix
/// only actually hit the same cache if they are routed to the same shard, which
/// is what a `prompt_cache_key` buys. Deriving it from the *stable prefix*
/// fingerprint (rather than a random per-run id) means every turn of one
/// logical thread — and every sub-agent that inherits the same system prompt
/// and tool set — routes together, which is exactly the population that shares
/// a prefix.
pub fn prompt_cache_key(request: &ModelRequest) -> Option<String> {
    let layout = super::PromptCacheLayout::from_request(request);
    if layout.prefix_ids().is_empty() {
        return None;
    }
    Some(format!("tap-{}", layout.fingerprint()))
}

/// Injects a derived [`PROMPT_CACHE_KEY_OPTION`] into `request.provider_options`
/// when the effective policy asks for prefix protection.
///
/// This is the *active* half of the prompt-cache tooling: until now the layout
/// types only ever **observed** a prefix, while
/// [`CachePolicy::protect_prompt_prefix`][super::CachePolicy::protect_prompt_prefix]
/// had no reader anywhere in the crate and so could not change any behaviour.
///
/// Precedence follows the rest of the crate: a caller who already set
/// `prompt_cache_key` in `provider_options` wins and is left untouched.
///
/// Returns `true` when an option was injected.
pub fn apply_prompt_cache_breakpoints(request: &mut ModelRequest) -> bool {
    let protect = request
        .cache_policy
        .as_ref()
        .is_some_and(|policy| policy.protect_prompt_prefix);
    if !protect {
        return false;
    }
    if request
        .provider_options
        .get(PROMPT_CACHE_KEY_OPTION)
        .is_some()
    {
        tinyagents_tracing::debug!(
            "[cache] prompt_cache_key already set by caller; leaving provider_options untouched"
        );
        return false;
    }
    let Some(derived) = prompt_cache_key(request) else {
        tinyagents_tracing::debug!(
            "[cache] protect_prompt_prefix is on but the request declares no cacheable prefix; \
             no prompt_cache_key derived"
        );
        return false;
    };
    if !request.provider_options.is_object() {
        request.provider_options = Value::Object(serde_json::Map::new());
    }
    if let Some(map) = request.provider_options.as_object_mut() {
        map.insert(
            PROMPT_CACHE_KEY_OPTION.to_string(),
            Value::String(derived.clone()),
        );
    }
    tinyagents_tracing::debug!(prompt_cache_key = %derived, "[cache] injected provider prompt-cache breakpoint");
    true
}
