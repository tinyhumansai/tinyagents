//! Offline published model prices, context windows and decorated-id lookup.
//!
//! Values preserve the July 2026 consumer snapshot; estimates remain best effort.
//! This module performs no I/O. Consumers own billing authority, runtime capability
//! overlays and product configuration.

/// Month the published values below were last verified. Bump when refreshing.
pub const PRICING_AS_OF: &str = "2026-07";

mod data;
mod types;

pub use data::KNOWN_MODEL_PRICING;
pub use types::ModelPrice;

/// Normalise a model string for matching: lower-case, trim, drop a trailing
/// `:tag` / `@date` decoration and a `[...]` suffix.
fn normalize(model: &str) -> String {
    let mut s = model.trim().to_ascii_lowercase();
    // Strip a `[1m]`-style context-window suffix.
    if let Some(idx) = s.find('[') {
        s.truncate(idx);
    }
    // Strip `:tag` (e.g. ollama-style) and `@date` (Vertex-style) decorations.
    for sep in [':', '@'] {
        if let Some(idx) = s.find(sep) {
            s.truncate(idx);
        }
    }
    s.trim().to_string()
}

/// Resolve a concrete model string to its catalogued row, if known.
///
/// Match order: exact canonical id → id with a leading `vendor/` segment
/// stripped → longest canonical id that the request extends with a snapshot
/// decoration only (`gpt-5.4-mini-2026-05-01`, `claude-sonnet-4-5-20250929`,
/// `gemini-2.5-pro-preview-06-05`; see [`is_decorated_id`]). Returns `None`
/// for unknown models — callers should fall back to a tier estimate.
///
/// Any other suffix names a different model, not a snapshot of a listed one:
/// `gpt-4.1-nano` is not priced like `gpt-4.1`, nor `o3-pro` like `o3`, so
/// those resolve to `None` rather than inheriting a neighbour's rates. The
/// snapshot also retains historical cached-rate assumptions and models no
/// long-context estimate tiers except MiniMax M3.
pub fn lookup(model: &str) -> Option<&'static ModelPrice> {
    let norm = normalize(model);
    if norm.is_empty() {
        return None;
    }
    if let Some(p) = KNOWN_MODEL_PRICING.iter().find(|p| p.model_id == norm) {
        return Some(p);
    }
    let bare = norm.rsplit('/').next().unwrap_or(norm.as_str());
    if let Some(p) = KNOWN_MODEL_PRICING.iter().find(|p| p.model_id == bare) {
        return Some(p);
    }
    KNOWN_MODEL_PRICING
        .iter()
        .filter(|p| is_decorated_id(&norm, p.model_id) || is_decorated_id(bare, p.model_id))
        .max_by_key(|p| p.model_id.len())
}

/// Whether `candidate` is `canonical` followed by `-` and a snapshot
/// decoration: `-`-separated tokens that are each a date or version fragment
/// (`2026-05-01`, `20250929`, `06-05`, `001`) or a release-channel word
/// (`latest`, `preview`, `exp`).
///
/// A prefix match alone is not enough: `gpt-4.1-nano` starts with `gpt-4.1`
/// and `o3-pro` with `o3`, and both are different models with their own
/// rates. Requiring the prefix also keeps short canonical ids out of unrelated
/// tokens (`proto3-chat` must not match the `o3` row, `solo1-7b` must not
/// match `o1`), which a plain substring search once got wrong and leaked a
/// bogus 200K context window through this catalog (issue #4249).
fn is_decorated_id(candidate: &str, canonical: &str) -> bool {
    let Some(rest) = candidate.strip_prefix(canonical) else {
        return false;
    };
    let Some(suffix) = rest.strip_prefix('-') else {
        return false;
    };
    !suffix.is_empty() && suffix.split('-').all(is_decoration_token)
}

/// One token of a snapshot decoration.
fn is_decoration_token(token: &str) -> bool {
    const CHANNELS: &[&str] = &["latest", "preview", "exp"];
    let dated = matches!(token.len(), 2 | 3 | 4 | 8) && token.bytes().all(|b| b.is_ascii_digit());
    dated || CHANNELS.contains(&token)
}

/// Estimate the USD cost of a single model call from catalogued per-MTok rates.
///
/// Prices the standard (cache-miss) input tokens, the cached-prefix input
/// tokens, and the output tokens separately. `cached_input_tokens` are billed at
/// the (usually cheaper) cached rate and are assumed to be a subset of
/// `input_tokens`, so the standard-rate portion is `input − cached`. Returns
/// `0.0` when the model is not catalogued (caller should treat as "unknown, not
/// free" — this is a best-effort estimate used when the provider does not report
/// a charged amount).
pub fn estimate_cost_usd(
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cached_input_tokens: u64,
) -> f64 {
    let Some(p) = lookup(model) else {
        return 0.0;
    };
    let cached = cached_input_tokens.min(input_tokens);
    let standard_input = input_tokens.saturating_sub(cached);
    if p.model_id == "minimax-m3" && input_tokens <= 512_000 {
        return (standard_input as f64) * 0.30 / 1_000_000.0
            + (cached as f64) * 0.06 / 1_000_000.0
            + (output_tokens as f64) * 1.20 / 1_000_000.0;
    }
    let per_tok = |mtok_rate: f64| mtok_rate / 1_000_000.0;
    (standard_input as f64) * per_tok(p.input_per_mtok_usd)
        + (cached as f64) * per_tok(p.cached_input_per_mtok_usd)
        + (output_tokens as f64) * per_tok(p.output_per_mtok_usd)
}

fn per_token(rate_per_mtok: f64) -> Option<f64> {
    (rate_per_mtok > 0.0).then_some(rate_per_mtok / 1_000_000.0)
}

const VISION_INPUT_MODELS: &[&str] = &["minimax-m3"];

// Only models with provider-published cache support are advertised as having
// prompt caching. Some rows, notably Mistral, carry estimated cache rates for
// cost modeling but do not establish that the provider supports the feature.
const PROMPT_CACHING_MODELS: &[&str] = &[
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "deepseek-chat",
    "deepseek-reasoner",
    "kimi-k2.6",
    "kimi-k2.5",
    "qwen3-max",
    "qwen-max",
    "qwen-plus",
    "qwen-flash",
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.7-highspeed",
];

/// Whether the published vision allowlist includes this decorated model id.
pub fn model_accepts_image_input(model_id: &str) -> bool {
    lookup(model_id).is_some_and(|price| VISION_INPUT_MODELS.contains(&price.model_id))
}

/// Project one published catalog row as flat metadata (not a tier-aware estimate) into a TinyAgents model-catalog entry.
///
/// This is intentionally a pricing/window projection only. Consumers still
/// derives live runtime capability flags (tools, vision, streaming) from the
/// provider adapter at construction time, because the static cost catalog does
/// not yet encode those fields authoritatively for every provider/model.
///
/// This preserves the migrated flat metadata projection. For MiniMax M3,
/// `estimate_cost_usd` applies a lower rate through 512K input tokens, while
/// this projection uses the row rates and intentionally contains no tiers.
/// Consumers must not treat these two paths as equivalent call estimates.
pub fn flat_catalog_entry(price: &ModelPrice, source: &str) -> crate::ModelCatalogEntry {
    crate::ModelCatalogEntry {
        provider: price.provider.to_string(),
        model_id: price.model_id.to_string(),
        aliases: Vec::new(),
        mode: "chat".to_string(),
        max_input_tokens: Some(u64::from(price.context_window)),
        max_output_tokens: None,
        deprecation_date: None,
        release_date: None,
        pricing: tinyagents_harness::cost::ModelPricing {
            input_per_token: per_token(price.input_per_mtok_usd),
            output_per_token: per_token(price.output_per_mtok_usd),
            cache_read_input_per_token: per_token(price.cached_input_per_mtok_usd),
            cache_creation_input_per_token: None,
            input_audio_per_token: None,
            output_reasoning_per_token: None,
            // The local price table is flat; context-size tiers are not modeled.
            tiers: Vec::new(),
        },
        capabilities: crate::ModelCapabilities {
            prompt_caching: PROMPT_CACHING_MODELS.contains(&price.model_id),
            vision: model_accepts_image_input(price.model_id),
            ..crate::ModelCapabilities::default()
        },
        source: source.to_string(),
        source_url: None,
        raw: serde_json::json!({ "pricing_as_of": PRICING_AS_OF }),
    }
}

#[cfg(test)]
#[path = "pricing_tests.rs"]
mod tests;
