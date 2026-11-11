//! Published price and context-window snapshot, last checked 2026-11-11.

use super::{LongContextTier, ModelPrice};

/// Published list prices and context windows for the default models the product
/// can route to.
///
/// Each provider block names the page its rows were checked against and the
/// date ([`super::PRICING_AS_OF`]). Rates are standard (not batch) list prices;
/// the cached rate is the cache-read price. Models their provider retired are
/// removed rather than kept at stale prices. Context
/// windows are the published maximums. Rows with a long-context price step are
/// the short-context rates; [`LONG_CONTEXT_TIERS`] holds the step.
pub const KNOWN_MODEL_PRICING: &[ModelPrice] = &[
    // ── Anthropic: platform.claude.com/docs/en/about-claude/pricing (2026-11-11).
    // Cache read = 0.1× input; 4.6 and later bill the full 1M window flat.
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-fable-5",
        input_per_mtok_usd: 10.00,
        cached_input_per_mtok_usd: 1.00,
        output_per_mtok_usd: 50.00,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-opus-4-8",
        input_per_mtok_usd: 5.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 25.00,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-opus-4-7",
        input_per_mtok_usd: 5.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 25.00,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-opus-4-6",
        input_per_mtok_usd: 5.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 25.00,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-opus-4-5",
        input_per_mtok_usd: 5.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 25.00,
        context_window: 200_000,
    },
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-sonnet-4-6",
        input_per_mtok_usd: 3.00,
        cached_input_per_mtok_usd: 0.30,
        output_per_mtok_usd: 15.00,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-sonnet-4-5",
        input_per_mtok_usd: 3.00,
        cached_input_per_mtok_usd: 0.30,
        output_per_mtok_usd: 15.00,
        context_window: 200_000,
    },
    ModelPrice {
        provider: "anthropic",
        model_id: "claude-haiku-4-5",
        input_per_mtok_usd: 1.00,
        cached_input_per_mtok_usd: 0.10,
        output_per_mtok_usd: 5.00,
        context_window: 200_000,
    },
    // ── OpenAI: developers.openai.com/api/docs/pricing (2026-11-11).
    // Cache read is 0.1× input on GPT-5.x and 0.25× on GPT-4.1 and o3.
    ModelPrice {
        provider: "openai",
        model_id: "gpt-5.5",
        input_per_mtok_usd: 5.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 30.00,
        context_window: 1_050_000,
    },
    ModelPrice {
        provider: "openai",
        model_id: "gpt-5.4",
        input_per_mtok_usd: 2.50,
        cached_input_per_mtok_usd: 0.25,
        output_per_mtok_usd: 15.00,
        context_window: 1_050_000,
    },
    ModelPrice {
        provider: "openai",
        model_id: "gpt-5.4-mini",
        input_per_mtok_usd: 0.75,
        cached_input_per_mtok_usd: 0.075,
        output_per_mtok_usd: 4.50,
        context_window: 400_000,
    },
    ModelPrice {
        provider: "openai",
        model_id: "gpt-5.4-nano",
        input_per_mtok_usd: 0.20,
        cached_input_per_mtok_usd: 0.02,
        output_per_mtok_usd: 1.25,
        context_window: 400_000,
    },
    ModelPrice {
        provider: "openai",
        model_id: "gpt-4.1",
        input_per_mtok_usd: 2.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 8.00,
        context_window: 1_047_576,
    },
    ModelPrice {
        provider: "openai",
        model_id: "gpt-4.1-mini",
        input_per_mtok_usd: 0.40,
        cached_input_per_mtok_usd: 0.10,
        output_per_mtok_usd: 1.60,
        context_window: 1_047_576,
    },
    ModelPrice {
        provider: "openai",
        model_id: "o3",
        input_per_mtok_usd: 2.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 8.00,
        context_window: 200_000,
    },
    // ── Google: aistudio.google.com/docs/pricing (2026-11-11). Text cache read
    // = 0.1× input. 2.5 Pro and Flash are deprecated and re-routed from
    // 2026-12-01.
    ModelPrice {
        provider: "google",
        model_id: "gemini-2.5-pro",
        input_per_mtok_usd: 1.25,
        cached_input_per_mtok_usd: 0.125,
        output_per_mtok_usd: 10.00,
        context_window: 1_048_576,
    },
    ModelPrice {
        provider: "google",
        model_id: "gemini-2.5-flash",
        input_per_mtok_usd: 0.30,
        cached_input_per_mtok_usd: 0.03,
        output_per_mtok_usd: 2.50,
        context_window: 1_048_576,
    },
    ModelPrice {
        provider: "google",
        model_id: "gemini-2.5-flash-lite",
        input_per_mtok_usd: 0.10,
        cached_input_per_mtok_usd: 0.01,
        output_per_mtok_usd: 0.40,
        context_window: 1_048_576,
    },
    // ── Moonshot: platform.kimi.ai/docs/pricing/chat (2026-11-11).
    ModelPrice {
        provider: "moonshot",
        model_id: "kimi-k2.6",
        input_per_mtok_usd: 0.95,
        cached_input_per_mtok_usd: 0.16,
        output_per_mtok_usd: 4.00,
        context_window: 262_144,
    },
    // ── Qwen: alibabacloud.com/help/en/model-studio/model-pricing, International
    // (2026-11-11). Cache read is the explicit-cache rate (0.1× input);
    // qwen-max has implicit cache only (0.2×).
    ModelPrice {
        provider: "qwen",
        model_id: "qwen-max",
        input_per_mtok_usd: 1.60,
        cached_input_per_mtok_usd: 0.32,
        output_per_mtok_usd: 6.40,
        context_window: 256_000,
    },
    ModelPrice {
        provider: "qwen",
        model_id: "qwen-plus",
        input_per_mtok_usd: 0.40,
        cached_input_per_mtok_usd: 0.04,
        output_per_mtok_usd: 1.20,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "qwen",
        model_id: "qwen-flash",
        input_per_mtok_usd: 0.05,
        cached_input_per_mtok_usd: 0.005,
        output_per_mtok_usd: 0.40,
        context_window: 1_000_000,
    },
    // ── Mistral: docs.mistral.ai model cards (2026-11-11). mistral-large is
    // Large 4 (26.10, published cache price), also listed under its own ids
    // mistral-large-4 and mistral-large-4-0 (a one-digit suffix is not a
    // snapshot decoration, so lookup would not reach the alias row); mistral-small is Small 4 (26.03);
    // ministral-8b is Ministral 3 8B (25.12); codestral is 25.08. Small, 8B and
    // Codestral publish no cache price, so theirs is an estimate (0.1× input).
    ModelPrice {
        provider: "mistral",
        model_id: "mistral-large",
        input_per_mtok_usd: 0.70,
        cached_input_per_mtok_usd: 0.07,
        output_per_mtok_usd: 2.10,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "mistral",
        model_id: "mistral-large-4",
        input_per_mtok_usd: 0.70,
        cached_input_per_mtok_usd: 0.07,
        output_per_mtok_usd: 2.10,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "mistral",
        model_id: "mistral-large-4-0",
        input_per_mtok_usd: 0.70,
        cached_input_per_mtok_usd: 0.07,
        output_per_mtok_usd: 2.10,
        context_window: 1_000_000,
    },
    ModelPrice {
        provider: "mistral",
        model_id: "mistral-small",
        input_per_mtok_usd: 0.15,
        cached_input_per_mtok_usd: 0.015,
        output_per_mtok_usd: 0.60,
        context_window: 256_000,
    },
    ModelPrice {
        provider: "mistral",
        model_id: "codestral",
        input_per_mtok_usd: 0.30,
        cached_input_per_mtok_usd: 0.03,
        output_per_mtok_usd: 0.90,
        context_window: 128_000,
    },
    ModelPrice {
        provider: "mistral",
        model_id: "ministral-8b",
        input_per_mtok_usd: 0.15,
        cached_input_per_mtok_usd: 0.015,
        output_per_mtok_usd: 0.15,
        context_window: 256_000,
    },
    // ── MiniMax: platform.minimax.io/docs/guides/pricing (2026-11-11). M3's row
    // is the ≤512K rate.
    ModelPrice {
        provider: "minimax",
        model_id: "minimax-m3",
        input_per_mtok_usd: 0.30,
        cached_input_per_mtok_usd: 0.06,
        output_per_mtok_usd: 1.20,
        context_window: 1_000_000,
    },
];

/// Published long-context price steps, from the same pages as the rows. Above
/// `above_input_tokens` input tokens (cached tokens included) the whole call
/// bills at the step's rates.
pub const LONG_CONTEXT_TIERS: &[LongContextTier] = &[
    // OpenAI: "Long context: >272K input tokens".
    LongContextTier {
        model_id: "gpt-5.5",
        above_input_tokens: 272_000,
        input_per_mtok_usd: 10.00,
        cached_input_per_mtok_usd: 1.00,
        output_per_mtok_usd: 45.00,
    },
    LongContextTier {
        model_id: "gpt-5.4",
        above_input_tokens: 272_000,
        input_per_mtok_usd: 5.00,
        cached_input_per_mtok_usd: 0.50,
        output_per_mtok_usd: 22.50,
    },
    // Google: prompts >200K.
    LongContextTier {
        model_id: "gemini-2.5-pro",
        above_input_tokens: 200_000,
        input_per_mtok_usd: 2.50,
        cached_input_per_mtok_usd: 0.25,
        output_per_mtok_usd: 15.00,
    },
    // Qwen: the 256K-1M input tier (non-thinking), billed per request.
    LongContextTier {
        model_id: "qwen-plus",
        above_input_tokens: 256_000,
        input_per_mtok_usd: 1.20,
        cached_input_per_mtok_usd: 0.12,
        output_per_mtok_usd: 3.60,
    },
    LongContextTier {
        model_id: "qwen-flash",
        above_input_tokens: 256_000,
        input_per_mtok_usd: 0.25,
        cached_input_per_mtok_usd: 0.025,
        output_per_mtok_usd: 2.00,
    },
    // MiniMax: "> 512k input tokens". The page does not say whether cached
    // tokens count toward the boundary; they are counted here.
    LongContextTier {
        model_id: "minimax-m3",
        above_input_tokens: 512_000,
        input_per_mtok_usd: 0.60,
        cached_input_per_mtok_usd: 0.12,
        output_per_mtok_usd: 2.40,
    },
];
