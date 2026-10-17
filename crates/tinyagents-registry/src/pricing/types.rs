//! Published price row vocabulary.

/// A single model's published per-million-token rates (USD) and context window.
#[derive(Debug, Clone, Copy)]
pub struct ModelPrice {
    /// Canonical provider slug, matching the `cloud_providers` type strings
    /// (`anthropic`, `openai`, `google`, `deepseek`, `moonshot`, `qwen`,
    /// `mistral`). Used as the `provider` field when pre-filling registry rows.
    pub provider: &'static str,
    /// Canonical, lower-case model id used for matching. Keep these distinctive
    /// (no bare `gpt-5`) so a decorated id resolves to one row.
    pub model_id: &'static str,
    /// USD per million standard (cache-miss) input tokens.
    pub input_per_mtok_usd: f64,
    /// USD per million cached-prefix input tokens. Best-effort: exact where the
    /// provider publishes it, otherwise the provider's typical cache discount.
    pub cached_input_per_mtok_usd: f64,
    /// USD per million output tokens.
    pub output_per_mtok_usd: f64,
    /// Maximum context window in tokens (published max input). Providers differ
    /// widely (128K–1M+); callers budget prompts / trigger compaction off this.
    pub context_window: u32,
}
