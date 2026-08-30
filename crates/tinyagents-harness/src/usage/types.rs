//! Token usage accounting types.

use serde::{Deserialize, Serialize};

/// Normalized token usage for a single model call.
///
/// Providers expose different breakdowns; fields default to zero so partial
/// data still produces a valid record. Detail fields (cache, reasoning) do not
/// need to sum to the totals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt/input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Completion/output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Total tokens (input + output) as reported by the provider when known.
    #[serde(default)]
    pub total_tokens: u64,
    /// Input tokens served from a provider prompt/KV cache.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Input tokens written into a provider prompt/KV cache.
    #[serde(default)]
    pub cache_creation_tokens: u64,
    /// Reasoning/thinking output tokens when the provider exposes them.
    #[serde(default)]
    pub reasoning_tokens: u64,
}

/// Aggregate usage across many calls, tracking both the call count and the
/// summed [`Usage`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageTotals {
    /// Number of accumulated calls.
    #[serde(default)]
    pub calls: u64,
    /// Summed usage across all accumulated calls.
    #[serde(default)]
    pub usage: Usage,
}
