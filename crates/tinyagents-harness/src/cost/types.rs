//! Cost accounting types.
//!
//! [`CostTotals`] is the additive value that lets cost roll up across a
//! recursive run tree (model call → run → parent run).

use serde::{Deserialize, Serialize};

/// Per-token pricing for a model.
///
/// Every field is optional: `None` means the price is unknown or does not
/// apply, rather than that the token class is free.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelPricing {
    /// Price per input token.
    #[serde(default)]
    pub input_per_token: Option<f64>,
    /// Price per output token.
    #[serde(default)]
    pub output_per_token: Option<f64>,
    /// Discounted price per cached input token.
    #[serde(default)]
    pub cache_read_input_per_token: Option<f64>,
    /// Price per input token written to a prompt cache.
    #[serde(default)]
    pub cache_creation_input_per_token: Option<f64>,
    /// Price per audio input token.
    #[serde(default)]
    pub input_audio_per_token: Option<f64>,
    /// Price per reasoning output token.
    #[serde(default)]
    pub output_reasoning_per_token: Option<f64>,
}

/// A breakdown of estimated cost for one or more model calls, in the pricing
/// table's currency (typically USD). All values are accumulating sums.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostTotals {
    /// Cost attributed to (non-cached) input tokens.
    #[serde(default)]
    pub input_cost: f64,
    /// Cost attributed to output tokens.
    #[serde(default)]
    pub output_cost: f64,
    /// Cost attributed to cache read and cache creation tokens.
    #[serde(default)]
    pub cache_cost: f64,
    /// Cost attributed to reasoning tokens.
    #[serde(default)]
    pub reasoning_cost: f64,
    /// Sum of all component costs.
    #[serde(default)]
    pub total_cost: f64,
}
