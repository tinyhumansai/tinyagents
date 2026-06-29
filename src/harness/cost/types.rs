//! Cost accounting types.
//!
//! [`CostTotals`] is the additive value that lets cost roll up across a
//! recursive run tree (model call → run → parent run).

use serde::{Deserialize, Serialize};

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
