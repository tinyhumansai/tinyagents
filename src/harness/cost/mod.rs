//! Cost accounting.
//!
//! Because [`CostTotals`] is additive, cost composes the same way the runtime
//! recurses: each model call's cost folds into its run, and a parent run rolls
//! up the cost of every nested sub-agent and sub-graph beneath it into one
//! total.
//!
//! [`estimate_cost`] prices a [`Usage`] record against a [`ModelPricing`] entry
//! from the registry catalog. [`CostTotals`] supports `+`/`+=` accumulation so
//! a run can roll up cost across many calls.

mod types;

use std::ops::{Add, AddAssign};

use crate::harness::usage::Usage;
use crate::registry::catalog::ModelPricing;

pub use types::*;

impl CostTotals {
    /// Creates an empty cost accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Recomputes `total_cost` from the component costs.
    fn recompute_total(&mut self) {
        self.total_cost =
            self.input_cost + self.output_cost + self.cache_cost + self.reasoning_cost;
    }
}

impl Add for CostTotals {
    type Output = CostTotals;

    fn add(mut self, rhs: CostTotals) -> CostTotals {
        self += rhs;
        self
    }
}

impl AddAssign for CostTotals {
    fn add_assign(&mut self, rhs: CostTotals) {
        self.input_cost += rhs.input_cost;
        self.output_cost += rhs.output_cost;
        self.cache_cost += rhs.cache_cost;
        self.reasoning_cost += rhs.reasoning_cost;
        self.recompute_total();
    }
}

/// Estimates the cost of a [`Usage`] record using per-token [`ModelPricing`].
///
/// Missing prices contribute zero. Cache read and cache creation tokens are
/// priced independently when the catalog provides those rates and folded into
/// `cache_cost`.
pub fn estimate_cost(pricing: &ModelPricing, usage: &Usage) -> CostTotals {
    let price = |rate: Option<f64>, tokens: u64| rate.unwrap_or(0.0) * tokens as f64;

    let mut totals = CostTotals {
        input_cost: price(pricing.input_per_token, usage.input_tokens),
        output_cost: price(pricing.output_per_token, usage.output_tokens),
        cache_cost: price(pricing.cache_read_input_per_token, usage.cache_read_tokens)
            + price(
                pricing.cache_creation_input_per_token,
                usage.cache_creation_tokens,
            ),
        reasoning_cost: price(pricing.output_reasoning_per_token, usage.reasoning_tokens),
        total_cost: 0.0,
    };
    totals.recompute_total();
    totals
}

#[cfg(test)]
mod test;
