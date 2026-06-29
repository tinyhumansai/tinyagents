//! Unit tests for cost accounting.
//!
//! These cover [`super::estimate_cost`] pricing each token class (input,
//! output, cache read/creation, reasoning) against a [`ModelPricing`] entry,
//! the handling of missing prices as zero cost, and [`super::CostTotals`]
//! `+`/`+=` accumulation with total recomputation.

use super::*;
use crate::harness::usage::Usage;
use crate::registry::catalog::ModelPricing;

fn pricing() -> ModelPricing {
    ModelPricing {
        input_per_token: Some(0.001),
        output_per_token: Some(0.002),
        cache_read_input_per_token: Some(0.0001),
        cache_creation_input_per_token: Some(0.0005),
        output_reasoning_per_token: Some(0.003),
        ..ModelPricing::default()
    }
}

#[test]
fn estimates_each_component() {
    let usage = Usage {
        input_tokens: 1000,
        output_tokens: 500,
        total_tokens: 1500,
        cache_read_tokens: 100,
        cache_creation_tokens: 200,
        reasoning_tokens: 10,
    };
    let cost = estimate_cost(&pricing(), &usage);
    assert!((cost.input_cost - 1.0).abs() < 1e-9);
    assert!((cost.output_cost - 1.0).abs() < 1e-9);
    // 100*0.0001 + 200*0.0005 = 0.01 + 0.1 = 0.11
    assert!((cost.cache_cost - 0.11).abs() < 1e-9);
    assert!((cost.reasoning_cost - 0.03).abs() < 1e-9);
    assert!((cost.total_cost - 2.14).abs() < 1e-9);
}

#[test]
fn missing_prices_are_zero() {
    let usage = Usage::new(100, 100);
    let cost = estimate_cost(&ModelPricing::default(), &usage);
    assert_eq!(cost.total_cost, 0.0);
}

#[test]
fn cost_totals_accumulate() {
    let mut totals = CostTotals::new();
    let usage = Usage::new(1000, 0);
    totals += estimate_cost(&pricing(), &usage);
    totals += estimate_cost(&pricing(), &usage);
    assert!((totals.input_cost - 2.0).abs() < 1e-9);
    assert!((totals.total_cost - 2.0).abs() < 1e-9);
}
