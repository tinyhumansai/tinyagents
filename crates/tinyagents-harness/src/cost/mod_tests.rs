//! Unit tests for cost accounting.
//!
//! These cover [`super::estimate_cost`] pricing each token class (input,
//! output, cache read/creation, reasoning) against a [`ModelPricing`] entry,
//! the handling of missing prices as zero cost, and [`super::CostTotals`]
//! `+`/`+=` accumulation with total recomputation.

use super::*;
use crate::cost::ModelPricing;
use tinyinference_llm::usage::Usage;

fn contribution(
    input_tokens: u64,
    output_tokens: u64,
    cached_input_tokens: u64,
    known_usd: f64,
    source: CostSource,
) -> UsageContribution {
    UsageContribution {
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cost: CostTally { known_usd, source },
    }
}

#[test]
fn inclusive_subagent_rollup_preserves_root_when_there_are_no_children() {
    let root = contribution(10, 20, 4, 0.75, CostSource::Estimated);

    assert_eq!(roll_up_inclusive_subagent_usage(root, []), root);
}

#[test]
fn inclusive_subagent_rollup_adds_root_and_each_inclusive_child_once() {
    let root = contribution(10, 20, 4, 0.75, CostSource::Charged);
    // These entries already include their own nested usage; the flat API
    // consumes only these totals and has no child tree to traverse.
    let inclusive_child = contribution(7, 5, 3, 0.5, CostSource::Charged);
    let second_child = contribution(11, 2, 8, 0.25, CostSource::Estimated);
    let result = roll_up_inclusive_subagent_usage(root, [inclusive_child, second_child]);

    assert_eq!(result.input_tokens, 28);
    assert_eq!(result.output_tokens, 27);
    assert_eq!(result.cached_input_tokens, 15);
    assert_eq!(result.cost.known_usd, 1.5);
    assert_eq!(result.cost.source, CostSource::Estimated);
    assert_eq!(result.cost.usd(), Some(1.5));
}

#[test]
fn inclusive_subagent_rollup_counts_equal_children_without_identity_deduplication() {
    let child = contribution(7, 5, 3, 0.5, CostSource::Charged);
    let result = roll_up_inclusive_subagent_usage(
        contribution(1, 2, 0, 0.25, CostSource::Charged),
        [child, child],
    );

    assert_eq!(result.input_tokens, 15);
    assert_eq!(result.output_tokens, 12);
    assert_eq!(result.cached_input_tokens, 6);
    assert_eq!(result.cost.known_usd, 1.25);
}

#[test]
fn inclusive_subagent_rollup_saturates_token_totals_and_keeps_cached_as_subset() {
    let result = roll_up_inclusive_subagent_usage(
        contribution(
            u64::MAX - 2,
            u64::MAX - 1,
            u64::MAX - 3,
            1.0,
            CostSource::Charged,
        ),
        [contribution(5, 4, 6, 2.0, CostSource::Charged)],
    );

    assert_eq!(result.input_tokens, u64::MAX);
    assert_eq!(result.output_tokens, u64::MAX);
    assert_eq!(result.cached_input_tokens, u64::MAX);
    assert_eq!(result.cost.known_usd, 3.0);
}

#[test]
fn inclusive_subagent_rollup_retains_known_cost_when_any_source_is_unknown() {
    let result = roll_up_inclusive_subagent_usage(
        contribution(1, 2, 0, 0.75, CostSource::Charged),
        [
            contribution(3, 4, 1, 0.5, CostSource::Estimated),
            contribution(5, 6, 2, 0.0, CostSource::Unknown),
        ],
    );

    assert_eq!(result.cost.known_usd, 1.25);
    assert_eq!(result.cost.source, CostSource::Unknown);
    assert_eq!(result.cost.usd(), None);
}

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
fn subagent_usage_rollup_deduplicates_replayed_runs_but_keeps_distinct_runs() {
    let usage = |run_id: &str, task_id: &str, input_tokens, cost| SubagentUsage {
        run_id: run_id.into(),
        task_id: task_id.into(),
        input_tokens,
        output_tokens: 7,
        cached_input_tokens: 5,
        cost: CostTally {
            known_usd: cost,
            source: CostSource::Estimated,
        },
        children: Vec::new(),
    };
    let first_run = usage("run-1", "task", 12, 0.25);
    let second_run = usage("run-2", "task", 8, 0.5);
    let rollup = roll_up_subagent_usage(
        100,
        20,
        40,
        CostTally {
            known_usd: 1.0,
            source: CostSource::Charged,
        },
        [first_run.clone(), first_run, second_run],
    );
    assert_eq!(rollup.input_tokens, 120);
    assert_eq!(rollup.output_tokens, 34);
    assert_eq!(rollup.cached_input_tokens, 50);
    assert_eq!(rollup.cost.known_usd, 1.75);
    assert_eq!(rollup.cost.source, CostSource::Estimated);
    assert_eq!(rollup.subagents.len(), 2);
}

#[test]
fn model_budget_clones_share_the_same_ledger() {
    use tinyinference_llm::model::budget::{Budget, SpendLimits};

    let budget = ModelBudget::new(
        Budget::new(SpendLimits {
            tokens: Some(50),
            cost_micros: Some(100),
        }),
        tinyinference_llm::model::budget::CallBudget {
            input_tokens: 1_000,
            output_tokens: 10,
            cost_micros: 20,
        },
    );
    let child = budget.clone();
    assert_eq!(child.ledger.snapshot(), budget.ledger.snapshot());
    assert_eq!(child.ledger.snapshot().spent.tokens, 0);
}

#[test]
fn nested_usage_rollup_counts_root_child_and_grandchild_once() {
    let grandchild = SubagentUsage {
        run_id: "grandchild-run".into(),
        task_id: "grandchild".into(),
        input_tokens: 3,
        output_tokens: 2,
        cached_input_tokens: 1,
        cost: CostTally {
            known_usd: 0.3,
            source: CostSource::Charged,
        },
        children: Vec::new(),
    };
    let child = SubagentUsage {
        run_id: "child-run".into(),
        task_id: "child".into(),
        input_tokens: 5,
        output_tokens: 4,
        cached_input_tokens: 2,
        cost: CostTally {
            known_usd: 0.5,
            source: CostSource::Charged,
        },
        children: vec![grandchild.clone()],
    };
    let rollup = roll_up_subagent_usage(
        10,
        8,
        4,
        CostTally {
            known_usd: 1.0,
            source: CostSource::Charged,
        },
        [child, grandchild],
    );
    assert_eq!(rollup.input_tokens, 18);
    assert_eq!(rollup.output_tokens, 14);
    assert_eq!(rollup.cached_input_tokens, 7);
    assert_eq!(rollup.cost.known_usd, 1.8);
    assert_eq!(rollup.subagents.len(), 2);
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
        charged_amount: None,
        context_window_tokens: None,
    };
    let cost = estimate_cost(&pricing(), &usage);
    // cache_read_tokens (100) and reasoning_tokens (10) are subsets of
    // input_tokens/output_tokens, so the standard-rate cost only applies to
    // the non-cached/non-reasoning remainder: (1000-100)*0.001 = 0.9,
    // (500-10)*0.002 = 0.98.
    assert!((cost.input_cost - 0.9).abs() < 1e-9);
    assert!((cost.output_cost - 0.98).abs() < 1e-9);
    // 100*0.0001 + 200*0.0005 = 0.01 + 0.1 = 0.11
    assert!((cost.cache_cost - 0.11).abs() < 1e-9);
    assert!((cost.reasoning_cost - 0.03).abs() < 1e-9);
    assert!((cost.total_cost - 2.02).abs() < 1e-9);
}

#[test]
fn cache_and_reasoning_tokens_are_not_double_counted() {
    // A fully-cached prompt: all 90k input tokens were served from cache.
    // Charging the full input_tokens at the standard rate *and* the
    // cache_read_tokens at the cache rate would double-bill the cached
    // tokens; only the cache rate should apply to them.
    let usage = Usage {
        input_tokens: 90_000,
        output_tokens: 100,
        total_tokens: 90_100,
        cache_read_tokens: 90_000,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
        charged_amount: None,
        context_window_tokens: None,
    };
    let cost = estimate_cost(&pricing(), &usage);
    assert_eq!(cost.input_cost, 0.0);
    assert!((cost.cache_cost - 9.0).abs() < 1e-9);
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

// ---------------------------------------------------------------------------
// Tiered pricing
// ---------------------------------------------------------------------------

fn tiered_pricing() -> ModelPricing {
    ModelPricing {
        // Flat rate is the fallback the highest tier partially relies on.
        input_per_token: Some(0.001),
        output_per_token: Some(0.002),
        tiers: vec![
            PriceTier {
                up_to_tokens: Some(200_000),
                input: Some(0.001),
                output: Some(0.002),
                cache_read: Some(0.0001),
                cache_write: None,
            },
            PriceTier {
                up_to_tokens: None,
                input: Some(0.002),
                output: Some(0.004),
                cache_read: None,
                cache_write: None,
            },
        ],
        ..ModelPricing::default()
    }
}

#[test]
fn tiered_pricing_uses_the_matching_capped_tier() {
    let usage = Usage::new(100_000, 1000);
    let cost = estimate_cost(&tiered_pricing(), &usage);
    assert!((cost.input_cost - 100.0).abs() < 1e-9);
    assert!((cost.output_cost - 2.0).abs() < 1e-9);
}

#[test]
fn tiered_pricing_falls_through_to_the_unlimited_tier_above_the_cap() {
    let usage = Usage::new(300_000, 1000);
    let cost = estimate_cost(&tiered_pricing(), &usage);
    // The unlimited tier's higher input rate (0.002) applies once the call's
    // input tokens exceed the capped tier's 200K threshold.
    assert!((cost.input_cost - 600.0).abs() < 1e-9);
    assert!((cost.output_cost - 4.0).abs() < 1e-9);
}

#[test]
fn tiered_pricing_falls_back_to_flat_rate_for_unset_tier_fields() {
    // The unlimited tier leaves `cache_read` unset; it must fall back to the
    // flat `cache_read_input_per_token` rather than pricing at zero.
    let mut pricing = tiered_pricing();
    pricing.cache_read_input_per_token = Some(0.0002);
    let usage = Usage {
        input_tokens: 300_000,
        output_tokens: 0,
        total_tokens: 300_000,
        cache_read_tokens: 50_000,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
        charged_amount: None,
        context_window_tokens: None,
    };
    let cost = estimate_cost(&pricing, &usage);
    assert!((cost.cache_cost - 10.0).abs() < 1e-9);
}

#[test]
fn no_tiers_declared_uses_flat_pricing() {
    assert!(select_tier(&pricing(), 1_000_000).is_none());
}

#[test]
fn select_tier_picks_the_smallest_sufficient_cap() {
    let pricing = tiered_pricing();
    assert_eq!(
        select_tier(&pricing, 50_000).unwrap().up_to_tokens,
        Some(200_000)
    );
    assert_eq!(
        select_tier(&pricing, 200_000).unwrap().up_to_tokens,
        Some(200_000)
    );
    assert_eq!(select_tier(&pricing, 200_001).unwrap().up_to_tokens, None);
}
