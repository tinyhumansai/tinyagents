//! Feature/integration tests for the harness token-usage and cost-accounting
//! infrastructure (`harness::usage` + `harness::cost`).
//!
//! These exercise the additive roll-up contracts that let a recursive run tree
//! fold per-call token counts and cost into a single total: `Usage`/`UsageTotals`
//! accumulation and the `estimate_cost` pricing rules (cache/reasoning subset
//! handling, missing-price behaviour, and `CostTotals` composition).
//!
//! Deterministic and fully offline — no provider calls.

use tinyagents::harness::cost::{CostTotals, estimate_cost};
use tinyagents::harness::usage::{Usage, UsageTotals};
use tinyagents::registry::catalog::ModelPricing;

// ── Usage accumulation ─────────────────────────────────────────────────────

#[test]
fn usage_new_sets_total_to_input_plus_output() {
    let usage = Usage::new(120, 30);
    assert_eq!(usage.input_tokens, 120);
    assert_eq!(usage.output_tokens, 30);
    assert_eq!(usage.total_tokens, 150);
    assert_eq!(usage.effective_total(), 150);
}

#[test]
fn effective_total_falls_back_when_provider_omitted_total() {
    // A record built by hand with total_tokens left at 0 still reports a
    // meaningful total via effective_total().
    let usage = Usage {
        input_tokens: 40,
        output_tokens: 10,
        total_tokens: 0,
        ..Usage::default()
    };
    assert_eq!(usage.effective_total(), 50);
}

#[test]
fn usage_add_sums_every_component_and_preserves_effective_total() {
    let a = Usage {
        input_tokens: 100,
        output_tokens: 20,
        total_tokens: 0, // provider omitted the total on this record
        cache_read_tokens: 30,
        cache_creation_tokens: 5,
        reasoning_tokens: 8,
    };
    let b = Usage::new(50, 10); // total_tokens = 60

    let combined = a + b;

    assert_eq!(combined.input_tokens, 150);
    assert_eq!(combined.output_tokens, 30);
    assert_eq!(combined.cache_read_tokens, 30);
    assert_eq!(combined.cache_creation_tokens, 5);
    assert_eq!(combined.reasoning_tokens, 8);
    // a.effective_total() (120, from input+output since total was 0) + 60.
    assert_eq!(combined.total_tokens, 180);
    assert_eq!(combined.effective_total(), 180);
}

#[test]
fn usage_add_assign_matches_add() {
    let mut acc = Usage::new(10, 2);
    acc += Usage::new(3, 1);
    assert_eq!(acc, Usage::new(13, 3));
}

#[test]
fn usage_totals_record_tracks_call_count_and_summed_usage() {
    let mut totals = UsageTotals::new();
    totals.record(Usage::new(10, 4));
    totals.record(Usage::new(6, 1));
    totals += Usage::new(4, 0);

    assert_eq!(totals.calls, 3);
    assert_eq!(totals.usage.input_tokens, 20);
    assert_eq!(totals.usage.output_tokens, 5);
    assert_eq!(totals.usage.effective_total(), 25);
}

#[test]
fn usage_totals_add_operator_records_one_call() {
    let totals = UsageTotals::new() + Usage::new(7, 3) + Usage::new(1, 1);
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.effective_total(), 12);
}

// ── Cost estimation ────────────────────────────────────────────────────────

fn pricing() -> ModelPricing {
    ModelPricing {
        input_per_token: Some(2.0),
        output_per_token: Some(4.0),
        cache_read_input_per_token: Some(0.5),
        cache_creation_input_per_token: Some(1.0),
        output_reasoning_per_token: Some(6.0),
        ..ModelPricing::default()
    }
}

#[test]
fn estimate_cost_prices_only_non_cached_non_reasoning_remainder() {
    // 100 input tokens, 20 of which were cache reads → 80 billed at 2.0 = 160.
    // 50 output tokens, 10 of which were reasoning → 40 billed at 4.0 = 160.
    // cache: 20 read * 0.5 + 4 creation * 1.0 = 14.
    // reasoning: 10 * 6.0 = 60.
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        cache_read_tokens: 20,
        cache_creation_tokens: 4,
        reasoning_tokens: 10,
    };
    let totals = estimate_cost(&pricing(), &usage);

    assert_eq!(totals.input_cost, 160.0);
    assert_eq!(totals.output_cost, 160.0);
    assert_eq!(totals.cache_cost, 14.0);
    assert_eq!(totals.reasoning_cost, 60.0);
    assert_eq!(totals.total_cost, 160.0 + 160.0 + 14.0 + 60.0);
}

#[test]
fn missing_prices_contribute_zero_rather_than_erroring() {
    // A pricing table with no rates at all yields an all-zero cost.
    let usage = Usage::new(1000, 500);
    let totals = estimate_cost(&ModelPricing::default(), &usage);
    assert_eq!(totals, CostTotals::default());
    assert_eq!(totals.total_cost, 0.0);
}

#[test]
fn cache_read_never_double_charges_the_input_rate() {
    // When every input token is a cache read, the standard input rate applies
    // to none of them — only the cache-read rate does.
    let usage = Usage {
        input_tokens: 40,
        cache_read_tokens: 40,
        ..Usage::default()
    };
    let totals = estimate_cost(&pricing(), &usage);
    assert_eq!(totals.input_cost, 0.0);
    assert_eq!(totals.cache_cost, 40.0 * 0.5);
}

#[test]
fn cost_totals_accumulate_across_calls() {
    let a = estimate_cost(&pricing(), &Usage::new(10, 5));
    let b = estimate_cost(&pricing(), &Usage::new(20, 10));

    let rolled = a + b;
    // input: (10+20)*2 = 60, output: (5+10)*4 = 60.
    assert_eq!(rolled.input_cost, 60.0);
    assert_eq!(rolled.output_cost, 60.0);
    assert_eq!(rolled.total_cost, 120.0);
}

#[test]
fn cost_totals_add_assign_recomputes_total() {
    let mut acc = CostTotals::new();
    acc += estimate_cost(&pricing(), &Usage::new(1, 0));
    acc += estimate_cost(&pricing(), &Usage::new(0, 1));
    // input 1*2 + output 1*4 = 6.
    assert_eq!(acc.total_cost, 6.0);
}
