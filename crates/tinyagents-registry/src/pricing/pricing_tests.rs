use super::*;

#[test]
fn exact_lookup_resolves_canonical_ids() {
    let p = lookup("claude-opus-4-8").expect("anthropic row");
    assert_eq!(p.provider, "anthropic");
    assert_eq!(p.input_per_mtok_usd, 5.00);
    assert_eq!(p.output_per_mtok_usd, 25.00);
    assert_eq!(p.cached_input_per_mtok_usd, 0.50);
    assert_eq!(p.context_window, 1_000_000);
}

#[test]
fn lookup_is_case_insensitive() {
    assert_eq!(lookup("GPT-4.1").unwrap().model_id, "gpt-4.1");
}

#[test]
fn lookup_strips_vendor_prefix_openrouter_style() {
    assert_eq!(
        lookup("anthropic/claude-sonnet-4-6").unwrap().model_id,
        "claude-sonnet-4-6"
    );
    assert_eq!(lookup("moonshot/kimi-k2.6").unwrap().model_id, "kimi-k2.6");
    assert_eq!(lookup("qwen/qwen-plus").unwrap().model_id, "qwen-plus");
}

#[test]
fn lookup_strips_context_and_tag_decorations() {
    assert_eq!(
        lookup("claude-opus-4-8[1m]").unwrap().model_id,
        "claude-opus-4-8"
    );
    assert_eq!(lookup("kimi-k2.6:turbo").unwrap().model_id, "kimi-k2.6");
    assert_eq!(
        lookup("claude-opus-4-5@20251101").unwrap().model_id,
        "claude-opus-4-5"
    );
}

#[test]
fn lookup_resolves_dated_ids_to_the_most_specific_row() {
    // `gpt-5.4` is also a prefix here, but `-mini-2026-05-01` is not a
    // decoration, so only the `gpt-5.4-mini` row qualifies.
    assert_eq!(
        lookup("gpt-5.4-mini-2026-05-01").unwrap().model_id,
        "gpt-5.4-mini"
    );
    assert_eq!(
        lookup("claude-sonnet-4-5-20250929").unwrap().model_id,
        "claude-sonnet-4-5"
    );
    assert_eq!(
        lookup("gemini-2.5-pro-preview-06-05").unwrap().model_id,
        "gemini-2.5-pro"
    );
    assert_eq!(
        lookup("gemini-2.5-flash-latest").unwrap().model_id,
        "gemini-2.5-flash"
    );
    assert_eq!(
        lookup("openai/gpt-5.4-2026-03-01").unwrap().model_id,
        "gpt-5.4"
    );
}

#[test]
fn unknown_variants_do_not_inherit_a_neighbours_rates() {
    // Each of these extends a catalogued id with something that is not a
    // snapshot decoration: a different model, which has its own prices.
    for id in ["gpt-4.1-nano", "o3-pro", "o3-mini", "gpt-5.4-mini-high"] {
        assert!(
            lookup(id).is_none(),
            "{id} resolved to {:?}",
            lookup(id).map(|p| p.model_id)
        );
    }
    // The estimator therefore reports "no estimate" rather than gpt-4.1's rates.
    assert_eq!(
        estimate_cost_usd("gpt-4.1-nano", 1_000_000, 1_000_000, 0),
        0.0
    );
}

#[test]
fn lookup_returns_none_for_unknown() {
    assert!(lookup("totally-made-up-model").is_none());
    assert!(lookup("").is_none());
    assert!(
        lookup("agentic-v1").is_none(),
        "abstract tiers aren't vendor models"
    );
}

#[test]
fn every_row_has_sane_values() {
    for p in KNOWN_MODEL_PRICING {
        assert!(p.input_per_mtok_usd > 0.0, "{}", p.model_id);
        assert!(p.output_per_mtok_usd > 0.0, "{}", p.model_id);
        assert!(p.context_window > 0, "{}", p.model_id);
        assert!(
            p.cached_input_per_mtok_usd <= p.input_per_mtok_usd,
            "{} cached should not exceed input",
            p.model_id
        );
    }
}

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
}

#[test]
fn estimate_prices_standard_input_and_output() {
    // opus-4-8: $5/$25 per MTok in/out. 1M in + 1M out, no cache.
    approx(
        estimate_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000, 0),
        30.00,
    );
}

#[test]
fn estimate_bills_cached_prefix_at_the_cheaper_rate() {
    // Fully cached input (cached == input) → cached rate only ($0.50/MTok).
    approx(
        estimate_cost_usd("claude-opus-4-8", 1_000_000, 0, 1_000_000),
        0.50,
    );
    // Half cached: 0.5M standard @ $5 + 0.5M cached @ $0.50 = 2.50 + 0.25.
    approx(
        estimate_cost_usd("claude-opus-4-8", 1_000_000, 0, 500_000),
        2.75,
    );
}

#[test]
fn estimate_clamps_cached_to_input() {
    // cached_input_tokens > input_tokens must not underflow or overcharge:
    // it is clamped to input, so this is billed as fully cached.
    approx(
        estimate_cost_usd("claude-opus-4-8", 1_000_000, 0, 5_000_000),
        0.50,
    );
}

#[test]
fn estimate_returns_zero_for_uncatalogued_models() {
    // "unknown, not free" — the caller treats 0.0 as no estimate available.
    assert_eq!(
        estimate_cost_usd("totally-made-up-model", 1_000_000, 1_000_000, 0),
        0.0
    );
}

#[test]
fn estimate_resolves_decorated_model_ids() {
    // The catalog lookup normalizes tags/suffixes, so a decorated id
    // (e.g. the runtime "[1m]" window tag) still prices correctly.
    approx(
        estimate_cost_usd("claude-opus-4-8[1m]", 1_000_000, 0, 0),
        5.00,
    );
}

#[test]
fn minimax_m3_uses_standard_tier_through_512k() {
    approx(estimate_cost_usd("minimax-m3", 512_000, 0, 0), 0.1536);
    approx(estimate_cost_usd("minimax-m3", 512_001, 0, 0), 0.3072006);
}

#[test]
fn short_ids_do_not_match_inside_unrelated_tokens() {
    assert!(lookup("proto3-chat").is_none());
    assert!(lookup("solo1-7b").is_none());
    assert_eq!(
        lookup("gpt-5.4-mini-2026-05-01").unwrap().model_id,
        "gpt-5.4-mini"
    );
}

#[test]
fn catalogue_tiers_and_call_estimates_agree_at_every_step() {
    use tinyagents_harness::cost::{ModelPricing, estimate_cost};
    use tinyinference_llm::usage::Usage;
    for tier in LONG_CONTEXT_TIERS {
        let row = lookup(tier.model_id).unwrap();
        let pricing: ModelPricing = flat_catalog_entry(row, "consumer").pricing;
        assert_eq!(pricing.tiers.len(), 2, "{}", tier.model_id);
        for input in [1_000, tier.above_input_tokens, tier.above_input_tokens + 1] {
            let usage = Usage {
                cache_read_tokens: input / 2,
                ..Usage::new(input, 1_000)
            };
            approx(
                estimate_cost(&pricing, &usage).total_cost,
                estimate_cost_usd(tier.model_id, input, 1_000, input / 2),
            );
        }
    }
}

#[test]
fn long_context_steps_bill_the_whole_call_at_the_higher_rate() {
    approx(estimate_cost_usd("gpt-5.5", 272_000, 0, 0), 1.36);
    approx(estimate_cost_usd("gpt-5.5", 272_001, 0, 0), 2.72001);
    approx(
        estimate_cost_usd("gemini-2.5-pro", 200_001, 1_000, 0),
        0.5150025,
    );
    approx(estimate_cost_usd("gpt-5.4-mini", 400_000, 0, 0), 0.30);
}

#[test]
fn rows_without_a_step_have_no_catalogue_tiers() {
    let entry = flat_catalog_entry(lookup("claude-opus-4-8").unwrap(), "consumer");
    assert!(entry.pricing.tiers.is_empty());
    assert!(long_context_tier("claude-opus-4-8").is_none());
    for tier in LONG_CONTEXT_TIERS {
        assert!(
            lookup(tier.model_id).is_some(),
            "{} has no row",
            tier.model_id
        );
    }
}

#[test]
fn image_support_resolves_decorated_minimax_ids() {
    assert!(model_accepts_image_input("minimax/minimax-m3-2026-07-01"));
}

#[test]
fn free_model_pricing_is_zero_and_does_not_change_reported_charges() {
    let pricing = lookup_pricing("vendor/model:FREE").expect("explicit free model");
    assert_eq!(pricing.input_per_token, Some(0.0));
    assert_eq!(pricing.output_per_token, Some(0.0));
    assert_eq!(pricing.cache_read_input_per_token, Some(0.0));

    let usage = tinyinference_llm::usage::Usage {
        input_tokens: 100,
        output_tokens: 20,
        ..Default::default()
    };
    assert_eq!(
        tinyagents_harness::cost::call_cost(
            Some(&pricing),
            &tinyagents_harness::cost::CallUsage {
                usage,
                charged_amount_usd: 0.004,
                cost_is_estimate: false,
                charge_reported: true,
            },
        )
        .usd(),
        Some(0.004),
        "provider-reported billing remains authoritative for free-labeled models"
    );
}

#[test]
fn approximate_cached_rates_do_not_advertise_prompt_caching() {
    let mistral = flat_catalog_entry(lookup("mistral-small").unwrap(), "consumer");
    assert!(!mistral.capabilities.prompt_caching);
}

#[test]
fn gpt5_and_gemini_cache_reads_are_a_tenth_of_input() {
    // #393: these were filled in as 0.25× input; the published rate is 0.1×.
    for id in [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ] {
        let row = lookup(id).unwrap();
        approx(row.cached_input_per_mtok_usd, row.input_per_mtok_usd / 10.0);
    }
}

#[test]
fn retired_models_are_gone_and_renamed_mistral_ids_carry_their_new_models() {
    for id in [
        "deepseek-chat",
        "deepseek-reasoner",
        "kimi-k2.5",
        "qwen3-max",
        "minimax-m2.7",
        "minimax-m2.7-highspeed",
        "mistral-medium",
    ] {
        assert!(lookup(id).is_none(), "{id} is retired");
        approx(estimate_cost_usd(id, 1_000, 1_000, 0), 0.0);
    }
    for (id, input, cached, output, window) in [
        ("mistral-large", 0.70, 0.07, 2.10, 1_000_000),
        ("mistral-small", 0.15, 0.015, 0.60, 256_000),
        ("ministral-8b", 0.15, 0.015, 0.15, 256_000),
    ] {
        let row = lookup(id).unwrap();
        approx(row.input_per_mtok_usd, input);
        approx(row.cached_input_per_mtok_usd, cached);
        approx(row.output_per_mtok_usd, output);
        assert_eq!(row.context_window, window, "{id}");
    }
    assert_eq!(
        lookup("ministral-8b-2512").unwrap().model_id,
        "ministral-8b"
    );
}
