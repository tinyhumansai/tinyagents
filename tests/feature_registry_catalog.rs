//! Feature tests for the offline model catalog surface
//! ([`ModelCatalog`]) of `src/registry/catalog.rs`.
//!
//! These exercise the *user-facing* catalog features a recursive run leans on
//! for the decisions surrounding a model call — deterministic, offline lookup
//! of prices, context windows, and capability flags — without a network round
//! trip. To stay fully deterministic and independent of the checked-in seed
//! prices (which may change), every test builds its own small snapshot from
//! JSON via [`ModelCatalog::from_json`]; one test additionally confirms the
//! embedded seed loads.

use tinyagents::registry::{ModelCatalog, ModelCatalogSnapshot};

/// A tiny, fully-specified two-model snapshot: a cheap tool-capable chat model
/// and a pricier vision/JSON-schema model. Pricing and capability flags are
/// chosen so cost and gating assertions are exact.
const CUSTOM_SNAPSHOT: &str = r#"
{
  "schema_version": 1,
  "snapshot_id": "test-snapshot-1",
  "created_at": "2026-01-01T00:00:00Z",
  "currency": "USD",
  "unit": "token",
  "description": "deterministic offline test catalog",
  "sources": [
    { "name": "hand-written", "url": "https://example.test/catalog", "retrieved_at": "2026-01-01T00:00:00Z" }
  ],
  "models": [
    {
      "provider": "acme",
      "model_id": "acme-small",
      "aliases": ["small", "acme/acme-small"],
      "mode": "chat",
      "max_input_tokens": 8000,
      "max_output_tokens": 2000,
      "pricing": { "input_per_token": 0.000001, "output_per_token": 0.000002 },
      "capabilities": { "streaming": true, "tool_calling": true, "system_messages": true },
      "source": "hand-written"
    },
    {
      "provider": "acme",
      "model_id": "acme-large",
      "aliases": ["large"],
      "mode": "chat",
      "max_input_tokens": 200000,
      "max_output_tokens": 16000,
      "pricing": { "input_per_token": 0.00001, "output_per_token": 0.00003 },
      "capabilities": { "tool_calling": true, "vision": true, "json_schema": true },
      "source": "hand-written"
    }
  ]
}
"#;

fn custom_catalog() -> ModelCatalog {
    ModelCatalog::from_json(CUSTOM_SNAPSHOT).expect("custom snapshot parses")
}

#[test]
fn embedded_seed_catalog_loads_with_provenance_and_entries() {
    let catalog = ModelCatalog::seed().expect("embedded seed snapshot loads");
    let snapshot = catalog.snapshot();

    // Provenance metadata is populated and the model list is non-empty.
    assert!(snapshot.schema_version >= 1);
    assert_eq!(snapshot.currency, "USD");
    assert_eq!(snapshot.unit, "token");
    assert!(!catalog.models().is_empty(), "seed carries model entries");
}

#[test]
fn custom_snapshot_exposes_metadata_sources_and_entries() {
    let catalog = custom_catalog();
    let snapshot = catalog.snapshot();

    assert_eq!(snapshot.schema_version, 1);
    assert_eq!(snapshot.snapshot_id, "test-snapshot-1");
    assert_eq!(snapshot.currency, "USD");
    assert_eq!(snapshot.unit, "token");
    assert_eq!(snapshot.sources.len(), 1);
    assert_eq!(snapshot.sources[0].name, "hand-written");
    assert_eq!(catalog.models().len(), 2);
}

#[test]
fn looks_up_models_by_provider_id_and_by_alias() {
    let catalog = custom_catalog();

    // Provider + canonical id.
    let by_id = catalog.get("acme", "acme-small").expect("canonical lookup");
    assert_eq!(by_id.model_id, "acme-small");

    // Provider + alias resolves to the same entry.
    let by_alias = catalog.get("acme", "small").expect("alias lookup");
    assert_eq!(by_alias.model_id, "acme-small");

    // Cross-provider lookup by id alone (first match), including via alias.
    assert_eq!(
        catalog.get_by_model_id("large").expect("alias id").model_id,
        "acme-large"
    );
    assert_eq!(
        catalog
            .get_by_model_id("acme/acme-small")
            .expect("namespaced alias")
            .model_id,
        "acme-small"
    );
}

#[test]
fn unknown_lookups_return_none_rather_than_error() {
    let catalog = custom_catalog();

    assert!(catalog.get("acme", "ghost-model").is_none());
    // Right id, wrong provider.
    assert!(catalog.get("other", "acme-small").is_none());
    assert!(catalog.get_by_model_id("nonexistent").is_none());
}

#[test]
fn pricing_supports_deterministic_rollup_cost_estimation() {
    let catalog = custom_catalog();

    // A parent step on the large model plus a delegated child step on the
    // cheaper small model. Estimating roll-up cost across the parent/child run
    // is exactly what the offline catalog is for.
    let large = catalog.get("acme", "acme-large").expect("large entry");
    let small = catalog.get("acme", "acme-small").expect("small entry");

    let cost = |entry: &tinyagents::registry::ModelCatalogEntry, input: f64, output: f64| -> f64 {
        input * entry.pricing.input_per_token.unwrap_or(0.0)
            + output * entry.pricing.output_per_token.unwrap_or(0.0)
    };

    // Parent: 1000 in, 500 out on the large model.
    let parent = cost(large, 1000.0, 500.0);
    assert!((parent - (1000.0 * 0.00001 + 500.0 * 0.00003)).abs() < 1e-12);

    // Child: 2000 in, 1000 out on the small model.
    let child = cost(small, 2000.0, 1000.0);
    assert!((child - (2000.0 * 0.000001 + 1000.0 * 0.000002)).abs() < 1e-12);

    // The recursive roll-up is the sum. Choosing the cheaper sub-model keeps
    // the delegated leg materially below the parent leg.
    let rollup = parent + child;
    assert!((rollup - (parent + child)).abs() < 1e-12);
    assert!(
        child < parent,
        "the cheaper sub-model costs less for more work"
    );
}

#[test]
fn capability_flags_gate_feature_dispatch() {
    let catalog = custom_catalog();

    let small = catalog.get("acme", "acme-small").expect("small entry");
    let large = catalog.get("acme", "acme-large").expect("large entry");

    // Tool calling: only hand tools to a sub-agent whose model advertises it.
    assert!(small.capabilities.tool_calling);
    assert!(large.capabilities.tool_calling);

    // Vision / JSON-schema: gated to the large model only.
    assert!(!small.capabilities.vision);
    assert!(large.capabilities.vision);
    assert!(!small.capabilities.json_schema);
    assert!(large.capabilities.json_schema);

    // Context windows are available for pre-dispatch budgeting.
    assert_eq!(small.max_input_tokens, Some(8000));
    assert_eq!(large.max_input_tokens, Some(200000));
}

#[test]
fn bridges_catalog_entry_into_a_runtime_profile() {
    let catalog = custom_catalog();

    // The convenience accessor hydrates a runtime ModelProfile so offline
    // catalog facts flow into capability-profile resolution.
    let profile = catalog.profile("acme", "acme-large").expect("profile");
    assert_eq!(profile.provider.as_deref(), Some("acme"));
    assert_eq!(profile.model.as_deref(), Some("acme-large"));
    assert!(profile.tool_calling);
    assert_eq!(profile.max_input_tokens, Some(200000));

    // A missing entry yields no profile.
    assert!(catalog.profile("acme", "ghost").is_none());
}

#[test]
fn snapshot_round_trips_through_serde() {
    let catalog = custom_catalog();
    let json = serde_json::to_string(catalog.snapshot()).expect("serialize");
    let back: ModelCatalogSnapshot = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back.models.len(), 2);
    assert_eq!(back.snapshot_id, "test-snapshot-1");
    // Re-wrapping the round-tripped snapshot preserves lookups.
    let rewrapped = ModelCatalog::from_snapshot(back);
    assert_eq!(
        rewrapped.get("acme", "small").expect("alias").model_id,
        "acme-small"
    );
}
