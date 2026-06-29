# Model Catalog And Local Snapshots

Parent module: [Registry](README.md).

The model catalog is a registry-owned data layer for provider model metadata. It
keeps a local snapshot of model ids, providers, context windows, output limits,
prices, modalities, and capabilities so RustAgents can validate and estimate
runs without hitting provider docs or APIs on every request.

The snapshot is deliberately local and timestamped. Provider pricing and model
limits change frequently, so the catalog must expose provenance and freshness
rather than claiming permanent truth.

## Responsibilities

- Store model metadata from multiple providers.
- Store input, output, cached-input, and special-mode pricing.
- Store context-window and max-output-token data.
- Store capabilities such as streaming, tool calling, vision, audio, JSON
  schema, prompt caching, and reasoning support.
- Store provider source URLs and snapshot time.
- Provide lookup by provider/model id and registry alias.
- Provide metadata used by model resolution, such as capability labels, context
  limits, deprecation state, and price estimates.
- Provide deterministic cost estimates for tests and local development.
- Allow refresh from external sources with an auditable diff.
- Preserve old snapshots for reproducibility when desired.

## Non-Responsibilities

- It is not a live provider billing oracle.
- It does not replace provider-side validation.
- It does not guarantee that a provider still serves a model after the snapshot
  time.
- It does not decide model routing by itself.

## Source Priority

Preferred source order:

1. provider official docs or provider machine-readable APIs when available
2. LiteLLM `model_prices_and_context_window.json`
3. curated RustAgents overrides
4. user application overrides

LiteLLM is useful because it already normalizes many provider price and context
fields into one JSON catalog. Provider official docs remain authoritative when a
conflict is found.

Useful source URLs:

- OpenAI pricing: <https://openai.com/api/pricing/>
- Anthropic pricing: <https://www.anthropic.com/pricing>
- Anthropic models overview: <https://docs.anthropic.com/en/docs/about-claude/models/overview>
- Google Gemini pricing: <https://ai.google.dev/gemini-api/docs/pricing>
- Google Gemini models: <https://ai.google.dev/gemini-api/docs/models>
- Vertex AI generative AI pricing:
  <https://cloud.google.com/vertex-ai/generative-ai/pricing>
- LiteLLM model catalog:
  <https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json>

## Snapshot File

Seed snapshot:

```text
docs/modules/registry/model-catalog.snapshot.json
```

This file is a small checked-in seed for schema design, examples, and tests. It
should not try to mirror every model from every provider. A future implementation
can add a generated snapshot under a crate data path such as:

```text
data/model-catalog/model-catalog.snapshot.json
```

or ship an optional compressed catalog behind a crate feature.

## Snapshot Shape

```json
{
  "schema_version": 1,
  "snapshot_id": "2026-06-29-litellm-seed",
  "created_at": "2026-06-29T00:00:00Z",
  "currency": "USD",
  "unit": "token",
  "sources": [
    {
      "name": "litellm",
      "url": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
      "retrieved_at": "2026-06-29T00:00:00Z"
    }
  ],
  "models": [
    {
      "provider": "openai",
      "model_id": "gpt-4.1",
      "mode": "chat",
      "max_input_tokens": 1047576,
      "max_output_tokens": 32768,
      "pricing": {
        "input_per_token": 0.000002,
        "output_per_token": 0.000008,
        "cache_read_input_per_token": 0.0000005
      },
      "capabilities": {
        "streaming": true,
        "tool_calling": true,
        "parallel_tool_calling": true,
        "json_schema": true,
        "vision": true,
        "prompt_caching": true
      },
      "source": "litellm"
    }
  ]
}
```

## Rust Types

```rust
pub struct ModelCatalog {
    snapshots: Vec<ModelCatalogSnapshot>,
    overrides: Vec<ModelCatalogOverride>,
}

pub struct ModelCatalogSnapshot {
    pub schema_version: u32,
    pub snapshot_id: String,
    pub created_at: SystemTime,
    pub currency: Currency,
    pub unit: PriceUnit,
    pub sources: Vec<ModelCatalogSource>,
    pub models: Vec<ModelCatalogEntry>,
}

pub struct ModelCatalogEntry {
    pub provider: ProviderId,
    pub model_id: String,
    pub aliases: Vec<String>,
    pub mode: ModelMode,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub pricing: ModelPricing,
    pub capabilities: ModelCapabilities,
    pub deprecation_date: Option<NaiveDate>,
    pub source: String,
    pub source_url: Option<String>,
    pub raw: serde_json::Value,
}

pub struct ModelPricing {
    pub input_per_token: Option<Decimal>,
    pub output_per_token: Option<Decimal>,
    pub cache_read_input_per_token: Option<Decimal>,
    pub cache_creation_input_per_token: Option<Decimal>,
    pub input_audio_per_token: Option<Decimal>,
    pub output_reasoning_per_token: Option<Decimal>,
    pub tiers: Vec<PricingTier>,
}

pub struct ModelCapabilities {
    pub streaming: bool,
    pub tool_calling: bool,
    pub parallel_tool_calling: bool,
    pub json_schema: bool,
    pub system_messages: bool,
    pub vision: bool,
    pub audio_input: bool,
    pub audio_output: bool,
    pub pdf_input: bool,
    pub prompt_caching: bool,
    pub reasoning: bool,
}
```

Use a decimal type for money rather than `f64` in implementation. The JSON seed
uses numbers for readability, but the Rust API should parse into exact decimal
values or string-backed decimals.

## Registry Integration

Model registration and model catalog lookup are related but separate:

```rust
registry.models().register("fast", openai_gpt_4_1_mini).await?;

let metadata = registry
    .model_catalog()
    .resolve(ModelRef::provider_model("openai", "gpt-4.1-mini"))?;
```

The model registry answers "what executable model handle should I use?" The
model catalog answers "what do we know about this provider model?"

Common joins:

- a registered model alias points at one catalog entry
- a request estimates price by resolving the model alias to a catalog entry
- a context limiter uses `max_input_tokens` and `max_output_tokens`
- a UI model picker lists executable aliases enriched with catalog metadata
- a harness model resolver rejects candidates that cannot satisfy required
  capabilities
- a resolved-model record stores both executable registry alias and catalog
  provider/model identity for replay

Catalog metadata should never be the only durable identity. A selected model
should be recorded as a `ResolvedModel` containing the executable registry name,
provider id, provider model id, catalog entry id or snapshot id when known, and
resolver source. This lets a later run decide whether it can reuse the same
model even if aliases or catalog snapshots have changed.

## Refresh Workflow

Refresh should be explicit and auditable:

1. Fetch upstream source catalogs.
2. Normalize into RustAgents schema.
3. Merge curated overrides.
4. Preserve raw provider fields.
5. Write a new snapshot with `created_at`, source URLs, and source checksums.
6. Diff old and new snapshots.
7. Run catalog validation tests.
8. Commit the new snapshot only with a clear reason.

Future command shape:

```sh
cargo run --example refresh_model_catalog -- \
  --source litellm \
  --output data/model-catalog/model-catalog.snapshot.json
```

Validation should fail on:

- duplicate `(provider, model_id)` pairs
- negative prices
- missing source
- output limit greater than total context when both are known and incompatible
- model alias collision
- invalid date format
- unknown provider id unless explicitly allowed

## Staleness Policy

Every snapshot should carry freshness metadata:

```rust
pub struct CatalogFreshness {
    pub created_at: SystemTime,
    pub max_age: Duration,
    pub source_count: usize,
    pub stale: bool,
}
```

Default policy:

- tests may use stale snapshots
- local development may warn on stale snapshots
- production cost enforcement should warn or fail when the snapshot is stale,
  depending on application policy
- provider responses always remain authoritative at call time

## Overrides

Applications need overrides for private models, fine-tunes, local models, and
enterprise contracts.

```json
{
  "provider": "openai",
  "model_id": "ft:gpt-4.1-mini:tenant:model",
  "base_model": "gpt-4.1-mini",
  "pricing": {
    "input_per_token": "contract",
    "output_per_token": "contract"
  },
  "source": "tenant_override"
}
```

Overrides should be layered after the base snapshot and marked in discovery
responses so UIs know whether a value came from upstream or local policy.

## Eventing

Catalog operations should emit registry events:

- `catalog.snapshot_loaded`
- `catalog.snapshot_failed`
- `catalog.lookup_started`
- `catalog.lookup_completed`
- `catalog.lookup_failed`
- `catalog.stale`
- `catalog.override_applied`
- `catalog.refresh_started`
- `catalog.refresh_completed`
- `catalog.refresh_failed`

## Milestones

### C1: Snapshot Schema

- checked-in seed snapshot
- serde structs
- validation tests
- lookup by provider/model id

### C2: Registry Join

- registered model alias points to catalog entry
- cost/context lookup helpers
- UI discovery includes catalog metadata

### C3: Refresh Tool

- fetch LiteLLM JSON
- normalize selected fields
- preserve raw payload
- write deterministic snapshot
- produce diff summary

### C4: Provider Overrides

- user override file
- provider-specific normalization patches
- stale snapshot policy
