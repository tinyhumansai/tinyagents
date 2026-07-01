//! Deterministic, offline model catalog.
//!
//! Where [`CapabilityRegistry`](crate::registry::CapabilityRegistry) resolves
//! *executable* capabilities by name, this module resolves *facts about
//! models* by name: a checked-in snapshot of provider model prices, context
//! windows, and capability flags. Recursive runs lean on it for the decisions
//! that surround a model call — estimating roll-up cost across parent/child
//! runs, choosing a cheaper sub-model for a delegated step, or gating a feature
//! (tool calling, JSON schema, vision) before a sub-agent is dispatched — all
//! without a network round-trip, so the default offline build stays
//! deterministic.
//!
//! The snapshot is embedded at compile time from
//! `docs/modules/registry/model-catalog.snapshot.json` and loaded via
//! [`ModelCatalog::seed`]; alternative snapshots can be supplied with
//! [`ModelCatalog::from_json`] or [`ModelCatalog::from_snapshot`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

const SEED_SNAPSHOT: &str = include_str!("../../docs/modules/registry/model-catalog.snapshot.json");

/// An in-memory, immutable view over a [`ModelCatalogSnapshot`].
///
/// Construct it from the embedded seed ([`ModelCatalog::seed`]) or from custom
/// JSON, then look entries up by `(provider, model_id)` or by model id alone.
/// Lookups also match an entry's [`aliases`](ModelCatalogEntry::aliases).
#[derive(Clone, Debug)]
pub struct ModelCatalog {
    snapshot: ModelCatalogSnapshot,
}

impl ModelCatalog {
    /// Wraps an already-parsed [`ModelCatalogSnapshot`].
    pub fn from_snapshot(snapshot: ModelCatalogSnapshot) -> Self {
        Self { snapshot }
    }

    /// Parses a catalog from a JSON snapshot string.
    ///
    /// # Errors
    ///
    /// Returns an error if `source` is not a valid [`ModelCatalogSnapshot`].
    pub fn from_json(source: &str) -> Result<Self> {
        let snapshot = serde_json::from_str(source)?;
        Ok(Self::from_snapshot(snapshot))
    }

    /// Loads the catalog from the snapshot embedded in the crate at build time.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded snapshot fails to parse (which would
    /// indicate a corrupted checked-in file).
    pub fn seed() -> Result<Self> {
        Self::from_json(SEED_SNAPSHOT)
    }

    /// Returns the underlying snapshot, including its metadata and sources.
    pub fn snapshot(&self) -> &ModelCatalogSnapshot {
        &self.snapshot
    }

    /// Returns all catalog entries.
    pub fn models(&self) -> &[ModelCatalogEntry] {
        &self.snapshot.models
    }

    /// Looks up an entry by `provider` and `model_id`, matching either the
    /// canonical [`model_id`](ModelCatalogEntry::model_id) or any of its
    /// [`aliases`](ModelCatalogEntry::aliases). Returns `None` if no entry for
    /// that provider matches.
    pub fn get(&self, provider: &str, model_id: &str) -> Option<&ModelCatalogEntry> {
        self.snapshot.models.iter().find(|entry| {
            entry.provider == provider
                && (entry.model_id == model_id
                    || entry.aliases.iter().any(|alias| alias == model_id))
        })
    }

    /// Looks up an entry by model id (or alias) across all providers, returning
    /// the first match. Use [`get`](Self::get) when the provider is known and
    /// the same id might appear under more than one provider.
    pub fn get_by_model_id(&self, model_id: &str) -> Option<&ModelCatalogEntry> {
        self.snapshot.models.iter().find(|entry| {
            entry.model_id == model_id || entry.aliases.iter().any(|alias| alias == model_id)
        })
    }

    /// Hydrates a runtime
    /// [`ModelProfile`][crate::harness::model::ModelProfile] from the catalog
    /// entry for `provider`/`model_id`, bridging offline catalog facts into the
    /// capability profile resolution and fallback consume. Returns `None` when
    /// no entry matches.
    pub fn profile(
        &self,
        provider: &str,
        model_id: &str,
    ) -> Option<crate::harness::model::ModelProfile> {
        self.get(provider, model_id)
            .map(crate::harness::model::ModelProfile::from_catalog_entry)
    }
}

/// The deserialized form of a model-catalog snapshot file.
///
/// Carries provenance metadata (schema version, snapshot id, creation time,
/// pricing currency/unit, and the [`sources`](Self::sources) it was derived
/// from) alongside the list of [`models`](Self::models).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogSnapshot {
    /// Version of the snapshot schema this file conforms to.
    pub schema_version: u32,
    /// Unique identifier for this snapshot revision.
    pub snapshot_id: String,
    /// ISO-8601 timestamp recording when the snapshot was generated.
    pub created_at: String,
    /// Currency that all pricing fields are denominated in (e.g. `"USD"`).
    pub currency: String,
    /// The unit prices are quoted per (e.g. `"token"`).
    pub unit: String,
    /// Optional human-readable description of the snapshot.
    #[serde(default)]
    pub description: Option<String>,
    /// Provenance entries describing where the snapshot data came from.
    #[serde(default)]
    pub sources: Vec<ModelCatalogSource>,
    /// The catalog entries themselves, one per model.
    #[serde(default)]
    pub models: Vec<ModelCatalogEntry>,
}

/// One provenance record for a [`ModelCatalogSnapshot`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogSource {
    /// Human-readable name of the source.
    pub name: String,
    /// URL the data was retrieved from.
    pub url: String,
    /// ISO-8601 timestamp recording when the source was retrieved.
    pub retrieved_at: String,
}

/// A single model's catalog record: identity, limits, pricing, and capability
/// flags.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogEntry {
    /// Provider that serves this model (e.g. `"openai"`, `"anthropic"`).
    pub provider: String,
    /// Canonical model identifier within the provider.
    pub model_id: String,
    /// Alternate identifiers that also resolve to this entry in lookups.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Serving mode for the model (e.g. `"chat"`, `"embedding"`).
    pub mode: String,
    /// Maximum number of input (context) tokens, when known.
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    /// Maximum number of output tokens, when known.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Announced deprecation date, when the provider has published one.
    #[serde(default)]
    pub deprecation_date: Option<String>,
    /// Per-token pricing for the model.
    #[serde(default)]
    pub pricing: ModelPricing,
    /// Capability flags advertised for the model.
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    /// Identifier of the upstream source this entry was derived from.
    pub source: String,
    /// Optional URL pointing at the source documentation for this entry.
    #[serde(default)]
    pub source_url: Option<String>,
    /// Raw provider payload preserved verbatim for fields not modeled above.
    #[serde(default)]
    pub raw: Value,
}

/// Per-token pricing for a model, in the snapshot's currency and unit.
///
/// Every field is optional: a `None` means the price is unknown or not
/// applicable to the model rather than free.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelPricing {
    /// Price per input (prompt) token.
    #[serde(default)]
    pub input_per_token: Option<f64>,
    /// Price per output (completion) token.
    #[serde(default)]
    pub output_per_token: Option<f64>,
    /// Discounted price per input token served from prompt cache.
    #[serde(default)]
    pub cache_read_input_per_token: Option<f64>,
    /// Price per input token when writing to the prompt cache.
    #[serde(default)]
    pub cache_creation_input_per_token: Option<f64>,
    /// Price per audio input token.
    #[serde(default)]
    pub input_audio_per_token: Option<f64>,
    /// Price per reasoning output token (for reasoning models that bill these
    /// separately).
    #[serde(default)]
    pub output_reasoning_per_token: Option<f64>,
}

/// Boolean capability flags advertised for a model.
///
/// These let recursive runs gate behavior before dispatch — for example,
/// refusing to hand tools to a sub-agent backed by a model whose
/// [`tool_calling`](Self::tool_calling) flag is `false`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelCapabilities {
    /// The model supports token streaming.
    #[serde(default)]
    pub streaming: bool,
    /// The model supports tool/function calling.
    #[serde(default)]
    pub tool_calling: bool,
    /// The model can request multiple tool calls in a single turn.
    #[serde(default)]
    pub parallel_tool_calling: bool,
    /// The model supports JSON-schema-constrained structured output.
    #[serde(default)]
    pub json_schema: bool,
    /// The model accepts system messages.
    #[serde(default)]
    pub system_messages: bool,
    /// The model accepts image input.
    #[serde(default)]
    pub vision: bool,
    /// The model accepts audio input.
    #[serde(default)]
    pub audio_input: bool,
    /// The model can produce audio output.
    #[serde(default)]
    pub audio_output: bool,
    /// The model accepts PDF input.
    #[serde(default)]
    pub pdf_input: bool,
    /// The model supports prompt caching.
    #[serde(default)]
    pub prompt_caching: bool,
    /// The model exposes explicit reasoning/thinking.
    #[serde(default)]
    pub reasoning: bool,
}

/// Tests that the embedded seed snapshot loads and that entries resolve by
/// canonical id and by alias.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_seed_model_catalog_snapshot() {
        let catalog = ModelCatalog::seed().unwrap();

        assert_eq!(catalog.snapshot().schema_version, 1);
        assert!(catalog.get("openai", "gpt-4.1").is_some());
        assert!(catalog.get("anthropic", "claude-sonnet-4").is_some());
        assert!(catalog.get("gemini", "gemini-2.5-flash").is_some());
    }

    #[test]
    fn looks_up_model_by_alias_or_id() {
        let catalog = ModelCatalog::seed().unwrap();

        let by_id = catalog.get_by_model_id("gemini/gemini-2.5-pro").unwrap();
        let by_alias = catalog.get_by_model_id("gemini-2.5-pro").unwrap();

        assert_eq!(by_id.model_id, by_alias.model_id);
    }

    #[test]
    fn bridges_catalog_entry_into_runtime_profile() {
        let catalog = ModelCatalog::seed().unwrap();
        let entry = catalog.get("openai", "gpt-4.1").unwrap();

        let profile = crate::harness::model::ModelProfile::from_catalog_entry(entry);
        assert_eq!(profile.provider.as_deref(), Some("openai"));
        assert_eq!(profile.model.as_deref(), Some("gpt-4.1"));
        // The catalog's advertised capability flags carry across the bridge.
        assert_eq!(profile.tool_calling, entry.capabilities.tool_calling);
        assert_eq!(profile.max_input_tokens, entry.max_input_tokens);

        // The convenience accessor returns the same bridged profile.
        let via_catalog = catalog.profile("openai", "gpt-4.1").unwrap();
        assert_eq!(via_catalog, profile);
    }
}
