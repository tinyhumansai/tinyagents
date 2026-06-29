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
    #[serde(default)]
    pub sources: Vec<ModelCatalogSource>,
    #[serde(default)]
    pub models: Vec<ModelCatalogEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogSource {
    pub name: String,
    pub url: String,
    pub retrieved_at: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogEntry {
    pub provider: String,
    pub model_id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub mode: String,
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub deprecation_date: Option<String>,
    #[serde(default)]
    pub pricing: ModelPricing,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    pub source: String,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub raw: Value,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelPricing {
    #[serde(default)]
    pub input_per_token: Option<f64>,
    #[serde(default)]
    pub output_per_token: Option<f64>,
    #[serde(default)]
    pub cache_read_input_per_token: Option<f64>,
    #[serde(default)]
    pub cache_creation_input_per_token: Option<f64>,
    #[serde(default)]
    pub input_audio_per_token: Option<f64>,
    #[serde(default)]
    pub output_reasoning_per_token: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub tool_calling: bool,
    #[serde(default)]
    pub parallel_tool_calling: bool,
    #[serde(default)]
    pub json_schema: bool,
    #[serde(default)]
    pub system_messages: bool,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub audio_input: bool,
    #[serde(default)]
    pub audio_output: bool,
    #[serde(default)]
    pub pdf_input: bool,
    #[serde(default)]
    pub prompt_caching: bool,
    #[serde(default)]
    pub reasoning: bool,
}

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
}
