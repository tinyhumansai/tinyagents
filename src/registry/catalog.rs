use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

const SEED_SNAPSHOT: &str =
    include_str!("../../docs/modules/registry/model-catalog.snapshot.json");

#[derive(Clone, Debug)]
pub struct ModelCatalog {
    snapshot: ModelCatalogSnapshot,
}

impl ModelCatalog {
    pub fn from_snapshot(snapshot: ModelCatalogSnapshot) -> Self {
        Self { snapshot }
    }

    pub fn from_json(source: &str) -> Result<Self> {
        let snapshot = serde_json::from_str(source)?;
        Ok(Self::from_snapshot(snapshot))
    }

    pub fn seed() -> Result<Self> {
        Self::from_json(SEED_SNAPSHOT)
    }

    pub fn snapshot(&self) -> &ModelCatalogSnapshot {
        &self.snapshot
    }

    pub fn models(&self) -> &[ModelCatalogEntry] {
        &self.snapshot.models
    }

    pub fn get(&self, provider: &str, model_id: &str) -> Option<&ModelCatalogEntry> {
        self.snapshot.models.iter().find(|entry| {
            entry.provider == provider
                && (entry.model_id == model_id || entry.aliases.iter().any(|alias| alias == model_id))
        })
    }

    pub fn get_by_model_id(&self, model_id: &str) -> Option<&ModelCatalogEntry> {
        self.snapshot
            .models
            .iter()
            .find(|entry| entry.model_id == model_id || entry.aliases.iter().any(|alias| alias == model_id))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogSnapshot {
    pub schema_version: u32,
    pub snapshot_id: String,
    pub created_at: String,
    pub currency: String,
    pub unit: String,
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
