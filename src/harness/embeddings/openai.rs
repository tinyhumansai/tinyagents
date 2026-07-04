//! [`OpenAiEmbeddingModel`]: an [`EmbeddingModel`] backed by the hosted
//! OpenAI embeddings endpoint (`POST {base_url}/embeddings`).
//!
//! Split out of `embeddings/mod.rs`; mirrors the transport pattern of
//! [`crate::harness::providers::openai::OpenAiModel`].

use async_trait::async_trait;
use serde_json::{Value, json};

use super::EmbeddingModel;
use crate::error::{Result, TinyAgentsError};

/// Default OpenAI embedding model id.
const DEFAULT_MODEL: &str = "text-embedding-3-small";
/// Default OpenAI API base URL (no trailing slash).
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Default dimensionality of `text-embedding-3-small`.
const DEFAULT_DIMENSIONS: usize = 1536;

/// An [`EmbeddingModel`] backed by the hosted OpenAI embeddings endpoint
/// (`POST {base_url}/embeddings`).
///
/// Construct one with [`OpenAiEmbeddingModel::new`] (plus the `with_*`
/// builders) or [`OpenAiEmbeddingModel::from_env`]. The model holds a
/// reusable [`reqwest::Client`] so repeated calls share a connection pool.
///
/// This adapter intentionally mirrors the transport pattern of
/// [`OpenAiModel`](crate::harness::providers::openai::OpenAiModel).
///
/// # Example
/// ```no_run
/// use tinyagents::harness::embeddings::OpenAiEmbeddingModel;
///
/// # fn main() -> tinyagents::Result<()> {
/// let model = OpenAiEmbeddingModel::from_env()?;
/// # let _ = model;
/// # Ok(())
/// # }
/// ```
pub struct OpenAiEmbeddingModel {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    dimensions: usize,
}

impl OpenAiEmbeddingModel {
    /// Creates a model with the given API key, the default model
    /// (`text-embedding-3-small`), and the default base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            dimensions: DEFAULT_DIMENSIONS,
        }
    }

    /// Overrides the embedding model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the API base URL; a trailing slash is trimmed so the joined
    /// endpoint is always `{base_url}/embeddings`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Overrides the reported dimensionality (and requests it from the API
    /// via the `dimensions` parameter, which `text-embedding-3-*` supports).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Builds a model from environment variables.
    ///
    /// Reads `OPENAI_API_KEY` (required), `OPENAI_EMBEDDING_MODEL`
    /// (optional), and `OPENAI_BASE_URL` (optional).
    ///
    /// # Errors
    /// Returns [`TinyAgentsError::Validation`] when `OPENAI_API_KEY` is
    /// missing or empty.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| {
                TinyAgentsError::Validation(
                    "OPENAI_API_KEY is not set; export it or add it to a .env file".to_string(),
                )
            })?;
        let mut model = Self::new(api_key);
        if let Ok(name) = std::env::var("OPENAI_EMBEDDING_MODEL")
            && !name.trim().is_empty()
        {
            model = model.with_model(name);
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL")
            && !url.trim().is_empty()
        {
            model = model.with_base_url(url);
        }
        Ok(model)
    }
}

#[async_trait]
impl EmbeddingModel for OpenAiEmbeddingModel {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url);
        let body = json!({
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                TinyAgentsError::Embedding(format!(
                    "openai embeddings request to {url} failed: {e}"
                ))
            })?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Embedding(format!("openai embeddings body read failed: {e}"))
        })?;
        if !status.is_success() {
            return Err(TinyAgentsError::Embedding(format!(
                "openai embeddings returned HTTP {status}: {text}"
            )));
        }

        let value: Value = serde_json::from_str(&text)?;
        let data = value
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| {
                TinyAgentsError::Embedding("openai embeddings response missing `data` array".into())
            })?;
        let mut vectors = Vec::with_capacity(data.len());
        for item in data {
            let embedding = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    TinyAgentsError::Embedding(
                        "openai embeddings response missing `embedding` array".into(),
                    )
                })?;
            vectors.push(
                embedding
                    .iter()
                    .map(|n| n.as_f64().unwrap_or(0.0) as f32)
                    .collect(),
            );
        }
        Ok(vectors)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
