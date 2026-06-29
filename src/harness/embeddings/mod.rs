//! Harness embeddings + retrieval module.
//!
//! Embeddings are provider-neutral dense vector representations used for
//! retrieval, semantic search, and retrieval-augmented prompt context. This
//! module owns:
//!
//! - [`EmbeddingModel`] — the trait every embedding provider implements.
//! - [`MockEmbeddingModel`] — a deterministic, offline implementation for tests.
//! - [`VectorStore`] — a trait for storing and searching dense vectors.
//! - [`InMemoryVectorStore`] — an in-process cosine-similarity vector store.
//! - [`Retriever`] — ties an embedding model and vector store together for
//!   indexing documents and answering text queries.
//! - [`cosine_similarity`] — the distance metric used by the in-memory store.
//!
//! With the `openai` feature enabled, [`OpenAiEmbeddingModel`] adds a hosted
//! provider backed by the OpenAI embeddings endpoint.
//!
//! The design mirrors LangChain's separation of concerns: chat models generate
//! messages, embedding models generate vectors, vector stores search vectors,
//! and retrievers return documents. Prompt and middleware code decides what
//! retrieved context enters a model request.
//!
//! # Example
//! ```
//! use std::sync::Arc;
//! use tinyagents::harness::embeddings::{InMemoryVectorStore, MockEmbeddingModel, Retriever};
//! use serde_json::json;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let retriever = Retriever::new(
//!     Arc::new(MockEmbeddingModel::new(64)),
//!     Arc::new(InMemoryVectorStore::new()),
//! );
//! retriever
//!     .index(vec![
//!         ("cats".into(), "cats are great pets".into(), json!({"topic": "animals"})),
//!         ("finance".into(), "the stock market crashed".into(), json!({"topic": "finance"})),
//!     ])
//!     .await
//!     .unwrap();
//!
//! let hits = retriever.retrieve("cats are great pets", 1).await.unwrap();
//! assert_eq!(hits[0].id, "cats");
//! # });
//! ```

mod types;

pub use types::*;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

// ── Vector math ───────────────────────────────────────────────────────────────

/// Computes the cosine similarity between two vectors.
///
/// Cosine similarity is the dot product of the vectors divided by the product
/// of their Euclidean norms, yielding a value in `[-1.0, 1.0]` where `1.0`
/// means identical direction, `0.0` means orthogonal, and `-1.0` means
/// opposite direction. It is invariant to vector magnitude.
///
/// Returns `0.0` when the vectors have different lengths or when either vector
/// has zero magnitude, so callers never observe `NaN` from a degenerate input.
///
/// # Example
/// ```
/// use tinyagents::harness::embeddings::cosine_similarity;
///
/// assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
/// assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
/// assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

// ── InMemoryVectorStore ───────────────────────────────────────────────────────

impl InMemoryVectorStore {
    /// Creates a new, empty in-memory vector store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }

    /// Returns `true` when the store holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add(&self, id: String, vector: Vec<f32>, metadata: Value) -> Result<()> {
        let mut entries = self.entries.lock().map_err(|e| {
            crate::error::TinyAgentsError::Embedding(format!("vector store lock poisoned: {e}"))
        })?;
        // Replace an existing entry with the same id so re-indexing updates it.
        if let Some(existing) = entries.iter_mut().find(|e| e.id == id) {
            existing.vector = vector;
            existing.metadata = metadata;
        } else {
            entries.push(VectorEntry {
                id,
                vector,
                metadata,
            });
        }
        Ok(())
    }

    async fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<ScoredDoc>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let entries = self.entries.lock().map_err(|e| {
            crate::error::TinyAgentsError::Embedding(format!("vector store lock poisoned: {e}"))
        })?;
        let mut scored: Vec<ScoredDoc> = entries
            .iter()
            .map(|e| ScoredDoc {
                id: e.id.clone(),
                score: cosine_similarity(vector, &e.vector),
                metadata: e.metadata.clone(),
            })
            .collect();
        // Sort by descending score; `total_cmp` keeps ordering total even with
        // NaN-free f32 scores (cosine_similarity never returns NaN).
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k);
        Ok(scored)
    }
}

// ── Retriever ─────────────────────────────────────────────────────────────────

impl Retriever {
    /// Creates a retriever from an embedding model and a vector store.
    pub fn new(
        model: std::sync::Arc<dyn EmbeddingModel>,
        store: std::sync::Arc<dyn VectorStore>,
    ) -> Self {
        Self { model, store }
    }

    /// Returns the embedding model backing this retriever.
    pub fn model(&self) -> &std::sync::Arc<dyn EmbeddingModel> {
        &self.model
    }

    /// Returns the vector store backing this retriever.
    pub fn store(&self) -> &std::sync::Arc<dyn VectorStore> {
        &self.store
    }

    /// Embeds and indexes a batch of documents.
    ///
    /// Each document is a `(id, text, metadata)` tuple: `text` is embedded with
    /// the configured model and stored under `id` along with `metadata`. The
    /// texts are embedded in a single batched [`EmbeddingModel::embed`] call.
    ///
    /// Re-indexing a document with an existing `id` replaces its vector and
    /// metadata in stores that support in-place update (such as
    /// [`InMemoryVectorStore`]).
    pub async fn index(&self, docs: Vec<(String, String, Value)>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = docs.iter().map(|(_, text, _)| text.clone()).collect();
        let vectors = self.model.embed(&texts).await?;
        for ((id, _text, metadata), vector) in docs.into_iter().zip(vectors) {
            self.store.add(id, vector, metadata).await?;
        }
        Ok(())
    }

    /// Embeds `query` and returns up to `top_k` most-similar documents.
    ///
    /// Results are sorted by descending cosine similarity (most similar first).
    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<ScoredDoc>> {
        let mut vectors = self.model.embed(&[query.to_string()]).await?;
        let query_vector = vectors.pop().unwrap_or_default();
        self.store.query(&query_vector, top_k).await
    }
}

// ── OpenAiEmbeddingModel (feature `openai`) ───────────────────────────────────

#[cfg(feature = "openai")]
mod openai {
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
                    TinyAgentsError::Embedding(
                        "openai embeddings response missing `data` array".into(),
                    )
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
}

#[cfg(feature = "openai")]
pub use openai::OpenAiEmbeddingModel;

#[cfg(test)]
mod test;
