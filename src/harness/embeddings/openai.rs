//! [`OpenAiEmbeddingModel`]: an [`EmbeddingModel`] backed by the hosted
//! OpenAI embeddings endpoint (`POST {base_url}/embeddings`).
//!
//! Split out of `embeddings/mod.rs`; mirrors the transport pattern of
//! [`crate::harness::providers::openai::OpenAiModel`].

use async_trait::async_trait;
use serde_json::{Value, json};

use super::EmbeddingModel;
use super::retry_after::{MAX_RETRIES, backoff_ms_for_attempt};
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
    send_dimensions: bool,
    requires_api_key: bool,
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
            send_dimensions: true,
            requires_api_key: true,
        }
    }

    /// Overrides the embedding model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = normalize_model_for_base_url(&self.base_url, &model.into());
        self
    }

    /// Overrides the API base URL; a trailing slash is trimmed so the joined
    /// endpoint is always `{base_url}/embeddings`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self.model = normalize_model_for_base_url(&self.base_url, &self.model);
        self
    }

    /// Overrides the reported dimensionality (and requests it from the API
    /// via the `dimensions` parameter, which `text-embedding-3-*` supports).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Controls whether the OpenAI-compatible `dimensions` field is sent.
    pub fn with_send_dimensions(mut self, send: bool) -> Self {
        self.send_dimensions = send;
        self
    }

    /// Controls whether an empty API key is rejected before making a request.
    pub fn with_required_api_key(mut self, required: bool) -> Self {
        self.requires_api_key = required;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn embeddings_url(&self) -> String {
        embeddings_url(&self.base_url)
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
    fn name(&self) -> &str {
        "openai"
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(index) = texts.iter().position(|text| text.trim().is_empty()) {
            return Err(TinyAgentsError::Validation(format!(
                "openai embed: refusing empty/whitespace input at index {index} of {} (model={})",
                texts.len(),
                self.model
            )));
        }
        if self.requires_api_key && self.api_key.trim().is_empty() {
            return Err(TinyAgentsError::Validation(format!(
                "Embedding API key not set (model={})",
                self.model
            )));
        }
        let url = self.embeddings_url();
        let mut body = json!({
            "model": self.model,
            "input": texts,
        });
        if self.send_dimensions && self.dimensions > 0 {
            body["dimensions"] = json!(self.dimensions);
        }

        let mut response = None;
        for attempt in 0..=MAX_RETRIES {
            super::rate_limit::acquire(&self.base_url).await;
            let mut request = self.client.post(&url).json(&body);
            if !self.api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {}", self.api_key));
            }
            let current = request.send().await.map_err(|e| {
                TinyAgentsError::Embedding(format!(
                    "openai embeddings request to {url} failed: {e}"
                ))
            })?;
            let retryable = matches!(current.status().as_u16(), 429 | 503);
            if retryable && attempt < MAX_RETRIES {
                let retry_after = current
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned);
                let status = current.status();
                let _ = current.text().await;
                let delay_ms = backoff_ms_for_attempt(attempt, retry_after.as_deref());
                tracing::debug!(
                    target: "tinyagents::embeddings::openai",
                    %status,
                    attempt,
                    delay_ms,
                    "[embeddings] retrying transient OpenAI-compatible response"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                continue;
            }
            response = Some(current);
            break;
        }
        let response = response.expect("bounded retry loop always records its final response");

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
            let vector = embedding
                .iter()
                .map(|n| {
                    n.as_f64().map(|value| value as f32).ok_or_else(|| {
                        TinyAgentsError::Embedding(
                            "openai embeddings response contains a non-numeric value".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if self.dimensions > 0 && vector.len() != self.dimensions {
                return Err(TinyAgentsError::Embedding(format!(
                    "openai embed dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    vector.len()
                )));
            }
            vectors.push(vector);
        }
        if vectors.len() != texts.len() {
            return Err(TinyAgentsError::Embedding(format!(
                "openai embed count mismatch: sent {} texts, got {} embeddings",
                texts.len(),
                vectors.len()
            )));
        }
        Ok(vectors)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

fn embeddings_url(base_url: &str) -> String {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return format!("{}/v1/embeddings", base_url.trim_end_matches('/'));
    };
    let path = url.path().trim_end_matches('/');
    if path.ends_with("/embeddings") {
        base_url.trim_end_matches('/').to_owned()
    } else if path.is_empty() || path == "/" {
        format!("{}/v1/embeddings", base_url.trim_end_matches('/'))
    } else {
        format!("{}/embeddings", base_url.trim_end_matches('/'))
    }
}

fn normalize_model_for_base_url(base_url: &str, model: &str) -> String {
    let is_gemini = reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| {
            host == "generativelanguage.googleapis.com"
                || host.ends_with(".generativelanguage.googleapis.com")
        });
    if is_gemini
        && !model.is_empty()
        && !model.starts_with("models/")
        && !model.starts_with("tunedModels/")
    {
        format!("models/{model}")
    } else {
        model.to_owned()
    }
}
