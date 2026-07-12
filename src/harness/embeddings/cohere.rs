//! Cohere-native embedding model using the `/v2/embed` contract.

use async_trait::async_trait;
use serde::Deserialize;

use super::EmbeddingModel;
use super::retry_after::{MAX_RETRIES, backoff_ms_for_attempt};
use crate::error::{Result, TinyAgentsError};

pub const COHERE_API_BASE: &str = "https://api.cohere.com";
pub const COHERE_DEFAULT_MODEL: &str = "embed-english-v3.0";
pub const COHERE_DEFAULT_DIMENSIONS: usize = 1024;

pub struct CohereEmbeddingModel {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimensions: usize,
    base_url: String,
    query_mode: bool,
}

impl CohereEmbeddingModel {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: COHERE_DEFAULT_MODEL.to_owned(),
            dimensions: COHERE_DEFAULT_DIMENSIONS,
            base_url: COHERE_API_BASE.to_owned(),
            query_mode: false,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim().trim_end_matches('/').to_owned();
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

#[derive(Deserialize)]
struct CohereResponse {
    embeddings: CohereEmbeddings,
}

#[derive(Deserialize)]
struct CohereEmbeddings {
    float: Vec<Vec<f32>>,
}

#[async_trait]
impl EmbeddingModel for CohereEmbeddingModel {
    fn name(&self) -> &str {
        "cohere"
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.api_key.trim().is_empty() {
            return Err(TinyAgentsError::Validation(
                "Cohere API key not set. Configure an API key before embedding.".into(),
            ));
        }
        if let Some(index) = texts.iter().position(|text| text.trim().is_empty()) {
            return Err(TinyAgentsError::Validation(format!(
                "cohere embed: refusing empty/whitespace input at index {index} of {}",
                texts.len()
            )));
        }

        let url = format!("{}/v2/embed", self.base_url);
        let mut body = serde_json::json!({
            "model": self.model,
            "texts": texts,
            "input_type": if self.query_mode { "search_query" } else { "search_document" },
            "embedding_types": ["float"],
        });
        if self.model.to_ascii_lowercase().contains("v4") && self.dimensions > 0 {
            body["output_dimension"] = serde_json::json!(self.dimensions);
        }

        let mut response = None;
        for attempt in 0..=MAX_RETRIES {
            super::rate_limit::acquire(&self.base_url).await;
            let current = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&body)
                .send()
                .await
                .map_err(|error| {
                    TinyAgentsError::Embedding(format!(
                        "Cohere embeddings request to {url} failed: {error}"
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
                    target: "tinyagents::embeddings::cohere",
                    %status,
                    attempt,
                    delay_ms,
                    "[embeddings] retrying transient Cohere response"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                continue;
            }
            response = Some(current);
            break;
        }

        let response = response.expect("bounded retry loop always records its final response");
        let status = response.status();
        let text = response.text().await.map_err(|error| {
            TinyAgentsError::Embedding(format!("Cohere embed response read failed: {error}"))
        })?;
        if !status.is_success() {
            return Err(TinyAgentsError::Embedding(format!(
                "Cohere embed API error ({status}): {text}"
            )));
        }

        let payload: CohereResponse = serde_json::from_str(&text).map_err(|error| {
            TinyAgentsError::Embedding(format!("Cohere embed response parse failed: {error}"))
        })?;
        let vectors = payload.embeddings.float;
        if vectors.len() != texts.len() {
            return Err(TinyAgentsError::Embedding(format!(
                "Cohere embed count mismatch: sent {} texts, got {} embeddings",
                texts.len(),
                vectors.len()
            )));
        }
        for (index, vector) in vectors.iter().enumerate() {
            if self.dimensions > 0 && vector.len() != self.dimensions {
                return Err(TinyAgentsError::Embedding(format!(
                    "Cohere embed dimension mismatch at index {index}: expected {}, got {}",
                    self.dimensions,
                    vector.len()
                )));
            }
        }
        Ok(vectors)
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let query_model = Self {
            client: self.client.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            dimensions: self.dimensions,
            base_url: self.base_url.clone(),
            query_mode: true,
        };
        let mut vectors = query_model.embed(&[query.to_owned()]).await?;
        Ok(vectors.pop().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_defaults_match_host_contract() {
        let model = CohereEmbeddingModel::new("key");
        assert_eq!(model.name(), "cohere");
        assert_eq!(model.model_id(), COHERE_DEFAULT_MODEL);
        assert_eq!(model.dimensions(), COHERE_DEFAULT_DIMENSIONS);
        assert_eq!(
            model.signature(),
            "provider=cohere;model=embed-english-v3.0;dims=1024"
        );
    }

    #[tokio::test]
    async fn empty_batch_short_circuits_before_key_validation() {
        let model = CohereEmbeddingModel::new("");
        assert!(model.embed(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn missing_key_fails_before_network() {
        let model = CohereEmbeddingModel::new("").with_base_url("http://127.0.0.1:1");
        let error = model.embed(&["hello".into()]).await.unwrap_err();
        assert!(error.to_string().contains("API key not set"));
    }
}
