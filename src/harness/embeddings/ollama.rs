//! Ollama `/api/embed` model with positional and NaN recovery guarantees.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::EmbeddingModel;
use crate::error::{Result, TinyAgentsError};

pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
pub const DEFAULT_OLLAMA_MODEL: &str = "bge-m3";
pub const DEFAULT_OLLAMA_DIMENSIONS: usize = 1024;

#[derive(Debug)]
pub struct OllamaEmbeddingModel {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimensions: usize,
}

impl OllamaEmbeddingModel {
    pub fn try_new(base_url: &str, model: &str, dimensions: usize) -> Result<Self> {
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: normalize_base_url(base_url)?,
            model: normalize_model(model)?,
            dimensions: if dimensions == 0 {
                DEFAULT_OLLAMA_DIMENSIONS
            } else {
                dimensions
            },
        })
    }

    pub fn new(base_url: &str, model: &str, dimensions: usize) -> Self {
        Self::try_new(base_url, model, dimensions).expect("invalid Ollama embedding configuration")
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn embed_url(&self) -> String {
        format!("{}/api/embed", self.base_url)
    }

    async fn request(&self, input: Vec<String>) -> Result<reqwest::Response> {
        self.client
            .post(self.embed_url())
            .json(&OllamaRequest {
                model: self.model.clone(),
                input,
            })
            .send()
            .await
            .map_err(|error| {
                TinyAgentsError::Embedding(format!(
                    "ollama embed request failed (is Ollama running at {}?): {error}",
                    self.base_url
                ))
            })
    }

    async fn embed_one_with_nan_recovery(&self, text: &str) -> Result<Vec<f32>> {
        let response = self.request(vec![text.to_owned()]).await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 500 && is_nan_encode_error(&body) {
                tracing::warn!(
                    target: "tinyagents::embeddings::ollama",
                    model = self.model,
                    "[embeddings] Ollama input produced NaN"
                );
                return Err(TinyAgentsError::Embedding(
                    "Ollama could not encode input without NaN values".into(),
                ));
            }
            return Err(ollama_http_error(status, &body));
        }
        let payload = parse_response(response).await?;
        if payload.embeddings.len() != 1 {
            return Err(TinyAgentsError::Embedding(format!(
                "ollama embed count mismatch: sent 1 text, got {} embeddings",
                payload.embeddings.len()
            )));
        }
        let vector = payload.embeddings.into_iter().next().unwrap();
        self.validate_dimensions(0, &vector)?;
        Ok(vector)
    }

    async fn embed_per_text(
        &self,
        total: usize,
        live: &[(usize, String)],
    ) -> Result<Vec<Vec<f32>>> {
        let mut output = vec![Vec::new(); total];
        for (index, text) in live {
            output[*index] = self.embed_one_with_nan_recovery(text).await?;
        }
        Ok(output)
    }

    fn validate_dimensions(&self, index: usize, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(TinyAgentsError::Embedding(format!(
                "ollama embed dimension mismatch at index {index}: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }
        Ok(())
    }
}

impl Default for OllamaEmbeddingModel {
    fn default() -> Self {
        Self::new(
            DEFAULT_OLLAMA_URL,
            DEFAULT_OLLAMA_MODEL,
            DEFAULT_OLLAMA_DIMENSIONS,
        )
    }
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
}

async fn parse_response(response: reqwest::Response) -> Result<OllamaResponse> {
    response.json().await.map_err(|error| {
        TinyAgentsError::Embedding(format!("ollama embed response parse failed: {error}"))
    })
}

fn ollama_http_error(status: reqwest::StatusCode, body: &str) -> TinyAgentsError {
    let detail = body.trim();
    TinyAgentsError::Embedding(format!(
        "ollama embed failed with status {status}{}",
        if detail.is_empty() {
            String::new()
        } else {
            format!(": {detail}")
        }
    ))
}

fn is_nan_encode_error(body: &str) -> bool {
    body.to_ascii_lowercase().contains("unsupported value: nan")
}

fn normalize_base_url(base_url: &str) -> Result<String> {
    let raw = if base_url.trim().is_empty() {
        DEFAULT_OLLAMA_URL
    } else {
        base_url.trim()
    };
    let url = reqwest::Url::parse(raw).map_err(|error| {
        TinyAgentsError::Validation(format!("invalid Ollama base_url `{raw}`: {error}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(TinyAgentsError::Validation(format!(
            "invalid Ollama base_url `{raw}`: expected an http:// or https:// URL"
        )));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(TinyAgentsError::Validation(format!(
            "invalid Ollama base_url `{raw}`: configure the server root without credentials"
        )));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(TinyAgentsError::Validation(format!(
            "invalid Ollama base_url `{raw}`: query strings and fragments are not supported"
        )));
    }
    let segments = url
        .path_segments()
        .map(|parts| {
            parts
                .filter(|part| !part.is_empty())
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let endpoint_suffix = segments
        .iter()
        .any(|part| matches!(part.as_str(), "api" | "v1"))
        || (segments.len() >= 2
            && segments[segments.len() - 2] == "chat"
            && segments[segments.len() - 1] == "completions");
    if endpoint_suffix {
        return Err(TinyAgentsError::Validation(format!(
            "invalid Ollama base_url `{raw}`: configure the Ollama server root, not an API endpoint"
        )));
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

fn normalize_model(model: &str) -> Result<String> {
    let model = if model.trim().is_empty() {
        DEFAULT_OLLAMA_MODEL.to_owned()
    } else {
        model.trim().to_owned()
    };
    if model.to_ascii_lowercase().starts_with("local-") {
        return Err(TinyAgentsError::Validation(format!(
            "invalid Ollama embedding model `{model}`: `local-*` IDs are virtual routing aliases; configure `{DEFAULT_OLLAMA_MODEL}` or another real model"
        )));
    }
    Ok(model)
}

#[async_trait]
impl EmbeddingModel for OllamaEmbeddingModel {
    fn name(&self) -> &str {
        "ollama"
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
        let live = texts
            .iter()
            .enumerate()
            .filter_map(|(index, text)| {
                let text = text.trim();
                (!text.is_empty()).then(|| (index, text.to_owned()))
            })
            .collect::<Vec<_>>();
        if live.is_empty() {
            return Ok(vec![Vec::new(); texts.len()]);
        }
        let input = live
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>();
        let response = self.request(input.clone()).await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 500 && is_nan_encode_error(&body) {
                tracing::warn!(
                    target: "tinyagents::embeddings::ollama",
                    batch = live.len(),
                    model = self.model,
                    "[embeddings] recovering Ollama NaN batch per text"
                );
                if live.len() == 1 {
                    return Err(TinyAgentsError::Embedding(
                        "Ollama could not encode input without NaN values".into(),
                    ));
                }
                return self.embed_per_text(texts.len(), &live).await;
            }
            return Err(ollama_http_error(status, &body));
        }

        let payload = parse_response(response).await?;
        if payload.embeddings.len() != input.len() {
            return Err(TinyAgentsError::Embedding(format!(
                "ollama embed count mismatch: sent {} texts, got {} embeddings",
                input.len(),
                payload.embeddings.len()
            )));
        }
        for (index, vector) in payload.embeddings.iter().enumerate() {
            self.validate_dimensions(index, vector)?;
        }
        let mut output = vec![Vec::new(); texts.len()];
        for ((index, _), vector) in live.iter().zip(payload.embeddings) {
            output[*index] = vector;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_identity_match_host() {
        let model = OllamaEmbeddingModel::default();
        assert_eq!(model.base_url(), DEFAULT_OLLAMA_URL);
        assert_eq!(model.model_id(), DEFAULT_OLLAMA_MODEL);
        assert_eq!(model.dimensions(), DEFAULT_OLLAMA_DIMENSIONS);
        assert_eq!(model.signature(), "provider=ollama;model=bge-m3;dims=1024");
    }

    #[test]
    fn validates_root_url_and_real_model() {
        assert!(OllamaEmbeddingModel::try_new("http://host:11434/api", "m", 1).is_err());
        assert!(OllamaEmbeddingModel::try_new("http://user:p@host:11434", "m", 1).is_err());
        assert!(OllamaEmbeddingModel::try_new("http://host:11434", "local-v1", 1).is_err());
    }

    #[tokio::test]
    async fn blank_inputs_preserve_positions_without_network() {
        let model = OllamaEmbeddingModel::default();
        let vectors = model.embed(&[" ".into(), "\n".into()]).await.unwrap();
        assert_eq!(vectors, vec![Vec::<f32>::new(), Vec::new()]);
    }

    #[test]
    fn recognizes_only_nan_encoding_failures() {
        assert!(is_nan_encode_error("unsupported value: NaN"));
        assert!(!is_nan_encode_error("model crashed"));
    }
}
