//! Bearer-authenticated OpenAI-compatible cloud embedding model.

use std::sync::Arc;

use async_trait::async_trait;

use super::{EmbeddingModel, OpenAiEmbeddingModel};
use crate::error::{Result, TinyAgentsError};

pub const DEFAULT_CLOUD_MODEL: &str = "embedding-v1";
pub const DEFAULT_CLOUD_DIMENSIONS: usize = 1024;

/// Resolves the current bearer token for each request.
pub type BearerResolver = Arc<dyn Fn() -> Result<String> + Send + Sync>;

/// Cloud model whose credential lifecycle remains owned by the host.
pub struct CloudEmbeddingModel {
    base_url: String,
    model: String,
    dimensions: usize,
    bearer: BearerResolver,
}

impl CloudEmbeddingModel {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        dimensions: usize,
        bearer: BearerResolver,
    ) -> Self {
        Self {
            base_url: base_url.into().trim().trim_end_matches('/').to_owned(),
            model: model.into(),
            dimensions,
            bearer,
        }
    }
}

#[async_trait]
impl EmbeddingModel for CloudEmbeddingModel {
    fn name(&self) -> &str {
        "cloud"
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
        if let Some(index) = texts.iter().position(|text| text.trim().is_empty()) {
            return Err(TinyAgentsError::Validation(format!(
                "cloud embed: refusing empty/whitespace input at index {index} of {} (model={})",
                texts.len(),
                self.model
            )));
        }
        let bearer = (self.bearer)()?;
        if bearer.trim().is_empty() {
            return Err(TinyAgentsError::Validation(
                "No backend session for cloud embeddings".into(),
            ));
        }
        OpenAiEmbeddingModel::new(bearer)
            .with_base_url(&self.base_url)
            .with_model(&self.model)
            .with_dimensions(self.dimensions)
            .with_send_dimensions(false)
            .with_required_api_key(true)
            .embed(texts)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn missing_bearer() -> BearerResolver {
        Arc::new(|| {
            Err(TinyAgentsError::Validation(
                "No backend session for cloud embeddings".into(),
            ))
        })
    }

    #[test]
    fn identity_matches_host_contract() {
        let model = CloudEmbeddingModel::new(
            "https://api.example/openai/v1/",
            DEFAULT_CLOUD_MODEL,
            DEFAULT_CLOUD_DIMENSIONS,
            missing_bearer(),
        );
        assert_eq!(model.name(), "cloud");
        assert_eq!(
            model.signature(),
            "provider=cloud;model=embedding-v1;dims=1024"
        );
    }

    #[tokio::test]
    async fn validation_precedes_bearer_resolution() {
        let model = CloudEmbeddingModel::new(
            "https://api.example/openai/v1",
            DEFAULT_CLOUD_MODEL,
            DEFAULT_CLOUD_DIMENSIONS,
            missing_bearer(),
        );
        assert!(model.embed(&[]).await.unwrap().is_empty());
        let error = model.embed(&[" ".into()]).await.unwrap_err();
        assert!(error.to_string().contains("empty/whitespace"));
    }
}
