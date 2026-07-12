//! Voyage AI embedding model.

use async_trait::async_trait;

use super::{EmbeddingModel, OpenAiEmbeddingModel};
use crate::error::Result;

pub const VOYAGE_API_BASE: &str = "https://api.voyageai.com/v1";
pub const VOYAGE_DEFAULT_MODEL: &str = "voyage-3-large";
pub const VOYAGE_DEFAULT_DIMENSIONS: usize = 1024;

/// Voyage's endpoint uses the OpenAI response shape without its `dimensions`
/// request parameter.
pub struct VoyageEmbeddingModel {
    inner: OpenAiEmbeddingModel,
}

impl VoyageEmbeddingModel {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_options(
            api_key,
            VOYAGE_DEFAULT_MODEL,
            VOYAGE_DEFAULT_DIMENSIONS,
            VOYAGE_API_BASE,
        )
    }

    pub fn with_options(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimensions: usize,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            inner: OpenAiEmbeddingModel::new(api_key)
                .with_model(model)
                .with_dimensions(dimensions)
                .with_base_url(base_url)
                .with_send_dimensions(false)
                .with_required_api_key(true),
        }
    }
}

#[async_trait]
impl EmbeddingModel for VoyageEmbeddingModel {
    fn name(&self) -> &str {
        "voyage"
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed(texts).await
    }
}
