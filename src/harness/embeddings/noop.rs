//! No-op embedding model for keyword-only retrieval.

use async_trait::async_trait;

use super::EmbeddingModel;
use crate::error::Result;

/// Embedding model used when semantic search is disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopEmbeddingModel;

#[async_trait]
impl EmbeddingModel for NoopEmbeddingModel {
    fn name(&self) -> &str {
        "none"
    }

    fn model_id(&self) -> &str {
        "none"
    }

    fn dimensions(&self) -> usize {
        0
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }
}
