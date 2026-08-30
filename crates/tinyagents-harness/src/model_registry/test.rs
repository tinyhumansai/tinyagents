//! Model registry unit tests.

use async_trait::async_trait;
use tinyinference::model::{ChatModel, ModelRequest, ModelResponse};

use super::*;

struct StaticModel;

#[async_trait]
impl ChatModel<()> for StaticModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference::Result<ModelResponse> {
        Ok(ModelResponse::assistant("ok"))
    }
}

#[test]
fn first_registration_becomes_default() {
    let mut registry = ModelRegistry::new();
    registry.register("default", Arc::new(StaticModel));
    assert_eq!(registry.default_name(), Some("default"));
    assert_eq!(registry.names(), vec!["default"]);
}
