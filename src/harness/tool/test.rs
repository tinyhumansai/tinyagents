//! Tests for the harness tool layer.
//!
//! Cover [`ToolSchema`]/[`ToolCall`]/[`ToolResult`] construction (including the
//! error path that preserves the call id) and [`ToolRegistry`] registration,
//! lookup, name listing, and schema collection.

use super::*;
use async_trait::async_trait;
use serde_json::json;

struct EchoTool;

#[async_trait]
impl Tool<()> for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "echoes its input"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "echo",
            "echoes its input",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        )
    }

    async fn call(&self, _state: &(), call: ToolCall) -> crate::Result<ToolResult> {
        let text = call
            .arguments
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        Ok(ToolResult::text(call.id, "echo", text))
    }
}

#[test]
fn registry_register_get_names_schemas() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));
    assert!(registry.get("echo").is_some());
    assert!(registry.get("missing").is_none());
    assert_eq!(registry.names(), vec!["echo".to_string()]);
    assert_eq!(registry.schemas().len(), 1);
    assert_eq!(registry.schemas()[0].name, "echo");
}

#[tokio::test]
async fn tool_call_round_trips() {
    let tool = EchoTool;
    let call = ToolCall::new("c-1", "echo", json!({"text": "hi"}));
    let result = tool.call(&(), call).await.unwrap();
    assert_eq!(result.call_id, "c-1");
    assert_eq!(result.content, "hi");
    assert!(!result.is_error());
}

#[test]
fn error_result_preserves_call_id() {
    let result = ToolResult::error("c-9", "echo", "boom");
    assert!(result.is_error());
    assert_eq!(result.call_id, "c-9");
    assert_eq!(result.error.as_deref(), Some("boom"));
}
