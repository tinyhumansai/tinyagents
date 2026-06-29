use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

impl ToolResult {
    pub fn text(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
            raw: None,
        }
    }
}

#[async_trait]
pub trait Tool<State>: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    async fn call(&self, state: &State, call: ToolCall) -> Result<ToolResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_call_new_sets_fields() {
        let call = ToolCall::new("search", json!({"q": "rust"}));
        assert_eq!(call.name, "search");
        assert_eq!(call.arguments, json!({"q": "rust"}));
    }

    #[test]
    fn tool_result_text_has_no_raw() {
        let result = ToolResult::text("search", "ok");
        assert_eq!(result.name, "search");
        assert_eq!(result.content, "ok");
        assert!(result.raw.is_none());
    }

    #[test]
    fn tool_call_serde_round_trip() {
        let call = ToolCall::new("calc", json!({"a": 1, "b": 2}));
        let encoded = serde_json::to_string(&call).unwrap();
        let decoded: ToolCall = serde_json::from_str(&encoded).unwrap();
        assert_eq!(call, decoded);
    }

    #[test]
    fn tool_call_arguments_default_when_missing() {
        // `arguments` is `#[serde(default)]` → absent field decodes to Null.
        let decoded: ToolCall = serde_json::from_str(r#"{"name": "noop"}"#).unwrap();
        assert_eq!(decoded.name, "noop");
        assert_eq!(decoded.arguments, Value::Null);
    }

    #[test]
    fn tool_result_serde_round_trip_skips_none_raw() {
        let result = ToolResult::text("t", "content");
        let encoded = serde_json::to_string(&result).unwrap();
        // `raw` is skipped when None.
        assert!(!encoded.contains("raw"));
        let decoded: ToolResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(result, decoded);
    }

    #[test]
    fn tool_result_serde_round_trip_with_raw() {
        let result = ToolResult {
            name: "t".into(),
            content: "c".into(),
            raw: Some(json!({"k": "v"})),
        };
        let encoded = serde_json::to_string(&result).unwrap();
        assert!(encoded.contains("raw"));
        let decoded: ToolResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(result, decoded);
    }

    struct EchoTool;

    #[async_trait]
    impl Tool<()> for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "echoes the provided arguments back as text"
        }

        async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::text(call.name, call.arguments.to_string()))
        }
    }

    #[tokio::test]
    async fn tool_impl_call_returns_result() {
        let tool = EchoTool;
        assert_eq!(tool.name(), "echo");
        assert!(!tool.description().is_empty());

        let call = ToolCall::new("echo", json!({"x": 1}));
        let result = tool.call(&(), call).await.unwrap();
        assert_eq!(result.name, "echo");
        assert_eq!(result.content, json!({"x": 1}).to_string());
    }
}
