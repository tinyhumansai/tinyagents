use super::*;
use crate::harness::message::Message;
use crate::harness::usage::Usage;
use async_trait::async_trait;
use serde_json::json;

struct StaticModel;

#[async_trait]
impl ChatModel<()> for StaticModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> crate::Result<ModelResponse> {
        Ok(ModelResponse::assistant("hello").with_usage(Usage::new(3, 1)))
    }
}

#[test]
fn request_builder_sets_fields() {
    let req = ModelRequest::new(vec![Message::user("hi")])
        .with_model("gpt")
        .with_temperature(0.5)
        .with_max_tokens(128)
        .with_timeout_ms(1000)
        .with_tool_choice(ToolChoice::Required)
        .with_tag("t");
    assert_eq!(req.model.as_deref(), Some("gpt"));
    assert_eq!(req.temperature, Some(0.5));
    assert_eq!(req.max_tokens, Some(128));
    assert_eq!(req.timeout_ms, Some(1000));
    assert_eq!(req.tool_choice, ToolChoice::Required);
    assert_eq!(req.tags, vec!["t".to_string()]);
}

#[test]
fn tool_choice_defaults_to_auto() {
    assert_eq!(ModelRequest::default().tool_choice, ToolChoice::Auto);
}

#[test]
fn cacheable_prefix_ids_in_order() {
    let req = ModelRequest::new(vec![]).with_cache_segments(vec![
        PromptSegment {
            id: "sys".into(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "tools".into(),
            role: SegmentRole::Tools,
            cacheable: true,
        },
        PromptSegment {
            id: "tail".into(),
            role: SegmentRole::Volatile,
            cacheable: false,
        },
    ]);
    assert_eq!(req.cacheable_prefix_ids(), vec!["sys", "tools"]);
}

#[test]
fn response_format_json_schema() {
    let fmt = ResponseFormat::json_schema("person", json!({"type": "object"}));
    match fmt {
        ResponseFormat::JsonSchema { name, .. } => assert_eq!(name, "person"),
        _ => panic!("expected json schema"),
    }
}

#[test]
fn response_helpers() {
    let resp = ModelResponse::assistant("hi").with_finish_reason("stop");
    assert_eq!(resp.text(), "hi");
    assert!(resp.tool_calls().is_empty());
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn registry_register_get_default_and_stream() {
    let mut registry: ModelRegistry<()> = ModelRegistry::new();
    registry.register("default", Arc::new(StaticModel));
    assert_eq!(registry.default_name(), Some("default"));
    assert!(registry.get("default").is_some());
    assert_eq!(registry.names(), vec!["default".to_string()]);

    let model = registry.default_model().unwrap();
    let resp = model.invoke(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(resp.text(), "hello");
    assert_eq!(resp.usage.unwrap().total_tokens, 4);

    let deltas = model.stream(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0].content, "hello");
}
