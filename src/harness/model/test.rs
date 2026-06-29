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
        .with_model_hint(ModelHint {
            model: "fast".into(),
            priority: 10,
            reason: Some("latency".into()),
        })
        .with_reuse_previous_model(true)
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
    assert_eq!(req.model_hints[0].model, "fast");
    assert!(req.reuse_previous_model);
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
    let resp = ModelResponse::assistant("hi")
        .with_finish_reason("stop")
        .with_resolved_model(ResolvedModel {
            name: "fast".into(),
            requested: Some("fast".into()),
            source: ModelResolutionSource::Hint,
        });
    assert_eq!(resp.text(), "hi");
    assert!(resp.tool_calls().is_empty());
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    assert_eq!(resp.resolved_model.unwrap().name, "fast");
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

#[tokio::test]
async fn registry_resolves_request_override_first() {
    let mut registry: ModelRegistry<()> = ModelRegistry::new();
    registry
        .register("default", Arc::new(StaticModel))
        .register("explicit", Arc::new(StaticModel));

    let request = ModelRequest::default()
        .with_model("explicit")
        .with_model_hint(ModelHint {
            model: "default".into(),
            priority: 100,
            reason: None,
        });

    let resolved = registry
        .resolve_request(&request, Some("default"), None)
        .unwrap()
        .resolved;

    assert_eq!(resolved.name, "explicit");
    assert_eq!(resolved.source, ModelResolutionSource::RequestOverride);
}

#[tokio::test]
async fn registry_reuses_previous_before_hints() {
    let mut registry: ModelRegistry<()> = ModelRegistry::new();
    registry
        .register("default", Arc::new(StaticModel))
        .register("previous", Arc::new(StaticModel))
        .register("hint", Arc::new(StaticModel));

    let request = ModelRequest::default()
        .with_reuse_previous_model(true)
        .with_model_hint(ModelHint {
            model: "hint".into(),
            priority: 100,
            reason: None,
        });

    let previous = ResolvedModel {
        name: "previous".into(),
        requested: Some("previous".into()),
        source: ModelResolutionSource::AgentDefault,
    };

    let resolved = registry
        .resolve_request(&request, Some("default"), Some(previous))
        .unwrap()
        .resolved;

    assert_eq!(resolved.name, "previous");
    assert_eq!(resolved.source, ModelResolutionSource::StateReuse);
}

#[tokio::test]
async fn registry_tries_hints_by_priority_then_agent_default_then_registry_default() {
    let mut registry: ModelRegistry<()> = ModelRegistry::new();
    registry
        .register("registry_default", Arc::new(StaticModel))
        .register("agent_default", Arc::new(StaticModel))
        .register("strong_hint", Arc::new(StaticModel));

    let request = ModelRequest::default()
        .with_model_hint(ModelHint {
            model: "missing".into(),
            priority: 100,
            reason: None,
        })
        .with_model_hint(ModelHint {
            model: "strong_hint".into(),
            priority: 10,
            reason: None,
        });

    let resolved = registry
        .resolve_request(&request, Some("agent_default"), None)
        .unwrap()
        .resolved;

    assert_eq!(resolved.name, "strong_hint");
    assert_eq!(resolved.source, ModelResolutionSource::Hint);

    let resolved = registry
        .resolve_request(&ModelRequest::default(), Some("agent_default"), None)
        .unwrap()
        .resolved;

    assert_eq!(resolved.name, "agent_default");
    assert_eq!(resolved.source, ModelResolutionSource::AgentDefault);

    let resolved = registry
        .resolve_request(&ModelRequest::default(), Some("missing"), None)
        .unwrap()
        .resolved;

    assert_eq!(resolved.name, "registry_default");
    assert_eq!(resolved.source, ModelResolutionSource::RegistryDefault);
}
