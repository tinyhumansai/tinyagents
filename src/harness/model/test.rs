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
fn response_format_auto_constructor() {
    let fmt = ResponseFormat::auto("person", json!({"type": "object"}));
    match fmt {
        ResponseFormat::Auto { name, .. } => assert_eq!(name, "person"),
        _ => panic!("expected auto"),
    }
}

#[test]
fn default_profile_is_conservative() {
    let profile = ModelProfile::default();
    assert!(!profile.tool_calling);
    assert!(!profile.native_structured_output);
    assert!(!profile.streaming);
    assert_eq!(profile.status, ModelStatus::Stable);
    // Default modalities are text-only.
    assert!(profile.modalities.text_in && profile.modalities.text_out);
    assert!(!profile.modalities.image_in);
}

#[test]
fn empty_capability_set_is_always_satisfied() {
    let profile = ModelProfile::default();
    assert!(profile.satisfies(&CapabilitySet::default()));
}

#[test]
fn profile_satisfies_matching_capabilities() {
    let profile = ModelProfile {
        tool_calling: true,
        streaming: true,
        json_schema: true,
        native_structured_output: true,
        max_input_tokens: Some(128_000),
        max_output_tokens: Some(8_000),
        ..ModelProfile::default()
    };
    let required = CapabilitySet {
        tool_calling: true,
        json_schema: true,
        native_structured_output: true,
        min_input_tokens: Some(100_000),
        min_output_tokens: Some(4_000),
        ..CapabilitySet::default()
    };
    assert!(profile.satisfies(&required));
}

#[test]
fn profile_does_not_satisfy_missing_capability() {
    let profile = ModelProfile {
        tool_calling: true,
        ..ModelProfile::default()
    };
    // Requires reasoning, which the profile does not advertise.
    let required = CapabilitySet {
        tool_calling: true,
        reasoning: true,
        ..CapabilitySet::default()
    };
    assert!(!profile.satisfies(&required));
}

#[test]
fn profile_token_requirement_fails_when_capacity_unknown_or_too_small() {
    // Unknown capacity fails a token requirement.
    let unknown = ModelProfile::default();
    assert!(!unknown.satisfies(&CapabilitySet {
        min_input_tokens: Some(1_000),
        ..CapabilitySet::default()
    }));

    // Known-but-too-small capacity also fails.
    let small = ModelProfile {
        max_output_tokens: Some(512),
        ..ModelProfile::default()
    };
    assert!(!small.satisfies(&CapabilitySet {
        min_output_tokens: Some(4_096),
        ..CapabilitySet::default()
    }));
}

#[test]
fn permissive_profile_satisfies_demanding_capability_set() {
    let profile = ModelProfile::permissive();
    let required = CapabilitySet {
        tool_calling: true,
        parallel_tool_calls: true,
        streaming: true,
        streaming_tool_chunks: true,
        native_structured_output: true,
        json_schema: true,
        reasoning: true,
        image_in: true,
        image_out: true,
        audio_in: true,
        audio_out: true,
        ..CapabilitySet::default()
    };
    assert!(profile.satisfies(&required));
}

#[test]
fn model_request_capability_and_provider_option_builders() {
    let caps = CapabilitySet {
        tool_calling: true,
        ..CapabilitySet::default()
    };
    let req = ModelRequest::new(vec![])
        .with_required_capabilities(caps.clone())
        .with_provider_options(json!({"top_k": 5}))
        .with_continuation_id("resp-1");
    assert_eq!(req.required_capabilities, Some(caps));
    assert_eq!(req.provider_options, json!({"top_k": 5}));
    assert_eq!(req.continuation_id.as_deref(), Some("resp-1"));
    // Defaults stay null/None so existing builders/tests are unaffected.
    assert!(ModelRequest::default().provider_options.is_null());
    assert!(ModelRequest::default().required_capabilities.is_none());
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
