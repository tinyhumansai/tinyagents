use std::sync::Arc;

use futures::stream;
use serde::Deserialize;
use serde_json::json;
use tinyagents::harness::cost::{CostTotals, estimate_cost};
use tinyagents::harness::message::{Message, MessageDelta};
use tinyagents::harness::model::{
    CapabilitySet, ChatModel, ModelHint, ModelProfile, ModelRegistry, ModelRequest,
    ModelResolutionSource, ModelResponse, ModelSelection, ModelStreamItem, PromptSegment,
    ProviderError, ResponseFormat, SegmentRole, StreamAccumulator, ToolChoice,
    collect_model_stream,
};
use tinyagents::harness::providers::{MockModel, ProviderKind, ProviderSpec};
use tinyagents::harness::stream::{StreamChunk, StreamMode, StreamSink, stream as filter_stream};
use tinyagents::harness::structured::{
    StructuredExtractor, StructuredStrategy, response_format_for_strategy,
};
use tinyagents::harness::tool::{ToolCall, ToolDelta, ToolFormat, ToolSchema};
use tinyagents::harness::usage::{Usage, UsageTotals};
use tinyagents::registry::catalog::{
    ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelCatalogSnapshot, ModelCatalogSource,
    ModelPricing,
};

#[tokio::test]
async fn mock_provider_invokes_streams_and_reports_usage() {
    let state = ();
    let echo = MockModel::echo();
    let request = ModelRequest::new(vec![
        Message::system("system text"),
        Message::user("hello mock provider"),
    ]);

    let response = echo.invoke(&state, request.clone()).await.unwrap();
    assert_eq!(response.text(), "hello mock provider");
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    assert_eq!(response.usage.unwrap().output_tokens, 5);
    assert_eq!(echo.call_count(), 1);

    let streamed = collect_model_stream(echo.stream(&state, request).await.unwrap())
        .await
        .unwrap();
    assert_eq!(streamed.text(), "hello mock provider");
    assert_eq!(echo.call_count(), 2);

    let scripted = MockModel::with_responses(vec![
        MockModel::text_response("first"),
        MockModel::text_response("second"),
    ]);
    assert_eq!(
        scripted
            .invoke(&state, ModelRequest::new(vec![]))
            .await
            .unwrap()
            .text(),
        "first"
    );
    assert_eq!(
        scripted
            .invoke(&state, ModelRequest::new(vec![]))
            .await
            .unwrap()
            .text(),
        "second"
    );
    assert_eq!(
        scripted
            .invoke(&state, ModelRequest::new(vec![]))
            .await
            .unwrap()
            .text(),
        "first"
    );

    let tool_model = MockModel::with_tool_call("lookup", json!({ "query": "rust" }));
    let tool_response = tool_model
        .invoke(&state, ModelRequest::new(vec![Message::user("call tool")]))
        .await
        .unwrap();
    assert_eq!(tool_response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(tool_response.tool_calls()[0].name, "lookup");
    assert_eq!(tool_response.tool_calls()[0].arguments["query"], "rust");
}

#[test]
fn provider_specs_infer_and_normalize_known_providers() {
    assert_eq!(ProviderKind::OpenAi.as_str(), "openai");
    assert_eq!(
        ProviderKind::infer("openai:gpt-4.1-mini"),
        Some(ProviderKind::OpenAi)
    );
    assert_eq!(
        ProviderKind::infer("claude-3-5-sonnet-latest"),
        Some(ProviderKind::Anthropic)
    );
    assert_eq!(
        ProviderKind::infer("mistralai:mistral-small-latest"),
        Some(ProviderKind::Mistral)
    );
    assert_eq!(ProviderKind::infer("unknown-model"), None);

    let spec = ProviderSpec::for_kind(ProviderKind::Ollama)
        .with_model("llama3.3")
        .with_base_url("http://localhost:11434/v1/")
        .with_provider("local")
        .with_api_key_env("LOCAL_KEY");
    assert_eq!(spec.kind, ProviderKind::Ollama);
    assert_eq!(spec.provider, "local");
    assert_eq!(spec.model, "llama3.3");
    assert_eq!(spec.base_url, "http://localhost:11434/v1");
    assert_eq!(spec.api_key_env.as_deref(), Some("LOCAL_KEY"));
    assert!(!ProviderSpec::for_kind(ProviderKind::Ollama).requires_api_key);
    assert!(ProviderSpec::for_kind(ProviderKind::Compatible).requires_api_key);
}

#[tokio::test]
async fn model_request_response_registry_and_stream_contracts_are_stable() {
    let schema = json!({ "type": "object", "properties": { "ok": { "type": "boolean" } } });
    let request = ModelRequest::new(vec![Message::user("choose")])
        .with_model("direct")
        .with_model_hint(ModelHint {
            model: "hinted".into(),
            priority: 10,
            reason: Some("cheaper".into()),
        })
        .with_reuse_previous_model(true)
        .with_temperature(0.2)
        .with_top_p(0.9)
        .with_max_tokens(128)
        .with_stop_sequences(["END", "STOP"])
        .with_seed(42)
        .with_timeout_ms(1_000)
        .with_tag("e2e")
        .with_tool_choice(ToolChoice::Tool("lookup".into()))
        .with_tools(vec![ToolSchema {
            name: "lookup".into(),
            description: "lookup things".into(),
            parameters: json!({ "type": "object" }),
            format: ToolFormat::PType {
                parameters: vec!["query".into()],
            },
        }])
        .with_response_format(ResponseFormat::auto("answer", schema.clone()))
        .with_cache_segments(vec![
            PromptSegment {
                id: "system".into(),
                role: SegmentRole::System,
                cacheable: true,
            },
            PromptSegment {
                id: "turn".into(),
                role: SegmentRole::Volatile,
                cacheable: false,
            },
        ])
        .with_required_capabilities(CapabilitySet {
            tool_calling: true,
            streaming: true,
            ..CapabilitySet::default()
        })
        .with_provider_options(json!({ "reasoning": "low" }))
        .with_provider_option("seed_control", json!(true))
        .with_continuation_id("resp_123");

    assert_eq!(request.cacheable_prefix_ids(), vec!["system"]);
    assert_eq!(request.model.as_deref(), Some("direct"));
    assert_eq!(request.stop_sequences, vec!["END", "STOP"]);
    assert_eq!(request.provider_options["reasoning"], "low");
    assert_eq!(request.provider_options["seed_control"], true);

    let resolved = tinyagents::harness::model::ResolvedModel {
        name: "direct".into(),
        requested: Some("direct".into()),
        source: ModelResolutionSource::RequestOverride,
    };
    let response = ModelResponse::assistant("done")
        .with_usage(Usage::new(2, 3))
        .with_finish_reason("stop")
        .with_resolved_model(resolved.clone());
    assert_eq!(response.text(), "done");
    assert_eq!(response.usage.unwrap().effective_total(), 5);
    assert_eq!(response.message.usage.unwrap().effective_total(), 5);
    assert_eq!(response.resolved_model, Some(resolved));

    let mut registry: ModelRegistry<()> = ModelRegistry::new();
    registry
        .register("default", Arc::new(MockModel::constant("default")))
        .register("direct", Arc::new(MockModel::constant("direct")))
        .register("hinted", Arc::new(MockModel::constant("hinted")));
    registry.set_default("default");

    let binding = registry
        .resolve_request(&request, Some("default"), None)
        .unwrap();
    assert_eq!(binding.resolved.name, "direct");
    assert_eq!(
        binding.resolved.source,
        ModelResolutionSource::RequestOverride
    );
    assert_eq!(registry.names(), vec!["default", "direct", "hinted"]);

    let fallback = registry
        .resolve(ModelSelection {
            hints: vec![
                ModelHint {
                    model: "missing".into(),
                    priority: 100,
                    reason: None,
                },
                ModelHint {
                    model: "hinted".into(),
                    priority: 1,
                    reason: None,
                },
            ],
            ..ModelSelection::default()
        })
        .unwrap();
    assert_eq!(fallback.resolved.name, "hinted");
    assert_eq!(fallback.resolved.source, ModelResolutionSource::Hint);

    let unsatisfied = registry.resolve(ModelSelection {
        required_capabilities: Some(CapabilitySet {
            min_input_tokens: Some(u64::MAX),
            ..CapabilitySet::default()
        }),
        ..ModelSelection::default()
    });
    assert!(unsatisfied.is_none());

    let mut accumulator = StreamAccumulator::new();
    accumulator.push(&ModelStreamItem::Started);
    accumulator.push(&ModelStreamItem::MessageDelta(MessageDelta {
        text: "hel".into(),
        reasoning: String::new(),
        tool_call: None,
    }));
    accumulator.push(&ModelStreamItem::ToolCallDelta(ToolDelta {
        call_id: "call-1".into(),
        content: r#"{"a":"#.into(),
        tool_name: None,
    }));
    accumulator.push(&ModelStreamItem::ToolCallDelta(ToolDelta {
        call_id: "call-1".into(),
        content: r#""b"}"#.into(),
        tool_name: None,
    }));
    accumulator.push(&ModelStreamItem::UsageDelta(Usage::new(4, 5)));
    accumulator.push(&ModelStreamItem::MessageDelta(MessageDelta {
        text: "lo".into(),
        reasoning: String::new(),
        tool_call: None,
    }));
    assert!(!accumulator.is_terminal());
    let merged = accumulator.finish().unwrap();
    assert_eq!(merged.text(), "hello");
    assert_eq!(merged.tool_calls()[0].id, "call-1");
    assert_eq!(merged.tool_calls()[0].arguments["a"], "b");
    assert_eq!(merged.usage.unwrap().effective_total(), 9);

    let mut completed = StreamAccumulator::new();
    completed.push(&ModelStreamItem::UsageDelta(Usage::new(1, 1)));
    completed.push(&ModelStreamItem::Completed(ModelResponse::assistant(
        "final",
    )));
    assert!(completed.is_terminal());
    let final_response = completed.finish().unwrap();
    assert_eq!(final_response.text(), "final");
    assert_eq!(final_response.usage.unwrap().effective_total(), 2);

    let failed_stream = Box::pin(stream::iter([ModelStreamItem::ProviderFailed(
        ProviderError {
            provider: "mock".into(),
            model: Some("bad".into()),
            status: Some(500),
            code: Some("internal".into()),
            message: "nope".into(),
            retryable: true,
            raw: Some(json!({ "error": "nope" })),
        },
    )]));
    let err = collect_model_stream(failed_stream).await.unwrap_err();
    // A streamed `ProviderFailed` now surfaces as a structured
    // `TinyAgentsError::Provider` whose `Display` renders the provider, HTTP
    // status, code, and message (preserving the classification the retry layer
    // needs) rather than the old flattened "<provider> provider error" string.
    assert!(
        err.to_string()
            .contains("mock returned HTTP 500 (internal): nope"),
        "unexpected error string: {err}"
    );
}

#[test]
fn model_profiles_stream_chunks_usage_and_cost_are_additive() {
    let profile = ModelProfile::permissive();
    assert!(profile.satisfies(&CapabilitySet {
        tool_calling: true,
        parallel_tool_calls: true,
        streaming: true,
        native_structured_output: true,
        json_schema: true,
        image_in: true,
        audio_out: true,
        ..CapabilitySet::default()
    }));
    assert!(!profile.satisfies(&CapabilitySet {
        min_input_tokens: Some(1),
        ..CapabilitySet::default()
    }));
    assert!(!ModelProfile::default().satisfies(&CapabilitySet {
        tool_calling: true,
        ..CapabilitySet::default()
    }));

    let chunks = vec![
        StreamChunk::Values(json!({ "state": 1 })),
        StreamChunk::Updates(json!({ "state": 2 })),
        StreamChunk::Message(MessageDelta {
            text: "delta".into(),
            reasoning: String::new(),
            tool_call: Some(ToolDelta {
                call_id: "tool-1".into(),
                content: "partial".into(),
                tool_name: None,
            }),
        }),
        StreamChunk::Debug("trace".into()),
        StreamChunk::Interrupt(json!({ "kind": "input" })),
        StreamChunk::Custom(json!({ "x": true })),
    ];
    assert_eq!(chunks[0].mode(), StreamMode::Values);
    assert_eq!(
        filter_stream(&chunks, &[StreamMode::Messages, StreamMode::Custom]),
        vec![chunks[2].clone(), chunks[5].clone()]
    );

    let mut sink = StreamSink::new([StreamMode::Messages]);
    assert!(sink.is_active(StreamMode::Messages));
    assert!(!sink.is_active(StreamMode::Debug));
    sink.push(chunks[2].clone());
    sink.push(chunks[3].clone());
    assert_eq!(sink.len(), 1);
    sink.enable(StreamMode::Debug);
    sink.push(chunks[3].clone());
    sink.disable(StreamMode::Messages);
    sink.push(chunks[2].clone());
    assert_eq!(sink.peek(), vec![chunks[2].clone(), chunks[3].clone()]);
    assert_eq!(sink.drain().len(), 2);
    assert!(sink.is_empty());
    assert_eq!(StreamSink::all().active_modes().len(), 6);

    let mut usage = Usage::new(10, 5);
    usage.cache_read_tokens = 3;
    usage.cache_creation_tokens = 2;
    usage.reasoning_tokens = 4;
    let mut totals = UsageTotals::new();
    totals.record(usage);
    totals += Usage::new(1, 2);
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.input_tokens, 11);
    assert_eq!(totals.usage.output_tokens, 7);

    let pricing = ModelPricing {
        input_per_token: Some(0.001),
        output_per_token: Some(0.002),
        cache_read_input_per_token: Some(0.0001),
        cache_creation_input_per_token: Some(0.0005),
        output_reasoning_per_token: Some(0.003),
        ..ModelPricing::default()
    };
    // cache_read_tokens (3) and reasoning_tokens (4) are subsets of
    // input_tokens/output_tokens, not additions to them, so the standard
    // rate only applies to the non-cached/non-reasoning remainder:
    // (10-3)*0.001 = 0.007, (5-4)*0.002 = 0.002.
    let cost = estimate_cost(&pricing, &usage);
    assert_eq!(cost.input_cost, 0.007);
    assert_eq!(cost.output_cost, 0.002);
    assert_eq!(cost.cache_cost, 0.0013);
    assert_eq!(cost.reasoning_cost, 0.012);
    assert!((cost.total_cost - 0.0223).abs() < f64::EPSILON);
    let combined = CostTotals::new() + cost + cost;
    assert!((combined.total_cost - 0.0446).abs() < f64::EPSILON);
}

#[test]
fn structured_output_supports_provider_schema_and_tool_fallbacks() {
    #[derive(Debug, Deserialize, PartialEq)]
    struct Score {
        score: u32,
    }

    let schema = json!({
        "type": "object",
        "properties": { "score": { "type": "integer" } },
        "required": ["score"]
    });

    assert_eq!(
        StructuredStrategy::for_profile(None),
        StructuredStrategy::ProviderSchema
    );
    assert_eq!(
        StructuredStrategy::for_profile(Some(&ModelProfile {
            tool_calling: true,
            ..ModelProfile::default()
        })),
        StructuredStrategy::ToolCall
    );

    let provider_format =
        response_format_for_strategy(StructuredStrategy::ProviderSchema, "score", schema.clone());
    assert_eq!(
        provider_format,
        ResponseFormat::json_schema("score", schema.clone())
    );
    let tool_format =
        response_format_for_strategy(StructuredStrategy::ToolCall, "score", schema.clone());
    assert_eq!(tool_format, ResponseFormat::Text);

    let extractor =
        StructuredExtractor::new(StructuredStrategy::ProviderSchema, "score", schema.clone());
    assert_eq!(extractor.schema(), &schema);
    let output = extractor
        .extract(&ModelResponse::assistant(r#"{"score":42}"#))
        .unwrap();
    assert_eq!(output.as_value()["score"], 42);
    assert_eq!(output.raw_text.as_deref(), Some(r#"{"score":42}"#));
    assert_eq!(output.parse::<Score>().unwrap(), Score { score: 42 });

    let invalid = extractor
        .extract(&ModelResponse::assistant("not json"))
        .unwrap_err();
    assert!(
        invalid
            .to_string()
            .contains("response text is not valid JSON")
    );

    let tool_response = ModelResponse {
        message: tinyagents::harness::message::AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![ToolCall {
                id: "tool-1".into(),
                name: "score".into(),
                arguments: json!({ "score": 7 }),
                invalid: None,
            }],
            usage: None,
        },
        usage: None,
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
    };
    let tool_output = StructuredExtractor::new(StructuredStrategy::ToolCall, "score", schema)
        .extract(&tool_response)
        .unwrap();
    assert_eq!(tool_output.as_value()["score"], 7);
    assert!(tool_output.raw_text.is_none());

    let missing = StructuredExtractor::new(
        StructuredStrategy::ToolCall,
        "missing",
        json!({ "type": "object" }),
    )
    .extract(&tool_response)
    .unwrap_err();
    assert!(missing.to_string().contains("no tool call"));
}

#[test]
fn model_catalog_loads_seed_and_custom_snapshots() {
    let snapshot = ModelCatalogSnapshot {
        schema_version: 1,
        snapshot_id: "test-snapshot".into(),
        created_at: "2026-06-29T00:00:00Z".into(),
        currency: "USD".into(),
        unit: "token".into(),
        description: Some("custom".into()),
        sources: vec![ModelCatalogSource {
            name: "unit".into(),
            url: "https://example.invalid/models".into(),
            retrieved_at: "2026-06-29T00:00:00Z".into(),
        }],
        models: vec![ModelCatalogEntry {
            provider: "mock".into(),
            model_id: "mock-large".into(),
            aliases: vec!["large".into()],
            mode: "chat".into(),
            max_input_tokens: Some(1024),
            max_output_tokens: Some(256),
            deprecation_date: None,
            pricing: ModelPricing {
                input_per_token: Some(0.1),
                output_per_token: Some(0.2),
                ..ModelPricing::default()
            },
            capabilities: ModelCapabilities {
                streaming: true,
                tool_calling: true,
                json_schema: true,
                prompt_caching: true,
                ..ModelCapabilities::default()
            },
            source: "unit".into(),
            source_url: Some("https://example.invalid/mock-large".into()),
            raw: json!({ "family": "mock" }),
        }],
    };

    let catalog = ModelCatalog::from_snapshot(snapshot.clone());
    assert_eq!(catalog.snapshot().snapshot_id, "test-snapshot");
    assert_eq!(catalog.models().len(), 1);
    assert_eq!(
        catalog.get("mock", "mock-large").unwrap().model_id,
        "mock-large"
    );
    assert_eq!(catalog.get("mock", "large").unwrap().model_id, "mock-large");
    assert!(
        catalog
            .get_by_model_id("large")
            .unwrap()
            .capabilities
            .tool_calling
    );
    assert!(catalog.get("other", "large").is_none());

    let json = serde_json::to_string(&snapshot).unwrap();
    let parsed = ModelCatalog::from_json(&json).unwrap();
    assert_eq!(parsed.snapshot().sources[0].name, "unit");

    let seed = ModelCatalog::seed().unwrap();
    assert_eq!(seed.snapshot().schema_version, 1);
    assert!(!seed.models().is_empty());
}
