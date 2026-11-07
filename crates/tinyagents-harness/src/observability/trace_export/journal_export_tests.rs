use std::borrow::Cow;

use serde_json::json;
use tinyinference_llm::usage::Usage;

use super::*;
use crate::events::AgentEvent;
use crate::ids::{CallId, EventId, RunId};
use crate::observability::trace_export::RunType;
use crate::observability::{AgentObservation, LangfuseClient};

const BRAND: ExportBrand<'static> = ExportBrand {
    product: "openhuman",
    version: "9.9.9",
};

fn obs(offset: u64, event: AgentEvent) -> AgentObservation {
    AgentObservation {
        event_id: EventId::new(format!("run-1-evt-{offset}")),
        run_id: RunId::new("run-1"),
        parent_run_id: None,
        root_run_id: RunId::new("run-1"),
        offset,
        ts_ms: 1_000 + offset,
        event,
    }
}

#[test]
fn trace_config_from_context_matches_span_trace_attribution() {
    let ctx = TraceContext::new("trace:req-1", Some("user-1".to_string()))
        .with_session_group("thread-abc")
        .with_client_id("socket-abc")
        .with_agent_id("researcher")
        .with_channel_source("chat")
        .with_run_type(RunType::InteractiveChat);

    let trace = trace_config_from_context(&ctx, "staging", &BRAND);
    assert_eq!(trace.trace_id.as_deref(), Some("trace:req-1"));
    assert_eq!(trace.name.as_deref(), Some("agent.turn:researcher"));
    assert_eq!(trace.user_id.as_deref(), Some("user-1"));
    assert_eq!(trace.session_id.as_deref(), Some("thread-abc"));
    assert_eq!(trace.environment.as_deref(), Some("staging"));
    assert_eq!(trace.tags, vec!["run:interactive_chat", "source:chat"]);
    assert_eq!(trace.metadata["client.id"], "socket-abc");
    assert_eq!(trace.metadata["agent.id"], "researcher");
    assert_eq!(trace.metadata["channel.source"], "chat");
    assert_eq!(trace.metadata["run_type"], "interactive_chat");
    assert_eq!(trace.metadata["app.version"], BRAND.version);
}

#[test]
fn trace_config_from_context_stamps_run_lineage() {
    // A spawned sub-agent: its run has a parent (the spawning turn) and a
    // root. Stamping these onto trace metadata is what links the sub-agent's
    // Langfuse trace back to the parent turn (#4657).
    let ctx = TraceContext::new("trace:req-1", None).with_run_lineage(
        Some("sub-run".to_string()),
        Some("parent-run".to_string()),
        Some("root-run".to_string()),
    );
    let trace = trace_config_from_context(&ctx, "staging", &BRAND);
    assert_eq!(trace.metadata["run_id"], "sub-run");
    assert_eq!(trace.metadata["parent_run_id"], "parent-run");
    assert_eq!(trace.metadata["root_run_id"], "root-run");
}

#[test]
fn trace_config_omits_parent_run_id_for_top_level_turn() {
    // A top-level turn has no parent; the key must stay absent (root == run).
    let ctx = TraceContext::new("trace:req-1", None).with_run_lineage(
        Some("run-1".to_string()),
        None,
        Some("run-1".to_string()),
    );
    let trace = trace_config_from_context(&ctx, "staging", &BRAND);
    assert_eq!(trace.metadata["run_id"], "run-1");
    assert_eq!(trace.metadata["root_run_id"], "run-1");
    assert!(
        trace.metadata.get("parent_run_id").is_none(),
        "top-level turn must not carry a parent_run_id"
    );
}

#[test]
fn trace_ctx_with_run_lineage_derives_from_subagent_observations() {
    // Sub-agent observations carry parent/root ids pointing at the spawning
    // turn; the derived trace context stamps them so the sub-agent's trace
    // links back instead of landing as a disconnected sibling (#4657).
    let observations = vec![AgentObservation {
        event_id: EventId::new("evt-1"),
        run_id: RunId::new("sub-run"),
        parent_run_id: Some(RunId::new("parent-run")),
        root_run_id: RunId::new("root-run"),
        offset: 1,
        ts_ms: 1_000,
        event: AgentEvent::ModelCompleted {
            call_id: CallId::new("model-1"),
            started_at_ms: Some(1_000),
            usage: Some(Usage::new(1, 1)),
            input: None,
            output: None,
        },
    }];
    let base = TraceContext::new("trace:req-1", None);

    let enriched = trace_ctx_with_run_lineage(&base, &observations);
    assert_eq!(enriched.run_id.as_deref(), Some("sub-run"));
    assert_eq!(enriched.parent_run_id.as_deref(), Some("parent-run"));
    assert_eq!(enriched.root_run_id.as_deref(), Some("root-run"));

    // An empty stream leaves the context untouched (no lineage invented).
    let untouched = trace_ctx_with_run_lineage(&base, &[]);
    assert!(untouched.run_id.is_none());
    assert!(untouched.parent_run_id.is_none());
    assert!(untouched.root_run_id.is_none());
}

#[test]
fn child_run_roots_its_own_trace_and_preserves_parent_lineage() {
    let child = AgentObservation {
        event_id: EventId::new("child-evt-1"),
        run_id: RunId::new("child-run"),
        parent_run_id: Some(RunId::new("parent-run")),
        root_run_id: RunId::new("root-run"),
        offset: 1,
        ts_ms: 1_000,
        event: AgentEvent::ModelCompleted {
            call_id: CallId::new("model-1"),
            started_at_ms: Some(900),
            usage: Some(Usage::new(10, 3)),
            input: None,
            output: None,
        },
    };
    let ctx = trace_ctx_with_run_lineage(
        &TraceContext::new("subagent:child-run", Some("user-1".into()))
            .with_session_group("thread-1")
            .with_run_lineage(
                Some("child-run".into()),
                Some("parent-run".into()),
                Some("root-run".into()),
            ),
        &root_subagent_observations(std::slice::from_ref(&child)),
    );
    let trace = trace_config_from_context(&ctx, "production", &BRAND);
    assert_eq!(trace.session_id.as_deref(), Some("thread-1"));
    assert_eq!(trace.metadata["parent_run_id"], "parent-run");
    assert_eq!(trace.metadata["root_run_id"], "root-run");
    let rooted = root_subagent_observations(&[child]);
    assert!(rooted[0].parent_run_id.is_none());
    assert_eq!(rooted[0].root_run_id.as_str(), "child-run");
    let client = LangfuseClient::proxy("https://api.tinyhumans.ai", "token").unwrap();
    let batch = client.build_ingestion_batch(trace, &rooted).unwrap();
    let child_run = batch["batch"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["type"] == "span-create" && event["body"]["name"] == "agent")
        .unwrap();
    assert!(child_run["body"].get("parentObservationId").is_none());
}

#[test]
fn journal_observation_content_follows_capture_gate() {
    let observations = vec![
        obs(
            1,
            AgentEvent::ModelCompleted {
                call_id: CallId::new("model-1"),
                started_at_ms: Some(1_000),
                usage: Some(Usage::new(10, 3)),
                input: Some(json!([{"role": "user", "content": "secret prompt"}])),
                output: Some(json!({"role": "assistant", "content": "secret reply"})),
            },
        ),
        obs(
            2,
            AgentEvent::ToolCompleted {
                parent_call_id: None,
                call_id: CallId::new("tool-1"),
                tool_name: "search".to_string(),
                started_at_ms: Some(1_010),
                input: Some(json!({"query": "secret"})),
                output: Some(json!("secret result")),
                duration_ms: Some(20),
                input_bytes: None,
                output_bytes: Some(13),
                error: None,
                metadata: None,
            },
        ),
    ];

    let off_ctx = TraceContext::new("trace:req-1", None);
    let filtered = observations_for_export(&off_ctx, &observations);
    assert!(matches!(filtered, Cow::Owned(_)));
    match &filtered[0].event {
        AgentEvent::ModelCompleted { input, output, .. } => {
            assert!(input.is_none());
            assert!(output.is_none());
        }
        other => panic!("unexpected event: {other:?}"),
    }
    match &filtered[1].event {
        AgentEvent::ToolCompleted { input, output, .. } => {
            assert!(input.is_none());
            assert!(output.is_none());
        }
        other => panic!("unexpected event: {other:?}"),
    }
    match &observations[0].event {
        AgentEvent::ModelCompleted { input, output, .. } => {
            assert!(input.is_some(), "source journal observation stays intact");
            assert!(output.is_some(), "source journal observation stays intact");
        }
        other => panic!("unexpected event: {other:?}"),
    }

    let on_ctx = TraceContext::new("trace:req-1", None).with_capture_content(true);
    let passthrough = observations_for_export(&on_ctx, &observations);
    assert!(matches!(passthrough, Cow::Borrowed(_)));
    match &passthrough[1].event {
        AgentEvent::ToolCompleted { input, output, .. } => {
            assert_eq!(input.as_ref(), Some(&json!({"query": "secret"})));
            assert_eq!(output.as_ref(), Some(&json!("secret result")));
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn run_telemetry_inserts_aggregate_generation() {
    let observations = vec![obs(
        1,
        AgentEvent::ModelCompleted {
            call_id: CallId::new("model-1"),
            started_at_ms: Some(1_000),
            usage: None,
            input: None,
            output: None,
        },
    )];
    let client = LangfuseClient::proxy("https://api.tinyhumans.ai", "token").expect("proxy client");
    let trace = trace_config_from_context(
        &TraceContext::new("trace:req-1", None),
        "production",
        &BRAND,
    );
    let mut payload = client
        .build_ingestion_batch(trace, &observations)
        .expect("batch");
    let telemetry = RunTotals {
        run_id: "req-1".to_string(),
        input_tokens: 120,
        output_tokens: 30,
        cached_input_tokens: 40,
        cost_usd: 0.0123,
        tool_count: 2,
        model: Some("managed.chat-v1".to_string()),
        provider: Some("managed".to_string()),
        error: None,
    };

    assert!(insert_run_telemetry_generation(
        &mut payload,
        Some(&telemetry),
        &BRAND
    ));
    let batch = payload["batch"].as_array().expect("batch array");
    assert_eq!(batch[1]["type"], "generation-create");
    let body = &batch[1]["body"];
    assert_eq!(body["id"], "trace:req-1:openhuman-run-telemetry");
    assert_eq!(body["name"], "run.total");
    assert_eq!(body["traceId"], "trace:req-1");
    assert_eq!(body["model"], "managed.chat-v1");
    assert_eq!(body["usageDetails"]["input"], 80);
    assert_eq!(body["usageDetails"]["output"], 30);
    assert_eq!(body["usageDetails"]["total"], 150);
    assert_eq!(body["usageDetails"]["cache_read_input_tokens"], 40);
    assert_eq!(body["costDetails"]["total"], 0.0123);
    assert_eq!(body["metadata"]["source"], "openhuman.run_telemetry");
    assert_eq!(body["metadata"]["run_id"], "req-1");
    assert_eq!(body["metadata"]["tool_count"], 2);
    assert_eq!(body["metadata"]["provider"], "managed");

    let batch_for = |observations: &[AgentObservation]| {
        client
            .build_ingestion_batch(
                trace_config_from_context(
                    &TraceContext::new("trace:req-1", None),
                    "production",
                    &BRAND,
                ),
                observations,
            )
            .unwrap()
    };
    let model_call = |id: &str, usage: Option<Usage>| {
        obs(
            1,
            AgentEvent::ModelCompleted {
                call_id: CallId::new(id),
                started_at_ms: Some(1_000),
                usage,
                input: None,
                output: None,
            },
        )
    };

    // Per-call usage covers the run: the aggregate adds only the missing cost.
    let with_usage = vec![model_call("model-1", Some(Usage::new(100, 20)))];
    let covered_totals = RunTotals {
        input_tokens: 100,
        output_tokens: 20,
        cached_input_tokens: 0,
        ..telemetry.clone()
    };
    let mut cost_only = batch_for(&with_usage);
    assert!(insert_run_telemetry_generation(
        &mut cost_only,
        Some(&covered_totals),
        &BRAND
    ));
    let body = &cost_only["batch"][1]["body"];
    assert_eq!(body["name"], "run.total");
    assert!(body.get("usageDetails").is_none(), "tokens counted twice");
    assert_eq!(body["costDetails"]["total"], 0.0123);
    let mut no_cost = batch_for(&with_usage);
    let zero_cost = RunTotals {
        cost_usd: 0.0,
        ..covered_totals.clone()
    };
    assert!(!insert_run_telemetry_generation(
        &mut no_cost,
        Some(&zero_cost),
        &BRAND
    ));

    // One call reported usage and one did not: only the uncovered remainder
    // goes on the aggregate.
    let partial = vec![
        model_call("model-1", Some(Usage::new(100, 20))),
        model_call("model-2", None),
    ];
    let mut remainder = batch_for(&partial);
    assert!(insert_run_telemetry_generation(
        &mut remainder,
        Some(&telemetry),
        &BRAND
    ));
    let usage = &remainder["batch"][1]["body"]["usageDetails"];
    assert_eq!(usage["input"], 0);
    assert_eq!(usage["cache_read_input_tokens"], 20);
    assert_eq!(usage["output"], 10);
    assert_eq!(usage["total"], 30);

    let mut already_charged = client
        .build_ingestion_batch(
            trace_config_from_context(
                &TraceContext::new("trace:req-1", None),
                "production",
                &BRAND,
            ),
            &with_usage,
        )
        .unwrap();
    already_charged["batch"][2]["body"]["costDetails"] = json!({ "total": 0.0123 });
    let before = already_charged["batch"].as_array().unwrap().len();
    assert!(!insert_run_telemetry_generation(
        &mut already_charged,
        Some(&telemetry),
        &BRAND
    ));
    assert_eq!(already_charged["batch"].as_array().unwrap().len(), before);
}
