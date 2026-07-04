//! Unit tests for the Langfuse ingestion exporter.

use super::*;
use crate::harness::events::AgentEvent;
use crate::harness::ids::{CallId, EventId, RunId};

fn obs(offset: u64, event: AgentEvent) -> AgentObservation {
    AgentObservation {
        event_id: EventId::new(format!("evt-{offset}")),
        run_id: RunId::new("run-1"),
        parent_run_id: None,
        root_run_id: RunId::new("root-1"),
        offset,
        ts_ms: 1_704_067_200_000 + offset,
        event,
    }
}

#[test]
fn normalizes_langfuse_endpoints() {
    let client = LangfuseClient::proxy("https://api.example.test", "token").unwrap();
    assert_eq!(
        client.endpoint(),
        "https://api.example.test/telemetry/langfuse/ingestion"
    );
    let client = LangfuseClient::proxy(
        "https://api.example.test/telemetry/langfuse/ingestion",
        "token",
    )
    .unwrap();
    assert_eq!(
        client.endpoint(),
        "https://api.example.test/telemetry/langfuse/ingestion"
    );
}

#[test]
fn builds_trace_and_generation_batch() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig {
                user_id: Some("user-1".to_string()),
                session_id: Some("thread-1".to_string()),
                ..Default::default()
            },
            &[
                obs(
                    0,
                    AgentEvent::RunStarted {
                        run_id: RunId::new("run-1"),
                        thread_id: None,
                    },
                ),
                obs(
                    1,
                    AgentEvent::ModelCompleted {
                        call_id: CallId::new("model-call"),
                        started_at_ms: None,
                        usage: Some(Usage {
                            input_tokens: 3,
                            output_tokens: 4,
                            total_tokens: 7,
                            ..Default::default()
                        }),
                        input: None,
                        output: None,
                    },
                ),
            ],
        )
        .unwrap();

    let events = batch["batch"].as_array().unwrap();
    assert_eq!(events[0]["type"], "trace-create");
    assert_eq!(events[0]["body"]["id"], "root-1");
    assert_eq!(events[0]["body"]["userId"], "user-1");
    // Trace metadata is defaulted from run lineage even without a caller value.
    assert_eq!(events[0]["body"]["metadata"]["root_run_id"], "root-1");
    assert_eq!(events[0]["body"]["metadata"]["run_id"], "run-1");
    assert_eq!(events[2]["type"], "generation-create");
    assert_eq!(events[2]["body"]["id"], "model-call");
    assert_eq!(events[2]["body"]["usage"]["input"], 3);
    // Payload-free generation: no input/output body fields.
    assert!(events[2]["body"].get("input").is_none());
    assert!(events[2]["body"].get("output").is_none());
    // Without a captured start time (a pre-`started_at_ms` journal entry) the
    // start falls back to the completion timestamp, a zero-width point.
    assert_eq!(events[2]["body"]["startTime"], events[2]["body"]["endTime"]);
}

#[test]
fn populates_generation_and_tool_io_when_captured() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig::default(),
            &[
                obs(
                    0,
                    AgentEvent::ModelCompleted {
                        call_id: CallId::new("model-call"),
                        started_at_ms: Some(1_704_067_199_000),
                        usage: None,
                        input: Some(json!([{ "role": "user", "content": "hi" }])),
                        output: Some(json!({ "content": "hello there" })),
                    },
                ),
                obs(
                    1,
                    AgentEvent::ToolCompleted {
                        call_id: CallId::new("tool-call"),
                        tool_name: "lookup".to_string(),
                        started_at_ms: Some(1_704_067_199_500),
                        input: Some(json!({ "query": "weather" })),
                        output: Some(json!("sunny")),
                        duration_ms: Some(250),
                        output_bytes: Some(5),
                        error: None,
                    },
                ),
            ],
        )
        .unwrap();

    let events = batch["batch"].as_array().unwrap();
    assert_eq!(events[1]["type"], "generation-create");
    assert_eq!(events[1]["body"]["input"][0]["content"], "hi");
    assert_eq!(events[1]["body"]["output"]["content"], "hello there");
    assert_eq!(events[2]["type"], "span-create");
    assert_eq!(events[2]["body"]["input"]["query"], "weather");
    assert_eq!(events[2]["body"]["output"], "sunny");

    // The loop-captured start time gives each observation a real duration
    // (start < end) instead of the zero-width start == end point.
    assert_eq!(events[1]["body"]["startTime"], iso_ms(1_704_067_199_000));
    assert_eq!(events[1]["body"]["endTime"], iso_ms(1_704_067_200_000));
    assert_eq!(events[2]["body"]["startTime"], iso_ms(1_704_067_199_500));
    // The tool span's end time is now start + the event's own duration_ms
    // (1_704_067_199_500 + 250), a real execution window rather than the
    // journal-append timestamp.
    assert_eq!(events[2]["body"]["endTime"], iso_ms(1_704_067_199_750));
    // Result size rides metadata even though this fixture also captured output.
    assert_eq!(events[2]["body"]["metadata"]["output_bytes"], 5);

    // Observation metadata carries only lineage + event kind, not the whole
    // event payload (which would duplicate input/output already in `body`).
    let gen_meta = &events[1]["body"]["metadata"];
    assert_eq!(gen_meta["event_kind"], "model.completed");
    assert!(
        gen_meta.get("event").is_none(),
        "full event payload must not be duplicated into metadata"
    );
    assert_eq!(gen_meta["run_id"], "run-1");
}

#[test]
fn merges_caller_trace_metadata_over_defaults() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig {
                metadata: json!({ "deployment": "prod", "root_run_id": "override" }),
                ..Default::default()
            },
            &[obs(
                0,
                AgentEvent::RunStarted {
                    run_id: RunId::new("run-1"),
                    thread_id: None,
                },
            )],
        )
        .unwrap();

    let meta = &batch["batch"].as_array().unwrap()[0]["body"]["metadata"];
    assert_eq!(meta["deployment"], "prod");
    // Caller keys win on collision with the defaulted lineage.
    assert_eq!(meta["root_run_id"], "override");
    assert_eq!(meta["run_id"], "run-1");
}
