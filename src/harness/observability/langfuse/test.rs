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
    // The observation id is namespaced by the (globally-unique) trace id so it
    // cannot collide across turns/threads that reuse the same run-scoped call_id.
    // Here the trace id defaults to the root run id ("root-1").
    assert_eq!(events[2]["body"]["id"], "root-1:model-call");
    // The raw call_id survives in metadata for in-run correlation.
    assert_eq!(events[2]["body"]["metadata"]["call_id"], "model-call");
    assert_eq!(events[2]["body"]["usage"]["input"], 3);
    // Payload-free generation: no input/output body fields.
    assert!(events[2]["body"].get("input").is_none());
    assert!(events[2]["body"].get("output").is_none());
    // Without a captured start time (a pre-`started_at_ms` journal entry) the
    // start falls back to the completion timestamp, a zero-width point.
    assert_eq!(events[2]["body"]["startTime"], events[2]["body"]["endTime"]);
}

#[test]
fn generation_carries_model_from_model_started() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig::default(),
            &[
                obs(
                    0,
                    AgentEvent::ModelStarted {
                        call_id: CallId::new("model-call"),
                        model: "managed.chat-v1".to_string(),
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
    let generation = events
        .iter()
        .find(|e| e["type"] == "generation-create")
        .expect("a generation-create observation");
    // The generation is projected from `ModelCompleted` (which carries no model);
    // it is correlated by `call_id` to the model that `ModelStarted` announced,
    // so Langfuse can map pricing instead of recording cost $0.
    assert_eq!(generation["body"]["model"], "managed.chat-v1");
    assert_eq!(generation["body"]["metadata"]["call_id"], "model-call");
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
    let generation = events
        .iter()
        .find(|e| e["type"] == "generation-create")
        .expect("a generation-create observation");
    let tool = events
        .iter()
        .find(|e| e["type"] == "span-create" && e["body"]["name"] == "lookup")
        .expect("the tool span");
    assert_eq!(generation["body"]["input"][0]["content"], "hi");
    assert_eq!(generation["body"]["output"]["content"], "hello there");
    assert_eq!(tool["body"]["input"]["query"], "weather");
    assert_eq!(tool["body"]["output"], "sunny");

    // The loop-captured start time gives each observation a real duration
    // (start < end) instead of the zero-width start == end point.
    assert_eq!(generation["body"]["startTime"], iso_ms(1_704_067_199_000));
    assert_eq!(generation["body"]["endTime"], iso_ms(1_704_067_200_000));
    assert_eq!(tool["body"]["startTime"], iso_ms(1_704_067_199_500));
    // The tool span's end time is now start + the event's own duration_ms
    // (1_704_067_199_500 + 250), a real execution window rather than the
    // journal-append timestamp.
    assert_eq!(tool["body"]["endTime"], iso_ms(1_704_067_199_750));
    // Result size rides metadata even though this fixture also captured output.
    assert_eq!(tool["body"]["metadata"]["output_bytes"], 5);

    // Both the generation and the tool span nest under the run's span, which
    // itself sits directly under the trace — the LangChain run-tree shape.
    assert_eq!(
        generation["body"]["parentObservationId"],
        "root-1:run:run-1"
    );
    assert_eq!(tool["body"]["parentObservationId"], "root-1:run:run-1");
    let run_span = events
        .iter()
        .find(|e| e["body"]["id"] == "root-1:run:run-1")
        .expect("a run span");
    assert_eq!(run_span["type"], "span-create");

    // Observation metadata carries only lineage + event kind, not the whole
    // event payload (which would duplicate input/output already in `body`).
    let gen_meta = &generation["body"]["metadata"];
    assert_eq!(gen_meta["event_kind"], "model.completed");
    assert!(
        gen_meta.get("event").is_none(),
        "full event payload must not be duplicated into metadata"
    );
    assert_eq!(gen_meta["run_id"], "run-1");
}

#[test]
fn call_scoped_observation_ids_are_unique_per_trace() {
    // Regression for the Langfuse id-collision bug: two turns on two different
    // threads reuse the SAME run-scoped call_id (`agent_turn-model-1`, the
    // deterministic id every interactive turn gets). Because Langfuse upserts
    // observations by `id` project-wide, a bare call_id made each new turn's
    // model generation overwrite the previous one onto the newest trace — so
    // every earlier trace lost its model usage/cost/content. The id must now be
    // distinct per trace.
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();

    let build = |trace_id: &str| {
        client
            .build_ingestion_batch(
                LangfuseTraceConfig {
                    trace_id: Some(trace_id.to_string()),
                    ..Default::default()
                },
                &[
                    obs(
                        1,
                        AgentEvent::ModelCompleted {
                            call_id: CallId::new("agent_turn-model-1"),
                            started_at_ms: None,
                            usage: Some(Usage {
                                input_tokens: 1,
                                output_tokens: 1,
                                total_tokens: 2,
                                ..Default::default()
                            }),
                            input: None,
                            output: None,
                        },
                    ),
                    obs(
                        2,
                        AgentEvent::ToolCompleted {
                            call_id: CallId::new("agent_turn-tool-1"),
                            tool_name: "lookup".to_string(),
                            started_at_ms: None,
                            input: None,
                            output: None,
                            duration_ms: None,
                            output_bytes: None,
                            error: None,
                        },
                    ),
                ],
            )
            .unwrap()
    };

    let extract_ids = |batch: &Value| -> (String, String) {
        let events = batch["batch"].as_array().unwrap();
        let gen_id = events
            .iter()
            .find(|e| e["type"] == "generation-create")
            .unwrap()["body"]["id"]
            .as_str()
            .unwrap()
            .to_string();
        // The tool span, not the run span (which is also a `span-create`).
        let span_id = events
            .iter()
            .find(|e| e["type"] == "span-create" && e["body"]["name"] == "lookup")
            .unwrap()["body"]["id"]
            .as_str()
            .unwrap()
            .to_string();
        (gen_id, span_id)
    };

    let (gen_a, span_a) = extract_ids(&build("thread-A:turn-1"));
    let (gen_b, span_b) = extract_ids(&build("thread-B:turn-2"));

    // Same call_id, different traces → different observation ids (no overwrite).
    assert_ne!(
        gen_a, gen_b,
        "model generation ids must not collide across traces"
    );
    assert_ne!(
        span_a, span_b,
        "tool span ids must not collide across traces"
    );
    assert_eq!(gen_a, "thread-A:turn-1:agent_turn-model-1");
    assert_eq!(span_b, "thread-B:turn-2:agent_turn-tool-1");
}

#[test]
fn sub_agent_run_nests_under_its_parent_run_span() {
    // A child run (distinct run_id, parented to the top-level run) exported in
    // its own batch must reference its parent's run span so the two batches
    // reconstruct one recursion tree under a shared trace.
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();

    let child = AgentObservation {
        event_id: EventId::new("evt-child"),
        run_id: RunId::new("child-run"),
        parent_run_id: Some(RunId::new("run-1")),
        root_run_id: RunId::new("root-1"),
        offset: 0,
        ts_ms: 1_704_067_200_500,
        event: AgentEvent::ModelCompleted {
            call_id: CallId::new("child-model"),
            started_at_ms: None,
            usage: None,
            input: None,
            output: None,
        },
    };

    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig {
                trace_id: Some("root-1".to_string()),
                ..Default::default()
            },
            &[child],
        )
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    // The child run gets a "sub-agent" span parented to the parent run's span,
    // and its generation nests under that sub-agent span.
    let run_span = events
        .iter()
        .find(|e| e["type"] == "span-create" && e["body"]["name"] == "sub-agent")
        .expect("a sub-agent run span");
    assert_eq!(run_span["body"]["id"], "root-1:run:child-run");
    assert_eq!(
        run_span["body"]["parentObservationId"], "root-1:run:run-1",
        "the sub-agent span nests under its parent run's span (from the parent batch)"
    );
    let generation = events
        .iter()
        .find(|e| e["type"] == "generation-create")
        .expect("the child generation");
    assert_eq!(
        generation["body"]["parentObservationId"],
        "root-1:run:child-run"
    );
}

#[test]
fn run_span_carries_run_error_and_window() {
    // RunStarted/RunFailed are folded into the run span rather than emitted as
    // standalone events: the span carries the start/end window and ERROR status.
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig::default(),
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
                    AgentEvent::RunFailed {
                        run_id: RunId::new("run-1"),
                        error: "boom".to_string(),
                    },
                ),
            ],
        )
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    // No standalone run.started / run.failed events survive.
    assert!(
        !events
            .iter()
            .any(|e| e["body"]["name"] == "run.started" || e["body"]["name"] == "run.failed"),
        "run-lifecycle events are consumed into the run span"
    );
    let run_span = events
        .iter()
        .find(|e| e["body"]["id"] == "root-1:run:run-1")
        .expect("a run span");
    assert_eq!(run_span["type"], "span-create");
    assert_eq!(run_span["body"]["level"], "ERROR");
    assert_eq!(run_span["body"]["statusMessage"], "boom");
    assert_eq!(run_span["body"]["startTime"], iso_ms(1_704_067_200_000));
    assert_eq!(run_span["body"]["endTime"], iso_ms(1_704_067_200_001));
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
