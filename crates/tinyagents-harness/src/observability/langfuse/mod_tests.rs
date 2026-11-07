//! Unit tests for the Langfuse ingestion exporter.

use super::*;
use crate::events::AgentEvent;
use crate::ids::{CallId, EventId, RunId};
use tinyinference_llm::message::MessageDelta;
use tinyinference_llm::usage::ChargedAmount;

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
fn generation_reports_native_usage_cost_and_first_token_time() {
    let client = LangfuseClient::proxy("https://backend.test", "t").unwrap();
    let call_id = CallId::new("model-call");
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig::default(),
            &[
                obs(
                    0,
                    AgentEvent::ModelStarted {
                        call_id: call_id.clone(),
                        model: "chat-v1".into(),
                    },
                ),
                obs(
                    25,
                    AgentEvent::ModelDelta {
                        run_id: RunId::new("run-1"),
                        call_id: call_id.clone(),
                        delta: MessageDelta::text("first token"),
                    },
                ),
                obs(
                    30,
                    AgentEvent::ModelDelta {
                        run_id: RunId::new("run-1"),
                        call_id: call_id.clone(),
                        delta: MessageDelta::text("more"),
                    },
                ),
                obs(
                    100,
                    AgentEvent::ModelCompleted {
                        call_id,
                        started_at_ms: Some(1_704_067_200_000),
                        usage: Some(Usage {
                            input_tokens: 100,
                            output_tokens: 20,
                            total_tokens: 120,
                            cache_read_tokens: 40,
                            reasoning_tokens: 5,
                            charged_amount: Some(ChargedAmount::usd_micros(125)),
                            ..Default::default()
                        }),
                        input: None,
                        output: None,
                    },
                ),
            ],
        )
        .unwrap();
    let generation = batch["batch"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["type"] == "generation-create")
        .unwrap();
    let body = &generation["body"];
    assert_eq!(body["model"], "chat-v1");
    assert_eq!(body["completionStartTime"], iso_ms(1_704_067_200_025));
    assert_eq!(body["startTime"], iso_ms(1_704_067_200_000));
    assert_eq!(body["endTime"], iso_ms(1_704_067_200_100));
    assert_eq!(body["usageDetails"]["input"], 60);
    assert_eq!(body["usageDetails"]["cache_read_input_tokens"], 40);
    assert_eq!(body["usageDetails"]["reasoning_output_tokens"], 5);
    assert_eq!(body["costDetails"]["total"], 0.000125);
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
                        parent_call_id: None,
                        call_id: CallId::new("tool-call"),
                        tool_name: "lookup".to_string(),
                        started_at_ms: Some(1_704_067_199_500),
                        input: Some(json!({ "query": "weather" })),
                        output: Some(json!("sunny")),
                        duration_ms: Some(250),
                        input_bytes: Some(19),
                        output_bytes: Some(5),
                        error: None,
                        metadata: None,
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
    assert_eq!(tool["body"]["metadata"]["input_bytes"], 19);

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
                            parent_call_id: None,
                            call_id: CallId::new("agent_turn-tool-1"),
                            tool_name: "lookup".to_string(),
                            started_at_ms: None,
                            input: None,
                            output: None,
                            duration_ms: None,
                            input_bytes: None,
                            output_bytes: None,
                            error: None,
                            metadata: None,
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
fn nested_tool_span_nests_under_its_parent_call_span() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let completed = |parent: Option<&str>, id: &str, name: &str| AgentEvent::ToolCompleted {
        parent_call_id: parent.map(CallId::new),
        call_id: CallId::new(id),
        tool_name: name.to_string(),
        started_at_ms: None,
        input: None,
        output: None,
        duration_ms: None,
        input_bytes: None,
        output_bytes: None,
        error: None,
        metadata: None,
    };
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig {
                trace_id: Some("trace-1".to_string()),
                ..Default::default()
            },
            &[
                obs(1, completed(Some("p1"), "p1/1", "leaf")),
                obs(2, completed(None, "p1", "caller")),
            ],
        )
        .unwrap();
    let events = batch["batch"].as_array().unwrap();
    let body = |name: &str| {
        events
            .iter()
            .find(|e| e["type"] == "span-create" && e["body"]["name"] == name)
            .unwrap()["body"]
            .clone()
    };
    let (leaf, caller) = (body("leaf"), body("caller"));
    assert_eq!(leaf["parentObservationId"], caller["id"]);
    assert_ne!(caller["parentObservationId"], caller["id"]);
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
                        outcome: None,
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

#[test]
fn builds_numeric_trace_score() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client.build_score_batch(
        LangfuseScore::numeric("trace-1", "helpfulness", 0.9).with_comment("solid answer"),
    );
    let event = &batch["batch"].as_array().unwrap()[0];
    assert_eq!(event["type"], "score-create");
    assert_eq!(event["body"]["traceId"], "trace-1");
    assert_eq!(event["body"]["name"], "helpfulness");
    assert_eq!(event["body"]["value"], 0.9);
    assert_eq!(event["body"]["dataType"], "NUMERIC");
    assert_eq!(event["body"]["comment"], "solid answer");
    // A trace-level score derives a deterministic, observation-free id.
    assert_eq!(event["body"]["id"], "trace-1:score:helpfulness");
    assert!(event["body"].get("observationId").is_none());
}

#[test]
fn builds_categorical_and_boolean_scores() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();

    let categorical =
        client.build_score_batch(LangfuseScore::categorical("trace-1", "verdict", "correct"));
    let cat_body = &categorical["batch"].as_array().unwrap()[0]["body"];
    assert_eq!(cat_body["value"], "correct");
    assert_eq!(cat_body["dataType"], "CATEGORICAL");

    let boolean = client.build_score_batch(LangfuseScore::boolean("trace-1", "passed", true));
    let bool_body = &boolean["batch"].as_array().unwrap()[0]["body"];
    // Boolean scores serialize as 1/0 with a BOOLEAN data type.
    assert_eq!(bool_body["value"], 1);
    assert_eq!(bool_body["dataType"], "BOOLEAN");
}

#[test]
fn score_can_target_a_single_observation() {
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client.build_score_batch(
        LangfuseScore::numeric("trace-1", "faithfulness", 1.0)
            .on_observation("trace-1:agent_turn-model-1"),
    );
    let body = &batch["batch"].as_array().unwrap()[0]["body"];
    assert_eq!(body["observationId"], "trace-1:agent_turn-model-1");
    // The id namespaces the observation so per-generation scores don't collide
    // with the trace-level score of the same name.
    assert_eq!(
        body["id"],
        "trace-1:trace-1:agent_turn-model-1:score:faithfulness"
    );
}

#[test]
fn derives_the_otlp_trace_endpoint_beside_the_ingestion_endpoint() {
    let direct = LangfuseClient::direct("https://lf.example.test/", "pk", "sk").unwrap();
    assert_eq!(
        direct.otlp_endpoint(),
        "https://lf.example.test/api/public/otel/v1/traces"
    );
    let proxy = LangfuseClient::proxy("https://api.example.test", "token").unwrap();
    assert_eq!(
        proxy.otlp_endpoint(),
        "https://api.example.test/telemetry/langfuse/otel/v1/traces"
    );
}

/// One request captured by [`serve_once_with`]: the request line, the
/// authorization header, the `x-langfuse-ingestion-version` header and the
/// decoded body.
type Captured = (String, String, String, String, String, Value);

/// Accepts one HTTP request on a loopback port, answers `status` with an empty
/// JSON object, and returns what it received.
fn serve_once(status: u16) -> (String, std::sync::mpsc::Receiver<Captured>) {
    serve_once_with(status, "{}")
}

/// [`serve_once`] answering with `response` as the JSON body.
fn serve_once_with(
    status: u16,
    response: &'static str,
) -> (String, std::sync::mpsc::Receiver<Captured>) {
    use std::io::{BufRead, BufReader, Read, Write};
    let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", socket.local_addr().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (stream, _) = socket.accept().unwrap();
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let (mut auth, mut version, mut custom, mut injected, mut len) = (
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            0_usize,
        );
        loop {
            let mut header = String::new();
            reader.read_line(&mut header).unwrap();
            if header.trim().is_empty() {
                break;
            }
            let (name, value) = header.split_once(':').unwrap();
            match name.trim().to_ascii_lowercase().as_str() {
                "authorization" => auth = value.trim().to_owned(),
                "x-langfuse-ingestion-version" => version = value.trim().to_owned(),
                "x-test-transport" => custom = value.trim().to_owned(),
                "x-test-header" => injected = value.trim().to_owned(),
                "content-length" => len = value.trim().parse().unwrap(),
                _ => {}
            }
        }
        let mut body = vec![0; len];
        reader.read_exact(&mut body).unwrap();
        let mut stream = reader.into_inner();
        write!(
            stream,
            "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response}",
            response.len()
        )
        .unwrap();
        tx.send((
            line.trim().to_owned(),
            auth,
            version,
            custom,
            injected,
            serde_json::from_slice(&body).unwrap(),
        ))
        .unwrap();
    });
    (base, rx)
}

#[tokio::test]
async fn sends_otlp_json_to_the_trace_endpoint_with_basic_auth() {
    let (base, rx) = serve_once(200);
    let client = LangfuseClient::direct(base, "pk-test", "sk-test").unwrap();
    let body = json!({"resourceSpans": []});
    client.send_otlp(body.clone()).await.unwrap();
    let (line, auth, _, _, _, received) = rx.recv().unwrap();
    assert_eq!(line, "POST /api/public/otel/v1/traces HTTP/1.1");
    assert_eq!(auth, "Basic cGstdGVzdDpzay10ZXN0");
    assert_eq!(received, body);
}

#[tokio::test]
async fn otlp_requests_select_the_langfuse_v4_ingestion_path() {
    for proxied in [false, true] {
        let (base, rx) = serve_once(200);
        let client = if proxied {
            LangfuseClient::proxy(base, "token").unwrap()
        } else {
            LangfuseClient::direct(base, "pk", "sk").unwrap()
        };
        client
            .send_otlp(json!({"resourceSpans": []}))
            .await
            .unwrap();
        let (_, _, version, _, _, _) = rx.recv().unwrap();
        assert_eq!(version, "4", "proxied={proxied}");
    }
}

#[tokio::test]
async fn reports_spans_the_otlp_endpoint_partially_rejected() {
    let (base, _rx) = serve_once_with(
        200,
        r#"{"partialSuccess":{"rejectedSpans":2,"errorMessage":"bad span"}}"#,
    );
    let client = LangfuseClient::direct(base, "pk", "sk").unwrap();
    let err = client
        .send_otlp(json!({"resourceSpans": []}))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("rejected 2"), "{err}");
}

#[tokio::test]
async fn proxy_uses_injected_transport_and_headers_before_partial_success_validation() {
    let (base, rx) = serve_once_with(
        200,
        r#"{"partialSuccess":{"rejectedSpans":"1","errorMessage":"invalid span"}}"#,
    );
    let mut defaults = reqwest::header::HeaderMap::new();
    defaults.insert("x-test-transport", "injected-client".parse().unwrap());
    let transport = reqwest::Client::builder()
        .default_headers(defaults)
        .build()
        .unwrap();
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-test-header", "injected-header".parse().unwrap());
    let client = LangfuseClient::proxy_with_transport(base, "token", transport, headers).unwrap();
    let error = client
        .send_otlp(json!({"resourceSpans": []}))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("rejected 1"), "{error}");
    let (_, auth, version, custom, injected, _) = rx.recv().unwrap();
    assert_eq!(auth, "Bearer token");
    assert_eq!(version, "4");
    assert_eq!(custom, "injected-client");
    assert_eq!(injected, "injected-header");
}

#[tokio::test]
async fn an_empty_partial_success_is_a_success() {
    let (base, _rx) = serve_once_with(200, r#"{"partialSuccess":{}}"#);
    let client = LangfuseClient::direct(base, "pk", "sk").unwrap();
    client
        .send_otlp(json!({"resourceSpans": []}))
        .await
        .unwrap();
}

#[tokio::test]
async fn reports_a_refused_otlp_request() {
    let (base, _rx) = serve_once(413);
    let client = LangfuseClient::proxy(base, "token").unwrap();
    assert!(
        client
            .send_otlp(json!({"resourceSpans": []}))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn sends_observations_as_v4_otlp_spans() {
    let (base, rx) = serve_once(200);
    let client = LangfuseClient::direct(base, "pk", "sk").unwrap();
    let trace = LangfuseTraceConfig {
        trace_id: Some("trace-otlp".into()),
        name: Some("agent.turn".into()),
        session_id: Some("thread-1".into()),
        ..Default::default()
    };
    let responses = client
        .send_observations_otlp(
            trace,
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
                        usage: None,
                        input: None,
                        output: None,
                    },
                ),
            ],
            "tinyagents",
        )
        .await
        .unwrap();
    assert_eq!(responses.len(), 1);
    let (line, _, version, _, _, body) = rx.recv().unwrap();
    assert_eq!(line, "POST /api/public/otel/v1/traces HTTP/1.1");
    assert_eq!(version, "4");
    let spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap();
    assert!(!spans.is_empty());
    for span in spans {
        let session = span["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|attr| attr["key"] == "langfuse.session.id")
            .map(|attr| attr["value"]["stringValue"].clone());
        assert_eq!(session, Some(json!("thread-1")));
    }
}
