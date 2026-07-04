//! Unit tests for the graph Langfuse exporter: batch shape, timed spans,
//! failure-level promotion, health telemetry, and trace-id alignment.

use super::*;
use crate::graph::observability::GraphObservation;
use crate::graph::stream::GraphEvent;
use crate::harness::ids::{CheckpointId, EventId, GraphId, NodeId, RunId, ThreadId};

/// `2024-01-01T00:00:00.000Z` in epoch millis; test deltas add onto this so ISO
/// assertions read as `…:00.0NNZ`.
const BASE: u64 = 1_704_067_200_000;

/// Builds an observation `delta_ms` after [`BASE`] with the given
/// offset/step/event under a fixed run.
fn obs(offset: u64, step: usize, delta_ms: u64, event: GraphEvent) -> GraphObservation {
    let ts_ms = BASE + delta_ms;
    GraphObservation {
        event_id: EventId::new(format!("evt-{offset}")),
        run_id: RunId::new("run-1"),
        root_run_id: RunId::new("root-1"),
        parent_run_id: None,
        thread_id: Some(ThreadId::new("thread-1")),
        graph_id: GraphId::new("demo-graph"),
        checkpoint_id: None,
        namespace: Vec::new(),
        step,
        offset,
        ts_ms,
        event,
    }
}

/// A representative run: start → step → node a (ok) → node b (fail) → step end
/// → run fail.
fn sample_run() -> Vec<GraphObservation> {
    let a = NodeId::new("a");
    let b = NodeId::new("b");
    vec![
        obs(
            0,
            0,
            1_000,
            GraphEvent::RunStarted {
                run_id: RunId::new("run-1"),
            },
        ),
        obs(
            1,
            1,
            1_010,
            GraphEvent::StepStarted {
                step: 1,
                active: vec![a.clone(), b.clone()],
            },
        ),
        obs(
            2,
            1,
            1_020,
            GraphEvent::NodeStarted {
                node: a.clone(),
                step: 1,
            },
        ),
        obs(
            3,
            1,
            1_045,
            GraphEvent::NodeCompleted {
                node: a.clone(),
                step: 1,
            },
        ),
        obs(
            4,
            1,
            1_050,
            GraphEvent::NodeStarted {
                node: b.clone(),
                step: 1,
            },
        ),
        obs(
            5,
            1,
            1_080,
            GraphEvent::NodeFailed {
                node: b.clone(),
                step: 1,
                error: "boom".to_string(),
            },
        ),
        obs(6, 1, 1_090, GraphEvent::StepCompleted { step: 1 }),
        obs(
            7,
            1,
            1_100,
            GraphEvent::RunFailed {
                run_id: RunId::new("run-1"),
                error: "boom".to_string(),
            },
        ),
    ]
}

fn exporter() -> GraphLangfuseExporter {
    GraphLangfuseExporter::new(
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "tok").unwrap(),
    )
}

/// Finds the first batch item of the given `type` whose body `id` matches.
fn find<'a>(batch: &'a [Value], ty: &str, id: &str) -> Option<&'a Value> {
    batch
        .iter()
        .find(|e| e["type"] == ty && e["body"]["id"] == id)
}

#[test]
fn empty_observations_are_rejected() {
    let err = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &[])
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[test]
fn trace_defaults_to_root_run_and_graph_id() {
    let batch = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &sample_run())
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    let trace = &events[0];
    assert_eq!(trace["type"], "trace-create");
    // Trace id aligns with the harness exporter (root run id) so agent tool
    // spans land under the same trace.
    assert_eq!(trace["body"]["id"], "root-1");
    assert_eq!(trace["body"]["name"], "demo-graph");
    assert_eq!(trace["body"]["sessionId"], "thread-1");
}

#[test]
fn config_overrides_trace_id_name_and_session() {
    let batch = exporter()
        .build_ingestion_batch(
            LangfuseTraceConfig {
                trace_id: Some("custom-trace".to_string()),
                name: Some("My Graph".to_string()),
                session_id: Some("sess-9".to_string()),
                user_id: Some("user-1".to_string()),
                ..Default::default()
            },
            &sample_run(),
        )
        .unwrap();
    let events = batch["batch"].as_array().unwrap();
    let trace = &events[0];
    assert_eq!(trace["body"]["id"], "custom-trace");
    assert_eq!(trace["body"]["name"], "My Graph");
    assert_eq!(trace["body"]["sessionId"], "sess-9");
    assert_eq!(trace["body"]["userId"], "user-1");
    // Spans reference the overridden trace id.
    let events: &[Value] = events;
    assert!(find(events, "span-create", "custom-trace:step:1").is_some());
}

#[test]
fn steps_and_nodes_become_timed_spans() {
    let batch = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &sample_run())
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    // Superstep span with real start/end times.
    let step = find(events, "span-create", "root-1:step:1").expect("step span");
    assert_eq!(step["body"]["name"], "step 1");
    assert_eq!(step["body"]["startTime"], "2024-01-01T00:00:01.010Z");
    assert_eq!(step["body"]["endTime"], "2024-01-01T00:00:01.090Z");
    // Steps parent to the trace, not another span.
    assert!(step["body"].get("parentObservationId").is_none());

    // Node a completed cleanly and is parented to its step span.
    let node_a = find(events, "span-create", "root-1:node:a:1").expect("node a span");
    assert_eq!(node_a["body"]["name"], "a");
    assert_eq!(node_a["body"]["parentObservationId"], "root-1:step:1");
    assert!(node_a["body"].get("level").is_none());
    assert_eq!(node_a["body"]["endTime"], "2024-01-01T00:00:01.045Z");
}

#[test]
fn node_failure_is_promoted_to_error_span() {
    let batch = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &sample_run())
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    let node_b = find(events, "span-create", "root-1:node:b:1").expect("node b span");
    assert_eq!(node_b["body"]["level"], "ERROR");
    assert_eq!(node_b["body"]["statusMessage"], "boom");
    assert_eq!(node_b["body"]["endTime"], "2024-01-01T00:00:01.080Z");
}

#[test]
fn run_failed_is_an_error_event_and_run_started_is_not_duplicated() {
    let batch = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &sample_run())
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    // RunStarted is represented by the trace, not a separate event.
    assert!(
        !events
            .iter()
            .any(|e| e["body"]["name"] == "run.started" && e["type"] == "event-create")
    );

    let failed = events
        .iter()
        .find(|e| e["body"]["name"] == "run.failed")
        .expect("run.failed event");
    assert_eq!(failed["type"], "event-create");
    assert_eq!(failed["body"]["level"], "ERROR");
    assert_eq!(failed["body"]["statusMessage"], "boom");
}

#[test]
fn health_telemetry_rides_on_trace_metadata() {
    let batch = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &sample_run())
        .unwrap();
    let events = batch["batch"].as_array().unwrap();
    let health = &events[0]["body"]["metadata"]["health"];

    assert_eq!(health["total_started"], 2);
    assert_eq!(health["total_completed"], 1);
    assert_eq!(health["total_failed"], 1);
    assert_eq!(health["run_failed"], true);
    let nodes = health["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0]["node"], "a");
    assert_eq!(nodes[1]["node"], "b");
    assert_eq!(nodes[1]["failed"], 1);
}

#[test]
fn open_node_span_has_start_but_no_end() {
    let a = NodeId::new("a");
    let observations = vec![
        obs(
            0,
            1,
            1_000,
            GraphEvent::StepStarted {
                step: 1,
                active: vec![a.clone()],
            },
        ),
        obs(
            1,
            1,
            1_010,
            GraphEvent::NodeStarted {
                node: a.clone(),
                step: 1,
            },
        ),
    ];
    let batch = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &observations)
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    let node = find(events, "span-create", "root-1:node:a:1").expect("open node span");
    assert_eq!(node["body"]["startTime"], "2024-01-01T00:00:01.010Z");
    assert!(node["body"].get("endTime").is_none());
}

#[test]
fn checkpoint_events_carry_coordinates_in_metadata() {
    let mut observations = sample_run();
    observations.push(GraphObservation {
        checkpoint_id: Some(CheckpointId::new("ckpt-7")),
        ..obs(
            8,
            1,
            1_110,
            GraphEvent::CheckpointSaved {
                checkpoint_id: CheckpointId::new("ckpt-7"),
            },
        )
    });
    let batch = exporter()
        .build_ingestion_batch(LangfuseTraceConfig::default(), &observations)
        .unwrap();
    let events = batch["batch"].as_array().unwrap();

    let ckpt = events
        .iter()
        .find(|e| e["body"]["name"] == "checkpoint.saved")
        .expect("checkpoint event");
    assert_eq!(ckpt["body"]["metadata"]["checkpoint_id"], "ckpt-7");
    assert_eq!(ckpt["body"]["metadata"]["step"], 1);
}
