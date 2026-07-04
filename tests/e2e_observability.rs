//! End-to-end coverage for the durable observability layer (offline).
//!
//! Two flows are exercised without any network or live provider:
//!
//! 1. A harness run whose [`RunContext::events`] sink fans every typed
//!    [`AgentEvent`] into a [`RedactingSink`] that wraps a [`JournalSink`]. The
//!    test asserts that the [`InMemoryEventJournal`] replays the run from offset
//!    `0`, that a configured secret is masked before it reaches the journal, and
//!    that the run's completion is observable both through a wired
//!    [`HarnessStatusStore`] and through the journaled events.
//!
//! 2. A graph run with a [`JournalGraphSink`] (wired through
//!    [`CompiledGraph::with_event_journal`]). The test asserts the
//!    [`GraphObservation`]s are replayable from offset `0` and carry their
//!    run/step coordinates.
//!
//! All assertions are structural (event kinds, offsets, ids, redaction) — never
//! model prose — so the test is deterministic.

use std::sync::Arc;

use serde_json::json;

use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::{AgentEvent, RecordingListener};
use tinyagents::harness::ids::{ExecutionStatus, RunId};
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, PayloadCapture, RunPolicy};
use tinyagents::harness::testkit::FakeTool;
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;
use tinyagents::{
    GraphBuilder, GraphEvent, GraphEventJournal, GraphLangfuseExporter, HarnessEventJournal,
    HarnessStatusStore, InMemoryEventJournal, InMemoryGraphEventJournal, InMemoryStatusStore,
    JournalSink, LangfuseClient, LangfuseTraceConfig, NodeContext, NodeResult, RedactingSink,
};

// The secret is embedded in the *registered model name* so it appears verbatim
// in the `model` field of every `AgentEvent::ModelStarted` the loop emits.
const MODEL_NAME: &str = "gpt-sk-SEKRET-key";
const SECRET: &str = "sk-SEKRET";

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, 2)),
        },
        usage: Some(Usage::new(4, 2)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
    }
}

#[tokio::test]
async fn harness_run_journals_redacted_replayable_events() {
    // A multi-step run (tool call then final answer) so the journal carries a
    // varied lifecycle: run/model/tool boundaries plus completion.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            MODEL_NAME,
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "lookup", json!({ "q": "x" })),
                text_response("done"),
            ])),
        )
        .set_default_model(MODEL_NAME)
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));

    // Durable journal + a plain recorder (live broadcast) to cross-check.
    let journal: Arc<InMemoryEventJournal> = Arc::new(InMemoryEventJournal::new());
    let recorder = Arc::new(RecordingListener::new());

    let run_id = RunId::new("run-obs");
    // RedactingSink wraps the JournalSink: every event is masked *before* it is
    // persisted, so the durable journal never sees the secret.
    let journal_sink = JournalSink::new(journal.clone(), run_id.clone());
    // Persistence is asynchronous; keep a handle to flush the shared drain
    // before reading the journal back. Clones share the same background worker.
    let journal_flush = journal_sink.clone();
    let redacting = RedactingSink::new(Arc::new(journal_sink), vec![SECRET.to_string()]);

    // Attach the sinks through RunContext.events, then drive the run in-context.
    let ctx: RunContext<()> = RunContext::new(RunConfig::new(run_id.as_str()), ());
    ctx.events.subscribe(Arc::new(redacting));
    ctx.events.subscribe(recorder.clone());

    let result = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("please look up")])
        .await
        .expect("run succeeds");

    assert_eq!(result.run.model_calls, 2);
    assert_eq!(result.run.tool_calls, 1);

    // --- Status surface: persist the returned status into a HarnessStatusStore
    // and confirm it reflects completion. ---
    let status_store = InMemoryStatusStore::new();
    status_store
        .put_status(result.status.clone())
        .await
        .expect("put status");
    let stored = status_store
        .get_status(run_id.as_str())
        .await
        .expect("get status")
        .expect("status present");
    assert_eq!(stored.status, ExecutionStatus::Completed);
    assert!(stored.ended_at.is_some());
    assert!(stored.error.is_none());

    // --- Journal replays the whole run from offset 0. ---
    journal_flush.flush();
    let all = journal
        .read_from(run_id.as_str(), 0)
        .await
        .expect("replay journal");
    assert!(
        all.len() >= 4,
        "expected run/model/tool/completion observations, got {}",
        all.len()
    );

    // Offsets are dense and monotonic from 0; lineage is stamped as top-level.
    for (i, obs) in all.iter().enumerate() {
        assert_eq!(obs.offset, i as u64, "dense offsets from 0");
        assert_eq!(obs.run_id.as_str(), run_id.as_str());
        assert_eq!(obs.root_run_id.as_str(), run_id.as_str());
        assert!(obs.parent_run_id.is_none(), "top-level run has no parent");
    }

    // The run lifecycle bookends the journaled stream.
    assert!(
        matches!(all.first().unwrap().event, AgentEvent::RunStarted { .. }),
        "first journaled event is RunStarted"
    );
    assert!(
        all.iter()
            .any(|o| matches!(o.event, AgentEvent::RunCompleted { .. })),
        "completion is observable from the journal"
    );

    // --- Redaction: the secret never reaches the journal, but a masked model
    // name does. The model field of ModelStarted carried the secret live. ---
    let journal_json = serde_json::to_string(&all).expect("serialize observations");
    assert!(
        !journal_json.contains(SECRET),
        "the secret must be masked before it is journaled"
    );
    assert!(
        journal_json.contains(RedactingSink::DEFAULT_MASK),
        "the redaction mask should appear where the secret was"
    );
    // Sanity: the *live* (un-redacted) recorder still saw the raw secret, proving
    // redaction happens at the sink boundary and not at emit time.
    let live_json = serde_json::to_string(&recorder.events()).expect("serialize live records");
    assert!(
        live_json.contains(SECRET),
        "the live broadcast carries full detail; redaction is sink-local"
    );
    // Replaying from a mid-stream offset returns exactly the tail.
    let tail = journal
        .read_from(run_id.as_str(), 2)
        .await
        .expect("replay tail");
    assert_eq!(tail.len(), all.len() - 2);
    assert_eq!(tail.first().unwrap().offset, 2);
}

#[tokio::test]
async fn harness_run_with_capture_exports_generation_and_tool_io() {
    // With PayloadCapture enabled, a real harness run journals model/tool
    // payloads on its completion events; feeding those durable observations
    // through the Langfuse exporter populates the generation Input/Output and
    // the tool-create Input/Output panels — the fix for tinyhumansai/tinyagents#6.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "capture-model",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "lookup", json!({ "q": "weather" })),
                text_response("all done"),
            ])),
        )
        .set_default_model("capture-model")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .with_policy(RunPolicy {
            capture: PayloadCapture::all(),
            ..Default::default()
        });

    let journal: Arc<InMemoryEventJournal> = Arc::new(InMemoryEventJournal::new());
    let run_id = RunId::new("run-capture");
    let ctx: RunContext<()> = RunContext::new(RunConfig::new(run_id.as_str()), ());
    let journal_sink = JournalSink::new(journal.clone(), run_id.clone());
    let journal_flush = journal_sink.clone();
    ctx.events.subscribe(Arc::new(journal_sink));

    harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("please look up")])
        .await
        .expect("run succeeds");
    // Persistence is asynchronous; block until the durable log catches up.
    journal_flush.flush();

    let observations = journal
        .read_from(run_id.as_str(), 0)
        .await
        .expect("replay journal");

    // The captured payloads survive on the durable observations.
    let model_completed = observations
        .iter()
        .find(|o| matches!(o.event, AgentEvent::ModelCompleted { .. }))
        .expect("a model completion is journaled");
    match &model_completed.event {
        AgentEvent::ModelCompleted { input, output, .. } => {
            assert!(input.is_some(), "captured model input rides the event");
            assert!(output.is_some(), "captured model output rides the event");
        }
        _ => unreachable!(),
    }

    // Export through the harness Langfuse client and assert the body carries I/O.
    let client = LangfuseClient::proxy("https://backend.test", "tok").expect("client");
    let batch = client
        .build_ingestion_batch(LangfuseTraceConfig::default(), &observations)
        .expect("build batch");
    let events = batch["batch"].as_array().expect("batch array");

    // Trace metadata is defaulted from the run lineage even with no caller value.
    assert_eq!(events[0]["type"], "trace-create");
    assert_eq!(events[0]["body"]["metadata"]["root_run_id"], "run-capture");

    let generations: Vec<_> = events
        .iter()
        .filter(|e| e["type"] == "generation-create")
        .collect();
    assert_eq!(generations.len(), 2, "one generation per model call");
    assert!(
        generations
            .iter()
            .all(|g| !g["body"]["input"].is_null() && !g["body"]["output"].is_null()),
        "every generation Input/Output panel is populated"
    );
    // The final generation's completion carries the answer text.
    assert_eq!(
        generations[1]["body"]["output"]["content"][0]["text"],
        "all done"
    );

    // Tool observations are exported as `span-create` (a valid Langfuse
    // ingestion type); `tool-create` is rejected by older/self-hosted Langfuse.
    let tool = events
        .iter()
        .find(|e| e["type"] == "span-create")
        .expect("a tool observation");
    assert_eq!(tool["body"]["input"]["q"], "weather");
    assert_eq!(tool["body"]["output"], "tool-output");
}

/// A two-node line graph over `i32` with overwrite semantics: `a -> b`.
fn line_graph() -> tinyagents::CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .expect("graph compiles")
}

#[tokio::test]
async fn graph_run_journals_replayable_observations_with_coords() {
    let journal = Arc::new(InMemoryGraphEventJournal::new());
    // `with_event_journal` wires a JournalGraphSink as the run's event sink.
    let graph = line_graph().with_event_journal(journal.clone());

    let run = graph.run(0).await.expect("graph run succeeds");
    let run_id = run.status.run_id.as_str().to_string();

    let obs = journal
        .read_from(&run_id, 0)
        .await
        .expect("replay graph journal");
    assert!(
        obs.len() >= 3,
        "expected several graph observations, got {}",
        obs.len()
    );

    // Dense, monotonic offsets from 0; run/graph coordinates are stamped.
    for (i, o) in obs.iter().enumerate() {
        assert_eq!(o.offset, i as u64, "dense offsets from 0");
        assert_eq!(o.run_id.as_str(), run_id);
        assert_eq!(o.root_run_id.as_str(), run_id);
        assert_eq!(&o.graph_id, graph.graph_id());
    }

    // Step coordinates are carried: the executor runs at least one superstep.
    assert!(
        obs.iter().any(|o| o.step >= 1),
        "observations should carry superstep coordinates"
    );

    // The run lifecycle bookends the stream.
    assert!(
        matches!(obs.first().unwrap().event, GraphEvent::RunStarted { .. }),
        "first observation is RunStarted"
    );
    assert!(
        obs.iter()
            .any(|o| matches!(o.event, GraphEvent::RunCompleted { .. })),
        "completion is observable from the graph journal"
    );

    // Replay from a mid-stream offset returns exactly the tail.
    let tail = journal.read_from(&run_id, 2).await.expect("replay tail");
    assert_eq!(tail.len(), obs.len() - 2);
    assert_eq!(tail.first().unwrap().offset, 2);

    // Reading an unknown run is empty, not an error.
    assert!(
        journal.read_from("nope", 0).await.unwrap().is_empty(),
        "unknown run replays empty"
    );
}

#[tokio::test]
async fn graph_observations_export_to_langfuse_trace_and_spans() {
    // Run a real graph, journal it, then feed the durable observations through
    // the graph Langfuse exporter and assert the offline batch structure.
    let journal = Arc::new(InMemoryGraphEventJournal::new());
    let graph = line_graph().with_event_journal(journal.clone());

    let run = graph.run(0).await.expect("graph run succeeds");
    let run_id = run.status.run_id.as_str().to_string();
    let observations = journal
        .read_from(&run_id, 0)
        .await
        .expect("replay graph journal");

    // The exporter reuses the harness Langfuse transport (proxy mode here).
    let exporter = GraphLangfuseExporter::new(
        LangfuseClient::proxy("https://backend.test", "tok").expect("client"),
    );
    let batch = exporter
        .build_ingestion_batch(LangfuseTraceConfig::default(), &observations)
        .expect("build batch");
    let events = batch["batch"].as_array().expect("batch array");

    // The trace is created first and its id defaults to the run's root run id,
    // aligning with the harness agent exporter for a unified trace.
    assert_eq!(events[0]["type"], "trace-create");
    assert_eq!(events[0]["body"]["id"], run_id);

    // Node health telemetry rides on the trace: two nodes ran and completed.
    let health = &events[0]["body"]["metadata"]["health"];
    assert_eq!(health["total_completed"], 2);
    assert_eq!(health["total_failed"], 0);
    assert_eq!(health["run_failed"], false);

    // Each node handler becomes a span, parented to its superstep span.
    let node_a = format!("{run_id}:node:a:1");
    let node_b = format!("{run_id}:node:b:2");
    let has_node_span = |id: &str| {
        events
            .iter()
            .any(|e| e["type"] == "span-create" && e["body"]["id"] == id)
    };
    assert!(has_node_span(&node_a), "node a span present");
    assert!(has_node_span(&node_b), "node b span present");

    // No node failed, so no span carries an ERROR level.
    assert!(
        !events
            .iter()
            .any(|e| e["type"] == "span-create" && e["body"]["level"] == "ERROR"),
        "healthy run has no ERROR spans"
    );
}
