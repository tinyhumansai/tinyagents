//! TRUE end-to-end: a legacy [`StateGraph`] node that drives a full
//! [`AgentHarness`] agent run.
//!
//! This composes three subsystems that the per-module integration tests
//! exercise in isolation:
//!
//! - the **graph** executor (`StateGraph` / `Node` / `NodeOutput`),
//! - the **harness** agent loop (model → tool → model), and
//! - the **testkit** (`ScriptedModel`, `FakeTool`, `EventRecorder`,
//!   `Trajectory`) plus a real **middleware** (`UsageAccountingMiddleware`).
//!
//! A single graph node owns an `Arc<AgentHarness>` and invokes it inside a
//! caller-supplied `RunContext` whose `EventSink` is wired to a shared
//! `EventRecorder`. After the graph reaches its terminal node we assert on the
//! accumulated state, the recorded event trajectory (the tool really ran), and
//! the usage totals folded by the middleware — never on model prose.

use std::sync::Arc;

use serde_json::json;

use rustagents::harness::context::{RunConfig, RunContext};
use rustagents::harness::message::{AssistantMessage, ContentBlock, Message};
use rustagents::harness::middleware::UsageAccountingMiddleware;
use rustagents::harness::model::ModelResponse;
use rustagents::harness::runtime::AgentHarness;
use rustagents::harness::testkit::{EventRecorder, FakeTool, ScriptedModel, Trajectory};
use rustagents::harness::tool::ToolCall;
use rustagents::harness::usage::Usage;
use rustagents::{Node, NodeOutput};

/// State threaded through the graph: the question to answer and the agent's
/// final text once the `agent` node has run.
#[derive(Clone, Debug, Default)]
struct AgentGraphState {
    question: String,
    answer: Option<String>,
}

/// Builds an assistant response that asks for a single tool call.
fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(6, 2)),
        },
        usage: Some(Usage::new(6, 2)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// Builds a plain-text final assistant response.
fn text_response(text: &str, input: u64, output: u64) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(input, output)),
        },
        usage: Some(Usage::new(input, output)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
    }
}

#[tokio::test]
async fn graph_node_drives_harness_agent_with_tool_loop() {
    // Shared recorder: the node feeds the agent run's events here so the test
    // can reconstruct a Trajectory after the graph completes.
    let recorder = Arc::new(EventRecorder::new());

    // Usage accounting middleware folds every model response's usage into a
    // running total we read back after the run.
    let usage_mw = Arc::new(UsageAccountingMiddleware::new());

    // Build the harness: script a tool call then a final answer, register a
    // real tool, and attach the usage middleware.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "scripted",
            Arc::new(ScriptedModel::new(vec![
                tool_call_response("call-1", "lookup_user", json!({ "user_id": "u-1" })),
                text_response("resolved via lookup", 4, 3),
            ])),
        )
        .set_default_model("scripted")
        .register_tool(Arc::new(FakeTool::returning("lookup_user", "Ada Lovelace")))
        .push_middleware(usage_mw.clone());
    let harness = Arc::new(harness);

    // A single graph node that runs the agent and folds the answer into state.
    let node_harness = harness.clone();
    let node_recorder = recorder.clone();
    let agent_node = Node::new("agent", move |mut state: AgentGraphState| {
        let harness = node_harness.clone();
        let recorder = node_recorder.clone();
        async move {
            // Wire the agent run's events into the shared recorder via a
            // caller-supplied RunContext.
            let ctx: RunContext<()> =
                RunContext::new(RunConfig::new("agent-node-run"), ()).with_events(recorder.sink());

            let run = harness
                .invoke_in_context(&(), ctx, vec![Message::user(state.question.clone())])
                .await?;

            state.answer = run.text();
            Ok(NodeOutput::end(state))
        }
    });

    let graph = rustagents::StateGraph::new()
        .add_node(agent_node)
        .set_start("agent")
        .add_end("agent");

    let run = graph
        .run(AgentGraphState {
            question: "who is user u-1?".to_string(),
            answer: None,
        })
        .await
        .expect("graph runs to completion");

    // Graph-level: the terminal node ran and the agent's answer landed in state.
    assert_eq!(run.visited, vec!["agent"]);
    assert_eq!(run.state.answer.as_deref(), Some("resolved via lookup"));

    // Harness-level via the trajectory: the tool actually fired, between two
    // model calls, and the run completed.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_tool_called("lookup_user");
    assert_eq!(traj.tool_call_count("lookup_user"), 1);
    traj.assert_model_called_times(2);
    traj.assert_completed();
    traj.assert_order(&["model.started", "lookup_user", "model.completed"])
        .expect("tool runs between the two model calls");

    // Middleware-level: usage was folded across both model calls.
    let totals = usage_mw.totals();
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.input_tokens, 6 + 4);
    assert_eq!(totals.usage.output_tokens, 2 + 3);
    assert_eq!(totals.usage.total_tokens, 15);
}
