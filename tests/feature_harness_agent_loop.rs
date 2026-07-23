//! Feature coverage for under-tested agent-loop behaviors.
//!
//! The existing `harness_agent_loop.rs` covers single-turn, a single-tool
//! round-trip, the `max_model_calls` cap, and usage accumulation. This file
//! adds distinct scenarios exercised entirely offline through the public
//! [`AgentHarness`] API with deterministic testkit doubles:
//!
//! - a single assistant turn requesting **multiple tool calls** dispatched to
//!   different tools, with every result threaded back into the transcript;
//! - the **`max_tool_calls`** cap (the tool-side sibling of `max_model_calls`);
//! - the lifecycle **event ordering** for a tool round-trip; and
//! - [`AgentHarness::invoke_with_status`] reporting a completed run.
//!
//! Behavioral assertions use the testkit [`Trajectory`] / [`EventRecorder`] and
//! the [`AgentRun`] counters — never the exact model prose.

use std::sync::Arc;

use serde_json::json;

use tinyagents::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::ids::ExecutionStatus;
use tinyagents::harness::limits::RunLimits;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
use tinyagents::harness::testkit::{EventRecorder, FakeTool, Trajectory};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// An assistant turn requesting several tool calls at once.
fn multi_tool_call_response(calls: Vec<ToolCall>) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some("msg-multi".into()),
            content: Vec::new(),
            tool_calls: calls,
            usage: Some(Usage::new(8, 3)),
        },
        usage: Some(Usage::new(8, 3)),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    multi_tool_call_response(vec![ToolCall::new(id, name, arguments)])
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, 2)),
        },
        usage: Some(Usage::new(4, 2)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// One assistant turn requests three tool calls across two distinct tools; the
/// loop executes all of them and threads each result into the transcript before
/// the final text turn.
#[tokio::test]
async fn parallel_tool_calls_in_one_turn_all_execute() {
    let recorder = EventRecorder::new();

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                multi_tool_call_response(vec![
                    ToolCall::new("c1", "alpha", json!({ "n": 1 })),
                    ToolCall::new("c2", "beta", json!({ "n": 2 })),
                    ToolCall::new("c3", "alpha", json!({ "n": 3 })),
                ]),
                text_response("all done"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("alpha", "A")))
        .register_tool(Arc::new(FakeTool::returning("beta", "B")));

    let ctx = RunContext::new(RunConfig::new("parallel-tools"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    // All three calls ran; two model turns total.
    assert_eq!(run.tool_calls, 3, "every requested tool call executed");
    assert_eq!(run.model_calls, 2);
    assert_eq!(run.text(), Some("all done".to_string()));

    // Each call's result landed in the transcript, keyed to its call id.
    let tool_texts: Vec<String> = run
        .messages
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .map(|m| m.text())
        .collect();
    assert_eq!(tool_texts, vec!["A", "B", "A"], "results in original order");

    // The trajectory records three tool starts (two on `alpha`, one on `beta`).
    let traj = Trajectory::from_events(recorder.events());
    assert_eq!(traj.tool_call_count("alpha"), 2);
    assert_eq!(traj.tool_call_count("beta"), 1);
    traj.assert_completed();
}

/// The `max_tool_calls` cap fails the run closed once the tool-call budget is
/// exhausted — the tool-side counterpart to the `max_model_calls` cap.
#[tokio::test]
async fn max_tool_calls_limit_returns_limit_exceeded() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    // A model that always requests a tool would loop forever without a cap.
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_tool_call("spin", json!({}))),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("spin", "again")))
        .with_policy(RunPolicy {
            // Generous model budget so the *tool* cap is what trips.
            limits: RunLimits::default()
                .with_max_tool_calls(2)
                .with_max_model_calls(50),
            ..RunPolicy::default()
        });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("the tool-call cap should be exceeded");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

/// A tool round-trip emits the lifecycle events in the documented order:
/// run start, first model completion, the tool bracket, the second model
/// completion, run completion.
#[tokio::test]
async fn tool_round_trip_emits_lifecycle_events_in_order() {
    let recorder = EventRecorder::new();

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("c1", "lookup", json!({})),
                text_response("final"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "out")));

    let ctx = RunContext::new(RunConfig::new("ordering"), ()).with_events(recorder.sink());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let kinds = recorder.kinds();
    // Run bracket present.
    assert_eq!(kinds.first().map(String::as_str), Some("run.started"));
    assert_eq!(kinds.last().map(String::as_str), Some("run.completed"));

    // The tool started and completed strictly between the two model completions.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_order(&[
        "run.started",
        "model.completed",
        "tool.started",
        "tool.completed",
        "model.completed",
        "run.completed",
    ])
    .expect("lifecycle events occur in the documented order");
}

/// `invoke_with_status` returns the [`AgentRun`] alongside a status snapshot
/// that reports the run completed and mirrors the run's tool-call count.
#[tokio::test]
async fn invoke_with_status_reports_completed_run() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("c1", "lookup", json!({})),
                text_response("done"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "out")));

    let result = harness
        .invoke_with_status(
            &(),
            (),
            RunConfig::new("with-status"),
            vec![Message::user("go")],
        )
        .await
        .expect("run succeeds");

    assert_eq!(result.run.tool_calls, 1);
    assert_eq!(result.run.model_calls, 2);
    // The status snapshot mirrors the completed run.
    assert_eq!(result.status.tool_calls, result.run.tool_calls);
    assert_eq!(
        result.status.status,
        ExecutionStatus::Completed,
        "a completed run reports Completed status: {:?}",
        result.status
    );
}

/// A distinct pair of tools is routed by name: the model asks for `beta` only,
/// so `alpha` is never invoked even though both are registered.
#[tokio::test]
async fn tools_are_dispatched_by_name() {
    let alpha = Arc::new(FakeTool::returning("alpha", "A"));
    let beta = Arc::new(FakeTool::returning("beta", "B"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("c1", "beta", json!({})),
                text_response("ok"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(alpha.clone())
        .register_tool(beta.clone());

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert!(alpha.calls().is_empty(), "the unrequested tool never ran");
    assert_eq!(beta.calls().len(), 1, "only the named tool ran");
}
