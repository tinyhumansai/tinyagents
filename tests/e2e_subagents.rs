//! TRUE end-to-end: agent-calling-agent composition (sub-agents).
//!
//! A parent [`AgentHarness`] whose scripted [`MockModel`] calls a
//! [`SubAgentTool`] drives a child [`AgentHarness`] (also a `MockModel`) and
//! composes the child's answer into a final assistant reply. The parent run's
//! [`EventSink`] is wired to a testkit [`EventRecorder`] so we can reconstruct
//! a [`Trajectory`] and assert *structurally* that the sub-agent really ran —
//! never on model prose.
//!
//! A second test exercises the deterministic recursion-depth guard: nesting a
//! sub-agent past the harness's `max_depth` fails fast with
//! [`RustAgentsError::SubAgentDepth`] *before* any model call, both through the
//! direct invoke path and through the tool path.

use std::sync::Arc;

use serde_json::json;

use rustagents::error::RustAgentsError;
use rustagents::harness::context::{RunConfig, RunContext};
use rustagents::harness::limits::RunLimits;
use rustagents::harness::message::{AssistantMessage, ContentBlock, Message};
use rustagents::harness::model::ModelResponse;
use rustagents::harness::providers::MockModel;
use rustagents::harness::runtime::{AgentHarness, RunPolicy};
use rustagents::harness::testkit::{EventRecorder, Trajectory};
use rustagents::harness::tool::{Tool, ToolCall};
use rustagents::harness::usage::Usage;
use rustagents::{SubAgent, SubAgentTool};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// A tool-call assistant turn: no text, a single tool call.
fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(9, 4)),
        },
        usage: Some(Usage::new(9, 4)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// A plain-text assistant turn.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(5, 3)),
        },
        usage: Some(Usage::new(5, 3)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// A child harness whose model always answers with `answer`.
fn child_harness(answer: &str) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("child-model", Arc::new(MockModel::constant(answer)));
    harness
}

/// A child harness capped at `max_depth`.
fn child_harness_with_max_depth(answer: &str, max_depth: usize) -> AgentHarness<()> {
    let mut harness = child_harness(answer);
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_depth(max_depth),
        ..RunPolicy::default()
    });
    harness
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn parent_drives_subagent_and_composes_answer() {
    // Child agent that "researches" and returns a fixed answer.
    let child = Arc::new(SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("RUST_IS_A_SYSTEMS_LANGUAGE")),
    ));
    let tool = Arc::new(SubAgentTool::new(child));

    // Parent: first turn delegates to the sub-agent tool, second turn composes
    // the final answer.
    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool(tool);
    parent.register_model(
        "parent-model",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "researcher", json!({ "input": "what is rust?" })),
            text_response("Based on the researcher: a systems language."),
        ])),
    );

    // Wire the parent run's events into a shared recorder so we can assert on
    // the trajectory (the sub-agent tool really ran) afterward.
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("parent-run"), ()).with_events(recorder.sink());

    let run = parent
        .invoke_in_context(&(), ctx, vec![Message::user("delegate this")])
        .await
        .expect("parent run succeeds");

    // Behavior / structure assertions — never on the exact prose.
    assert_eq!(run.tool_calls, 1, "parent invoked the sub-agent tool once");
    assert_eq!(run.model_calls, 2, "parent made two model calls");
    assert_eq!(
        run.text(),
        Some("Based on the researcher: a systems language.".to_string())
    );

    // The child's answer was woven into the parent transcript as a tool message.
    let child_answer_present = run
        .messages
        .iter()
        .any(|m| matches!(m, Message::Tool(_)) && m.text() == "RUST_IS_A_SYSTEMS_LANGUAGE");
    assert!(
        child_answer_present,
        "the sub-agent's answer should appear as a tool result in the parent transcript"
    );

    // Trajectory assertion: the sub-agent (exposed as the `researcher` tool)
    // really ran, and the run completed cleanly.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_tool_called("researcher");
    assert_eq!(traj.tool_call_count("researcher"), 1);
    traj.assert_model_called_times(2);
    traj.assert_completed();
    traj.assert_order(&["run.started", "researcher", "run.completed"])
        .expect("sub-agent tool runs between run start and completion");
}

#[tokio::test]
async fn nesting_past_max_depth_is_a_deterministic_error() {
    // Cap the child harness at depth 1: a child run is allowed at depth 1
    // (parent_depth 0) but not at depth 2 (parent_depth 1).
    let subagent = Arc::new(SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    ));

    // Within the cap: parent_depth 0 -> child depth 1.
    let ok_run = subagent
        .invoke(&(), (), 0, "ok")
        .await
        .expect("child run at depth 1 is within the cap");
    assert_eq!(ok_run.text(), Some("ok".to_string()));

    // Direct invoke past the cap: parent_depth 1 -> child depth 2 > cap of 1.
    let err = subagent
        .invoke(&(), (), 1, "too deep")
        .await
        .expect_err("child depth 2 exceeds the cap");
    assert!(
        matches!(err, RustAgentsError::SubAgentDepth(1)),
        "expected SubAgentDepth(1), got {err:?}"
    );

    // Tool path past the cap: constructing the tool at parent_depth 1 makes the
    // child run at depth 2, which also exceeds the cap deterministically.
    let tool = SubAgentTool::new(subagent).with_parent_depth(1);
    let tool_err = tool
        .call(&(), ToolCall::new("c1", "deep", json!({ "input": "x" })))
        .await
        .expect_err("the tool surfaces the same depth error");
    assert!(
        matches!(tool_err, RustAgentsError::SubAgentDepth(1)),
        "expected SubAgentDepth(1) from the tool path, got {tool_err:?}"
    );
}
