//! Tests for first-class sub-agents and recursion-depth tracking.
//!
//! These exercise:
//! - a parent harness whose scripted [`MockModel`] calls a [`SubAgentTool`],
//!   driving a child harness and composing the child's answer,
//! - direct [`SubAgent::invoke`] returning the child [`AgentRun`] at depth 1,
//! - the depth guard producing [`TinyAgentsError::SubAgentDepth`] when nested
//!   too deep (both via direct invoke and via the tool path),
//! - sub-agent lifecycle events emitted onto a shared sink.

use std::sync::Arc;

use serde_json::json;

use crate::error::TinyAgentsError;
use crate::harness::events::{AgentEvent, EventSink, RecordingListener};
use crate::harness::limits::RunLimits;
use crate::harness::message::{AssistantMessage, ContentBlock, Message};
use crate::harness::model::ModelResponse;
use crate::harness::providers::MockModel;
use crate::harness::runtime::{AgentHarness, RunPolicy};
use crate::harness::subagent::{SubAgent, SubAgentTool};
use crate::harness::tool::{Tool, ToolCall};
use crate::harness::usage::Usage;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Builds a tool-call assistant response (no text, one tool call).
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

/// Builds a plain-text assistant response.
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

/// Builds a child harness whose model always answers with `answer`.
fn child_harness(answer: &str) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("child-model", Arc::new(MockModel::constant(answer)));
    harness
}

/// Builds a child harness with a custom `max_depth` policy.
fn child_harness_with_max_depth(answer: &str, max_depth: usize) -> AgentHarness<()> {
    let mut harness = child_harness(answer);
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_depth(max_depth),
        ..RunPolicy::default()
    });
    harness
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn subagent_tool_drives_child_and_composes_answer() {
    let child = Arc::new(SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("the child answer")),
    ));
    let tool = Arc::new(SubAgentTool::new(child));

    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool(tool);
    parent.register_model(
        "parent-model",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "researcher", json!({ "input": "what is rust?" })),
            text_response("composed final answer"),
        ])),
    );

    let run = parent
        .invoke_default(&(), vec![Message::user("delegate this")])
        .await
        .expect("parent run succeeds");

    assert_eq!(run.tool_calls, 1);
    assert_eq!(run.model_calls, 2);
    assert_eq!(run.text(), Some("composed final answer".to_string()));

    // The child's answer is woven into the parent transcript as a tool message.
    let has_child_answer = run
        .messages
        .iter()
        .any(|m| matches!(m, Message::Tool(_)) && m.text() == "the child answer");
    assert!(
        has_child_answer,
        "child answer should appear as a tool result"
    );
}

#[tokio::test]
async fn direct_invoke_returns_child_run_at_depth_one() {
    let subagent = SubAgent::new(
        "helper",
        "a helper agent",
        Arc::new(child_harness("hi from child")),
    )
    .with_system_prompt("You are a helper.");

    let run = subagent
        .invoke(&(), (), 0, "hello")
        .await
        .expect("child run succeeds");

    assert_eq!(run.text(), Some("hi from child".to_string()));
    // system prompt + user input + assistant reply.
    assert_eq!(run.messages.len(), 3);
    assert!(matches!(run.messages[0], Message::System(_)));
}

#[tokio::test]
async fn invoke_at_max_depth_errors() {
    // Child harness caps depth at 1: a child run is allowed at depth 1
    // (parent_depth 0) but not at depth 2 (parent_depth 1).
    let subagent = SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    );

    // parent_depth 0 -> child depth 1 -> within the cap.
    subagent
        .invoke(&(), (), 0, "ok")
        .await
        .expect("depth 1 within cap");

    // parent_depth 1 -> child depth 2 -> exceeds the cap of 1.
    let err = subagent
        .invoke(&(), (), 1, "too deep")
        .await
        .expect_err("depth 2 exceeds cap");
    assert!(matches!(err, TinyAgentsError::SubAgentDepth(1)));
}

#[tokio::test]
async fn tool_path_enforces_depth_limit() {
    let subagent = Arc::new(SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    ));
    // Invoke the child at parent_depth 1 -> child depth 2 -> exceeds cap of 1.
    let tool = SubAgentTool::new(subagent).with_parent_depth(1);

    let err = tool
        .call(&(), ToolCall::new("c1", "deep", json!({ "input": "x" })))
        .await
        .expect_err("tool surfaces the depth error");
    assert!(matches!(err, TinyAgentsError::SubAgentDepth(1)));
}

#[tokio::test]
async fn invoke_with_events_emits_lifecycle_on_shared_sink() {
    let subagent = SubAgent::new(
        "observed",
        "an observed agent",
        Arc::new(child_harness("done")),
    );

    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    subagent
        .invoke_with_events(&(), (), 0, "go", &sink)
        .await
        .expect("child run succeeds");

    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();

    // Sub-agent lifecycle brackets the child run, and the child's own RunStarted
    // also lands on the shared sink.
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::SubAgentStarted { name, depth } if name == "observed" && *depth == 1
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::SubAgentCompleted { name, depth } if name == "observed" && *depth == 1
    )));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunStarted { .. }))
    );
}

#[test]
fn subagent_tool_schema_uses_name_and_description() {
    let subagent = Arc::new(SubAgent::new(
        "writer",
        "writes prose",
        Arc::new(child_harness("x")),
    ));
    let tool = SubAgentTool::new(subagent);
    let schema = tool.schema();
    assert_eq!(schema.name, "writer");
    assert_eq!(schema.description, "writes prose");
    assert_eq!(schema.parameters["required"][0], "input");
}
