//! End-to-end coverage for [`UnknownToolPolicy`] recovery behavior.
//!
//! Each test drives a real [`AgentHarness`] with a scripted [`MockModel`] that
//! calls an unregistered tool, then asserts how the configured
//! [`UnknownToolPolicy`] steers the run: hard failure, recoverable tool-error
//! injection, rewrite-to-a-real-tool, rewrite fallback, and bounded recovery
//! under the tool-call limit. Where events matter, an [`EventRecorder`] is
//! attached through a [`RunContext`] so the emitted
//! [`AgentEvent::UnknownToolCall`] can be inspected.

use std::sync::Arc;

use serde_json::json;

use tinyagents::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::AgentEvent;
use tinyagents::harness::limits::RunLimits;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy, UnknownToolPolicy};
use tinyagents::harness::testkit::{EventRecorder, FakeTool};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;

// ── Scripted response helpers ─────────────────────────────────────────────────

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(6, 2)),
        },
        usage: Some(Usage::new(6, 2)),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(3, 1)),
        },
        usage: Some(Usage::new(3, 1)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

/// Finds the single recorded [`AgentEvent::UnknownToolCall`], asserting the
/// `kind()` label and returning the requested name and recovery string.
fn single_unknown_tool_event(events: &[AgentEvent]) -> (String, String) {
    let mut found: Option<(String, String)> = None;
    for event in events {
        if let AgentEvent::UnknownToolCall {
            requested_name,
            recovery,
            ..
        } = event
        {
            assert_eq!(event.kind(), "tool.unknown");
            assert!(
                found.is_none(),
                "expected exactly one UnknownToolCall event, got a second"
            );
            found = Some((requested_name.clone(), recovery.clone()));
        }
    }
    found.expect("an UnknownToolCall event should have been recorded")
}

// ── 1. Fail policy (default) ──────────────────────────────────────────────────

#[tokio::test]
async fn fail_policy_errors_on_unregistered_tool() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("missing", json!({}))),
    );
    // Default policy is UnknownToolPolicy::Fail; no tool registered.

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("Fail policy must abort on an unregistered tool");

    match err {
        TinyAgentsError::ToolNotFound(name) => assert_eq!(name, "missing"),
        other => panic!("expected ToolNotFound(\"missing\"), got {other:?}"),
    }
}

// ── 2. ReturnToolError recovers and emits an event ────────────────────────────

#[tokio::test]
async fn return_tool_error_recovers_and_emits_event() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "missing", json!({})),
            text_response("recovered"),
        ])),
    );
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::ReturnToolError,
        ..RunPolicy::default()
    });

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("return-tool-error"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("ReturnToolError is recoverable");

    assert_eq!(run.final_response.unwrap().text(), "recovered");

    // The injected tool-error message names the requested tool for repair.
    let injected = run
        .messages
        .iter()
        .any(|m| m.text().contains("unknown tool `missing`"));
    assert!(
        injected,
        "a tool-error message naming `missing` should be in the transcript"
    );

    let (requested_name, recovery) = single_unknown_tool_event(&recorder.events());
    assert_eq!(requested_name, "missing");
    assert_eq!(recovery, "tool_error");
}

#[tokio::test]
async fn return_tool_error_preserves_original_arguments() {
    let original = json!({ "query": "weather", "n": 3 });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "search", original.clone()),
            text_response("recovered"),
        ])),
    );
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::ReturnToolError,
        ..RunPolicy::default()
    });

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("preserve-args"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("ReturnToolError is recoverable");

    // The event carries the original arguments verbatim.
    let args = recorder.events().into_iter().find_map(|e| match e {
        AgentEvent::UnknownToolCall { arguments, .. } => Some(arguments),
        _ => None,
    });
    assert_eq!(args, Some(original));

    // The injected repair message echoes the arguments for the model.
    assert!(
        run.messages
            .iter()
            .any(|m| m.text().contains("\"query\":\"weather\"")),
        "the injected message should echo the original arguments"
    );
}

// ── 3. Rewrite retargets to a real, registered tool ───────────────────────────

#[tokio::test]
async fn rewrite_retargets_to_real_tool() {
    let fake = Arc::new(FakeTool::returning("lookup", "out"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "missing", json!({})),
            text_response("done"),
        ])),
    );
    harness.register_tool(fake.clone());
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::Rewrite {
            tool_name: "lookup".into(),
        },
        ..RunPolicy::default()
    });

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("rewrite"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("rewrite to a registered tool recovers");

    assert_eq!(run.final_response.unwrap().text(), "done");
    // The rewritten call actually executed the real lookup tool exactly once.
    assert_eq!(fake.calls().len(), 1);

    let (requested_name, recovery) = single_unknown_tool_event(&recorder.events());
    assert_eq!(requested_name, "missing");
    assert_eq!(recovery, "rewrite:lookup");
}

// ── 4. Rewrite to a missing target falls back to ReturnToolError ──────────────

#[tokio::test]
async fn rewrite_to_missing_target_falls_back_to_tool_error() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "missing", json!({})),
            text_response("recovered"),
        ])),
    );
    // Rewrite target "nope" is itself unregistered → fall back to tool-error.
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::Rewrite {
            tool_name: "nope".into(),
        },
        ..RunPolicy::default()
    });

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("rewrite-fallback"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("rewrite fallback still recovers");

    assert_eq!(run.final_response.unwrap().text(), "recovered");

    let injected = run
        .messages
        .iter()
        .any(|m| m.text().contains("unknown tool `missing`"));
    assert!(
        injected,
        "fallback should inject a tool-error naming the original `missing` tool"
    );

    let (requested_name, recovery) = single_unknown_tool_event(&recorder.events());
    assert_eq!(requested_name, "missing");
    assert_eq!(
        recovery, "tool_error",
        "an unregistered rewrite target falls back to tool_error recovery"
    );
}

// ── 5. Recovery is bounded by the tool-call limit ─────────────────────────────

#[tokio::test]
async fn recovery_is_bounded_by_tool_call_limit() {
    // MockModel::with_tool_call repeats the same unknown call on every
    // invocation, so without a bound the loop would spin forever.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("missing", json!({}))),
    );
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::ReturnToolError,
        limits: RunLimits::default().with_max_tool_calls(3),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("an always-unknown model must terminate via the limit");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );
}
