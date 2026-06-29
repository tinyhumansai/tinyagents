//! Tests for orchestrator → sub-agent steering.
//!
//! Unit tests exercise [`apply_pending_steering`] directly; integration tests
//! drive a full [`AgentHarness`] run with a [`SteeringHandle`] attached to its
//! [`RunContext`] and assert both the transcript outcome and the observable
//! [`AgentEvent::Steered`] events via an [`EventRecorder`].

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::json;

use crate::error::{Result, RustAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::AgentEvent;
use crate::harness::message::Message;
use crate::harness::model::{ChatModel, ModelRequest, ModelResponse};
use crate::harness::providers::MockModel;
use crate::harness::runtime::AgentHarness;
use crate::harness::steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringOutcome, SteeringPolicy,
    apply_pending_steering,
};
use crate::harness::testkit::{EventRecorder, Trajectory};
use crate::harness::usage::Usage;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Builds a plain-text assistant response.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: crate::harness::message::AssistantMessage {
            id: None,
            content: vec![crate::harness::message::ContentBlock::Text(
                text.to_string(),
            )],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(1, 1)),
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// A model that records every request it receives. On its first call it pushes
/// a follow-up steering command into a shared [`SteeringHandle`] (simulating an
/// orchestrator reacting mid-run) and returns a tool call so the loop iterates
/// again; on later calls it returns a final text answer.
struct RecordingModel {
    requests: Mutex<Vec<ModelRequest>>,
    calls: Mutex<usize>,
    steer_on_first: Mutex<Option<SteeringCommand>>,
    handle: SteeringHandle,
}

#[async_trait]
impl ChatModel<()> for RecordingModel {
    async fn invoke(&self, _state: &(), request: ModelRequest) -> Result<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        let first = *calls == 1;
        drop(calls);

        if first {
            if let Some(cmd) = self.steer_on_first.lock().unwrap().take() {
                self.handle.send(cmd);
            }
            // Ask for a tool so the loop runs another model call after the
            // steering checkpoint drains the queued command.
            Ok(ModelResponse {
                message: crate::harness::message::AssistantMessage {
                    id: Some("m1".to_string()),
                    content: Vec::new(),
                    tool_calls: vec![crate::harness::tool::ToolCall::new("c1", "noop", json!({}))],
                    usage: Some(Usage::new(1, 1)),
                },
                usage: Some(Usage::new(1, 1)),
                finish_reason: Some("tool_calls".to_string()),
                raw: None,
                resolved_model: None,
            })
        } else {
            Ok(text_response("done"))
        }
    }
}

/// A no-op tool that lets the loop iterate.
struct NoopTool;

#[async_trait]
impl crate::harness::tool::Tool<()> for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "noop"
    }
    fn schema(&self) -> crate::harness::tool::ToolSchema {
        crate::harness::tool::ToolSchema::new("noop", "noop", json!({"type": "object"}))
    }
    async fn call(
        &self,
        _state: &(),
        call: crate::harness::tool::ToolCall,
    ) -> Result<crate::harness::tool::ToolResult> {
        Ok(crate::harness::tool::ToolResult::text(
            call.id, "noop", "ok",
        ))
    }
}

// ── Unit tests: apply_pending_steering ────────────────────────────────────────

#[test]
fn no_handle_is_continue_and_silent() {
    let recorder = EventRecorder::new();
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_events(recorder.sink());
    let mut messages = vec![Message::user("hi")];

    let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();

    assert_eq!(outcome, SteeringOutcome::Continue);
    assert_eq!(messages.len(), 1);
    assert!(recorder.events().is_empty());
}

#[test]
fn inject_message_appends_to_transcript_and_emits_event() {
    let recorder = EventRecorder::new();
    let handle =
        SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage));
    handle.send(SteeringCommand::InjectMessage(Message::user(
        "focus on billing",
    )));

    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ())
        .with_events(recorder.sink())
        .with_steering(handle);
    let mut messages = vec![Message::user("hi")];

    let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();

    assert_eq!(outcome, SteeringOutcome::Continue);
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].text(), "focus on billing");
    assert_eq!(
        recorder.events(),
        vec![AgentEvent::Steered {
            command_kind: "inject_message".to_string(),
            accepted: true,
        }]
    );
}

#[test]
fn redirect_appends_system_instruction() {
    let handle = SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::Redirect));
    handle.send(SteeringCommand::Redirect {
        instruction: "compare against policy v3".to_string(),
    });
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_steering(handle);
    let mut messages = vec![Message::user("hi")];

    apply_pending_steering(&mut ctx, &mut messages).unwrap();

    assert!(matches!(messages[1], Message::System(_)));
    assert_eq!(
        messages[1].text(),
        "[steering:redirect] compare against policy v3"
    );
}

#[test]
fn set_metadata_replaces_config_metadata() {
    let handle = SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::SetMetadata));
    handle.send(SteeringCommand::SetMetadata {
        metadata: json!({"reviewed": true}),
    });
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_steering(handle);
    let mut messages = Vec::new();

    apply_pending_steering(&mut ctx, &mut messages).unwrap();

    assert_eq!(ctx.config.metadata, json!({"reviewed": true}));
}

#[test]
fn pause_then_resume_in_same_batch_nets_to_continue() {
    let handle = SteeringHandle::new(SteeringPolicy::allow_all());
    handle.send(SteeringCommand::Pause);
    handle.send(SteeringCommand::Resume);
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_steering(handle);
    let mut messages = Vec::new();

    let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();
    assert_eq!(outcome, SteeringOutcome::Continue);
}

#[test]
fn pause_alone_nets_to_pause() {
    let handle = SteeringHandle::new(SteeringPolicy::allow_all());
    handle.send(SteeringCommand::Pause);
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_steering(handle);
    let mut messages = Vec::new();

    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Pause
    );
}

#[test]
fn cancel_wins_over_later_commands() {
    let handle = SteeringHandle::new(SteeringPolicy::allow_all());
    handle.send(SteeringCommand::Cancel);
    handle.send(SteeringCommand::InjectMessage(Message::user("ignored")));
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_steering(handle);
    let mut messages = Vec::new();

    let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();
    assert_eq!(outcome, SteeringOutcome::Cancel);
    // The injection after the cancel is never applied.
    assert!(messages.is_empty());
}

#[test]
fn disallowed_command_is_rejected_with_steering_error_and_event() {
    let recorder = EventRecorder::new();
    // Policy permits Pause but not Cancel.
    let handle = SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::Pause));
    handle.send(SteeringCommand::Cancel);
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ())
        .with_events(recorder.sink())
        .with_steering(handle);
    let mut messages = Vec::new();

    let err = apply_pending_steering(&mut ctx, &mut messages).unwrap_err();
    assert!(matches!(err, RustAgentsError::Steering(_)), "got {err:?}");
    assert_eq!(
        recorder.events(),
        vec![AgentEvent::Steered {
            command_kind: "cancel".to_string(),
            accepted: false,
        }]
    );
}

// ── Serde ─────────────────────────────────────────────────────────────────────

#[test]
fn steering_command_round_trips_through_json() {
    let cmd = SteeringCommand::Redirect {
        instruction: "x".to_string(),
    };
    let json = serde_json::to_string(&cmd).unwrap();
    let back: SteeringCommand = serde_json::from_str(&json).unwrap();
    assert_eq!(cmd, back);
}

// ── Integration: full agent-loop runs ──────────────────────────────────────────

#[tokio::test]
async fn orchestrator_injects_message_mid_run_next_model_call_sees_it() {
    let recorder = EventRecorder::new();
    let handle =
        SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage));

    let model = Arc::new(RecordingModel {
        requests: Mutex::new(Vec::new()),
        calls: Mutex::new(0),
        steer_on_first: Mutex::new(Some(SteeringCommand::InjectMessage(Message::user(
            "STEER: focus on billing",
        )))),
        handle: handle.clone(),
    });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.register_tool(Arc::new(NoopTool));

    let ctx: RunContext = RunContext::new(RunConfig::new("run-inject"), ())
        .with_events(recorder.sink())
        .with_steering(handle);

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("start")])
        .await
        .expect("run succeeds");

    assert_eq!(run.text(), Some("done".to_string()));

    // The second model call must have seen the injected instruction.
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let second = &requests[1];
    assert!(
        second
            .messages
            .iter()
            .any(|m| m.text() == "STEER: focus on billing"),
        "second model request did not contain the injected steering message: {:?}",
        second.messages
    );

    // Observable via the event stream.
    let trajectory = Trajectory::from_events(recorder.events());
    trajectory.assert_order(&["agent.steered"]).unwrap();
    assert!(recorder.events().iter().any(|e| matches!(
        e,
        AgentEvent::Steered { command_kind, accepted: true } if command_kind == "inject_message"
    )));
}

#[tokio::test]
async fn cancel_terminates_the_run() {
    let recorder = EventRecorder::new();
    let handle = SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::Cancel));
    // Queue the cancel before the run starts; the first checkpoint drains it.
    handle.send(SteeringCommand::Cancel);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("never reached")));

    let ctx: RunContext = RunContext::new(RunConfig::new("run-cancel"), ())
        .with_events(recorder.sink())
        .with_steering(handle);

    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user("start")])
        .await
        .expect_err("run should be cancelled");

    assert!(matches!(err, RustAgentsError::Cancelled), "got {err:?}");

    // No model call ever happened and the cancel + failure are observable.
    let trajectory = Trajectory::from_events(recorder.events());
    assert_eq!(trajectory.model_call_count(), 0);
    assert!(trajectory.failed());
    assert!(recorder.events().iter().any(|e| matches!(
        e,
        AgentEvent::Steered { command_kind, accepted: true } if command_kind == "cancel"
    )));
}

#[tokio::test]
async fn disallowed_command_fails_the_run() {
    let recorder = EventRecorder::new();
    // Empty policy: every command is rejected.
    let handle = SteeringHandle::new(SteeringPolicy::new());
    handle.send(SteeringCommand::InjectMessage(Message::user("nope")));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("never reached")));

    let ctx: RunContext = RunContext::new(RunConfig::new("run-reject"), ())
        .with_events(recorder.sink())
        .with_steering(handle);

    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user("start")])
        .await
        .expect_err("run should fail on disallowed steering");

    assert!(matches!(err, RustAgentsError::Steering(_)), "got {err:?}");
    assert!(recorder.events().iter().any(|e| matches!(
        e,
        AgentEvent::Steered { command_kind, accepted: false } if command_kind == "inject_message"
    )));
}
