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

use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::events::AgentEvent;
use crate::runtime::AgentHarness;
use crate::steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringOutcome, SteeringPolicy,
    SteeringTarget, apply_pending_steering,
};
use crate::testkit::{EventRecorder, Trajectory};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolResult};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Builds a plain-text assistant response.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: tinyinference_llm::message::AssistantMessage {
            id: None,
            content: vec![tinyinference_llm::message::ContentBlock::Text(
                text.to_string(),
            )],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(1, 1)),
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
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
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
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
                message: tinyinference_llm::message::AssistantMessage {
                    id: Some("m1".to_string()),
                    content: Vec::new(),
                    tool_calls: vec![ToolCall::new("c1", "noop", json!({}))],
                    usage: Some(Usage::new(1, 1)),
                },
                usage: Some(Usage::new(1, 1)),
                finish_reason: Some("tool_calls".to_string()),
                raw: None,
                resolved_model: None,
                continue_turn: None,
                served_from_cache: false,
                correlation: None,
                resolved_route: None,
            })
        } else {
            Ok(text_response("done"))
        }
    }
}

/// A no-op tool that lets the loop iterate.
struct NoopTool;

#[async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "noop"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
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
fn disallowed_command_is_rejected_with_steered_event_and_the_run_continues() {
    let recorder = EventRecorder::new();
    // Policy permits Pause but not Cancel.
    let handle = SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::Pause));
    handle.send(SteeringCommand::Cancel);
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ())
        .with_events(recorder.sink())
        .with_steering(handle);
    let mut messages = Vec::new();

    // A disallowed command is rejected on its own; it no longer fails the
    // checkpoint (I-5/M-7).
    let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();
    assert_eq!(outcome, SteeringOutcome::Continue);
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

    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");

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
async fn disallowed_command_is_skipped_and_the_run_still_completes() {
    let recorder = EventRecorder::new();
    // Empty policy: every command is rejected.
    let handle = SteeringHandle::new(SteeringPolicy::new());
    handle.send(SteeringCommand::InjectMessage(Message::user("nope")));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("reached")));

    let ctx: RunContext = RunContext::new(RunConfig::new("run-reject"), ())
        .with_events(recorder.sink())
        .with_steering(handle);

    // A disallowed steering command no longer kills the run (I-5/M-7); it is
    // rejected individually and the loop continues.
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("start")])
        .await
        .expect("run should complete despite the rejected steering command");
    assert_eq!(run.text(), Some("reached".to_string()));

    assert!(recorder.events().iter().any(|e| matches!(
        e,
        AgentEvent::Steered { command_kind, accepted: false } if command_kind == "inject_message"
    )));
}

/// Queue accessors must recover from a poisoned mutex (a panic in another
/// holder) instead of panicking: steering is reachable from arbitrary caller
/// threads, so a single panicking user must not brick the queue.
#[test]
fn steering_queue_recovers_from_poisoned_lock() {
    let handle = SteeringHandle::allow_all();
    handle.send(SteeringCommand::InjectMessage(Message::user("before")));

    // Poison the queue mutex by panicking while holding it.
    let poisoner = handle.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poisoner.inner.queue.lock().unwrap();
        panic!("poison the steering queue");
    })
    .join();

    // Every accessor still works on the recovered state.
    assert!(!handle.is_empty());
    assert_eq!(handle.pending(), 1);
    handle.send(SteeringCommand::InjectMessage(Message::user("after")));
    assert_eq!(handle.drain().len(), 2);
    assert!(handle.is_empty());
}

// ── I-5/M-7: a disallowed command in a batch is rejected individually ─────────

#[test]
fn a_rejected_command_in_a_batch_does_not_drop_the_allowed_ones() {
    // Regression test (I-5/M-7): `apply_pending_steering` used to validate
    // the whole drained batch up front and refuse it entirely — including
    // commands the policy *did* permit — the moment one command in it was
    // disallowed, and the caller's `?` then killed the run. A command the
    // policy disallows must be rejected on its own; every allowed command in
    // the same batch still applies, and the checkpoint does not error.
    let recorder = EventRecorder::new();
    let handle = SteeringHandle::new(
        SteeringPolicy::new()
            .allow(SteeringCommandKind::InjectMessage)
            .allow(SteeringCommandKind::SetMetadata),
    );
    handle.send(SteeringCommand::InjectMessage(Message::user("first")));
    handle.send(SteeringCommand::SetMetadata {
        metadata: serde_json::json!({"tag": "applied"}),
    });
    // Not allowed → rejected individually, the rest of the batch still runs.
    handle.send(SteeringCommand::Cancel);
    handle.send(SteeringCommand::InjectMessage(Message::user("last")));

    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ())
        .with_events(recorder.sink())
        .with_steering(handle);
    let mut messages = Vec::new();

    let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();
    assert_eq!(outcome, SteeringOutcome::Continue);

    assert_eq!(
        messages,
        vec![Message::user("first"), Message::user("last")],
        "allowed commands in the batch should still have applied"
    );
    assert_eq!(
        ctx.config.metadata,
        serde_json::json!({"tag": "applied"}),
        "the allowed SetMetadata command should still have applied"
    );
    // Every command gets its own event: accepted, accepted, rejected, accepted.
    assert_eq!(
        recorder.events(),
        vec![
            AgentEvent::Steered {
                command_kind: "inject_message".to_string(),
                accepted: true,
            },
            AgentEvent::Steered {
                command_kind: "set_metadata".to_string(),
                accepted: true,
            },
            AgentEvent::Steered {
                command_kind: "cancel".to_string(),
                accepted: false,
            },
            AgentEvent::Steered {
                command_kind: "inject_message".to_string(),
                accepted: true,
            },
        ]
    );
}

// ── LOOP-8(b): a pause is latched and resumable across checkpoints ────────────

#[test]
fn a_pause_survives_the_batch_and_holds_later_checkpoints() {
    // Regression test (LOOP-8b): `Resume` only cancelled a `Pause` from the
    // *same* drained batch, and the outcome was recomputed from scratch each
    // checkpoint — so a pause applied at one checkpoint silently evaporated at
    // the next, and could never be deliberately resumed either.
    let handle = SteeringHandle::allow_all();
    handle.send(SteeringCommand::Pause);
    let mut ctx: RunContext =
        RunContext::new(RunConfig::new("r"), ()).with_steering(handle.clone());
    let mut messages = Vec::new();

    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Pause
    );
    assert!(handle.is_paused());

    // A later checkpoint with an EMPTY queue must stay paused.
    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Pause,
        "the pause evaporated at the next checkpoint"
    );
}

#[test]
fn a_pause_is_resumable_from_a_later_batch() {
    let handle = SteeringHandle::allow_all();
    let mut ctx: RunContext =
        RunContext::new(RunConfig::new("r"), ()).with_steering(handle.clone());
    let mut messages = Vec::new();

    handle.send(SteeringCommand::Pause);
    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Pause
    );

    // The resume arrives long after the pause was applied — the case that was
    // impossible before.
    handle.send(SteeringCommand::Resume);
    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Continue,
        "a pause applied in an earlier batch was unresumable"
    );
    assert!(!handle.is_paused());
    assert!(handle.pause_state().is_none());
}

#[test]
fn pause_state_makes_a_paused_run_distinguishable_from_an_empty_answer() {
    // The loop reports `final_response: None` for both a pause and an empty
    // model turn; `pause_state()` is what tells the caller which happened.
    let handle = SteeringHandle::allow_all();
    handle.send(SteeringCommand::PauseWith {
        reason: "waiting for human approval".into(),
    });
    let mut ctx: RunContext =
        RunContext::new(RunConfig::new("r"), ()).with_steering(handle.clone());
    let mut messages = Vec::new();

    let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();
    assert!(outcome.is_pause());

    let state = handle
        .pause_state()
        .expect("a Pause outcome must always carry a PauseState");
    assert_eq!(state.reason.as_deref(), Some("waiting for human approval"));
    assert_eq!(state.paused_at_checkpoint, 0);

    // A run that was never paused has no state at all.
    assert!(SteeringHandle::allow_all().pause_state().is_none());
}

#[test]
fn a_repeated_pause_keeps_the_original_reason_and_checkpoint() {
    let handle = SteeringHandle::allow_all();
    let mut ctx: RunContext =
        RunContext::new(RunConfig::new("r"), ()).with_steering(handle.clone());
    let mut messages = Vec::new();

    handle.send(SteeringCommand::PauseWith {
        reason: "first reason".into(),
    });
    apply_pending_steering(&mut ctx, &mut messages).unwrap();

    handle.send(SteeringCommand::PauseWith {
        reason: "second reason".into(),
    });
    apply_pending_steering(&mut ctx, &mut messages).unwrap();

    let state = handle.pause_state().expect("still paused");
    assert_eq!(state.reason.as_deref(), Some("first reason"));
    assert_eq!(state.paused_at_checkpoint, 0);
}

#[test]
fn pause_with_is_gated_by_the_same_policy_kind_as_pause() {
    assert_eq!(
        SteeringCommand::PauseWith {
            reason: "why".into()
        }
        .kind(),
        SteeringCommandKind::Pause
    );

    // A policy that forbids Pause forbids PauseWith too: rejected
    // individually, and the checkpoint continues rather than the run dying.
    let handle = SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::Resume));
    handle.send(SteeringCommand::PauseWith {
        reason: "why".into(),
    });
    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_steering(handle);
    let mut messages = Vec::new();
    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Continue
    );
}

#[test]
fn handle_resume_clears_a_latch_without_going_through_the_queue() {
    let handle = SteeringHandle::allow_all();
    handle.send(SteeringCommand::Pause);
    let mut ctx: RunContext =
        RunContext::new(RunConfig::new("r"), ()).with_steering(handle.clone());
    let mut messages = Vec::new();
    apply_pending_steering(&mut ctx, &mut messages).unwrap();

    let cleared = handle.resume().expect("a pause was in effect");
    assert_eq!(cleared.paused_at_checkpoint, 0);
    assert!(!handle.is_paused());
    assert!(handle.resume().is_none(), "resume must be idempotent");

    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Continue
    );
}

#[test]
fn pause_with_round_trips_through_json() {
    let command = SteeringCommand::PauseWith {
        reason: "human review".into(),
    };
    let json = serde_json::to_value(&command).expect("serialize");
    let back: SteeringCommand = serde_json::from_value(json).expect("deserialize");
    assert_eq!(back, command);
}

// ── I-5: a child run only drains commands addressed to it ─────────────────────

#[test]
fn root_addressed_command_is_not_consumed_by_a_child() {
    // Regression test (I-5): `RunContext::child` used to hand the child a bare
    // clone of the parent's `SteeringHandle`, so a command an orchestrator
    // addressed to the parent (the default target) could be drained and
    // applied by whichever sub-agent reached its checkpoint first.
    let handle = SteeringHandle::allow_all();
    let mut parent: RunContext =
        RunContext::new(RunConfig::new("parent"), ()).with_steering(handle.clone());
    let child_config = RunConfig::new("child");
    let mut child: RunContext = parent.child(child_config, ()).unwrap();

    // Addressed to the default target (Root == the parent).
    handle.send(SteeringCommand::InjectMessage(Message::user(
        "for the parent",
    )));

    let mut child_messages = Vec::new();
    let outcome = apply_pending_steering(&mut child, &mut child_messages).unwrap();
    assert_eq!(outcome, SteeringOutcome::Continue);
    assert!(
        child_messages.is_empty(),
        "child drained a command addressed to the root: {child_messages:?}"
    );

    // The parent's own checkpoint still sees it.
    let mut parent_messages = Vec::new();
    apply_pending_steering(&mut parent, &mut parent_messages).unwrap();
    assert_eq!(parent_messages, vec![Message::user("for the parent")]);
}

#[test]
fn run_addressed_command_reaches_only_that_run() {
    let handle = SteeringHandle::allow_all();
    let parent: RunContext =
        RunContext::new(RunConfig::new("parent"), ()).with_steering(handle.clone());
    let mut child_a: RunContext = parent.child(RunConfig::new("child-a"), ()).unwrap();
    let mut child_b: RunContext = parent.child(RunConfig::new("child-b"), ()).unwrap();

    handle.send_to(
        SteeringTarget::Run(child_a.run_id().clone()),
        SteeringCommand::InjectMessage(Message::user("for child-a only")),
    );

    let mut a_messages = Vec::new();
    apply_pending_steering(&mut child_a, &mut a_messages).unwrap();
    assert_eq!(a_messages, vec![Message::user("for child-a only")]);

    let mut b_messages = Vec::new();
    apply_pending_steering(&mut child_b, &mut b_messages).unwrap();
    assert!(
        b_messages.is_empty(),
        "a command addressed to child-a leaked into child-b: {b_messages:?}"
    );
}

#[test]
fn all_addressed_command_is_drained_by_whichever_run_checkpoints_first() {
    // `SteeringTarget::All` matches any handle sharing the queue, but delivery
    // is still pull-and-consume-once: whichever run reaches its checkpoint
    // first drains it, exactly like the pre-routing behaviour for every
    // command. It is documented that way (see `SteeringTarget::All`), unlike
    // `Root`/`Run(id)` which are exclusive to one run by construction.
    let handle = SteeringHandle::allow_all();
    let parent: RunContext =
        RunContext::new(RunConfig::new("parent"), ()).with_steering(handle.clone());
    let mut child: RunContext = parent.child(RunConfig::new("child"), ()).unwrap();
    let mut parent = parent;

    handle.send_all(SteeringCommand::InjectMessage(Message::user("broadcast")));

    let mut child_messages = Vec::new();
    apply_pending_steering(&mut child, &mut child_messages).unwrap();
    assert_eq!(child_messages, vec![Message::user("broadcast")]);

    // Already drained by the child; the parent's checkpoint sees nothing.
    let mut parent_messages = Vec::new();
    apply_pending_steering(&mut parent, &mut parent_messages).unwrap();
    assert!(parent_messages.is_empty());
}
