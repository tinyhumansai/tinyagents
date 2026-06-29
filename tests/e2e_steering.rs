//! End-to-end tests for orchestrator → sub-agent steering.
//!
//! These compose the steering channel with a real `AgentHarness` run driven by
//! a deterministic `ScriptedModel`, asserting that policy-permitted steering
//! commands mutate the run as documented and that the effects are observable on
//! the event stream — all offline (no network).

use std::sync::Arc;

use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::message::Message;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, ScriptedModel, Trajectory};
use tinyagents::{
    RustAgentsError, SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringPolicy,
};

/// Builds a harness whose only model is a `ScriptedModel`; returns both the
/// harness and the shared model handle so the test can inspect the requests the
/// model actually received.
fn harness_with_scripted(replies: Vec<&str>) -> (AgentHarness<(), ()>, Arc<ScriptedModel>) {
    let model = Arc::new(ScriptedModel::replies(replies));
    let mut harness = AgentHarness::<(), ()>::new();
    harness.register_model("scripted", model.clone());
    harness.set_default_model("scripted");
    (harness, model)
}

#[tokio::test]
async fn injected_message_reaches_the_next_model_call() {
    let (harness, model) = harness_with_scripted(vec!["done"]);

    // Orchestrator pre-queues an instruction before the run starts. The loop
    // drains it at the steering checkpoint before the first model call.
    let steering = SteeringHandle::allow_all();
    steering.send(SteeringCommand::InjectMessage(Message::user(
        "ORCHESTRATOR: always answer in French.",
    )));

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("steer-inject"), ())
        .with_steering(steering)
        .with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("Hello")])
        .await
        .expect("run completes");

    // The model received the injected instruction in its transcript.
    let requests = model.requests();
    assert_eq!(requests.len(), 1, "exactly one model call");
    let saw_injection = requests[0]
        .messages
        .iter()
        .any(|m| m.text().contains("always answer in French"));
    assert!(saw_injection, "injected steering message reached the model");

    // The steering action is observable on the event stream.
    let kinds = recorder.kinds();
    assert!(
        kinds.iter().any(|k| k.ends_with("steered")),
        "a Steered event was emitted, got {kinds:?}"
    );

    // And the run still completed normally.
    Trajectory::from_events(recorder.events()).assert_completed();
    assert_eq!(run.model_calls, 1);
}

#[tokio::test]
async fn cancel_terminates_the_run() {
    let (harness, _model) = harness_with_scripted(vec!["should never be returned"]);

    let steering = SteeringHandle::allow_all();
    steering.send(SteeringCommand::Cancel);

    let ctx = RunContext::new(RunConfig::new("steer-cancel"), ()).with_steering(steering);

    let result = harness
        .invoke_in_context(&(), ctx, vec![Message::user("Hi")])
        .await;

    assert!(
        matches!(result, Err(RustAgentsError::Cancelled)),
        "Cancel steers the run to terminate with Cancelled, got {result:?}"
    );
}

#[tokio::test]
async fn redirect_appends_a_system_instruction() {
    let (harness, model) = harness_with_scripted(vec!["ok"]);

    let steering = SteeringHandle::allow_all();
    steering.send(SteeringCommand::Redirect {
        instruction: "switch to a terse style".into(),
    });

    let ctx = RunContext::new(RunConfig::new("steer-redirect"), ()).with_steering(steering);

    harness
        .invoke_in_context(&(), ctx, vec![Message::user("Explain Rust traits")])
        .await
        .expect("run completes");

    let requests = model.requests();
    let saw_redirect = requests[0]
        .messages
        .iter()
        .any(|m| m.text().contains("switch to a terse style"));
    assert!(saw_redirect, "redirect instruction reached the model");
}

#[tokio::test]
async fn disallowed_command_is_rejected_by_policy() {
    let (harness, _model) = harness_with_scripted(vec!["ok"]);

    // Policy permits injecting messages but NOT cancelling.
    let policy = SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage);
    let steering = SteeringHandle::new(policy);
    steering.send(SteeringCommand::Cancel);

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("steer-deny"), ())
        .with_steering(steering)
        .with_events(recorder.sink());

    let result = harness
        .invoke_in_context(&(), ctx, vec![Message::user("Hi")])
        .await;

    assert!(
        matches!(result, Err(RustAgentsError::Steering(_))),
        "a command outside the policy allowlist is rejected, got {result:?}"
    );
    // The rejection is observable (a Steered event with accepted = false).
    assert!(
        recorder.kinds().iter().any(|k| k.ends_with("steered")),
        "a Steered event was emitted for the rejected command"
    );
}
