//! Regression coverage for run-scoped call limits in the agent loop.
//!
//! Three defects are pinned here:
//!
//! - **LOOP-1**: an explicitly-set `RunConfig` call cap was silently widened by
//!   the harness `RunPolicy` default, so `with_max_model_calls(2)` ran 25.
//! - **LOOP-9b**: cap exhaustion was only ever a hard error, discarding the
//!   whole run even under `LimitBehavior::StopWithPartial`.
//! - **LOOP-12**: the working transcript was dropped on every failure path.

use std::sync::Arc;

use serde_json::json;

use tinyagents_harness::TinyAgentsError;
use tinyagents_harness::context::RunConfig;
use tinyagents_harness::limits::{LimitBehavior, RunLimits};
use tinyinference::message::Message;
use tinyinference::providers::MockModel;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::FakeTool;

/// A harness whose model always asks for the same tool, so the loop only ever
/// stops because a cap stops it.
fn spinning_harness() -> (AgentHarness<()>, Arc<MockModel>) {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let model = Arc::new(MockModel::with_tool_call("spin", json!({})));
    harness.register_model("mock", model.clone());
    harness.register_tool(Arc::new(FakeTool::returning("spin", "again")));
    (harness, model)
}

/// LOOP-1: an explicitly-set `RunConfig` model-call cap is a **ceiling**. The
/// permissive `RunPolicy` default (25) must not widen it.
///
/// Before the fix the loop overwrote the tracker's caps with the policy's by
/// plain assignment, so this ran 25 model calls instead of 2 — an embedder
/// budgeting a cheap sub-agent at 2 calls silently burned 25.
#[tokio::test]
async fn explicit_run_config_model_cap_is_not_widened_by_the_policy_default() {
    let (harness, model) = spinning_harness();

    let err = harness
        .invoke(
            &(),
            (),
            RunConfig::new("capped").with_max_model_calls(2),
            vec![Message::user("go")],
        )
        .await
        .expect_err("the explicit cap should trip");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
    assert_eq!(
        model.call_count(),
        2,
        "the explicitly-set cap of 2 must bound the run, not the policy default of 25"
    );
}

/// LOOP-1, tool axis: the same ceiling rule applies to `max_tool_calls`.
#[tokio::test]
async fn explicit_run_config_tool_cap_is_not_widened_by_the_policy_default() {
    let (harness, model) = spinning_harness();

    let err = harness
        .invoke(
            &(),
            (),
            RunConfig::new("capped").with_max_tool_calls(1),
            vec![Message::user("go")],
        )
        .await
        .expect_err("the explicit tool cap should trip");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
    // Turn 1 spends the single tool call; turn 2 requests a second and trips.
    assert_eq!(
        model.call_count(),
        2,
        "the explicit tool cap of 1 must bound the run, not the policy default of 50"
    );
}

/// LOOP-1, the legitimate loosening case that must keep working: an **unset**
/// `RunConfig` cap merely defaulted, so a policy configuring a higher cap is
/// the only real source of truth and wins.
#[tokio::test]
async fn unset_run_config_cap_lets_the_policy_raise_the_limit() {
    let (mut harness, model) = spinning_harness();
    harness.with_policy(RunPolicy {
        limits: RunLimits::default()
            .with_max_model_calls(30)
            .with_max_tool_calls(1000),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("the policy cap should trip");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
    assert!(
        err.to_string().contains("30"),
        "expected the policy limit (30) to be reported, got: {err}"
    );
    assert_eq!(
        model.call_count(),
        30,
        "an unset config cap must not clamp the policy's higher cap back to the default"
    );
}

/// LOOP-9b: under `LimitBehavior::StopWithPartial` the model-call cap ends the
/// run **cleanly**, keeping the transcript, counters, and usage rather than
/// discarding all of it behind a `LimitExceeded`.
#[tokio::test]
async fn model_cap_stops_with_the_partial_run_under_stop_with_partial() {
    let (mut harness, _model) = spinning_harness();
    harness.with_policy(RunPolicy {
        limits: RunLimits::default()
            .with_max_model_calls(2)
            .with_behavior(LimitBehavior::StopWithPartial),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("StopWithPartial must not fail the run");

    assert_eq!(run.model_calls, 2);
    assert!(
        run.messages.len() > 1,
        "the partial transcript must survive, got {:?}",
        run.messages.len()
    );
    assert!(
        run.tool_calls >= 1,
        "work completed before the cap must be reported"
    );
}

/// LOOP-12: even on the hard-error path the accumulated transcript is worth
/// keeping. `invoke_collecting_partial` hands back the partial run alongside
/// the error instead of dropping it.
#[tokio::test]
async fn a_failed_run_still_returns_its_partial_transcript() {
    let (harness, _model) = spinning_harness();

    let outcome = harness
        .invoke_collecting_partial(
            &(),
            (),
            RunConfig::new("partial").with_max_model_calls(2),
            vec![Message::user("go")],
        )
        .await;

    let error = outcome.error.expect("the cap should trip");
    assert!(
        matches!(error, TinyAgentsError::LimitExceeded(_)),
        "got {error:?}"
    );
    assert!(
        outcome.run.messages.len() > 1,
        "the working transcript must be preserved on the failure path, got {:?}",
        outcome.run.messages
    );
    assert_eq!(outcome.run.model_calls, 2);
}
