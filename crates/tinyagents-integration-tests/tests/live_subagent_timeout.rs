//! LIVE: a real OpenAI sub-agent with a tiny wall-clock cap times out.
//!
//! The child [`AgentHarness`] is driven by a real [`OpenAiModel`] but its policy
//! caps wall-clock time at 1ms. Any real HTTP round-trip vastly exceeds that, so
//! the per-model-call budget interrupts the in-flight request and
//! [`SubAgent::invoke`] returns `Err(TinyAgentsError::Timeout(..))`.
//!
//! # Skips gracefully
//!
//! This test is `#[ignore]`d and only runs opted in via
//! `tests/common/live.rs::require_live`, so the default `cargo test` passes
//! with no key configured and never dials a real provider by accident.

mod common;

#[tokio::test]
#[ignore = "network: set TINYAGENTS_LIVE=1 and run with --ignored"]
async fn live_openai_subagent_times_out_on_tiny_budget() {
    use std::sync::Arc;

    use tinyagents_harness::SubAgent;
    use tinyagents_harness::error::TinyAgentsError;
    use tinyagents_harness::limits::RunLimits;
    use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
    use tinyinference_llm::providers::openai::OpenAiModel;

    if !common::live::require_live(&["OPENAI_API_KEY"]) {
        return;
    }

    // Child agent backed by a real model, with a 1ms wall-clock cap. A real
    // network call cannot complete in 1ms, so the per-call budget trips.
    let mut child: AgentHarness<()> = AgentHarness::new();
    child
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");
    child.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_wall_clock_ms(Some(1)),
        ..RunPolicy::default()
    });

    let subagent = SubAgent::new(
        "slow_lookup",
        "Answers questions via a real model.",
        Arc::new(child),
    );

    let err = subagent
        .invoke(&(), (), 0, "What is 2 + 2?")
        .await
        .expect_err("a 1ms budget must trip on any real network call");

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
}
