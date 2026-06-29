//! LIVE: a real OpenAI sub-agent with a tiny wall-clock cap times out.
//!
//! The child [`AgentHarness`] is driven by a real [`OpenAiModel`] but its policy
//! caps wall-clock time at 1ms. Any real HTTP round-trip vastly exceeds that, so
//! the per-model-call budget interrupts the in-flight request and
//! [`SubAgent::invoke`] returns `Err(TinyAgentsError::Timeout(..))`.
//!
//! # Skips gracefully
//!
//! Gated behind `#[cfg(feature = "openai")]`; returns early (after an
//! `eprintln!`) when `OPENAI_API_KEY` is unset.

#[cfg(feature = "openai")]
#[tokio::test]
async fn live_openai_subagent_times_out_on_tiny_budget() {
    use std::sync::Arc;

    use tinyagents::SubAgent;
    use tinyagents::error::TinyAgentsError;
    use tinyagents::harness::limits::RunLimits;
    use tinyagents::harness::providers::openai::OpenAiModel;
    use tinyagents::harness::runtime::{AgentHarness, RunPolicy};

    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!(
            "skipping live_openai_subagent_times_out_on_tiny_budget: OPENAI_API_KEY is not set"
        );
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
