//! Live steering test against the real OpenAI API.
//!
//! This test is `#[ignore]`d and only runs opted in via
//! `tests/common/live.rs::require_live`, so `cargo test` stays green without
//! credentials and never dials a real provider by accident. Run it for real
//! with:
//!
//! ```text
//! TINYAGENTS_LIVE=1 cargo test --test live_steering -- --ignored --nocapture
//! ```

mod common;

use std::sync::Arc;

use tinyagents_graph::*;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::*;
use tinyagents_language::*;
use tinyagents_registry::*;
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::openai::OpenAiModel;

#[tokio::test]
#[ignore = "network: set TINYAGENTS_LIVE=1 and run with --ignored"]
async fn orchestrator_steers_a_real_openai_run() {
    if !common::live::require_live(&["OPENAI_API_KEY"]) {
        return;
    }

    let model = OpenAiModel::from_env().expect("OpenAiModel from env");
    let mut harness = AgentHarness::<(), ()>::new();
    harness.register_model("openai", Arc::new(model));
    harness.set_default_model("openai");

    // The orchestrator injects an overriding instruction before the model call.
    let steering = SteeringHandle::allow_all();
    steering.send(SteeringCommand::Redirect {
        instruction: "Ignore the question's topic. Reply with EXACTLY the single \
                       word BANANA in uppercase and nothing else."
            .into(),
    });

    let ctx = RunContext::new(RunConfig::new("live-steer"), ()).with_steering(steering);

    let run = harness
        .invoke_in_context(
            &(),
            ctx,
            vec![Message::user("Tell me about the history of Rome.")],
        )
        .await
        .expect("live steered run completes");

    let answer = run.text().unwrap_or_default();
    eprintln!("steered answer: {answer:?}");
    assert!(
        answer.to_uppercase().contains("BANANA"),
        "the steered instruction should dominate the response, got {answer:?}"
    );
}
