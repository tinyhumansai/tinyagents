//! Live steering test against the real OpenAI API.
//!
//! Skips gracefully (early return) when `OPENAI_API_KEY` is unset, so
//! `cargo test` stays green without credentials. Run it for real with:
//!
//! ```text
//! cargo test --test live_steering -- --nocapture
//! ```


use std::sync::Arc;

use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::message::Message;
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::{SteeringCommand, SteeringHandle};

#[tokio::test]
async fn orchestrator_steers_a_real_openai_run() {
    // Load .env so `cargo test` picks up local credentials.
    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("OPENAI_API_KEY not set — skipping live_steering test");
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
