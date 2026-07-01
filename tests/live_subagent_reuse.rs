//! LIVE end-to-end: post-completion sub-agent REUSE against a real OpenAI model.
//!
//! A real OpenAI sub-agent answers a first question, then a follow-up that
//! *depends on the first answer* — proving the sub-agent was reused with its
//! prior conversation context intact (rather than killed and restarted). This
//! is the network-backed sibling of `e2e_subagent_reuse.rs`.
//!
//! # Skips gracefully
//!
//! Returns early (after an `eprintln!`) when `OPENAI_API_KEY` is unset, so the
//! default `cargo test` passes with no key configured.

#[tokio::test]
async fn live_openai_subagent_reused_with_carried_context() {
    use std::sync::Arc;

    use tinyagents::harness::message::Message;
    use tinyagents::harness::providers::openai::OpenAiModel;
    use tinyagents::harness::runtime::AgentHarness;
    use tinyagents::harness::testkit::{EventRecorder, Trajectory};
    use tinyagents::{SubAgent, SubAgentSession};

    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!(
            "skipping live_openai_subagent_reused_with_carried_context: OPENAI_API_KEY is not set"
        );
        return;
    }

    // A real OpenAI sub-agent we will reuse across two turns.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");

    let subagent = Arc::new(
        SubAgent::new(
            "assistant",
            "a helpful assistant that remembers the conversation",
            Arc::new(harness),
        )
        .with_system_prompt(
            "You are a concise assistant. Answer using the full prior conversation context.",
        ),
    );

    let recorder = EventRecorder::new();
    let mut session = SubAgentSession::new(subagent).with_events(recorder.sink());

    // ── Turn 1: establish a fact the follow-up will depend on. ───────────────
    let first = session
        .send(
            &(),
            (),
            vec![Message::user(
                "My favorite color is teal. Please acknowledge it in one short sentence.",
            )],
        )
        .await
        .expect("first live sub-agent run");
    let first_text = first.text().unwrap_or_default();
    assert!(
        !first_text.trim().is_empty(),
        "first turn should produce a non-empty answer"
    );

    // ── Turn 2: a follow-up that ONLY succeeds if prior context was reused. ───
    let second = session
        .send(
            &(),
            (),
            vec![Message::user(
                "What is my favorite color? Answer with just the color word.",
            )],
        )
        .await
        .expect("second live sub-agent run (reused)");
    let second_text = second.text().unwrap_or_default().to_lowercase();

    assert_eq!(session.turns(), 2);
    assert!(
        second_text.contains("teal"),
        "the reused sub-agent should recall the earlier answer (favorite color), got: {second_text:?}"
    );

    // The reuse is observable in the trajectory.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_completed();
    assert!(
        recorder
            .events()
            .iter()
            .any(|e| matches!(e, tinyagents::harness::events::AgentEvent::SubAgentReused { turn, .. } if *turn == 1)),
        "the second send should record a SubAgentReused event"
    );
}
