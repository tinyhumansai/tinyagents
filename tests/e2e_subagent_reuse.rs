//! TRUE end-to-end (offline): post-completion sub-agent REUSE with human input.
//!
//! An orchestrator calls a sub-agent, then takes (simulated) human input, then
//! calls the **same** sub-agent again — reusing it and carrying the prior
//! conversation context forward instead of killing/restarting it. This is the
//! reuse flow exposed by [`SubAgentSession`], distinct from steering (which
//! interrupts a still-running agent).
//!
//! The sub-agent is driven by a deterministic [`ScriptedModel`] so the test is
//! fully offline. We assert *structurally* — the composed result, the carried
//! context (via the model's recorded second request), and the reuse trajectory
//! (via a testkit [`Trajectory`]) — never on LLM prose.

use std::sync::Arc;

use tinyagents::harness::message::Message;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, ScriptedModel, Trajectory};
use tinyagents::{SubAgent, SubAgentSession};

#[tokio::test]
async fn orchestrator_reuses_subagent_with_human_input_between() {
    // A persistent "travel concierge" sub-agent. The scripted model answers the
    // first question, then a context-dependent follow-up.
    let model = Arc::new(ScriptedModel::replies(vec![
        "Tokyo is a great choice for a spring trip.",
        "For Tokyo in spring, pack light layers and a rain jacket.",
    ]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("scripted", model.clone());

    let subagent = Arc::new(
        SubAgent::new(
            "concierge",
            "a persistent travel concierge",
            Arc::new(harness),
        )
        .with_system_prompt("You are a helpful travel concierge."),
    );
    let subagent_handle = Arc::clone(&subagent);

    // The orchestrator wires the session's events into a recorder so the reuse
    // is observable as a trajectory.
    let recorder = EventRecorder::new();
    let mut session = SubAgentSession::new(subagent).with_events(recorder.sink());

    // ── Turn 1: orchestrator asks on behalf of the user ──────────────────────
    let first = session
        .send(
            &(),
            (),
            vec![Message::user("Where should I travel this spring?")],
        )
        .await
        .expect("first sub-agent run");
    assert_eq!(
        first.text(),
        Some("Tokyo is a great choice for a spring trip.".to_string())
    );

    // ── Simulated human-in-the-loop input arrives out-of-band ────────────────
    let human_followup = "Great — what should I pack for there?";

    // ── Turn 2: the SAME sub-agent is reused, carrying prior context ──────────
    let second = session
        .send(&(), (), vec![Message::user(human_followup)])
        .await
        .expect("second sub-agent run (reused)");
    assert_eq!(
        second.text(),
        Some("For Tokyo in spring, pack light layers and a rain jacket.".to_string())
    );

    // The same SubAgent instance was reused — never reconstructed.
    assert!(
        Arc::ptr_eq(session.subagent(), &subagent_handle),
        "the orchestrator reused the same sub-agent across turns"
    );
    assert_eq!(session.turns(), 2);

    // Carried context: the second model request contained the first turn's
    // question, the first answer, AND the human follow-up.
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "one model request per reused send");
    let second_texts: Vec<String> = requests[1].messages.iter().map(Message::text).collect();
    assert!(second_texts.contains(&"Where should I travel this spring?".to_string()));
    assert!(second_texts.contains(&"Tokyo is a great choice for a spring trip.".to_string()));
    assert!(second_texts.contains(&human_followup.to_string()));

    // The accumulating session transcript reflects the full conversation.
    let transcript_texts: Vec<String> = session.transcript().iter().map(Message::text).collect();
    assert!(transcript_texts.contains(&human_followup.to_string()));
    assert!(
        transcript_texts
            .contains(&"For Tokyo in spring, pack light layers and a rain jacket.".to_string())
    );

    // Trajectory: both child runs completed, and the second send is recorded as
    // a reuse (`subagent.reused`) bracketed by start/complete events.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_completed();
    traj.assert_order(&[
        "subagent.started",
        "subagent.completed",
        "subagent.reused",
        "subagent.completed",
    ])
    .expect("first run completes, then the sub-agent is reused for the second");
}
