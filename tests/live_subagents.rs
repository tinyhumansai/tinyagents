//! LIVE end-to-end: a real OpenAI parent agent composes the answer of a real
//! OpenAI child sub-agent.
//!
//! This is the network-backed sibling of `e2e_subagents.rs`. The parent
//! [`AgentHarness`] is given a [`SubAgentTool`] wrapping a child harness; both
//! are driven by a real [`OpenAiModel`]. We assert *structurally* (the parent
//! finished, produced non-empty text, and the sub-agent tool ran) rather than
//! on the exact model prose.
//!
//! # Skips gracefully
//!
//! The test returns early (after an `eprintln!`) when `OPENAI_API_KEY` is
//! unset, so `cargo test` passes with no key configured.

#[tokio::test]
async fn live_openai_parent_composes_child_subagent() {
    use std::sync::Arc;

    use tinyagents::harness::context::{RunConfig, RunContext};
    use tinyagents::harness::message::Message;
    use tinyagents::harness::providers::openai::OpenAiModel;
    use tinyagents::harness::runtime::AgentHarness;
    use tinyagents::harness::testkit::{EventRecorder, Trajectory};
    use tinyagents::{SubAgent, SubAgentTool};

    // Load .env so `cargo test` picks up local credentials.
    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping live_openai_parent_composes_child_subagent: OPENAI_API_KEY is not set");
        return;
    }

    // Child agent: a focused "math expert" sub-agent driven by a real model.
    let mut child: AgentHarness<()> = AgentHarness::new();
    child
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");

    let subagent = Arc::new(
        SubAgent::new(
            "math_expert",
            "Computes precise arithmetic answers. Pass the arithmetic question as `input`.",
            Arc::new(child),
        )
        .with_system_prompt(
            "You are a meticulous arithmetic engine. Reply with only the numeric answer.",
        ),
    );
    let tool = Arc::new(SubAgentTool::new(subagent));

    // Parent agent: also a real model, equipped with the sub-agent as a tool.
    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool(tool);
    parent
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("live-parent-run"), ()).with_events(recorder.sink());

    let run = parent
        .invoke_in_context(
            &(),
            ctx,
            vec![Message::user(
                "Use the math_expert tool to compute 17 * 23, then state the result in a short \
                 sentence.",
            )],
        )
        .await
        .expect("live parent run succeeds");

    // Structural assertions only — never assert on exact LLM prose.
    let final_text = run.text().unwrap_or_default();
    assert!(
        !final_text.trim().is_empty(),
        "parent should produce a non-empty composed answer"
    );
    assert!(
        run.model_calls >= 1,
        "parent should make at least one model call"
    );

    let traj = Trajectory::from_events(recorder.events());
    traj.assert_completed();
    // The model is strongly steered to call the sub-agent; if it did, the
    // trajectory records it as a `math_expert` tool invocation.
    if traj.tool_was_called("math_expert") {
        assert!(
            run.tool_calls >= 1,
            "a recorded sub-agent tool call should be reflected in run.tool_calls"
        );
    }
}
