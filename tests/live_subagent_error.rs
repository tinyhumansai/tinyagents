//! LIVE: a real OpenAI sub-agent whose only tool fails surfaces the failure.
//!
//! The child [`AgentHarness`] is driven by a real [`OpenAiModel`] and given a
//! single tool [`FakeTool::failing`] plus a directive system prompt that makes
//! the model call it. The tool returns `Err(TinyAgentsError::Tool(..))`, which
//! propagates out of the child agent loop, so [`SubAgent::invoke`] returns the
//! failure.
//!
//! # Skips gracefully
//!
//! Gated behind `#[cfg(feature = "openai")]`; returns early (after an
//! `eprintln!`) when `OPENAI_API_KEY` is unset.

#[cfg(feature = "openai")]
#[tokio::test]
async fn live_openai_subagent_surfaces_tool_failure() {
    use std::sync::Arc;

    use tinyagents::SubAgent;
    use tinyagents::error::TinyAgentsError;
    use tinyagents::harness::providers::openai::OpenAiModel;
    use tinyagents::harness::runtime::AgentHarness;
    use tinyagents::harness::testkit::FakeTool;

    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping live_openai_subagent_surfaces_tool_failure: OPENAI_API_KEY is not set");
        return;
    }

    // Child agent backed by a real model, with a single tool that always fails.
    let mut child: AgentHarness<()> = AgentHarness::new();
    child
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");
    child.register_tool(Arc::new(FakeTool::failing("lookup", "upstream 500")));

    let subagent = SubAgent::new(
        "lookup_agent",
        "Looks things up. Always uses the `lookup` tool to answer.",
        Arc::new(child),
    )
    .with_system_prompt(
        "You must answer every request by calling the `lookup` tool exactly once with any \
         arguments. Never answer directly; always call `lookup` first.",
    );

    let err = subagent
        .invoke(&(), (), 0, "Look up the capital of France.")
        .await
        .expect_err("the failing tool must surface as an error from the sub-agent run");

    // The failure surfaces as the tool error (its message is propagated).
    match err {
        TinyAgentsError::Tool(msg) => assert!(
            msg.contains("upstream 500"),
            "expected the tool error message to propagate, got {msg:?}"
        ),
        other => panic!("expected TinyAgentsError::Tool, got {other:?}"),
    }
}
