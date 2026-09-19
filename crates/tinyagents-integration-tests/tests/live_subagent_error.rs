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
//! This test is `#[ignore]`d and only runs opted in via
//! `tests/common/live.rs::require_live`, so the default `cargo test` passes
//! with no key configured and never dials a real provider by accident.

mod common;

#[tokio::test]
#[ignore = "network: set TINYAGENTS_LIVE=1 and run with --ignored"]
async fn live_openai_subagent_surfaces_tool_failure() {
    use std::sync::Arc;

    use tinyagents_harness::SubAgent;
    use tinyagents_harness::error::TinyAgentsError;
    use tinyagents_harness::runtime::AgentHarness;
    use tinyagents_harness::testkit::FakeTool;
    use tinyinference_llm::providers::openai::OpenAiModel;

    if !common::live::require_live(&["OPENAI_API_KEY"]) {
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
