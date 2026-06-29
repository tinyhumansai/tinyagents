//! TRUE end-to-end (offline): sub-agent ERROR propagation.
//!
//! A [`SubAgent`] is built over a child [`AgentHarness`] whose model always
//! requests a tool that fails ([`FakeTool::failing`]). The tool failure
//! propagates out of the child agent loop, so:
//!
//! - [`SubAgent::invoke`] returns `Err(TinyAgentsError::Tool(..))`,
//! - [`SubAgentTool::call`] surfaces the *same* error by propagation (its
//!   `call` does `self.subagent.invoke(..).await?`, so the contract is a
//!   propagated `Err`, not an error-tagged `ToolResult`), and
//! - an orchestrator that calls the failing sub-agent-as-tool observes the
//!   failure: its run errors out and a `RunFailed` event is recorded.
//!
//! All assertions are structural / on the error variant — never on model prose.

use std::sync::Arc;

use serde_json::json;

use tinyagents::error::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::message::Message;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, FakeTool, Trajectory};
use tinyagents::harness::tool::{Tool, ToolCall};
use tinyagents::{SubAgent, SubAgentTool};

/// Builds a child harness whose model always asks for the `broken` tool, which
/// fails with `Err(TinyAgentsError::Tool("boom"))`.
fn failing_child_harness() -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "child-model",
        Arc::new(MockModel::with_tool_call("broken", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::failing("broken", "boom")));
    harness
}

#[tokio::test]
async fn subagent_invoke_propagates_tool_failure() {
    let subagent = SubAgent::new(
        "broken_worker",
        "a worker whose tool always fails",
        Arc::new(failing_child_harness()),
    );

    let err = subagent
        .invoke(&(), (), 0, "do the thing")
        .await
        .expect_err("the child tool failure must propagate out of SubAgent::invoke");

    match err {
        TinyAgentsError::Tool(msg) => assert_eq!(msg, "boom"),
        other => panic!("expected TinyAgentsError::Tool(\"boom\"), got {other:?}"),
    }
}

#[tokio::test]
async fn subagent_tool_call_surfaces_failure_as_err() {
    // Contract: SubAgentTool::call propagates the child failure as an `Err`
    // (it does not swallow it into an error-tagged ToolResult).
    let subagent = Arc::new(SubAgent::new(
        "broken_worker",
        "a worker whose tool always fails",
        Arc::new(failing_child_harness()),
    ));
    let tool = SubAgentTool::new(subagent);

    let err = tool
        .call(&(), ToolCall::new("c1", "broken_worker", json!({ "input": "x" })))
        .await
        .expect_err("SubAgentTool::call must surface the child failure as an Err");

    match err {
        TinyAgentsError::Tool(msg) => assert_eq!(msg, "boom"),
        other => panic!("expected TinyAgentsError::Tool(\"boom\"), got {other:?}"),
    }
}

#[tokio::test]
async fn orchestrator_observes_failing_subagent_tool() {
    // An orchestrator equipped with the failing sub-agent as a tool. Its model
    // delegates to the tool on the first turn; the tool's propagated Err aborts
    // the orchestrator run and is recorded as a RunFailed event.
    let subagent = Arc::new(SubAgent::new(
        "broken_worker",
        "a worker whose tool always fails",
        Arc::new(failing_child_harness()),
    ));
    let tool = Arc::new(SubAgentTool::new(subagent));

    let mut orchestrator: AgentHarness<()> = AgentHarness::new();
    orchestrator.register_tool(tool);
    orchestrator.register_model(
        "parent-model",
        Arc::new(MockModel::with_tool_call(
            "broken_worker",
            json!({ "input": "delegate" }),
        )),
    );

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("orchestrator-run"), ()).with_events(recorder.sink());

    let err = orchestrator
        .invoke_in_context(&(), ctx, vec![Message::user("delegate this")])
        .await
        .expect_err("the failing sub-agent tool must abort the orchestrator run");

    match err {
        TinyAgentsError::Tool(msg) => assert_eq!(msg, "boom"),
        other => panic!("expected TinyAgentsError::Tool(\"boom\"), got {other:?}"),
    }

    // The orchestrator run emitted a RunFailed event (on_error fan-out path).
    let traj = Trajectory::from_events(recorder.events());
    assert!(
        traj.failed(),
        "orchestrator run should record a RunFailed event when the sub-agent tool fails"
    );
}
