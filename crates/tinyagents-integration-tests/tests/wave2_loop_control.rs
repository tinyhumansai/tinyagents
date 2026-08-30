//! Regression coverage for loop control-flow checkpoints.
//!
//! - **LOOP-7**: `MiddlewareControl` was drained only *after* a model call, so a
//!   control requested from `after_tool` was honored one full model call late —
//!   an extra billable provider round trip after a guardrail (or a human gate)
//!   had already said stop.
//! - **LOOP-8**: a steering pause fell through to the success epilogue, so a
//!   paused run was indistinguishable from one whose model returned an empty
//!   final answer.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::TinyAgentsError;
use tinyagents_harness::context::{MiddlewareControl, RunConfig, RunContext};
use tinyagents_harness::ids::ExecutionStatus;
use tinyinference::message::Message;
use tinyagents_harness::middleware::Middleware;
use tinyinference::providers::MockModel;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::steering::{SteeringCommand, SteeringHandle, SteeringPolicy};
use tinyagents_harness::testkit::FakeTool;
use tinyagents_harness::tool::ToolResult;

/// Requests a control outcome from `after_tool` — the natural place for a
/// post-hoc guardrail or a budget stop that only knows once the result is in.
struct StopAfterToolMiddleware {
    control: MiddlewareControl,
}

#[async_trait]
impl Middleware<(), ()> for StopAfterToolMiddleware {
    fn name(&self) -> &str {
        "stop-after-tool"
    }

    async fn after_tool(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _result: &mut ToolResult,
    ) -> tinyagents_harness::Result<()> {
        ctx.request_control(self.control.clone());
        Ok(())
    }
}

fn spinning_harness() -> (AgentHarness<()>, Arc<MockModel>) {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let model = Arc::new(MockModel::with_tool_call("spin", json!({})));
    harness.register_model("mock", model.clone());
    harness.register_tool(Arc::new(FakeTool::returning("spin", "again")));
    (harness, model)
}

/// LOOP-7: `StopWithFinal` raised from `after_tool` stops the loop **on that
/// turn**, not after another model call.
#[tokio::test]
async fn stop_with_final_from_after_tool_is_honored_before_the_next_model_call() {
    let (mut harness, model) = spinning_harness();
    harness.push_middleware(Arc::new(StopAfterToolMiddleware {
        control: MiddlewareControl::StopWithFinal("stopped".to_string()),
    }));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run stops cleanly");

    assert_eq!(run.text(), Some("stopped".to_string()));
    assert_eq!(
        model.call_count(),
        1,
        "a control raised from after_tool must not cost another model call"
    );
}

/// LOOP-7: the same for `Interrupt`, where the wasted call is the more
/// expensive mistake — a human gate already said stop.
#[tokio::test]
async fn interrupt_from_after_tool_is_honored_before_the_next_model_call() {
    let (mut harness, model) = spinning_harness();
    harness.push_middleware(Arc::new(StopAfterToolMiddleware {
        control: MiddlewareControl::Interrupt {
            node: "approval".to_string(),
            message: "needs a human".to_string(),
        },
    }));

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("the interrupt surfaces");

    assert!(
        matches!(err, TinyAgentsError::Interrupted { .. }),
        "got {err:?}"
    );
    assert_eq!(
        model.call_count(),
        1,
        "an interrupt raised from after_tool must not cost another billable model call"
    );
}

/// LOOP-8: a steering pause is a distinct, resumable outcome — reported as
/// interrupted, carrying the pause reason, and keeping the transcript.
#[tokio::test]
async fn a_steering_pause_is_distinguishable_from_a_clean_finish() {
    let (harness, _model) = spinning_harness();

    let steering = SteeringHandle::new(SteeringPolicy::allow_all());
    steering.send(SteeringCommand::PauseWith {
        reason: "waiting for a human".to_string(),
    });

    let ctx: RunContext<()> = RunContext::new(RunConfig::new("paused"), ()).with_steering(steering);

    let result = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a pause is not a failure");

    let pause = result
        .run
        .paused
        .as_ref()
        .expect("a paused run must say so, not look like an empty final answer");
    assert_eq!(pause.reason.as_deref(), Some("waiting for a human"));
    assert!(result.run.final_response.is_none());
    assert_eq!(
        result.status.status,
        ExecutionStatus::Interrupted,
        "a paused run must not be reported as completed"
    );
}

/// A run with no steering still completes normally — the pause path must not
/// leak into the ordinary case.
#[tokio::test]
async fn an_unsteered_run_still_reports_completed() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("done")));

    let result = harness
        .invoke_with_status(&(), (), RunConfig::new("plain"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert!(result.run.paused.is_none());
    assert_eq!(result.status.status, ExecutionStatus::Completed);
}
