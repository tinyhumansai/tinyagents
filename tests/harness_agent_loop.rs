//! End-to-end coverage for the harness agent loop.
//!
//! Exercises a single-turn response, a multi-step tool loop, the
//! `max_model_calls` limit, and usage accumulation. Behavioral assertions use
//! the testkit [`Trajectory`] (tool-was-called, model-call count) plus the
//! [`AgentRun`] counters — never the exact model prose.
//!
//! The harness owns its internal [`EventSink`], so to feed a `Trajectory` we
//! subscribe a [`RecordingListener`] to the run's sink from a small middleware
//! at `before_agent` time. All subsequent model/tool/run events are captured.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents::TinyAgentsError;
use tinyagents::harness::context::RunContext;
use tinyagents::harness::events::RecordingListener;
use tinyagents::harness::limits::RunLimits;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::middleware::Middleware;
use tinyagents::harness::model::{
    CapabilitySet, ChatModel, ModelHint, ModelProfile, ModelRequest, ModelResolutionSource,
    ModelResponse,
};
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
use tinyagents::harness::testkit::{FakeTool, Trajectory};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;

/// Middleware that subscribes a shared [`RecordingListener`] to the run's event
/// sink so the test can reconstruct a [`Trajectory`] afterwards.
struct CaptureMiddleware {
    listener: Arc<RecordingListener>,
}

#[async_trait]
impl Middleware<(), ()> for CaptureMiddleware {
    fn name(&self) -> &str {
        "capture"
    }

    async fn before_agent(&self, ctx: &mut RunContext<()>, _state: &()) -> tinyagents::Result<()> {
        ctx.events.subscribe(self.listener.clone());
        Ok(())
    }
}

/// Builds the recording listener and a [`Trajectory`] from its captured events.
fn trajectory(listener: &Arc<RecordingListener>) -> Trajectory {
    let events = listener.events().into_iter().map(|r| r.event).collect();
    Trajectory::from_events(events)
}

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

fn text_response(text: &str, input: u64, output: u64) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(input, output)),
        },
        usage: Some(Usage::new(input, output)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

struct ProfiledIntegrationModel {
    profile: ModelProfile,
    text: &'static str,
}

#[async_trait]
impl ChatModel<()> for ProfiledIntegrationModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyagents::Result<ModelResponse> {
        Ok(ModelResponse::assistant(self.text))
    }
}

struct RequireJsonSchemaMiddleware;

#[async_trait]
impl Middleware<(), ()> for RequireJsonSchemaMiddleware {
    fn name(&self) -> &str {
        "require_json_schema"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> tinyagents::Result<()> {
        request.required_capabilities = Some(CapabilitySet {
            json_schema: true,
            ..CapabilitySet::default()
        });
        request.model_hints.push(ModelHint {
            model: "capable".to_string(),
            priority: 1,
            reason: Some("needs json schema".to_string()),
        });
        Ok(())
    }
}

#[tokio::test]
async fn single_turn_response_completes_with_one_model_call() {
    let listener = Arc::new(RecordingListener::new());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(MockModel::constant("hello there")))
        .set_default_model("mock")
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 1);
    assert_eq!(run.tool_calls, 0);
    assert!(run.text().is_some(), "a final response should be produced");

    let traj = trajectory(&listener);
    traj.assert_model_called_times(1);
    traj.assert_completed();
    assert!(!traj.failed());
}

#[tokio::test]
async fn required_capabilities_select_capable_model() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "incapable",
            Arc::new(ProfiledIntegrationModel {
                profile: ModelProfile::default(),
                text: "incapable",
            }),
        )
        .set_default_model("incapable")
        .register_model(
            "capable",
            Arc::new(ProfiledIntegrationModel {
                profile: ModelProfile {
                    json_schema: true,
                    ..ModelProfile::default()
                },
                text: "capable",
            }),
        )
        .push_middleware(Arc::new(RequireJsonSchemaMiddleware));

    let run = harness
        .invoke_default(&(), vec![Message::user("pick a model")])
        .await
        .expect("capable hinted model satisfies required capabilities");

    assert_eq!(run.text(), Some("capable".to_string()));
    let resolved = run
        .final_response
        .expect("final response")
        .resolved_model
        .expect("resolved model metadata");
    assert_eq!(resolved.name, "capable");
    assert_eq!(resolved.source, ModelResolutionSource::Hint);
}

#[tokio::test]
async fn multi_step_tool_loop_calls_tool_then_finishes() {
    let listener = Arc::new(RecordingListener::new());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "lookup", json!({ "q": "x" })),
                text_response("done", 4, 2),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));

    let run = harness
        .invoke_default(&(), vec![Message::user("please look up")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 2);
    assert_eq!(run.tool_calls, 1);

    let traj = trajectory(&listener);
    // Structural assertions only — never the model prose.
    traj.assert_tool_called("lookup");
    assert_eq!(traj.tool_call_count("lookup"), 1);
    traj.assert_model_called_times(2);
    traj.assert_completed();
    // The tool was started before the second (final) model call.
    traj.assert_order(&["lookup", "model.completed"])
        .expect("tool runs before the final model completion");
}

#[tokio::test]
async fn max_model_calls_limit_returns_limit_exceeded() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    // A model that always asks for a tool would loop forever without a cap.
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_tool_call("spin", json!({}))),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("spin", "again")))
        .with_policy(RunPolicy {
            limits: RunLimits::default().with_max_model_calls(1),
            ..RunPolicy::default()
        });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("the model-call cap should be exceeded");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn usage_accumulates_across_model_calls() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "lookup", json!({})),
                text_response("done", 4, 2),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "out")));

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    // Two model calls folded into the totals: 7+4 input, 3+2 output.
    assert_eq!(run.usage.calls, 2);
    assert_eq!(run.usage.usage.input_tokens, 11);
    assert_eq!(run.usage.usage.output_tokens, 5);
    assert_eq!(run.usage.usage.total_tokens, 16);
}
