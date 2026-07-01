//! End-to-end coverage for two harness control surfaces:
//!
//! - **Part A — [`MiddlewareControl`]**: a lifecycle middleware requests a
//!   control outcome on the [`RunContext`] in `after_model`, and the agent loop
//!   honors it at its safe checkpoint after each model response. `StopWithFinal`
//!   ends the run with a fixed final response (no further tools run) while still
//!   emitting a `run.completed` event; `Interrupt` surfaces as
//!   [`TinyAgentsError::Interrupted`].
//! - **Part B — [`SteeringRegistry`] bridge**: the model-visible
//!   `orchestrate_steer` tool looks a live task up in a [`SteeringRegistry`] and
//!   delivers a real [`SteeringCommand`] into the run executing that task. When
//!   no handle is registered (or the tool has no registry) the request is
//!   recorded but not delivered (`accepted = false`); unknown commands fail
//!   validation.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents::graph::orchestration::{
    InMemoryTaskStore, OrchestrationTaskKind, OrchestrationTaskSpec, OrchestrationTool,
    OrchestrationToolKind, SteeringRegistry, TaskStore,
};
use tinyagents::harness::context::{MiddlewareControl, RunConfig, RunContext};
use tinyagents::harness::ids::TaskId;
use tinyagents::harness::message::Message;
use tinyagents::harness::middleware::Middleware;
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::steering::{SteeringCommand, SteeringHandle};
use tinyagents::harness::testkit::{EventRecorder, FakeTool};
use tinyagents::harness::tool::{Tool, ToolCall};
use tinyagents::{Result, TinyAgentsError};

// ── Part A: MiddlewareControl ────────────────────────────────────────────────

/// Middleware that stops the run with a fixed final response after the model
/// call, exercising [`MiddlewareControl::StopWithFinal`].
struct StopControlMiddleware;

#[async_trait]
impl Middleware<(), ()> for StopControlMiddleware {
    fn name(&self) -> &str {
        "stop_control"
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> Result<()> {
        ctx.request_control(MiddlewareControl::StopWithFinal("stopped".into()));
        Ok(())
    }
}

/// Middleware that requests an interrupt after the model call, exercising
/// [`MiddlewareControl::Interrupt`].
struct InterruptControlMiddleware;

#[async_trait]
impl Middleware<(), ()> for InterruptControlMiddleware {
    fn name(&self) -> &str {
        "interrupt_control"
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> Result<()> {
        ctx.request_control(MiddlewareControl::Interrupt {
            node: "review".into(),
            message: "needs approval".into(),
        });
        Ok(())
    }
}

#[tokio::test]
async fn stop_with_final_control_ends_run_before_tools() {
    // The model asks for a tool it would normally execute and loop on, but the
    // control middleware stops the run at the checkpoint after the model call.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("lookup", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "out")));
    harness.push_middleware(Arc::new(StopControlMiddleware));

    // Capture events by driving the run inside a context whose sink the recorder
    // observes.
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("stop-control"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("control stop yields a completed run");

    assert_eq!(
        run.final_response.expect("final response present").text(),
        "stopped",
        "StopWithFinal must set the fixed final text"
    );
    assert_eq!(
        run.tool_calls, 0,
        "the tool must never run because the loop stopped at the checkpoint"
    );
    assert!(
        recorder.kinds().iter().any(|k| k == "run.completed"),
        "a StopWithFinal run must still emit a run.completed event, got {:?}",
        recorder.kinds()
    );
    assert!(
        recorder.kinds().iter().any(|k| k == "control.applied"),
        "honoring a control outcome must be journaled as control.applied, got {:?}",
        recorder.kinds()
    );
}

#[tokio::test]
async fn interrupt_control_surfaces_as_interrupted_error() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hi")));
    harness.push_middleware(Arc::new(InterruptControlMiddleware));

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("Interrupt control must surface as an error");

    match err {
        TinyAgentsError::Interrupted { node, message } => {
            assert_eq!(node, "review");
            assert_eq!(message, "needs approval");
        }
        other => panic!("expected Interrupted, got {other:?}"),
    }
}

// ── Part B: SteeringRegistry bridge ──────────────────────────────────────────

/// Builds a store holding one live (running) sub-agent task and returns the
/// store together with the task id.
fn live_task() -> (Arc<dyn TaskStore>, TaskId) {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let id = TaskId::new("child-1");
    store
        .insert(OrchestrationTaskSpec::new(
            id.as_str(),
            OrchestrationTaskKind::SubAgent {
                agent: "worker".into(),
            },
        ))
        .expect("insert task spec");
    store.mark_running(&id).expect("mark task running");
    (store, id)
}

/// Calls the steer tool for `task_id` with `command` (and optional extra args),
/// returning the `accepted` flag from the tool result payload.
async fn steer_accepted(
    tool: &OrchestrationTool,
    task_id: &TaskId,
    args: serde_json::Value,
) -> bool {
    let call = ToolCall::new(
        "call-steer",
        "orchestrate_steer",
        args_with_task(task_id, args),
    );
    let result = Tool::<()>::call(tool, &(), call)
        .await
        .expect("steer tool call succeeds");
    result
        .raw
        .and_then(|raw| raw.get("accepted").and_then(|v| v.as_bool()))
        .unwrap_or_else(|| panic!("steer result missing `accepted` boolean"))
}

/// Merges the task id into the caller-supplied steer arguments.
fn args_with_task(task_id: &TaskId, mut args: serde_json::Value) -> serde_json::Value {
    let obj = args.as_object_mut().expect("steer args must be an object");
    obj.insert("task_id".into(), json!(task_id.as_str()));
    args
}

#[tokio::test]
async fn steer_delivers_commands_to_registered_handle() {
    let (store, id) = live_task();
    let handle = SteeringHandle::allow_all();

    let registry = SteeringRegistry::new();
    registry.register(id.clone(), handle.clone());

    let tool =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store).with_steering(registry.clone());

    // `pause` and `cancel` flow through the model-visible tool schema and are
    // delivered to the live handle looked up by task id.
    assert!(
        steer_accepted(&tool, &id, json!({"command": "pause"})).await,
        "pause must be delivered to the live registered handle"
    );
    assert!(
        steer_accepted(&tool, &id, json!({"command": "cancel"})).await,
        "cancel must be delivered to the live registered handle"
    );

    // The `orchestrate_steer` schema is `additionalProperties: false` and only
    // permits `task_id`/`command`/`payload`, so a `redirect` cannot carry its
    // top-level `instruction` through the tool. Exercise the Redirect variant of
    // the bridge via the same live handle the registry hands back from `get`.
    let live = registry.get(&id).expect("registry returns the live handle");
    live.send(SteeringCommand::Redirect {
        instruction: "go north".into(),
    });

    let delivered = handle.drain();
    assert_eq!(
        delivered,
        vec![
            SteeringCommand::Pause,
            SteeringCommand::Cancel,
            SteeringCommand::Redirect {
                instruction: "go north".into()
            },
        ],
        "the handle must receive each delivered command in order"
    );
}

#[tokio::test]
async fn steer_without_registered_handle_is_not_accepted() {
    let (store, id) = live_task();

    // A registry exists but has no handle registered for this task.
    let registry = SteeringRegistry::new();
    let tool_with_empty_registry =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone()).with_steering(registry);
    assert!(
        !steer_accepted(&tool_with_empty_registry, &id, json!({"command": "pause"})).await,
        "with no handle registered the steer request must not be accepted"
    );

    // No registry attached to the tool at all.
    let tool_without_registry = OrchestrationTool::new(OrchestrationToolKind::Steer, store);
    assert!(
        !steer_accepted(&tool_without_registry, &id, json!({"command": "pause"})).await,
        "without a SteeringRegistry the steer request must not be accepted"
    );
}

#[tokio::test]
async fn steer_rejects_unknown_command_with_validation_error() {
    let (store, id) = live_task();
    let handle = SteeringHandle::allow_all();
    let registry = SteeringRegistry::new();
    registry.register(id.clone(), handle);
    let tool = OrchestrationTool::new(OrchestrationToolKind::Steer, store).with_steering(registry);

    let call = ToolCall::new(
        "call-steer",
        "orchestrate_steer",
        json!({"task_id": id.as_str(), "command": "explode"}),
    );
    let err = Tool::<()>::call(&tool, &(), call)
        .await
        .expect_err("an unknown steering command must error");

    assert!(
        matches!(err, TinyAgentsError::Validation(_)),
        "unknown steering command must surface as a validation error, got {err:?}"
    );
}

#[tokio::test]
async fn preloaded_cancel_steering_cancels_a_real_run() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("done")));

    // Attach a steering handle already carrying a Cancel command; the agent loop
    // drains it at the checkpoint before the first model call and ends the run.
    let steering = SteeringHandle::allow_all();
    steering.send(SteeringCommand::Cancel);
    let ctx = RunContext::new(RunConfig::new("cancel-steer"), ()).with_steering(steering);

    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect_err("a preloaded Cancel steering command must cancel the run");

    assert!(
        matches!(err, TinyAgentsError::Cancelled),
        "steered Cancel must surface as Cancelled, got {err:?}"
    );
}
