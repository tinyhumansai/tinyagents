//! Unit tests for [`SubAgentNode`](super::SubAgentNode): a graph node that
//! delegates to a harness agent resolved by name from a
//! [`CapabilityRegistry`](crate::registry::CapabilityRegistry).
//!
//! These exercise:
//! - a graph node delegating to a registered agent and mapping its answer back
//!   into parent state,
//! - the child run being recorded as a distinct, root-preserving child on the
//!   parent execution rollup (with rolled-up usage),
//! - child harness events forwarded onto the node's event sink,
//! - resolving an unregistered agent failing with `Capability`,
//! - the work budget tripping `LimitExceeded`.

use std::sync::Arc;

use super::*;
use crate::graph::builder::GraphBuilder;
use crate::harness::message::AssistantMessage;
use crate::harness::model::ModelResponse;
use crate::harness::providers::MockModel;
use crate::harness::runtime::AgentHarness;
use crate::harness::subagent::SubAgent;
use crate::harness::testkit::{EventRecorder, FakeTool, ScriptedModel};
use crate::harness::tool::ToolCall;
use crate::harness::usage::Usage;
use crate::registry::CapabilityRegistry;

/// Builds a tool-call assistant response (no text, one tool call).
fn tool_call_response(id: &str, name: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, serde_json::json!({}))],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

/// Builds a registry holding one agent named `name` whose model always answers
/// with `answer`.
fn registry_with_agent(name: &str, answer: &str) -> Arc<CapabilityRegistry> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("m", Arc::new(MockModel::constant(answer)));
    let subagent = Arc::new(SubAgent::new(name, "test agent", Arc::new(harness)));

    let mut registry: CapabilityRegistry = CapabilityRegistry::new();
    registry
        .register_agent(HarnessSubAgent::new(subagent).into_dyn())
        .unwrap();
    Arc::new(registry)
}

/// A `String`-state graph with one sub-agent node delegating to `agent`.
fn graph_delegating_to(
    agent: &str,
    registry: Arc<CapabilityRegistry>,
    node: SubAgentNode<String, String>,
) -> crate::graph::CompiledGraph<String, String> {
    let _ = agent;
    GraphBuilder::<String, String>::overwrite()
        .add_node("delegate", subagent_node(node, registry))
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn delegates_to_registered_agent_and_maps_output() {
    let registry = registry_with_agent("researcher", "the answer is 42");
    let node = SubAgentNode::<String, String>::from_fns(
        "researcher",
        |s: &String| SubAgentInput::prompt(s.clone()),
        |out: SubAgentOutput| out.text,
    );
    let graph = graph_delegating_to("researcher", registry, node);

    let run = graph.run("what is the answer?".to_string()).await.unwrap();
    assert_eq!(run.state, "the answer is 42");
}

#[tokio::test]
async fn records_distinct_root_preserving_child_run_with_usage() {
    let registry = registry_with_agent("researcher", "hello");
    let node = SubAgentNode::<String, String>::from_fns(
        "researcher",
        |s: &String| SubAgentInput::prompt(s.clone()),
        |out: SubAgentOutput| out.text,
    );
    let graph = graph_delegating_to("researcher", registry, node);

    let run = graph.run("hi".to_string()).await.unwrap();

    assert_eq!(run.child_runs.len(), 1);
    let child = &run.child_runs[0];
    // Distinct child run id, sharing the parent run's root.
    assert_ne!(child.run_id, run.run_id);
    assert_eq!(child.root_run_id, run.root_run_id);
    assert_eq!(child.node.as_str(), "delegate");
    assert_eq!(child.graph_id.as_str(), "agent:researcher");
    // The child's usage was rolled up onto the parent execution.
    assert!(child.usage.usage.effective_total() > 0);
    // The run tree exposes the same lineage.
    assert_eq!(run.run_tree().children.len(), 1);
}

#[tokio::test]
async fn forwards_child_events_onto_node_sink() {
    let registry = registry_with_agent("worker", "done");
    let recorder = EventRecorder::new();
    let node = SubAgentNode::<String, String>::from_fns(
        "worker",
        |s: &String| SubAgentInput::prompt(s.clone()),
        |out: SubAgentOutput| out.text,
    )
    .with_events(recorder.sink());
    let graph = graph_delegating_to("worker", registry, node);

    graph.run("go".to_string()).await.unwrap();

    let kinds = recorder.kinds();
    assert!(
        kinds.iter().any(|k| k == "subagent.started"),
        "expected subagent.started, got {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| k == "subagent.completed"),
        "expected subagent.completed, got {kinds:?}"
    );
}

#[tokio::test]
async fn unregistered_agent_fails_with_capability() {
    let registry: Arc<CapabilityRegistry> = Arc::new(CapabilityRegistry::new());
    let node = SubAgentNode::<String, String>::from_fns(
        "missing",
        |s: &String| SubAgentInput::prompt(s.clone()),
        |out: SubAgentOutput| out.text,
    );
    let graph = graph_delegating_to("missing", registry, node);

    let err = graph.run("x".to_string()).await.unwrap_err();
    assert!(
        matches!(err, crate::TinyAgentsError::Capability(_)),
        "expected Capability error, got {err:?}"
    );
}

#[tokio::test]
async fn budget_trips_limit_exceeded() {
    // A child agent that makes a tool call then answers needs two model calls;
    // a 1-call model budget trips the node.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "m",
        Arc::new(ScriptedModel::new(vec![
            tool_call_response("c1", "noop"),
            ModelResponse::assistant("final"),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::returning("noop", "ok")));
    let subagent = Arc::new(SubAgent::new(
        "twostep",
        "two-step agent",
        Arc::new(harness),
    ));

    let mut registry: CapabilityRegistry = CapabilityRegistry::new();
    registry
        .register_agent(HarnessSubAgent::new(subagent).into_dyn())
        .unwrap();

    let policy = SubAgentPolicy::default().with_budget(SubAgentBudget {
        max_model_calls: Some(1),
        max_tool_calls: None,
    });
    let node = SubAgentNode::<String, String>::from_fns(
        "twostep",
        |s: &String| SubAgentInput::prompt(s.clone()),
        |out: SubAgentOutput| out.text,
    )
    .with_policy(policy);
    let graph = graph_delegating_to("twostep", Arc::new(registry), node);

    let err = graph.run("go".to_string()).await.unwrap_err();
    assert!(
        matches!(err, crate::TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );
}
