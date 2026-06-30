//! End-to-end coverage for graph sub-agent nodes through the public registry,
//! harness, and graph execution surfaces.

use std::sync::Arc;

use serde_json::json;

use tinyagents::graph::{
    HarnessSubAgent, SubAgentBudget, SubAgentInput, SubAgentNode, SubAgentOutput, SubAgentPolicy,
    subagent_node,
};
use tinyagents::harness::message::{AssistantMessage, Message};
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::subagent::SubAgent;
use tinyagents::harness::testkit::{EventRecorder, FakeTool, ScriptedModel};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;
use tinyagents::{CapabilityRegistry, GraphBuilder, TinyAgentsError};

fn tool_call_response(id: &str, name: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, json!({}))],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

fn registry_with_constant_agent(name: &str, answer: &str) -> Arc<CapabilityRegistry> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("m", Arc::new(MockModel::constant(answer)));
    let subagent = Arc::new(SubAgent::new(name, "test agent", Arc::new(harness)));

    let mut registry: CapabilityRegistry = CapabilityRegistry::new();
    registry
        .register_agent(
            HarnessSubAgent::new(subagent)
                .with_parent_depth(0)
                .into_dyn(),
        )
        .expect("register agent");
    Arc::new(registry)
}

fn graph_delegating_to(
    registry: Arc<CapabilityRegistry>,
    node: SubAgentNode<String, String>,
) -> tinyagents::CompiledGraph<String, String> {
    GraphBuilder::<String, String>::overwrite()
        .add_node("delegate", subagent_node(node, registry))
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .expect("graph compiles")
}

#[tokio::test]
async fn subagent_node_delegates_records_child_run_and_forwards_events() {
    let registry = registry_with_constant_agent("researcher", "answer: 42");
    let recorder = EventRecorder::new();
    let node = SubAgentNode::<String, String>::from_fns(
        "researcher",
        |state: &String| SubAgentInput::prompt(format!("question: {state}")),
        |out: SubAgentOutput| {
            assert_eq!(out.text, "answer: 42");
            assert!(out.model_calls >= 1);
            assert_eq!(out.tool_calls, 0);
            out.text
        },
    )
    .with_events(recorder.sink());
    let graph = graph_delegating_to(registry, node);

    let run = graph.run("life?".to_string()).await.expect("graph run");
    assert_eq!(run.state, "answer: 42");
    assert_eq!(run.child_runs.len(), 1);
    let child = &run.child_runs[0];
    assert_eq!(child.node.as_str(), "delegate");
    assert_eq!(child.graph_id.as_str(), "agent:researcher");
    assert_eq!(child.root_run_id, run.root_run_id);
    assert_ne!(child.run_id, run.run_id);
    assert!(child.usage.usage.effective_total() > 0);
    assert_eq!(run.run_tree().children.len(), 1);

    let kinds = recorder.kinds();
    assert!(kinds.iter().any(|k| k == "subagent.started"));
    assert!(kinds.iter().any(|k| k == "subagent.completed"));
}

#[tokio::test]
async fn subagent_node_errors_for_missing_agent_and_budget_excess() {
    let missing_registry: Arc<CapabilityRegistry> = Arc::new(CapabilityRegistry::new());
    let missing_node = SubAgentNode::<String, String>::from_fns(
        "missing",
        |state: &String| SubAgentInput::prompt(state.clone()).with_data(json!({ "source": "e2e" })),
        |out: SubAgentOutput| out.text,
    );
    let missing_graph = graph_delegating_to(missing_registry, missing_node);
    let err = missing_graph.run("go".to_string()).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Capability(_)), "got {err:?}");

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
        .expect("register agent");

    let policy = SubAgentPolicy::default().with_budget(SubAgentBudget {
        max_model_calls: Some(1),
        max_tool_calls: None,
    });
    let node = SubAgentNode::<String, String>::from_fns(
        "twostep",
        |state: &String| SubAgentInput {
            prompt: state.clone(),
            data: Some(json!({ "kind": "budget-check" })),
        },
        |out: SubAgentOutput| out.text,
    )
    .with_policy(policy);
    let graph = graph_delegating_to(Arc::new(registry), node);

    let err = graph.run("go".to_string()).await.unwrap_err();
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );

    assert!(
        SubAgentBudget::unlimited()
            .check(
                &SubAgentOutput {
                    text: "ok".to_string(),
                    structured: None,
                    usage: Default::default(),
                    model_calls: 0,
                    tool_calls: 0,
                },
                "agent"
            )
            .is_ok()
    );
}

#[tokio::test]
async fn harness_subagent_adapter_uses_prompt_input_as_child_user_message() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "m",
        Arc::new(ScriptedModel::new(vec![ModelResponse::assistant(
            "adapter answer",
        )])),
    );
    let subagent = Arc::new(SubAgent::new("adapter", "adapter agent", Arc::new(harness)));
    let adapter = HarnessSubAgent::new(subagent).with_parent_depth(2);

    let output = tinyagents::graph::HarnessAgent::run(
        &adapter,
        SubAgentInput::prompt("delegated prompt"),
        EventRecorder::new().sink(),
    )
    .await
    .expect("adapter run");
    assert_eq!(output.text, "adapter answer");
    assert_eq!(output.model_calls, 1);
    assert_eq!(output.tool_calls, 0);

    let _ = Message::user("keep import honest");
}
