//! Deterministic end-to-end fuzz matrix for graph/agent composition.
//!
//! These tests generate a compact matrix of graph shapes and agent behaviors:
//! whole-state (overwrite) graphs, durable partial-update linear/conditional
//! graphs, command fan-out graphs, and adapter-subgraph graphs. Graph nodes
//! drive real
//! `AgentHarness` runs whose scripted models may make ordinary tool calls,
//! sub-agent tool calls, or both. The assertions are structural: graph visit
//! paths, reducer state, model/tool event counts, and tool-result transcript
//! evidence rather than exact prose.

use std::sync::Arc;

use serde_json::json;

use tinyagents::graph::ClosureStateReducer;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::EventSink;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, FakeTool, ScriptedModel, Trajectory};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;
use tinyagents::{
    Command, GraphBuilder, NodeContext, NodeResult, Result, StateReducer, SubAgent, SubAgentTool,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphShape {
    WholeStateLinear,
    DurableLinear,
    DurableConditional,
    DurableCommandFanout,
    DurableAdapterSubgraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Scenario {
    id: usize,
    shape: GraphShape,
    parallel: bool,
    branch_count: usize,
    use_regular_tool: bool,
    use_subagent: bool,
    route_right: bool,
}

impl Scenario {
    fn label(self) -> String {
        format!(
            "scenario-{}::{:?}::parallel={}::branches={}::regular={}::subagent={}::right={}",
            self.id,
            self.shape,
            self.parallel,
            self.branch_count,
            self.use_regular_tool,
            self.use_subagent,
            self.route_right
        )
    }

    fn expected_tools_per_agent(self) -> usize {
        usize::from(self.use_regular_tool) + usize::from(self.use_subagent)
    }

    fn expected_parent_model_calls_per_agent(self) -> usize {
        if self.expected_tools_per_agent() == 0 {
            1
        } else {
            2
        }
    }

    fn expected_observed_model_calls_per_agent(self) -> usize {
        self.expected_parent_model_calls_per_agent() + usize::from(self.use_subagent)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FuzzState {
    scenario_id: usize,
    path: Vec<String>,
    agent_answers: Vec<String>,
    regular_tool_results: usize,
    subagent_tool_results: usize,
    model_calls: usize,
    tool_calls: usize,
    branch_values: Vec<i32>,
    total: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FuzzUpdate {
    Mark(String),
    Agent {
        node: String,
        answer: String,
        model_calls: usize,
        tool_calls: usize,
        saw_regular_tool: bool,
        saw_subagent_tool: bool,
    },
    Branch {
        node: String,
        value: i32,
    },
    Total(i32),
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            id: 0,
            shape: GraphShape::WholeStateLinear,
            parallel: false,
            branch_count: 0,
            use_regular_tool: false,
            use_subagent: false,
            route_right: false,
        },
        Scenario {
            id: 1,
            shape: GraphShape::WholeStateLinear,
            parallel: false,
            branch_count: 0,
            use_regular_tool: true,
            use_subagent: true,
            route_right: false,
        },
        Scenario {
            id: 2,
            shape: GraphShape::DurableLinear,
            parallel: false,
            branch_count: 0,
            use_regular_tool: true,
            use_subagent: false,
            route_right: false,
        },
        Scenario {
            id: 3,
            shape: GraphShape::DurableLinear,
            parallel: false,
            branch_count: 0,
            use_regular_tool: false,
            use_subagent: true,
            route_right: false,
        },
        Scenario {
            id: 4,
            shape: GraphShape::DurableConditional,
            parallel: false,
            branch_count: 0,
            use_regular_tool: true,
            use_subagent: true,
            route_right: true,
        },
        Scenario {
            id: 5,
            shape: GraphShape::DurableConditional,
            parallel: false,
            branch_count: 0,
            use_regular_tool: false,
            use_subagent: true,
            route_right: false,
        },
        Scenario {
            id: 6,
            shape: GraphShape::DurableCommandFanout,
            parallel: false,
            branch_count: 2,
            use_regular_tool: true,
            use_subagent: false,
            route_right: false,
        },
        Scenario {
            id: 7,
            shape: GraphShape::DurableCommandFanout,
            parallel: true,
            branch_count: 3,
            use_regular_tool: true,
            use_subagent: true,
            route_right: false,
        },
        Scenario {
            id: 8,
            shape: GraphShape::DurableCommandFanout,
            parallel: true,
            branch_count: 4,
            use_regular_tool: false,
            use_subagent: true,
            route_right: false,
        },
        Scenario {
            id: 9,
            shape: GraphShape::DurableAdapterSubgraph,
            parallel: false,
            branch_count: 0,
            use_regular_tool: true,
            use_subagent: false,
            route_right: false,
        },
        Scenario {
            id: 10,
            shape: GraphShape::DurableAdapterSubgraph,
            parallel: false,
            branch_count: 0,
            use_regular_tool: true,
            use_subagent: true,
            route_right: false,
        },
    ]
}

fn reducer()
-> ClosureStateReducer<FuzzState, FuzzUpdate, impl Fn(FuzzState, FuzzUpdate) -> Result<FuzzState>> {
    ClosureStateReducer::new(|mut state: FuzzState, update: FuzzUpdate| {
        match update {
            FuzzUpdate::Mark(node) => state.path.push(node),
            FuzzUpdate::Agent {
                node,
                answer,
                model_calls,
                tool_calls,
                saw_regular_tool,
                saw_subagent_tool,
            } => {
                state.path.push(node);
                state.agent_answers.push(answer);
                state.model_calls += model_calls;
                state.tool_calls += tool_calls;
                if saw_regular_tool {
                    state.regular_tool_results += 1;
                }
                if saw_subagent_tool {
                    state.subagent_tool_results += 1;
                }
            }
            FuzzUpdate::Branch { node, value } => {
                state.path.push(node);
                state.branch_values.push(value);
            }
            FuzzUpdate::Total(total) => {
                state.path.push("aggregate".to_string());
                state.total = Some(total);
            }
        }
        Ok(state)
    })
}

fn tool_call_response(calls: Vec<ToolCall>) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some("tool-turn".to_string()),
            content: Vec::new(),
            tool_calls: calls,
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

fn text_response(text: impl Into<String>) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(5, 2)),
        },
        usage: Some(Usage::new(5, 2)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
    }
}

fn child_harness(answer: String) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("child", Arc::new(ScriptedModel::replies(vec![answer])));
    harness
}

fn parent_harness(scenario: Scenario, node: &str) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();

    let mut calls = Vec::new();
    if scenario.use_regular_tool {
        harness.register_tool(Arc::new(FakeTool::returning(
            "lookup",
            format!("lookup:{}:{node}", scenario.id),
        )));
        calls.push(ToolCall::new(
            format!("{}-{node}-lookup", scenario.id),
            "lookup",
            json!({ "node": node }),
        ));
    }
    if scenario.use_subagent {
        let child = Arc::new(SubAgent::new(
            "delegate",
            "delegate deterministic work to a child agent",
            Arc::new(child_harness(format!("child:{}:{node}", scenario.id))),
        ));
        harness.register_tool(Arc::new(SubAgentTool::new(child)));
        calls.push(ToolCall::new(
            format!("{}-{node}-delegate", scenario.id),
            "delegate",
            json!({ "input": format!("child task for {node}") }),
        ));
    }

    let final_answer = format!("final:{}:{node}", scenario.id);
    let responses = if calls.is_empty() {
        vec![text_response(final_answer)]
    } else {
        vec![tool_call_response(calls), text_response(final_answer)]
    };
    harness.register_model("parent", Arc::new(ScriptedModel::new(responses)));
    harness
}

async fn run_agent_node(
    scenario: Scenario,
    node: &'static str,
    events: EventSink,
) -> Result<FuzzUpdate> {
    let harness = parent_harness(scenario, node);
    let ctx = RunContext::new(RunConfig::new(format!("agent-{}-{node}", scenario.id)), ())
        .with_events(events);

    let run = harness
        .invoke_in_context(
            &(),
            ctx,
            vec![Message::user(format!("run {} in {node}", scenario.label()))],
        )
        .await?;

    let saw_regular_tool = run
        .messages
        .iter()
        .any(|message| matches!(message, Message::Tool(_)) && message.text().contains("lookup:"));
    let saw_subagent_tool = run
        .messages
        .iter()
        .any(|message| matches!(message, Message::Tool(_)) && message.text().contains("child:"));

    Ok(FuzzUpdate::Agent {
        node: node.to_string(),
        answer: run.text().unwrap_or_default(),
        model_calls: run.model_calls,
        tool_calls: run.tool_calls,
        saw_regular_tool,
        saw_subagent_tool,
    })
}

async fn run_whole_state_linear(scenario: Scenario, recorder: &EventRecorder) -> Result<FuzzState> {
    let events = recorder.sink();
    // A whole-state (overwrite) durable graph: each node returns the full next
    // state. The agent node folds its update through the shared reducer itself.
    let graph = GraphBuilder::<FuzzState, FuzzState>::overwrite()
        .add_node(
            "prep",
            |mut state: FuzzState, _ctx: NodeContext| async move {
                state.path.push("prep".to_string());
                Ok(NodeResult::Update(state))
            },
        )
        .add_node("agent", move |state: FuzzState, _ctx: NodeContext| {
            let events = events.clone();
            async move {
                let update = run_agent_node(scenario, "agent", events).await?;
                let state = reducer().apply(state, update)?;
                Ok(NodeResult::Update(state))
            }
        })
        .set_entry("prep")
        .add_edge("prep", "agent")
        .set_finish("agent")
        .compile()?;

    let run = graph
        .run(FuzzState {
            scenario_id: scenario.id,
            ..FuzzState::default()
        })
        .await?;

    assert_visited(&run.visited, &["prep", "agent"], &scenario.label());
    Ok(run.state)
}

async fn run_durable_scenario(scenario: Scenario, recorder: &EventRecorder) -> Result<FuzzState> {
    match scenario.shape {
        GraphShape::DurableLinear => run_durable_linear(scenario, recorder).await,
        GraphShape::DurableConditional => run_durable_conditional(scenario, recorder).await,
        GraphShape::DurableCommandFanout => run_durable_command_fanout(scenario, recorder).await,
        GraphShape::DurableAdapterSubgraph => {
            run_durable_adapter_subgraph(scenario, recorder).await
        }
        GraphShape::WholeStateLinear => {
            unreachable!("whole-state scenarios run via run_whole_state_linear")
        }
    }
}

async fn run_durable_linear(scenario: Scenario, recorder: &EventRecorder) -> Result<FuzzState> {
    let events = recorder.sink();
    let graph = GraphBuilder::<FuzzState, FuzzUpdate>::new()
        .set_reducer(reducer())
        .add_node("prep", |_state: FuzzState, _ctx: NodeContext| async move {
            Ok(NodeResult::Update(FuzzUpdate::Mark("prep".to_string())))
        })
        .add_node("agent", move |_state: FuzzState, _ctx: NodeContext| {
            let events = events.clone();
            async move {
                Ok(NodeResult::Update(
                    run_agent_node(scenario, "agent", events).await?,
                ))
            }
        })
        .set_entry("prep")
        .add_edge("prep", "agent")
        .set_finish("agent")
        .compile()?;

    let run = graph
        .run(FuzzState {
            scenario_id: scenario.id,
            ..FuzzState::default()
        })
        .await?;
    assert_visited(&run.visited, &["prep", "agent"], &scenario.label());
    Ok(run.state)
}

async fn run_durable_conditional(
    scenario: Scenario,
    recorder: &EventRecorder,
) -> Result<FuzzState> {
    let left_events = recorder.sink();
    let right_events = recorder.sink();
    let expected_agent = if scenario.route_right {
        "agent_right"
    } else {
        "agent_left"
    };

    let graph = GraphBuilder::<FuzzState, FuzzUpdate>::new()
        .set_reducer(reducer())
        .add_node("prep", |_state: FuzzState, _ctx: NodeContext| async move {
            Ok(NodeResult::Update(FuzzUpdate::Mark("prep".to_string())))
        })
        .add_node("route", |_state: FuzzState, _ctx: NodeContext| async move {
            Ok(NodeResult::Update(FuzzUpdate::Mark("route".to_string())))
        })
        .add_node("agent_left", move |_state: FuzzState, _ctx: NodeContext| {
            let events = left_events.clone();
            async move {
                Ok(NodeResult::Update(
                    run_agent_node(scenario, "agent_left", events).await?,
                ))
            }
        })
        .add_node(
            "agent_right",
            move |_state: FuzzState, _ctx: NodeContext| {
                let events = right_events.clone();
                async move {
                    Ok(NodeResult::Update(
                        run_agent_node(scenario, "agent_right", events).await?,
                    ))
                }
            },
        )
        .set_entry("prep")
        .add_edge("prep", "route")
        .add_conditional_edges(
            "route",
            move |_state: &FuzzState| {
                if scenario.route_right {
                    "right".to_string()
                } else {
                    "left".to_string()
                }
            },
            [("left", "agent_left"), ("right", "agent_right")],
        )
        .set_finish("agent_left")
        .set_finish("agent_right")
        .compile()?;

    let run = graph
        .run(FuzzState {
            scenario_id: scenario.id,
            ..FuzzState::default()
        })
        .await?;
    assert_visited(
        &run.visited,
        &["prep", "route", expected_agent],
        &scenario.label(),
    );
    Ok(run.state)
}

async fn run_durable_command_fanout(
    scenario: Scenario,
    recorder: &EventRecorder,
) -> Result<FuzzState> {
    let mut builder = GraphBuilder::<FuzzState, FuzzUpdate>::new()
        .with_parallel(scenario.parallel)
        .set_reducer(reducer())
        .add_node(
            "dispatch",
            move |_state: FuzzState, _ctx: NodeContext| async move {
                let targets = (0..scenario.branch_count)
                    .map(|idx| format!("branch_{idx}"))
                    .collect::<Vec<_>>();
                Ok(NodeResult::Command(Command::default().with_goto(targets)))
            },
        )
        .add_node(
            "aggregate",
            |_state: FuzzState, _ctx: NodeContext| async move {
                Ok(NodeResult::Update(FuzzUpdate::Total(100)))
            },
        )
        .set_entry("dispatch")
        .mark_command_routing("dispatch")
        .set_finish("aggregate");

    for idx in 0..scenario.branch_count {
        let node = format!("branch_{idx}");
        let node_for_handler = node.clone();
        let events = recorder.sink();
        builder = builder
            .add_node(node.clone(), move |_state: FuzzState, _ctx: NodeContext| {
                let events = events.clone();
                let node_for_update = node_for_handler.clone();
                async move {
                    if idx == 0 {
                        Ok(NodeResult::Update(
                            run_agent_node(scenario, "branch_0", events).await?,
                        ))
                    } else {
                        Ok(NodeResult::Update(FuzzUpdate::Branch {
                            node: node_for_update,
                            value: idx as i32,
                        }))
                    }
                }
            })
            .add_edge(node, "aggregate");
    }

    let graph = builder.compile()?;
    let run = graph
        .run(FuzzState {
            scenario_id: scenario.id,
            ..FuzzState::default()
        })
        .await?;

    assert!(
        run.visited.iter().any(|node| node.as_str() == "dispatch"),
        "{}",
        scenario.label()
    );
    assert!(
        run.visited.iter().any(|node| node.as_str() == "aggregate"),
        "{}",
        scenario.label()
    );
    assert_eq!(run.state.total, Some(100), "{}", scenario.label());
    Ok(run.state)
}

async fn run_durable_adapter_subgraph(
    scenario: Scenario,
    recorder: &EventRecorder,
) -> Result<FuzzState> {
    let child = GraphBuilder::<FuzzState, FuzzUpdate>::new()
        .set_reducer(reducer())
        .add_node(
            "child_prep",
            |_state: FuzzState, _ctx: NodeContext| async move {
                Ok(NodeResult::Update(FuzzUpdate::Mark(
                    "child_prep".to_string(),
                )))
            },
        )
        .add_node(
            "child_done",
            |_state: FuzzState, _ctx: NodeContext| async move {
                Ok(NodeResult::Update(FuzzUpdate::Branch {
                    node: "child_done".to_string(),
                    value: 7,
                }))
            },
        )
        .set_entry("child_prep")
        .add_edge("child_prep", "child_done")
        .set_finish("child_done")
        .compile()?;

    let events = recorder.sink();
    let graph = GraphBuilder::<FuzzState, FuzzUpdate>::new()
        .set_reducer(reducer())
        .add_node(
            "subgraph",
            tinyagents::graph::subgraph::adapter_subgraph_node(
                child,
                |parent: &FuzzState| parent.clone(),
                |_parent: &FuzzState, child_state: FuzzState| FuzzUpdate::Branch {
                    node: "subgraph".to_string(),
                    value: child_state.branch_values.iter().sum(),
                },
            ),
        )
        .add_node("agent", move |_state: FuzzState, _ctx: NodeContext| {
            let events = events.clone();
            async move {
                Ok(NodeResult::Update(
                    run_agent_node(scenario, "agent", events).await?,
                ))
            }
        })
        .set_entry("subgraph")
        .add_edge("subgraph", "agent")
        .set_finish("agent")
        .compile()?;

    let run = graph
        .run(FuzzState {
            scenario_id: scenario.id,
            ..FuzzState::default()
        })
        .await?;
    assert_visited(&run.visited, &["subgraph", "agent"], &scenario.label());
    Ok(run.state)
}

fn assert_visited(actual: &[tinyagents::harness::ids::NodeId], expected: &[&str], label: &str) {
    let actual = actual.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(actual, expected, "{label}");
}

fn assert_state_invariants(scenario: Scenario, state: &FuzzState) {
    assert_eq!(state.scenario_id, scenario.id, "{}", scenario.label());
    assert!(
        !state.path.is_empty(),
        "graph should commit at least one update for {}",
        scenario.label()
    );

    let expected_agent_runs = match scenario.shape {
        GraphShape::DurableCommandFanout => 1,
        _ => 1,
    };
    assert_eq!(
        state.agent_answers.len(),
        expected_agent_runs,
        "{}",
        scenario.label()
    );
    assert_eq!(
        state.model_calls,
        expected_agent_runs * scenario.expected_parent_model_calls_per_agent(),
        "{}",
        scenario.label()
    );
    assert_eq!(
        state.tool_calls,
        expected_agent_runs * scenario.expected_tools_per_agent(),
        "{}",
        scenario.label()
    );
    assert_eq!(
        state.regular_tool_results,
        expected_agent_runs * usize::from(scenario.use_regular_tool),
        "{}",
        scenario.label()
    );
    assert_eq!(
        state.subagent_tool_results,
        expected_agent_runs * usize::from(scenario.use_subagent),
        "{}",
        scenario.label()
    );

    if matches!(scenario.shape, GraphShape::DurableCommandFanout) {
        assert_eq!(state.total, Some(100), "{}", scenario.label());
        assert_eq!(
            state.branch_values.len(),
            scenario.branch_count.saturating_sub(1),
            "{}",
            scenario.label()
        );
    }

    if matches!(scenario.shape, GraphShape::DurableAdapterSubgraph) {
        assert_eq!(state.branch_values, vec![7], "{}", scenario.label());
    }
}

fn assert_event_invariants(scenario: Scenario, recorder: &EventRecorder) {
    let trajectory = Trajectory::from_events(recorder.events());
    trajectory.assert_completed();
    trajectory.assert_model_called_times(scenario.expected_observed_model_calls_per_agent());
    if scenario.use_regular_tool {
        trajectory.assert_tool_called("lookup");
        assert_eq!(
            trajectory.tool_call_count("lookup"),
            1,
            "{}",
            scenario.label()
        );
    }
    if scenario.use_subagent {
        trajectory.assert_tool_called("delegate");
        assert_eq!(
            trajectory.tool_call_count("delegate"),
            1,
            "{}",
            scenario.label()
        );
    }
}

#[tokio::test]
async fn generated_graph_agent_scenarios_keep_e2e_invariants() {
    for scenario in scenarios() {
        let recorder = EventRecorder::new();
        let state = match scenario.shape {
            GraphShape::WholeStateLinear => run_whole_state_linear(scenario, &recorder).await,
            _ => run_durable_scenario(scenario, &recorder).await,
        }
        .unwrap_or_else(|error| panic!("{} failed: {error}", scenario.label()));

        assert_state_invariants(scenario, &state);
        assert_event_invariants(scenario, &recorder);
    }
}
