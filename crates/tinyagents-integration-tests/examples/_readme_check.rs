use std::sync::Arc;
use tinyagents_graph::*;
use tinyagents_harness::runtime::AgentHarness;
use tinyinference::message::Message;
use tinyinference::providers::openai::OpenAiModel;

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<Message>,
    needs_tool: bool,
}

#[allow(dead_code)]
async fn graph_snippet() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let graph = GraphBuilder::<AgentState, AgentState>::overwrite()
        .add_node("agent", |mut state: AgentState, _ctx: NodeContext| async move {
            state.messages.push(Message::assistant("checking the local tool"));
            Ok(NodeResult::Update(state))
        })
        .add_node("tool", |mut state: AgentState, _ctx: NodeContext| async move {
            state.messages.push(Message::tool("echo", "tool result"));
            state.needs_tool = false;
            Ok(NodeResult::Update(state))
        })
        .set_entry("agent")
        .add_conditional_edges(
            "agent",
            |state: &AgentState| if state.needs_tool { "tool".to_string() } else { "done".to_string() },
            [("tool", "tool"), ("done", END)],
        )
        .add_edge("tool", "agent")
        .compile()?;

    let _run = graph.run(AgentState { messages: vec![], needs_tool: true }).await?;
    Ok(())
}

#[allow(dead_code)]
async fn harness_snippet() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let model = OpenAiModel::from_env()?;
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("openai", Arc::new(model)).set_default_model("openai");

    let run = harness
        .invoke_default(&(), vec![Message::user("What is a Rust trait?")])
        .await?;
    println!("{}", run.text().unwrap_or_default());
    Ok(())
}

fn main() {}
