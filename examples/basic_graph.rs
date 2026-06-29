//! A minimal durable graph: a whole-state agent/tool loop.
//!
//! Builds a [`GraphBuilder`] over `Update == State` with the overwrite reducer
//! (each node returns the full next state). The `agent` node appends an
//! assistant message and is wired with conditional edges: while `needs_tool` is
//! set the run routes to `tool`, otherwise it ends. The `tool` node appends a
//! tool result, clears the flag, and loops back to `agent`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example basic_graph
//! ```

use tinyagents::graph::END;
use tinyagents::harness::message::Message;
use tinyagents::{GraphBuilder, NodeContext, NodeResult, Result};

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<Message>,
    needs_tool: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let graph = GraphBuilder::<AgentState, AgentState>::overwrite()
        .add_node(
            "agent",
            |mut state: AgentState, _ctx: NodeContext| async move {
                state
                    .messages
                    .push(Message::assistant("I should check the local tool."));
                Ok(NodeResult::Update(state))
            },
        )
        .add_node(
            "tool",
            |mut state: AgentState, _ctx: NodeContext| async move {
                state
                    .messages
                    .push(Message::tool("echo", "tool result: hello from tinyagents"));
                state.needs_tool = false;
                // A whole-state continue: the static edge `tool -> agent` routes us.
                Ok(NodeResult::Update(state))
            },
        )
        .set_entry("agent")
        .add_conditional_edges(
            "agent",
            |state: &AgentState| {
                if state.needs_tool {
                    "tool".to_string()
                } else {
                    "done".to_string()
                }
            },
            [("tool", "tool"), ("done", END)],
        )
        .add_edge("tool", "agent")
        .compile()?;

    let run = graph
        .run(AgentState {
            messages: vec![Message::user("Can you use a tool?")],
            needs_tool: true,
        })
        .await?;

    for message in &run.state.messages {
        let role = match message {
            Message::System(_) => "system",
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::Tool(_) => "tool",
        };
        println!("{role}: {}", message.text());
    }
    println!("visited: {:?}", run.visited);

    Ok(())
}
