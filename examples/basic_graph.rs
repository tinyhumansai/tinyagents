use rustagents::{ChatMessage, Node, NodeOutput, Result, StateGraph};

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<ChatMessage>,
    needs_tool: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let graph = StateGraph::new()
        .add_node(Node::new("agent", |mut state: AgentState| async move {
            state
                .messages
                .push(ChatMessage::assistant("I should check the local tool."));

            if state.needs_tool {
                Ok(NodeOutput::route(state, "tool"))
            } else {
                Ok(NodeOutput::end(state))
            }
        }))
        .add_node(Node::new("tool", |mut state: AgentState| async move {
            state.messages.push(ChatMessage::tool(
                "echo",
                "tool result: hello from RustAgents",
            ));
            state.needs_tool = false;
            Ok(NodeOutput::continue_with(state))
        }))
        .set_start("agent")
        .add_conditional_edges("agent", [("tool", "tool")])
        .add_edge("tool", "agent");

    let run = graph
        .run(AgentState {
            messages: vec![ChatMessage::user("Can you use a tool?")],
            needs_tool: true,
        })
        .await?;

    for message in run.state.messages {
        println!("{:?}: {}", message.role, message.content);
    }

    Ok(())
}
