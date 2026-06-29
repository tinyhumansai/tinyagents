# RustAgents

RustAgents is an open source, modular agentic harness for Rust.

While building OpenHuman, the TinyHumans team found that Rust did not have an
agent harness that felt comparable to the mature options available in Python
and TypeScript. Python and TypeScript developers could reach for projects such
as LangChain, LangGraph, Pydantic AI, and other orchestration frameworks, but
Rust developers did not have a similarly capable, ergonomic, and extensible
foundation for building agentic systems.

RustAgents takes inspiration from the best ideas across those harnesses and
brings them into a Rust-native design: strongly typed state, async-first model
and tool abstractions, explicit graph execution, and a small public API that can
grow without hiding the runtime behavior.

RustAgents is a gift from TinyHumans to the Rust community: the harness the team
wanted while building OpenHuman, released openly so Rust developers can build
production agent workflows without having to recreate the same foundation from
scratch.

The crate currently provides:

- chat message primitives
- async chat model and tool traits
- executable state graphs with direct and conditional routing
- examples and tests for the first public API

See [docs/SPEC.md](docs/SPEC.md) for the system specification across the
harness, graph runtime, and expressive language.

## Install

```toml
[dependencies]
rustagents = { path = "." }
```

## Example

```rust
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
            state.messages.push(ChatMessage::assistant("I should use a tool."));

            if state.needs_tool {
                Ok(NodeOutput::route(state, "tool"))
            } else {
                Ok(NodeOutput::end(state))
            }
        }))
        .add_node(Node::new("tool", |mut state: AgentState| async move {
            state.messages.push(ChatMessage::tool("echo", "tool result"));
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

    println!("{:#?}", run.visited);
    Ok(())
}
```

Run the bundled example:

```sh
cargo run --example basic_graph
```

## Core Concepts

`ChatModel<State>` is the provider abstraction. Implement it for OpenAI,
Anthropic, local models, or test doubles.

`Tool<State>` is the tool abstraction. Tools receive immutable access to the
current state plus a structured `ToolCall`.

`StateGraph<State>` is the LangGraph-inspired runtime. Nodes own async handlers
that can continue to the next edge, route through conditional edges, or end the
run with a final state.

## Development

```sh
cargo fmt
cargo test
```
