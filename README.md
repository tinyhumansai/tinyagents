# TinyAgents

[![CI](https://github.com/tinyhumansai/rustagents/actions/workflows/ci.yml/badge.svg)](https://github.com/tinyhumansai/rustagents/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

RustAgents is a Rust-native framework for building LLM agents as typed,
inspectable workflows.

The project starts from a simple belief: agent systems should not be piles of
callbacks and hidden loops. They should be explicit programs with state, edges,
policies, tools, model calls, checkpoints, and events that you can read, test,
debug, serialize, and run again.

RustAgents brings that shape to Rust. It combines strongly typed application
state, async model and tool traits, executable graph primitives, and a roadmap
for declarative agent workflow languages that LLMs can safely author, inspect,
compile, and run.

## Why TinyAgents

Python and TypeScript have mature agent frameworks. Rust developers deserve the
same level of orchestration power without giving up Rust's clarity, type system,
performance, and production discipline.

RustAgents is designed for teams that want to build real agent products:

- typed state instead of unstructured runtime bags
- explicit graph execution instead of implicit control flow
- model and tool traits that are easy to test
- deterministic routing around nondeterministic LLM calls
- observable runs with room for streaming, checkpoints, interrupts, and replay
- declarative workflow definitions that can be reviewed before execution
- parallel agents, sub-agent fanout, context forking, and blocking child agents
  as first-class workflow concepts

The long-term goal is not just "call an LLM from Rust." The goal is to make
Rust a serious home for durable agent runtimes.

## The Big Idea: Declarative Workflows For LLMs

LLMs are good at proposing plans, but raw generated code is a dangerous
execution boundary. RustAgents is moving toward a safer path: an expressive,
declarative workflow language for graph blueprints.

A `.rag` workflow should describe what an agent system is allowed to do:

- which models, tools, stores, and sub-agents are available
- which nodes exist
- how state flows between nodes
- when work fans out in parallel
- how child agents join back into parent state
- which agents are blocking, optional, racing, or quorum-based
- where checkpoints, interrupts, retries, budgets, and policies apply

That means an LLM can create or modify a workflow without receiving arbitrary
host-code execution. The generated plan becomes source, the source becomes an
AST, the AST is validated against registries and policy, and only then does it
compile into the same graph runtime that hand-written Rust uses.

Conceptually:

```rustagents
graph research_review {
  start supervisor

  channel messages messages
  channel findings append
  channel critiques append

  node supervisor {
    kind agent
    model "default"
    tools ["search", "read_repo"]

    send [
      research_agent { question: input.question },
      verifier_agent { question: input.question }
    ]

    join review
  }

  node review {
    kind sub_agent
    agent "final_reviewer"
    blocking true
    next END
  }
}
```

This is the power RustAgents is built around: agents can author workflows that
spawn other agents, run branches in parallel, call blocking reviewers, fork
context safely, and merge outputs through typed reducers while the runtime keeps
policy and observability intact.

## Current Status

RustAgents is early. The crate currently provides the foundation:

- chat message primitives
- async chat model and tool traits
- executable state graphs with direct and conditional routing
- graph validation, recursion limits, visited-node traces, and examples
- extensive architecture docs for the harness, graph runtime, registry,
  expressive language, and REPL language

The docs describe the target system in more detail than the current crate
implements. That is intentional: the public repository is both a working Rust
library and an open design space for the agent runtime Rust should have.

## Install

Until the crate is published, use the repository directly:

```toml
[dependencies]
rustagents = { git = "https://github.com/tinyhumansai/rustagents" }
```

For local development:

```toml
[dependencies]
rustagents = { path = "." }
```

## Quick Example

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

## Architecture

RustAgents is organized around five major surfaces:

- **Harness**: provider-neutral models, tools, middleware, prompts, context,
  memory, streaming, observability, retries, caching, and test doubles.
- **Graph runtime**: typed state graphs, nodes, routing, reducers, commands,
  parallel execution, checkpointing, interrupts, subgraphs, and sub-agents.
- **Registry**: named models, tools, agents, graphs, stores, middleware, and
  policies that declarative workflows can bind to safely.
- **Expressive language**: `.rag` graph blueprints that are readable by humans,
  authorable by agents, and compiled through the same validation path.
- **REPL language**: `.ragsh` interactive orchestration for inspecting,
  scripting, and controlling harness and graph runs through registered
  capabilities.

Start with the system specification:

- [System specification](docs/spec/README.md)
- [Harness module](docs/modules/harness/README.md)
- [Graph module](docs/modules/graph/README.md)
- [Parallel agents and context forking](docs/modules/graph/parallel-agents-forking.md)
- [Expressive language](docs/modules/expressive-language/README.md)
- [REPL language](docs/modules/repl-language/README.md)
- [Registry module](docs/modules/registry/README.md)

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --all-targets
cargo test
```

Run the example:

```sh
cargo run --example basic_graph
```

## Contributing

RustAgents is open source and welcomes focused contributions. The highest-value
work right now is small, well-tested improvements to the core graph API,
harness traits, docs, examples, and the declarative language design.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

RustAgents is licensed under [GPL-3.0-only](LICENSE).

Built by TinyHumans for the Rust agent ecosystem.
