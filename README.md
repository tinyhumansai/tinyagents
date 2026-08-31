<h1 align="center">TinyAgents</h1>

<p align="center">
 <img src="https://github.com/tinyhumansai/tinyagents/raw/main/docs/readme.png" alt="The Tet" />
</p>

<p align="center">
 <a href="https://github.com/tinyhumansai/tinyagents/actions/workflows/ci.yml"><img src="https://github.com/tinyhumansai/tinyagents/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
 <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3" /></a>
</p>

TinyAgents is a small, provider-neutral agent harness for Rust, plus a durable
typed state-graph runtime. It takes its shape from
[LangChain](https://www.langchain.com/) (models, tools, middleware, structured
output, streaming, usage/cost) and
[LangGraph](https://www.langchain.com/langgraph) (`START`/`END`, nodes,
conditional edges, channels/reducers, checkpoints, interrupts, subgraphs, time
travel) — rebuilt as ordinary, typed Rust with no hidden magic.

It is for Rust services that need to call models and tools in a loop, want
that loop to be resumable and inspectable, and would rather not carry a
Python runtime or a framework's DSL to get there.

## What's inside

TinyAgents is a Cargo workspace, not one crate. Depend on the pieces you need:

- **`tinyagents-harness`** — provider-neutral model calls, typed tools,
  middleware, structured output, streaming, usage/cost accounting, retries,
  caching, and memory. Features: `sqlite`, `tools`, `multimodal`, `tracing`.
- **`tinyagents-graph`** — a LangGraph-style durable, typed state graph:
  `START`/`END`, nodes, conditional edges, `Send` fanout, reducers/channels,
  checkpoints, interrupts, subgraphs, and time travel. Features: `sqlite`,
  `tracing`.
- **`tinyagents-language`** — the `.rag` blueprint format: a declarative,
  side-effect-free workflow description that lexes, parses, and compiles into
  the same graph and harness types as hand-written Rust.
- **`tinyagents-registry`** — a named capability catalog (models, tools,
  agents, graphs, routers) that `.rag` and application code bind against by
  name, plus an offline model price/capability catalog.
- **`tinyagents-session`** — a SQLite-backed store for session history,
  messages, tool calls, cost, and run lineage.
- **`tinyagents-tracing`** — the `tracing` macros the other crates gate behind
  their `tracing` feature. Compiled out by default.
- **`tinyagents-integration-tests`** — cross-crate tests and the runnable
  examples referenced below (not published, workspace-internal).

None of the crates are published to crates.io (`publish = false` in every
`Cargo.toml`); use path or git dependencies.

## Quick start

None of the crates ship on crates.io, so add them as git or path dependencies:

```toml
[dependencies]
tinyagents-harness = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-harness" }
tinyagents-graph = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-graph" }
tinyagents-language = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-language" }
tinyagents-registry = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-registry" }
```

A minimal typed graph — a whole-state agent/tool loop (trimmed from
[`examples/basic_graph.rs`](crates/tinyagents-integration-tests/examples/basic_graph.rs)):

```rust
use tinyagents_graph::*;
use tinyinference::message::Message;

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<Message>,
    needs_tool: bool,
}

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

let run = graph.run(AgentState { messages: vec![], needs_tool: true }).await?;
```

Run it for real:

```sh
git clone git@github.com:tinyhumansai/tinyagents.git
cd tinyagents
cargo run -p tinyagents-integration-tests --example basic_graph
```

A one-shot model call through the harness (`export OPENAI_API_KEY=...` then
`cargo run -p tinyagents-integration-tests --example openai_chat`):

```rust
use std::sync::Arc;
use tinyagents_harness::runtime::AgentHarness;
use tinyinference::message::Message;
use tinyinference::providers::openai::OpenAiModel;

let model = OpenAiModel::from_env()?;
let mut harness: AgentHarness<()> = AgentHarness::new();
harness.register_model("openai", Arc::new(model)).set_default_model("openai");

let run = harness
    .invoke_default(&(), vec![Message::user("What is a Rust trait?")])
    .await?;
println!("{}", run.text().unwrap_or_default());
```

## Graph runtime

`tinyagents-graph` is a durable, typed state graph modeled on LangGraph:
`START`/`END` markers, nodes, static and conditional edges, `Command`-based
routing, `Send` fanout, reducers over named channels, checkpointing (with an
optional `sqlite` backend), interrupts, streaming events, topology export, and
replay/time travel across superstep boundaries. A node can embed another
compiled graph as a subgraph, so a whole workflow can appear as a single step
inside a larger one.

## Harness

`tinyagents-harness` runs the model/tool agent loop: provider-neutral model
calls, typed tool definitions, middleware, structured output, streaming,
usage and cost accounting, retries and limits, response caching, memory, and
a testkit for exercising the loop without a live provider. An agent can be
wrapped as a tool and handed to another agent (`SubAgent` /
`SubAgentSession` / `SubAgentTool`), which is how multi-agent orchestration
is composed — plain function composition, not a distinct execution mode.

## Registry

`tinyagents-registry` is a name-addressable catalog of models, tools, agents,
graphs, and routers. `.rag` blueprints and application code both resolve
capabilities by name against it rather than holding direct handles, which is
what lets a blueprint be validated against exactly the capabilities a host
chose to register.

## `.rag` blueprint language

`tinyagents-language` implements `.rag`: a declarative, side-effect-free
format for describing a graph's state channels, nodes, routes, and named
capability references. It compiles through a fixed pipeline —
`source -> lexer -> tokens -> parser -> AST -> compiler -> Blueprint` — into
the same `tinyagents-graph` and `tinyagents-harness` types produced by
hand-written Rust. It can only reference capabilities by name; it has no way
to embed arbitrary code, so a blueprint is bound and validated against a
registry before it runs. See
[`examples/rag_blueprint.rs`](crates/tinyagents-integration-tests/examples/rag_blueprint.rs).

## Providers

Every provider speaks the OpenAI Chat Completions wire format, so one adapter
reaches all of them; only the base URL and model differ. Built-in presets:
OpenAI, Anthropic (via its OpenAI-compatible endpoint), DeepSeek, Groq, xAI,
OpenRouter, Together, Mistral, and Ollama (local). Any other OpenAI-compatible
endpoint works by base URL — see [`providers.env.example`](providers.env.example)
for the full list and configuration format.

## Examples

All live in
[`crates/tinyagents-integration-tests/examples/`](crates/tinyagents-integration-tests/examples/):

- **`basic_graph`** — a minimal typed state graph: `START`, nodes, edges, `END`.
- **`complex_graph`** — conditional routing, fanout, and richer topology.
- **`durable_graph`** — checkpoints, resume, and time travel over supersteps.
- **`resilient_graph`** — node-level retry over transient failures, with a
  resumable checkpoint.
- **`agent_loop_tools`** — the agent/tool loop the harness runs.
- **`orchestrator_subagents`** — an orchestrator agent that resolves and calls
  sub-agents by name from the registry.
- **`rag_blueprint`** — parse and compile a `.rag` workflow, then bind it
  against a registry.
- **`openai_self_blueprint`** — a model emits a `.rag` blueprint that is
  compiled and run.
- **`goals_and_todos`** — a durable goal driving a task-board kanban on one
  thread.
- **`openai_chat`**, **`openai_tools`**, **`openai_structured`**,
  **`openai_graph_agent`** — provider-backed chat, tool calling, structured
  output, and a graph-driven agent.
- **`subconscious_loop`** — an offline, testable autonomous closed-loop
  harness (see its own
  [README](crates/tinyagents-integration-tests/examples/subconscious_loop/README.md)).

OpenAI-backed examples require `OPENAI_API_KEY` at run time.

## Documentation

- [`docs/spec/README.md`](docs/spec/README.md) — architecture specification.
- [Wiki](https://github.com/tinyhumansai/tinyagents/wiki) — Harness, Graph
  Runtime, Registry, Expressive Language, Providers, Quick Start,
  Examples, Development.

## Development

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --all-targets
cargo test --workspace
```

### Verifying real providers (BYOK)

`cargo test` never touches the network. To check the providers you hold keys
for — a chat call, a streaming call, and a tool call each, reported as a
`provider | PASS/FAIL(reason) | latency(ms)` table:

```sh
cp providers.env.example providers.env   # fill in the keys you have; blank => skipped
PROVIDER_MATRIX=1 cargo test -p tinyagents-integration-tests --test live_provider_matrix -- --nocapture
```

Dialling is opt-in through `PROVIDER_MATRIX=1`, so a bare `cargo test` stays
offline even with a fully configured `providers.env`. `providers.env` is
gitignored — never commit real keys.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

TinyAgents is licensed under [GPL-3.0-only](LICENSE).
