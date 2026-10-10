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
  caching, and a Claude Code CLI model adapter with stream-json,
  session, authentication, and MCP endpoint support. Features: `sqlite`,
  `storage-drivers`, `builtin-tools` (`tools` is a deprecated alias),
  `multimodal`, `png-optimize` (opt-in PNG re-compression), `media`,
  `claude-code` and `langfuse` (both default), `tracing` (forwards to `tinytools-agent/tracing` for its parsing diagnostics; the harness's own instrumentation is always compiled in).
- **`tinyagents-graph`** — a LangGraph-style durable, typed state graph:
  `START`/`END`, nodes, conditional edges, `Send` fanout, reducers/channels,
  checkpoints, interrupts, subgraphs, and time travel. Features: `sqlite`,
  `tracing`.
- **`tinyagents-tasks`** — graph-independent detached-task machinery: task
  model, `TaskStore` (in-memory and JSONL), `DetachedTaskRegistry`, restart
  reconciliation, the durable completion router (push a finished child to its
  parent, redelivered after a restart), and the `spawn`/`await`/`cancel`/`steer`
  orchestration tools.
  Re-exported from `tinyagents_graph::orchestration` for compatibility.
- **`tinyagents-registry`** — a named capability catalog (models, tools,
  agents, graphs, and routers), plus an offline model price/capability catalog.
- **`tinyagents-session`** — a SQLite-backed store for session history,
  messages, tool calls, cost, and run lineage, plus JSONL transcripts and the
  `threads` chat thread/message store. Transcript spend separates charged and
  estimated costs from unknown or legacy usage; see the
  [cost provenance contract](crates/tinyagents-session/src/transcript/README.md#spend-and-cost-provenance-spendrs).
- **`tinyagents-definition`** — the host-owned agent definition vocabulary:
  identity, description, declared model/tools/delegates, and a read-only
  catalogue seam. Authorization, prompt construction, and execution stay with
  the host and harness.
- **`tinyagents-runtime`** — host-neutral stateful turns over the harness and
  append-only transcript seam; hosts retain policy, prompt composition,
  authorization, and durable-dialect conversion.
- **`tinyagents-live`** — live (realtime voice) agent sessions: declares a
  harness's tools to a [`tinyliveagents`](https://github.com/tinyhumansai/tinyliveagents)
  provider (Gemini Live, ElevenLabs Agents, Sarvam) and runs every tool call
  the model makes through the harness's own admission and middleware
  pipeline, so host policy applies to spoken requests too.
- **`tinyagents-orchestration`** — host-neutral composition of durable
  multi-agent work (teams and workflows) over the graph, harness, and session
  layers; depends one-way on those crates and stays host-free.
- **`tinyagents-integration-tests`** — cross-crate tests and the runnable
  examples referenced below (not published, workspace-internal).

## Quick start

None of the crates are published to crates.io (`publish = false` in every
`Cargo.toml`), so add them as git or path dependencies:

```toml
[dependencies]
tinyagents-harness = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-harness" }
tinyagents-graph = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-graph" }
tinyagents-registry = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-registry" }
# The code samples below build `Message` and provider types directly from
# TinyInference, the message/model crate TinyAgents is built on. It is a
# separate git dependency, not re-exported by the crates above.
tinyinference-llm = { git = "https://github.com/tinyhumansai/tinyinference", package = "tinyinference-llm" }
```

A minimal typed graph — a whole-state agent/tool loop (trimmed from
[`examples/basic_graph.rs`](crates/tinyagents-integration-tests/examples/basic_graph.rs)):

```rust
use tinyagents_graph::*;
use tinyinference_llm::message::Message;

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
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::openai::OpenAiModel;

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
usage and cost accounting, retries and limits, response caching, and a testkit
for exercising the loop without a live provider. Memory, workspace lifecycle,
authorization, and persistence policy stay in the host and can wrap a complete
run with `AgentMiddleware`.

## Subagent orchestration

`tinyagents-orchestration` owns child-agent composition. `SubAgentTool` is a
typed parent-context dispatcher: its default `background` mode starts a child
in the background and returns a stable job id immediately, while `inline` mode
waits for and returns the child's final result in the same call. `SubAgentJobsTool` queries job
status/results and `SubAgentMessageTool` sends messages to a live job;
hosts register these control tools over the same explicitly shared
`SubAgentJobRegistry`. `SubAgentSession` covers retained post-completion
conversations, while `SubagentDriver` coordinates durable lifecycle
preparation, execution, pause, resume, and persistence. Teams and workflow DAGs
are intentionally outside this focused crate.

## Session runtime

`tinyagents-runtime` owns mutable model history for one host-owned
conversation, a stable prompt prefix, a frozen tool declaration snapshot, and
the sequencing around one append-only transcript commit. A host supplies the
driver, its lossless transcript codec, and lifecycle hooks. On a driver error,
the runtime can commit recoverable logical history with an interrupted,
display-only partial in the same history operation; model-context replay omits
that partial. A codec can also derive `TurnUsage` from its explicit host
context after the driver runs; that usage is attached to the same atomic
append's final assistant row for both success and recoverable partials. A
post-commit hook observes durable successes but cannot change their result.
See [the runtime module](docs/modules/runtime/README.md).

## Registry

`tinyagents-registry` is a name-addressable catalog of models, tools, agents,
graphs, and routers. Application code resolves capabilities by name against it
rather than holding direct handles.

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

- **`basic_graph`**, **`complex_graph`**, **`durable_graph`**,
  **`resilient_graph`** — a minimal typed graph, then conditional
  routing/fanout, checkpoint/resume/time-travel, and node-level retry.
- **`agent_loop_tools`** — the agent/tool loop the harness runs.
- **`orchestrator_subagents`** — an orchestrator agent that resolves and calls
  sub-agents by name from the registry.
- **`goals_and_todos`** — a durable goal driving a todo checklist on one
  thread.
- **`openai_chat`**, **`openai_tools`**, **`openai_structured`**,
  **`openai_graph_agent`** — provider-backed chat, tool calling, structured
  output, and a graph-driven agent (all need `OPENAI_API_KEY`).
- **`subconscious_loop`** — an offline, testable autonomous closed-loop
  harness (see its own
  [README](crates/tinyagents-integration-tests/examples/subconscious_loop/README.md)).

## Documentation

- [`docs/spec/README.md`](docs/spec/README.md) — architecture specification.
- [Wiki](https://github.com/tinyhumansai/tinyagents/wiki) — Harness, Graph
  Runtime, Registry, Providers, Quick Start,
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
