<h1 align="center">TinyAgents</h1>

<p align="center">
 <img src="https://github.com/tinyhumansai/tinyagents/raw/main/docs/readme.png" alt="The Tet" />
</p>

<p align="center">
 <a href="https://github.com/tinyhumansai/tinyagents/actions/workflows/ci.yml"><img src="https://github.com/tinyhumansai/tinyagents/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
 <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3" /></a>
</p>

**TinyAgents is a durable agent and graph harness for Rust.** It is a typed,
checkpointed runtime where language models call models, agents call agents,
graphs run graphs, and a model can author, compile, and run the very workflow it
is standing inside — all as inspectable, checkpointed, policy-checked Rust.

## Recursion, without an embedded interpreter

Most agent frameworks stuff everything into one ever-growing context window and
hope the model copes. TinyAgents takes the other stance: a long task is an
external *environment* that gets decomposed, and the runtime is re-entrant, so a
model can recurse over pieces of it instead of swallowing the whole thing at
once. The concrete surfaces:

- **Sub-agents (agents calling agents).** A harness agent is exposed *as a tool*
  to another agent, so orchestration is literally a model calling a model
  (`SubAgent`, `SubAgentSession`, `SubAgentTool`).
- **Recursion policy + depth tracking.** The runtime tracks `root_run_id` /
  `parent_run_id`, enforces a recursion limit, and rolls child runs' events,
  usage, and cost up to the parent as first-class observable runs.
- **Graphs that run graphs.** A node can embed another compiled graph, so a
  whole compiled workflow can appear as a single step inside another one.
- **Self-authoring (the deepest recursion).** A model can emit a `.rag`
  blueprint that compiles through the *same* registry-bound compiler path as a
  human-authored file, then runs on the *same* runtime the model is already
  executing in. The harness can describe and re-enter itself.

One language, one runtime: `.rag` blueprints lower into the exact same `graph` +
`harness` types as hand-written Rust — a language whose programs *are* the
runtime that interprets them.

**Not in this workspace, by design:** the scripted CodeAct/REPL loop — an embedded
interpreter (Rhai, Python, JavaScript) executing model-written code cells whose
only host surface is capability calls. That is a host concern. TinyAgents gives
it everything it needs (the capability `registry`, the harness, typed
`SessionId`/`CellId`/`CallId`, and the `repl_agent` node kind for binding a
host-provided scripted node by name) without pulling an interpreter into your
dependency graph.

## Features

- **Harness** — provider-neutral model calls, typed tools, middleware,
  structured output, streaming, usage/cost accounting, retries and limits,
  response caching, memory/embeddings, summarization, steering, and a testkit.
- **Graph runtime** — LangGraph-style durable, typed state graphs: `START`/`END`,
  nodes, edges, conditional routing, commands, `Send` fanout, reducers/channels,
  checkpoints, interrupts, subgraphs, streaming, topology export, and time
  travel.
- **Registry** — a named capability catalog (models, tools, agents, graphs,
  stores, middleware, policy) that `.rag` binds by name.
- **`.rag` expressive language** — a declarative, side-effect-free blueprint
  format that compiles (lexer → parser → compiler) into the runtime; the safe
  boundary for agent-authored plans.
- **Recursion & sub-agents** — agents-as-tools, subgraphs, depth tracking, and a
  recursion policy so deep call trees stay bounded and observable.
- **Durability & checkpoints** — resume long runs, replay history, and travel
  back in time across superstep boundaries.
- **Provider-neutral** — one interface across hosted and local providers; swap
  models without rewriting workflows.
- **Observability** — normalized events, usage, and cost that roll up across
  recursive child runs.
- **Structured output & streaming** — typed responses and incremental token
  streams at the harness boundary.

## Architecture

```text
                         +-----------------------+
                         |   .rag blueprint      |
                         | declarative workflow  |
                         +-----------+-----------+
                                     |
                                     |  compile / lower (by name)
                                     v
+-------------+        +-------------------------------------------+
| Application |------->| Capability Registry                       |
| Rust code   |        | models | tools | agents | graphs | policy |
+------+------+        +---------------------+---------------------+
       |                                     |
       |                                     v
       |              +-------------------------------------------+
       +------------->| Durable Graph Runtime                     |
                      | typed state | nodes | edges | checkpoints |
                      +---------------------+---------------------+
                                            |
                                            v
                      +-------------------------------------------+
                      | Agent Harness                             |
                      | prompts | tools | middleware | usage/cost |
                      +----+--------------------------+-----------+
                           |                          |
                           v                          v
                 +------------------+        +------------------+
                 | Model Providers  |        | Typed Tools      |
                 | OpenAI/Anthropic |        | local functions  |
                 | Ollama/etc.      |        | external systems |
                 +------------------+        +------------------+
```

The recursion loop — agents call agents, and graphs run graphs:

```text
        +-------+
        | START |
        +---+---+
            |
            v
      +-------------+        a sub-agent is just a tool,
      | Agent Node  |        and a tool may itself be a
      +------+------+        whole compiled graph...
             |
      +------+-------------------------+
      |              |                 |
 needs tool     calls sub-agent    done
      |              |                 |
      v              v                 v
+-----------+  +---------------+    +-----+
| Tool Node |  | SubAgent /    |    | END |
+-----+-----+  | Subgraph Node |    +-----+
      |        +-------+-------+
      |                |  depth +1, recursion policy,
      |                |  child run rolls up usage/cost
      +-- loops back --+--- re-enters the runtime ---+
          to Agent Node     (graph -> subgraph -> graph)
```

## Quick start

Add only the TinyAgents crates your project uses. There is deliberately no
`tinyagents` facade crate, so migrating from the former monolith is a breaking
change:

```toml
[dependencies]
tinyagents-harness = "2.1.1"
tinyagents-graph = "2.1.1"
tinyagents-language = "2.1.1"
tinyagents-registry = "2.1.1"
tinyagents-session = "2.1.1"
```

The OpenAI (and OpenAI-compatible) provider is compiled in by default; the
build stays offline unless you actually make a call. Optional features live on
the crate that owns the behavior: `tinyagents-harness` provides `sqlite`,
`tools`, `multimodal`, and `tracing`; `tinyagents-graph` provides `sqlite` and
`tracing`; and the registry and session crates forward `tracing`. Tracing
instrumentation is disabled unless that feature is selected.

To explore locally:

```sh
git clone git@github.com:tinyhumansai/tinyagents.git
cd tinyagents
cargo run -p tinyagents-integration-tests --example basic_graph
```

OpenAI-backed examples need an API key:

```sh
export OPENAI_API_KEY=...
cargo run -p tinyagents-integration-tests --example openai_chat
```

Bound individual tool calls by installing shared timeout settings on the
harness. Tools return `ToolTimeout::Inherit` by default; they may instead opt
out with `Unbounded` or request a clamped explicit `Millis` budget:

```rust
use tinyagents_harness::{AgentHarness, ToolTimeoutSettings};

let mut harness: AgentHarness<()> = AgentHarness::new();
harness.with_tool_timeout_settings(ToolTimeoutSettings::new(
    120_000, // inherited default
    1_000,   // minimum explicit budget
    3_600_000,
    5_000,   // scheduling grace for explicit budgets
));
```

A per-tool deadline produces a recoverable tool-error message and the agent
loop continues, so the model can retry or choose another tool. The run's
wall-clock deadline remains the outer hard abort. Clones of the settings share
their inherited value, allowing a host to update it without rebuilding a
harness.

Export durable harness observations to Langfuse with the embedded client:

```rust
use tinyagents_harness::{LangfuseClient, LangfuseTraceConfig};

let client = LangfuseClient::proxy("https://api.tinyhumans.ai", backend_jwt)?;
client
    .send_observations(
        LangfuseTraceConfig {
            user_id: Some("user_123".to_string()),
            session_id: Some("thread_abc".to_string()),
            ..Default::default()
        },
        &observations,
    )
    .await?;
```

`LangfuseClient::proxy` sends to the backend
`/telemetry/langfuse/ingestion` endpoint with bearer auth. Use
`LangfuseClient::direct(langfuse_url, public_key, secret_key)` when an
application is allowed to talk to Langfuse directly.

Graph runs export the same way through `GraphLangfuseExporter`, which reuses the
harness `LangfuseClient` transport and turns supersteps and nodes into timed
spans (failures promoted to `ERROR`), with per-node **tool health** telemetry
attached to the trace:

```rust
use tinyagents_graph::GraphLangfuseExporter;
use tinyagents_harness::{LangfuseClient, LangfuseTraceConfig};

let exporter = GraphLangfuseExporter::new(LangfuseClient::from_env()?);
let observations = journal.read_from(run_id, 0).await?;
exporter
    .send_observations(LangfuseTraceConfig::default(), &observations)
    .await?;
```

Because a graph run and the agent runs its nodes spawn share the same
`root_run_id` — the default Langfuse `traceId` for both exporters — exporting a
graph run and its child agents lands every step, node, model generation, and
tool call under one trace for full end-to-end observability.

## Examples to explore

All live in
[`crates/tinyagents-integration-tests/examples/`](crates/tinyagents-integration-tests/examples/):

- **`basic_graph`** — a minimal typed state graph: `START`, nodes, edges, `END`.
- **`complex_graph`** — conditional routing, fanout, and richer topology.
- **`durable_graph`** — checkpoints, resume, and time-travel over supersteps.
- **`resilient_graph`** — node-level retry over transient failures, plus a
  resumable failure checkpoint that `retry` restarts after an outage clears.
- **`agent_loop_tools`** — the agent ↔ tool loop the harness runs.
- **`orchestrator_subagents`** — **recursion in action:** an orchestrator agent
  that calls sub-agents as tools, with depth tracking and rolled-up usage.
- **`openai_self_blueprint`** — **the deepest recursion:** a model authors a
  `.rag` blueprint that is compiled and run on the same runtime.
- **`rag_blueprint`** — load and run a declarative `.rag` workflow.
- **`goals_and_todos`** — a durable `ThreadGoal` driving a `TaskBoard` kanban
  on one thread.
- **`subconscious_loop`** — an offline, testable autonomous closed-loop
  harness (see
  [`examples/subconscious_loop/README.md`](crates/tinyagents-integration-tests/examples/subconscious_loop/README.md)).
- **`openai_chat`** — a single provider-backed chat turn.
- **`openai_tools`** — tool calling against a hosted model.
- **`openai_structured`** — typed structured output.
- **`openai_graph_agent`** — a provider-backed agent driven inside a graph.

OpenAI-backed examples require `OPENAI_API_KEY` at run time.

## Documentation

- [Harness API](https://docs.rs/tinyagents-harness)
- [Graph API](https://docs.rs/tinyagents-graph)
- [Language API](https://docs.rs/tinyagents-language)
- [Registry API](https://docs.rs/tinyagents-registry)
- [Session API](https://docs.rs/tinyagents-session)
- [Wiki home](https://github.com/tinyhumansai/tinyagents/wiki)
  - [Recursion and sub-agents](https://github.com/tinyhumansai/tinyagents/wiki/Recursion-and-RLM)
  - [Harness](https://github.com/tinyhumansai/tinyagents/wiki/Harness)
  - [Graph runtime](https://github.com/tinyhumansai/tinyagents/wiki/Graph-Runtime)
  - [Registry](https://github.com/tinyhumansai/tinyagents/wiki/Registry)
  - [Expressive language `.rag`](https://github.com/tinyhumansai/tinyagents/wiki/Expressive-Language-RAG)
  - [Providers](https://github.com/tinyhumansai/tinyagents/wiki/Providers)
  - [Quick start](https://github.com/tinyhumansai/tinyagents/wiki/Quick-Start)
  - [Examples](https://github.com/tinyhumansai/tinyagents/wiki/Examples)
  - [Development](https://github.com/tinyhumansai/tinyagents/wiki/Development)

Contributors working directly in the repository should also read the checked-in
architecture specification under [`docs/spec/README.md`](docs/spec/README.md).

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
offline even with a fully configured `providers.env`.

`providers.env` is gitignored — never commit real keys. See
[`crates/tinyagents-harness/src/providers/openai/README.md`](crates/tinyagents-harness/src/providers/openai/README.md)
for the configuration format.

## Contributing

TinyAgents welcomes focused contributions that improve the graph runtime,
harness contracts, the registry, the `.rag` language, provider
adapters, tests, examples, and documentation.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

TinyAgents is licensed under [GPL-3.0-only](LICENSE).

Built by TinyHumans for the Rust agent ecosystem.
