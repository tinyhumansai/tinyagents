<h1 align="center">TinyAgents</h1>

<p align="center">
 <img src="https://github.com/tinyhumansai/tinyagents/raw/main/docs/readme.png" alt="The Tet" />
</p>

<p align="center">
 <a href="https://crates.io/crates/tinyagents"><img src="https://img.shields.io/crates/v/tinyagents.svg" alt="crates.io" /></a>
 <a href="https://docs.rs/tinyagents"><img src="https://docs.rs/tinyagents/badge.svg" alt="docs.rs" /></a>
 <a href="https://github.com/tinyhumansai/tinyagents/actions/workflows/ci.yml"><img src="https://github.com/tinyhumansai/tinyagents/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
 <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3" /></a>
</p>

**TinyAgents is a recursive language-model (RLM) harness for Rust.** It is a
typed, durable runtime where language models call models, agents call agents,
graphs run graphs, and a model can author, compile, and run the very workflow it
is standing inside — all as inspectable, checkpointed, policy-checked Rust.

## What is an RLM, and why recursive?

Most agent frameworks stuff everything into one ever-growing context window and
hope the model copes. **Recursive Language Models (RLMs)** take a different
stance: a long prompt is treated as an external *environment* that the model
explores through a REPL — examining it, decomposing it, and **recursively calling
itself (or sub-models) over snippets** instead of swallowing the whole thing at
once. This mitigates "context rot" and lets effective context exceed the raw
window.

The idea comes from recent research:

- **Paper:** "Recursive Language Models," Alex L. Zhang, Tim Kraska, Omar Khattab
  (MIT CSAIL), 2025 — [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- **Blog:** Alex L. Zhang, "Recursive Language Models" —
  <https://alexzhang13.github.io/blog/2025/rlm/>
- **Reference implementation:** <https://github.com/alexzhang13/rlm>

TinyAgents is **inspired by and architected around** the RLM execution model — a
production-shaped Rust harness for building RLM-style systems. It does not claim
to reproduce the paper's benchmark numbers; instead it brings the *execution
model* to Rust as concrete, implemented surfaces:

- **Sub-agents (agents calling agents).** A harness agent is exposed *as a tool*
  to another agent, so orchestration is literally a model calling a model
  (`SubAgent`, `SubAgentSession`, `SubAgentTool`).
- **Recursion policy + depth tracking.** The runtime tracks `root_run_id` /
  `parent_run_id`, enforces a recursion limit, and rolls child runs' events,
  usage, and cost up to the parent as first-class observable runs.
- **Graphs that run graphs.** A node can embed another compiled graph, and the
  `.ragsh` REPL can drive a graph from inside a graph node (graph → REPL →
  graph).
- **The REPL as the RLM core.** In `.ragsh`, context and prompts are runtime
  *values*, not just prompt text. The model writes small programs, inspects their
  output, calls sub-models / sub-agents / sub-graphs as functions, and iterates —
  the RLM/CodeAct loop.
- **Self-authoring (the deepest recursion).** A model can emit a `.rag`
  blueprint that compiles through the *same* registry-bound compiler path as a
  human-authored file, then runs on the *same* runtime the model is already
  executing in. The harness can describe and re-enter itself.

Two languages, one runtime: `.rag` (declarative blueprint) and `.ragsh`
(imperative REPL) both lower into the exact same `graph` + `harness` types as
hand-written Rust — a language whose programs *are* the runtime that interprets
them.

## Features

- **Harness** — provider-neutral model calls, typed tools, middleware,
  structured output, streaming, usage/cost accounting, retries and limits,
  response caching, memory/embeddings, summarization, steering, and a testkit.
- **Graph runtime** — LangGraph-style durable, typed state graphs: `START`/`END`,
  nodes, edges, conditional routing, commands, `Send` fanout, reducers/channels,
  checkpoints, interrupts, subgraphs, streaming, topology export, and time
  travel.
- **Registry** — a named capability catalog (models, tools, agents, graphs,
  stores, middleware, policy) that `.rag` and `.ragsh` bind by name.
- **`.rag` expressive language** — a declarative, side-effect-free blueprint
  format that compiles (lexer → parser → compiler) into the runtime; the safe
  boundary for agent-authored plans.
- **`.ragsh` REPL language** — imperative, capability-bound interactive
  orchestration; the RLM/CodeAct loop surface.
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
            +-----------------------+      +-----------------------+
            |   .rag blueprint      |      |   .ragsh REPL         |
            | declarative workflow  |      | imperative RLM loop   |
            +-----------+-----------+      +-----------+-----------+
                        \                              /
                         \   compile / lower (by name) /
                          v                            v
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
          to Agent Node     (graph -> REPL -> graph)
```

## Quick start

Add TinyAgents to your project:

```toml
[dependencies]
tinyagents = "1.5"
```

The OpenAI (and OpenAI-compatible) provider is compiled in by default; the
build stays offline unless you actually make a call. Two optional Cargo
features gate heavier dependencies: `sqlite` (embedded SQLite checkpointer)
and `repl` (embedded Rhai engine for the `.ragsh` session runtime).

To explore locally:

```sh
git clone git@github.com:tinyhumansai/tinyagents.git
cd tinyagents
cargo run --example basic_graph
```

OpenAI-backed examples need an API key:

```sh
export OPENAI_API_KEY=...
cargo run --example openai_chat
```

Export durable harness observations to Langfuse with the embedded client:

```rust
use tinyagents::{LangfuseClient, LangfuseTraceConfig};

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
use tinyagents::{GraphLangfuseExporter, LangfuseClient, LangfuseTraceConfig};

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

All live in [`examples/`](examples/):

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
  harness (see [`examples/subconscious_loop/README.md`](examples/subconscious_loop/README.md)).
- **`openai_chat`** — a single provider-backed chat turn.
- **`openai_tools`** — tool calling against a hosted model.
- **`openai_structured`** — typed structured output.
- **`openai_graph_agent`** — a provider-backed agent driven inside a graph.

OpenAI-backed examples require `OPENAI_API_KEY` at run time.

## Documentation

- [crates.io](https://crates.io/crates/tinyagents)
- [docs.rs API reference](https://docs.rs/tinyagents)
- [Wiki home](https://github.com/tinyhumansai/tinyagents/wiki)
  - [Recursion and the RLM model](https://github.com/tinyhumansai/tinyagents/wiki/Recursion-and-RLM)
  - [Harness](https://github.com/tinyhumansai/tinyagents/wiki/Harness)
  - [Graph runtime](https://github.com/tinyhumansai/tinyagents/wiki/Graph-Runtime)
  - [Registry](https://github.com/tinyhumansai/tinyagents/wiki/Registry)
  - [Expressive language `.rag`](https://github.com/tinyhumansai/tinyagents/wiki/Expressive-Language-RAG)
  - [REPL language `.ragsh`](https://github.com/tinyhumansai/tinyagents/wiki/REPL-Language-RAGSH)
  - [Providers](https://github.com/tinyhumansai/tinyagents/wiki/Providers)
  - [Quick start](https://github.com/tinyhumansai/tinyagents/wiki/Quick-Start)
  - [Examples](https://github.com/tinyhumansai/tinyagents/wiki/Examples)
  - [Development](https://github.com/tinyhumansai/tinyagents/wiki/Development)

Contributors working directly in the repository should also read the checked-in
architecture specification under [`docs/spec/README.md`](docs/spec/README.md).

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --all-targets
cargo test
```

## Contributing

TinyAgents welcomes focused contributions that improve the graph runtime,
harness contracts, the registry, the `.rag` / `.ragsh` languages, provider
adapters, tests, examples, and documentation.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

TinyAgents is licensed under [GPL-3.0-only](LICENSE).

Built by TinyHumans for the Rust agent ecosystem.
