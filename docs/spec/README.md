# TinyAgents System Specification

TinyAgents is a Rust-native LLM application framework inspired by LangChain,
LangGraph, and CodeAct-style recursive language-model runtimes. The system is
organized around five modules:

1. the harness
2. the graph
3. the registry
4. the expressive language
5. the REPL language

The goal is to make agent systems easy to define, inspect, run, test, and
eventually serialize without hiding the Rust types that make production systems
reliable.

## Reference Positioning

TinyAgents should synthesize the reference systems rather than clone any one of
them:

- LangGraph contributes the durable execution model: explicit state graphs,
  virtual `START` and `END`, Pregel-style supersteps, reducers/channels,
  commands, `Send` fanout, checkpointing, interrupts, subgraphs, streaming, and
  time travel.
- LangChain contributes the harness model: provider-neutral models, tools,
  middleware, runtime context, memory, retrieval, structured output, tracing,
  usage, cost, and conformance tests for integrations.
- `rust-langgraph` shows the Rust-facing precedent for a stateful graph runtime
  with nodes, conditional edges, checkpoints, streaming, optional model
  adapters, and ReAct/tool helpers. TinyAgents should go deeper on typed state,
  harness composition, registries, and language-backed graph definitions.
- OpenHuman PR #4261 contributes the closest product-shaped precedent: a
  harness-decoupled graph engine, persistent checkpoints, HITL, graph
  observability, blueprints, JSON-RPC run control, and a behavior-preserving
  cutover from an implicit turn loop to an explicit phase machine.
- RLM contributes the REPL/code-act model: context and prompts as runtime
  values, recursive sub-model or sub-agent calls as functions, persistent
  session variables, trajectory logging, and sandbox choices.

The target architecture is therefore layered: the harness owns model/tool
execution and policies, the graph owns deterministic state transition and
durability, the registry owns named capabilities, `.rag` owns serializable graph
blueprints, and `.ragsh` owns capability-bound interactive orchestration. No
layer should bypass another layer's safety, policy, observability, or test
contracts.

## Detailed Module Docs

- [Harness module](../modules/harness/README.md)
  - [Context](../modules/harness/context.md)
  - [Model and providers](../modules/harness/model.md)
  - [Embeddings and retrieval](../modules/harness/embeddings.md)
  - [Prompt](../modules/harness/prompt.md)
  - [Tool](../modules/harness/tool.md)
  - [Middleware](../modules/harness/middleware.md)
  - [Sub-agent and orchestrator steering](../modules/harness/subagent-steering.md)
  - [Structured output](../modules/harness/structured-output.md)
  - [Limits, retry, fallback, and rate limiting](../modules/harness/limits-retry.md)
  - [Summarization](../modules/harness/summarization.md)
  - [Usage](../modules/harness/usage.md)
  - [Cost](../modules/harness/cost.md)
  - [Cache](../modules/harness/cache.md)
  - [Streaming](../modules/harness/streaming.md)
  - [Store](../modules/harness/store.md)
  - [Observability and events](../modules/harness/observability.md)
  - [Testkit](../modules/harness/testkit.md)
- [Graph module](../modules/graph/README.md)
  - [Package and core types](../modules/graph/package.md)
  - [Builder and compile contract](../modules/graph/builder.md)
  - [Node model](../modules/graph/nodes.md)
  - [State, channels, and updates](../modules/graph/state-channels.md)
  - [Edges, routing, commands, and sends](../modules/graph/routing.md)
  - [Execution model and parallelization](../modules/graph/execution.md)
  - [Parallel agents and context forking](../modules/graph/parallel-agents-forking.md)
  - [Checkpointing, durability, state inspection, and time travel](../modules/graph/checkpointing.md)
  - [Interrupts and resume](../modules/graph/interrupts.md)
  - [Streaming and events](../modules/graph/streaming.md)
  - [Observability and tracing](../modules/graph/observability.md)
  - [Runtime context and policies](../modules/graph/runtime-policy.md)
  - [Fault tolerance](../modules/graph/fault-tolerance.md)
  - [Subgraphs](../modules/graph/subgraphs.md)
  - [Sub-agents and recursion](../modules/graph/subagents-recursion.md)
  - [Memory and stores boundary](../modules/graph/memory-boundary.md)
  - [Visualization, introspection, and testkit](../modules/graph/visualization-testkit.md)
  - [Implementation milestones](../modules/graph/milestones.md)
- [Registry module](../modules/registry/README.md)
  - [Design](../modules/registry/design.md)
  - [Model catalog and local snapshots](../modules/registry/model-catalog.md)
- [Expressive language module](../modules/expressive-language/README.md)
- [REPL language module](../modules/repl-language/README.md)
  - [Design](../modules/repl-language/design.md)

Docs should follow the module layout. Do not place standalone specification
files directly in `docs/` or `docs/modules/`; each high-level topic should have
its own directory with a `README.md` entrypoint and any supporting files beside
it.

## Design Goals

- Make simple agent workflows concise.
- Make complex workflows explicit, inspectable, and testable.
- Treat graph execution as a first-class runtime, not an incidental callback
  chain.
- Keep model providers, tools, memory, and tracing behind stable traits.
- Support both Rust builder APIs and a compact expressive language for workflow
  definitions.
- Support a capability-bound REPL language for interactive graph and harness
  orchestration.
- Allow agents to author, inspect, compile, and run graph blueprints through the
  same registry-bound compiler path used by human-authored `.rag` files.
- Allow parent orchestrators and humans to steer orchestrator agents and
  sub-agents through typed, policy-checked, observable commands.
- Prefer deterministic state transitions around inherently nondeterministic LLM
  calls.
- Keep every generated or hand-authored graph explainable as topology,
  capabilities, policies, state channels, checkpoints, and events.

## Module 1: Harness

The harness is the provider-neutral runtime for model calls, tools,
middleware, structured output, streaming, usage/cost, retry/limits, cache,
memory/embeddings, sub-agents, and steering. See
[`harness-spec.md`](harness-spec.md) for the full specification (core types,
model/tool/message abstractions, agent loop, middleware, memory, structured
output, observability, and testability), and
[`docs/modules/harness/README.md`](../modules/harness/README.md) for the
per-topic implementation docs.

## Module 2: Graph

The graph is the durable, typed state-graph runtime: `START`/`END`, nodes,
reducers/channels, routing, supersteps, checkpointing, interrupts, streaming,
subgraphs, and execution guarantees. See [`graph-spec.md`](graph-spec.md) for
the full specification, and
[`docs/modules/graph/README.md`](../modules/graph/README.md) for the
per-topic implementation docs.

## Module 3: Expressive Language

The `.rag` expressive language is a declarative, side-effect-free blueprint
format that compiles through lexer -> parser -> compiler into the same
graph/harness runtime types as hand-written Rust. See
[`expressive-language-spec.md`](expressive-language-spec.md) for the goals,
grammar sketch, and compilation pipeline, and
[`docs/modules/expressive-language/README.md`](../modules/expressive-language/README.md)
for implementation status.

## Package Layout

The crate is a single library at the repository root (`Cargo.toml`), with
`src/lib.rs` re-exporting the public surface and `src/error.rs` holding the
crate-wide error type. Each of the five surfaces lives in its own module
directory:

```text
src/
  error.rs
  lib.rs
  graph/       # durable typed state graphs (checkpoint, interrupt, streaming, ...)
  harness/     # provider-neutral model calls, tools, middleware, streaming, ...
  language/    # the declarative `.rag` blueprint format (lexer/parser/compiler)
  registry/    # the named capability catalog (models, tools, agents, stores, ...)
  repl/        # the imperative `.ragsh` session runtime
```

Provider implementations (OpenAI and the OpenAI-compatible endpoints for
Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together, and Mistral)
live inside `src/harness/providers/` and are compiled in unconditionally.
Two Cargo features gate optional dependencies: `sqlite` (embedded SQLite
checkpointer) and `repl` (embedded Rhai engine for `.ragsh` sessions).

## Milestones

All five milestones below have shipped as of v1.5.0.

### Milestone 1: Core Runtime (shipped)

Chat message primitives, the model and tool traits, the state graph with
direct and conditional edges, and the initial test/example suite.

### Milestone 2: Harness (shipped)

The `AgentHarness` type, model and tool registries, run context, callback
events, run status store, durable event journal, cache-backed observability
projections, and mock model/tool testkit utilities.

### Milestone 3: Expressive Language (shipped)

The `.rag` AST, lexer, parser, compiler into the graph runtime, parse/
validation diagnostics with source spans, and example `.rag` workflow files
(see `examples/rag_blueprint.rs`, `examples/openai_self_blueprint.rs`).

### Milestone 4: Provider Integrations (shipped)

OpenAI and OpenAI-compatible provider adapters (Anthropic, Ollama, DeepSeek,
Groq, xAI, OpenRouter, Together, Mistral), plus the offline deterministic
mock provider.

### Milestone 5: Production Runtime Features (shipped)

Streaming events, checkpointing and resume support, the graph run status
store, event journal with listener replay, graph export, and an embedded
Langfuse tracing integration (`LangfuseClient`, `GraphLangfuseExporter`).

## Open Questions

Historical decisions that have since been settled, kept for context:

- The expressive language file extension is `.rag` (interactive/imperative
  orchestration uses the separate `.ragsh` extension).
- State schemas remain Rust-owned; `.rag` binds to them by name through the
  registry rather than declaring schemas itself.
- Provider crates live in this crate as always-compiled modules behind
  `src/harness/providers/`, not separate crates or feature flags.
- Memory and embeddings are async, matching the rest of the harness surface.

Remaining open question:

- Should graph nodes support typed route enums as a stronger alternative to
  string-keyed conditional routing before further serialization work lands?
