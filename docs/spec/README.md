# TinyAgents System Specification

TinyAgents is a Rust-native LLM application framework inspired by LangChain and
LangGraph. The system is organized as focused crates for five public surfaces:

1. the harness
2. the graph
3. the registry
4. the expressive language
5. durable sessions

Scripted, interpreter-backed orchestration (a CodeAct/REPL loop over
model-written code cells) is deliberately a *host* concern built on top of these
crates, not a surface this workspace ships. See "Host-side surfaces" below.

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
- CodeAct/recursive-language-model runtimes contribute the recursion model:
  context and prompts as runtime values, recursive sub-model or sub-agent calls
  as functions, persistent session variables, and trajectory logging. TinyAgents
  provides the primitives (registry capabilities, sub-agents, session/cell/call
  ids, event journals); the interpreter and its sandbox stay host-side.

The target architecture is therefore layered: the harness owns model/tool
execution and policies, the graph owns deterministic state transition and
durability, the registry owns named capabilities, and `.rag` owns serializable
graph blueprints. No layer should bypass another layer's safety, policy,
observability, or test contracts.

## Host-side surfaces

Some things a recursive agent system needs are intentionally *not* implemented
here, because a host can implement them on top of the four modules and because
shipping them would drag an embedded interpreter into every dependent's build:

- the scripted CodeAct/REPL session loop (an embedded Rhai / Python / JavaScript
  interpreter running model-written code cells)
- the driver loop that prompts a model for the next code cell and feeds the
  cell's output back in

What these crates provide for those hosts: the capability `registry` (so a script
can only reach named `llm` / `tool` / `agent` capabilities), the harness and its
sub-agent recursion accounting, typed `SessionId` / `CellId` / `CallId`, the
event journal, and the `.rag` `repl_agent` node kind, which binds a
host-provided scripted node to a registered `Script` component by name.

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

Per-tool deadlines are opt-in at the harness boundary through
`AgentHarness::with_tool_timeout_settings`. A tool's `ToolTimeout` policy is
resolved from the final post-middleware call: `Inherit` uses the shared dynamic
default, `Millis` is clamped and padded with configured grace, and `Unbounded`
has no per-tool deadline. Expiry is a recoverable tool result returned to the
model; only the enclosing run wall-clock deadline aborts the run.

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

The repository root is a virtual Cargo workspace. There is no `tinyagents`
compatibility facade: applications depend directly on the packages whose APIs
they use. Shared runtime errors live in `tinyagents-harness`.

```text
crates/
  tinyagents-harness/           # models, tools, middleware, providers, runtime
  tinyagents-language/          # .rag lexer, parser, compiler, and resolver
  tinyagents-graph/             # durable typed state graphs
  tinyagents-registry/          # named capabilities and model catalog
  tinyagents-session/           # durable session history and run ledger
  tinyagents-tracing/           # shared opt-in tracing macros
  tinyagents-integration-tests/ # cross-crate tests and runnable examples
```

Provider implementations (OpenAI and the OpenAI-compatible endpoints for
Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together, and Mistral)
live inside `crates/tinyagents-harness/src/providers/` and are compiled in
unconditionally. Optional features are owned by their packages. Tracing calls
and the direct `tracing` dependency are disabled unless a package's `tracing`
feature is enabled.

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
(see the examples in `crates/tinyagents-integration-tests/examples/`).

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

- The expressive language file extension is `.rag`. Interactive/imperative
  orchestration was prototyped as a separate `.ragsh` surface and has since been
  removed from this crate as a host concern.
- State schemas remain Rust-owned; `.rag` binds to them by name through the
  registry rather than declaring schemas itself.
- Providers remain always-compiled modules of `tinyagents-harness`, rather
  than becoming one crate per provider.
- Memory and embeddings are async, matching the rest of the harness surface.

Remaining open question:

- Should graph nodes support typed route enums as a stronger alternative to
  string-keyed conditional routing before further serialization work lands?
