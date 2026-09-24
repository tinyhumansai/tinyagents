# TinyAgents System Specification

TinyAgents is a small, provider-neutral agent harness for Rust, plus a durable
typed state-graph runtime. It takes its shape from LangChain (models, tools,
middleware, structured output, streaming, usage/cost) and LangGraph
(`START`/`END`, nodes, conditional edges, channels/reducers, checkpoints,
interrupts, subgraphs, time travel) — rebuilt as ordinary, typed Rust. The
system is organized as five public crates:

1. the harness
2. the graph
3. the registry
4. durable sessions
5. host-neutral session runtime

The goal is to make agent systems easy to define, inspect, run, test, and
serialize without hiding the Rust types that make production systems reliable.

## Reference Positioning

TinyAgents synthesizes the reference systems rather than cloning either one:

- LangGraph contributes the durable execution model: explicit state graphs,
  virtual `START` and `END`, Pregel-style supersteps, reducers/channels,
  commands, `Send` fanout, checkpointing, interrupts, subgraphs, streaming, and
  time travel.
- LangChain contributes the harness model: provider-neutral models, tools,
  middleware, runtime context, memory, retrieval, structured output, tracing,
  usage, cost, and conformance tests for integrations.

The target architecture is layered: the harness owns model/tool execution and
policies, the graph owns deterministic state transition and durability, the
registry owns named capabilities. No layer should bypass another layer's safety, policy,
observability, or test contracts.

## Detailed Module Docs

- [Harness module](../modules/harness/README.md)
  - [Context](../modules/harness/context.md)
  - [Model and providers](../modules/harness/model.md)
  - [Embeddings and retrieval](../modules/harness/embeddings.md)
  - [Media generation](../modules/harness/media.md)
  - [Prompt](../modules/harness/prompt.md)
  - [Tool](../modules/harness/tool.md)
  - [Tool exposure and discovery](../modules/harness/tool-discovery.md)
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
  - [Performance and capacity testing](../modules/harness/performance.md)
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
- [Session runtime module](../modules/runtime/README.md)
- [Runtime comparison and execution plan](../runtime-comparison/README.md)

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

Tool schemas are advertised by exposure, not by registration. Only
`ToolExposure::Direct` tools appear in a request's `tools` array; `Deferred`
tools are indexed per run and reached through the intrinsic `tool_search` /
`tool_call` bridge, whose `tool_call` is unwrapped to the real tool before
admission so policy and authorization see the true name. The `tools` array
therefore stays byte-stable for a whole run when no per-turn exposure
middleware (dynamic/contextual tool selection, tool-policy filtering) changes
the advertised direct set, which is what a provider prompt cache depends on.
`RunPolicy::tool_schemas` optionally projects and byte-budgets every schema.

## Module 2: Graph

The graph is the durable, typed state-graph runtime: `START`/`END`, nodes,
reducers/channels, routing, supersteps, checkpointing, interrupts, streaming,
subgraphs, and execution guarantees. See [`graph-spec.md`](graph-spec.md) for
the full specification, and
[`docs/modules/graph/README.md`](../modules/graph/README.md) for the
per-topic implementation docs.

## Package Layout

The repository root is a virtual Cargo workspace. There is no `tinyagents`
compatibility facade: applications depend directly on the packages whose APIs
they use. Shared runtime errors live in `tinyagents-harness`.

```text
crates/
  tinyagents-harness/           # models, tools, middleware, providers, runtime
  tinyagents-graph/             # durable typed state graphs
  tinyagents-registry/          # named capabilities and model catalog
  tinyagents-session/           # durable session history and run ledger
  tinyagents-definition/        # host-owned agent definition vocabulary
  tinyagents-runtime/           # host-neutral stateful harness sessions
  tinyagents-orchestration/     # host-neutral team/workflow composition over graph+harness+session
  tinyagents-integration-tests/ # cross-crate tests and runnable examples
```

Provider implementations (OpenAI and the OpenAI-compatible endpoints for
Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together, and Mistral)
live inside `crates/tinyagents-harness/src/providers/` and are compiled in
unconditionally. Optional features are owned by their packages. Tracing calls
and the direct `tracing` dependency are disabled unless a package's `tracing`
feature is enabled.

## Milestones

All four milestones below have shipped as of v1.5.0.

### Milestone 1: Core Runtime (shipped)

Chat message primitives, the model and tool traits, the state graph with
direct and conditional edges, and the initial test/example suite.

### Milestone 2: Harness (shipped)

The `AgentHarness` type, model and tool registries, run context, callback
events, run status store, durable event journal, cache-backed observability
projections, and mock model/tool testkit utilities.

### Milestone 3: Provider Integrations (shipped)

OpenAI and OpenAI-compatible provider adapters (Anthropic, Ollama, DeepSeek,
Groq, xAI, OpenRouter, Together, Mistral), plus the offline deterministic
mock provider.

### Milestone 4: Production Runtime Features (shipped)

Streaming events, checkpointing and resume support, the graph run status
store, event journal with listener replay, graph export, and an embedded
Langfuse tracing integration (`LangfuseClient`, `GraphLangfuseExporter`).

## Open Questions

Historical decisions that have since been settled, kept for context:

- Providers remain always-compiled modules of `tinyagents-harness`, rather
  than becoming one crate per provider.
- Memory and embeddings are async, matching the rest of the harness surface.

Remaining open question:

- Should graph nodes support typed route enums as a stronger alternative to
  string-keyed conditional routing before further serialization work lands?
