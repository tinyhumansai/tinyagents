# Harness Design Notes

Split out of [`README.md`](README.md) to keep that file under the repo's
500-line Markdown limit. This file holds the harness's core-type sketch; see
[`langchain-parity.md`](langchain-parity.md) for the LangChain feature-parity
checklist, and the module README for the package shape and feature index.

## Core Types

Illustrative sketch of the harness's central types (not the literal current
struct definitions — see `crates/tinyagents-harness/src/context/types.rs` and
`crates/tinyagents-harness/src/runtime/types.rs` for the real fields):

```rust
pub struct AgentHarness<State, Ctx = ()> {
    models: ModelRegistry<State, Ctx>,
    embeddings: EmbeddingRegistry<Ctx>,
    tools: ToolRegistry<State, Ctx>,
    middleware: MiddlewareStack<State, Ctx>,
    memory: Option<Arc<dyn ShortTermMemory<State>>>,
    stores: StoreRegistry,
    policy: RunPolicy,
}

pub struct RunConfig {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub configurable: serde_json::Value,
    pub timeout: Option<Duration>,
    pub max_model_calls: usize,
    pub max_tool_calls: usize,
}

pub struct RunContext<Ctx = ()> {
    pub config: RunConfig,
    pub data: Ctx,
    pub events: EventSink,
    pub stores: StoreRegistry,
    pub cancellation: CancellationToken,
}
```

`RunConfig` is serializable invocation policy and identity. `RunContext` is the
runtime dependency container. This split keeps tests deterministic and prevents
global singletons.

Nested model calls, tools, sub-agents, and graph nodes must inherit the root run
id, selected tags, inherited metadata, event sink, cancellation token, stores,
usage tracker, cost tracker, and configured budget policy. They may add local
tags and metadata, but they must not mutate parent config in place.

Nested runs may also receive steering commands. Steering is explicit runtime
control from a parent orchestrator, human, graph supervisor, middleware, or
test. A steered run must record actor, target, policy, payload summary, and the
safe boundary where the command was applied. See
[Sub-agent and orchestrator steering](subagent-steering.md).
