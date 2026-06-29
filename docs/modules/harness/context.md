# Harness Context Feature

The context feature owns what the harness knows at run time: identities,
metadata, user/runtime data, configured limits, available stores, event sinks,
and model context-window pressure.

## Responsibilities

- Carry `run_id`, `thread_id`, `parent_run_id`, and `root_run_id`.
- Carry local and inherited tags/metadata.
- Carry runtime configurable values.
- Carry cancellation.
- Carry store, event, cache, usage, and cost handles.
- Track context-window budget for the selected model.
- Expose context to models, tools, middleware, and graph nodes.
- Provide inherited context to nested model calls, tools, sub-agents, and graph
  nodes without relying on global variables.
- Hide runtime-only values from model-visible tool schemas.

## Source Inspiration

LangChain's `RunnableConfig` and v1 `ModelRequest.runtime` pass tags,
metadata, configurable values, callbacks, and runtime context through nested
calls:

- <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/config.py>
- <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py>
- <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/tools/tool_node.py>

TinyAgents should use typed Rust context values instead of dynamic Python
dictionaries where possible, while preserving a JSON metadata/configurable
escape hatch for app-level data.

## Core Types

```rust
pub struct RunContext<Ctx = ()> {
    pub config: RunConfig,
    pub data: Ctx,
    pub stores: StoreRegistry,
    pub events: EventSink,
    pub usage: UsageTracker,
    pub costs: CostTracker,
    pub cache: CacheRegistry,
    pub cancellation: CancellationToken,
}

pub struct ContextWindow {
    pub model: ModelName,
    pub max_tokens: usize,
    pub reserved_output_tokens: usize,
    pub estimated_prompt_tokens: usize,
}
```

## Context Pressure

Before every model call, the harness estimates whether the request fits. If not,
it applies configured policies in order:

1. drop nonessential retrieved context
2. trim old messages
3. summarize old messages
4. compact tool outputs
5. fail with a context-limit error

Every action emits an event.

## Inheritance Rules

Nested calls inherit:

- root run id
- parent run id
- thread id unless overridden
- event sink
- cancellation token
- stores
- usage and cost trackers
- cache registry
- inherited tags and metadata
- budget and limit policies

Nested calls may add local tags and metadata. They must not mutate parent config
in place. This keeps traces and tests deterministic.

## Runtime Injection

Tools and middleware may receive runtime-only values such as stores,
cancellation, event sinks, and typed app context. Those values must not appear in
model-visible JSON schemas. The tool feature owns schema hiding; the context
feature owns safe access to the values.
