# Harness Context Feature

The context feature owns what the harness knows at run time: identities,
metadata, user/runtime data, configured limits, available stores, event sinks,
and model context-window pressure.

## Responsibilities

- Carry `run_id`, `thread_id`, `parent_run_id`, and `root_run_id`.
- Carry local and inherited tags/metadata.
- Carry runtime configurable values.
- Carry store, event, cache, usage, and cost handles.
- Track context-window budget for the selected model.
- Expose context to models, tools, middleware, and graph nodes.

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
