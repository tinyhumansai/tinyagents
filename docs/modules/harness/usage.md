# Harness Usage Feature

Usage accounting normalizes token and cache usage across providers.

## Responsibilities

- Track prompt tokens.
- Track completion tokens.
- Track cached input tokens.
- Track reasoning tokens when providers expose them.
- Track tool/model call counts.
- Roll usage up by call, node, agent, graph, thread, and run.
- Emit usage events.

## Core Types

```rust
pub struct UsageRecord {
    pub run_id: RunId,
    pub component: ComponentId,
    pub call_id: CallId,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub reasoning_tokens: usize,
    pub total_tokens: usize,
}
```

Provider adapters should map provider-specific usage into this normalized shape.
When providers do not return usage, the harness may estimate tokens and mark the
record as estimated.
