# Harness Cache Feature

Caching avoids repeated model, prompt, summary, and artifact work when policy
allows it.

## Responsibilities

- Build stable cache keys.
- Cache prompt rendering.
- Cache model responses where safe.
- Cache summaries.
- Cache tool artifacts.
- Record cache hits and misses.
- Feed cached token counts into usage and cost accounting.

## Cache Policy

```rust
pub struct CachePolicy {
    pub enabled: bool,
    pub ttl: Option<Duration>,
    pub scope: CacheScope,
    pub include_tools: bool,
    pub include_model_responses: bool,
}
```

Cache keys must include every behavior-affecting input: model, messages, tools,
tool schemas, response format, provider options, and relevant metadata. Unsafe
or side-effecting tool calls should not be cached by default.
