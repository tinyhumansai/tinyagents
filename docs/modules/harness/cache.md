# Harness Cache Feature

Caching avoids repeated model, prompt, summary, and artifact work when policy
allows it.

## Responsibilities

- Build stable cache keys.
- Cache prompt rendering.
- Cache embeddings where safe.
- Cache model responses where safe.
- Cache summaries.
- Cache tool artifacts.
- Record cache hits and misses.
- Feed cached token counts into usage and cost accounting.
- Distinguish local response caching from provider prompt caching.
- Emit cache events with key fingerprints rather than full sensitive payloads.
- Support in-memory, store-backed, and provider-specific cache metadata.

## Source Inspiration

LangChain core has a beta local model cache with `lookup`, `update`, and async
variants:

- <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/caches.py>

Provider prompt caching is different. It usually affects provider billing and
usage metadata, not whether RustAgents skips the provider call entirely.

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

## Cache Key Inputs

Model response cache keys should include:

- provider and model id
- canonical serialized messages
- content block order and ids when ids affect behavior
- tool declarations and schemas
- tool choice
- response format
- normalized model settings
- provider options
- relevant metadata/configurable values
- prompt template version
- middleware version or policy fingerprint when middleware changes requests

Cache keys should store fingerprints, not raw prompts, where the backing store
may be inspected by humans or external systems.

Embedding cache keys should include provider, model, input text, document/query
mode, requested dimensions, preprocessing version, and provider options. A query
embedding and document embedding for the same text must not share a key unless
the provider adapter explicitly declares they are equivalent.

## Cache Decisions

Every lookup should produce a decision:

- disabled by policy
- skipped because request is unsafe
- miss
- hit
- stale
- write skipped
- write completed

The usage feature should record provider prompt-cache hits separately from local
response-cache hits.
