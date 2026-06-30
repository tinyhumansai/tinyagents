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
- Preserve provider prompt/KV-cache stability through explicit prompt segment
  boundaries.
- Track which middleware invalidated or preserved provider prompt-cache
  prefixes.
- Emit cache events with key fingerprints rather than full sensitive payloads.
- Support in-memory, store-backed, and provider-specific cache metadata.

## Source Inspiration

LangChain core has a beta local model cache with `lookup`, `update`, and async
variants:

- <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/caches.py>

Provider prompt caching is different. It usually affects provider billing and
usage metadata, not whether TinyAgents skips the provider call entirely.

## Provider Prompt And KV Cache

Provider prompt caching, prefix caching, and KV-cache reuse are first-class
targets. The harness must make it hard to accidentally invalidate a large stable
prefix by inserting volatile context near the front of a request.

Prompt assembly should support explicit segments:

```rust
pub struct PromptSegment {
    pub id: PromptSegmentId,
    pub cache_role: CacheRole,
    pub content: Vec<Message>,
    pub fingerprint: PromptFingerprint,
}

pub enum CacheRole {
    StablePrefix,
    StableButProviderSpecific,
    VolatileTail,
    NeverCache,
}
```

Stable prefix segments are for content that should remain byte/token stable
across many turns:

- system prompts
- policy and safety text
- reusable developer instructions
- tool declarations and schemas
- structured output schemas
- long-lived examples
- durable project or tenant context

Volatile tail segments are for content likely to change every turn:

- latest user message
- current retrieved documents
- timestamps and run ids
- tool results
- scratchpads and temporary reasoning traces
- per-run configurable metadata

Middleware that edits prompts must report whether it changed the stable prefix
or only the volatile tail. This lets tests, traces, and cost accounting explain
why provider prompt-cache hits were preserved or lost.

## KV-Cache-Safe Layout Rules

Request builders and middleware should follow these rules:

- never insert timestamps, run ids, random ids, or dynamic retrieval output into
  a stable prefix by default
- append volatile context after stable instructions and schemas
- keep stable tool/schema serialization canonical and deterministic
- preserve segment ordering unless a middleware explicitly declares a cache
  layout migration
- fingerprint prompt segments separately from the full request
- include middleware policy fingerprints when a middleware can affect
  model-visible bytes
- preserve `thread_id` across parent agents, sub-agents, graph subgraphs, and
  harness calls so provider-side prompt/KV-cache attribution remains stable for
  the logical conversation
- map stable `thread_id` or tenant/user identity into provider-specific cache
  headers when required by a provider policy; for example, Fireworks-style
  integrations may need a deterministic user/cache identifier header to reuse
  provider cache safely
- emit `cache.layout_preserved`, `cache.layout_changed`, and
  `cache.prefix_invalidated` events for observability

Regression tests should be able to assert that a prompt edit preserves the
stable prefix fingerprint even if the full request changes.

## Cache Policy

```rust
pub struct CachePolicy {
    pub enabled: bool,
    pub ttl: Option<Duration>,
    pub scope: CacheScope,
    pub include_tools: bool,
    pub include_model_responses: bool,
    pub preserve_provider_prefix: bool,
    pub stable_prefix_min_tokens: Option<usize>,
}
```

Cache keys must include every behavior-affecting input: model, messages, tools,
tool schemas, response format, provider options, and relevant metadata. Unsafe
or side-effecting tool calls should not be cached by default.

The local response cache key is a SHA-256 digest of canonical request JSON.
Prompt text is not embedded directly in the key, but every serialized
behavior-affecting request field participates in the digest.

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
- prompt segment ids and segment fingerprints
- provider prompt-cache options
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
- provider prefix preserved
- provider prefix invalidated
- write skipped
- write completed

The usage feature should record provider prompt-cache hits separately from local
response-cache hits.
