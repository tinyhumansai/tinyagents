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

As implemented today (`harness::cache::CachePolicy`):

```rust
pub struct CachePolicy {
    pub response_cache_enabled: bool,
    pub protect_prompt_prefix: bool,
    pub ttl_ms: Option<u64>,
    pub namespace: Option<String>,
}
```

`ttl_ms` and `namespace` cover the `ttl` / `scope` this spec asks for. The
remaining aspirational fields (`include_tools`, `include_model_responses`,
`stable_prefix_min_tokens`) are **not implemented**; tools are always part of
the key and there is no minimum-prefix threshold.

Unsafe or side-effecting tool calls should not be cached by default.

### Key composition

The key is a two-part composition, never the prompt alone:

```text
scoped_cache_key(cache_key(request), model.cache_identity(), streaming, namespace)
```

- `cache_key(request)` is a SHA-256 digest over per-message and per-tool frames
  plus an **explicit allowlist projection** of the behaviour-affecting
  parameters. The projection destructures `ModelRequest` exhaustively, so adding
  a request field is a compile error until someone decides whether it belongs in
  the key. Fields that cannot change the answer — `tags`, `timeout_ms`,
  `metadata`, `cache_policy`, `prompt_fingerprint`, `cache_segments` — are
  deliberately excluded; folding them in gave a caller who put a run id in
  `metadata` a permanent 0% hit rate.
- `cache_identity()` names the provider family, model id, API base URL, optional
  scope, and a **fingerprint** of the credential. It is computed *after* model
  resolution, because the real model is chosen by `ModelRegistry::resolve_request`
  and the endpoint and credential live inside the `Arc<dyn ChatModel>`, never in
  the request. Without it one shared cache serves a hosted harness's answer to a
  local one.
- `streaming` is a parameter of the call rather than a request field, so it is
  folded explicitly; a warm streaming run is not served an entry written by a
  unary run.

Raw credentials never reach a key: `credential_fingerprint` hashes them first.

### Write rules

- Only the **primary** model's answer is written under its own key. When the
  fallback chain answers, the write is skipped — otherwise the primary's key is
  poisoned (permanently, absent a TTL) with a different model's response.
- A cache read or write failure is logged and ignored. The provider call has
  already succeeded and been paid for; discarding its answer because the cache
  was unavailable is strictly worse than not caching.
- A cache hit is stamped `ModelResponse::served_from_cache` so token/cost
  accounting can tell a replay from a real call and not re-bill it.
- A cache hit on a **streaming** run is replayed as synthetic `ModelDelta`
  events (text, then one per tool call) so warm and cold runs are
  observationally identical.

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

Implemented today: `AgentEvent::CacheHit` / `AgentEvent::CacheMiss` are emitted
as events, and the "no lookup happened at all" half is reported as a
`CacheSkipReason` (`no_cache_attached`, `policy_disabled`,
`multi_turn_transcript`) on a `[cache]`-prefixed debug log. A cache also exposes
`ResponseCache::stats() -> CacheStats` (hits, misses, writes, evictions,
expirations, entries, bytes). The remaining decisions are not yet distinct
events.

The usage feature should record provider prompt-cache hits separately from local
response-cache hits.

## Backends

- `InMemoryResponseCache` — bounded on **both** an entry count and an
  approximate byte budget (an entry count alone does not bound memory when
  responses are long-context). Recency is an ordered map, not a linear scan.
- `SqliteResponseCache` (feature `sqlite`) — durable, WAL, `(ns, key)` primary
  key with an `expiry` column and a lazy purge on read. Namespaces do not
  cross-serve and clear independently.
- `SingleFlight` — collapses concurrent identical misses into one provider call
  so N simultaneous callers do not all pay for the same answer. Errors are not
  shared: a follower whose leader failed runs its own call.
