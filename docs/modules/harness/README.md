# Harness Module Specification

The harness is the orchestration layer around LLM calls. It owns model
registration, tool registration, prompt assembly, middleware, memory, event
streaming, tracing, retries, limits, summarization, caching, usage accounting,
pricing, and test support.

The harness should be usable in three modes:

1. direct model invocation
2. model plus tools agent loop
3. graph node runtime dependency

It should not require the graph module. The graph module can depend on harness
traits, but a user should be able to call a model or run a tool loop without
constructing a graph.

## Source Inspiration

Primary references:

- <https://docs.langchain.com/oss/python/langchain/agents>
- <https://docs.langchain.com/oss/python/langchain/models>
- <https://docs.langchain.com/oss/python/langchain/tools>
- <https://docs.langchain.com/oss/python/langchain/runtime>
- <https://docs.langchain.com/oss/python/langchain/short-term-memory>
- <https://docs.langchain.com/oss/python/langchain/structured-output>
- <https://docs.langchain.com/oss/python/langchain/middleware/built-in>
- <https://docs.langchain.com/oss/python/langchain/streaming>
- <https://docs.langchain.com/oss/python/langchain/observability>
- <https://docs.langchain.com/oss/python/langchain/test>
- LangChain callback usage tracking code:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/usage.py>
- LangChain store and chat history code:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/stores.py>
  and
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/chat_history.py>
- LangChain v1 agent factory:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py>
- LangChain v1 agent middleware types and built-ins:
  <https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware>
- LangChain structured output strategies:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py>
- LangChain model profiles:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/model_profile.py>
  and
  <https://github.com/langchain-ai/langchain/tree/master/libs/model-profiles>
- LangChain message and content-block model:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/messages>
- LangChain embeddings, vector stores, retrievers, and indexing:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/embeddings>
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/vectorstores>
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/retrievers.py>
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/indexing>
- LangChain runnable config, fallbacks, retry, and event streams:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/runnables>
- OpenHuman PR #4261 agent graph implementation:
  <https://github.com/tinyhumansai/openhuman/pull/4261>
- OpenHuman PR #4261 state graph files:
  `src/openhuman/agent_graph/graph/`, `checkpoint/`, `hitl/`,
  `observability/`, `definitions/`, `blueprint/`, `live/`, `ops.rs`, and
  `schemas.rs`
- LangChain callbacks, tracers, and usage accounting:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/callbacks>
  and
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tracers>
- LangChain standard integration tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>

LangChain separates durable core primitives (`libs/core`), the v1 agent facade
(`libs/langchain_v1`), legacy/classic integrations (`libs/langchain`), partner
providers (`libs/partners`), model capability data (`libs/model-profiles`), and
standard provider test suites (`libs/standard-tests`). RustAgents should keep a
similar separation of concerns even though it is a single Rust crate today:
harness traits first, feature-gated provider adapters second, and compatibility
tests for every adapter.

## Responsibilities

- Normalize user input into structured messages.
- Build model requests from messages, prompts, tools, memory, and config.
- Track context-window pressure and choose trimming or summarization policies.
- Preserve provider prompt/KV-cache stability by making stable prompt prefixes
  explicit and keeping volatile context out of those prefixes by default.
- Dispatch model calls through provider-neutral traits.
- Dispatch embedding calls through provider-neutral traits.
- Dispatch tool calls through a registry with schema validation.
- Expose retrievers and vector stores for retrieval-augmented context assembly.
- Run the standard model-tool-model agent loop.
- Represent the standard agent loop as an explicit state machine when callers
  need inspection, checkpointing, HITL, branch/fork semantics, or graph-node
  execution.
- Apply middleware before and after model calls, tool calls, retries, and errors.
- Enforce model-call limits, tool-call limits, timeouts, and retry policy.
- Emit typed events for tracing, streaming, and tests.
- Persist short-term thread memory when configured.
- Expose durable stores through runtime context.
- Store run data, messages, events, tool artifacts, and application records
  through pluggable backends.
- Track prompt tokens, completion tokens, cached tokens, model prices, and
  run-level cost.
- Enforce optional token budgets and dollar budgets.
- Cache reusable prompts, model responses, tool artifacts, and summaries when
  policy allows.
- Distinguish local response caching from provider prompt/KV-cache reuse and
  expose cache layout events when middleware changes model-visible prompt
  segments.
- Provide deterministic test utilities.
- Describe provider capability profiles so middleware can choose safe defaults.
- Translate between provider-native message formats and RustAgents messages.
- Support dynamic runtime context injection into tools and middleware without
  exposing private state to model-visible schemas.
- Support model fallback, tool retry, rate limiting, and human interruption as
  explicit policies rather than ad hoc callbacks.
- Support durable graph runs with pause/resume, checkpoint listing, and
  inspectable node transitions.
- Support per-agent execution blueprints that describe how an agent runs
  separately from what its prompt says.
- Provide a standard conformance test suite for model, tool, store, stream, and
  middleware implementations.

## Non-Responsibilities

- It does not own graph topology.
- It does not decide graph routing except inside an agent-loop node.
- It does not persist graph checkpoints; that belongs to the graph module.
- It does not hide provider-specific metadata when users need it.
- It does not execute arbitrary workflow language source directly.
- It does not require every provider to support every modality or output
  strategy; capability profiles describe those differences.
- It does not make hidden network calls from tools, middleware, or stores unless
  the configured implementation does so explicitly.

## Package Shape

Each substantial harness feature gets its own module. This is not just file
organization; it is an ownership rule. If a feature will need its own traits,
errors, tests, middleware, or provider adapters, it belongs in its own submodule.

Target layout:

```text
src/harness/
  mod.rs
  agent_loop.rs
  cache.rs
  context.rs
  cost.rs
  embeddings.rs
  events.rs
  graph_runtime.rs
  limits.rs
  memory.rs
  message.rs
  middleware.rs
  model.rs
  prompt.rs
  providers.rs
  retry.rs
  runtime.rs
  stream.rs
  summarization.rs
  structured.rs
  store.rs
  testkit.rs
  tool.rs
  usage.rs
```

The current crate already has top-level `chat.rs`, `model.rs`, and `tool.rs`.
Those can either stay as public re-exports or move under `harness/` once the API
settles.

Feature ownership:

- `agent_loop`: default model-tool-model loop.
- `cache`: prompt, provider prompt/KV-cache, response, summary, and artifact
  cache policy.
- `context`: `RunConfig`, `RunContext`, inherited metadata, runtime values.
- `cost`: model pricing, budget policy, and cost rollups.
- `embeddings`: embedding providers, vector stores, retrievers, indexing, and
  retrieval-context records.
- `events`: typed harness events, sinks, streams, redaction adapters.
- `graph_runtime`: explicit state graphs, node commands, reducers,
  checkpointing, HITL, run records, and graph execution blueprints.
- `limits`: model-call, tool-call, concurrency, timeout, and recursion policy.
- `memory`: short-term thread memory and long-term stores.
- `message`: structured messages, content blocks, tool call correlation.
- `middleware`: before/after/wrap hooks and middleware stack ordering.
- `model`: provider-neutral model traits, requests, responses, streams.
- `prompt`: prompt templates, rendering, and dynamic prompt context.
- `providers`: feature-gated provider adapters.
- `retry`: retry classification, backoff, attempt accounting.
- `runtime`: high-level `AgentHarness` builder/facade.
- `stream`: token streams, tool progress streams, event streams, adapters.
- `summarization`: context summaries, message compaction, summary provenance.
- `structured`: typed response formats and validation.
- `store`: JSONL, file, MongoDB, in-memory, and other persistence backends.
- `testkit`: fakes, recorders, deterministic ids, trajectory assertions.
- `tool`: tool traits, schemas, validation, execution, result formatting.
- `usage`: token accounting, cached token tracking, context-window estimates.

Feature details:

- [Context feature](context.md)
- [Model and provider feature](model.md)
- [Embeddings and retrieval feature](embeddings.md)
- [State graph runtime feature](state-graph.md)
- [Prompt feature](prompt.md)
- [Tool feature](tool.md)
- [Middleware feature](middleware.md)
- [Structured output feature](structured-output.md)
- [Limits, retry, fallback, and rate limiting](limits-retry.md)
- [Summarization feature](summarization.md)
- [Usage feature](usage.md)
- [Cost feature](cost.md)
- [Cache feature](cache.md)
- [Streaming feature](streaming.md)
- [Store feature](store.md)
- [Observability and events](observability.md)
- [Testkit feature](testkit.md)

## LangChain Feature Parity Map

This map is not a mandate to clone LangChain. It is a checklist of proven
surface area that RustAgents should intentionally support, adapt, or reject.

| LangChain area | Source | RustAgents harness implication |
| --- | --- | --- |
| `create_agent` factory | `libs/langchain_v1/langchain/agents/factory.py` | `AgentHarness` should compose model selection, tool execution, middleware, structured output, runtime context, and graph-node compatibility behind one builder while keeping traits reusable outside the facade. |
| Agent middleware | `libs/langchain_v1/langchain/agents/middleware/types.py` | Middleware needs before/after hooks, streaming delta hooks, and wrap hooks that can replace the model/tool call, inject commands, short-circuit, or jump to `model`, `tools`, or `end`. |
| Built-in middleware | `libs/langchain_v1/langchain/agents/middleware/*.py` | Ship focused middleware for summarization, context compression, transcript compression, retrieval compression, output compression, prompt cache layout guards, context editing, PII redaction, model/tool limits, retries, fallback, tool selection, human-in-the-loop, shell/file-search style privileged tools, and todo/task state. |
| Structured output | `libs/langchain_v1/langchain/agents/structured_output.py` | Support provider-native schemas and artificial tool-call schemas, with typed validation, retryable validation errors, union/oneOf variants, and configurable error handling. |
| Message model | `libs/core/langchain_core/messages/*.py` | Use typed content blocks for text, JSON, image, audio, file, tool call, tool result, reasoning, citations, refusal/safety, and provider extension data. |
| Content translation | `libs/core/langchain_core/messages/block_translators/*.py` | Provider adapters must translate to/from the canonical RustAgents message model without losing ids, tool-call chunks, reasoning, usage, or provider metadata. |
| Model profiles | `libs/core/langchain_core/language_models/model_profile.py` | Store model capability metadata: context limits, modalities, tool calling, tool-choice support, streaming tool chunks, structured output, reasoning output, temperature, attachments, status, and release dates. |
| Embeddings | `libs/core/langchain_core/embeddings/embeddings.py` | Define provider-neutral embedding traits for documents and queries, with batch, async, dimensionality, provider metadata, usage, cost, cache, and fake deterministic implementations. |
| OpenHuman agent graph | `openhuman#4261`, `src/openhuman/agent_graph/graph/*` | Add a LangGraph-style state-machine runtime: typed state reducers, async nodes, static/conditional/fork edges, Pregel super-steps, compile validation, cancellation, max-step guards, interrupts, and resume. |
| OpenHuman checkpointer | `openhuman#4261`, `src/openhuman/agent_graph/checkpoint/*` | Persist graph runs and checkpoints through a pluggable `Checkpointer`, with in-memory tests and durable SQLite-style production storage. |
| OpenHuman graph blueprints | `openhuman#4261`, `src/openhuman/agent_graph/blueprint/*` | Keep per-agent execution topology in `graph.rs`-style blueprints next to prompts, so "what the agent says" and "how the agent runs" are inspectable separately. |
| OpenHuman live turn graph | `openhuman#4261`, `src/openhuman/agent_graph/live/*` and `agent/harness/engine/core.rs` | Preserve the hot-path turn contract while making phases explicit: dispatch, parse, stop check, tools, compact, loop, finalize, max-iteration checkpoint. |
| Vector stores | `libs/core/langchain_core/vectorstores/base.py`, `in_memory.py` | Support add/update/delete/get-by-id, similarity search, score-threshold search, MMR search, metadata filters, async variants, and in-memory test stores. |
| Retrievers and indexing | `libs/core/langchain_core/retrievers.py`, `indexing/*.py` | Treat retrievers as query-to-document components with events, tags, metadata, and record-manager-backed incremental indexing for dedupe and cleanup. |
| Tool runtime injection | `langgraph.prebuilt.ToolRuntime` as re-exported by `libs/langchain_v1/langchain/tools/tool_node.py` | Tools should receive typed runtime context, state, store handles, stream writers, and cancellation handles through Rust parameters, not model-visible JSON schema fields. |
| Callback/tracer events | `libs/core/langchain_core/callbacks` and `libs/core/langchain_core/tracers` | Emit typed events for every lifecycle boundary and expose sinks for tracing, streaming, logs, tests, and future UI replay. |
| Runnables config | `libs/core/langchain_core/runnables/config.py` | `RunConfig` should carry tags, metadata, configurable values, concurrency, recursion, callbacks/events, and stable run identity through nested calls. |
| Retry/fallback/rate limit | `libs/core/langchain_core/runnables/retry.py`, `fallbacks.py`, `rate_limiters.py` | Policies should distinguish retryable transport errors, provider errors, validation errors, tool errors, budget failures, and rate-limit waits. |
| Cache | `libs/core/langchain_core/caches.py` | Separate local response cache from provider prompt/KV-cache reuse, preserve stable prefix layout, and include all behavior-affecting request fields in keys. |
| Stores and chat history | `libs/core/langchain_core/stores.py`, `chat_history.py` | Keep generic stores separate from conversation memory and graph checkpoints. |
| Standard tests | `libs/standard-tests` | Add reusable conformance tests so provider adapters prove tool calling, structured output, streaming, usage, callbacks/events, multimodal input, Unicode, and error behavior. |

## Core Types

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
    pub max_concurrency: usize,
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

## Messages

Messages are the harness's internal data model. Raw strings should only appear
at API boundaries.

```rust
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

pub enum ContentBlock {
    Text(String),
    Json(serde_json::Value),
    Image(ImageRef),
    Audio(AudioRef),
    File(FileRef),
    ToolCall(ToolCallBlock),
    ToolResult(ToolResultBlock),
    Reasoning(ReasoningBlock),
    Citation(CitationBlock),
    Refusal(RefusalBlock),
    ProviderExtension(serde_json::Value),
}

pub struct AssistantMessage {
    pub id: Option<String>,
    pub content: Vec<ContentBlock>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub provider: Option<ProviderMetadata>,
}

pub struct ToolMessage {
    pub tool_call_id: String,
    pub name: String,
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}
```

Required message properties:

- stable role
- structured content blocks
- assistant tool calls
- tool call ids
- tool result correlation
- usage metadata
- provider extension escape hatch
- invalid or partially parsed tool calls for streaming/provider repair
- reasoning, citation, refusal, and safety blocks when providers expose them
- provider response ids for continuation/resume APIs

The first public API can keep `ChatMessage` as a simple compatibility type, but
the harness internals should move toward richer messages before provider
integrations are added.

Provider adapters are responsible for converting between provider payloads and
this model. The conversion must be round-trip safe for supported fields and must
preserve unknown provider fields in `ProviderExtension` rather than dropping
them. Streaming adapters must merge message chunks deterministically.

## Model Registry

The model registry maps names to provider-neutral implementations.

```rust
pub struct ModelRegistry<State, Ctx = ()> {
    models: HashMap<ModelName, Arc<dyn ChatModel<State, Ctx>>>,
    default: Option<ModelName>,
}

#[async_trait]
pub trait ChatModel<State, Ctx = ()>: Send + Sync {
    async fn invoke(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ModelRequest,
    ) -> Result<ModelResponse>;

    async fn stream(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ModelRequest,
    ) -> Result<ModelStream>;
}
```

`ModelRequest` should contain:

- model id or registry alias
- messages
- tool declarations
- tool choice policy
- response format
- temperature
- max tokens
- stop sequences
- timeout
- retry policy
- tags and metadata
- provider options
- capability requirements such as `requires_tool_calling`,
  `requires_structured_output`, `requires_image_input`, or
  `requires_tool_call_streaming`
- cache policy
- rate-limit policy
- continuation or previous-response id where a provider supports it

`ModelResponse` should contain:

- assistant message
- usage
- finish reason
- raw provider metadata
- structured response when requested
- provider response id
- safety/refusal metadata
- retry and cache metadata
- elapsed time and timing breakdown

Provider integrations should be optional features:

- `provider-openai`
- `provider-anthropic`
- `provider-ollama`
- `provider-mock`

Every provider adapter must expose a `ModelProfile`. Middleware and builders
should use profiles to reject impossible requests early, choose provider-native
structured output only when supported, reserve context window budget, and decide
whether streamed tool-call chunks can be trusted.

## Tool Registry

The tool registry owns available tools and their schemas.

```rust
pub struct ToolRegistry<State, Ctx = ()> {
    tools: HashMap<ToolName, Arc<dyn Tool<State, Ctx>>>,
}

#[async_trait]
pub trait Tool<State, Ctx = ()>: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> ToolSchema;

    async fn call(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        call: ToolCall,
    ) -> Result<ToolResult>;
}
```

Tool schema requirements:

- name
- description
- JSON schema compatible input shape
- optional output schema
- safety metadata
- timeout override
- retry override
- model-visible flag for each argument
- injected-runtime argument declarations that are hidden from model schemas
- side-effect and idempotency metadata
- confirmation policy for destructive operations
- artifact output policy

Tool call requirements:

- `id`
- `name`
- `arguments`
- provider metadata
- originating model call id
- validation status
- retry attempt

Tool result requirements:

- `tool_call_id`
- `name`
- content
- raw structured value
- elapsed time
- error flag
- artifact references
- user-visible summary
- redacted event payload

Tool names should default to ASCII `snake_case`. The registry should reject
duplicate names and invalid names.

## Agent Loop

The default loop is the LangChain-style model-tool loop:

```text
input messages
  -> build request
  -> call model
  -> if assistant has tool calls:
       validate tool calls
       execute tools
       append tool messages
       repeat
  -> final assistant message
```

Detailed lifecycle:

1. Create `RunConfig` and `RunContext`.
2. Load short-term memory for `thread_id` if configured.
3. Normalize input into messages.
4. Apply prompt templates and dynamic context.
5. Select model.
6. Select exposed tools.
7. Run `before_model` middleware, including prompt/cache-layout guards and
   pre-call compression.
8. Invoke or stream the model through `wrap_model` middleware.
9. Run `on_model_delta` middleware for streamed chunks.
10. Run `after_model` middleware, including post-call compression and summary
    persistence.
11. Emit model events and append assistant message.
12. If tool calls exist, validate name, schema, and limits.
13. Run `before_tool` middleware per call.
14. Execute tools serially or concurrently according to policy.
15. Run `on_tool_delta` middleware for tool progress streams.
16. Run `after_tool` middleware per result.
17. Append tool messages.
18. Repeat until no tool calls remain.
19. Validate structured output if configured.
20. Persist short-term memory.
21. Emit final event and return `AgentRun`.

Hard limits:

- `max_model_calls`
- `max_tool_calls`
- `max_concurrency`
- wall-clock timeout
- per-call timeout
- retry budget

The loop must fail closed when a limit is reached.

## Middleware

Middleware is the main extension point for behavior that cuts across providers,
tools, and graph nodes.

```rust
#[async_trait]
pub trait Middleware<State, Ctx = ()>: Send + Sync {
    async fn before_model(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &mut ModelRequest,
    ) -> Result<()>;

    async fn on_model_delta(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        delta: &mut ModelDelta,
    ) -> Result<()>;

    async fn after_model(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        response: &mut ModelResponse,
    ) -> Result<()>;

    async fn before_tool(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        call: &mut ToolCall,
    ) -> Result<()>;

    async fn on_tool_delta(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        delta: &mut ToolDelta,
    ) -> Result<()>;

    async fn after_tool(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        result: &mut ToolResult,
    ) -> Result<()>;

    async fn on_error(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        error: &RustAgentsError,
    ) -> Result<()>;
}
```

Middleware ordering is stable and explicit. Middleware runs in registration
order for `before_*` hooks, registration order for streaming delta hooks, and
reverse order for `after_*` hooks. Wrap hooks should surround the full model or
tool operation when middleware needs setup, streaming inspection, and teardown
as one unit.

Built-in middleware candidates:

- tracing middleware
- retry middleware
- timeout middleware
- model fallback middleware
- token-bucket rate limiter middleware
- prompt cache layout guard middleware
- message trimming middleware
- summarization middleware
- context compression middleware
- transcript compression middleware
- retrieval compression middleware
- streaming delta compression middleware
- output compression middleware
- context editing middleware
- tool allowlist middleware
- dynamic tool selection middleware
- guardrail middleware
- PII detection/redaction middleware
- human-in-the-loop middleware
- shell/filesystem privilege boundary middleware
- structured output validator
- rate limiter

Wrap hooks should exist in addition to before/after hooks. A wrap hook receives a
request plus a handler and can call the handler, replace the request, retry,
fallback to another model/tool, short-circuit with a response, or return a
control command. Before/after hooks are simpler and should remain available for
common mutation and observation cases.

## Memory And Stores

Memory and storage are related but not the same feature. `memory` owns
conversation semantics. `store` owns persistence backends.

Memory has two layers conceptually:

```text
short-term memory: thread-scoped conversation state
long-term store: cross-thread application data
```

Short-term memory:

- keyed by `thread_id`
- loaded before an agent loop
- updated after successful loop completion
- optionally trimmed or summarized
- useful for conversation continuity

Stores:

- available through `RunContext`
- namespaced
- typed where possible
- usable by tools and middleware
- not automatically injected into prompts unless middleware does it
- reusable by memory, event recording, tool artifacts, and web UIs

Suggested traits:

```rust
#[async_trait]
pub trait ShortTermMemory<State>: Send + Sync {
    async fn load(&self, thread_id: &ThreadId) -> Result<Option<State>>;
    async fn save(&self, thread_id: &ThreadId, state: &State) -> Result<()>;
}
```

The storage layer should be a separate harness feature:

```rust
#[async_trait]
pub trait Store: Send + Sync {
    async fn get(&self, key: StoreKey) -> Result<Option<StoreValue>>;
    async fn put(&self, key: StoreKey, value: StoreValue) -> Result<()>;
    async fn delete(&self, key: StoreKey) -> Result<()>;
    async fn scan(&self, prefix: StoreKeyPrefix) -> Result<Vec<StoreRecord>>;
}

#[async_trait]
pub trait AppendStore: Send + Sync {
    async fn append(&self, stream: StoreStream, value: StoreValue) -> Result<StoreOffset>;
    async fn read_from(&self, stream: StoreStream, offset: StoreOffset) -> Result<Vec<StoreRecord>>;
}

pub enum StoreValue {
    Json(serde_json::Value),
    Bytes(Vec<u8>),
    Text(String),
}
```

Initial store backends:

- `InMemoryStore`: deterministic tests and examples.
- `JsonlStore`: append-only local development, replayable event logs, and cheap
  debugging.
- `FileStore`: local artifacts such as tool outputs, provider payload snapshots,
  and prompt fixtures.
- `MongoStore`: durable application/runtime records for server deployments.

Later store backends:

- SQLite for single-node durable local apps.
- Postgres for multi-tenant production apps.
- S3-compatible blob store for large artifacts.
- Redis for short-lived cache/session data.

Store data classes:

- run records
- thread records
- normalized messages
- event envelopes
- tool call records
- model call records
- structured outputs
- user/application memory
- tool artifacts and blobs

Backend selection should be per store namespace:

```rust
let stores = StoreRegistry::new()
    .register("events", JsonlStore::new("./data/events.jsonl"))
    .register("threads", MongoStore::new(mongo, "threads"))
    .register("artifacts", FileStore::new("./data/artifacts"));
```

Store events should flow through `harness::events` or the registry event bus:

- `store.read`
- `store.write`
- `store.append`
- `store.delete`
- `store.error`

Sensitive store fields must support redaction before event emission.

## Structured Output

Structured output should support:

- provider-native schema mode
- tool-call fallback mode
- JSON parsing mode for simple local models

```rust
pub enum ResponseFormat {
    Text,
    JsonSchema(JsonSchema),
    ProviderNative(JsonSchema),
    ToolStrategy { tool_name: String, schema: JsonSchema },
}
```

The final run result should keep messages and structured output separate:

```rust
pub struct AgentRun<State, Output = ()> {
    pub state: State,
    pub messages: Vec<Message>,
    pub structured_response: Option<Output>,
    pub events: Vec<AgentEvent>,
}
```

## Events And Streaming

The harness event stream should be typed, not a string callback.

```rust
pub enum AgentEvent {
    RunStarted { run_id: RunId, thread_id: Option<ThreadId> },
    ModelStarted { call_id: CallId, model: ModelName },
    ModelDelta { call_id: CallId, delta: MessageDelta },
    ModelCompleted { call_id: CallId, usage: Option<Usage> },
    ToolStarted { call_id: CallId, tool_name: ToolName },
    ToolDelta { call_id: CallId, delta: ToolDelta },
    ToolCompleted { call_id: CallId, tool_name: ToolName },
    MiddlewareStarted { name: String },
    MiddlewareCompleted { name: String },
    RetryScheduled { call_id: CallId, attempt: usize },
    Custom { name: String, payload: serde_json::Value },
    RunCompleted { run_id: RunId },
    RunFailed { run_id: RunId, error: String },
}
```

Streaming modes:

- `messages`: model deltas and final messages
- `tools`: tool start, progress, result
- `updates`: state or memory updates
- `events`: all low-level events
- `final`: final output only

## Errors

Harness errors should distinguish:

- invalid request
- missing model
- missing tool
- invalid tool schema
- invalid tool arguments
- provider authentication failure
- provider rate limit
- provider server error
- timeout
- retry exhausted
- structured output validation failure
- middleware failure
- memory failure

Retry policy should only retry explicitly retryable classes by default:

- network interruption
- timeout
- rate limit
- provider 5xx

Do not retry authentication, schema, malformed request, or missing tool errors
unless a user explicitly overrides policy.

## Testkit

`harness::testkit` should be part of the early API.

Utilities:

- `FakeChatModel`
- `ScriptedChatModel`
- `FakeStreamingModel`
- `FakeTool`
- `InMemoryShortTermMemory`
- `InMemoryStore`
- `EventRecorder`
- deterministic ids
- deterministic clock
- trajectory assertions

Example trajectory assertion:

```rust
assert_trajectory(run.events())
    .model_called("default")
    .tool_called("lookup_user")
    .model_called("default")
    .completed();
```

## Implementation Milestones

### H1: Current Minimal Traits

- Keep `ChatMessage`.
- Keep `ChatModel`.
- Keep `Tool`.
- Add better tool call ids.

### H2: Registries And Context

- Add `ModelRegistry`.
- Add `ToolRegistry`.
- Add `RunConfig`.
- Add `RunContext`.

### H3: Agent Loop

- Implement model-tool loop.
- Enforce limits.
- Add fake model and fake tool tests.

### H4: Middleware And Events

- Add middleware stack.
- Add typed events.
- Add event recorder.

### H5: Memory And Structured Output

- Add short-term memory trait.
- Add store trait.
- Add structured response format.

### H6: Providers

- Add feature-gated provider crates or modules.
- Start with mock and one hosted provider.
