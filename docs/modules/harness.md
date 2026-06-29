# Harness Module Specification

The harness is the orchestration layer around LLM calls. It owns model
registration, tool registration, prompt assembly, middleware, memory, event
streaming, tracing, retries, limits, and test support.

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
- LangChain store and chat history code:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/stores.py>
  and
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/chat_history.py>

## Responsibilities

- Normalize user input into structured messages.
- Build model requests from messages, prompts, tools, memory, and config.
- Dispatch model calls through provider-neutral traits.
- Dispatch tool calls through a registry with schema validation.
- Run the standard model-tool-model agent loop.
- Apply middleware before and after model calls, tool calls, retries, and errors.
- Enforce model-call limits, tool-call limits, timeouts, and retry policy.
- Emit typed events for tracing, streaming, and tests.
- Persist short-term thread memory when configured.
- Expose durable stores through runtime context.
- Store run data, messages, events, tool artifacts, and application records
  through pluggable backends.
- Provide deterministic test utilities.

## Non-Responsibilities

- It does not own graph topology.
- It does not decide graph routing except inside an agent-loop node.
- It does not persist graph checkpoints; that belongs to the graph module.
- It does not hide provider-specific metadata when users need it.
- It does not execute arbitrary workflow language source directly.

## Package Shape

Each substantial harness feature gets its own module. This is not just file
organization; it is an ownership rule. If a feature will need its own traits,
errors, tests, middleware, or provider adapters, it belongs in its own submodule.

Target layout:

```text
src/harness/
  mod.rs
  agent_loop.rs
  context.rs
  events.rs
  limits.rs
  memory.rs
  message.rs
  middleware.rs
  model.rs
  prompt.rs
  providers.rs
  retry.rs
  runtime.rs
  structured.rs
  store.rs
  testkit.rs
  tool.rs
```

The current crate already has top-level `chat.rs`, `model.rs`, and `tool.rs`.
Those can either stay as public re-exports or move under `harness/` once the API
settles.

Feature ownership:

- `agent_loop`: default model-tool-model loop.
- `context`: `RunConfig`, `RunContext`, inherited metadata, runtime values.
- `events`: typed harness events, sinks, streams, redaction adapters.
- `limits`: model-call, tool-call, concurrency, timeout, and recursion policy.
- `memory`: short-term thread memory and long-term stores.
- `message`: structured messages, content blocks, tool call correlation.
- `middleware`: before/after/wrap hooks and middleware stack ordering.
- `model`: provider-neutral model traits, requests, responses, streams.
- `prompt`: prompt templates, rendering, and dynamic prompt context.
- `providers`: feature-gated provider adapters.
- `retry`: retry classification, backoff, attempt accounting.
- `runtime`: high-level `AgentHarness` builder/facade.
- `structured`: typed response formats and validation.
- `store`: JSONL, file, MongoDB, in-memory, and other persistence backends.
- `testkit`: fakes, recorders, deterministic ids, trajectory assertions.
- `tool`: tool traits, schemas, validation, execution, result formatting.

Feature details:

- [Store feature](harness/store.md)

## Core Types

```rust
pub struct AgentHarness<State, Ctx = ()> {
    models: ModelRegistry<State, Ctx>,
    tools: ToolRegistry<State, Ctx>,
    middleware: MiddlewareStack<State, Ctx>,
    memory: Option<Arc<dyn ShortTermMemory<State>>>,
    stores: StoreRegistry,
    policy: RunPolicy,
}

pub struct RunConfig {
    pub run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
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
}
```

`RunConfig` is serializable invocation policy and identity. `RunContext` is the
runtime dependency container. This split keeps tests deterministic and prevents
global singletons.

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

The first public API can keep `ChatMessage` as a simple compatibility type, but
the harness internals should move toward richer messages before provider
integrations are added.

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

`ModelResponse` should contain:

- assistant message
- usage
- finish reason
- raw provider metadata
- structured response when requested

Provider integrations should be optional features:

- `provider-openai`
- `provider-anthropic`
- `provider-ollama`
- `provider-mock`

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

Tool call requirements:

- `id`
- `name`
- `arguments`
- provider metadata

Tool result requirements:

- `tool_call_id`
- `name`
- content
- raw structured value
- elapsed time
- error flag

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
7. Run `before_model` middleware.
8. Invoke or stream the model.
9. Run `after_model` middleware.
10. Emit model events and append assistant message.
11. If tool calls exist, validate name, schema, and limits.
12. Run `before_tool` middleware per call.
13. Execute tools serially or concurrently according to policy.
14. Run `after_tool` middleware per result.
15. Append tool messages.
16. Repeat until no tool calls remain.
17. Validate structured output if configured.
18. Persist short-term memory.
19. Emit final event and return `AgentRun`.

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
order for `before_*` hooks and reverse order for `after_*` hooks.

Built-in middleware candidates:

- tracing middleware
- retry middleware
- timeout middleware
- message trimming middleware
- summarization middleware
- tool allowlist middleware
- guardrail middleware
- structured output validator
- rate limiter

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
