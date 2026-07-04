# Harness Runtime: Tools, Agent Loop, Middleware, Memory

Continues from [`README.md`](README.md): tool registry, agent loop,
middleware, and memory/stores.

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
        error: &TinyAgentsError,
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

