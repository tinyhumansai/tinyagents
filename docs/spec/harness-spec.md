# Harness Module Specification

The harness is the outer runtime for LLM applications. In LangChain terms, this
is the layer around a model call that owns the agent loop, prompt/context
assembly, tool execution, middleware, memory, streaming, tracing, retries, and
testability.

The harness must stay composable. It should not be a single monolithic `Agent`
type that hides every behavior. A direct model call, a model-plus-tools loop, and
a graph node that invokes a model should all share the same harness primitives.

### Source Inspiration

The harness design is informed by LangChain's docs on agents, chat models, tools,
runtime context, memory, structured output, middleware, streaming, tracing, and
testing:

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

### Responsibilities

- Register chat model providers.
- Resolve model calls from request overrides, reusable state, model hints,
  agent defaults, registry defaults, and fallback policy.
- Register tools and validate tool calls against schemas.
- Build model requests from state, prompts, memory, and runtime context.
- Apply prompt and message templates.
- Preserve provider prompt/KV-cache stability by keeping cacheable prompt
  prefixes deterministic and isolating volatile context near the tail of model
  requests.
- Manage per-run config such as run ids, thread ids, metadata, tags, deadlines,
  max concurrency, model limits, tool limits, and
  cancellation.
- Provide middleware hooks before and after model calls, tool calls, and errors.
- Provide middleware hooks during streaming model calls so compression,
  redaction, observability, and adaptive context algorithms can inspect deltas
  without replacing provider adapters.
- Emit typed events for observability and streaming.
- Write readable run status records for direct model calls, agent loops, and
  graph-node child harness calls.
- Maintain append-only event journals when durable listener replay is
  configured.
- Enforce retry, timeout, model-call, tool-call, and recursion policies.
- Accept sub-agent and orchestrator steering commands from humans, parent
  agents, graph supervisors, middleware, and tests at safe loop boundaries.
- Normalize model and tool errors into framework errors.
- Persist resolved model identity in responses, events, usage/cost records, run
  status, and durable agent or graph state so later calls can reuse it.
- Provide test doubles for models, tools, stores, clocks, and ids.

### Core Types

```rust
pub struct AgentHarness<State, Ctx = ()> {
    models: ModelRegistry<State, Ctx>,
    tools: ToolRegistry<State, Ctx>,
    middleware: MiddlewareStack<State, Ctx>,
    policy: RunPolicy,
}

pub struct RunConfig {
    pub run_id: String,
    pub thread_id: Option<String>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub timeout_ms: Option<u64>,
    pub max_model_calls: usize,
    pub max_tool_calls: usize,
}

pub struct RunContext<Ctx = ()> {
    pub config: RunConfig,
    pub data: Ctx,
    pub stores: StoreRegistry,
    pub events: EventSink,
}
```

`RunConfig` is stable invocation identity and policy. `RunContext` is the
per-run dependency bag. Keeping those separate prevents global state and makes
unit tests straightforward.

### Model Abstraction

Models should be provider-agnostic. The graph layer should never know whether a
node uses OpenAI, Anthropic, Ollama, a local model, or a test fake.

```rust
#[async_trait]
pub trait ChatModel<State>: Send + Sync {
    async fn invoke(
        &self,
        state: &State,
        request: ModelRequest,
    ) -> Result<ModelResponse>;

    async fn stream(
        &self,
        state: &State,
        request: ModelRequest,
    ) -> Result<ModelStream> {
        default_stream_from_invoke(self, state, request).await
    }
}
```

`ModelRequest` should grow beyond the current minimal version:

- model hints and reusable resolved-model policy
- messages
- tools available for this call
- tool choice policy
- response format
- model id/provider override
- temperature
- max tokens
- timeout
- retry policy
- local response cache policy
- provider prompt-cache policy
- cacheable prompt prefix boundaries
- ephemeral/non-cacheable context boundaries
- prompt layout fingerprint
- tags and metadata

`ModelResponse` and agent state should record a `ResolvedModel` with registry
name, provider, provider model id, catalog snapshot/entry when known, resolver
source, and fallback history. This record is the durable answer to which model
actually ran, and it may be reused by later calls when policy allows.

Provider prompt caching is different from local response caching. The harness
must support extreme prompt caching for providers with KV-cache or
prompt-prefix-cache behavior. That means request construction must be able to
mark stable message and tool-schema prefixes, preserve their byte/token order
across turns, and append volatile state, retrieved context, scratchpads, and
per-run metadata after those stable prefixes. Middleware that compresses,
trims, summarizes, or injects context must declare whether it changes the
cacheable prefix, the volatile tail, or only non-model-visible metadata.

The cache contract should prevent accidental KV-cache busting:

- stable system prompts, policy text, tool declarations, schema text, and
  reusable instruction blocks should have explicit prefix segment ids
- volatile values such as timestamps, run ids, retrieved documents, current
  tool results, and user-specific ephemeral context should stay out of the
  cacheable prefix unless a policy explicitly opts in
- request builders should preserve segment order and canonical serialization
- middleware must emit a cache-layout event when it mutates prompt segments
- tests should be able to assert whether a change preserves or invalidates the
  provider prompt-cache prefix

Initial provider implementations should be optional feature flags:

- `openai`
- `anthropic`
- `ollama`
- `mock`

### Message Model

Messages are the internal currency of the harness. The framework should not pass
raw strings after initial user input normalization.

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
```

The message model should preserve:

- role
- content blocks
- assistant tool calls
- tool call ids
- tool result ids
- usage metadata
- provider extensions

Tool call ids are mandatory once tool execution is implemented because they are
the correlation key between assistant requests and tool messages.

### Tool Abstraction

Tools are typed capabilities exposed to agents. The initial executor can accept
JSON arguments, but the registry should store schema metadata from the start.

```rust
#[async_trait]
pub trait Tool<State>: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> ToolSchema;
    async fn call(&self, state: &State, call: ToolCall) -> Result<ToolResult>;
}
```

Tool calls must be observable and replayable. Each call should record:

- tool name
- arguments
- result content
- raw provider result when available
- elapsed time
- error details

Tool names should be ASCII and `snake_case` by default. This keeps names
portable across providers that are strict about tool naming.

### Agent Loop

The default harness loop should be:

1. Build `RunContext`.
2. Load short-term memory for `thread_id` when configured.
3. Build a `ModelRequest`.
4. Run pre-request middleware that can edit prompts, context, cache layout,
   compression state, and provider options.
5. Run wrap middleware around the invoke or stream call for retry, fallback,
   rate limiting, tracing, and replacement.
6. Run streaming middleware while model deltas arrive, including compression,
   redaction, tool-call reconstruction, usage accounting, and adaptive
   cancellation.
7. Run post-response middleware that can validate, compress, summarize,
   persist, or transform the model response.
8. If the assistant produced tool calls, validate and execute them.
9. Append tool result messages.
10. Repeat until no tool calls remain or limits are reached.
11. Persist updated short-term memory and return the final output.

Limits are not optional. The harness should enforce:

- maximum model calls per run
- maximum tool calls per run
- maximum wall-clock duration
- maximum retries per call
- optional maximum concurrency for parallel tool calls

### Middleware

Middleware is the primary extension point for behavior that should not be baked
into the model or graph APIs.

```rust
#[async_trait]
pub trait Middleware<State, Ctx = ()>: Send + Sync {
    async fn before_agent(&self, ctx: &mut RunContext<Ctx>, state: &State) -> Result<()>;
    async fn after_agent(&self, ctx: &mut RunContext<Ctx>, state: &State, run: &mut AgentRun) -> Result<()>;
    async fn before_model(&self, ctx: &mut RunContext<Ctx>, state: &State, request: &mut ModelRequest) -> Result<()>;
    async fn on_model_delta(&self, ctx: &mut RunContext<Ctx>, state: &State, delta: &mut ModelDelta) -> Result<()>;
    async fn after_model(&self, ctx: &mut RunContext<Ctx>, state: &State, response: &mut ModelResponse) -> Result<()>;
    async fn before_tool(&self, ctx: &mut RunContext<Ctx>, state: &State, call: &mut ToolCall) -> Result<()>;
    async fn on_tool_delta(&self, ctx: &mut RunContext<Ctx>, state: &State, delta: &mut ToolDelta) -> Result<()>;
    async fn after_tool(&self, ctx: &mut RunContext<Ctx>, state: &State, result: &mut ToolResult) -> Result<()>;
    async fn on_error(&self, ctx: &mut RunContext<Ctx>, error: &TinyAgentsError) -> Result<()>;
}
```

Wrap middleware should also exist around model calls and tool calls. A
compression algorithm often needs to wrap the entire model operation so it can
prepare context before the call, inspect streaming deltas during the call, and
commit summaries or cache metadata after the final response.

Expected middleware:

- retry and timeout policy
- prompt injection
- prompt cache layout protection
- provider prompt-cache/KV-cache hints
- dynamic tool filtering
- guardrails
- context compression
- transcript compression
- retrieved-context compression
- output compression
- streaming delta compression
- message trimming
- summarization
- structured output validation
- tracing
- rate limiting

### Memory

Memory should be a harness capability. The graph runtime should handle
checkpointed graph execution; the harness should handle conversation and
application memory.

Memory is split into two concepts:

- short-term memory: thread-scoped conversation state, usually backed by graph
  checkpoints or a conversation checkpoint store
- long-term memory: cross-thread application data exposed through a store trait

Memory backends should start with:

- in-memory store for tests
- file-backed store for local development
- trait boundary for external stores

Trimming and summarization should be explicit policies, not hidden behavior.
Compression is a broader middleware family than summarization. The harness
should support pre-call compression of old messages and retrieved context,
during-call compression or redaction of streaming deltas, and post-call
compression of transcripts, tool artifacts, reasoning traces, and memory
records. Compression middleware must preserve provenance: the original source
ids, token estimates, cache segment ids, and enough metadata to explain why a
message was removed, replaced, or summarized.

### Structured Output

The harness should support typed output using two strategies:

- provider-native schema enforcement when the model supports it
- tool-call-based structured output fallback

The user-facing API should allow:

```rust
let output: MyType = harness
    .with_response_format(ResponseFormat::json_schema::<MyType>())
    .invoke(state)
    .await?
    .structured_response()?;
```

The final structured value should be separate from final chat messages so users
can inspect both.

### Observability

Every run should be traceable through typed events and readable through a
compact execution status store. The status store is the answer to "what is this
run doing now?"; the event stream and journal are the answer to "what happened?"

The canonical feature references are:

- [Harness observability and events](../modules/harness/observability.md)
- [Harness store](../modules/harness/store.md)
- [Harness streaming](../modules/harness/streaming.md)
- [Harness cache](../modules/harness/cache.md)

At minimum, the harness should emit:

- run started
- model requested
- model token delta
- model responded
- tool requested
- tool token or progress delta
- tool responded
- state update
- middleware started
- middleware completed
- retry scheduled
- route selected
- run completed
- run failed

The event stream should be structured data so it can feed logs,
OpenTelemetry, test recorders, durable JSONL/MongoDB journals, or a custom UI.

```rust
pub enum AgentEvent {
    RunStarted { run_id: String, thread_id: Option<String> },
    ModelStarted { call_id: String, model: String },
    ModelDelta { call_id: String, delta: MessageDelta },
    ModelCompleted { call_id: String, usage: Option<Usage> },
    ToolStarted { call_id: String, tool_name: String },
    ToolCompleted { call_id: String, tool_name: String },
    RetryScheduled { call_id: String, attempt: usize },
    RunCompleted { run_id: String },
    RunFailed { run_id: String, error: String },
}
```

The harness should also expose a compact run-status record:

```rust
pub struct HarnessRunStatus {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub component: ComponentId,
    pub status: ExecutionStatus,
    pub current_phase: HarnessPhase,
    pub model_calls: usize,
    pub tool_calls: usize,
    pub active_model_call: Option<CallId>,
    pub active_tool_calls: Vec<CallId>,
    pub last_event_id: Option<EventId>,
    pub usage: UsageTotals,
    pub cost: CostTotals,
    pub started_at: SystemTime,
    pub updated_at: SystemTime,
    pub ended_at: Option<SystemTime>,
    pub error: Option<HarnessErrorSummary>,
}
```

Status records are operational snapshots. They should not include full prompts,
tool outputs, or raw provider payloads. Event journals are append-only and
should support listener replay by stream offset. Derived observability
projections such as latest status, usage rollups, cost rollups, and timing
summaries may be cached, but every cached projection must include a source event
offset and projection version.

### Testability

The harness should ship a `testkit` module early. It should include:

- fake chat model with scripted responses
- fake streaming model
- fake tool
- in-memory stores
- deterministic run id generator
- deterministic clock
- event recorder
- trajectory assertions that check tool calls and state changes without relying
  on exact LLM prose

