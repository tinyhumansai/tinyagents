# RustAgents System Specification

RustAgents is a Rust-native LLM application framework inspired by LangChain,
LangGraph, and CodeAct-style recursive language-model runtimes. The system is
organized around five modules:

1. the harness
2. the graph
3. the registry
4. the expressive language
5. the REPL language

The goal is to make agent systems easy to define, inspect, run, test, and
eventually serialize without hiding the Rust types that make production systems
reliable.

## Reference Positioning

RustAgents should synthesize the reference systems rather than clone any one of
them:

- LangGraph contributes the durable execution model: explicit state graphs,
  virtual `START` and `END`, Pregel-style supersteps, reducers/channels,
  commands, `Send` fanout, checkpointing, interrupts, subgraphs, streaming, and
  time travel.
- LangChain contributes the harness model: provider-neutral models, tools,
  middleware, runtime context, memory, retrieval, structured output, tracing,
  usage, cost, and conformance tests for integrations.
- `rust-langgraph` shows the Rust-facing precedent for a stateful graph runtime
  with nodes, conditional edges, checkpoints, streaming, optional model
  adapters, and ReAct/tool helpers. RustAgents should go deeper on typed state,
  harness composition, registries, and language-backed graph definitions.
- OpenHuman PR #4261 contributes the closest product-shaped precedent: a
  harness-decoupled graph engine, persistent checkpoints, HITL, graph
  observability, blueprints, JSON-RPC run control, and a behavior-preserving
  cutover from an implicit turn loop to an explicit phase machine.
- RLM contributes the REPL/code-act model: context and prompts as runtime
  values, recursive sub-model or sub-agent calls as functions, persistent
  session variables, trajectory logging, and sandbox choices.

The target architecture is therefore layered: the harness owns model/tool
execution and policies, the graph owns deterministic state transition and
durability, the registry owns named capabilities, `.rag` owns serializable graph
blueprints, and `.ragsh` owns capability-bound interactive orchestration. No
layer should bypass another layer's safety, policy, observability, or test
contracts.

## Detailed Module Docs

- [Harness module](../modules/harness/README.md)
  - [Context](../modules/harness/context.md)
  - [Model and providers](../modules/harness/model.md)
  - [Embeddings and retrieval](../modules/harness/embeddings.md)
  - [Prompt](../modules/harness/prompt.md)
  - [Tool](../modules/harness/tool.md)
  - [Middleware](../modules/harness/middleware.md)
  - [Sub-agent and orchestrator steering](../modules/harness/subagent-steering.md)
  - [Structured output](../modules/harness/structured-output.md)
  - [Limits, retry, fallback, and rate limiting](../modules/harness/limits-retry.md)
  - [Summarization](../modules/harness/summarization.md)
  - [Usage](../modules/harness/usage.md)
  - [Cost](../modules/harness/cost.md)
  - [Cache](../modules/harness/cache.md)
  - [Streaming](../modules/harness/streaming.md)
  - [Store](../modules/harness/store.md)
  - [Observability and events](../modules/harness/observability.md)
  - [Testkit](../modules/harness/testkit.md)
- [Graph module](../modules/graph/README.md)
  - [Package and core types](../modules/graph/package.md)
  - [Builder and compile contract](../modules/graph/builder.md)
  - [Node model](../modules/graph/nodes.md)
  - [State, channels, and updates](../modules/graph/state-channels.md)
  - [Edges, routing, commands, and sends](../modules/graph/routing.md)
  - [Execution model and parallelization](../modules/graph/execution.md)
  - [Parallel agents and context forking](../modules/graph/parallel-agents-forking.md)
  - [Checkpointing, durability, state inspection, and time travel](../modules/graph/checkpointing.md)
  - [Interrupts and resume](../modules/graph/interrupts.md)
  - [Streaming and events](../modules/graph/streaming.md)
  - [Observability and tracing](../modules/graph/observability.md)
  - [Runtime context and policies](../modules/graph/runtime-policy.md)
  - [Fault tolerance](../modules/graph/fault-tolerance.md)
  - [Subgraphs](../modules/graph/subgraphs.md)
  - [Sub-agents and recursion](../modules/graph/subagents-recursion.md)
  - [Memory and stores boundary](../modules/graph/memory-boundary.md)
  - [Visualization, introspection, and testkit](../modules/graph/visualization-testkit.md)
  - [Implementation milestones](../modules/graph/milestones.md)
- [Registry module](../modules/registry/README.md)
  - [Design](../modules/registry/design.md)
  - [Model catalog and local snapshots](../modules/registry/model-catalog.md)
- [Expressive language module](../modules/expressive-language/README.md)
- [REPL language module](../modules/repl-language/README.md)
  - [Design](../modules/repl-language/design.md)

Docs should follow the module layout. Do not place standalone specification
files directly in `docs/` or `docs/modules/`; each high-level topic should have
its own directory with a `README.md` entrypoint and any supporting files beside
it.

## Design Goals

- Make simple agent workflows concise.
- Make complex workflows explicit, inspectable, and testable.
- Treat graph execution as a first-class runtime, not an incidental callback
  chain.
- Keep model providers, tools, memory, and tracing behind stable traits.
- Support both Rust builder APIs and a compact expressive language for workflow
  definitions.
- Support a capability-bound REPL language for interactive graph and harness
  orchestration.
- Allow agents to author, inspect, compile, and run graph blueprints through the
  same registry-bound compiler path used by human-authored `.rag` files.
- Allow parent orchestrators and humans to steer orchestrator agents and
  sub-agents through typed, policy-checked, observable commands.
- Prefer deterministic state transitions around inherently nondeterministic LLM
  calls.
- Keep every generated or hand-authored graph explainable as topology,
  capabilities, policies, state channels, checkpoints, and events.

## Module 1: Harness

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
    async fn on_error(&self, ctx: &mut RunContext<Ctx>, error: &RustAgentsError) -> Result<()>;
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

## Module 2: Graph

The graph is the workflow runtime. It executes stateful nodes, applies state
updates, follows direct or conditional edges, records execution history, handles
interrupts, and returns a final state.

The first implementation can stay sequential, but the module should be designed
toward LangGraph's durable execution model: compiled graphs, virtual `START` and
`END` nodes, supersteps, reducer-driven state updates, checkpoints, interrupts,
commands, streaming, and subgraphs.

### Source Inspiration

The graph design is informed by LangGraph's docs on the graph API, reducers,
commands, persistence, checkpointers, interrupts, streaming, subgraphs, and fault
tolerance:

- <https://docs.langchain.com/oss/python/langgraph/graph-api>
- <https://docs.langchain.com/oss/python/langgraph/persistence>
- <https://docs.langchain.com/oss/python/langgraph/checkpointers>
- <https://docs.langchain.com/oss/python/langgraph/interrupts>
- <https://docs.langchain.com/oss/python/langgraph/streaming>
- <https://docs.langchain.com/oss/python/langgraph/event-streaming>
- <https://docs.langchain.com/oss/python/langgraph/use-subgraphs>
- <https://docs.langchain.com/oss/python/langgraph/fault-tolerance>

### Responsibilities

- Store named nodes.
- Store direct and conditional edges.
- Validate graph structure at compile time.
- Produce an immutable executable graph.
- Run async node handlers.
- Route based on node output or command output.
- Apply partial state updates through reducers.
- Enforce recursion limits.
- Persist checkpoints at safe boundaries.
- Support interrupts and resume.
- Stream typed execution events.
- Write readable execution status records for graph runs.
- Maintain append-only graph event journals for external listeners.
- Cache derived graph observability projections without making them the source
  of truth.
- Return final state and execution history.
- Support graph visualization and serialization later.

### Core Concepts

`State` is user-owned application state. RustAgents should never require a
specific state shape for hand-written Rust graphs.

`Node<State>` is an async unit of work.

`NodeOutput<State>` controls execution in the current scaffold:

- `Continue(State)` follows a direct edge.
- `Route { state, route }` follows a conditional edge.
- `End(State)` stops execution.

The target design should evolve this into partial updates and commands:

```rust
pub enum NodeResult<Update> {
    Update(Update),
    Command(Command<Update>),
    Interrupt(Interrupt),
}

pub struct Command<Update> {
    pub update: Option<Update>,
    pub goto: Vec<NodeId>,
    pub resume: Option<serde_json::Value>,
}
```

`GraphBuilder<State, Update>` should own graph construction. `CompiledGraph`
should own execution. This separates user-friendly mutation during setup from a
validated immutable runtime.

```rust
let graph = GraphBuilder::new()
    .add_node("agent", agent_node)
    .add_node("tools", tools_node)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", route_agent)
    .add_edge("tools", "agent")
    .compile()?;
```

### State Updates And Reducers

LangGraph nodes return partial state updates. RustAgents should adopt the same
direction because it enables parallel execution, replay, checkpointing, and
clearer node contracts.

The default reducer should be overwrite. Users should be able to opt into
reducers for fields that accumulate values:

- append list
- merge messages by id
- set union
- numeric min/max
- custom reducer

Possible Rust shape:

```rust
pub trait Reducer<T>: Send + Sync {
    fn reduce(&self, current: T, update: T) -> Result<T>;
}

pub trait StateReducer<State, Update>: Send + Sync {
    fn apply(&self, state: State, update: Update) -> Result<State>;
}
```

For milestone 1, whole-state updates are acceptable. For durable parallel graph
execution, partial updates and reducers should be introduced before
checkpoint/resume semantics harden.

### Graph Lifecycle

1. Define state.
2. Define update type if partial updates are enabled.
3. Create graph builder.
4. Add nodes.
5. Add direct or conditional edges.
6. Add `START` edge.
7. Compile and validate the graph.
8. Run graph with initial state and runtime config.
9. Inspect final state, checkpoints, events, and visited nodes.

### Routing Semantics

Direct routing:

```text
START -> agent -> summarize -> END
```

Conditional routing:

```text
START -> agent
agent --tool--> tools
agent --final--> END
tools ---------> agent
```

Conditional routes may start as explicit strings. Later versions should support
typed route enums or route newtypes so Rust users can avoid typo-prone strings.

Nodes should not mix static outgoing edges and dynamic command-based routing in
the same execution mode unless the behavior is deliberately specified. A strict
compile-time validation rule is preferable: a node has either normal outgoing
edges or command routing, not both.

### Supersteps

The target executor should be superstep-based:

1. Take the current active node set.
2. Run all active nodes for the step, respecting concurrency policy.
3. Collect partial state updates, commands, interrupts, and errors.
4. Apply reducers at the step boundary.
5. Persist a checkpoint.
6. Select the next active nodes.
7. Stop when the active set is empty or reaches `END`.

The first implementation can run one node at a time, but checkpointing and
parallel execution should use superstep boundaries as the durable unit. Do not
checkpoint mid-node.

### Checkpointing And Persistence

Graph checkpointing is not the same as harness memory. Checkpoints are
thread-scoped graph execution snapshots used for resume, interrupts, and fault
tolerance.

```rust
#[async_trait]
pub trait Checkpointer<State>: Send + Sync {
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId>;
    async fn get(&self, thread_id: &str, checkpoint_id: Option<&str>) -> Result<Option<Checkpoint<State>>>;
    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>>;
}
```

A checkpoint should contain:

- thread id
- checkpoint id
- parent checkpoint id
- namespace
- state snapshot
- next active nodes
- completed tasks for the superstep
- pending writes
- interrupts
- metadata

Interrupted or failed nodes may rerun from the beginning. Node authors must make
side effects idempotent or isolate side effects behind tools/middleware that can
record exactly-once intent.

### Interrupts And Resume

Interrupts support human-in-the-loop and external approval flows.

```rust
pub struct Interrupt {
    pub id: String,
    pub node: NodeId,
    pub payload: serde_json::Value,
}
```

Resume should use a command-style API:

```rust
graph.resume(
    RunConfig::thread("support-123"),
    Command::resume(json!({ "approved": true })),
).await?;
```

The default semantic should match LangGraph: resuming restarts the interrupted
node and replays until the interrupt point using stored resume values. That is
more durable than trying to suspend an async Rust stack.

### Streaming

The graph should expose low-level runtime events, higher-level projections, a
status store, and optional durable replay for outside listeners. The canonical
feature references are:

- [Graph streaming and events](../modules/graph/streaming.md)
- [Graph observability and tracing](../modules/graph/observability.md)
- [Graph checkpointing and state inspection](../modules/graph/checkpointing.md)
- [Graph memory and stores boundary](../modules/graph/memory-boundary.md)

Low-level events:

- node started
- node completed
- node failed
- state update
- checkpoint saved
- task scheduled
- interrupt emitted
- route selected

High-level stream modes:

- values: full state snapshots
- updates: partial state updates
- messages: model/message deltas emitted by harness nodes
- debug: verbose executor events
- interrupts: interrupt payloads
- custom: user events

The graph should also expose a compact run-status record:

```rust
pub struct GraphRunStatus {
    pub run_id: RunId,
    pub root_run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub thread_id: Option<ThreadId>,
    pub graph_id: GraphId,
    pub checkpoint_id: Option<CheckpointId>,
    pub checkpoint_namespace: Vec<String>,
    pub status: ExecutionStatus,
    pub current_step: usize,
    pub active_nodes: Vec<NodeId>,
    pub pending_interrupts: Vec<InterruptId>,
    pub last_event_id: Option<EventId>,
    pub started_at: SystemTime,
    pub updated_at: SystemTime,
    pub ended_at: Option<SystemTime>,
    pub error: Option<GraphErrorSummary>,
}
```

Graph status records are not checkpoints. Checkpoints preserve resumable graph
state; status records summarize live and recent execution for observers. A
graph event journal should let listeners subscribe live or replay from a stored
offset by run id, root run id, thread id, graph id, node id, event kind, or
namespace. Derived projections such as latest status by thread, task timing
rollups, checkpoint summaries, and introspection snapshots may be cached when
they include source coordinates: run id, checkpoint id, namespace, step, event
offset, and projection version.

### Subgraphs

Subgraphs should be executable graphs that can be used as nodes.

Two modes are needed:

- shared-state subgraph: parent and child graph use the same state channels
- adapter subgraph: wrapper node maps parent state into child state and maps the
  child result back into parent state

Checkpoint namespaces are required so parent and child checkpoint ids do not
collide.

### Execution Guarantees

The graph runtime should guarantee:

- every visited node existed at validation time
- every configured edge points to an existing node
- conditional routes fail clearly when missing
- recursion limit failures are deterministic
- checkpoint writes happen at configured execution boundaries
- interrupted runs can be resumed only when checkpointing is configured
- final state is returned exactly once

The graph runtime should not guarantee:

- deterministic LLM output
- tool idempotency
- provider-specific retry behavior
- persistence across process restarts unless a checkpointer is configured
- exactly-once side effects inside node code

### Future Graph Features

- graph serialization to JSON
- Mermaid export
- parallel branches
- joins
- typed route enums
- static graph analysis
- graph diffing
- graph snapshots for tests
- durable task queue integration

## Module 3: Expressive Language

The expressive language is a compact way to define agent workflows without
writing all builder calls manually. It should compile into the same graph and
harness types as Rust code.

This language is not meant to replace Rust. It is a workflow definition layer for
fast iteration, examples, documentation, and eventually user-authored agent
plans.

It is also the safe boundary for agent-authored graph plans. A REPL or model may
propose `.rag` source, but that source must pass through the same parser,
diagnostics, registry binding, allowlist checks, review gates, and graph
compiler as human-authored source before it can run.

### Goals

- Make common agent graphs readable at a glance.
- Keep syntax close to graph intent.
- Compile into explicit RustAgents structures.
- Preserve source locations for helpful errors.
- Avoid embedding arbitrary code in the first version.
- Describe state channels, reducers, policies, subgraphs, sub-agents,
  interrupts, joins, and fanout as declarative graph primitives.
- Produce inspectable blueprints that can be reviewed, diffed, registered, and
  tested.

### Non-Goals

- It is not a general-purpose programming language.
- It is not a prompt templating language by itself.
- It should not execute untrusted code.
- It should not bypass Rust type checks for stateful logic.
- It should not install model-generated topology directly into the graph
  runtime.

### Initial Syntax Sketch

```rustagents
graph support_agent {
  defaults {
    recursion_limit 50
    checkpoint inherit
  }

  start agent

  channel messages messages
  channel tool_calls append

  node agent {
    kind agent
    model "default"
    prompt "You are a concise support agent."
    tools ["lookup_user", "create_ticket"]
    routes {
      tool_call -> tools
      final -> END
    }
  }

  node tools {
    kind tool_executor
    next agent
  }
}
```

### Minimal Grammar

```text
program       = graph_decl*
graph_decl    = "graph" ident "{" graph_item* "}"
graph_item    = start_decl | defaults_decl | channel_decl | node_decl | edge_decl
start_decl    = "start" ident
defaults_decl = "defaults" object
channel_decl  = "channel" ident reducer_ref
node_decl     = "node" ident "{" node_item* "}"
node_item     = kind_decl | model_decl | prompt_decl | tools_decl | next_decl | routes_decl
kind_decl     = "kind" ident
model_decl    = "model" string
prompt_decl   = "prompt" string
tools_decl    = "tools" "[" string_list? "]"
next_decl     = "next" ident
routes_decl   = "routes" "{" route_decl* "}"
route_decl    = ident "->" (ident | "END")
edge_decl     = ident "->" ident
```

The full language target is broader than this minimal grammar. It should grow
toward commands, `Send` fanout, joins/barriers, subgraphs, sub-agents,
`repl_agent` nodes, interrupts, registered route functions, graph defaults,
capability allowlists, blueprint provenance, and deterministic graph diffs. See
[the expressive language module](../modules/expressive-language/README.md) for
the canonical target.

### Compilation Pipeline

1. Parse source into an AST.
2. Validate identifiers and route targets.
3. Lower AST into graph builder calls.
4. Bind model and tool references through the harness.
5. Return a compiled workflow object.

### Error Requirements

Errors should include:

- file name when available
- line and column
- invalid token or missing token
- unknown node name
- duplicate node name
- missing start node
- route target that does not exist
- model or tool reference that is not registered in the harness

### Runtime Relationship

The expressive language should produce the same runtime structures as hand-written
Rust:

```text
source -> parser -> AST -> compiler -> StateGraph<State> + Harness bindings
```

The graph runtime should not know whether a graph came from Rust builders or the
expressive language.

For generated source, the runtime relationship is:

```text
REPL/model proposal -> .rag source or AST -> parser -> diagnostics -> resolver
  -> policy/review gate -> compiler -> GraphBuilder + Harness bindings
  -> CompiledGraph -> optional registry registration
```

## Package Layout

Target module layout:

```text
src/
  chat.rs
  error.rs
  graph.rs
  harness.rs
  language/
    ast.rs
    lexer.rs
    parser.rs
    compiler.rs
    mod.rs
  model.rs
  tool.rs
```

Provider implementations should live behind feature flags:

```text
src/providers/
  openai.rs
  anthropic.rs
  ollama.rs
  mock.rs
```

## Milestones

### Milestone 1: Core Runtime

- Chat message primitives.
- Model trait.
- Tool trait.
- State graph with direct and conditional edges.
- Basic tests and examples.

### Milestone 2: Harness

- Harness type.
- Model registry.
- Tool registry.
- Run context.
- Callback events.
- Run status store.
- Durable event journal.
- Cache-backed observability projections.
- Mock model and mock tool utilities.

### Milestone 3: Expressive Language Preview

- AST.
- Parser for a small graph definition language.
- Compiler into `StateGraph`.
- Helpful parse and validation errors.
- Example `.rag` or `.rustagents` workflow file.

### Milestone 4: Provider Integrations

- OpenAI chat model provider.
- Anthropic chat model provider.
- Local/mock provider.
- Provider feature flags.

### Milestone 5: Production Runtime Features

- Streaming events.
- Checkpointing.
- Resume support.
- Graph run status store.
- Graph event journal and listener replay.
- Graph export.
- Tracing integration.

## Open Questions

- Should the expressive language file extension be `.rag`, `.rustagents`, or
  something shorter?
- Should state schemas be declared in the language, or should state remain purely
  Rust-owned?
- Should graph nodes support typed route enums before serialization support?
- Should provider crates live in this crate behind feature flags or in separate
  crates?
- Should memory be synchronous, async, or both?
