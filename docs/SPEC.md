# RustAgents System Specification

RustAgents is a Rust-native LLM application framework inspired by LangChain and
LangGraph. The system is organized around three modules:

1. the harness
2. the graph
3. the expressive language

The goal is to make agent systems easy to define, inspect, run, test, and
eventually serialize without hiding the Rust types that make production systems
reliable.

## Design Goals

- Make simple agent workflows concise.
- Make complex workflows explicit, inspectable, and testable.
- Treat graph execution as a first-class runtime, not an incidental callback
  chain.
- Keep model providers, tools, memory, and tracing behind stable traits.
- Support both Rust builder APIs and a compact expressive language for workflow
  definitions.
- Prefer deterministic state transitions around inherently nondeterministic LLM
  calls.

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
- Manage per-run config such as run ids, thread ids, metadata, tags, deadlines,
  max concurrency, model limits, tool limits, and
  cancellation.
- Provide middleware hooks before and after model calls, tool calls, and errors.
- Emit typed events for observability and streaming.
- Enforce retry, timeout, model-call, tool-call, and recursion policies.
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
- tags and metadata

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
4. Run `before_model` middleware.
5. Invoke or stream the model.
6. Run `after_model` middleware.
7. If the assistant produced tool calls, validate and execute them.
8. Append tool result messages.
9. Repeat until no tool calls remain or limits are reached.
10. Persist updated short-term memory and return the final output.

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
    async fn before_model(&self, ctx: &mut RunContext<Ctx>, state: &State, request: &mut ModelRequest) -> Result<()>;
    async fn after_model(&self, ctx: &mut RunContext<Ctx>, state: &State, response: &mut ModelResponse) -> Result<()>;
    async fn before_tool(&self, ctx: &mut RunContext<Ctx>, state: &State, call: &mut ToolCall) -> Result<()>;
    async fn after_tool(&self, ctx: &mut RunContext<Ctx>, state: &State, result: &mut ToolResult) -> Result<()>;
    async fn on_error(&self, ctx: &mut RunContext<Ctx>, error: &RustAgentsError) -> Result<()>;
}
```

Expected middleware:

- retry and timeout policy
- prompt injection
- dynamic tool filtering
- guardrails
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

Every run should be traceable through typed events. At minimum, the harness
should emit:

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

The event stream should be structured data so it can later feed logs, OpenTelemetry,
or a custom UI.

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

The graph should expose low-level runtime events and higher-level projections.

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

### Goals

- Make common agent graphs readable at a glance.
- Keep syntax close to graph intent.
- Compile into explicit RustAgents structures.
- Preserve source locations for helpful errors.
- Avoid embedding arbitrary code in the first version.

### Non-Goals

- It is not a general-purpose programming language.
- It is not a prompt templating language by itself.
- It should not execute untrusted code.
- It should not bypass Rust type checks for stateful logic.

### Initial Syntax Sketch

```rustagents
graph support_agent {
  start agent

  node agent {
    model "default"
    prompt "You are a concise support agent."
    routes {
      tool_call -> tools
      final -> end
    }
  }

  node tools {
    tool_choice auto
    next agent
  }
}
```

### Minimal Grammar

```text
program       = graph_decl*
graph_decl    = "graph" ident "{" graph_item* "}"
graph_item    = start_decl | node_decl | edge_decl
start_decl    = "start" ident
node_decl     = "node" ident "{" node_item* "}"
node_item     = model_decl | prompt_decl | next_decl | routes_decl | tool_decl
model_decl    = "model" string
prompt_decl   = "prompt" string
next_decl     = "next" ident
routes_decl   = "routes" "{" route_decl* "}"
route_decl    = ident "->" (ident | "end")
tool_decl     = "tool_choice" ident
edge_decl     = ident "->" ident
```

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
