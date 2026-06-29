# Registry Module Specification

Parent module: [Registry](README.md).

The registry module is the coordination layer for TinyAgents. It registers
agents, graphs, tools, models, stores, middleware, and listeners, then provides
an event-friendly runtime surface for external systems such as web UIs, CLIs,
logs, tests, and distributed supervisors.

The registry is deliberately simple at its core: a Rust registry of named
components plus an event bus. It should not become a hidden global runtime.
Users can own one registry per application, test, tenant, workspace, or server.

## Source Inspiration

Primary code references:

- LangChain `RunnableConfig` carries tags, metadata, callbacks,
  `max_concurrency`, `recursion_limit`, configurable values, and run ids:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/config.py>
- LangChain callback managers propagate child callbacks and parent run ids:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/manager.py>
- LangChain tools own name, description, schema, callbacks, tags, metadata,
  error handling, response format, and provider extras:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools/base.py>
- `langchain-rust` agents expose tools through `Vec<Arc<dyn Tool>>` and executor
  code maps tool names to trait objects:
  <https://github.com/Abraxas-365/langchain-rust/blob/main/src/agent/agent.rs>
  and
  <https://github.com/Abraxas-365/langchain-rust/blob/main/src/agent/executor.rs>
- `langchain-rust` tools use a simple Rust trait with name, description,
  parameter schema, parsing, and async execution:
  <https://github.com/Abraxas-365/langchain-rust/blob/main/src/tools/tool.rs>

## Responsibilities

- Register named agents.
- Register compiled graphs.
- Register tools.
- Register model providers.
- Register model aliases, resolver policies, and agent model defaults.
- Register middleware and stores.
- Register event listeners.
- Provide lookup and discovery APIs.
- Normalize component names.
- Validate duplicate or incompatible registrations.
- Emit registration, lifecycle, and execution events.
- Route events to outside listeners.
- Support parallel agent and graph execution with correlation ids.
- Provide a test recorder for event assertions.

## Non-Responsibilities

- It does not execute graph nodes itself.
- It does not call LLM providers itself.
- It does not validate tool arguments beyond registry-level schema checks.
- It does not persist checkpoints unless a checkpointer is registered as a
  component.
- It does not force a singleton global registry.

## Package Shape

Target layout:

```text
src/registry/
  mod.rs
  agent.rs
  catalog.rs
  component.rs
  discovery.rs
  events.rs
  graph.rs
  listener.rs
  model.rs
  names.rs
  pricing.rs
  scope.rs
  snapshot.rs
  store.rs
  testkit.rs
  tool.rs
```

## Core Concept

The registry owns named component handles:

```rust
pub struct Registry<State, Ctx = ()> {
    agents: AgentRegistry<State, Ctx>,
    graphs: GraphRegistry<State, Ctx>,
    models: ModelRegistry<State, Ctx>,
    tools: ToolRegistry<State, Ctx>,
    stores: StoreRegistry,
    middleware: MiddlewareRegistry<State, Ctx>,
    listeners: ListenerRegistry,
    bus: EventBus,
}
```

The registry is cloneable through `Arc`:

```rust
pub type SharedRegistry<State, Ctx = ()> = Arc<Registry<State, Ctx>>;
```

The registry should be passable into:

- harness runs
- graph runs
- tool calls
- web server request handlers
- background workers
- tests

The registry keeps lookup separate from execution:

```text
ComponentRef
  -> registry.resolve(ref, run_config)
  -> ComponentDescriptor
  -> ComponentFactory
  -> executable component
  -> component-owned runtime events
```

Registry operations emit registry lifecycle events. Model, tool, graph, and
agent execution spans are emitted by the component runtimes.

## Component Identity

Every registered component has an identity.

```rust
pub struct ComponentId {
    pub kind: ComponentKind,
    pub namespace: Option<String>,
    pub name: String,
    pub version: Option<String>,
}

pub enum ComponentKind {
    Agent,
    Graph,
    Model,
    Tool,
    Store,
    Middleware,
    Listener,
}
```

Name rules:

- ASCII by default
- lowercase `snake_case`
- no whitespace
- slash-separated namespaces are allowed through `namespace/name`
- versions are optional but immutable once registered

The registry should reject duplicate ids unless replacement is explicitly
requested.

Component ids are durable. They must not be raw Rust type paths. A Rust type can
move modules without changing the persisted component id.

Alias and migration support:

```rust
pub struct ComponentAlias {
    pub from: ComponentId,
    pub to: ComponentId,
    pub reason: String,
}
```

The registry should resolve aliases before lookup and emit
`registry.alias_resolved` so old graph specs or expressive-language files can
survive component renames.

## Component Metadata

Every component may expose metadata for discovery and UI rendering.

```rust
pub struct ComponentMetadata {
    pub id: ComponentId,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub schema: Option<serde_json::Value>,
    pub capabilities: Vec<String>,
    pub created_by: Option<String>,
    pub provider: Option<String>,
}
```

Examples:

- a tool exposes input/output schema
- a graph exposes node and edge summary
- an agent exposes model name and tool names
- a listener exposes supported event filters

## Component Factories

Some components are static trait objects. Others need per-run construction from
config. The registry should support both through factories.

```rust
#[async_trait]
pub trait ComponentFactory<State, Ctx = ()>: Send + Sync {
    type Output: Send + Sync;

    async fn instantiate(
        &self,
        descriptor: &ComponentDescriptor,
        config: &RunConfig,
        registry: SharedRegistry<State, Ctx>,
    ) -> Result<Self::Output>;
}

pub struct ComponentDescriptor {
    pub id: ComponentId,
    pub metadata: ComponentMetadata,
    pub dependencies: Vec<ComponentId>,
    pub event_kinds: Vec<EventKind>,
}
```

Use factories for:

- model aliases resolved from runtime config
- tenant-specific tool wrappers
- dynamic subagents
- fake replacements in tests
- lazily initialized expensive providers

Use direct `Arc<dyn Trait>` entries for simple tools and stores.

## Agent Registration

Agents are executable harness configurations. They may be simple model-tool loops
or wrappers around compiled graphs.

```rust
#[async_trait]
pub trait RegisteredAgent<State, Ctx = ()>: Send + Sync {
    fn metadata(&self) -> ComponentMetadata;

    async fn invoke(
        &self,
        input: AgentInput,
        ctx: RunContext<Ctx>,
        registry: SharedRegistry<State, Ctx>,
    ) -> Result<AgentRun<State>>;
}
```

Registration:

```rust
registry
    .agents()
    .register("support_agent", support_agent)
    .await?;
```

Lookup:

```rust
let agent = registry.agents().get("support_agent")?;
let run = agent.invoke(input, ctx, registry.clone()).await?;
```

Agent events:

- `agent.registered`
- `agent.started`
- `agent.completed`
- `agent.failed`
- `agent.child_started`
- `agent.child_completed`

Agent metadata should also declare model selection policy when the agent does
not want to rely only on the registry default:

```rust
pub struct AgentModelPolicy {
    pub default_model: Option<ModelRef>,
    pub fallback_models: Vec<ModelRef>,
    pub hints: Vec<ModelHint>,
    pub required_capabilities: CapabilitySet,
    pub reuse_resolved_model: bool,
    pub inherit_parent_model: InheritancePolicy,
}
```

The registry stores these declarations; the harness applies them during a run.
This keeps model selection inspectable without making the registry call model
providers.

## Graph Registration

Graphs are compiled graph runtimes.

```rust
pub struct RegisteredGraph<State, Ctx = ()> {
    pub metadata: ComponentMetadata,
    pub graph: Arc<CompiledGraph<State, Ctx>>,
}
```

Registration:

```rust
registry
    .graphs()
    .register("support_flow", compiled_graph)
    .await?;
```

Graph metadata should include:

- node ids
- edge count
- route names
- start node
- whether checkpoints are required
- whether interrupts are possible
- declared input/output schemas when available

Graph events:

- `graph.registered`
- `graph.started`
- `graph.step_started`
- `graph.node_started`
- `graph.node_completed`
- `graph.route_selected`
- `graph.interrupted`
- `graph.checkpoint_saved`
- `graph.completed`
- `graph.failed`

## Tool Registration

Tools are named callable capabilities. The registry should own normalized name
lookup so executors do not rebuild ad hoc maps per run.

```rust
pub struct RegisteredTool<State, Ctx = ()> {
    pub metadata: ComponentMetadata,
    pub tool: Arc<dyn Tool<State, Ctx>>,
}

pub struct ToolRegistry<State, Ctx = ()> {
    tools: DashMap<ToolName, RegisteredTool<State, Ctx>>,
}
```

Registration:

```rust
registry.tools().register(my_tool).await?;
```

Lookup:

```rust
let tool = registry.tools().get("lookup_user")?;
```

Tool events:

- `tool.registered`
- `tool.started`
- `tool.progress`
- `tool.completed`
- `tool.failed`

The registry should keep tool schema and metadata discoverable for web UIs.

## Model Registration

Models are provider-neutral chat model handles.

```rust
pub struct RegisteredModel<State, Ctx = ()> {
    pub metadata: ComponentMetadata,
    pub model: Arc<dyn ChatModel<State, Ctx>>,
}
```

Model metadata should include:

- provider
- model id
- catalog entry id when known
- aliases
- streaming support
- tool-calling support
- structured-output support
- default request policy
- resolver tags such as `fast`, `cheap`, `local`, `long_context`,
  `reasoning`, or app-defined labels

Model events:

- `model.registered`
- `model.alias_registered`
- `model.resolution_started`
- `model.resolution_candidate_rejected`
- `model.resolved`
- `model.started`
- `model.delta`
- `model.completed`
- `model.failed`

## Model Resolution Registry Contract

The registry owns the names and metadata used by model resolution. The harness
owns the actual selection for a run because it has request state, agent state,
runtime policy, budget, and current context-window pressure.

Registry responsibilities:

- map aliases and tags to registered executable model handles
- join executable models with model catalog metadata
- expose resolver policies for tenants, workspaces, agents, and tests
- validate duplicate aliases and conflicting model labels
- expose discovery data for UIs and model pickers
- emit model-resolution events

Non-responsibilities:

- it does not silently choose a model without the harness/run policy
- it does not call providers to test availability during ordinary lookup
- it does not mutate agent state with resolved-model records

Suggested shape:

```rust
pub struct ModelResolver {
    pub aliases: ModelAliasMap,
    pub catalog: ModelCatalog,
    pub policies: Vec<ModelResolverPolicy>,
}

pub struct ModelResolverPolicy {
    pub scope: RegistryScope,
    pub allowed_models: Vec<ModelRef>,
    pub denied_models: Vec<ModelRef>,
    pub fallback_order: Vec<ModelRef>,
    pub labels: HashMap<String, Vec<ModelRef>>,
}
```

Resolution should return both an executable handle and a durable
`ResolvedModel` record. The handle is process-local; the record is persisted in
state, events, checkpoints, usage, and cost rows.

## Store And Checkpointer Registration

Stores and checkpointers should be registry components so agents and graphs can
share them without globals.

Registered stores:

- short-term memory
- long-term key/value store
- vector store
- graph checkpointer
- event sink

Store events:

- `store.registered`
- `store.read`
- `store.write`
- `store.failed`

Store read/write events may need redaction controls.

## Listener Registration

Listeners observe events. They should be trait objects so web UIs, tests,
OpenTelemetry exporters, logs, and custom dashboards can all subscribe.

```rust
#[async_trait]
pub trait EventListener: Send + Sync {
    fn id(&self) -> ListenerId;
    fn filter(&self) -> EventFilter;
    async fn on_event(&self, event: RegistryEvent) -> Result<()>;
}
```

Listener examples:

- in-memory event recorder
- stdout logger
- tracing subscriber bridge
- websocket broadcaster
- Server-Sent Events broadcaster
- OpenTelemetry exporter
- metrics collector
- test assertion recorder

Listener lifecycle events:

- `listener.registered`
- `listener.failed`
- `listener.removed`

## Event Model

All events should share an envelope.

```rust
pub struct RegistryEvent {
    pub id: EventId,
    pub time: SystemTime,
    pub run: Option<RunRef>,
    pub parent: Option<EventId>,
    pub component: ComponentId,
    pub kind: EventKind,
    pub level: EventLevel,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub payload: EventPayload,
}
```

Correlation fields:

```rust
pub struct RunRef {
    pub run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub span_id: SpanId,
}
```

`RunRef` should be created from one propagated `RunConfig`. Do not pass tracing
ids, registry ids, and tool ids through separate ad hoc channels.

Metadata inheritance:

```rust
pub struct RunConfig {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub tags: MetadataScope<Vec<String>>,
    pub metadata: MetadataScope<serde_json::Value>,
    pub max_concurrency: Option<usize>,
    pub recursion_limit: Option<usize>,
    pub configurable: serde_json::Value,
}

pub struct MetadataScope<T> {
    pub local: T,
    pub inherited: T,
}
```

Inherited tags and metadata propagate to child components. Local metadata stays
on the current component span. This distinction lets UIs filter by durable
context like tenant, thread, agent, or tool package without polluting every child
span with local retry and provider details.

Event levels:

- trace
- debug
- info
- warn
- error

Event payloads:

```rust
pub enum EventPayload {
    Agent(AgentEvent),
    Graph(GraphEvent),
    Model(ModelEvent),
    Tool(ToolEvent),
    Store(StoreEvent),
    Listener(ListenerEvent),
    Registry(RegistryLifecycleEvent),
    Custom(serde_json::Value),
}
```

The envelope gives external listeners enough context to render parallel runs
without guessing parent/child relationships.

Recommended serialized event names:

```text
on_registry_lookup_start
on_registry_lookup_end
on_registry_resolve_start
on_registry_resolve_end
on_component_instantiate_start
on_component_instantiate_end
on_agent_start
on_agent_stream
on_agent_end
on_agent_error
on_graph_start
on_graph_stream
on_graph_end
on_graph_error
on_tool_start
on_tool_stream
on_tool_end
on_tool_error
on_model_start
on_model_stream
on_model_end
on_model_error
```

The Rust enum names can be idiomatic, but serialized event names should remain
stable for web UIs and persisted traces.

## Event Bus

The event bus is responsible for fanout.

```rust
#[async_trait]
pub trait EventBus: Send + Sync {
    async fn emit(&self, event: RegistryEvent) -> Result<()>;
    async fn subscribe(&self, filter: EventFilter) -> Result<EventSubscription>;
}
```

Implementation requirements:

- nonblocking emit path where possible
- bounded queues
- backpressure policy
- listener error isolation
- event redaction hook
- deterministic in-memory implementation for tests
- async stream subscription for web UIs

Backpressure policies:

- block
- drop oldest
- drop newest
- fail run

Default policy should be bounded and fail noisy in tests, but avoid crashing a
production run because a UI listener disconnected.

## Event Filters

Listeners need filters.

```rust
pub struct EventFilter {
    pub kinds: Option<Vec<EventKind>>,
    pub component_kinds: Option<Vec<ComponentKind>>,
    pub component_names: Option<Vec<String>>,
    pub run_id: Option<RunId>,
    pub thread_id: Option<ThreadId>,
    pub tags: Vec<String>,
    pub min_level: EventLevel,
}
```

Example filters:

- all events for one `run_id`
- only graph node events
- only tool failures
- only events tagged `tenant:acme`
- only model token deltas for streaming text

## Static And Dynamic Components

The registry should distinguish static registered components from runtime
components.

Static components:

- registered before a run
- listed in discovery APIs
- have durable ids
- validated when graphs compile

Runtime components:

- supplied by middleware or run config
- may not be discoverable before a run
- must be executable through a dynamic resolver
- must still emit events with a component id or temporary runtime id

Dynamic tools are useful for middleware and subagents, but unknown tool names
should not silently execute. A dynamic resolver must explicitly accept the tool
name and produce a registered-like descriptor.

## Parallel Agents And Hierarchical Runs

Parallel execution requires explicit hierarchy.

When an agent spawns child agents or graph branches:

- every child run gets a new `run_id`
- every child run inherits `root_run_id`
- every child run records `parent_run_id`
- tags and metadata inherit by default
- child components may append tags
- event ordering is per listener stream, not global causality

Example:

```text
root run: support_agent
  child run: retrieval_graph
    node: search
    node: rerank
  child run: draft_agent
    model: default
    tool: lookup_user
```

The web UI should be able to reconstruct this tree from event envelopes alone.

## Web UI Integration

The registry should support UI-facing APIs:

```rust
GET /registry/components
GET /registry/components/{kind}/{name}
GET /runs/{run_id}/events
GET /threads/{thread_id}/runs
GET /events/stream?run_id=...
POST /agents/{name}/invoke
POST /graphs/{name}/invoke
```

The Rust library should not ship a web server initially, but the event and
discovery APIs should make one straightforward.

UI event needs:

- stable component ids
- display names
- graph topology metadata
- tool schemas
- run tree correlation
- streaming model deltas
- interrupt payloads
- checkpoint ids
- final outputs
- redacted error details

## Stream Transformers

Stream transformers derive UI-friendly views from raw events.

```rust
#[async_trait]
pub trait StreamTransformer: Send + Sync {
    fn id(&self) -> &str;
    fn input_filter(&self) -> EventFilter;
    async fn transform(&self, event: RegistryEvent) -> Result<Vec<RegistryEvent>>;
}
```

Examples:

- tool-call timeline transformer
- subagent tree transformer
- model-token text transformer
- graph-state diff transformer
- cost/usage accumulator

Transformers should be registry extensions. They should not be hardcoded into
every tool, graph, or model implementation.

## Redaction And Safety

Events may contain sensitive data. The registry must support redaction.

Redaction hooks:

- before event leaves component
- before event reaches bus
- per listener

Fields that may require redaction:

- API keys
- provider request bodies
- tool arguments
- tool outputs
- memory values
- user messages
- environment details

Default behavior:

- metadata values are JSON only
- known secret keys are redacted
- raw provider payloads are opt-in
- listeners can request safe summaries

## Registration Lifecycle

Registration should be explicit and validated.

```rust
registry.register_tool(tool).await?;
registry.register_model("default", model).await?;
registry.register_graph("support_flow", graph).await?;
registry.register_agent("support_agent", agent).await?;
registry.register_listener(websocket_listener).await?;
```

Lifecycle events:

- `registry.component_registered`
- `registry.component_replaced`
- `registry.component_removed`
- `registry.lookup_started`
- `registry.lookup_completed`
- `registry.resolve_started`
- `registry.resolve_completed`
- `registry.instantiate_started`
- `registry.instantiate_completed`
- `registry.alias_resolved`
- `registry.validation_failed`

Validation:

- duplicate names
- invalid names
- missing dependencies
- incompatible state type where statically knowable
- tool schema invalid
- graph references missing component

## Discovery API

Discovery lets UIs and orchestrators inspect capabilities.

```rust
pub trait Discoverable {
    fn metadata(&self) -> ComponentMetadata;
    fn dependencies(&self) -> Vec<ComponentId>;
}
```

Discovery output should include:

- component id
- component kind
- description
- tags
- schema
- dependencies
- event kinds emitted
- run modes supported

## Error Model

Registry errors should distinguish:

- duplicate component
- component not found
- invalid component name
- invalid schema
- missing dependency
- listener failure
- event queue full
- component type mismatch
- redaction failure
- registration locked

Listener failures should emit events but should not fail the run unless the
listener is marked required.

## Testkit

`registry::testkit` should include:

- in-memory registry builder
- event recorder listener
- event snapshot assertions
- fake tool registration
- fake graph registration
- fake agent registration
- deterministic event ids
- deterministic timestamps

Example:

```rust
let recorder = EventRecorder::new();
let registry = TestRegistry::new()
    .with_listener(recorder.clone())
    .with_tool(fake_tool("lookup_user"))
    .with_agent(fake_agent("support_agent"))
    .build();

registry.agent("support_agent")?.invoke(input).await?;

recorder.assert()
    .saw("agent.started")
    .saw("tool.started")
    .saw("tool.completed")
    .saw("agent.completed");
```

## Implementation Milestones

### R1: Component Registries

- `ComponentId`
- `ComponentMetadata`
- `ToolRegistry`
- `ModelRegistry`
- `GraphRegistry`
- duplicate validation

### R2: Event Envelope

- `RegistryEvent`
- `RunRef`
- `EventPayload`
- in-memory event bus

### R3: Listeners

- `EventListener`
- filters
- event recorder
- stdout listener

### R4: Agent And Graph Registration

- registered agent trait
- registered graph wrapper
- lookup and invoke helpers

### R5: Parallel Run Correlation

- parent/root run ids
- child run creation
- inherited tags and metadata
- event tree assertions

### R6: UI Streaming Surface

- async subscriptions
- websocket/SSE example
- redaction policy
- component discovery JSON
