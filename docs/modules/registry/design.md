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


---

Continues in [`events.md`](events.md) (store/checkpointer registration, the
event model, event bus, and filters) and [`operations.md`](operations.md)
(lifecycle, discovery, error model, testkit, and milestones).
