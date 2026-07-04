# Registry Operations And Lifecycle

Continues from [`design.md`](design.md) and [`events.md`](events.md):
static/dynamic components, parallel agents, web UI integration, stream
transformers, redaction, registration lifecycle, discovery, error model,
testkit, and implementation milestones.

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
