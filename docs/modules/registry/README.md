# Registry Module Specification

The registry module is the coordination and discovery layer for TinyAgents. It
registers agents, graphs, tools, models, stores, middleware, listeners, and
runtime data catalogs, then provides an event-friendly surface for web UIs,
CLIs, tests, and distributed supervisors.

The registry is deliberately explicit. It should not become a hidden global
runtime. Users can own one registry per application, test, tenant, workspace, or
server.

## Detailed Module Docs

- [Design](design.md)
- [Model catalog and local snapshots](model-catalog.md)

## Responsibilities

- Register named agents.
- Register compiled graphs.
- Register tools.
- Register model providers.
- Register middleware and stores.
- Register event listeners.
- Register local model metadata snapshots.
- Provide lookup and discovery APIs.
- Normalize component names.
- Validate duplicate or incompatible registrations.
- Emit registration, lifecycle, and execution events.
- Route events to outside listeners.
- Support parallel agent and graph execution with correlation ids.
- Provide a test recorder for event assertions.
- Provide local lookup for model prices, capabilities, and context windows.

## Non-Responsibilities

- It does not execute graph nodes itself.
- It does not call LLM providers itself.
- It does not validate tool arguments beyond registry-level schema checks.
- It does not persist checkpoints unless a checkpointer is registered as a
  component.
- It does not force a singleton global registry.
- It does not guarantee live provider pricing without an explicit snapshot
  refresh.

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

Documentation layout:

```text
docs/modules/registry/
  README.md
  design.md
  model-catalog.md
  model-catalog.snapshot.json
```

## Core Concept

The registry owns named component handles and local metadata catalogs:

```rust
pub struct Registry<State, Ctx = ()> {
    agents: AgentRegistry<State, Ctx>,
    graphs: GraphRegistry<State, Ctx>,
    models: ModelRegistry<State, Ctx>,
    tools: ToolRegistry<State, Ctx>,
    stores: StoreRegistry,
    middleware: MiddlewareRegistry<State, Ctx>,
    listeners: ListenerRegistry,
    model_catalog: ModelCatalog,
    bus: EventBus,
}
```

Runtime execution still belongs to the harness and graph modules. The registry
answers questions such as:

- Which model aliases are available?
- Which tool schemas are available?
- Which graph ids can be invoked?
- What are the known context-window and price details for this model?
- Which events should be emitted and where should they go?

## Local Model Catalog

The local model catalog is a snapshot, not a source of truth. It gives
TinyAgents deterministic offline behavior for:

- cost estimates
- context-window checks
- provider capability discovery
- model picker UIs
- tests
- request validation before provider calls

Snapshots must carry source URLs, fetch time, normalization version, and per-row
provenance. See [model catalog and local snapshots](model-catalog.md).
