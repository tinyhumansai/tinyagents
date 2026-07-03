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

## Introspection And Diagnostics

The `CapabilityRegistry` exposes machine-readable views of what is registered so
CLIs, UIs, and audit logs can render and diff the catalog without owning the
live handles.

### Component kinds

`ComponentKind` partitions the registry namespace and now has **12** variants.
Alongside `Model`, `Tool`, `Graph`, `Router`, `Reducer`, `Store`, `Agent`, and
`Script` (a REPL script a `repl_agent` node may reference), four kinds cover the
runtime's durable roles:

| Kind | `as_str` |
| --- | --- |
| `Middleware` | `"middleware"` |
| `Checkpointer` | `"checkpointer"` |
| `TaskStore` | `"task_store"` |
| `Listener` | `"listener"` |

### Snapshots

`CapabilityRegistry::snapshot()` returns a serializable `RegistrySnapshot`
sorted by `(kind, name)` for diff-friendly output. It carries both the
registered `components` (each a `ComponentMetadata`) **and** every alias as an
`aliases: Vec<AliasBinding { kind, alias, canonical }>`, so a CLI can enumerate
the alternate names that resolve to each canonical component. The snapshot
round-trips through serde for audit logs.

```rust
let mut reg = CapabilityRegistry::<()>::new();
reg.register_model("gpt-4o", model)?;
reg.alias(ComponentKind::Model, "default", "gpt-4o")?;

let snapshot = reg.snapshot();
assert_eq!(snapshot.aliases.len(), 1);
assert_eq!(snapshot.aliases[0].alias, "default");
assert_eq!(snapshot.aliases[0].canonical, "gpt-4o");
assert_eq!(snapshot.aliases[0].kind, ComponentKind::Model);
```

`RegistrySnapshot::to_dot()` renders a Graphviz DOT document clustering
components by kind.

### Health diagnostics

`CapabilityRegistry::diagnostics()` returns actionable `RegistryDiagnostic`
findings sorted by `(kind, name)`:

- **alias shadows a component** (`DiagnosticSeverity::Warning`) — an alias has
  the same name as a registered component of that kind, so the alias is
  unreachable.
- **dangling alias** (`Error`) — an alias resolves to a name that is not a
  registered component.
- **name reused across kinds** (`Warning`) — the same name is registered under
  more than one kind. Registration only rejects same-`(kind, name)` duplicates,
  so a name legally shared across kinds is flagged for audits.

```rust
let mut reg = CapabilityRegistry::<()>::new();
reg.register_model("shared", model)?;
reg.register_router("shared")?; // legal: different kind

let diags = reg.diagnostics();
assert_eq!(diags.len(), 1);
assert_eq!(diags[0].name, "shared");
assert!(diags[0].message.contains("multiple kinds"));
```

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
