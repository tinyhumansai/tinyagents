# Registry Events And Persistence

Continues from [`design.md`](design.md): store and checkpointer
registration, listener registration, the event model, event bus, and
event filters.

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

