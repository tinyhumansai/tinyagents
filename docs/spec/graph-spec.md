# Graph Module Specification

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

`State` is user-owned application state. TinyAgents should never require a
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

LangGraph nodes return partial state updates. TinyAgents should adopt the same
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

