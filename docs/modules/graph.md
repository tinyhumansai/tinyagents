# Graph Module Specification

The graph module is the workflow runtime. It owns topology, state transitions,
routing, execution history, checkpointing, interrupts, streaming, parallel
execution, sub-agent nodes, recursive graph calls, and depth tracking.

The graph module should be usable without the expressive language. The expressive
language compiles into graph structures; the graph runtime should not know or
care where a graph came from.

## Source Inspiration

Primary references:

- <https://docs.langchain.com/oss/python/langgraph/graph-api>
- <https://docs.langchain.com/oss/python/langgraph/persistence>
- <https://docs.langchain.com/oss/python/langgraph/checkpointers>
- <https://docs.langchain.com/oss/python/langgraph/interrupts>
- <https://docs.langchain.com/oss/python/langgraph/streaming>
- <https://docs.langchain.com/oss/python/langgraph/event-streaming>
- <https://docs.langchain.com/oss/python/langgraph/use-subgraphs>
- <https://docs.langchain.com/oss/python/langgraph/fault-tolerance>

## Responsibilities

- Build named node graphs.
- Validate topology before execution.
- Compile a mutable builder into an immutable executable graph.
- Execute async nodes.
- Apply state updates through reducer policies.
- Route through direct, conditional, and command-based edges.
- Enforce recursion and concurrency policy.
- Execute parallel branches.
- Represent sub-agents as graph nodes.
- Support recursive graph/subgraph invocation with explicit depth tracking.
- Persist checkpoints at execution boundaries.
- Support interrupts and resume.
- Emit typed execution events.
- Represent subgraphs as executable nodes.
- Export graph structure for visualization and tests.

## Non-Responsibilities

- It does not own chat model provider logic.
- It does not own tool schema validation.
- It does not implement prompt templating.
- It does not manage long-term application memory.
- It does not parse the expressive language.

## Package Shape

Target layout:

```text
src/graph/
  mod.rs
  builder.rs
  checkpoint.rs
  command.rs
  compile.rs
  edge.rs
  error.rs
  event.rs
  executor.rs
  interrupt.rs
  node.rs
  parallel.rs
  reducer.rs
  recursion.rs
  state.rs
  stream.rs
  subagent.rs
  subgraph.rs
  testkit.rs
```

The current scaffold has a single `src/graph.rs`. It can remain for milestone 1,
then split into the package above once the graph contract becomes larger.

## Core Types

```rust
pub struct GraphBuilder<State, Update = State> {
    nodes: HashMap<NodeId, Node<State, Update>>,
    edges: EdgeSet,
    reducer: Arc<dyn StateReducer<State, Update>>,
    defaults: GraphDefaults,
}

pub struct CompiledGraph<State, Update = State> {
    graph_id: GraphId,
    nodes: Arc<HashMap<NodeId, Node<State, Update>>>,
    edges: Arc<EdgeSet>,
    reducer: Arc<dyn StateReducer<State, Update>>,
    defaults: GraphDefaults,
}

pub struct GraphRun<State> {
    pub state: State,
    pub visited: Vec<NodeId>,
    pub checkpoints: Vec<CheckpointId>,
    pub interrupts: Vec<Interrupt>,
    pub max_depth: usize,
}
```

The builder is mutable and ergonomic. The compiled graph is immutable,
validated, cheap to clone, and safe to run concurrently.

## Node Model

Nodes are async units of work. They receive state and runtime context and return
a result that may update state, route execution, or interrupt.

```rust
#[async_trait]
pub trait GraphNode<State, Update = State, Ctx = ()>: Send + Sync {
    async fn run(
        &self,
        state: StateView<'_, State>,
        ctx: &mut GraphContext<Ctx>,
    ) -> Result<NodeResult<Update>>;
}

pub enum NodeResult<Update> {
    Update(Update),
    Command(Command<Update>),
    Interrupt(Interrupt),
}
```

Milestone 1 can keep closure-based nodes:

```rust
Node::new("agent", |state| async move {
    Ok(NodeOutput::continue_with(state))
})
```

The target API should support trait-backed nodes so provider nodes, subgraph
nodes, test nodes, and language-compiled nodes can share one representation.

## State And Updates

The current scaffold returns whole state from each node. The durable graph design
should move toward partial updates.

```rust
pub trait StateReducer<State, Update>: Send + Sync {
    fn apply(&self, state: State, update: Update) -> Result<State>;
}
```

Reducer policies:

- overwrite
- append
- message merge by id
- set union
- numeric min/max
- custom reducer

Why partial updates matter:

- parallel branches can update different fields
- checkpoints can store pending writes
- failed parallel nodes can rerun without discarding completed writes
- tests can assert precise state changes
- language-defined nodes can have simple update contracts

Recommended staged path:

1. keep whole-state `NodeOutput<State>`
2. add `StateReducer<State, Update>` behind a new builder
3. add partial update examples
4. make reducer-based execution the default for durable graphs

## Edges And Routing

Reserved virtual nodes:

- `START`
- `END`

Direct edge:

```text
START -> agent
agent -> summarize
summarize -> END
```

Conditional edge:

```text
agent --tool--> tools
agent --final--> END
tools ---------> agent
```

Command routing:

```rust
Command::new()
    .update(update)
    .goto(["tools"])
```

Validation rules:

- graph must have exactly one start path unless multi-start is explicitly added
- every edge source exists, except `START`
- every edge target exists, except `END`
- duplicate node ids are rejected
- duplicate route names from a node are rejected
- conditional route targets are validated at compile time when known
- a node should not mix static outgoing edges with command routing unless that
  mixed behavior is explicitly enabled

Typed routes should be added after string routes:

```rust
enum AgentRoute {
    Tool,
    Final,
}
```

## Execution Model

Milestone 1 executor:

- sequential
- one active node at a time
- whole-state updates
- direct or string conditional routes
- recursion limit

Target executor:

- superstep-based
- multiple active nodes per step
- reducer-applied updates at step boundaries
- checkpoint after step completion
- support pending writes from completed parallel nodes
- resume from checkpoint
- recursive call tracking
- child graph and child agent run tracking

Superstep lifecycle:

1. Load current active nodes.
2. Emit step started event.
3. Run active nodes with concurrency policy.
4. Collect updates, commands, interrupts, and errors.
5. Apply reducers to successful updates.
6. Persist pending writes and checkpoint.
7. Select next active nodes.
8. Emit step completed event.

Checkpointing mid-node should be avoided. Async Rust stack suspension is not a
stable persistence primitive; node rerun semantics are easier to reason about.

## Parallelization

Parallel execution is expressed as multiple active nodes in a superstep. A node
can route to more than one next node through conditional routing or a command.

```rust
Command::new()
    .update(update)
    .goto(["retrieve_docs", "lookup_user", "score_risk"])
```

Parallel execution rules:

- all active nodes in a superstep read the same state snapshot
- each node returns a partial update, command, interrupt, or error
- reducers merge successful updates at the step boundary
- a failed required node fails the superstep unless policy says otherwise
- completed writes can be preserved as pending writes when checkpointing supports
  them
- concurrency is bounded by graph and run config

Parallelism must be visible in events:

- `StepStarted { active: [...] }`
- `NodeStarted`
- `NodeCompleted`
- `NodeFailed`
- `StateUpdated`
- `StepCompleted`

## Sub-Agents

Sub-agents are harness agents registered in the registry and invoked from graph
nodes. They are not special opaque side effects; they are child runs with their
own events, usage, cost, and failures.

```rust
pub struct SubAgentNode {
    pub agent: ComponentId,
    pub input_mapper: InputMapper,
    pub output_mapper: OutputMapper,
    pub policy: SubAgentPolicy,
}
```

Sub-agent requirements:

- create a child `run_id`
- preserve `root_run_id`
- set `parent_run_id` to the graph node run
- forward harness events through graph/registry streaming
- apply timeout, retry, and budget policy
- map child output into parent graph update
- make child usage and cost visible in parent rollups

Sub-agents allow a graph to coordinate specialized workers while keeping each
worker independently observable.

## Recursion And Depth Tracking

The graph should allow recursive execution, but only with explicit limits and
tracking.

Recursive cases:

- a graph invokes itself as a subgraph
- a graph invokes a subgraph that eventually invokes the parent graph
- an agent node calls another agent that calls back into the same graph
- a router intentionally loops through nodes until state converges

Required tracking:

```rust
pub struct RecursionFrame {
    pub graph_id: GraphId,
    pub node_id: Option<NodeId>,
    pub run_id: RunId,
    pub depth: usize,
    pub parent: Option<RunId>,
}

pub struct RecursionPolicy {
    pub max_depth: usize,
    pub max_visits_per_node: Option<usize>,
    pub max_total_steps: usize,
}
```

Rules:

- every graph/subgraph/sub-agent call pushes a recursion frame
- every return pops the frame
- depth is emitted on graph and registry events
- exceeding `max_depth` fails with a clear recursion error
- node-loop recursion and graph-call recursion are tracked separately
- checkpoint metadata includes the current recursion stack

Depth tracking is part of observability, not just safety. Web UIs should be able
to render nested runs and recursive calls without reconstructing them from logs.

## Commands

Commands combine state update and routing.

```rust
pub struct Command<Update> {
    pub update: Option<Update>,
    pub goto: Vec<NodeId>,
    pub resume: Option<serde_json::Value>,
    pub graph: CommandGraphTarget,
}

pub enum CommandGraphTarget {
    Current,
    Parent,
    Subgraph(GraphId),
}
```

Use commands for:

- dynamic routing
- human approval resume values
- parent graph handoff from subgraphs
- node-local state update plus routing

Do not require users to split one conceptual node decision into separate
mutation and route functions.

## Checkpointing

Checkpointing is graph runtime persistence. It is separate from harness memory.

```rust
#[async_trait]
pub trait Checkpointer<State>: Send + Sync {
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId>;
    async fn get(
        &self,
        thread_id: &ThreadId,
        checkpoint_id: Option<&CheckpointId>,
    ) -> Result<Option<Checkpoint<State>>>;
    async fn list(&self, thread_id: &ThreadId) -> Result<Vec<CheckpointMetadata>>;
}
```

Checkpoint fields:

- thread id
- graph id
- checkpoint id
- parent checkpoint id
- namespace
- state snapshot
- next active nodes
- pending writes
- completed tasks
- interrupts
- metadata
- created timestamp

Backends:

- in-memory
- file-backed JSON
- SQLite
- Postgres later

Execution guarantee:

- checkpoint at superstep boundary
- interrupted nodes rerun from the beginning on resume
- completed writes in a failed superstep can be preserved once pending writes
  are implemented

## Interrupts And Resume

Interrupts pause execution and return control to the caller.

```rust
pub struct Interrupt {
    pub id: InterruptId,
    pub node: NodeId,
    pub payload: serde_json::Value,
    pub order: usize,
}
```

Resume API:

```rust
compiled_graph
    .resume(
        RunConfig::thread("support-123"),
        Command::resume(json!({ "approved": true })),
    )
    .await?;
```

Rules:

- interrupts require a checkpointer
- resume requires `thread_id`
- the interrupted node restarts
- multiple interrupts inside one node are matched by order
- node code before the interrupt must be deterministic or idempotent

## Streaming

The graph event stream should be typed.

```rust
pub enum GraphEvent {
    RunStarted { run_id: RunId, graph_id: GraphId },
    RunStreamingStarted { run_id: RunId },
    StepStarted { step: usize, active: Vec<NodeId> },
    NodeStarted { node: NodeId },
    NodeCompleted { node: NodeId },
    NodeFailed { node: NodeId, error: String },
    StateUpdated { node: NodeId, update: serde_json::Value },
    RouteSelected { node: NodeId, routes: Vec<NodeId> },
    SubAgentStarted { node: NodeId, agent: ComponentId, child_run_id: RunId },
    SubAgentCompleted { node: NodeId, agent: ComponentId, child_run_id: RunId },
    RecursionDepthChanged { depth: usize },
    CheckpointSaved { checkpoint_id: CheckpointId },
    InterruptEmitted { interrupt: Interrupt },
    RunCompleted { run_id: RunId },
    RunFailed { run_id: RunId, error: String },
    Custom { name: String, payload: serde_json::Value },
}
```

Stream modes:

- `events`: all events
- `updates`: state updates only
- `values`: full state snapshots
- `interrupts`: interrupt payloads
- `debug`: executor internals
- `messages`: harness message deltas emitted by model nodes
- `subagents`: child agent lifecycle and output events
- `depth`: recursion frame/depth changes

The graph stream should be able to forward harness events from graph nodes while
preserving node id and run id.

Streaming requirements:

- graph runs can be consumed as an async stream
- streaming does not require waiting for final state
- every streamed event carries run hierarchy and recursion depth
- harness streams from model/tool/sub-agent nodes are forwarded with graph node
  context
- subscribers can filter to graph events, harness events, sub-agent events, or
  state updates

## Subgraphs

Subgraphs are compiled graphs used as nodes.

Two modes:

```text
shared-state subgraph
parent State == child State

adapter subgraph
parent State -> child State -> parent Update
```

Subgraph requirements:

- namespace checkpoint ids
- emit nested events with parent node id
- support isolated per-invocation memory by default
- support thread-scoped child persistence by explicit configuration
- allow child command to route back to parent when configured

## Fault Tolerance

Fault tolerance policy should be configurable at graph and node levels.

```rust
pub struct NodePolicy {
    pub timeout: Option<Duration>,
    pub retry: RetryPolicy,
    pub error_handler: ErrorHandler,
}
```

Default behavior:

- node error fails the run
- retry only if node policy allows it
- timeout fails the node
- checkpoint remains at last completed boundary

Future behavior:

- route errors to error handler node
- retry with backoff
- skip optional node
- mark partial failure in state

## Visualization And Introspection

The graph should export:

- node ids
- edge list
- conditional routes
- start and end paths
- node metadata
- graph validation report

Export formats:

- JSON
- Mermaid
- DOT later

Test snapshots can use the same export.

## Errors

Graph errors should distinguish:

- missing start
- duplicate node
- missing node
- missing edge target
- duplicate route
- missing route
- invalid command target
- recursion limit
- checkpoint required
- checkpoint missing
- interrupt resume mismatch
- reducer conflict
- node timeout
- node failure

## Testkit

`graph::testkit` should include:

- no-op node
- scripted route node
- failing node
- interrupting node
- in-memory checkpointer
- event recorder
- graph snapshot assertion
- checkpoint assertion
- route assertion

Example:

```rust
assert_graph(run)
    .visited(["agent", "tools", "agent"])
    .routed("agent", "tool")
    .completed();
```

## Implementation Milestones

### G1: Current Sequential Runtime

- `Node`
- `NodeOutput`
- `StateGraph`
- direct edges
- conditional edges
- recursion limit

### G2: Builder And Compile Step

- introduce `GraphBuilder`
- introduce `CompiledGraph`
- move validation to `compile`
- add `START` and `END`

### G3: Commands And Typed Events

- add `Command`
- add `GraphEvent`
- add stream/event recorder

### G4: Partial Updates And Reducers

- add `StateReducer`
- add update type parameter
- add append and message reducers

### G5: Checkpointing And Interrupts

- add `Checkpointer`
- add in-memory backend
- add interrupt/resume API

### G6: Supersteps And Subgraphs

- add multi-active-node executor
- add subgraph node
- add checkpoint namespaces

### G7: Sub-Agents And Recursion

- add `SubAgentNode`
- add child run hierarchy
- add recursion stack
- add depth events
- add max-depth policy

### G8: Graph Streaming

- expose async graph event stream
- forward harness streams with node context
- add stream filters for updates, values, messages, subagents, and depth
