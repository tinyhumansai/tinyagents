# Harness State Graph Runtime Feature

The state graph runtime is the explicit state-machine form of the harness. It
models an agent run as named nodes, edges, routing commands, typed working state,
checkpointed run records, and resumable interrupts.

This is not a replacement for the simple direct harness path. A direct model
call and a model-plus-tools loop should remain easy. The graph runtime is for
long-running, inspectable, branchy, resumable, human-reviewed, or UI-controlled
agent work.

## Source Inspiration

OpenHuman PR #4261 implements a LangGraph-style runtime that is directly
relevant to RustAgents:

- PR: <https://github.com/tinyhumansai/openhuman/pull/4261>
- generic engine: `src/openhuman/agent_graph/graph/`
- run/checkpoint persistence: `src/openhuman/agent_graph/checkpoint/`
- HITL interrupts: `src/openhuman/agent_graph/hitl/`
- graph lifecycle events: `src/openhuman/agent_graph/observability/`
- built-in graph definitions: `src/openhuman/agent_graph/definitions/`
- per-agent blueprints: `src/openhuman/agent_graph/blueprint/`
- JSON-RPC surface: `src/openhuman/agent_graph/{ops,schemas}.rs`
- live turn graph bridge: `src/openhuman/agent_graph/live/`
- behavior-preserving turn state machine: `src/openhuman/agent/harness/engine/core.rs`

Key design lesson from the PR: keep the generic graph engine decoupled from the
agent harness, then bridge product-specific model/tool/memory behavior through
nodes and runtime adapters.

## Responsibilities

- Define graph state and merge/reducer semantics.
- Define async node execution.
- Support static edges, conditional edges, fork/fan-out edges, and finish nodes.
- Support node commands: continue, goto, fork, interrupt, and end.
- Compile and validate graph topology before execution.
- Execute graphs with deterministic super-step ordering.
- Support branch state merging.
- Enforce cancellation and max-step guards.
- Persist run records and checkpoints.
- Support pause/resume through human-in-the-loop interrupts.
- Emit typed graph lifecycle events.
- Expose graph definitions, run records, checkpoints, and per-agent blueprints
  to UIs and tests.
- Provide deterministic graph definitions and fake nodes for E2E tests.

## Non-Responsibilities

- It does not require all harness calls to use graphs.
- It does not own provider adapters.
- It does not own tool implementations.
- It does not replace the graph module's broader workflow/topology APIs.
- It does not hide model/tool safety checks inside graph routing.
- It does not allow provider-supplied tool calls to bypass the per-turn
  advertised tool allowlist.

## Core Types

```rust
pub trait GraphState:
    Clone + Serialize + DeserializeOwned + Send + Sync + 'static
{
    fn merge(&mut self, other: Self) -> Result<()>;
}

#[async_trait]
pub trait Node<S: GraphState>: Send + Sync {
    async fn run(&self, state: S, ctx: &NodeCtx<'_>) -> Result<NodeOutput<S>>;
}

pub struct NodeOutput<S> {
    pub state: S,
    pub command: Command,
}

pub enum Command {
    Continue,
    Goto(NodeId),
    Fork(Vec<NodeId>),
    Interrupt(InterruptRequest),
    End,
}
```

`GraphState::merge` is the reducer contract. Parallel branches receive cloned
state and then merge back. Reducers must account for shared pre-fork state so
they do not double-count base data.

## Builder And Compile Validation

```rust
pub struct StateGraph<S: GraphState> {
    name: GraphName,
    nodes: HashMap<NodeId, Arc<dyn Node<S>>>,
    edges: HashMap<NodeId, Edge<S>>,
    entry: Option<NodeId>,
    finish: HashSet<NodeId>,
    max_steps: u32,
}
```

Builder surface:

- `add_node(id, node)`
- `add_edge(from, to)`
- `add_conditional_edges(from, targets, router)`
- `add_fork(from, targets)`
- `set_entry_point(id)`
- `set_finish_point(id)`
- `set_max_steps(max)`
- `compile()`

`compile()` must fail on:

- missing entry point
- unknown entry node
- edge source that is not a node
- edge target that is not a node or `END`
- conditional target not declared
- node with no outgoing edge and not marked as finish

Compile failures should be distinct from runtime failures so tests can assert
whether a graph definition is malformed or execution failed.

## Execution Semantics

Execution follows Pregel-style super-steps:

1. Start with a frontier of `(entry_node, initial_state)`.
2. Run every node in the current frontier.
3. Collect each node's state and command.
4. Compute the next frontier.
5. Merge states that converge on the same next node.
6. Stop when all branches end, an interrupt pauses the run, cancellation fires,
   or the max-step guard trips.

The runtime should keep deterministic ordering for tests and transcripts. A
stable ordering such as `BTreeMap<NodeId, S>` for next-frontier convergence is
preferred.

## Human-In-The-Loop

Human review is a first-class node outcome, not an ad hoc exception.

```rust
pub struct InterruptRequest {
    pub kind: String,
    pub question: String,
    pub options: Vec<String>,
    pub resume_to: Option<NodeId>,
}

pub trait ApplyResume {
    fn apply_resume(&mut self, input: &str);
}
```

When a node returns `Command::Interrupt`, the runtime should:

- persist a paused run record
- write a checkpoint snapshot
- emit a graph paused event
- return the interrupt payload to the caller
- resume only when the run is still paused
- fold the resume input into state before continuing
- continue from `resume_to` or a validated static successor

Resume must reject completed, failed, missing, or non-paused runs.

## Checkpointing

```rust
#[async_trait]
pub trait Checkpointer: Send + Sync {
    async fn save_run(&self, rec: &GraphRunRecord) -> Result<()>;
    async fn load_run(&self, run_id: &RunId) -> Result<Option<GraphRunRecord>>;
    async fn list_runs(&self, limit: usize, offset: usize) -> Result<Vec<GraphRunRecord>>;
    async fn save_checkpoint(&self, cp: &Checkpoint) -> Result<CheckpointId>;
    async fn list_checkpoints(&self, run_id: &RunId) -> Result<Vec<Checkpoint>>;
}
```

Run records should include:

- run id
- graph name
- status
- created and updated timestamps
- last node
- super-step count
- serialized state
- interrupt payload when paused
- error text when failed
- node transitions

Checkpoints should include:

- checkpoint id
- run id
- step
- node
- label such as `start`, `pause:approval`, `complete`, or `failed`
- serialized state
- timestamp

Backends:

- `InMemoryCheckpointer` for tests
- SQLite for durable local runs
- future Postgres/Mongo/object-store-backed checkpointers for server use

Run listing must be newest-first and paged. Timestamps should be normalized to
UTC or compared as parsed absolute times rather than relying on arbitrary string
ordering.

## Graph Events

Event kinds:

- `graph.run.started`
- `graph.node.entered`
- `graph.node.completed`
- `graph.run.paused`
- `graph.run.completed`
- `graph.run.failed`
- `graph.checkpoint.saved`
- `graph.resume.started`
- `graph.resume.completed`

Node-completed events should include run id, graph name, step, node id, command
label, and elapsed time. Run events should include status, last node, steps,
checkpoint id when available, and error/interrupt metadata.

## Definitions And Blueprints

The runtime needs two definition forms:

- executable definitions: compiled Rust node graphs
- inspectable blueprints: serializable node/edge topology for UIs, tests, and
  agent catalogs

```rust
pub enum NodeKind {
    Dispatch,
    Parse,
    StopCheck,
    Tools,
    Compact,
    Finalize,
    Hitl,
    Delegate(AgentId),
    Custom(String),
}

pub enum EdgeSpec {
    Static { from: NodeId, to: NodeId },
    Conditional { from: NodeId, on: String, targets: Vec<NodeId> },
    Fork { from: NodeId, targets: Vec<NodeId> },
}

pub struct GraphBlueprint {
    pub name: String,
    pub entry: NodeId,
    pub finish: Vec<NodeId>,
    pub nodes: Vec<NodeSpec>,
    pub edges: Vec<EdgeSpec>,
}
```

Per-agent convention:

- `prompt` defines what the agent says
- `graph` defines how the agent runs
- loader tests validate every built-in agent graph
- UIs can inspect the graph without running it

Reusable blueprint shapes:

- `canonical_turn`: `dispatch -> parse -> stop_check -> tools -> compact ->
  dispatch`, or `finalize`
- `single_shot`: `dispatch -> finalize`
- `plan_execute_review`: `plan -> execute -> review -> finalize`, with reject
  looping back to execute
- `delegate`: orchestrator-style delegation and join patterns

## Live Turn Bridge

The graph runtime must support real agent turns without forcing non-clonable
provider/tool state into `GraphState`.

Pattern:

- keep provider, tool executor, message history, and cost state in a
  run-scoped machine behind an async mutex or equivalent exclusive owner
- keep graph-visible state small and serializable
- use graph nodes to drive live phases
- preserve legacy turn contracts while migrating phase by phase

Canonical live phases:

- `dispatch`: context guard, stop hooks, trim request copy, call provider,
  append assistant message, record usage/cost
- `parse`: decide final answer, tool loop, repeat-output breaker, or iteration
  cap
- `tools`: validate requested tools against the advertised allowlist, parse
  arguments, execute tools, append tool messages, enforce failure breaker and
  early-exit tools
- `compact`: summarize or trim before looping
- `finalize`: return outcome and preserve final history
- `max_iterations`: invoke checkpoint strategy

Safety requirements:

- provider tool calls must be checked against tools advertised for that turn
- malformed tool arguments must fail closed and must not execute side-effecting
  tools with default empty arguments
- native assistant tool-call history and tool result messages must preserve
  provider call ids
- early-exit tools must surface pause semantics without pretending a final
  answer was produced

## RPC Or Control Surface

The harness should expose a control surface equivalent to:

- `graph_definition_list`
- `graph_agent_list`
- `graph_agent_get`
- `graph_run`
- `graph_run_list`
- `graph_run_get`
- `graph_checkpoint_list`
- `graph_resume`

Read-only methods should return bare values where the surrounding RPC protocol
already has its own envelope. Mutation methods may include logs or audit
messages. The important contract is that UI clients can list definitions, start
runs, inspect paused/completed runs, browse checkpoints, and resume HITL runs.

## Testing Requirements

Graph runtime tests should cover:

- linear graph
- conditional routing
- cycles with max-step guard
- fork and merge
- merge failure
- HITL pause and resume
- reject loop and approve completion
- unknown node compile error
- dangling node compile error
- cancellation
- node error propagation
- checkpoint round trip
- run upsert
- run listing pagination and newest-first order
- checkpoint ordering and latest checkpoint
- RPC run/list/get/checkpoint/resume flow
- per-agent blueprint validation for every built-in agent
- live turn graph preserves tool advertisement
- live turn graph rejects unknown tools
- live turn graph rejects malformed tool arguments without execution
- live turn graph final output depends on actual tool execution

## Implementation Notes

OpenHuman PR #4261 intentionally keeps the generic graph engine independent from
the agent harness. RustAgents should keep the same boundary:

- graph core depends only on graph traits, ids, commands, state, cancellation,
  and progress sinks
- harness integration lives in nodes, definitions, middleware, and runtime
  adapters
- graph checkpointing belongs to graph runtime, while harness stores still own
  events, artifacts, messages, and application data
- the simple harness path remains available for direct calls and ordinary
  model-tool loops
