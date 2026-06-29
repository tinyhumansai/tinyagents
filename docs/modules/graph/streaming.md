# Graph Streaming And Events

The graph stream should support low-level event streams and high-level
projection streams.

Core stream modes:

- `values`: full state values after each step
- `updates`: per-node/per-task state updates
- `messages`: harness message or token deltas emitted by model nodes
- `custom`: arbitrary user stream writes from inside nodes
- `checkpoints`: checkpoint payloads
- `tasks`: task start and task result payloads
- `debug`: checkpoints plus task internals
- `events`: all graph lifecycle events

Typed stream part:

```rust
pub enum StreamPart<State, Output> {
    Values {
        namespace: Vec<String>,
        data: Output,
        interrupts: Vec<Interrupt>,
    },
    Updates {
        namespace: Vec<String>,
        data: IndexMap<NodeId, StateUpdate>,
    },
    Messages {
        namespace: Vec<String>,
        message: Message,
        metadata: StreamMetadata,
    },
    Custom {
        namespace: Vec<String>,
        data: serde_json::Value,
    },
    Checkpoint {
        namespace: Vec<String>,
        data: CheckpointPayload<State>,
    },
    Tasks {
        namespace: Vec<String>,
        data: TaskStreamPayload,
    },
    Debug {
        namespace: Vec<String>,
        data: DebugPayload<State>,
    },
}
```

Event stream:

```rust
pub enum GraphEvent {
    RunStarted { run_id: RunId, graph_id: GraphId },
    RunStreamingStarted { run_id: RunId },
    StepStarted { step: usize, active: Vec<NodeId> },
    TaskStarted { task_id: TaskId, node: NodeId, triggers: Vec<String> },
    TaskCompleted { task_id: TaskId, node: NodeId },
    TaskCached { task_id: TaskId, node: NodeId },
    TaskFailed { task_id: TaskId, node: NodeId, error: String },
    StateUpdated { node: NodeId, update: serde_json::Value },
    RouteSelected { node: NodeId, routes: Vec<RouteTarget> },
    ContextForked { parent_task_id: TaskId, child_task_id: TaskId },
    ContextForkJoined { parent_task_id: TaskId, child_task_id: TaskId },
    SubgraphStarted { node: NodeId, child_run_id: RunId, namespace: Vec<String> },
    SubgraphCompleted { node: NodeId, child_run_id: RunId },
    SubAgentStarted { node: NodeId, agent: ComponentId, child_run_id: RunId },
    SubAgentCompleted { node: NodeId, agent: ComponentId, child_run_id: RunId },
    RecursionDepthChanged { depth: usize },
    CheckpointSaved { checkpoint_id: CheckpointId },
    InterruptEmitted { interrupt: Interrupt },
    RunDraining { run_id: RunId, reason: String },
    RunCompleted { run_id: RunId },
    RunFailed { run_id: RunId, error: String },
    Custom { name: String, payload: serde_json::Value },
}
```

Streaming requirements:

- graph runs can be consumed as an async stream
- streaming does not require waiting for final state
- every streamed event carries run id, thread id, namespace, step, and node/task
  metadata when available
- subgraph streams preserve nested namespaces
- harness streams from model/tool/sub-agent nodes are forwarded with graph node
  context
- subscribers can filter graph events, harness events, sub-agent events, state
  updates, task payloads, messages, and checkpoints
- a typed run stream should expose final output, interrupted status, and pending
  interrupts even when the caller only subscribed to a subset of projections
