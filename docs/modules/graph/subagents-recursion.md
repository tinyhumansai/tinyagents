# Graph Sub-Agents, Recursion, And Depth Tracking

Sub-agents are harness agents registered in the registry and invoked from graph
nodes. They are not opaque side effects; they are child runs with their own
events, usage, cost, failures, and optional streams.

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
- set `parent_run_id` to the graph node/task run
- forward harness events through graph/registry streaming
- apply timeout, retry, cache, and budget policy
- map child output into parent graph update
- make child usage and cost visible in parent rollups
- include child run references in checkpoints and task events

Sub-agents allow a graph to coordinate specialized workers while keeping each
worker independently observable and testable.

Parallel sub-agent fanout should use the graph's context-forking contract so
child agents inherit run identity, stores, stream sinks, and cache handles while
receiving isolated mutable task scope. See
[Parallel agents and context forking](parallel-agents-forking.md).

## Recursion And Depth Tracking

The graph should allow recursive execution, but only with explicit limits and
tracking.

Recursive cases:

- a graph invokes itself as a subgraph
- a graph invokes a subgraph that eventually invokes the parent graph
- an agent node calls another agent that calls back into the same graph
- a router intentionally loops through nodes until state converges
- a `Send` fanout schedules many recursive child tasks

Required tracking:

```rust
pub struct RecursionFrame {
    pub graph_id: GraphId,
    pub node_id: Option<NodeId>,
    pub run_id: RunId,
    pub task_id: Option<TaskId>,
    pub namespace: Vec<String>,
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
- exceeding `max_depth` fails with a clear graph recursion error
- node-loop recursion and graph-call recursion are tracked separately
- checkpoint metadata includes the current recursion stack
- UIs should be able to render nested runs without reconstructing them from logs
