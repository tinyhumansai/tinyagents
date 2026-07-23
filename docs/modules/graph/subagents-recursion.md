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

## Steering Sub-Agents

Graph-backed sub-agents can be steered by a parent orchestrator, a graph
supervisor node, middleware, a human operator, or a test. Steering should use
the harness steering command contract, then lower into graph commands,
checkpoint updates, or child-run delivery depending on the target state. See
[Sub-agent and orchestrator steering](../harness/subagent-steering.md).

Supported graph-level steering cases:

- parent orchestrator sends additional instructions to a running child
  sub-agent
- parent orchestrator narrows a child sub-agent's tool or model policy
- parent orchestrator asks a child sub-agent for status before join
- human pauses, resumes, redirects, approves, rejects, or cancels a specific
  child sub-agent run
- human steers the parent orchestrator while children continue or pause under
  policy
- graph supervisor node redirects a child task to a review, retry, or finalize
  node

Graph steering targets must be addressable by run-tree metadata:

```rust
pub enum GraphSteeringTarget {
    ParentRun(RunId),
    ChildRun { parent_run_id: RunId, child_run_id: RunId },
    Task { run_id: RunId, task_id: TaskId },
    Node { run_id: RunId, node_id: NodeId },
    Namespace { run_id: RunId, namespace: Vec<String> },
}
```

Rules:

- parent runs can steer descendants by default, not unrelated runs
- humans can steer any run or child task allowed by the control surface policy
- steering a child must not mutate parent state directly
- child steering is delivered as structured input with actor/provenance, not as
  anonymous user text
- accepted child steering produces child events and parent rollup events
- rejected child steering includes a policy reason
- graph reducers own any state change caused by steering
- checkpoint metadata records pending, applied, and rejected steering commands

When a sub-agent is paused on an interrupt, steering can target that exact
interrupt or child namespace. Resuming a child interrupt should restart or
continue only that child task unless the steering policy explicitly escalates to
the parent graph.

## Orchestrator Steering

The parent orchestrator is itself steerable. Human or supervisor steering can:

- add or replace orchestration instructions at the next safe boundary
- narrow delegation policy before more sub-agents are spawned
- cancel pending child fanout
- redirect the graph to a review, retry, join, or finalize node
- apply reducer-mediated state corrections

The orchestrator must treat steering as external control with provenance. It
should not silently merge human instructions into its own assistant messages or
pretend parent-agent steering came from the original user.

## Steering Events

Graph event streams should include steering alongside sub-agent lifecycle:

- `steering.requested`
- `steering.accepted`
- `steering.rejected`
- `steering.delivered`
- `steering.applied`
- `subagent.steering_received`
- `subagent.steering_rejected`

Every steering event must include root run id, parent run id, child run id when
available, task id when graph-backed, namespace, actor, command kind, policy id,
correlation id, and checkpoint id when available.

## Detached Task Runtime Registry

`DetachedTaskRegistry<Metadata, Status>` complements the durable `TaskStore`
with process-local executor handles. Applications register a task's owner,
metadata, status watch receiver, cancellation token, and abort handle, while
the task's live `SteeringHandle` remains addressable through the shared
`SteeringRegistry`.

The registry enforces owner checks for wait and steering, keeps timed-out waits
registered, prunes terminal waits, cancels cooperatively before hard-aborting,
and sweeps terminal entries when its soft cap is reached. It never evicts live
work to satisfy the cap and does not duplicate durable lifecycle records.
After a restart, applications reconcile live `TaskStore` records that have no
matching runtime entry according to their executor policy.

## Listing Orchestration Tasks {#orchestrate-list}

The model-facing `orchestrate_list` tool enumerates managed tasks visible to the
current orchestration scope, filtered through `OrchestrationTaskFilter`. Beyond
the run-tree fields (`parent_run_id`, `root_run_id`, `thread_id`, `node_id`,
`status`), the filter supports:

- `kind` — a task-kind discriminant label matching
  `OrchestrationTaskKind::as_str` (`"graph"`, `"sub_agent"`, `"tool"`,
  `"external_process"`).
- `created_after_ms` / `created_before_ms` — an inclusive created-at window in
  Unix-epoch **milliseconds**, parsed into `SystemTime` bounds against each
  record's `created_at`.

`OrchestrationTaskFilter` exposes `with_kind(label)` and
`created_between(after, before)` builders (each bound is `Option<SystemTime>`),
plus a `matches(record)` predicate. The tool schema accepts the same names:

```rust
// orchestrate_list arguments (all optional):
json!({ "kind": "sub_agent" })            // only sub-agent tasks
json!({ "created_after_ms": 0 })          // created at/after the epoch (all)
json!({ "created_before_ms": 0 })         // created at/before the epoch (none)
```

For example, spawning one `graph` task and one `sub_agent` task and then listing
with `{ "kind": "sub_agent" }` returns just the sub-agent record; a
`{ "created_before_ms": 0 }` upper bound excludes everything created just now,
while `{ "created_after_ms": 0 }` includes both. The full orchestration tool set
(`orchestrate_spawn`, `orchestrate_await`, `orchestrate_list`, and the rest) is
described in the Graph Runtime wiki page.

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
