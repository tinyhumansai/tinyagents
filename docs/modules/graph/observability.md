# Graph Observability And Tracing

LangGraph has several observability surfaces:

- stream modes for values, updates, messages, custom data, checkpoints, tasks,
  debug payloads, and execution events
- `debug` streams that wrap checkpoint, task, and task-result payloads with step
  and timestamp metadata
- callback/tracing integration through the LangChain callback manager
- `ls_integration = "langgraph"` metadata so traces can identify LangGraph runs
- graph lifecycle callbacks for interrupt and resume events
- LangSmith tracing configuration in the SDK
- thread-level stream modes for run lifecycle and state updates
- nested namespace propagation for subgraph streams

TinyAgents should expose observability as a graph feature, not only as a side
effect of streaming. Streaming is the transport; observability is the semantic
contract for what every run, step, task, checkpoint, interrupt, subgraph, and
child agent must report.

## Execution Status Store

Graph runs need a readable status surface in addition to transient event
streams. A caller, web UI, supervisor, or test should be able to ask what a graph
is doing without holding the original stream handle.

The graph runtime should write a compact status record at every execution
boundary:

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
    pub metadata: serde_json::Value,
}

pub enum ExecutionStatus {
    Pending,
    Running,
    Waiting,
    Interrupted,
    Draining,
    Completed,
    Failed,
    Cancelled,
}
```

Required status transitions:

- `Pending` before the first step is scheduled.
- `Running` while one or more graph tasks are executing.
- `Waiting` when the executor is idle but expects more work, such as async
  checkpoint persistence or external resume input.
- `Interrupted` when a human-in-the-loop interrupt has been emitted and the run
  cannot continue without a resume command.
- `Draining` after cancellation or failure while child tasks, streams, and
  checkpoint writes are closing.
- `Completed`, `Failed`, or `Cancelled` as terminal states.

The status store is not a replacement for checkpoints. Checkpoints preserve
resumable state; status records summarize live and recent execution for
observers. A checkpointer may persist status records in the same backend, but
the API should keep `Checkpointer` and `GraphStatusStore` separate.

```rust
#[async_trait]
pub trait GraphStatusStore: Send + Sync {
    async fn put_status(&self, status: GraphRunStatus) -> Result<()>;
    async fn get_status(&self, run_id: RunId) -> Result<Option<GraphRunStatus>>;
    async fn list_by_thread(&self, thread_id: ThreadId) -> Result<Vec<GraphRunStatus>>;
}
```

Status writes should be idempotent by `run_id` and monotonic by `updated_at` or
step. Readers must never need to deserialize full graph state to answer basic
questions such as "is this run still active?", "which node is executing?", or
"which interrupt is waiting?".

Run-level latency can be derived from `started_at` and either `ended_at` or
`updated_at`. This status-derived view is compact and cheap, but detailed step
and node latency comes from durable observations.

## Observability Event Model

Every graph event should carry stable envelope data:

```rust
pub struct GraphObservation {
    pub event_id: EventId,
    pub run_id: RunId,
    pub root_run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub thread_id: Option<ThreadId>,
    pub checkpoint_id: Option<CheckpointId>,
    pub checkpoint_namespace: Vec<String>,
    pub graph_id: GraphId,
    pub node_id: Option<NodeId>,
    pub task_id: Option<TaskId>,
    pub step: Option<usize>,
    pub time: SystemTime,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub kind: GraphObservationKind,
}
```

Required observation kinds:

- `graph.started`
- `graph.completed`
- `graph.failed`
- `graph.draining`
- `step.started`
- `step.completed`
- `task.started`
- `task.completed`
- `task.failed`
- `task.cached`
- `state.updated`
- `route.selected`
- `checkpoint.saved`
- `checkpoint.loaded`
- `interrupt.emitted`
- `interrupt.resumed`
- `subgraph.started`
- `subgraph.completed`
- `subagent.started`
- `subagent.completed`
- `stream.opened`
- `stream.closed`
- `debug.payload`

## Latency Metrics

The observability module exposes `GraphLatencyMetrics` for durable graph
latency rollups. Metrics are derived from `GraphObservation` timestamps, so a
late UI or supervisor can compute them from the event journal without holding
the original in-process stream.

Graph latency metrics include:

- `run_elapsed_ms`: first `run.started` to first terminal `run.completed` or
  `run.failed`
- step latency: `step.started` to `step.completed`, correlated by superstep
- node latency: `node.started` to `node.completed` or `node.failed`,
  correlated by `(node_id, step)`
- total, max, and average latency helpers for completed steps
- total, max, and average latency helpers for completed node handlers

```rust
let observations = journal.read_from(run_id, 0).await?;
let metrics = GraphLatencyMetrics::from_observations(&observations);

let run_ms = metrics.run_elapsed_ms;
let slowest_node_ms = metrics.max_node_ms;
let average_step_ms = metrics.average_step_ms();
```

Incomplete work is ignored. A `step.started` without `step.completed`, or a
`node.started` without `node.completed` / `node.failed`, does not produce a
latency record because there is no terminal timestamp. Failed nodes are retained
with `failed = true` so dashboards can separate slow successes from slow
failures.

### Node / Tool Health

A graph node is the graph's unit of work and is frequently a delegated agent or
tool call (`SubAgentNode`), so per-node success/failure counts double as **tool
health** telemetry. `GraphHealthSummary::from_observations` rolls the same
durable stream into:

- `total_started` / `total_completed` / `total_failed` node counts
- `run_failed` — whether the run emitted `run.failed`
- a per-node `GraphNodeHealth` entry (`started`, `completed`, `failed`), sorted
  by node id, with `failure_rate()` and `is_healthy()` helpers
- `is_healthy()`, `failure_rate()`, and `unhealthy_nodes()` at the run level

```rust
let observations = journal.read_from(run_id, 0).await?;
let health = GraphHealthSummary::from_observations(&observations);

if !health.is_healthy() {
    for node in health.unhealthy_nodes() {
        eprintln!("{} failed {}x", node.node.as_str(), node.failed);
    }
}
```

The summary is attached to the Langfuse trace metadata (below) so it is visible
alongside the exported spans.

## Event Journal And Listener Replay

Outside listeners need two paths:

- live subscription for events emitted after the listener attaches
- replay from a durable offset for events that were emitted earlier

The graph should therefore support an append-only event journal when configured.
Each journal record should include the observation envelope, stream namespace,
monotonic offset, and redaction status. Consumers can store the last seen offset
and resume later without inspecting checkpoints.

```rust
#[async_trait]
pub trait GraphEventJournal: Send + Sync {
    async fn append(&self, observation: GraphObservation) -> Result<EventOffset>;
    async fn read_from(
        &self,
        run_id: RunId,
        offset: EventOffset,
    ) -> Result<Vec<GraphObservationRecord>>;
}
```

Live listeners should be registered against filters rather than ad hoc callback
closures:

- run id, root run id, thread id, graph id, node id, or namespace
- event kind families such as graph, step, task, state, checkpoint, interrupt,
  subgraph, sub-agent, harness child, or debug
- redaction profile
- replay offset

Listener delivery is best-effort by default. If a deployment requires durable
delivery, the listener should read from the event journal and acknowledge
offsets outside the graph executor.

## Observability Cache

Some observability projections are expensive or repetitive to compute. The graph
module should permit cache-backed projections without making cached data the
source of truth.

Cacheable graph observability records include:

- latest status by run id
- latest status by thread id
- node/task timing rollups
- graph latency rollups from durable observations
- latest state-update summary per step
- checkpoint metadata summaries
- visualization/introspection snapshots
- listener replay cursors for test and UI adapters

Cache records must include source coordinates: run id, checkpoint id,
checkpoint namespace, step, event offset, and projection version. If any source
coordinate changes, the projection is stale and must be recomputed.

## Trace Integration

Graph tracing should support:

- run names
- run ids supplied by callers
- inherited root and parent run ids
- tags and metadata
- per-node/task timing
- retry and cache metadata
- checkpoint ids and checkpoint namespaces
- graph recursion depth
- subgraph namespace paths
- child harness model/tool events linked to the graph node/task that emitted
  them

The graph should not require a specific tracing backend. It should expose typed
events and adapters:

- no-op observer
- in-memory observer for tests
- JSONL observer for local debugging
- tracing-span adapter
- a Langfuse ingestion exporter (`GraphLangfuseExporter`, see below)
- OpenTelemetry adapter later
- LangSmith-compatible exporter later

## Langfuse Export

`GraphLangfuseExporter` is the concrete, implemented exporter for the Langfuse
ingestion API. It is pull-based and best-effort: read a run's observations back
from a journal, build a batch, and send it — it never sits on the executor hot
path.

The exporter reuses the harness `LangfuseClient` for transport (Basic or Bearer
auth, endpoint normalization, `207 Multi-Status` handling), so a single set of
credentials serves both the agent and graph sides:

```rust
let exporter = GraphLangfuseExporter::from_env()?; // or ::new(client)
let observations = journal.read_from(run_id, 0).await?;
exporter
    .send_observations(LangfuseTraceConfig::default(), &observations)
    .await?;
```

The batch maps the graph run onto Langfuse's trace/observation model:

- one `trace-create` (id defaults to the run's `root_run_id`; name defaults to
  the `graph_id`; session defaults to the run's `thread_id`) with the
  `GraphHealthSummary` folded into trace metadata
- one structural `span-create` for the graph run (`{trace}:run:{run_id}`, named
  for the `graph_id`), parented to the trace, bracketing the whole run
- one timed `span-create` per superstep (`{trace}:step:{n}`), parented to the
  graph-run span
- one timed `span-create` per node handler (`{trace}:node:{name}:{step}`),
  parented to its superstep span; `node.failed` promotes the span to `ERROR`
  level with the rendered error as `statusMessage`
- one timed `span-create` per embedded subgraph, parented to the graph-run span
- an `event-create` for every remaining observation (routes, checkpoints,
  interrupts, custom writes, run lifecycle), parented to the graph-run span,
  with `run.failed` mapped to `ERROR`

Still-running work (a `node.started` with no terminal event) is exported as an
open span with a start time but no end time.

### Unified Traces With Agents And Tools

Because a graph run and the agent runs its nodes spawn share the same
`root_run_id`, and both the graph and harness exporters default their `traceId`
to that root run id, exporting a graph run **and** its child agent runs lands
every graph step, node, model generation, and tool call under one Langfuse
trace. This is what makes full end-to-end telemetry — including tool health and
tool timing — visible in a single trace tree.

The nesting is exact, not just co-located: the graph-run span
(`{trace}:run:{run_id}`) uses the same id scheme the harness exporter parents
its agent run spans to. A sub-agent a node spawns carries `parent_run_id` equal
to the graph run id, so its harness-exported run span resolves to
`{trace}:run:{graph_run_id}` and nests directly under the graph-run span rather
than floating at the trace root — the graph, its nodes, and their agents form
one contiguous tree.

## Debug Payloads

Debug streams are for executor introspection. They should be structured enough
for tests and UIs, but they are allowed to expose implementation details that
normal event consumers do not need.

Debug payloads should include:

- checkpoint payloads with state values, next nodes, tasks, metadata, and parent
  config
- task start payloads with task id, node name, input, triggers, and metadata
- task result payloads with task id, node name, error, interrupts, and writes
- step number and timestamp for every debug item

## Thread-Level Observability

Server and SDK use cases need thread-level observation in addition to run-level
streams. The graph layer should make it possible to subscribe to:

- run modes plus run completion events
- lifecycle only
- state updates only

This is separate from a single graph run stream. Thread streams watch a durable
thread as runs are created, resumed, interrupted, updated, or completed.

## Redaction And Export

Observation sinks must support redaction at export boundaries. Internal graph
events may include full state updates and task inputs; persistent or remote
observers may need to redact:

- state values
- model messages
- tool arguments/results
- store keys
- checkpoint metadata
- provider payloads
- user identifiers
- secrets and credentials

Redaction must preserve structural fields such as ids, node names, step numbers,
event kinds, timings, and checkpoint coordinates so traces remain useful.
