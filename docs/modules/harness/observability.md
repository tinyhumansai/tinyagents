# Harness Observability And Events

Observability is a first-class harness surface. Every important lifecycle step
should emit typed events that can drive logs, traces, streaming UIs, tests, and
durable run replay.

## Source Inspiration

LangChain uses callbacks, tracers, event streams, usage callbacks, and LangSmith
integration:

- callbacks:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/callbacks>
- tracers:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tracers>
- usage callback:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/usage.py>
- runnable event stream tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/tests/unit_tests/runnables>

TinyAgents should use typed Rust events rather than stringly callback names.

## Responsibilities

- Emit typed lifecycle events.
- Support multiple event sinks.
- Support redaction before persistence or external export.
- Support streaming subscribers.
- Support deterministic test collectors.
- Support durable event journals through `store`.
- Preserve parent/root run relationships.
- Include usage, cost, cache, retry, fallback, and latency data.
- Capture model-call and tool-call latency as first-class metrics.
- Include graph run ids, graph names, super-step indexes, node ids, routing
  commands, checkpoint ids, and interrupt metadata for state-graph runs.

## Execution Status Store

Harness runs need a readable status surface just like graph runs. Direct model
calls, model-tool loops, and harness calls made from graph nodes should all
publish their current execution status so external listeners can inspect runs
without holding an in-process stream.

```rust
pub struct HarnessRunStatus {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub component: ComponentId,
    pub status: ExecutionStatus,
    pub current_phase: HarnessPhase,
    pub model_calls: usize,
    pub tool_calls: usize,
    pub active_model_call: Option<CallId>,
    pub active_tool_calls: Vec<CallId>,
    pub last_event_id: Option<EventId>,
    pub usage: UsageTotals,
    pub cost: CostTotals,
    pub started_at: SystemTime,
    pub updated_at: SystemTime,
    pub ended_at: Option<SystemTime>,
    pub error: Option<HarnessErrorSummary>,
    pub metadata: serde_json::Value,
}

pub enum HarnessPhase {
    Starting,
    LoadingMemory,
    BuildingPrompt,
    CallingModel,
    ValidatingStructuredOutput,
    CallingTools,
    Summarizing,
    SavingMemory,
    Completed,
    Failed,
    Cancelled,
}
```

Status should move through `Pending`, `Running`, `Waiting`, `Completed`,
`Failed`, or `Cancelled` states while `current_phase` describes the active
harness operation. A graph node that invokes a harness should be able to read
the child harness status through `parent_run_id` and `root_run_id`.

Run-level latency can be derived from `started_at` and either `ended_at` or
`updated_at`. This status-derived view is compact and cheap, but it only reports
end-to-end elapsed time; per-call latency comes from the event journal.

```rust
#[async_trait]
pub trait HarnessStatusStore: Send + Sync {
    async fn put_status(&self, status: HarnessRunStatus) -> Result<()>;
    async fn get_status(&self, run_id: RunId) -> Result<Option<HarnessRunStatus>>;
    async fn list_by_thread(&self, thread_id: ThreadId) -> Result<Vec<HarnessRunStatus>>;
}
```

The status store should be pluggable and can be backed by in-memory state,
JSONL, MongoDB, SQLite, or another store backend. Status writes must be compact:
they should include counters, ids, phase, error summaries, and timestamps, not
full prompts, tool outputs, or provider payloads.

## Event Shape

```rust
pub struct HarnessEvent {
    pub id: EventId,
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub component: ComponentId,
    pub time: SystemTime,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub kind: HarnessEventKind,
}
```

Event kinds should include:

- `run.started`
- `run.completed`
- `run.failed`
- `model.started`
- `model.delta`
- `model.completed`
- `model.failed`
- `tool.started`
- `tool.progress`
- `tool.completed`
- `tool.failed`
- `middleware.started`
- `middleware.completed`
- `middleware.failed`
- `memory.loaded`
- `memory.saved`
- `summary.created`
- `cache.hit`
- `cache.miss`
- `usage.recorded`
- `cost.recorded`
- `retry.scheduled`
- `fallback.selected`
- `rate_limit.waited`
- `limit.reached`
- `stream.closed`
- `graph.run.started`
- `graph.node.entered`
- `graph.node.completed`
- `graph.run.paused`
- `graph.run.completed`
- `graph.run.failed`
- `graph.checkpoint.saved`

## Latency Metrics

The observability module exposes `AgentLatencyMetrics` for durable latency
rollups. Metrics are derived from `AgentObservation` timestamps rather than
payload metadata, so they work for in-memory journals, JSONL-backed journals,
and redacted sinks that preserve structural ids.

Agent latency metrics include:

- `run_elapsed_ms`: first `run.started` to first terminal `run.completed` or
  `run.failed`
- model call latency: `model.started` to `model.completed`, correlated by
  `CallId`
- tool call latency: `tool.started` to `tool.completed`, correlated by `CallId`
- total, max, and average latency helpers for completed model calls
- total, max, and average latency helpers for completed tool calls

Tool-call latency is required because tools are often the slowest part of an
agent turn: retrieval, browser work, RPC calls, file processing, and sub-agent
tools all need to be visible independently from provider latency.

```rust
let observations = journal.read_from(run_id, 0).await?;
let metrics = AgentLatencyMetrics::from_observations(&observations);

let run_ms = metrics.run_elapsed_ms;
let slowest_tool_ms = metrics.max_tool_ms;
let average_tool_ms = metrics.average_tool_ms();
```

Incomplete calls are ignored. A `model.started` without `model.completed` or a
`tool.started` without `tool.completed` does not produce a latency record
because there is no terminal timestamp. Failed runs still report
`run_elapsed_ms` when `run.failed` is observed.

## Event Journal And Listener Replay

The harness should support both live event subscribers and durable replay. A UI
or supervisor can then connect after a run has started and reconstruct what
happened from the last known offset.

```rust
#[async_trait]
pub trait HarnessEventJournal: Send + Sync {
    async fn append(&self, event: HarnessEvent) -> Result<EventOffset>;
    async fn read_from(
        &self,
        run_id: RunId,
        offset: EventOffset,
    ) -> Result<Vec<HarnessEventRecord>>;
}
```

Listener filters should include:

- run id, root run id, parent run id, thread id, component id, model id, or tool
  name
- event families such as run, model, tool, middleware, memory, summary, cache,
  usage, cost, retry, fallback, rate-limit, store, or stream
- minimum severity for errors and warnings
- replay offset
- redaction profile

Live event delivery should not block model or tool execution unless the run
policy explicitly requires a mandatory observer. Durable listeners should replay
from the journal rather than relying on in-memory broadcast delivery.

## Observability Cache

The harness cache feature can store derived observability projections. These
records help dashboards and tests read status quickly, but they are not the
authoritative source of run history.

Cacheable observability projections include:

- latest run status by run id
- latest thread status rollup
- usage and cost rollups
- model-call and tool-call timing summaries
- agent latency rollups from durable observations
- cache-hit summaries by model, tool, or prompt version
- redacted prompt/message previews for UIs
- event replay cursors for test collectors

Every cached projection must include a source event offset and projection
version. When the event offset advances, the cached projection is stale.

## Event Sinks

```rust
#[async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: HarnessEvent) -> Result<()>;
}
```

Built-in sinks:

- no-op sink
- in-memory collector
- stdout/debug sink
- JSONL append sink
- broadcast stream sink
- redacting sink wrapper
- fan-out sink

Event emission should not make the model or tool call fail unless policy says
observability is required. When best-effort sinks fail, the harness should emit
or record a sink error where possible.

## Stable Event Ids And Delta Attribution

Emitted `EventId`s are scoped by a stream prefix so they are stable across
process restarts and never collide between concurrent runs. An `EventSink`
built with `EventSink::with_stream_id(prefix)` mints ids of the form
`{prefix}-evt-{offset}`, where `offset` is the sink's monotonic emission
counter. `RunContext` seeds the prefix from the run id
(`EventSink::with_stream_id(config.run_id.as_str())`), so replaying the same run
re-mints identical ids for the same `(stream_id, offset)`. `EventSink::new()`
(the default) instead uses a process-unique prefix (`s<n>`) so two default sinks
never collide at offset 0.

```rust
use tinyagents::harness::events::{AgentEvent, EventSink};

// Two "processes" replaying the same run re-mint identical, stable ids.
let first = EventSink::with_stream_id("run-42");
let second = EventSink::with_stream_id("run-42");
let a = first.emit(AgentEvent::StateUpdate);
assert_eq!(a.id.as_str(), "run-42-evt-0");
assert_eq!(a.id, second.emit(AgentEvent::StateUpdate).id);

// A different run never collides even though both restart at offset 0.
let other = EventSink::with_stream_id("run-99");
assert_ne!(other.emit(AgentEvent::StateUpdate).id, a.id);
```

Because a sink is shared across a recursive run tree, a streamed delta must be
attributable to *its* run without depending on which sink instance delivered it.
`AgentEvent::ModelDelta` therefore carries `run_id` alongside `call_id` and the
`MessageDelta`, so a UI can route each streamed chunk to its run/thread lineage
directly. The `invoke_streaming*` agent-loop methods emit one
`ModelDelta { run_id, .. }` per streamed delta, each tagged with the emitting
run's id.

## Redaction

Redaction should happen at sink boundaries. Internal typed events may carry
full details while a persistent or remote sink can receive redacted payloads.

Redaction policies should handle:

- message content
- tool arguments
- tool results
- provider raw payloads
- metadata
- store keys
- secrets
- PII
- artifact paths

Redacted events should preserve structural fields such as ids, event kinds,
timings, and counters so traces remain useful.

## Graph Events

State-graph event payloads should be rich enough for a UI to render run
progress without loading checkpoints:

- run id
- graph name
- step or super-step index
- node id
- rendered command such as `continue`, `goto:node`, `fork:[a,b]`,
  `interrupt:approval`, or `end`
- elapsed milliseconds
- lifecycle status
- interrupt kind and resume node when paused
- error class and message when failed

Graph event tests should filter by both event kind and run id. A shared global
event bus can otherwise pick up unrelated graph events from parallel tests.
