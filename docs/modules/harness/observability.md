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

RustAgents should use typed Rust events rather than stringly callback names.

## Responsibilities

- Emit typed lifecycle events.
- Support multiple event sinks.
- Support redaction before persistence or external export.
- Support streaming subscribers.
- Support deterministic test collectors.
- Support durable event journals through `store`.
- Preserve parent/root run relationships.
- Include usage, cost, cache, retry, fallback, and timing data.

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
