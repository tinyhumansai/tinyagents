# Harness Store Feature

The store feature provides durable and ephemeral storage for the harness. It is
used by memory, events, model calls, tool calls, artifacts, web UIs, and tests.

The store is not graph checkpointing. Graph checkpoints belong to the graph
module. Harness stores record application/runtime data around LLM orchestration.
When the harness state-graph runtime is enabled, graph checkpointers own the
serialized graph state snapshots while harness stores can still index the run's
events, artifacts, model calls, tool calls, and UI-facing summaries.

The store may also back status and event-journal features for the harness and
for graph integrations. That does not make those records generic memory: status
records are operational snapshots, event records are append-only execution
history, and checkpoints remain graph-owned resumable state.

## Source Inspiration

LangChain separates generic stores from chat history:

- `BaseStore`, `InMemoryStore`, and byte stores:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/stores.py>
- `BaseChatMessageHistory` and in-memory chat history:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/chat_history.py>
- local file store:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/storage/file_system.py>

RustAgents should follow the separation, but make event-friendly storage a
first-class harness feature.

LangChain v1 agents can also receive graph/store runtime objects through tool
runtime injection. RustAgents should expose stores through `RunContext` and
`ToolRuntime`, while still hiding store handles from model-visible tool schemas.

## Responsibilities

- Provide key/value storage.
- Provide append-only stream storage.
- Store messages, run records, events, tool records, model records, and artifacts.
- Store embedding records and vector/retrieval metadata when configured.
- Store graph run indexes, graph event journals, and graph UI summaries when
  configured, without taking ownership of checkpoint snapshots.
- Support in-memory tests.
- Support JSONL local development.
- Support MongoDB server deployments.
- Support file/blob artifacts.
- Emit store events.
- Apply redaction rules to event payloads.
- Keep backend-specific code out of memory, tools, and agent loop modules.
- Keep store interfaces separate from model context selection.
- Support namespaces so applications can isolate memory, events, artifacts, and
  cache data.
- Provide deterministic ordering for test stores.

## Non-Responsibilities

- It does not decide what enters a prompt.
- It does not summarize memory.
- It does not checkpoint graph execution.
- It does not replace the graph checkpointer used for pause/resume.
- It does not replace the registry.
- It does not require one global database.

## Core Traits

```rust
#[async_trait]
pub trait Store: Send + Sync {
    async fn get(&self, key: StoreKey) -> Result<Option<StoreValue>>;
    async fn put(&self, key: StoreKey, value: StoreValue) -> Result<()>;
    async fn delete(&self, key: StoreKey) -> Result<()>;
    async fn scan(&self, prefix: StoreKeyPrefix) -> Result<Vec<StoreRecord>>;
}

#[async_trait]
pub trait AppendStore: Send + Sync {
    async fn append(&self, stream: StoreStream, value: StoreValue) -> Result<StoreOffset>;
    async fn read_from(
        &self,
        stream: StoreStream,
        offset: StoreOffset,
    ) -> Result<Vec<StoreRecord>>;
}
```

Value model:

```rust
pub enum StoreValue {
    Json(serde_json::Value),
    Text(String),
    Bytes(Vec<u8>),
}

pub struct StoreRecord {
    pub key: StoreKey,
    pub value: StoreValue,
    pub version: Option<String>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub metadata: serde_json::Value,
}
```

## Store Registry

Stores are namespaced.

```rust
pub struct StoreRegistry {
    stores: HashMap<StoreNamespace, Arc<dyn Store>>,
    append_stores: HashMap<StoreNamespace, Arc<dyn AppendStore>>,
}
```

Example:

```rust
let stores = StoreRegistry::new()
    .register("threads", MongoStore::new(mongo, "threads"))
    .register_append("events", JsonlStore::new("./data/events.jsonl"))
    .register("artifacts", FileStore::new("./data/artifacts"))
    .register("cache", InMemoryStore::new());
```

`RunContext` should expose the registry:

```rust
ctx.stores.get("threads")?.put(key, value).await?;
ctx.stores.append("events")?.append(stream, value).await?;
```

## Backends

### InMemoryStore

Use for:

- unit tests
- examples
- deterministic replay
- local prototyping

Properties:

- no durability
- easiest to assert
- should support deterministic ordering
- should be cloneable through `Arc`

### JsonlStore

Use for:

- local development
- event journals
- run replay
- debugging
- append-only audit trails

Properties:

- one JSON record per line
- append-only by default
- easy to tail
- easy to inspect with shell tools
- supports replay from offset

Record shape:

```json
{"stream":"runs/support-123/events","offset":42,"time":"2026-06-29T00:00:00Z","value":{"kind":"tool.completed"}}
```

JSONL is not ideal for high-concurrency writes unless guarded by a writer task or
file lock. The implementation should document its concurrency guarantees.

### FileStore

Use for:

- artifacts
- large tool outputs
- provider raw payload snapshots
- prompt fixtures
- binary files

Properties:

- key maps to safe relative path
- blocks path traversal
- optional sidecar metadata
- can pair with JSONL event records that reference artifact keys

### MongoStore

Use for:

- server deployments
- thread records
- run records
- message history
- event query APIs
- user/application memory

Properties:

- indexes by thread id, run id, component id, timestamp
- stores JSON-like documents naturally
- supports UI queries better than JSONL
- useful for multi-agent dashboards

Suggested collections:

- `threads`
- `runs`
- `run_status`
- `messages`
- `events`
- `event_offsets`
- `tool_calls`
- `model_calls`
- `artifacts`
- `memory`
- `embeddings`
- `retrievals`
- `graph_runs`
- `graph_events`

Minimum indexes:

- `events.run_id + events.time`
- `events.thread_id + events.time`
- `events.run_id + events.offset`
- `run_status.run_id`
- `run_status.thread_id + run_status.updated_at`
- `runs.thread_id + runs.created_at`
- `messages.thread_id + messages.created_at`
- `tool_calls.run_id + tool_calls.call_id`
- `model_calls.run_id + model_calls.call_id`
- `retrievals.run_id + retrievals.time`
- `embeddings.model + embeddings.created_at`
- `graph_runs.status + graph_runs.updated_at`
- `graph_events.run_id + graph_events.time`

### Future Backends

- SQLite for local durable single-process apps.
- Postgres for production relational deployments.
- Redis for cache/session data.
- S3-compatible blob store for artifacts.
- Vector store adapter for retrieval memory.

## Store Events

Store operations that affect harness-visible state should emit events:

- store get
- store put
- store delete
- append
- read from stream
- artifact written
- artifact read
- backend error

Payloads should be redacted by sink policy. Events should include namespace,
key or key fingerprint, stream name, offset when applicable, elapsed time, and
backend name.

## Status And Event Journal Records

Store-backed observability uses two different record families:

- status records are overwritten by id and answer "what is running now?"
- event journal records are append-only and answer "what happened?"

Harness status records should store the compact `HarnessRunStatus` shape from
[observability](observability.md). Graph integrations may store graph status
records in the same backend, but graph-specific fields such as checkpoint id,
checkpoint namespace, step, active nodes, and interrupts remain graph-owned.

Event journal records should be addressable by stream and offset:

```rust
pub struct StoredEventRecord {
    pub stream: StoreStream,
    pub offset: StoreOffset,
    pub event_id: EventId,
    pub run_id: RunId,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub time: SystemTime,
    pub kind: String,
    pub redaction: RedactionStatus,
    pub value: StoreValue,
}
```

Recommended streams:

- `runs/{run_id}/events`
- `threads/{thread_id}/events`
- `roots/{root_run_id}/events`
- `components/{component_id}/events`

The same event may appear in multiple query indexes, but one append stream
should be the authoritative ordered source for replay. Consumers should store
the last processed offset, not the last timestamp.

## Data Records

Run record:

```rust
pub struct StoredRun {
    pub run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub component_id: ComponentId,
    pub status: RunStatus,
    pub started_at: SystemTime,
    pub ended_at: Option<SystemTime>,
    pub metadata: serde_json::Value,
}
```

Run status record:

```rust
pub struct StoredRunStatus {
    pub run_id: RunId,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub component_id: ComponentId,
    pub status: RunStatus,
    pub phase: Option<String>,
    pub last_event_id: Option<EventId>,
    pub updated_at: SystemTime,
    pub ended_at: Option<SystemTime>,
    pub error: Option<ErrorSummary>,
    pub metadata: serde_json::Value,
}
```

Message record:

```rust
pub struct StoredMessage {
    pub id: MessageId,
    pub thread_id: ThreadId,
    pub run_id: RunId,
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
    pub tool_call_id: Option<String>,
    pub created_at: SystemTime,
}
```

Tool call record:

```rust
pub struct StoredToolCall {
    pub call_id: CallId,
    pub run_id: RunId,
    pub tool: ComponentId,
    pub arguments: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub status: CallStatus,
    pub started_at: SystemTime,
    pub ended_at: Option<SystemTime>,
}
```

Event record should use the registry event envelope when the registry module is
enabled.

## Event Emission

Store operations should emit events:

- `store.get.started`
- `store.get.completed`
- `store.put.started`
- `store.put.completed`
- `store.append.started`
- `store.append.completed`
- `store.delete.started`
- `store.delete.completed`
- `store.error`

Event payloads must not include full values by default. Emit summaries:

- namespace
- key
- stream
- offset
- value type
- byte size
- redaction status
- elapsed time

## Redaction

Stores may contain sensitive data.

Default redaction:

- do not emit raw values in events
- redact keys containing `secret`, `token`, `password`, `api_key`
- redact provider raw request/response bodies unless explicitly enabled
- allow per-namespace redaction policies

## Relationship To Memory

`memory` should depend on `store`, not the reverse.

Example:

```text
memory::ConversationMemory
  -> store::StoreRegistry["threads"]
  -> MongoStore or JsonlStore
```

This keeps memory policy testable while allowing the backing storage to change.

## Relationship To Registry

The registry can register stores as components for discovery and event routing.
The harness can also use a local `StoreRegistry` without the full component
registry in small applications.

## Implementation Milestones

### S1: Store Traits

- `Store`
- `AppendStore`
- `StoreValue`
- `StoreRecord`
- `StoreRegistry`

### S2: In-Memory Backend

- deterministic map store
- deterministic append store
- test assertions

### S3: JSONL Backend

- append-only stream
- replay from offset
- local event journal example

### S4: File Backend

- safe relative paths
- artifact metadata
- binary values

### S5: MongoDB Backend

- feature-gated `mongodb`
- run/message/event collections
- indexes
- integration test behind env var

### S6: Redaction And Events

- store operation events
- value summaries
- namespace redaction policies
