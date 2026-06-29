# Graph Checkpointing, Durability, State Inspection, And Time Travel

Checkpointing is graph runtime persistence. It is separate from harness memory
and long-term stores.

LangGraph stores graph state through checkpointers. The checkpoint package
defines this as the graph persistence layer: it saves graph state at every
superstep and enables human-in-the-loop, memory between interactions, durable
execution, and replay. Its core persistence unit is a checkpoint tuple: the
checkpoint itself plus config, metadata, parent config, and pending writes.

Checkpoint coordinates are thread-based:

- `thread_id` identifies the checkpoint lineage for a conversation, workflow, or
  tenant-isolated run series.
- `checkpoint_id` optionally selects a specific point in that thread, including
  time-travel or replay from the middle of a thread.
- `checkpoint_ns` scopes nested graph/subgraph state so subgraphs can share a
  checkpointer without colliding with parent checkpoints.

```rust
#[async_trait]
pub trait Checkpointer: Send + Sync {
    async fn get_tuple(
        &self,
        query: CheckpointQuery,
    ) -> Result<Option<CheckpointTuple>>;

    async fn list(
        &self,
        query: CheckpointListQuery,
    ) -> Result<Vec<CheckpointTuple>>;

    async fn put(
        &self,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Result<CheckpointConfig>;

    async fn put_writes(
        &self,
        config: CheckpointConfig,
        writes: Vec<PendingWrite>,
        task_id: TaskId,
        task_path: TaskPath,
    ) -> Result<()>;
}
```

Checkpoint tuple:

```rust
pub struct CheckpointTuple {
    pub config: CheckpointConfig,
    pub checkpoint: Checkpoint,
    pub metadata: CheckpointMetadata,
    pub parent_config: Option<CheckpointConfig>,
    pub pending_writes: Vec<PendingWrite>,
}
```

Checkpoint fields:

- version
- checkpoint id
- thread id
- checkpoint namespace
- graph id
- run id
- timestamp
- channel values
- channel versions
- versions seen by each node
- updated channels
- next active nodes
- pending sends
- pending writes
- task outcomes
- interrupts
- parent checkpoint config
- metadata source: `input`, `loop`, `update`, or `fork`

Durability modes:

- `sync`: persist before the next step starts.
- `async`: persist while the next step executes.
- `exit`: persist only when the graph exits.

Backends:

- in-memory
- file-backed JSON/JSONL
- SQLite
- Postgres later

Thread operations:

- list checkpoints for a thread
- delete all checkpoints for a thread
- delete checkpoints by run id
- copy a thread to a new thread id
- prune checkpoints with a documented strategy

Delta channels require careful copy and prune semantics. A checkpoint backend
must not keep only the latest checkpoint if the latest checkpoint depends on
ancestor pending writes or a previous delta snapshot.

## Long-Term Stores

LangGraph also has a `BaseStore`, but that is not the same thing as graph state
checkpointing. Stores provide long-term memory that can persist across threads
and conversations. They support hierarchical namespaces, key-value items,
metadata, and optional vector search.

RustAgents should mirror this separation:

- checkpointers store execution state needed to resume a graph exactly
- stores hold application memory, records, artifacts, and searchable data that
  graph or harness nodes may read and write

Compiled graphs may receive a store registry at compile/run time, and
`GraphContext` may expose stores to nodes, but the executor must not use stores
as a substitute for checkpoints.

## State Inspection And Time Travel

Compiled graphs with checkpointing should expose:

- `get_state(thread_id, checkpoint_id)`
- `get_state_history(thread_id, before, limit, filter)`
- `update_state(thread_id, values, as_node, task_id)`
- `bulk_update_state(thread_id, supersteps)`
- `fork_state(source_checkpoint, target_thread_id)`

State snapshots contain:

- current values
- next node names
- config used to fetch the snapshot
- checkpoint metadata
- creation timestamp
- parent config
- tasks for the next step
- task errors and results from attempted work
- pending interrupts

Manual state updates are graph writes. They must pass through the same channel
reducers, produce checkpoint metadata with source `update`, and validate
`as_node` when a caller attributes the write to a node.

Time travel is implemented by invoking or streaming from an older checkpoint
config or by forking a thread. It must not mutate old checkpoint records.
