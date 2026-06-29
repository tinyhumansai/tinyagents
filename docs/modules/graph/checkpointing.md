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

TinyAgents should mirror this separation:

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

## Implemented (TinyAgents)

The checkpoint core lives in `src/graph/checkpoint/`:

- `Checkpoint<State>` — the persisted superstep snapshot (thread/checkpoint ids,
  parent lineage, namespace, committed state, next/completed nodes, pending
  writes, interrupts, and free-form metadata).
- `CheckpointMetadata` — the lightweight listing record. Its `source` field is a
  typed `CheckpointSource` (no longer a bare string).
- `CheckpointSource` — `Input | Loop | Update | Fork` with serde (lowercase wire
  form) and `Display`. `CheckpointSource::parse` recovers it from a string.
- `DurabilityMode` — `Sync | Async | Exit`, default `Sync`. `Sync` persists a
  checkpoint before the next step starts. `Async` persists the boundary state
  once committed (today identical to `Sync`; the variant documents moving
  persistence off the critical path). `Exit` persists only the terminal
  checkpoint and any interrupt boundary (interrupts must persist so the run can
  resume). Set it with `CompiledGraph::with_durability(mode)`.
- `CheckpointConfig { thread_id, checkpoint_id, namespace }` — checkpoint
  coordinates. `CheckpointConfig::latest(thread_id)` addresses the newest
  checkpoint at the root namespace.
- `CheckpointTuple<State> { config, checkpoint, parent_config, pending_writes }`
  — the documented core persistence unit.
- `Checkpointer::get_tuple(config)` — a default trait method composed from `get`
  so every backend gets it for free; it resolves the concrete config and the
  parent's config from the loaded record.

The `Checkpointer` trait retains its existing `put` / `get` / `list` surface.
Two backends are bundled:

- `InMemoryCheckpointer` — an `Arc<Mutex<..>>` map, cheap to clone (clones share
  storage), for tests and ephemeral runs.
- `FileCheckpointer` — a durable JSON/JSONL backend that survives process
  restarts. Each thread maps to one append-only `<thread>.jsonl` file under a
  base directory (one serialized `Checkpoint` per line, in insertion order).
  `put` appends a line; `get`/`list` stream the thread file; `delete_*`/`prune`
  rewrite it (and remove it once empty); `copy_thread` copies the file with the
  `thread_id` rewritten on every record. Thread ids are percent-escaped into a
  single safe filename component, and `list_threads` recovers each canonical
  thread id from the first record rather than un-escaping the filename. The
  `Checkpointer` impl is bound by `State: Serialize + DeserializeOwned` (the
  trait itself stays bound-free, so non-serializable states still use the
  in-memory path). `Checkpoint<State>` derives serde's conditional
  (de)serialization for this.
- `SqliteCheckpointer` — a durable, queryable backend behind the optional
  `sqlite` cargo feature (`rusqlite` with the `bundled` SQLite). Open a file with
  `SqliteCheckpointer::open(path)` or an ephemeral database with
  `SqliteCheckpointer::in_memory()`; clones share one `Arc<Mutex<Connection>>`, so
  in-memory clones share data. Each checkpoint is one row in a `checkpoints` table
  keyed by `(thread_id, checkpoint_id)`: the full record is stored as JSON in a
  `record` column, while the parent id, namespace (json), next nodes (json),
  source, step, run id, and an interrupts flag are projected into their own
  columns so thread listing and parent-chain walks are served by indexes
  (`idx_checkpoints_thread`, `idx_checkpoints_lookup`) without deserializing whole
  states. A monotonic `seq` primary key preserves insertion order, so `get(None)`
  returns the most recent row, `get(Some(id))` the latest row with that id, and
  `list` walks rows in insertion order — matching the other backends. Like
  `FileCheckpointer`, the impl is bound by `State: Serialize + DeserializeOwned`.
  Postgres backends remain future work.

### Thread operations

`Checkpoint` and `CheckpointMetadata` now carry an optional `run_id`
(back-compatible — pre-existing/manual records leave it `None`). The executor
stamps every boundary checkpoint with the producing run id.

The `Checkpointer` trait exposes the documented thread operations. Three are
storage-specific primitives (no default body): `list_threads`, `delete_thread`,
and the low-level `delete_checkpoints(thread_id, ids)`. The higher-level
operations are default trait methods composed from those plus `list`/`get`/`put`,
so every backend inherits them:

- `delete_by_run(thread_id, run_id)` — deletes the checkpoints stamped with a
  run id (composed from `list` + `delete_checkpoints`), returning the count.
- `copy_thread(source, target)` — deep-copies every checkpoint into a new thread
  id, preserving each record's `checkpoint_id` and `parent_checkpoint_id` so the
  lineage spine stays walkable for time-travel/resume (composed from `list` +
  `get` + `put`).
- `prune(thread_id, keep_last)` — retains the most recent `keep_last`
  checkpoints **plus the full `parent_checkpoint_id` ancestor chain of every
  retained checkpoint**, then deletes the rest. Protecting the entire ancestor
  chain is what honors the delta-channel warning: a kept checkpoint that stores
  only a delta (or depends on an ancestor's pending writes/snapshot) can never
  be orphaned from the state it needs. `keep_last == 0` is clamped to `1` so the
  latest checkpoint always survives.

`InMemoryCheckpointer` implements only the three storage primitives; it inherits
`delete_by_run`, `copy_thread`, and `prune` from the trait defaults.

### State inspection & time travel

`CompiledGraph` exposes the documented inspection/time-travel surface when a
checkpointer is configured (every method returns `TinyAgentsError::Checkpoint`
if it is not). A `StateSnapshot<State>` bundles the committed `values`, the
`next_nodes`/`tasks` that would run on resume, the `config` addressing the
snapshot, its `parent_config`, the listing `metadata`, and any
`pending_interrupts`.

- `get_state(thread_id, checkpoint_id)` — loads a snapshot (latest when
  `checkpoint_id` is `None`); `Ok(None)` for an unknown thread/checkpoint.
- `get_state_history(thread_id, limit)` — snapshots newest-first, walking the
  `parent_checkpoint_id` lineage back from the latest checkpoint; `limit` caps
  the count.
- `update_state(thread_id, update, as_node)` — a manual graph write. The
  `update` is folded through the same `StateReducer` the executor uses, on top
  of the thread's latest committed state, and persisted as a new checkpoint with
  source `update`. `as_node` must name a real node (`MissingNode` otherwise); the
  write is attributed to it and the new checkpoint's pending nodes become that
  node's routing successors. With `as_node == None` the latest pending set is
  preserved.
- `bulk_update_state(thread_id, updates)` — applies a sequence of
  `(update, as_node)` pairs as successive `update` checkpoints, each layered on
  the previous one's committed state; returns the last config (errors on an
  empty sequence).
- `fork_state(source_thread, source_checkpoint_id, target_thread)` — copies a
  checkpoint into a new thread as a fresh root (no parent) with source `fork`.
  The source record is read with `get` and never mutated, so forks are
  non-destructive time travel.

Time-travel resume is `resume_from(thread_id, target, command)` where `target`
is a `ResumeTarget` (`Latest` or `Checkpoint(id)`). `resume` is shorthand for
`ResumeTarget::Latest`. Resuming from an older checkpoint replays its pending
nodes forward (applying `command`'s resume value to any interrupted node)
without rewriting history — new boundary checkpoints are appended to the thread.
