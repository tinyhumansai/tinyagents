# graph::checkpoint

The `Checkpointer` trait and its backends — the durability layer that makes
the recursive graph runtime resumable and time-travelable.

In a recursive-language-model harness, runs nest: a graph node can run another
compiled graph, which can run another, each producing its own state.
Checkpointing snapshots every level of that tree at superstep boundaries and
keys them by `thread_id` / `namespace` so a parent and its embedded subgraphs
never collide (see `graph::subgraph`). Persisting committed state at each
boundary is what lets a run be paused on an interrupt, resumed later, forked,
or replayed for time-travel debugging.

Checkpoints are written **at superstep boundaries only — never mid-node** —
so resuming always reruns a node from its start. There is no partial-node
durability; a node handler must be safe to re-run from scratch if the process
crashes mid-execution.

## Public surface

### The trait

`Checkpointer<State>` (async, `Send + Sync`):

- `put(checkpoint) -> Result<CheckpointId>` — persists a checkpoint, returns
  its id.
- `get(thread_id, checkpoint_id: Option<&str>) -> Result<Option<Checkpoint<State>>>`
  — loads a checkpoint; `None` id loads the latest for the thread.
- A namespace-scoped variant of `get` restricts the lookup to checkpoints
  whose stored namespace matches, which is what keeps a parent run and the
  subgraphs it embeds — sharing a thread id but differing in namespace — from
  loading each other's checkpoints on resume or inspection.
- Additional methods for listing checkpoint history and pruning lineage (see
  `mod.rs` for the full trait).

### Backends

- `FileCheckpointer` (`file.rs`) — durable, file-backed implementation; no
  external dependencies.
- `SqliteCheckpointer` (`sqlite.rs`, feature `sqlite`) — Sqlite-backed
  implementation for concurrent/multi-process access.
- An in-memory backend is also available for tests (see `mod.rs`/`test.rs`);
  suitable only for a single process's lifetime.

### Types (`types.rs`)

- `Checkpoint<State>` — the persisted record: committed `state`, `next_nodes`
  (the pending activation set), `interrupts`, source, and lineage pointers.
  `Checkpoint::to_metadata()` projects the listing-relevant fields into
  `CheckpointMetadata` so a `StateSnapshot` (from `graph::compiled`) and
  `Checkpointer::list` always agree on what a checkpoint "is."
- `CheckpointTuple<State>` — a checkpoint plus its `config` and
  `parent_config`, the shape returned by `get`/history lookups.
- `CheckpointConfig` — addresses a checkpoint by `thread_id` (+ optional
  `checkpoint_id`); `CheckpointConfig::latest(thread_id)` is the common case.
- `CheckpointMetadata` — the compact, listing-facing view of a checkpoint.
- `CheckpointSource` — why a checkpoint was written (superstep boundary,
  interrupt, resumable-failure boundary, ...); round-trips through
  `as_str()` / `parse()`.
- `DurabilityMode` — how aggressively the executor checkpoints (e.g. every
  boundary vs. only on interrupt/failure); set via
  `CompiledGraph::with_durability`.
- `PendingActivation` — a scheduled-but-not-yet-run node activation persisted
  across a checkpoint boundary.
- `BarrierArrivals` — tracks which parallel branches have arrived at a join
  barrier, persisted so a resumed run doesn't re-run already-arrived branches.
- `PendingWrite` — a reducer write buffered but not yet folded into committed
  state at the point the checkpoint was taken.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `Checkpoint`, `CheckpointTuple`, `CheckpointConfig`, `CheckpointMetadata`, `CheckpointSource`, `DurabilityMode`, `PendingActivation`, `BarrierArrivals`, `PendingWrite`. |
| `mod.rs` | The `Checkpointer` trait and the in-memory backend. |
| `file.rs` | `FileCheckpointer`. |
| `sqlite.rs` | `SqliteCheckpointer` (feature `sqlite`). |
| `test.rs` | Unit tests (put/get round-trips, namespace scoping, history, pruning). |

## Operational constraints

- Namespace scoping is load-bearing for subgraph isolation: a checkpointer
  implementation that ignores namespace when a caller asks for it will let a
  parent and a subgraph run collide on resume. Any new backend must honor the
  namespace-scoped `get`/`list` contract.
- `FileCheckpointer` and `SqliteCheckpointer` are safe across process
  restarts; the in-memory backend is not — never use it where resumability
  after a crash is required.
- Checkpoint ids must be collision-free across process restarts (see
  `graph::compiled`'s `next_checkpoint_id`, which delegates to
  `harness::ids::new_checkpoint_id` rather than a process-local counter) or
  lineage pruning and time-travel resume can corrupt.
- `DurabilityMode` trades persistence frequency for write volume — a mode that
  only checkpoints on interrupt/failure means a mid-run crash loses all
  progress since the last such boundary, not just the current node.
