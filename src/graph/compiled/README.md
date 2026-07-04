# graph::compiled

The superstep executor for the durable graph — `CompiledGraph<State, Update>`.

This is the engine that makes the recursive runtime durable: it drives a
compiled graph in checkpointed supersteps, and because a node handler may
recurse into another compiled graph (a subgraph) or a sub-agent, every level
of that recursion is observed through the same step/boundary/checkpoint
discipline — child runs roll their state, events, and interrupts up through
the parent's reducer and checkpointer.

## Superstep loop

Each step:

1. Take the active node set.
2. Run each active node against the committed state snapshot.
3. Collect updates / commands / interrupts.
4. Apply the reducer at the step boundary.
5. Persist a checkpoint at the boundary (when a checkpointer is configured).
6. Select the next active set.

The loop stops when the active set empties, every branch reaches `END`, an
interrupt pauses the run, or the recursion limit is hit (a deterministic
`TinyAgentsError::RecursionLimit`).

## Sequential vs. parallel steps

By default execution is sequential within a step. When the graph is compiled
with `GraphBuilder::with_parallel`, a step with more than one active node runs
every branch concurrently via `futures::future::join_all`, but the data flow
is identical to the sequential case: each branch reads the same committed
snapshot (its own clone), and results fold into the reducer in deterministic
active-set order at the step boundary — the merged state is reproducible
regardless of which branch finishes first.

Concurrency and interrupt semantics:

- All active branches in a parallel step start before any is awaited, and all
  are driven to completion (`join_all`) before the step boundary runs.
- Branch results are folded in active-set index order — the reducer is the
  fan-in/join, with lower-index branches' updates applied first.
- The **lowest-index** branch that errors or interrupts is the step's terminal
  outcome. Updates produced by lower-index successful branches are still
  applied/persisted; an error persists a resumable failure boundary (below)
  and aborts; an interrupt persists a checkpoint whose pending nodes are that
  branch and every later active node.
- Because branches run on cloned snapshots and never share mutable state,
  concurrency is data-race free — the reducer alone resolves conflicting
  writes, deterministically, by index.

## Network resilience and resumable failures

Two opt-in mechanisms make a run durable under transient failure and
restartable after a hard one:

- **Node retry** (`CompiledGraph::with_node_retry`) — a node whose handler
  fails with a retryable error (`harness::retry::is_retryable` — the model/tool
  transient class) is re-run from its start up to the policy's attempt cap,
  emitting `GraphEvent::NodeRetryScheduled` and sleeping the opt-in backoff
  between attempts. A single network blip is absorbed without touching the
  run.
- **Resumable failure** — when a handler fails beyond the retry budget (or the
  error is non-retryable), the executor does not discard the step. On a
  checkpointed thread it folds the branches that already completed into
  committed state and persists a failure-boundary checkpoint whose
  `next_nodes` schedule the failed node (and the not-yet-run tail) for a later
  `CompiledGraph::resume` / `CompiledGraph::retry`, with the error and failed
  node stamped into checkpoint metadata. The run reports `Failed` (carrying
  that checkpoint id) and returns the error. A caller can restart it as-is, or
  edit state with `CompiledGraph::update_state` before resuming to continue on
  operator feedback. Without a checkpointer the run aborts immediately, as
  before.

## Public surface

### Construction / configuration (builders, taken by value)

`with_checkpointer`, `with_event_sink`, `with_durability`, `with_node_retry`,
`with_namespace`, `with_recursion_policy`, `with_recursion_frames`,
`with_recursion_node`, `with_event_journal`, `with_status_store`.

Accessors: `graph_id()`, `name()`, `namespace()`.

### Running

- `run(state)` — fresh, un-checkpointed run (or checkpointed with an
  auto-generated thread if a checkpointer is set).
- `run_with_inputs(..)` / `run_with_thread(thread_id, state)` /
  `run_with_thread_inputs(..)` — thread-scoped variants for checkpointed runs
  and multi-entry-point graphs.
- `resume(..)` / `resume_from(..)` — continue a checkpointed thread from its
  last (or a specified) checkpoint, including past an interrupt or a resumable
  failure boundary.
- `retry(thread_id)` — re-run the failed node(s) recorded in the last
  failure-boundary checkpoint.

### State inspection / time travel

- `get_state(..)` / `get_state_history(..)` — read the current or historical
  `StateSnapshot<State>` for a thread.
- `update_state(..)` / `bulk_update_state(..)` — operator-driven state edits
  between runs (e.g. before a `resume`/`retry`).
- `fork_state(..)` — branch a new thread from an existing checkpoint.

### Types

- `CompiledGraph<State, Update>` — the executor itself.
- `GraphExecution<State>` — the result of a run: final/paused state, status,
  interrupts, run tree.
- `GraphInput` — seeds a run at a specific node with a payload
  (`GraphInput::start` / `::new` / `::node`).
- `StateSnapshot<State>` — a point-in-time view (`values`, `tasks`,
  `next_nodes`, `config`, `metadata`, `parent_config`, `pending_interrupts`)
  returned by state-inspection calls, projected from a `CheckpointTuple` via
  `Checkpoint::to_metadata` so it always matches what
  `Checkpointer::list` reports.
- `ResumeTarget` — selects which checkpoint/branch a resume targets.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `CompiledGraph`, `GraphExecution`, `GraphInput`, `StateSnapshot`, `ResumeTarget`. |
| `mod.rs` | Module wiring: shared helpers (`Activation`, checkpoint-id/snapshot projection, barrier persistence) and the builder/configuration `impl` (`with_*`, `emit`). |
| `executor.rs` | The superstep loop and run/resume entry points (`run`, `resume`, `retry`, `execute`, `execute_run`, node retry, resumable-failure persistence). |
| `state_api.rs` | State inspection / time travel (`get_state`, `get_state_history`, `update_state`, `bulk_update_state`, `fork_state`). |
| `routing.rs` | Resolving a completed step's active set into the next superstep's activations (`route`, `route_completed`, interrupt-durability preconditions). |
| `test.rs` | Unit tests (sequential/parallel steps, retry, resumable failure, resume/fork, state history). |

## Operational constraints

- Node retry and resumable failure both require a checkpointer to have
  observable durability; without one, retries still happen in-process but a
  hard failure aborts the run instead of leaving a resumable boundary.
- Parallel-step determinism depends on branches never mutating shared state
  outside their own snapshot clone — a node handler that reaches around the
  snapshot (e.g. into shared interior-mutable state) breaks the "index order
  resolves conflicts" guarantee.
- `id` generation (`next_checkpoint_id`) is collision-free across process
  restarts by delegating to `harness::ids::new_checkpoint_id`, not a
  process-local counter — this matters for resumed threads restarted in a new
  process.
