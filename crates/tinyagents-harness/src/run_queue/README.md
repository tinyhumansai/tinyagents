# harness::run_queue

A generic, thread-safe, multi-lane FIFO queue for messages that arrive while a
run is active.

## Why this exists

Hosts decide *which* incoming events should be queued rather than applied
immediately (a steering instruction, a deferred follow-up, extra context to
fold in later) and retain ownership of the queued payload. This module owns
only the reusable FIFO mechanics for the three lanes an agent runtime can
drain at safe iteration boundaries — it has no opinion on what `T` is or when
a lane should be drained.

## On the agent loop path (A4)

A `RunQueueHandle` (`Arc<RunQueue<Message>>`) attached with
`RunContext::with_run_queue` is drained by the built-in agent loop: `Steer`
after each tool batch and at a natural finish, `Followup` at a natural finish
when no steer is pending (running one more turn), `Collect` once at run end
onto `AgentRun::collected`. `RunPolicy::queue_mode` (`All` | `OneAtATime`)
sets how many items a boundary takes; each application emits
`AgentEvent::QueuedMessageApplied { lane, count }`. Middleware stops, limit
stops, pauses, and deferrals leave the queue untouched. Details:
`docs/modules/harness/runtime.md`, "Queued steering and follow-ups".

## Public surface

- [`RunQueue<T>`] — the queue itself. `new`/`Default` create an empty queue;
  `push(lane, item)` appends; `drain(lane)` empties one lane in FIFO order and
  returns its contents; `push_now`/`drain_now` are their synchronous forms,
  for `Drop` impls and other non-async callers; `take(lane, mode)` is the
  `QueueMode`-aware form;
  `status()` snapshots per-lane depth; `clear()` empties every lane and
  returns how many items were dropped.
  `edit_first_where(predicate, edit)` changes one pending payload in place;
  `move_first_where(predicate, lane, index)` relocates one pending item to
  a destination lane and position. Both return `false` if delivery already
  consumed the item. Indices address the destination after removal and clamp
  to its length. Other items keep their relative order.
- [`QueueLane`] — which lane an item belongs to: `Steer` (inject at the next
  safe boundary as an instruction), `Followup` (dispatch as a fresh turn once
  the active run completes), `Collect` (handed back to the host at run end as
  collected context, never injected).
- [`QueueMode`] — how many items `take` (and the loop) consume per call:
  `OneAtATime` or `All` (default).
- [`RunQueueHandle`] — `Arc<RunQueue<Message>>`, the loop-consumable form.
- [`QueueStatus`] — a `Serialize`-able snapshot of per-lane and total pending
  counts.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `RunQueue<T>` and its private `RunQueueInner<T>` storage. |
| `types.rs` | `QueueLane`, `QueueMode`, `QueueStatus`, `RunQueueHandle`. |
| `mod_tests.rs` | Per-lane push/drain ordering, status snapshots, `clear`, and lane independence. |

## Operational constraints

- Every method serializes through a single `std::sync::Mutex` guarding all
  three lanes together. It is held only for in-memory operations and never
  across an `.await`, so the `async` methods never block a runtime thread for
  long and the synchronous `push_now`/`drain_now` are safe anywhere. The
  queue is meant to hold a handful of pending items, not a high-throughput
  channel.
- Edit callbacks run under that lock. Validate replacement data before calling,
  keep callbacks brief, and never call back into the queue. A move to `Steer`
  still waits for a safe runtime boundary; it does not cancel an active tool.
- `SteeringForwarderGuard` drains and delivers under its shutdown gate and
  requeues residual (delivered-but-unapplied) steers synchronously in `Drop`,
  so once the guard has dropped, every steer the run did not apply is back in
  the queue and readable at once.
- `drain` removes and returns items in FIFO order; it does not re-queue on
  failure — a caller that fails to act on a drained item is responsible for
  re-pushing it if that is the desired behavior.
- The queue has no size limit; unbounded growth is a host responsibility to
  guard against if untrusted input can push into a lane.
