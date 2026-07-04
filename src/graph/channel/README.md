# graph::channel

Channel-per-field state model (additive) for the graph runtime — an opt-in
alternative to plain whole-state overwrite that gives individual state fields
their own merge semantics and concurrent-write conflict detection.

`ChannelState` implements `graph::reducer::StateReducer`, so a channel graph
runs on the **unchanged executor**: the executor folds a superstep's branch
results one at a time (`state = reducer.apply(state, update)` per branch's
`ChannelUpdate`), and because `ChannelState` is itself a reducer, each `apply`
dispatches every write in the update to the owning channel's `Channel::merge`.
No executor changes were needed to add this model.

## Channel kinds and merge rules

- **Aggregate channels** — `Topic`, `BinaryAggregate`, `Delta`, `Messages`,
  `Barrier`, `NamedBarrier`. `Channel::allows_concurrent` is `true`: when two
  fan-out branches write the same channel in one superstep, both writes fold
  in deterministic active-set index order.
- **Overwrite channels** — `LastValue`, `Ephemeral`, `Untracked`.
  `allows_concurrent` is `false`: a second same-step write to one of these
  raises `TinyAgentsError::InvalidConcurrentUpdate`, because there is no
  deterministic winner to pick.

## Concurrent-write conflict detection

Because the executor applies a step's updates as a contiguous batch, "same
step" is tracked by stamping each `ChannelUpdate` with the node's `ctx.step`
via `ChannelUpdate::at_step`. When updates are stamped, the reducer resets its
per-step bookkeeping (and clears `Ephemeral` channels) whenever the step
number advances.

Unstamped updates are each treated as their own step — last-value writes
always win, with no conflict detection and no ephemeral clearing — so
existing whole-state habits (a node that writes without stamping) keep
working and conflict detection is strictly **opt-in**.

## Public surface

- `Channel` (trait) — defines `merge` and `allows_concurrent` for one field's
  storage/merge policy.
- `LastValue<T>` — overwrite semantics; last write in a step wins, concurrent
  writes conflict.
- `Ephemeral<T>` — like `LastValue` but cleared at the start of every step
  (useful for per-step scratch signals).
- `Untracked<T>` — overwrite semantics, opts a field entirely out of conflict
  detection even when stamped.
- `Topic<T>` / `Messages<T>` — append-only aggregate channels; concurrent
  writes in one step all append.
- `BinaryAggregate<T>` — combines concurrent writes with a user-supplied
  binary operator.
- `Delta<T>` — accumulates numeric/structural deltas across concurrent
  writes.
- `Barrier` / `NamedBarrier` — join-coordination channels used to detect when
  every expected fan-out branch has arrived.
- `ChannelSet` — the map of named channels making up a graph's channel-typed
  state.
- `ChannelState` — wraps a `ChannelSet` and implements `StateReducer`; the
  type a `CompiledGraph` is parameterized over when using the channel model.
- `ChannelUpdate` — the per-branch write payload; `ChannelUpdate::at_step(n)`
  opts it into step-stamped conflict detection.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Type definitions and the channel/model overview. |
| `mod.rs` | `Channel::merge` rules, `ChannelSet` map operations, the `ChannelState` ⇒ `StateReducer` bridge. |
| `test.rs` | Unit tests (each channel kind's merge rule, conflict detection, step-stamping behavior). |

## Operational constraints

- Conflict detection only fires for stamped updates (`ChannelUpdate::at_step`)
  — a node handler that forgets to stamp silently loses the safety net and
  falls back to last-write-wins.
- `Ephemeral` channels are cleared on step advance, **not** on read — reading
  one after the step in which it was written but before the next step still
  sees the value.
- Mixing channel-typed state with plain whole-state overwrite in the same
  graph is possible (channels are just another reducer) but conflates two
  conflict-detection models; prefer one model per graph for predictability.
