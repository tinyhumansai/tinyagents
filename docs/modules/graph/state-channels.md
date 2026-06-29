# Graph State, Channels, And Updates

The current scaffold returns whole state from each node. Durable graphs should
move to partial state updates applied through channels. A channel owns the
current value, the accepted update type, checkpoint representation, reducer, and
step-boundary merge behavior.

```rust
pub trait Channel: Send + Sync {
    type Value;
    type Update;
    type Checkpoint;

    fn get(&self) -> Result<Self::Value>;
    fn update(&mut self, values: Vec<Self::Update>) -> Result<ChannelChange>;
    fn checkpoint(&self) -> Result<Self::Checkpoint>;
    fn restore(checkpoint: Self::Checkpoint) -> Result<Self>
    where
        Self: Sized;
    fn consume(&mut self) -> Result<ChannelChange> {
        Ok(ChannelChange::Unchanged)
    }
    fn finish(&mut self) -> Result<ChannelChange> {
        Ok(ChannelChange::Unchanged)
    }
}
```

Required channel policies:

- `LastValue`: accepts one update per step and overwrites the value.
- `Overwrite`: explicit overwrite marker for aggregate channels.
- `BinaryAggregate`: applies a binary reducer such as append, add, min, or max.
- `Topic`: pub/sub collection, optionally accumulating across steps.
- `Ephemeral`: value exists only for one step or one trigger.
- `Barrier`: waits until named sources have all arrived.
- `NamedBarrier`: tracks named arrivals for join semantics.
- `Messages`: message merge by id for chat histories.
- `Delta`: stores compact deltas plus periodic snapshots for large append-heavy
  channels.
- `Untracked`: excluded from checkpointing when safe.

Why channel-level reducers matter:

- parallel branches can write different fields safely
- map-reduce fanout can aggregate many outputs in one step
- checkpoints can store pending writes instead of only final whole-state values
- failed parallel nodes can rerun without discarding completed writes
- tests can assert exact writes and reducer behavior
- generated/language-defined nodes can use simple partial update contracts

The graph should support both root state and object state. A single root channel
is useful for scalar workflows; multi-key state is the default for agent graphs.

## Implemented additive model (`graph::channel`)

The channel model is shipped **additively**: the monolithic `State` +
`StateReducer` path is unchanged, and channels are an opt-in alternative that
runs on the *existing* executor. The implementation lives in
`src/graph/channel/` and is `serde_json::Value`-backed for generality.

- `Channel` (object-safe trait): per-key `merge(current, incoming) -> value`
  plus `allows_concurrent`, `is_ephemeral`, `is_tracked`, and `is_ready`
  (barrier) hooks. Concrete channels: `LastValue` (overwrite), `Topic`
  (append into an array), `Delta` (numeric accumulate), `Messages` (merge by
  `id`), `Ephemeral` (overwrite, cleared next step), `Untracked` (overwrite,
  excluded from snapshots), `Barrier`/`NamedBarrier` (count/name fan-in with
  readiness), and `BinaryAggregate` (fold via a closure or any
  `Reducer<Value>`).
- `ChannelSet`: a named map of `Box<dyn Channel>` plus their current values,
  with `add_channel`/`with_channel`, `apply_update(name, value)`, `get`,
  `is_ready`, and `snapshot()` (the tracked, durable view).
- `ChannelState`: a graph `State` wrapping a `ChannelSet`. It implements
  `StateReducer<ChannelState, ChannelUpdate>` for itself (the `&self` reducer
  receiver is unused; merge rules travel inside the running state), so a
  channel graph is built directly with
  `GraphBuilder::<ChannelState, ChannelUpdate>::new().set_reducer(ChannelState::new())`.
- `ChannelUpdate`: a batch of `(name, value)` writes a node returns
  (`ChannelUpdate::new().set(..).set(..)`). Stamp it with the producing node's
  superstep via `.at_step(ctx.step)`.

### Concurrent-write conflict detection

The executor folds a step's branch updates one at a time. Stamping each
`ChannelUpdate` with `ctx.step` lets the reducer group a step's writes: when the
step number advances it resets its per-step bookkeeping and clears `Ephemeral`
channels. A second same-step write to a non-aggregate channel (`LastValue`,
`Ephemeral`, `Untracked` — `allows_concurrent == false`) raises
`TinyAgentsError::InvalidConcurrentUpdate`; aggregate channels
(`allows_concurrent == true`) merge both writes in deterministic active-set
index order. Cross-step overwrites and repeated writes inside one update are
last-wins, not conflicts. Unstamped updates are treated as independent steps
(no conflict detection, no ephemeral clearing), preserving the simplest path.
