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
