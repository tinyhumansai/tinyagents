//! Channel-per-field state types.
//!
//! This module defines the *channel* state model described in
//! `docs/modules/graph/state-channels.md`. It is an **additive** alternative to
//! the monolithic `State` + [`crate::graph::StateReducer`] path: instead of one
//! whole-state value with one reducer, state is split into independently-named
//! *channels*, each owning its own current value and its own binary merge rule.
//!
//! The model is built from four pieces:
//!
//! - [`Channel`]: the per-key merge policy (overwrite, aggregate, append, …).
//! - [`ChannelSet`]: a named map of channels plus their current values.
//! - [`ChannelUpdate`]: a batch of `(channel_name, value)` writes a node returns.
//! - [`ChannelState`]: a concrete graph `State` wrapping a [`ChannelSet`] that
//!   implements [`crate::graph::StateReducer<ChannelState, ChannelUpdate>`], so a
//!   channel graph is just `GraphBuilder<ChannelState, ChannelUpdate>` running on
//!   the unchanged executor.
//!
//! Values are [`serde_json::Value`]-backed for generality so channels compose
//! with checkpointing, export, and language-defined nodes without per-graph type
//! parameters.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::Result;

/// A single named state channel: it owns the merge rule that folds an incoming
/// update value into the channel's current value at a superstep boundary.
///
/// Channels are object-safe and value-typed ([`serde_json::Value`]) so a
/// [`ChannelSet`] can hold a heterogeneous map of `Box<dyn Channel>`. Each
/// channel decides:
///
/// - [`Channel::merge`] — how an incoming write combines with the current value.
/// - [`Channel::allows_concurrent`] — whether two branches in the *same* step may
///   both write this channel (aggregates: yes; last-value: no, see
///   [`crate::TinyAgentsError::InvalidConcurrentUpdate`]).
/// - [`Channel::is_ephemeral`] — whether the value is cleared at the start of the
///   next step (one-shot channels).
/// - [`Channel::is_tracked`] — whether the value appears in [`ChannelSet::snapshot`]
///   and is considered part of durable state.
/// - [`Channel::is_ready`] — barrier readiness (defaults to always ready).
pub trait Channel: Send + Sync {
    /// A short, stable kind tag used for debugging and topology export.
    fn kind(&self) -> &'static str;

    /// Folds `incoming` into the channel's `current` value, returning the new
    /// value. `current` is `None` the first time the channel is written.
    ///
    /// Takes `current` **by value** so accumulating channels (topics, barriers,
    /// message logs) can reuse the existing backing `Vec`/`Map` in place instead
    /// of cloning the entire accumulated value on every merge, which was
    /// O(existing) allocation per write.
    fn merge(&self, current: Option<Value>, incoming: Value) -> Result<Value>;

    /// Whether more than one concurrent branch may write this channel within a
    /// single superstep. Aggregates (append/fold/accumulate/barrier) return
    /// `true`; overwrite-style channels return `false` and trigger
    /// [`crate::TinyAgentsError::InvalidConcurrentUpdate`] on a same-step clash.
    fn allows_concurrent(&self) -> bool {
        false
    }

    /// Whether the channel's value is cleared at the start of the next step.
    fn is_ephemeral(&self) -> bool {
        false
    }

    /// Whether the channel participates in [`ChannelSet::snapshot`] / durable
    /// state. [`Untracked`] returns `false`.
    fn is_tracked(&self) -> bool {
        true
    }

    /// Barrier readiness: whether the channel has received all the inputs it is
    /// waiting for. Non-barrier channels are always ready.
    fn is_ready(&self, _current: Option<&Value>) -> bool {
        true
    }

    /// Clones the channel into a fresh box (enables `Clone` for [`ChannelSet`]).
    fn clone_box(&self) -> Box<dyn Channel>;
}

impl Clone for Box<dyn Channel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for Box<dyn Channel> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Channel")
            .field("kind", &self.kind())
            .finish()
    }
}

/// Overwrite channel: each write replaces the value (last-value semantics).
///
/// Rejects concurrent same-step writes from multiple branches with
/// [`crate::TinyAgentsError::InvalidConcurrentUpdate`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LastValue;

/// Append channel: accumulates writes into a JSON array across steps.
///
/// A scalar write is pushed as one element; an array write extends the list.
/// Allows concurrent same-step writes (the order follows deterministic
/// active-set index order).
#[derive(Clone, Copy, Debug, Default)]
pub struct Topic;

/// Numeric accumulator: each write is added to the running total.
///
/// Integer writes stay integers; any float write promotes the total to a float.
/// Allows concurrent same-step writes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Delta;

/// Message-merge channel: maintains a JSON array of message objects deduplicated
/// by their `id` field. An incoming message whose `id` matches an existing entry
/// replaces it in place; otherwise it is appended. Allows concurrent same-step
/// writes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Messages;

/// One-shot overwrite channel whose value is cleared at the start of the next
/// step (see [`ChannelUpdate::at_step`] for how step boundaries are detected).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ephemeral;

/// Overwrite channel excluded from [`ChannelSet::snapshot`] and durable-state
/// views. Useful for scratch values that should not be checkpointed.
#[derive(Clone, Copy, Debug, Default)]
pub struct Untracked;

/// Count-based barrier: accumulates writes into a JSON array and is *ready* only
/// once it has collected at least `expected` arrivals. Allows concurrent
/// same-step writes (fan-in is the whole point).
#[derive(Clone, Copy, Debug)]
pub struct Barrier {
    /// Number of arrivals required before [`Channel::is_ready`] returns `true`.
    pub expected: usize,
}

/// Name-based barrier: accumulates writes into a JSON object keyed by arrival
/// name and is *ready* only once every name in `expected` has arrived. Each
/// incoming write is a JSON object whose keys are merged into the accumulator.
/// Allows concurrent same-step writes.
#[derive(Clone, Debug)]
pub struct NamedBarrier {
    /// The set of names that must all arrive before the barrier is ready.
    pub expected: Vec<String>,
}

/// Binary-aggregate channel: folds writes through a user-supplied binary
/// closure (append, add, min, max, custom). The first write becomes the value
/// directly; subsequent writes are `fold(current, incoming)`. Allows concurrent
/// same-step writes.
#[derive(Clone)]
pub struct BinaryAggregate {
    pub(crate) fold: std::sync::Arc<dyn Fn(Value, Value) -> Result<Value> + Send + Sync>,
}

impl std::fmt::Debug for BinaryAggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryAggregate").finish_non_exhaustive()
    }
}

/// A named map of channels plus their current [`serde_json::Value`]s.
///
/// The channel definitions (merge rules) live alongside the values, so the set
/// is self-describing: merging an update only needs the set itself. Construct
/// one with [`ChannelSet::new`] and register channels with
/// [`ChannelSet::with_channel`]; feed writes through
/// [`ChannelSet::apply_update`]; read the durable view with
/// [`ChannelSet::snapshot`].
#[derive(Default)]
pub struct ChannelSet {
    pub(crate) channels: HashMap<String, Box<dyn Channel>>,
    pub(crate) values: HashMap<String, Value>,
}

impl Clone for ChannelSet {
    fn clone(&self) -> Self {
        Self {
            channels: self
                .channels
                .iter()
                .map(|(k, v)| (k.clone(), v.clone_box()))
                .collect(),
            values: self.values.clone(),
        }
    }
}

impl std::fmt::Debug for ChannelSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kinds: BTreeMap<&str, &'static str> = self
            .channels
            .iter()
            .map(|(k, v)| (k.as_str(), v.kind()))
            .collect();
        f.debug_struct("ChannelSet")
            .field("channels", &kinds)
            .field("values", &self.values)
            .finish()
    }
}

/// A batch of `(channel_name, value)` writes returned by a node.
///
/// Build one with [`ChannelUpdate::new`] and chain [`ChannelUpdate::set`]. Tag
/// it with [`ChannelUpdate::at_step`] (passing `ctx.step`) to opt into
/// same-step concurrent-write conflict detection and ephemeral clearing — see
/// the module docs and [`ChannelState`].
#[derive(Clone, Debug, Default)]
pub struct ChannelUpdate {
    pub(crate) writes: Vec<(String, Value)>,
    pub(crate) step: Option<usize>,
}

/// A concrete graph `State` wrapping a [`ChannelSet`].
///
/// `ChannelState` implements [`crate::graph::StateReducer<ChannelState,
/// ChannelUpdate>`] for itself, so a channel-based graph is built directly with
/// `GraphBuilder<ChannelState, ChannelUpdate>` and runs on the unchanged
/// executor: each superstep the executor folds every branch's
/// [`ChannelUpdate`] into the committed `ChannelState` through this reducer,
/// dispatching each write to its channel's merge rule.
///
/// The reducer's `&self` receiver is unused — the merge rules travel inside the
/// running state's [`ChannelSet`] — so any `ChannelState` value (for example
/// [`ChannelState::default`]) can be passed to `set_reducer`.
#[derive(Clone, Debug, Default)]
pub struct ChannelState {
    pub(crate) set: ChannelSet,
    /// The step number of the writes currently accumulated in `step_writes`;
    /// `0` before the first stamped update is seen.
    pub(crate) current_step: usize,
    /// Per-channel write counts within `current_step`, used to detect
    /// concurrent writes to non-aggregate channels.
    pub(crate) step_writes: HashMap<String, usize>,
}
