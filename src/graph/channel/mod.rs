//! Channel-per-field state model (additive).
//!
//! See [`types`] for the type definitions and the high-level model. This file
//! supplies the concrete [`Channel`] merge rules, the [`ChannelSet`] map
//! operations, and the [`ChannelState`] ⇒ [`StateReducer`] bridge that lets a
//! channel graph run on the existing executor.
//!
//! ## How a channel graph runs on the unchanged executor
//!
//! The executor folds a superstep's branch results one at a time:
//! `state = reducer.apply(state, update)` for each branch's
//! [`ChannelUpdate`]. [`ChannelState`] is its own reducer, so each `apply`
//! dispatches every write in the update to the owning channel's
//! [`Channel::merge`].
//!
//! ## Concurrent-write conflict detection
//!
//! When two fan-out branches write the *same* channel in *one* superstep, the
//! merge must decide whether that is legal:
//!
//! - **Aggregate channels** ([`Topic`], [`BinaryAggregate`], [`Delta`],
//!   [`Messages`], [`Barrier`], [`NamedBarrier`]) set
//!   [`Channel::allows_concurrent`] to `true`; both writes fold in
//!   deterministic active-set index order.
//! - **Overwrite channels** ([`LastValue`], [`Ephemeral`], [`Untracked`])
//!   return `false`; a second same-step write to such a channel raises
//!   [`TinyAgentsError::InvalidConcurrentUpdate`] because there is no
//!   deterministic winner.
//!
//! Because the executor applies a step's updates as a contiguous batch, "same
//! step" is tracked by stamping each [`ChannelUpdate`] with the node's
//! `ctx.step` via [`ChannelUpdate::at_step`]. When updates are stamped, the
//! reducer resets its per-step bookkeeping (and clears [`Ephemeral`] channels)
//! whenever the step number advances. Unstamped updates are each treated as
//! their own step (last-value writes always win, no conflict detection and no
//! ephemeral clearing) — so existing whole-state habits keep working and
//! conflict detection is strictly opt-in.

mod types;

pub use types::{
    Barrier, BinaryAggregate, Channel, ChannelSet, ChannelState, ChannelUpdate, Delta, Ephemeral,
    LastValue, Messages, NamedBarrier, Topic, Untracked,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use serde_json::Value;

use crate::graph::reducer::StateReducer;
use crate::{Result, TinyAgentsError};

// --- Channel merge rules ---

impl Channel for LastValue {
    fn kind(&self) -> &'static str {
        "last_value"
    }

    fn merge(&self, _current: Option<&Value>, incoming: Value) -> Result<Value> {
        Ok(incoming)
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Topic {
    fn kind(&self) -> &'static str {
        "topic"
    }

    fn merge(&self, current: Option<&Value>, incoming: Value) -> Result<Value> {
        let mut list = match current {
            Some(Value::Array(items)) => items.clone(),
            Some(other) => vec![other.clone()],
            None => Vec::new(),
        };
        match incoming {
            Value::Array(items) => list.extend(items),
            other => list.push(other),
        }
        Ok(Value::Array(list))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Delta {
    fn kind(&self) -> &'static str {
        "delta"
    }

    fn merge(&self, current: Option<&Value>, incoming: Value) -> Result<Value> {
        let add_err =
            || TinyAgentsError::Graph("Delta channel only accepts numeric writes".to_string());
        let incoming_num = incoming.as_f64().ok_or_else(add_err)?;
        let Some(current) = current else {
            return Ok(incoming);
        };
        let current_num = current.as_f64().ok_or_else(add_err)?;

        // Stay in integer space when both operands are integers.
        if current.is_i64() && incoming.is_i64() {
            let sum = current.as_i64().unwrap() + incoming.as_i64().unwrap();
            return Ok(Value::from(sum));
        }
        Ok(Value::from(current_num + incoming_num))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Messages {
    fn kind(&self) -> &'static str {
        "messages"
    }

    fn merge(&self, current: Option<&Value>, incoming: Value) -> Result<Value> {
        let mut list = match current {
            Some(Value::Array(items)) => items.clone(),
            Some(_) => {
                return Err(TinyAgentsError::Graph(
                    "Messages channel value must be a JSON array".to_string(),
                ));
            }
            None => Vec::new(),
        };
        let incoming = match incoming {
            Value::Array(items) => items,
            other => vec![other],
        };
        // Build an id -> index map over the existing list once (O(existing)) so
        // each incoming message is an O(1) lookup instead of a linear scan.
        // Previously this dedup was O(existing x incoming), which bit at a few
        // thousand messages.
        let mut index: HashMap<String, usize> = list
            .iter()
            .enumerate()
            .filter_map(|(i, existing)| {
                existing
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|id| (id.to_string(), i))
            })
            .collect();
        for msg in incoming {
            match msg.get("id").and_then(Value::as_str).map(str::to_string) {
                // Keyed message: replace the same id in place, or append and
                // remember its position for later incoming writes.
                Some(id) => match index.get(&id) {
                    Some(&i) => list[i] = msg,
                    None => {
                        index.insert(id, list.len());
                        list.push(msg);
                    }
                },
                // Unkeyed message: always appended (unchanged behavior).
                None => list.push(msg),
            }
        }
        Ok(Value::Array(list))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Ephemeral {
    fn kind(&self) -> &'static str {
        "ephemeral"
    }

    fn merge(&self, _current: Option<&Value>, incoming: Value) -> Result<Value> {
        Ok(incoming)
    }

    fn is_ephemeral(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Untracked {
    fn kind(&self) -> &'static str {
        "untracked"
    }

    fn merge(&self, _current: Option<&Value>, incoming: Value) -> Result<Value> {
        Ok(incoming)
    }

    fn is_tracked(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Barrier {
    /// Creates a count-based barrier that is ready after `expected` arrivals.
    pub fn new(expected: usize) -> Self {
        Self { expected }
    }
}

impl Channel for Barrier {
    fn kind(&self) -> &'static str {
        "barrier"
    }

    fn merge(&self, current: Option<&Value>, incoming: Value) -> Result<Value> {
        let mut list = match current {
            Some(Value::Array(items)) => items.clone(),
            Some(other) => vec![other.clone()],
            None => Vec::new(),
        };
        match incoming {
            Value::Array(items) => list.extend(items),
            other => list.push(other),
        }
        Ok(Value::Array(list))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn is_ready(&self, current: Option<&Value>) -> bool {
        current
            .and_then(Value::as_array)
            .map(|items| items.len() >= self.expected)
            .unwrap_or(self.expected == 0)
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl NamedBarrier {
    /// Creates a name-based barrier that is ready once every name has arrived.
    pub fn new(expected: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            expected: expected.into_iter().map(Into::into).collect(),
        }
    }
}

impl Channel for NamedBarrier {
    fn kind(&self) -> &'static str {
        "named_barrier"
    }

    fn merge(&self, current: Option<&Value>, incoming: Value) -> Result<Value> {
        let mut map = match current {
            Some(Value::Object(map)) => map.clone(),
            Some(_) => {
                return Err(TinyAgentsError::Graph(
                    "NamedBarrier channel value must be a JSON object".to_string(),
                ));
            }
            None => serde_json::Map::new(),
        };
        let Value::Object(incoming) = incoming else {
            return Err(TinyAgentsError::Graph(
                "NamedBarrier writes must be JSON objects of named arrivals".to_string(),
            ));
        };
        for (key, value) in incoming {
            map.insert(key, value);
        }
        Ok(Value::Object(map))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn is_ready(&self, current: Option<&Value>) -> bool {
        let Some(Value::Object(map)) = current else {
            return self.expected.is_empty();
        };
        self.expected.iter().all(|name| map.contains_key(name))
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(self.clone())
    }
}

impl BinaryAggregate {
    /// Creates an aggregate channel from a binary fold closure. The first write
    /// becomes the value directly; later writes are `fold(current, incoming)`.
    pub fn new<F>(fold: F) -> Self
    where
        F: Fn(Value, Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            fold: Arc::new(fold),
        }
    }

    /// Builds an aggregate channel from a [`crate::graph::Reducer<Value>`].
    pub fn from_reducer<R>(reducer: R) -> Self
    where
        R: crate::graph::Reducer<Value> + 'static,
    {
        Self::new(move |current, incoming| reducer.reduce(current, incoming))
    }
}

impl Channel for BinaryAggregate {
    fn kind(&self) -> &'static str {
        "binary_aggregate"
    }

    fn merge(&self, current: Option<&Value>, incoming: Value) -> Result<Value> {
        match current {
            Some(current) => (self.fold)(current.clone(), incoming),
            None => Ok(incoming),
        }
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(self.clone())
    }
}

// --- ChannelSet ---

impl ChannelSet {
    /// Creates an empty channel set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `channel` under `name`, returning the set for chaining.
    pub fn with_channel(
        mut self,
        name: impl Into<String>,
        channel: impl Channel + 'static,
    ) -> Self {
        self.add_channel(name, channel);
        self
    }

    /// Registers `channel` under `name`.
    pub fn add_channel(&mut self, name: impl Into<String>, channel: impl Channel + 'static) {
        self.channels.insert(name.into(), Box::new(channel));
    }

    /// Returns the current value of `name`, if any has been written.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    /// Whether `name` is a registered channel.
    pub fn contains(&self, name: &str) -> bool {
        self.channels.contains_key(name)
    }

    /// Whether the channel `name` permits concurrent same-step writes. Errors
    /// if `name` is not a registered channel.
    pub fn allows_concurrent(&self, name: &str) -> Result<bool> {
        self.channel(name).map(|c| c.allows_concurrent())
    }

    /// Whether the barrier (or other) channel `name` has received everything it
    /// is waiting for. Non-barrier channels are always ready. Errors if `name`
    /// is not registered.
    pub fn is_ready(&self, name: &str) -> Result<bool> {
        let channel = self.channel(name)?;
        Ok(channel.is_ready(self.values.get(name)))
    }

    /// Folds `value` into the channel `name` via its merge rule. Errors with
    /// [`TinyAgentsError::Graph`] if `name` is not a registered channel.
    pub fn apply_update(&mut self, name: &str, value: Value) -> Result<()> {
        let channel = self.channel(name)?;
        let merged = channel.merge(self.values.get(name), value)?;
        self.values.insert(name.to_string(), merged);
        Ok(())
    }

    /// Returns the tracked channel values as an ordered map, excluding
    /// [`Untracked`] channels. This is the durable/inspectable state view.
    pub fn snapshot(&self) -> BTreeMap<String, Value> {
        self.values
            .iter()
            .filter(|(name, _)| {
                self.channels
                    .get(*name)
                    .map(|c| c.is_tracked())
                    .unwrap_or(true)
            })
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    /// Clears the value of every [`Ephemeral`] channel. Called at the start of a
    /// new step by [`ChannelState`].
    pub(crate) fn clear_ephemeral(&mut self) {
        let ephemeral: Vec<String> = self
            .channels
            .iter()
            .filter(|(_, c)| c.is_ephemeral())
            .map(|(name, _)| name.clone())
            .collect();
        for name in ephemeral {
            self.values.remove(&name);
        }
    }

    fn channel(&self, name: &str) -> Result<&dyn Channel> {
        self.channels
            .get(name)
            .map(AsRef::as_ref)
            .ok_or_else(|| TinyAgentsError::Graph(format!("unknown channel `{name}`")))
    }
}

// --- ChannelUpdate ---

impl ChannelUpdate {
    /// Creates an empty update.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a `(name, value)` write, returning the update for chaining.
    pub fn set(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.writes.push((name.into(), value.into()));
        self
    }

    /// Stamps the update with the producing node's superstep (`ctx.step`),
    /// enabling same-step concurrent-write conflict detection and ephemeral
    /// clearing. Without a stamp each update is treated as its own step.
    pub fn at_step(mut self, step: usize) -> Self {
        self.step = Some(step);
        self
    }

    /// Whether the update carries no writes.
    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

// --- ChannelState ---

impl ChannelState {
    /// Creates a state with no channels.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `channel` under `name`, returning the state for chaining. Use
    /// this to declare a graph's channel schema before running.
    pub fn with_channel(
        mut self,
        name: impl Into<String>,
        channel: impl Channel + 'static,
    ) -> Self {
        self.set.add_channel(name, channel);
        self
    }

    /// Borrows the underlying [`ChannelSet`].
    pub fn channels(&self) -> &ChannelSet {
        &self.set
    }

    /// Returns the current value of channel `name`, if written.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.set.get(name)
    }

    /// Returns the tracked channel values (see [`ChannelSet::snapshot`]).
    pub fn snapshot(&self) -> BTreeMap<String, Value> {
        self.set.snapshot()
    }

    /// Whether channel `name` is a satisfied barrier (see
    /// [`ChannelSet::is_ready`]).
    pub fn is_ready(&self, name: &str) -> Result<bool> {
        self.set.is_ready(name)
    }

    /// Folds a [`ChannelUpdate`] into this state, dispatching each write to its
    /// channel's merge rule. This is the core reducer step.
    ///
    /// When the update is stamped (via [`ChannelUpdate::at_step`]) with a step
    /// number that differs from the last one seen, the per-step write tracking
    /// is reset and [`Ephemeral`] channels are cleared before the writes apply.
    /// A second write to a non-aggregate channel within the same stamped step
    /// raises [`TinyAgentsError::InvalidConcurrentUpdate`].
    pub fn merge(mut self, update: ChannelUpdate) -> Result<Self> {
        match update.step {
            Some(step) if step != self.current_step => {
                self.current_step = step;
                self.step_writes.clear();
                self.set.clear_ephemeral();
            }
            Some(_) => {}
            None => {
                // Unstamped updates are independent: no cross-update detection.
                self.step_writes.clear();
            }
        }

        // Distinct channels touched by this single update (a node writing the
        // same channel twice in one update is last-wins, not a conflict).
        let mut distinct: Vec<&str> = Vec::new();
        for (name, _) in &update.writes {
            if !distinct.contains(&name.as_str()) {
                distinct.push(name.as_str());
            }
        }

        // Validate before mutating so a conflicting step never commits partial
        // writes.
        for name in &distinct {
            let allows = self.set.allows_concurrent(name)?;
            let count = self.step_writes.get(*name).copied().unwrap_or(0) + 1;
            if count > 1 && !allows {
                return Err(TinyAgentsError::InvalidConcurrentUpdate(format!(
                    "channel `{name}` received {count} concurrent writes in one step but is not an aggregate channel"
                )));
            }
        }

        let touched: HashSet<String> = distinct.iter().map(|n| n.to_string()).collect();
        for name in touched {
            *self.step_writes.entry(name).or_insert(0) += 1;
        }
        for (name, value) in update.writes {
            self.set.apply_update(&name, value)?;
        }
        Ok(self)
    }
}

/// `ChannelState` is its own [`StateReducer`]: the `&self` receiver is unused
/// (merge rules live in the running `state`'s [`ChannelSet`]), so any
/// `ChannelState` may be passed to `set_reducer`.
impl StateReducer<ChannelState, ChannelUpdate> for ChannelState {
    fn apply(&self, state: ChannelState, update: ChannelUpdate) -> Result<ChannelState> {
        state.merge(update)
    }
}

#[cfg(test)]
mod test;
