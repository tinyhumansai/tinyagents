//! Reducer traits and built-in reducer markers.
//!
//! Reducers define how concurrent or sequential writes merge into channel
//! values at superstep boundaries. Two traits are provided:
//!
//! - [`Reducer<T>`]: merges two values of the same channel type. Used for
//!   channel-style state where each key has its own merge policy.
//! - [`StateReducer<State, Update>`]: merges a partial `Update` into the whole
//!   `State`. This is the milestone-1 contract used by the executor: whole-state
//!   updates are acceptable, while the partial-update path enables typed,
//!   channel-like merging without rewriting node code.

use std::marker::PhantomData;

use crate::Result;

/// Merges two values of the same channel type.
pub trait Reducer<T>: Send + Sync {
    /// Merges `update` into `current`, producing the new channel value.
    fn reduce(&self, current: T, update: T) -> Result<T>;
}

/// Merges a partial `Update` into the whole graph `State`.
pub trait StateReducer<State, Update>: Send + Sync {
    /// Applies `update` to `state`, producing the new state.
    fn apply(&self, state: State, update: Update) -> Result<State>;
}

/// Overwrites the current value with the update (last-value semantics).
#[derive(Clone, Copy, Debug, Default)]
pub struct OverwriteReducer;

/// Appends the update vector onto the current vector.
#[derive(Clone, Copy, Debug, Default)]
pub struct AppendReducer;

/// Unions the update vector into the current vector, skipping duplicates and
/// preserving first-seen order.
#[derive(Clone, Copy, Debug, Default)]
pub struct SetUnionReducer;

/// Keeps the smaller of the two values.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinReducer;

/// Keeps the larger of the two values.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxReducer;

/// A custom binary [`Reducer`] backed by a closure.
pub struct ClosureReducer<T, F> {
    pub(crate) f: F,
    pub(crate) _marker: PhantomData<fn(T, T) -> T>,
}

/// A [`StateReducer`] that overwrites whole state with the update (the default
/// for `State == Update` graphs).
#[derive(Clone, Copy, Debug, Default)]
pub struct OverwriteStateReducer;

/// A custom [`StateReducer`] backed by a closure.
pub struct ClosureStateReducer<State, Update, F> {
    pub(crate) f: F,
    pub(crate) _marker: PhantomData<fn(State, Update) -> State>,
}
