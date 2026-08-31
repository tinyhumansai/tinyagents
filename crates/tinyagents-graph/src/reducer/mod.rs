//! Reducer trait implementations.
//!
//! Reducers are the deterministic fan-in point of the recursive runtime: when a
//! superstep runs several active nodes — including parallel branches or a node
//! that merged results from a subgraph or sub-agent — the executor folds their
//! updates back into one committed state through a reducer, in deterministic
//! active-set order, so merged state is reproducible regardless of completion
//! order. This module supplies the built-in channel/state reducers and the
//! closure-backed escape hatches; see [`types`] for the trait and marker
//! definitions.

mod types;

pub use types::{
    AppendReducer, ClosureReducer, ClosureStateReducer, MaxReducer, MinReducer, OverwriteReducer,
    OverwriteStateReducer, Reducer, SetUnionReducer, StateReducer,
};

use std::collections::HashSet;
use std::marker::PhantomData;

use crate::Result;

impl<T> Reducer<T> for OverwriteReducer
where
    T: Send + Sync,
{
    fn reduce(&self, _current: T, update: T) -> Result<T> {
        Ok(update)
    }
}

impl<T> Reducer<Vec<T>> for AppendReducer
where
    T: Send + Sync,
{
    fn reduce(&self, mut current: Vec<T>, mut update: Vec<T>) -> Result<Vec<T>> {
        current.append(&mut update);
        Ok(current)
    }
}

impl<T> Reducer<Vec<T>> for SetUnionReducer
where
    T: Eq + std::hash::Hash + Clone + Send + Sync,
{
    fn reduce(&self, current: Vec<T>, update: Vec<T>) -> Result<Vec<T>> {
        let mut seen: HashSet<T> = current.iter().cloned().collect();
        let mut out = current;
        for item in update {
            if seen.insert(item.clone()) {
                out.push(item);
            }
        }
        Ok(out)
    }
}

impl<T> Reducer<T> for MinReducer
where
    T: PartialOrd + Send + Sync,
{
    fn reduce(&self, current: T, update: T) -> Result<T> {
        Ok(if update < current { update } else { current })
    }
}

impl<T> Reducer<T> for MaxReducer
where
    T: PartialOrd + Send + Sync,
{
    fn reduce(&self, current: T, update: T) -> Result<T> {
        Ok(if update > current { update } else { current })
    }
}

impl<T, F> ClosureReducer<T, F>
where
    F: Fn(T, T) -> Result<T> + Send + Sync,
{
    /// Creates a custom reducer from a binary merge closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: PhantomData,
        }
    }
}

impl<T, F> Reducer<T> for ClosureReducer<T, F>
where
    T: Send + Sync,
    F: Fn(T, T) -> Result<T> + Send + Sync,
{
    fn reduce(&self, current: T, update: T) -> Result<T> {
        (self.f)(current, update)
    }
}

impl<State> StateReducer<State, State> for OverwriteStateReducer
where
    State: Send + Sync,
{
    fn apply(&self, _state: State, update: State) -> Result<State> {
        Ok(update)
    }
}

impl<State, Update, F> ClosureStateReducer<State, Update, F>
where
    F: Fn(State, Update) -> Result<State> + Send + Sync,
{
    /// Creates a custom state reducer from an `apply` closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: PhantomData,
        }
    }
}

impl<State, Update, F> StateReducer<State, Update> for ClosureStateReducer<State, Update, F>
where
    State: Send + Sync,
    Update: Send + Sync,
    F: Fn(State, Update) -> Result<State> + Send + Sync,
{
    fn apply(&self, state: State, update: Update) -> Result<State> {
        (self.f)(state, update)
    }
}

#[cfg(test)]
mod test;
