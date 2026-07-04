//! Pluggable event sinks for graph streaming and observability — how a
//! recursive run makes its progress visible to the outside (and to a parent
//! run).
//!
//! The durable executor emits a stream of [`GraphEvent`]s as it walks
//! supersteps, schedules tasks, updates state, saves checkpoints, and raises
//! interrupts. Routing those events into a [`GraphEventSink`] is what lets a
//! REPL, a UI, or an enclosing graph watch a subgraph or sub-agent execute in
//! real time; because every event is tagged with its node and step, the streams
//! of nested runs can be merged and attributed back up the run tree.
//!
//! See [`types`] for the event and stream-mode definitions. The executor emits
//! [`GraphEvent`]s into an optional [`GraphEventSink`]; callers can plug in a
//! [`NoopSink`], a test-friendly [`CollectingSink`], or any custom transport.

mod types;

pub use types::{GraphEvent, StreamMode};

use std::sync::{Arc, Mutex};

/// A pluggable target for low-level graph events.
pub trait GraphEventSink: Send + Sync {
    /// Receives one graph event. Implementations must not block the executor.
    fn emit(&self, event: GraphEvent);

    /// Blocks until every event emitted so far has been durably handled.
    ///
    /// Sinks that persist asynchronously (off the executor thread) override this
    /// so callers can guarantee the durable log has caught up — for example
    /// before reading a journal back. The executor calls it after a terminal
    /// run event. The default is a no-op for synchronous/in-memory sinks.
    fn flush(&self) {}
}

/// A sink that drops every event.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopSink;

impl GraphEventSink for NoopSink {
    fn emit(&self, _event: GraphEvent) {}
}

/// A sink that records every event for inspection in tests and UIs.
#[derive(Clone, Default)]
pub struct CollectingSink {
    events: Arc<Mutex<Vec<GraphEvent>>>,
}

impl CollectingSink {
    /// Creates an empty collecting sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a clone of the recorded events.
    pub fn events(&self) -> Vec<GraphEvent> {
        self.events.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Returns the number of recorded events.
    pub fn len(&self) -> usize {
        self.events.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Returns true when no events were recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl GraphEventSink for CollectingSink {
    fn emit(&self, event: GraphEvent) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(event);
        }
    }
}

#[cfg(test)]
mod test;
