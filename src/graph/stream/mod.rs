//! Pluggable event sinks for graph streaming and observability.
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
