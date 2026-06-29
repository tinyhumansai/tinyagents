//! Higher-level streaming projections for the harness.
//!
//! This module provides:
//!
//! - [`StreamMode`] — an enum selecting which chunk categories a consumer
//!   wants to receive.
//! - [`StreamChunk`] — a typed union of all chunk categories (state snapshots,
//!   diffs, model deltas, debug output, interrupts, and custom extensions).
//! - [`StreamSink`] — a synchronous, mode-filtered buffer for [`StreamChunk`]s.
//! - [`stream`] — a convenience helper that filters a slice of chunks by a
//!   set of modes and returns the matching subset.
//!
//! The stream module is **independent** of `crate::harness::events`: it
//! provides a higher-level projection API without importing the event bus.
//! Integration between event delivery and stream chunks is the responsibility
//! of the harness runtime.

mod types;

pub use types::*;

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// StreamSink impls
// ---------------------------------------------------------------------------

impl StreamSink {
    /// Creates a sink that accepts chunks for the given modes.
    ///
    /// Pass an empty slice to create a sink that discards every chunk (useful
    /// as a no-op sink in tests).
    pub fn new(modes: impl IntoIterator<Item = StreamMode>) -> Self {
        Self {
            active_modes: modes.into_iter().collect(),
            buffer: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Creates a sink that accepts **all** chunk modes.
    pub fn all() -> Self {
        Self::new([
            StreamMode::Values,
            StreamMode::Updates,
            StreamMode::Messages,
            StreamMode::Debug,
            StreamMode::Interrupts,
            StreamMode::Custom,
        ])
    }

    /// Returns `true` when the given mode is active on this sink.
    pub fn is_active(&self, mode: StreamMode) -> bool {
        self.active_modes.contains(&mode)
    }

    /// Returns a reference to the set of active modes.
    pub fn active_modes(&self) -> &HashSet<StreamMode> {
        &self.active_modes
    }

    /// Adds a mode to the active set.
    pub fn enable(&mut self, mode: StreamMode) {
        self.active_modes.insert(mode);
    }

    /// Removes a mode from the active set. Chunks of that mode already in the
    /// buffer are not removed.
    pub fn disable(&mut self, mode: StreamMode) {
        self.active_modes.remove(&mode);
    }

    /// Buffers a chunk if its mode is active; silently discards it otherwise.
    pub fn push(&self, chunk: StreamChunk) {
        if self.active_modes.contains(&chunk.mode()) {
            self.buffer.borrow_mut().push(chunk);
        }
    }

    /// Returns all buffered chunks in push order and clears the buffer.
    pub fn drain(&self) -> Vec<StreamChunk> {
        self.buffer.borrow_mut().drain(..).collect()
    }

    /// Returns the number of buffered chunks.
    pub fn len(&self) -> usize {
        self.buffer.borrow().len()
    }

    /// Returns `true` when no chunks are buffered.
    pub fn is_empty(&self) -> bool {
        self.buffer.borrow().is_empty()
    }

    /// Peeks at the buffered chunks without consuming them.
    pub fn peek(&self) -> Vec<StreamChunk> {
        self.buffer.borrow().clone()
    }
}

// ---------------------------------------------------------------------------
// stream() helper
// ---------------------------------------------------------------------------

/// Filters a collection of [`StreamChunk`]s to those matching a set of
/// [`StreamMode`]s and returns the matching chunks in input order.
///
/// This is a synchronous, allocation-based helper for contexts where the full
/// chunk list is already collected (for example in tests or post-processing).
/// It does not require an async runtime.
///
/// # Example
///
/// ```
/// use tinyagents::harness::stream::{stream, StreamChunk, StreamMode};
/// use tinyagents::harness::message::MessageDelta;
///
/// let chunks = vec![
///     StreamChunk::Message(MessageDelta { text: "hi".into(), tool_call: None }),
///     StreamChunk::Debug("trace".into()),
///     StreamChunk::Updates(serde_json::json!({"key": "value"})),
/// ];
///
/// let messages = stream(&chunks, &[StreamMode::Messages]);
/// assert_eq!(messages.len(), 1);
///
/// let all = stream(&chunks, &[StreamMode::Messages, StreamMode::Debug, StreamMode::Updates]);
/// assert_eq!(all.len(), 3);
/// ```
pub fn stream(chunks: &[StreamChunk], modes: &[StreamMode]) -> Vec<StreamChunk> {
    let mode_set: HashSet<StreamMode> = modes.iter().copied().collect();
    chunks
        .iter()
        .filter(|c| mode_set.contains(&c.mode()))
        .cloned()
        .collect()
}

#[cfg(test)]
#[cfg(test)]
mod test;
