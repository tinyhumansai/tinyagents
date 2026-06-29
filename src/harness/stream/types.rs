//! Type definitions for the harness streaming surface.
//!
//! All structs, enums, and traits in this module form the public API of
//! `crate::harness::stream`. Implementations, free functions, and tests live
//! in the sibling `mod.rs` and `test.rs` files.
//!
//! The stream module is intentionally independent of `crate::harness::events`
//! so it can be used without the full observability stack.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::harness::message::MessageDelta;

// ---------------------------------------------------------------------------
// StreamMode
// ---------------------------------------------------------------------------

/// Selects which categories of [`StreamChunk`]s a consumer wants to receive.
///
/// Consumers subscribe to one or more modes; a [`StreamSink`] only buffers
/// chunks that match the active mode set. Using a narrow mode set reduces
/// allocation and processing in tight loops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamMode {
    /// Raw state value snapshots (full key/value state blobs).
    Values,
    /// Incremental state diffs (only changed keys).
    Updates,
    /// Model token and message deltas.
    Messages,
    /// Verbose debug output: intermediate steps, routing decisions, phase
    /// transitions, and internal notes.
    Debug,
    /// Human-in-the-loop interrupt notifications.
    Interrupts,
    /// Caller-defined extension chunks passed through without filtering.
    Custom,
}

// ---------------------------------------------------------------------------
// StreamChunk
// ---------------------------------------------------------------------------

/// A single item produced by a streaming harness or graph run.
///
/// Each variant corresponds to a [`StreamMode`]: a chunk is buffered by a
/// [`StreamSink`] only when its matching mode is active.
///
/// All variants derive `Clone` so chunks can be fanned out to multiple
/// consumers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StreamChunk {
    /// A complete state value snapshot (corresponds to [`StreamMode::Values`]).
    Values(serde_json::Value),

    /// An incremental state diff (corresponds to [`StreamMode::Updates`]).
    Updates(serde_json::Value),

    /// An incremental model response fragment (corresponds to
    /// [`StreamMode::Messages`]).
    Message(MessageDelta),

    /// A verbose debug string (corresponds to [`StreamMode::Debug`]).
    Debug(String),

    /// A human-in-the-loop interrupt payload (corresponds to
    /// [`StreamMode::Interrupts`]).
    ///
    /// The inner value carries interrupt metadata (kind, resume node, etc.)
    /// as untyped JSON so the interrupt shape can vary without a hard coupling
    /// to a specific interrupt type.
    Interrupt(serde_json::Value),

    /// A caller-defined chunk that bypasses mode filtering and is always
    /// buffered when [`StreamMode::Custom`] is active.
    Custom(serde_json::Value),
}

impl StreamChunk {
    /// Returns the [`StreamMode`] that gates this chunk variant.
    ///
    /// Use this to implement custom routing or logging without matching on the
    /// full enum.
    pub fn mode(&self) -> StreamMode {
        match self {
            StreamChunk::Values(_) => StreamMode::Values,
            StreamChunk::Updates(_) => StreamMode::Updates,
            StreamChunk::Message(_) => StreamMode::Messages,
            StreamChunk::Debug(_) => StreamMode::Debug,
            StreamChunk::Interrupt(_) => StreamMode::Interrupts,
            StreamChunk::Custom(_) => StreamMode::Custom,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamSink
// ---------------------------------------------------------------------------

/// An in-process buffer for [`StreamChunk`]s filtered by an active set of
/// [`StreamMode`]s.
///
/// Producers call [`StreamSink::push`] to submit chunks; the sink silently
/// discards chunks whose mode is not in the active set. Consumers call
/// [`StreamSink::drain`] to retrieve and clear all buffered chunks.
///
/// `StreamSink` is deliberately synchronous and single-threaded. Wrap it in
/// `Arc<Mutex<…>>` when sharing across threads.
///
/// # Example
///
/// ```
/// use tinyagents::harness::stream::{StreamChunk, StreamMode, StreamSink};
/// use tinyagents::harness::message::MessageDelta;
///
/// let sink = StreamSink::new(vec![StreamMode::Messages]);
/// sink.push(StreamChunk::Message(MessageDelta { text: "hello".into(), tool_call: None }));
/// sink.push(StreamChunk::Debug("ignored".into()));
///
/// let chunks = sink.drain();
/// assert_eq!(chunks.len(), 1);
/// ```
pub struct StreamSink {
    /// The set of modes whose chunks are accepted by this sink.
    pub(crate) active_modes: HashSet<StreamMode>,
    /// Buffered chunks, in push order.
    pub(crate) buffer: std::cell::RefCell<Vec<StreamChunk>>,
}
