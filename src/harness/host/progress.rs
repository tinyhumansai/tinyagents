//! Back-pressured delivery of run events to an out-of-process consumer.
//!
//! See [`ProgressSink`] for the trait contract and [`NoopProgressSink`] for the
//! inert default.

use async_trait::async_trait;

use crate::error::Result;
use crate::harness::events::EventRecord;

/// Asynchronous, back-pressured delivery of run events to an out-of-process
/// consumer.
///
/// Complements [`EventListener`][crate::harness::events::EventListener], which
/// is synchronous and must not block the emitting step. A `ProgressSink` may
/// await — it exists for consumers behind a bounded channel, a socket, or an
/// IPC boundary, where dropping events silently is not acceptable. It carries
/// the crate's own [`EventRecord`] vocabulary; projecting that into a
/// presentation model is the embedder's job and stays outside the crate.
///
/// This is a *delivery* seam, not a transport. It has no receive side: an
/// interactive loop that reads input from a terminal or a chat platform is host
/// surface and does not belong behind this trait.
#[async_trait]
pub trait ProgressSink<State: Send + Sync>: Send + Sync {
    /// Whether a consumer is attached.
    ///
    /// Callers may use this to skip assembling expensive payloads (full tool
    /// output, model input/output) when nothing will read them. A sink that
    /// returns `false` must still tolerate [`Self::deliver`].
    fn is_connected(&self, state: &State) -> bool {
        let _ = state;
        true
    }

    /// Delivers one record in offset order. `record` is borrowed; clone it to
    /// retain it past the call. An `Err` means the consumer is gone; the
    /// runtime logs it and continues rather than failing the run.
    async fn deliver(&self, state: &State, record: &EventRecord) -> Result<()>;
}

/// A [`ProgressSink`] with no consumer behind it.
///
/// [`is_connected`][ProgressSink::is_connected] deliberately overrides the
/// trait default and returns `false`, so callers take their "nobody is
/// watching" fast path and skip assembling payloads nothing will read.
/// [`deliver`][ProgressSink::deliver] still succeeds, per the contract that a
/// disconnected sink must tolerate delivery.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopProgressSink;

impl NoopProgressSink {
    /// Creates the sink.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State: Send + Sync> ProgressSink<State> for NoopProgressSink {
    fn is_connected(&self, _state: &State) -> bool {
        false
    }

    async fn deliver(&self, _state: &State, _record: &EventRecord) -> Result<()> {
        Ok(())
    }
}
