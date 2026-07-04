//! Background drain worker for durable event sinks.
//!
//! Durable sinks implement the *synchronous* `EventListener`/`GraphEventSink`
//! hooks but persist through *async* journal/store APIs. Bridging that boundary
//! inline with `futures::executor::block_on` blocked a tokio worker thread for
//! the whole append (a file write, in the worst case) on the run's critical
//! path, and risked a deadlock on a current-thread runtime.
//!
//! [`AppendWorker`] moves persistence off the emitting thread entirely: each
//! `submit` pushes the payload onto a **bounded** channel that a dedicated
//! background thread drains, awaiting the append on its own single-thread
//! runtime.
//!
//! # Backpressure & drop policy
//! The channel is bounded (see [`DEFAULT_DRAIN_CAPACITY`]). `submit` never
//! blocks the emitting thread: if the queue is full the observation is
//! **dropped** and counted in [`AppendWorker::dropped`], trading completeness
//! of the durable log for run latency (a slow or stuck backend must never stall
//! the run). Callers that need a lossless log can inspect the dropped count.
//!
//! # Error policy
//! Append errors are **not** silently discarded: the drain loop reports each
//! failure to stderr with the sink name. Persistence remains best-effort — an
//! error never propagates back into the run — but it is observable.
//!
//! # Ordering & durability boundary
//! A single drain thread preserves submit order. [`AppendWorker::flush`] blocks
//! until every payload queued so far has been persisted, and `Drop` flushes
//! before tearing the thread down, so no buffered observation is lost on a
//! graceful shutdown.

use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;

use crate::error::Result;

/// Default bounded-queue capacity for a durable drain worker.
///
/// Sized so a transient backend stall buffers a healthy burst of events before
/// the drop policy engages, without letting the queue grow without bound.
pub(crate) const DEFAULT_DRAIN_CAPACITY: usize = 1024;

/// Messages carried over the drain channel.
enum Msg<T> {
    /// A payload to persist.
    Item(T),
    /// A flush barrier: the drain loop acks once it reaches this marker, which
    /// (by FIFO ordering) means every earlier `Item` has been persisted.
    Flush(Sender<()>),
}

/// A background worker that drains submitted payloads into an async append sink.
///
/// See the [module docs](self) for the backpressure, error, and durability
/// semantics.
pub(crate) struct AppendWorker<T: Send + 'static> {
    /// Bounded submit channel. Wrapped in `Option` only so [`Drop`] can drop the
    /// sender before joining the drain thread; it is always `Some` otherwise.
    tx: Option<SyncSender<Msg<T>>>,
    /// Count of payloads dropped because the queue was full (or disconnected).
    dropped: Arc<AtomicU64>,
    /// Handle to the drain thread, joined on drop.
    handle: Option<JoinHandle<()>>,
    /// Human-readable sink name used in error reports.
    name: &'static str,
}

impl<T: Send + 'static> AppendWorker<T> {
    /// Spawns a drain worker.
    ///
    /// `append` is invoked once per submitted payload on the drain thread's
    /// runtime; it returns the async append future. `name` labels the drain
    /// thread and error reports.
    pub(crate) fn spawn<F, Fut>(name: &'static str, capacity: usize, append: F) -> Self
    where
        F: Fn(T) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>>,
    {
        let (tx, rx) = sync_channel::<Msg<T>>(capacity.max(1));
        let handle = std::thread::Builder::new()
            .name(format!("tinyagents-{name}-drain"))
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    // Without a runtime we cannot await appends; drain and drop.
                    Err(_) => {
                        while rx.recv().is_ok() {}
                        return;
                    }
                };
                rt.block_on(async move {
                    while let Ok(msg) = rx.recv() {
                        match msg {
                            Msg::Item(item) => {
                                if let Err(e) = append(item).await {
                                    eprintln!("tinyagents: {name} durable append failed: {e}");
                                }
                            }
                            Msg::Flush(ack) => {
                                let _ = ack.send(());
                            }
                        }
                    }
                });
            })
            .expect("spawn durable-drain thread");
        Self {
            tx: Some(tx),
            dropped: Arc::new(AtomicU64::new(0)),
            handle: Some(handle),
            name,
        }
    }

    /// Queues `item` for durable persistence without blocking.
    ///
    /// If the bounded queue is full the item is dropped and counted (see
    /// [`Self::dropped`]).
    pub(crate) fn submit(&self, item: T) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(Msg::Item(item)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns the number of payloads dropped because the queue was full.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Blocks until every payload submitted before this call has been persisted.
    pub(crate) fn flush(&self) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        // Blocking send (not `try_send`) so a momentarily full queue does not
        // drop the flush barrier; the drain thread is actively consuming.
        if tx.send(Msg::Flush(ack_tx)).is_ok() {
            let _ = ack_rx.recv();
        }
    }
}

impl<T: Send + 'static> fmt::Debug for AppendWorker<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AppendWorker")
            .field("name", &self.name)
            .field("dropped", &self.dropped.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Drop for AppendWorker<T> {
    fn drop(&mut self) {
        // Persist anything still queued, then drop the sender so the drain loop
        // observes a closed channel and exits, then join the thread.
        self.flush();
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
