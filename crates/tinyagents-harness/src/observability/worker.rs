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
//! Append errors are **not** silently discarded, but neither are they allowed to
//! flood the host's log. Every failed append is counted in
//! [`AppendWorker::append_failures`] (a lifetime total, distinct from the
//! queue-full [`AppendWorker::dropped`] count), and the drain loop reports
//! through `tracing` on the `tinyagents::observability` target with a `sink`
//! field:
//!
//! - the **first** failure of a failure run is reported at `ERROR` — durable
//!   observations are being lost;
//! - subsequent failures are counted silently, with a reminder at `WARN` at most
//!   once per [`APPEND_REPORT_COOLDOWN`]; each reminder carries the *latest*
//!   error, so a changed cause (read-only volume → full disk) still surfaces
//!   within one cooldown;
//! - the first success after a failure run emits one `WARN` recovery summary
//!   carrying how many observations were lost while the sink was down;
//! - if the worker shuts down while still failing, it emits one final `WARN`
//!   summary so a never-recovering run is not silently quiet.
//!
//! The worker keeps attempting every item while degraded: the attempt *is* the
//! recovery detector, it runs off the run's critical path, and skipping it would
//! turn a transient blip into guaranteed loss of everything still queued. Items
//! that fail are lost, but counted.
//!
//! Persistence remains best-effort — an error never propagates back into the run
//! — but it is observable. Note that reporting goes through `tracing`, so an
//! embedder with no subscriber installed sees nothing at all: installing one is
//! how a host observes durable-log loss. [`AppendWorker::append_failures`]
//! counts it, but — like the queue-full [`AppendWorker::dropped`] count it
//! mirrors — it is crate-internal and is not reachable from a host application
//! through [`JournalSink`](super::JournalSink) or
//! [`JsonlSink`](super::JsonlSink).
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
use std::time::{Duration, Instant};

use crate::error::Result;

/// Default bounded-queue capacity for a durable drain worker.
///
/// Sized so a transient backend stall buffers a healthy burst of events before
/// the drop policy engages, without letting the queue grow without bound.
pub const DEFAULT_DRAIN_CAPACITY: usize = 1024;

/// Minimum interval between repeated reports of an *ongoing* append-failure run.
///
/// A persistent sink failure (a volume flipped read-only, a full disk) fails
/// every queued observation identically; without a cooldown that is one log line
/// per arriving event. The first failure is always reported; after that the
/// drain loop stays quiet for this long between reminders.
pub(crate) const APPEND_REPORT_COOLDOWN: Duration = Duration::from_secs(300);

/// Whether an ongoing failure run is due for a reminder report.
///
/// `last` is when this run last reported (`None` before its first report).
/// Pure and clock-free so the cooldown policy is unit-testable without a
/// subscriber or a fake clock.
pub(super) fn should_report(last: Option<Instant>, now: Instant, cooldown: Duration) -> bool {
    match last {
        None => true,
        // Saturating: a non-monotonic `now` yields zero, not a panic.
        Some(previous) => now.saturating_duration_since(previous) >= cooldown,
    }
}

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
pub struct AppendWorker<T: Send + 'static> {
    /// Bounded submit channel. Wrapped in `Option` only so [`Drop`] can drop the
    /// sender before joining the drain thread; it is always `Some` otherwise.
    tx: Option<SyncSender<Msg<T>>>,
    /// Count of payloads dropped because the queue was full (or disconnected).
    dropped: Arc<AtomicU64>,
    /// Lifetime count of payloads whose durable append returned an error.
    append_failures: Arc<AtomicU64>,
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
    pub fn spawn<F, Fut>(name: &'static str, capacity: usize, append: F) -> Self
    where
        F: Fn(T) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>>,
    {
        Self::spawn_with_cooldown(name, capacity, APPEND_REPORT_COOLDOWN, append)
    }

    /// Spawns a drain worker with an explicit failure-report cooldown.
    ///
    /// Same as [`Self::spawn`], which supplies [`APPEND_REPORT_COOLDOWN`]. The
    /// cooldown is a parameter so the suppression and reminder paths can be
    /// exercised deterministically, without waiting on wall-clock time.
    pub fn spawn_with_cooldown<F, Fut>(
        name: &'static str,
        capacity: usize,
        cooldown: Duration,
        append: F,
    ) -> Self
    where
        F: Fn(T) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>>,
    {
        let (tx, rx) = sync_channel::<Msg<T>>(capacity.max(1));
        let append_failures = Arc::new(AtomicU64::new(0));
        let failures = Arc::clone(&append_failures);
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
                    // Failure-run state: how many appends have failed since the
                    // last success, and when this run last reported. Local to
                    // the drain thread — no shared state, no timer tasks.
                    let mut failure_run: u64 = 0;
                    let mut last_report: Option<Instant> = None;
                    while let Ok(msg) = rx.recv() {
                        match msg {
                            Msg::Item(item) => match append(item).await {
                                Ok(()) => {
                                    if failure_run > 0 {
                                        tinyagents_tracing::warn!(
                                            target: "tinyagents::observability",
                                            sink = name,
                                            lost = failure_run,
                                            "[observability] durable append recovered; {failure_run} observation(s) lost while the sink was failing"
                                        );
                                        failure_run = 0;
                                        last_report = None;
                                    }
                                }
                                Err(error) => {
                                    failures.fetch_add(1, Ordering::Relaxed);
                                    failure_run += 1;
                                    let now = Instant::now();
                                    if failure_run == 1 {
                                        last_report = Some(now);
                                        tinyagents_tracing::error!(
                                            target: "tinyagents::observability",
                                            sink = name,
                                            error = %error,
                                            "[observability] durable append failed; the observation is lost and repeats are suppressed until the sink recovers"
                                        );
                                    } else if should_report(last_report, now, cooldown) {
                                        last_report = Some(now);
                                        tinyagents_tracing::warn!(
                                            target: "tinyagents::observability",
                                            sink = name,
                                            error = %error,
                                            failures = failure_run,
                                            "[observability] durable append failed; still failing after {failure_run} consecutive observations"
                                        );
                                    }
                                }
                            },
                            Msg::Flush(ack) => {
                                let _ = ack.send(());
                            }
                        }
                    }
                    // The channel closed mid-failure: report once on the way out
                    // so a run that never recovered is not silently quiet.
                    if failure_run > 0 {
                        tinyagents_tracing::warn!(
                            target: "tinyagents::observability",
                            sink = name,
                            lost = failure_run,
                            "[observability] durable append failed; sink never recovered before shutdown, {failure_run} observation(s) lost"
                        );
                    }
                });
            })
            .expect("spawn durable-drain thread");
        Self {
            tx: Some(tx),
            dropped: Arc::new(AtomicU64::new(0)),
            append_failures,
            handle: Some(handle),
            name,
        }
    }

    /// Queues `item` for durable persistence without blocking.
    ///
    /// If the bounded queue is full the item is dropped and counted (see
    /// [`Self::dropped`]).
    pub fn submit(&self, item: T) {
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
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Returns the number of payloads whose durable append returned an error.
    ///
    /// Distinct from [`Self::dropped`]: those never reached the sink, these were
    /// attempted and rejected. Reporting is `tracing`-based and suppressed while
    /// a failure run continues, so this counter is the only subscriber-free
    /// signal of durable-log loss.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn append_failures(&self) -> u64 {
        self.append_failures.load(Ordering::Relaxed)
    }

    /// Blocks until every payload submitted before this call has been persisted.
    pub fn flush(&self) {
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
            .field(
                "append_failures",
                &self.append_failures.load(Ordering::Relaxed),
            )
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
