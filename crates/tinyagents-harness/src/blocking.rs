//! Shared helper for running blocking (synchronous file/DB) work off the
//! tokio runtime.
//!
//! Several backends — [`crate::store::FileStore`], the JSONL append store, and
//! [`crate::cache::SqliteResponseCache`] under the `sqlite` feature — perform
//! blocking I/O (`std::fs::*`, `rusqlite` calls) that must never run directly
//! inside an `async fn` body, since that stalls whichever tokio worker thread
//! happens to poll it. [`run_blocking`] offloads the work via
//! `tokio::task::spawn_blocking` when a runtime is present, and falls back to
//! running it inline when there is none (e.g. a synchronous caller outside any
//! runtime, such as some test harnesses).

use crate::error::{Result, TinyAgentsError};

/// Runs `work` off the async runtime via `spawn_blocking`, falling back to
/// running it inline when no tokio runtime is currently entered.
pub(crate) async fn run_blocking<F, T>(work: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle
            .spawn_blocking(work)
            .await
            .map_err(|e| TinyAgentsError::Validation(format!("blocking task error: {e}")))?,
        Err(_) => work(),
    }
}

#[cfg(test)]
mod test {
    use super::run_blocking;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// A slow, synchronous "store" body run through `run_blocking`. On a
    /// `current_thread` runtime, a call that blocks the worker thread inline
    /// (e.g. `std::thread::sleep` called directly in an `async fn`) would
    /// starve every other task, including a concurrent timer. Routing it
    /// through `run_blocking` must let the timer still fire while the slow
    /// work is in flight (I-4).
    #[tokio::test(flavor = "current_thread")]
    async fn run_blocking_does_not_stall_the_runtime() {
        let timer_fired = Arc::new(AtomicBool::new(false));
        let timer_fired_task = Arc::clone(&timer_fired);

        let timer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            timer_fired_task.store(true, Ordering::SeqCst);
        });

        let slow_work = run_blocking(move || -> crate::error::Result<()> {
            std::thread::sleep(Duration::from_millis(200));
            Ok(())
        });

        // The slow blocking work and the short timer run concurrently; if
        // blocking I/O were run inline on the current_thread runtime, the
        // timer would never fire before `slow_work` completes because the
        // single worker would be parked in `std::thread::sleep`.
        slow_work.await.unwrap();
        assert!(
            timer_fired.load(Ordering::SeqCst),
            "concurrent timer should have fired while the blocking work ran off-thread"
        );
        timer.await.unwrap();
    }
}
