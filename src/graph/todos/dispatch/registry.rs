//! Registry of in-flight autonomous runs, keyed by the session they stream into.
//!
//! Autonomous card runs are detached tasks, not turns a chat channel knows
//! about, so a "stop" arriving through the normal path has nothing to cancel.
//! Registering each run's [`AbortHandle`](tokio::task::AbortHandle) here gives
//! that path a handle to pull.
//!
//! The registry's real job is **deciding who cleans up**. A run that finishes
//! naturally and a cancel that arrives at the same moment both try to write the
//! card's terminal state. Both must go through [`ActiveRunRegistry::take`] (or
//! [`take_if`](ActiveRunRegistry::take_if)), and only one of them gets `Some` —
//! so the write-back happens exactly once.

use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::watch;
use tokio::task::AbortHandle;

/// A live run's cancellation handles plus whatever context its host needs to
/// finish the card off.
///
/// `Context` is the host's own payload — typically the board coordinates and
/// anything else its write-back needs. The crate never inspects it.
#[derive(Debug)]
pub struct ActiveRun<Context> {
    /// The run this entry belongs to.
    pub run_id: String,
    /// The card it is executing.
    pub card_id: String,
    /// Aborts the detached run task.
    pub abort: AbortHandle,
    /// Stops the run's background heartbeat.
    pub heartbeat_cancel: watch::Sender<bool>,
    /// Host-supplied context, returned with the entry when it is taken.
    pub context: Context,
}

impl<Context> ActiveRun<Context> {
    /// Abort the run task and stop its heartbeat.
    ///
    /// Does **not** write the card back — that is the caller's job, because
    /// only the caller knows what terminal state the card should land in.
    pub fn cancel(&self) {
        self.abort.abort();
        let _ = self.heartbeat_cancel.send(true);
    }
}

/// A `session_thread_id → ActiveRun` map with race-free removal.
#[derive(Debug, Default)]
pub struct ActiveRunRegistry<Context> {
    runs: Mutex<HashMap<String, ActiveRun<Context>>>,
}

impl<Context> ActiveRunRegistry<Context> {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            runs: Mutex::new(HashMap::new()),
        }
    }

    /// Record `run` as the live run on `thread_id`, returning any entry it
    /// displaced (which the caller should cancel).
    pub fn register(&self, thread_id: impl Into<String>, run: ActiveRun<Context>) -> Option<ActiveRun<Context>> {
        self.lock().insert(thread_id.into(), run)
    }

    /// Remove and return the run on `thread_id`, if any.
    ///
    /// Whoever gets `Some` owns the terminal card write-back.
    pub fn take(&self, thread_id: &str) -> Option<ActiveRun<Context>> {
        self.lock().remove(thread_id)
    }

    /// Remove and return the run on `thread_id`, but only when it is the run
    /// `run_id` names. `None` for `run_id` removes whatever is there.
    ///
    /// The match and the removal happen under one lock acquisition. A
    /// peek-then-take would leave a window in which the matched run finishes
    /// and a *newer* run takes its place before removal — cancelling the new
    /// run instead of the intended one.
    pub fn take_if(&self, thread_id: &str, run_id: Option<&str>) -> Option<ActiveRun<Context>> {
        let mut runs = self.lock();
        if let Some(run_id) = run_id {
            match runs.get(thread_id) {
                None => {
                    tracing::debug!(
                        thread_id = %thread_id,
                        request_run_id = %run_id,
                        "[graph:todos:dispatch] scoped cancel ignored: no active run on thread"
                    );
                    return None;
                }
                Some(active) if active.run_id != run_id => {
                    tracing::debug!(
                        thread_id = %thread_id,
                        request_run_id = %run_id,
                        active_run_id = %active.run_id,
                        "[graph:todos:dispatch] scoped cancel ignored: run id mismatch"
                    );
                    return None;
                }
                _ => {}
            }
        }
        runs.remove(thread_id)
    }

    /// Whether a run is registered for `thread_id`.
    pub fn contains(&self, thread_id: &str) -> bool {
        self.lock().contains_key(thread_id)
    }

    /// Number of live runs.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no runs are live.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The session thread ids with a live run, in unspecified order.
    pub fn thread_ids(&self) -> Vec<String> {
        self.lock().keys().cloned().collect()
    }

    /// Remove every entry, returning them so the caller can cancel each.
    pub fn drain(&self) -> Vec<(String, ActiveRun<Context>)> {
        self.lock().drain().collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, ActiveRun<Context>>> {
        // A panic inside the map would leave it poisoned; the map holds no
        // invariant that a panic could break, so recovering is safe and keeps
        // one bad run from wedging cancellation for every other thread.
        self.runs.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
