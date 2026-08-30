//! Per-thread async lock maps that do not leak.
//!
//! The goals and todos stores serialize their `load → mutate → put` cycles
//! with one async mutex per thread id. A naive `HashMap<String, Arc<Mutex<()>>>`
//! grows forever in a long-lived process: every thread id ever touched keeps
//! one `Arc<Mutex<()>>` (and its key) alive until process exit.
//!
//! [`ThreadLockMap`] fixes the leak with a **weak-value map**: the map holds
//! [`Weak`] references, so a thread's mutex is deallocated as soon as the last
//! caller drops its `Arc`. Dead `Weak` entries (and their keys) are reclaimed
//! by an opportunistic amortized sweep on insertion, so the map's size stays
//! proportional to the number of *recently active* threads rather than every
//! thread id ever seen.
//!
//! # Correctness under concurrency
//! Handing out locks goes through one internal `std::sync::Mutex`, so lookup
//! and insertion are atomic: while any task holds (or awaits) a thread's
//! mutex, its `Arc` is alive, `Weak::upgrade` succeeds, and every concurrent
//! caller for the same thread id receives **the same mutex**. Only once every
//! clone has been dropped can a later caller mint a fresh mutex — at which
//! point no one holds the old one, so mutual exclusion is preserved.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex, Weak};

use tokio::sync::Mutex;

/// Smallest map size at which an insertion triggers a sweep of dead entries.
const SWEEP_MIN: usize = 16;

/// A `thread_id → async mutex` map holding weak references, so unused mutexes
/// (and eventually their map entries) are reclaimed instead of leaking.
///
/// Callers get an owned `Arc<tokio::sync::Mutex<()>>` from
/// [`lock_for`](Self::lock_for) and hold it for the duration of their critical
/// section; the map itself never keeps a mutex alive.
#[derive(Debug)]
pub(crate) struct ThreadLockMap {
    /// Human-readable owner name used in poisoned-lock panic messages.
    what: &'static str,
    /// The weak-value map plus its sweep bookkeeping.
    inner: StdMutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    /// `thread_id → weak handle` to that thread's mutex.
    locks: HashMap<String, Weak<Mutex<()>>>,
    /// Map size at which the next insertion sweeps dead entries. Doubling it
    /// after each sweep keeps the amortized sweep cost O(1) per insertion.
    sweep_at: usize,
}

impl ThreadLockMap {
    /// Creates an empty lock map. `what` names the owner in panic messages
    /// (e.g. `"goal lock map"`).
    pub(crate) fn new(what: &'static str) -> Self {
        Self {
            what,
            inner: StdMutex::new(Inner {
                locks: HashMap::new(),
                sweep_at: SWEEP_MIN,
            }),
        }
    }

    /// Returns the dedicated async mutex for `thread_id`, creating it when the
    /// thread has no live mutex.
    ///
    /// All concurrent callers passing the same `thread_id` receive the same
    /// mutex for as long as at least one of them keeps its `Arc` alive.
    pub(crate) fn lock_for(&self, thread_id: &str) -> Arc<Mutex<()>> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|_| panic!("{} poisoned", self.what));
        if let Some(existing) = inner.locks.get(thread_id).and_then(Weak::upgrade) {
            return existing;
        }
        let lock = Arc::new(Mutex::new(()));
        inner
            .locks
            .insert(thread_id.to_string(), Arc::downgrade(&lock));
        if inner.locks.len() >= inner.sweep_at {
            inner.locks.retain(|_, weak| weak.strong_count() > 0);
            inner.sweep_at = (inner.locks.len() * 2).max(SWEEP_MIN);
        }
        lock
    }

    /// Returns the number of map entries, live or dead (test instrumentation).
    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|_| panic!("{} poisoned", self.what))
            .locks
            .len()
    }
}

#[cfg(test)]
mod test;
