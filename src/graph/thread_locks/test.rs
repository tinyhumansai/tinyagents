//! Unit tests for [`ThreadLockMap`]: identity while held, reclamation of
//! dropped locks, and mutual exclusion under concurrent access.

use std::sync::Arc;

use super::{SWEEP_MIN, ThreadLockMap};

#[test]
fn same_thread_id_returns_same_mutex_while_held() {
    let map = ThreadLockMap::new("test lock map");
    let a = map.lock_for("t-1");
    let b = map.lock_for("t-1");
    assert!(Arc::ptr_eq(&a, &b), "live lock must be shared");

    let other = map.lock_for("t-2");
    assert!(
        !Arc::ptr_eq(&a, &other),
        "distinct threads get distinct locks"
    );
}

#[test]
fn dropped_locks_are_swept_from_the_map() {
    let map = ThreadLockMap::new("test lock map");
    // Insert (and immediately drop) far more locks than the sweep threshold.
    for i in 0..(SWEEP_MIN * 8) {
        let _ = map.lock_for(&format!("t-{i}"));
    }
    // Every insertion after the map reached the threshold swept dead entries,
    // so the map never accumulates all `SWEEP_MIN * 8` dead handles.
    assert!(
        map.entry_count() <= SWEEP_MIN,
        "dead entries must be reclaimed, got {}",
        map.entry_count()
    );
}

#[test]
fn held_locks_survive_the_sweep() {
    let map = ThreadLockMap::new("test lock map");
    let held = map.lock_for("t-held");
    for i in 0..(SWEEP_MIN * 8) {
        let _ = map.lock_for(&format!("t-{i}"));
    }
    // The held lock is still the one the map hands out.
    assert!(Arc::ptr_eq(&held, &map.lock_for("t-held")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_callers_are_mutually_exclusive() {
    let map = Arc::new(ThreadLockMap::new("test lock map"));
    // A deliberately non-atomic critical section: read, yield, write.
    let counter = Arc::new(std::sync::Mutex::new(0u64));

    let mut handles = Vec::new();
    for _ in 0..32 {
        let map = Arc::clone(&map);
        let counter = Arc::clone(&counter);
        handles.push(tokio::spawn(async move {
            for _ in 0..25 {
                let lock = map.lock_for("t-shared");
                let _guard = lock.lock().await;
                let read = *counter.lock().unwrap();
                tokio::task::yield_now().await;
                *counter.lock().unwrap() = read + 1;
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }

    // Without per-thread mutual exclusion the read-yield-write races and the
    // final count comes up short.
    assert_eq!(*counter.lock().unwrap(), 32 * 25);
}
