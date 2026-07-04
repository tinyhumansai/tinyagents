//! Tests for the harness store backends.
//!
//! Cover [`super::InMemoryStore`] put/get/delete/list round-tripping and that
//! [`super::StoreRegistry`] provides a working default store plus named
//! registration and lookup.

use std::sync::Arc;

use super::*;
use serde_json::json;

#[tokio::test]
async fn in_memory_store_put_get_delete() {
    let store = InMemoryStore::new();

    // Key absent before first write.
    assert_eq!(store.get("ns", "k").await.unwrap(), None);

    // Write and read back.
    store.put("ns", "k", json!({"x": 1})).await.unwrap();
    assert_eq!(store.get("ns", "k").await.unwrap(), Some(json!({"x": 1})));

    // Delete removes the key.
    store.delete("ns", "k").await.unwrap();
    assert_eq!(store.get("ns", "k").await.unwrap(), None);
}

#[tokio::test]
async fn in_memory_store_list() {
    let store = InMemoryStore::new();
    store.put("ns", "a", json!(1)).await.unwrap();
    store.put("ns", "b", json!(2)).await.unwrap();
    let mut keys = store.list("ns").await.unwrap();
    keys.sort();
    assert_eq!(keys, vec!["a", "b"]);
}

#[tokio::test]
async fn store_registry_default_store_is_usable() {
    let reg = StoreRegistry::new();
    let store = reg.default_store();
    store.put("ns", "key", json!("value")).await.unwrap();
    assert_eq!(store.get("ns", "key").await.unwrap(), Some(json!("value")));
}

#[tokio::test]
async fn store_registry_named_store() {
    let mut reg = StoreRegistry::new();
    reg.register("cache", Arc::new(InMemoryStore::new()));
    let cache = reg.get("cache").expect("cache store registered");
    cache.put("ns", "k", json!(42)).await.unwrap();
    assert_eq!(cache.get("ns", "k").await.unwrap(), Some(json!(42)));
    assert!(reg.get("missing").is_none());
}

#[tokio::test]
async fn in_memory_delete_missing_is_noop() {
    let store = InMemoryStore::new();
    // Deleting a key (or whole namespace) that was never written is a no-op.
    store.delete("ns", "never").await.unwrap();
    // list on an unwritten namespace returns empty.
    assert!(store.list("empty-ns").await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_namespaces_are_isolated() {
    let store = InMemoryStore::new();
    store.put("a", "k", json!(1)).await.unwrap();
    store.put("b", "k", json!(2)).await.unwrap();
    assert_eq!(store.get("a", "k").await.unwrap(), Some(json!(1)));
    assert_eq!(store.get("b", "k").await.unwrap(), Some(json!(2)));
    assert_eq!(store.list("a").await.unwrap(), vec!["k"]);
}

// ── FileStore end-to-end ──────────────────────────────────────────────────────

/// A self-cleaning temp directory rooted under [`std::env::temp_dir`].
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("tinyagents-store-test-{tag}-{nanos}"));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn file_store_put_get_delete_list() {
    let dir = TempDir::new("crud");
    let store = FileStore::new(&dir.0);

    // Missing key returns None before any write.
    assert_eq!(store.get("threads", "t1").await.unwrap(), None);
    // Listing a namespace that has never been written returns empty.
    assert!(store.list("threads").await.unwrap().is_empty());

    // put + get round-trip.
    store
        .put("threads", "t1", json!({"role": "user"}))
        .await
        .unwrap();
    assert_eq!(
        store.get("threads", "t1").await.unwrap(),
        Some(json!({"role": "user"}))
    );

    // Multiple keys across namespaces.
    store.put("threads", "t2", json!(2)).await.unwrap();
    store.put("events", "e1", json!("ev")).await.unwrap();

    let mut threads = store.list("threads").await.unwrap();
    threads.sort();
    assert_eq!(threads, vec!["t1", "t2"]);
    assert_eq!(store.list("events").await.unwrap(), vec!["e1"]);

    // delete removes only the targeted key.
    store.delete("threads", "t1").await.unwrap();
    assert_eq!(store.get("threads", "t1").await.unwrap(), None);
    assert_eq!(store.list("threads").await.unwrap(), vec!["t2"]);

    // Deleting a missing key is a no-op (no error).
    store.delete("threads", "absent").await.unwrap();
}

#[tokio::test]
async fn file_store_rejects_unsafe_names() {
    let dir = TempDir::new("sanitize");
    let store = FileStore::new(&dir.0);

    // Path separators are rejected on every operation.
    assert!(store.get("../etc", "passwd").await.is_err());
    assert!(store.put("ns", "a/b", json!(1)).await.is_err());
    assert!(store.delete("ns", "a\\b").await.is_err());
    assert!(store.list("with space").await.is_err());

    // Empty names are rejected.
    assert!(store.put("", "k", json!(1)).await.is_err());
    assert!(store.get("ns", "").await.is_err());

    // All-dot names are rejected (path-traversal guard): a namespace is joined
    // onto the root without a suffix, so `".."` would escape the store root.
    assert!(store.put("..", "k", json!(1)).await.is_err());
    assert!(store.get(".", "k").await.is_err());
    assert!(store.list("...").await.is_err());

    // Allowed characters: alphanumerics, hyphen, underscore, dot.
    assert!(store.put("ns-1", "key_2.v3", json!(1)).await.is_ok());
    assert_eq!(store.get("ns-1", "key_2.v3").await.unwrap(), Some(json!(1)));
}

#[tokio::test]
async fn file_store_overwrites_existing_key() {
    let dir = TempDir::new("overwrite");
    let store = FileStore::new(&dir.0);
    store.put("ns", "k", json!("first")).await.unwrap();
    store.put("ns", "k", json!("second")).await.unwrap();
    assert_eq!(store.get("ns", "k").await.unwrap(), Some(json!("second")));
}

#[tokio::test]
async fn store_registry_default_store_is_stable() {
    let reg = StoreRegistry::new();
    // Two handles to the default store share the same backing data (Arc clone).
    reg.default_store().put("ns", "k", json!(1)).await.unwrap();
    assert_eq!(
        reg.default_store().get("ns", "k").await.unwrap(),
        Some(json!(1))
    );
}

#[tokio::test]
async fn store_registry_register_replaces() {
    let mut reg = StoreRegistry::new();
    reg.register("s", Arc::new(InMemoryStore::new()));
    reg.get("s")
        .unwrap()
        .put("ns", "k", json!(1))
        .await
        .unwrap();

    // Re-registering under the same name replaces the previous store.
    reg.register("s", Arc::new(InMemoryStore::new()));
    assert_eq!(reg.get("s").unwrap().get("ns", "k").await.unwrap(), None);
}

// ── InMemoryAppendStore ────────────────────────────────────────────────────────

#[tokio::test]
async fn in_memory_append_returns_increasing_offsets() {
    let store = InMemoryAppendStore::new();
    assert_eq!(store.len("evts").await.unwrap(), 0);

    assert_eq!(store.append("evts", json!({"n": 0})).await.unwrap(), 0);
    assert_eq!(store.append("evts", json!({"n": 1})).await.unwrap(), 1);
    assert_eq!(store.append("evts", json!({"n": 2})).await.unwrap(), 2);

    // len equals the offset the next append will receive.
    assert_eq!(store.len("evts").await.unwrap(), 3);
}

#[tokio::test]
async fn in_memory_read_from_returns_tail() {
    let store = InMemoryAppendStore::new();
    for n in 0..5 {
        store.append("s", json!(n)).await.unwrap();
    }

    // Whole stream from offset 0.
    let all = store.read_from("s", 0).await.unwrap();
    assert_eq!(all.len(), 5);
    assert_eq!(all[0], (0, json!(0)));
    assert_eq!(all[4], (4, json!(4)));

    // Tail from a mid offset.
    let tail = store.read_from("s", 3).await.unwrap();
    assert_eq!(tail, vec![(3, json!(3)), (4, json!(4))]);

    // Reading from len (or beyond) yields nothing.
    assert!(store.read_from("s", 5).await.unwrap().is_empty());
    assert!(store.read_from("s", 99).await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_append_streams_are_isolated() {
    let store = InMemoryAppendStore::new();
    store.append("a", json!("a0")).await.unwrap();
    store.append("b", json!("b0")).await.unwrap();
    store.append("a", json!("a1")).await.unwrap();

    assert_eq!(store.len("a").await.unwrap(), 2);
    assert_eq!(store.len("b").await.unwrap(), 1);
    // An unwritten stream reads back empty without error.
    assert!(store.read_from("missing", 0).await.unwrap().is_empty());
    assert_eq!(store.len("missing").await.unwrap(), 0);
}

#[tokio::test]
async fn in_memory_append_bounded_evicts_oldest_and_keeps_offsets_monotonic() {
    let store = InMemoryAppendStore::new().with_max_entries_per_stream(3);
    for i in 0..5u64 {
        // Offsets keep advancing past the cap.
        assert_eq!(store.append("s", json!(i)).await.unwrap(), i);
    }

    // Logical length is unaffected by eviction: the next append gets offset 5.
    assert_eq!(store.len("s").await.unwrap(), 5);

    // Only the newest 3 entries remain, at their original offsets.
    let all = store.read_from("s", 0).await.unwrap();
    assert_eq!(
        all,
        vec![(2, json!(2)), (3, json!(3)), (4, json!(4))],
        "evicted offsets are no longer readable"
    );

    // Reading from a retained offset returns exactly the tail.
    assert_eq!(
        store.read_from("s", 3).await.unwrap(),
        vec![(3, json!(3)), (4, json!(4))]
    );
    // Reading past the end is empty, not an error.
    assert!(store.read_from("s", 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_append_bound_applies_per_stream() {
    let store = InMemoryAppendStore::new().with_max_entries_per_stream(2);
    for i in 0..3u64 {
        store.append("a", json!(i)).await.unwrap();
    }
    store.append("b", json!("b0")).await.unwrap();

    // Stream `a` was trimmed; stream `b` is untouched.
    assert_eq!(store.read_from("a", 0).await.unwrap().len(), 2);
    assert_eq!(
        store.read_from("b", 0).await.unwrap(),
        vec![(0, json!("b0"))]
    );
}

#[tokio::test]
async fn in_memory_append_unbounded_by_default() {
    let store = InMemoryAppendStore::new();
    for i in 0..100u64 {
        store.append("s", json!(i)).await.unwrap();
    }
    assert_eq!(store.read_from("s", 0).await.unwrap().len(), 100);
    assert_eq!(store.len("s").await.unwrap(), 100);
}

// ── JsonlAppendStore ───────────────────────────────────────────────────────────

#[tokio::test]
async fn jsonl_append_returns_increasing_offsets_and_reads_tail() {
    let dir = TempDir::new("jsonl-basic");
    let store = JsonlAppendStore::new(&dir.0);

    assert_eq!(store.append("evts", json!({"n": 0})).await.unwrap(), 0);
    assert_eq!(store.append("evts", json!({"n": 1})).await.unwrap(), 1);
    assert_eq!(store.append("evts", json!({"n": 2})).await.unwrap(), 2);
    assert_eq!(store.len("evts").await.unwrap(), 3);

    let tail = store.read_from("evts", 1).await.unwrap();
    assert_eq!(tail, vec![(1, json!({"n": 1})), (2, json!({"n": 2}))]);

    // A stream that was never written reads back empty.
    assert!(store.read_from("other", 0).await.unwrap().is_empty());
    assert_eq!(store.len("other").await.unwrap(), 0);
}

#[tokio::test]
async fn jsonl_round_trips_across_two_store_instances() {
    let dir = TempDir::new("jsonl-reopen");

    {
        let first = JsonlAppendStore::new(&dir.0);
        first
            .append("runs", json!({"kind": "started"}))
            .await
            .unwrap();
        first.append("runs", json!({"kind": "step"})).await.unwrap();
    }

    // A fresh store on the same directory sees prior entries and continues the
    // offset sequence.
    let second = JsonlAppendStore::new(&dir.0);
    assert_eq!(second.len("runs").await.unwrap(), 2);
    assert_eq!(
        second
            .append("runs", json!({"kind": "finished"}))
            .await
            .unwrap(),
        2
    );

    let all = second.read_from("runs", 0).await.unwrap();
    assert_eq!(
        all,
        vec![
            (0, json!({"kind": "started"})),
            (1, json!({"kind": "step"})),
            (2, json!({"kind": "finished"})),
        ]
    );
}

#[tokio::test]
async fn jsonl_append_offsets_stay_correct_under_many_appends() {
    // Exercise the cached-offset path: after the first append learns the length
    // from disk, later appends must keep incrementing without re-reading, and
    // every record must be readable back in order.
    let dir = TempDir::new("jsonl-many");
    let store = JsonlAppendStore::new(&dir.0);

    for n in 0..50u64 {
        let offset = store.append("evts", json!({ "n": n })).await.unwrap();
        assert_eq!(offset, n, "append offsets must be dense and monotonic");
    }
    assert_eq!(store.len("evts").await.unwrap(), 50);

    let all = store.read_from("evts", 0).await.unwrap();
    assert_eq!(all.len(), 50);
    for (i, (offset, value)) in all.iter().enumerate() {
        assert_eq!(*offset, i as u64);
        assert_eq!(value, &json!({ "n": i as u64 }));
    }
}

#[tokio::test]
async fn file_store_put_is_atomic_and_leaves_no_temp_files() {
    let dir = TempDir::new("atomic");
    let store = FileStore::new(&dir.0);

    // Overwrite the same key repeatedly; the final read must be a complete,
    // well-formed value and no scratch (`.tmp`) files may be left behind.
    for n in 0..5 {
        store
            .put(
                "ns",
                "key",
                json!({ "value": n, "payload": "x".repeat(1024) }),
            )
            .await
            .unwrap();
    }

    let got = store.get("ns", "key").await.unwrap().unwrap();
    assert_eq!(got["value"], json!(4));
    assert_eq!(store.list("ns").await.unwrap(), vec!["key".to_string()]);

    let leftover: Vec<_> = std::fs::read_dir(dir.0.join("ns"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp"))
        .collect();
    assert!(
        leftover.is_empty(),
        "atomic put must not leave temp files: {leftover:?}"
    );
}

#[tokio::test]
async fn jsonl_rejects_unsafe_stream_names() {
    let dir = TempDir::new("jsonl-sanitize");
    let store = JsonlAppendStore::new(&dir.0);

    assert!(store.append("../etc", json!(1)).await.is_err());
    assert!(store.append("a/b", json!(1)).await.is_err());
    assert!(store.append("", json!(1)).await.is_err());
    assert!(store.append("..", json!(1)).await.is_err());
    assert!(store.read_from("with space", 0).await.is_err());
    assert!(store.len("a\\b").await.is_err());

    // Allowed characters round-trip.
    assert_eq!(store.append("runs-1.events_v2", json!(1)).await.unwrap(), 0);
}
