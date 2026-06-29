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
