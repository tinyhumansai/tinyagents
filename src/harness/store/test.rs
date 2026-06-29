//! Tests added in a later pass.
//!
//! Smoke test confirming that [`super::InMemoryStore`] round-trips a single
//! key and that [`super::StoreRegistry`] provides a working default store.

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
