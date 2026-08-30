//! Module-local tests for the hierarchical store.

use super::*;
use crate::store::Store as FlatStore;

fn ns(segments: &[&str]) -> Namespace {
    Namespace::new(segments.iter().copied()).expect("valid namespace")
}

#[test]
fn namespace_validation_rejects_the_unaddressable() {
    assert!(Namespace::new(Vec::<String>::new()).is_err(), "empty");
    assert!(Namespace::new([""]).is_err(), "empty segment");
    assert!(Namespace::new(["a.b"]).is_err(), "separator in segment");
    assert!(Namespace::new(["langgraph", "x"]).is_err(), "reserved root");
    assert!(Namespace::new(["users", "alice"]).is_ok());
}

#[test]
fn namespaces_know_their_own_prefixes_and_suffixes() {
    let n = ns(&["users", "alice", "memories"]);
    assert!(n.starts_with(&["users".into()]));
    assert!(n.starts_with(&["users".into(), "alice".into()]));
    assert!(!n.starts_with(&["users".into(), "bob".into()]));
    assert!(n.ends_with(&["memories".into()]));
    assert!(!n.ends_with(&["alice".into()]));
}

#[tokio::test]
async fn put_get_delete_roundtrip() {
    let store = InMemoryNamespacedStore::new();
    let n = ns(&["users", "alice"]);
    store
        .put(&n, "k", serde_json::json!({"v": 1}))
        .await
        .unwrap();
    let item = store.get(&n, "k").await.unwrap().expect("stored");
    assert_eq!(item.value, serde_json::json!({"v": 1}));
    assert_eq!(item.namespace, n);
    store.delete(&n, "k").await.unwrap();
    assert!(store.get(&n, "k").await.unwrap().is_none());
}

/// A subtree search is what the flat store could never express.
#[tokio::test]
async fn search_walks_a_namespace_subtree() {
    let store = InMemoryNamespacedStore::new();
    store
        .put(&ns(&["users", "alice"]), "a", serde_json::json!({"n": 1}))
        .await
        .unwrap();
    store
        .put(&ns(&["users", "bob"]), "b", serde_json::json!({"n": 2}))
        .await
        .unwrap();
    store
        .put(&ns(&["teams", "core"]), "c", serde_json::json!({"n": 3}))
        .await
        .unwrap();

    let found = store
        .search(SearchQuery {
            namespace_prefix: vec!["users".into()],
            ..SearchQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(found.len(), 2, "only the `users` subtree");
}

#[tokio::test]
async fn search_applies_comparison_filters() {
    let store = InMemoryNamespacedStore::new();
    let n = ns(&["items"]);
    for (key, score) in [("low", 1), ("mid", 5), ("high", 9)] {
        store
            .put(
                &n,
                key,
                serde_json::json!({"score": score, "meta": {"kind": "x"}}),
            )
            .await
            .unwrap();
    }

    let mut filter = std::collections::HashMap::new();
    filter.insert("score".to_string(), FilterOp::Gte(5.0));
    let found = store
        .search(SearchQuery {
            namespace_prefix: vec!["items".into()],
            filter,
            ..SearchQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(found.len(), 2);

    // Nested dotted paths resolve.
    let mut filter = std::collections::HashMap::new();
    filter.insert(
        "meta.kind".to_string(),
        FilterOp::Eq(serde_json::json!("x")),
    );
    assert_eq!(
        store
            .search(SearchQuery {
                namespace_prefix: vec!["items".into()],
                filter,
                ..SearchQuery::default()
            })
            .await
            .unwrap()
            .len(),
        3
    );

    // `$exists` distinguishes absent from null.
    let mut filter = std::collections::HashMap::new();
    filter.insert("missing".to_string(), FilterOp::Exists(false));
    assert_eq!(
        store
            .search(SearchQuery {
                namespace_prefix: vec!["items".into()],
                filter,
                ..SearchQuery::default()
            })
            .await
            .unwrap()
            .len(),
        3
    );
}

#[tokio::test]
async fn search_paginates_deterministically() {
    let store = InMemoryNamespacedStore::new();
    let n = ns(&["items"]);
    for i in 0..5 {
        store
            .put(&n, &format!("k{i}"), serde_json::json!(i))
            .await
            .unwrap();
    }
    let page = |offset| SearchQuery {
        namespace_prefix: vec!["items".into()],
        limit: Some(2),
        offset,
        ..SearchQuery::default()
    };
    let first = store.search(page(0)).await.unwrap();
    let second = store.search(page(2)).await.unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(second.len(), 2);
    assert!(
        first.iter().all(|a| second.iter().all(|b| a.key != b.key)),
        "pages must not overlap — which requires a deterministic order"
    );
}

#[tokio::test]
async fn list_namespaces_honours_wildcards_and_depth() {
    let store = InMemoryNamespacedStore::new();
    for path in [
        vec!["users", "alice", "memories"],
        vec!["users", "bob", "memories"],
        vec!["teams", "core", "notes"],
    ] {
        store
            .put(&ns(&path), "k", serde_json::json!(1))
            .await
            .unwrap();
    }

    let all = store
        .list_namespaces(ListNamespacesQuery::default())
        .await
        .unwrap();
    assert_eq!(all.len(), 3);

    // `*` matches exactly one segment.
    let wildcard = store
        .list_namespaces(ListNamespacesQuery {
            prefix: Some(vec!["users".into(), "*".into()]),
            ..ListNamespacesQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(wildcard.len(), 2);

    let by_suffix = store
        .list_namespaces(ListNamespacesQuery {
            suffix: Some(vec!["memories".into()]),
            ..ListNamespacesQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(by_suffix.len(), 2);

    // Truncating to depth 1 collapses the tree to its roots.
    let roots = store
        .list_namespaces(ListNamespacesQuery {
            max_depth: Some(1),
            ..ListNamespacesQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(roots.len(), 2, "`users` and `teams`: {roots:?}");
}

/// TTL is the bounded-growth answer the flat store never had. An expired item
/// must be invisible on read *before* any sweep runs.
#[tokio::test]
async fn expired_items_are_invisible_and_reclaimable() {
    let store = InMemoryNamespacedStore::new();
    let n = ns(&["cache"]);
    // A negative-in-effect lifetime: expire essentially immediately.
    store
        .put_with_ttl(&n, "k", serde_json::json!(1), Some(f64::MIN_POSITIVE))
        .await
        .unwrap();
    store
        .put_with_ttl(&n, "keep", serde_json::json!(2), None)
        .await
        .unwrap();

    assert!(
        store.get(&n, "k").await.unwrap().is_none(),
        "an expired item is invisible on read, not only after a sweep"
    );
    assert!(store.get(&n, "keep").await.unwrap().is_some());
    let found = store
        .search(SearchQuery {
            namespace_prefix: vec!["cache".into()],
            ..SearchQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(found.len(), 1, "search skips expired items too");
}

#[tokio::test]
async fn a_default_ttl_applies_to_writes_that_do_not_set_one() {
    let store = InMemoryNamespacedStore::new().with_ttl(TtlConfig {
        default_ttl_minutes: Some(60.0),
        refresh_on_read: false,
    });
    let n = ns(&["cache"]);
    store.put(&n, "k", serde_json::json!(1)).await.unwrap();
    let item = store.get(&n, "k").await.unwrap().unwrap();
    assert!(
        item.expires_at_ms.is_some(),
        "the store default applies when the write does not name a TTL"
    );
}

/// `batch` is the single abstract method, so a multi-op request must come back
/// positionally aligned — every convenience method depends on that.
#[tokio::test]
async fn batch_results_align_positionally_with_the_request() {
    let store = InMemoryNamespacedStore::new();
    let n = ns(&["b"]);
    let results = store
        .batch(&[
            StoreOp::Put {
                namespace: n.clone(),
                key: "k".into(),
                value: Some(serde_json::json!("v")),
                ttl_minutes: None,
            },
            StoreOp::Get {
                namespace: n.clone(),
                key: "k".into(),
                refresh_ttl: None,
            },
            StoreOp::Search(SearchQuery {
                namespace_prefix: vec!["b".into()],
                ..SearchQuery::default()
            }),
            StoreOp::ListNamespaces(ListNamespacesQuery::default()),
        ])
        .await
        .unwrap();
    assert_eq!(results.len(), 4);
    assert!(matches!(results[0], StoreResult::Ack));
    assert!(matches!(results[1], StoreResult::Item(Some(_))));
    assert!(matches!(&results[2], StoreResult::Items(v) if v.len() == 1));
    assert!(matches!(&results[3], StoreResult::Namespaces(v) if v.len() == 1));
}

/// The new trait must be additive: existing flat-`Store` callers keep working.
#[tokio::test]
async fn the_flat_store_surface_still_works() {
    let store = FlatNamespacedStore::new(InMemoryNamespacedStore::new());
    store
        .put("events", "e1", serde_json::json!({"a": 1}))
        .await
        .unwrap();
    assert_eq!(
        FlatStore::get(&store, "events", "e1").await.unwrap(),
        Some(serde_json::json!({"a": 1}))
    );
    assert_eq!(store.list("events").await.unwrap(), vec!["e1".to_string()]);
    FlatStore::delete(&store, "events", "e1").await.unwrap();
    assert!(
        FlatStore::get(&store, "events", "e1")
            .await
            .unwrap()
            .is_none()
    );
    assert!(store.list("events").await.unwrap().is_empty());
}
