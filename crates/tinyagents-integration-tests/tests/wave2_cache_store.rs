//! Wave 2 — [`ResponseCache`] storage capabilities.
//!
//! Covers C-TTL (expiry, `clear`, namespacing), C-BYTES (a byte bound as well
//! as an entry count), C-STATS (counters), C-SINGLEFLIGHT (stampede
//! protection), C-SQLITE-CACHE (durability) and CACHE-9 (LRU recency without a
//! linear scan).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tinyagents_harness::cache::{CachePolicy, InMemoryResponseCache, ResponseCache, SingleFlight};
use tinyinference::model::ModelResponse;

// ── C-TTL ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn an_expired_entry_is_a_miss_and_is_dropped() {
    // Without a TTL, a poisoned entry is permanent in a cache that is never
    // cleared — which is what made CACHE-2's cross-model poisoning forever.
    let cache = InMemoryResponseCache::new();
    cache
        .put_with_ttl(
            "k",
            ModelResponse::assistant("stale"),
            Some(Duration::from_millis(20)),
        )
        .await
        .unwrap();
    assert!(
        cache.get("k").await.unwrap().is_some(),
        "live before expiry"
    );

    tokio::time::sleep(Duration::from_millis(40)).await;
    assert!(
        cache.get("k").await.unwrap().is_none(),
        "an expired entry must read as a miss"
    );
    assert_eq!(
        cache.stats().entries,
        0,
        "the expired entry must be dropped on the way past, not merely hidden"
    );
    assert_eq!(cache.stats().expirations, 1);
}

#[tokio::test]
async fn put_without_a_ttl_never_expires() {
    let cache = InMemoryResponseCache::new();
    cache.put("k", ModelResponse::assistant("v")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(cache.get("k").await.unwrap().is_some());
}

#[tokio::test]
async fn clear_drops_every_entry() {
    let cache = InMemoryResponseCache::new();
    cache.put("a", ModelResponse::assistant("a")).await.unwrap();
    cache.put("b", ModelResponse::assistant("b")).await.unwrap();
    cache.clear().await.unwrap();
    assert!(cache.get("a").await.unwrap().is_none());
    assert!(cache.get("b").await.unwrap().is_none());
    assert_eq!(cache.stats().entries, 0);
}

#[test]
fn cache_policy_carries_ttl_and_namespace() {
    let policy = CachePolicy::enabled()
        .with_ttl(Duration::from_secs(90))
        .with_namespace("tenant-7");
    assert!(policy.response_cache_enabled);
    assert_eq!(policy.ttl(), Some(Duration::from_secs(90)));
    assert_eq!(policy.namespace.as_deref(), Some("tenant-7"));
    // Round-trips, so a policy can be persisted with a saved agent config.
    let json = serde_json::to_string(&policy).unwrap();
    let back: CachePolicy = serde_json::from_str(&json).unwrap();
    assert_eq!(back, policy);
    // And a policy serialized before these fields existed still deserializes.
    let legacy: CachePolicy =
        serde_json::from_str(r#"{"response_cache_enabled":true,"protect_prompt_prefix":false}"#)
            .unwrap();
    assert!(legacy.ttl().is_none());
}

// ── C-BYTES ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_cache_is_bounded_by_bytes_as_well_as_entries() {
    // 1024 long-context responses carrying large tool payloads is hundreds of
    // megabytes; an entry count alone does not bound memory.
    let cache = InMemoryResponseCache::with_bounds(1024, 4_000);
    let big = || ModelResponse::assistant("x".repeat(1_500));

    for i in 0..10 {
        cache.put(&format!("k{i}"), big()).await.unwrap();
    }
    let stats = cache.stats();
    assert!(
        stats.bytes <= 4_000,
        "the byte budget must bound the cache: {} bytes retained",
        stats.bytes
    );
    assert!(
        stats.entries < 10,
        "entries must have been evicted to stay under the byte budget"
    );
    assert!(stats.evictions > 0);
    assert!(
        cache.get("k9").await.unwrap().is_some(),
        "the most recent write always survives"
    );
}

// ── C-STATS ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn stats_report_hits_misses_and_writes() {
    let cache = InMemoryResponseCache::new();
    assert!(cache.get("nope").await.unwrap().is_none());
    cache.put("k", ModelResponse::assistant("v")).await.unwrap();
    assert!(cache.get("k").await.unwrap().is_some());
    assert!(cache.get("k").await.unwrap().is_some());

    let stats = cache.stats();
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.writes, 1);
    assert_eq!(stats.entries, 1);
    assert!(stats.bytes > 0);
}

// ── CACHE-9 ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn recency_tracking_stays_correct_at_scale() {
    // The recency index moved from a linear `VecDeque` scan (up to `capacity`
    // string comparisons plus a memmove on every hit) to an ordered map. This
    // asserts the *behaviour* the rewrite had to preserve.
    let cache = InMemoryResponseCache::with_capacity(64);
    for i in 0..64 {
        cache
            .put(&format!("k{i}"), ModelResponse::assistant("v"))
            .await
            .unwrap();
    }
    // Keep the oldest key hot.
    for _ in 0..5 {
        assert!(cache.get("k0").await.unwrap().is_some());
    }
    // Overflow by one: the victim must be `k1`, not the freshly-touched `k0`.
    cache
        .put("overflow", ModelResponse::assistant("v"))
        .await
        .unwrap();
    assert!(cache.get("k0").await.unwrap().is_some(), "k0 was hot");
    assert!(cache.get("k1").await.unwrap().is_none(), "k1 was the LRU");
    assert_eq!(cache.stats().entries, 64);
}

// ── C-SINGLEFLIGHT ───────────────────────────────────────────────────────────

#[tokio::test]
async fn concurrent_identical_calls_collapse_into_one() {
    // N concurrent identical requests all missed and all called the provider;
    // N-1 of those calls were paid for and thrown away.
    let flight = SingleFlight::new();
    let calls = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let flight = flight.clone();
        let calls = calls.clone();
        handles.push(tokio::spawn(async move {
            flight
                .run("same-key", || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok(ModelResponse::assistant("one answer"))
                })
                .await
        }));
    }

    let mut followers = 0;
    for handle in handles {
        let (response, was_follower) = handle.await.unwrap().expect("call succeeds");
        assert_eq!(response.text(), "one answer");
        if was_follower {
            followers += 1;
        }
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "eight identical concurrent calls must reach the provider once"
    );
    assert_eq!(
        followers, 7,
        "seven callers rode along on the leader's call"
    );
    assert_eq!(flight.inflight_len(), 0, "the key is retired when done");
}

#[tokio::test]
async fn a_follower_runs_its_own_call_when_the_leader_fails() {
    // An error is not a value worth sharing: one caller's transient 503 must
    // not become every concurrent caller's failure.
    let flight = SingleFlight::new();
    let calls = Arc::new(AtomicUsize::new(0));

    let leader = {
        let flight = flight.clone();
        let calls = calls.clone();
        tokio::spawn(async move {
            flight
                .run("k", || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Err(tinyagents_harness::TinyAgentsError::Model(
                        "boom".to_string(),
                    ))
                })
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(10)).await;
    let follower = {
        let flight = flight.clone();
        let calls = calls.clone();
        tokio::spawn(async move {
            flight
                .run("k", || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(ModelResponse::assistant("recovered"))
                })
                .await
        })
    };

    assert!(leader.await.unwrap().is_err());
    let (response, was_follower) = follower.await.unwrap().expect("follower recovers");
    assert_eq!(response.text(), "recovered");
    assert!(!was_follower, "the follower ran its own call");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn distinct_keys_do_not_block_each_other() {
    let flight = SingleFlight::new();
    let calls = Arc::new(AtomicUsize::new(0));
    for key in ["a", "b", "c"] {
        let calls = calls.clone();
        let (_, was_follower) = flight
            .run(key, || async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(ModelResponse::assistant(key))
            })
            .await
            .unwrap();
        assert!(!was_follower);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn cancelling_a_leader_releases_followers() {
    let flight = SingleFlight::new();
    let (started, started_rx) = tokio::sync::oneshot::channel();
    let leader_flight = flight.clone();
    let leader = tokio::spawn(async move {
        leader_flight
            .run("cancelled", || async move {
                let _ = started.send(());
                std::future::pending::<tinyagents_harness::Result<ModelResponse>>().await
            })
            .await
    });
    started_rx.await.unwrap();
    leader.abort();
    let recovered = tokio::time::timeout(
        Duration::from_secs(1),
        flight.run("cancelled", || async {
            Ok(ModelResponse::assistant("recovered"))
        }),
    )
    .await
    .expect("follower must not wait behind a cancelled leader")
    .unwrap();
    assert_eq!(recovered.0.text(), "recovered");
    assert!(!recovered.1);
    assert_eq!(flight.inflight_len(), 0);
}

// ── C-SQLITE-CACHE ───────────────────────────────────────────────────────────

#[cfg(feature = "sqlite")]
mod sqlite_backend {
    use super::*;
    use tinyagents_harness::Result;
    use tinyagents_harness::cache::SqliteResponseCache;

    #[tokio::test]
    async fn sqlite_cache_round_trips_and_expires() -> Result<()> {
        let cache = SqliteResponseCache::in_memory()?;
        assert!(cache.get("k").await?.is_none());

        cache.put("k", ModelResponse::assistant("durable")).await?;
        let hit = cache.get("k").await?.expect("stored");
        assert_eq!(hit.text(), "durable");
        assert_eq!(cache.stats().entries, 1);

        cache
            .put_with_ttl(
                "short",
                ModelResponse::assistant("stale"),
                Some(Duration::from_millis(20)),
            )
            .await?;
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(
            cache.get("short").await?.is_none(),
            "an expired row must read as a miss"
        );
        assert_eq!(
            cache.stats().entries,
            1,
            "the expired row must be purged lazily on read"
        );
        Ok(())
    }

    #[tokio::test]
    async fn sqlite_cache_survives_a_dropped_handle() -> Result<()> {
        // Durability is the whole point: `InMemoryResponseCache` loses
        // everything when the value is dropped, so every restart pays the
        // provider bill again.
        let dir = std::env::temp_dir().join(format!("tinyagents-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("responses.sqlite3");
        let _ = std::fs::remove_file(&path);

        {
            let cache = SqliteResponseCache::open(&path)?;
            cache
                .put("k", ModelResponse::assistant("persisted"))
                .await?;
        }
        let reopened = SqliteResponseCache::open(&path)?;
        let hit = reopened.get("k").await?.expect("survives a reopen");
        assert_eq!(hit.text(), "persisted");

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[tokio::test]
    async fn sqlite_namespaces_do_not_cross_serve_and_clear_independently() -> Result<()> {
        let base = SqliteResponseCache::in_memory()?;
        let tenant_a = base.with_namespace("tenant-a");
        let tenant_b = base.with_namespace("tenant-b");

        tenant_a.put("k", ModelResponse::assistant("a")).await?;
        assert!(
            tenant_b.get("k").await?.is_none(),
            "one tenant must never be served another's entry"
        );

        tenant_b.put("k", ModelResponse::assistant("b")).await?;
        tenant_a.clear().await?;
        assert!(tenant_a.get("k").await?.is_none());
        assert_eq!(
            tenant_b.get("k").await?.expect("untouched").text(),
            "b",
            "clearing one namespace must not touch another"
        );
        Ok(())
    }
}
