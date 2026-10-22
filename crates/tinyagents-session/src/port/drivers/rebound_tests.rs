use super::*;
use crate::testkit::conformance::{
    session_store_conformance, session_store_isolation_conformance, session_store_lease_conformance,
};
use crate::transcript::{SessionRef, TranscriptMessage, TranscriptMeta, TranscriptTurn};
use serde_json::Value;
use tinystoragedrivers_core::{Capabilities, MemoryStorage, Scope, ScopedStorage};

#[tokio::test]
async fn registry_rebind_failure_keeps_every_provider_on_its_original_backend() {
    let registry = SessionStoreRegistry::default();
    let original: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let alice = RebindableSessionStores::new_in_tenant(
        &registry,
        Arc::clone(&original),
        Scope::tenant("alice").unwrap(),
        false,
    )
    .unwrap();
    let bob = RebindableSessionStores::new_in_tenant(
        &registry,
        Arc::clone(&original),
        Scope::tenant("bob").unwrap(),
        false,
    )
    .unwrap();
    let alice_live = alice.for_agent("agent");
    let bob_live = bob.for_agent("agent");
    let alice_fresh = alice.for_agent("fresh-agent");
    let bob_fresh = bob.for_agent("fresh-agent");
    for live in [&alice_live, &bob_live, &alice_fresh, &bob_fresh] {
        live.kv
            .put("marker", "backend", serde_json::json!("original"))
            .await
            .unwrap();
    }
    drop(alice_fresh);
    drop(bob_fresh);

    let candidate: Arc<dyn StorageBackend> = Arc::new(RejectBobBackend {
        inner: MemoryStorage::new(),
    });
    assert!(registry.rebind(candidate).is_err());

    for live in [&alice_live, &bob_live] {
        assert_eq!(
            live.kv.get("marker", "backend").await.unwrap(),
            Some(serde_json::json!("original")),
            "existing live handle must remain on the original backend"
        );
    }
    for provider in [&alice, &bob] {
        assert_eq!(
            provider
                .for_agent("fresh-agent")
                .kv
                .get("marker", "backend")
                .await
                .unwrap(),
            Some(serde_json::json!("original")),
            "provider current backend must remain original"
        );
    }
}

#[test]
fn registry_keeps_independent_initial_backends_until_explicit_rebind() {
    let registry = SessionStoreRegistry::default();
    let first_backend: Arc<dyn StorageBackend> = Arc::new(NamedBackend {
        inner: MemoryStorage::new(),
        name: "first",
    });
    let second_backend: Arc<dyn StorageBackend> = Arc::new(NamedBackend {
        inner: MemoryStorage::new(),
        name: "second",
    });
    let first =
        RebindableSessionStores::new_in(&registry, Arc::clone(&first_backend), false).unwrap();
    let second = RebindableSessionStores::new_in(&registry, second_backend, false).unwrap();

    assert!(
        first
            .for_agent("agent")
            .transcripts
            .destination_key()
            .unwrap()
            .starts_with("first://")
    );
    assert!(
        second
            .for_agent("agent")
            .transcripts
            .destination_key()
            .unwrap()
            .starts_with("second://"),
        "registration alone must not rebind a provider to another provider's initial backend"
    );
}

#[derive(Debug)]
struct NamedBackend {
    inner: MemoryStorage,
    name: &'static str,
}

impl StorageBackend for NamedBackend {
    fn driver(&self) -> &'static str {
        self.name
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn for_scope(&self, scope: &Scope) -> std::result::Result<ScopedStorage, StorageError> {
        let scoped = self.inner.for_scope(scope)?;
        Ok(ScopedStorage::new(
            scope.clone(),
            self.name,
            Arc::clone(scoped.documents()),
            Arc::clone(scoped.streams()),
            Arc::clone(scoped.blobs()),
        ))
    }
}

#[derive(Debug)]
struct RejectBobBackend {
    inner: MemoryStorage,
}

impl StorageBackend for RejectBobBackend {
    fn driver(&self) -> &'static str {
        self.inner.driver()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn for_scope(&self, scope: &Scope) -> std::result::Result<ScopedStorage, StorageError> {
        if scope.as_str().starts_with("u:bob") {
            return Err(StorageError::invalid_input("injected bob scope failure"));
        }
        self.inner.for_scope(scope)
    }
}

#[tokio::test]
async fn rebound_provider_passes_full_session_store_conformance() {
    let registry = SessionStoreRegistry::default();
    let provider =
        RebindableSessionStores::new_in(&registry, Arc::new(MemoryStorage::new()), false).unwrap();
    session_store_conformance(&provider).await;
    session_store_isolation_conformance(&provider).await;
    session_store_lease_conformance(&provider).await;
}

fn append_session_row(stores: &AgentStores, session: &SessionRef, content: &str) {
    let meta = TranscriptMeta {
        agent_name: "lease-agent".into(),
        agent_id: session.agent_id.clone(),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some(session.session_key.clone()),
        task_id: None,
        session_id: Some(session.session_id()),
        parent_session_id: session.parent_session_id(),
    };
    let row = TranscriptMessage::user(content);
    let history = stores
        .transcripts
        .open_session(session, meta.clone())
        .unwrap();
    history
        .append_turn(TranscriptTurn {
            prev: &[],
            next: std::slice::from_ref(&row),
            meta: &meta,
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
}

fn session_contents(stores: &AgentStores, session: &SessionRef) -> Vec<String> {
    stores
        .transcripts
        .read_session_transcript(session)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect()
}

#[tokio::test]
async fn a_lease_keeps_its_backend_while_new_opens_follow_rebind() {
    let registry = SessionStoreRegistry::default();
    let old_backend = Arc::new(MemoryStorage::new());
    let provider = RebindableSessionStores::new_in(&registry, old_backend.clone(), false).unwrap();
    let lease = provider.lease_for_agent("agent-one").unwrap();
    let session = SessionRef::scoped("lease-thread", "agent-one");
    append_session_row(lease.stores(), &session, "old backend transcript");
    lease
        .stores()
        .kv
        .put("prefs", "key", serde_json::json!("leased"))
        .await
        .unwrap();
    let new_backend = Arc::new(MemoryStorage::new());
    assert_eq!(registry.rebind(new_backend.clone()).unwrap(), 1);
    assert_eq!(
        lease.stores().kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("leased"))
    );
    assert_eq!(
        session_contents(lease.stores(), &session),
        vec!["old backend transcript"]
    );
    let new_lease = provider.lease_for_agent("agent-one").unwrap();
    assert_eq!(
        new_lease.stores().kv.get("prefs", "key").await.unwrap(),
        None
    );
    assert!(
        new_lease
            .stores()
            .transcripts
            .read_session_transcript(&session)
            .is_none()
    );
    append_session_row(new_lease.stores(), &session, "new backend transcript");
    assert_eq!(
        session_contents(new_lease.stores(), &session),
        vec!["new backend transcript"]
    );
    new_lease
        .stores()
        .kv
        .put("prefs", "key", serde_json::json!("new"))
        .await
        .unwrap();
    let old = DriverSessionStores::new(old_backend, Scope::local())
        .unwrap()
        .try_for_agent("agent-one")
        .unwrap();
    assert_eq!(
        old.kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("leased"))
    );
}

#[tokio::test]
async fn actor_bound_driver_provider_can_open_a_pinned_lease() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let root = DriverSessionStores::new(
        Arc::new(MemoryStorage::new()),
        Scope::tenant("alice").unwrap(),
    )
    .unwrap();
    let actor = root.bind_actor(Arc::clone(&backend)).unwrap().unwrap();
    let lease = actor.lease_for_agent("agent").unwrap();
    let session = SessionRef::scoped("actor-lease-thread", "agent");
    append_session_row(lease.stores(), &session, "actor transcript");
    lease
        .stores()
        .kv
        .put("prefs", "key", serde_json::json!("actor"))
        .await
        .unwrap();
    let direct = DriverSessionStores::new(backend, Scope::tenant("alice").unwrap())
        .unwrap()
        .try_for_agent("agent")
        .unwrap();
    assert_eq!(
        direct.kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("actor"))
    );
    let second_lease = actor.lease_for_agent("agent").unwrap();
    assert_eq!(
        session_contents(second_lease.stores(), &session),
        vec!["actor transcript"],
        "reusing one actor-bound provider must preserve its admitted backend across operations"
    );
}

#[tokio::test]
async fn actor_bound_leases_keep_the_admitted_tenant_separate() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let alice_root = DriverSessionStores::new(
        Arc::new(MemoryStorage::new()),
        Scope::tenant("alice").unwrap(),
    )
    .unwrap();
    let bob_root = DriverSessionStores::new(
        Arc::new(MemoryStorage::new()),
        Scope::tenant("bob").unwrap(),
    )
    .unwrap();
    let alice_provider = alice_root
        .bind_actor(Arc::clone(&backend))
        .unwrap()
        .unwrap();
    let bob_provider = bob_root.bind_actor(Arc::clone(&backend)).unwrap().unwrap();
    let alice = alice_provider.lease_for_agent("same-agent").unwrap();
    let bob = bob_provider.lease_for_agent("same-agent").unwrap();
    let session = SessionRef::scoped("tenant-thread", "same-agent");
    append_session_row(alice.stores(), &session, "alice transcript");
    alice
        .stores()
        .kv
        .put("prefs", "key", serde_json::json!("alice"))
        .await
        .unwrap();
    assert_eq!(bob.stores().kv.get("prefs", "key").await.unwrap(), None);
    assert!(
        bob.stores()
            .transcripts
            .read_session_transcript(&session)
            .is_none()
    );
    assert_eq!(
        session_contents(alice.stores(), &session),
        vec!["alice transcript"]
    );
    assert_eq!(
        alice.stores().kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("alice"))
    );
}

#[tokio::test]
async fn leases_keep_tenant_scope_and_agent_isolation() {
    let backend = Arc::new(MemoryStorage::new());
    let alice = DriverSessionStores::new(backend.clone(), Scope::tenant("alice").unwrap()).unwrap();
    let bob = DriverSessionStores::new(backend, Scope::tenant("bob").unwrap()).unwrap();
    let alice_lease = alice.lease_for_agent("agent").unwrap();
    let bob_lease = bob.lease_for_agent("agent").unwrap();
    let session = SessionRef::scoped("tenant-thread", "agent");
    append_session_row(alice_lease.stores(), &session, "alice transcript");
    alice_lease
        .stores()
        .kv
        .put("prefs", "key", serde_json::json!("alice"))
        .await
        .unwrap();
    assert_eq!(
        bob_lease.stores().kv.get("prefs", "key").await.unwrap(),
        None
    );
    assert!(
        bob_lease
            .stores()
            .transcripts
            .read_session_transcript(&session)
            .is_none()
    );
}

#[tokio::test]
async fn retained_and_cloned_handles_follow_rebind_without_crossing_agent_scopes() {
    let registry = SessionStoreRegistry::default();
    let old_backend = Arc::new(MemoryStorage::new());
    let provider = RebindableSessionStores::new_in(&registry, old_backend.clone(), false).unwrap();
    let first = provider.for_agent("agent-one");
    let first_clone = first.clone();
    let second = provider.for_agent("agent-two");
    first
        .kv
        .put("prefs", "key", serde_json::json!("old-one"))
        .await
        .unwrap();
    second
        .kv
        .put("prefs", "key", serde_json::json!("old-two"))
        .await
        .unwrap();

    let new_backend = Arc::new(MemoryStorage::new());
    assert_eq!(registry.rebind(new_backend.clone()).unwrap(), 1);
    assert_eq!(first_clone.kv.get("prefs", "key").await.unwrap(), None);
    first_clone
        .kv
        .put("prefs", "key", serde_json::json!("new-one"))
        .await
        .unwrap();
    second
        .kv
        .put("prefs", "key", serde_json::json!("new-two"))
        .await
        .unwrap();

    assert_eq!(
        first.kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("new-one"))
    );
    assert_eq!(
        second.kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("new-two"))
    );
    let fresh_driver = DriverSessionStores::new(new_backend, Scope::local()).unwrap();
    assert_eq!(
        fresh_driver
            .for_agent("agent-one")
            .kv
            .get("prefs", "key")
            .await
            .unwrap(),
        Some(serde_json::json!("new-one"))
    );
    let old_driver = DriverSessionStores::new(old_backend, Scope::local()).unwrap();
    assert_eq!(
        old_driver
            .for_agent("agent-one")
            .kv
            .get("prefs", "key")
            .await
            .unwrap(),
        Some(serde_json::json!("old-one"))
    );
}

#[tokio::test]
async fn retained_journal_handle_routes_later_appends_to_new_backend() {
    let registry = SessionStoreRegistry::default();
    let provider =
        RebindableSessionStores::new_in(&registry, Arc::new(MemoryStorage::new()), false).unwrap();
    let stores = provider.for_agent("agent");
    stores
        .journal
        .append("turns", serde_json::json!({"where":"old"}))
        .await
        .unwrap();
    provider.rebind(Arc::new(MemoryStorage::new())).unwrap();
    assert_eq!(stores.journal.len("turns").await.unwrap(), 0);
    stores
        .journal
        .append("turns", serde_json::json!({"where":"new"}))
        .await
        .unwrap();
    assert_eq!(stores.journal.len("turns").await.unwrap(), 1);
}

struct BlockingStore {
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

#[derive(Debug)]
struct GateBackend {
    inner: MemoryStorage,
    gate: Mutex<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>>,
}

#[tokio::test]
async fn provider_registered_after_rebind_uses_registry_current_backend() {
    let registry = SessionStoreRegistry::default();
    let initial: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let provider =
        RebindableSessionStores::new_unregistered(Arc::clone(&initial), Scope::local(), false)
            .unwrap();
    let candidate: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let candidate_stores = DriverSessionStores::new(Arc::clone(&candidate), Scope::local())
        .unwrap()
        .for_agent("agent");
    candidate_stores
        .kv
        .put("marker", "backend", serde_json::json!("candidate"))
        .await
        .unwrap();

    assert_eq!(registry.rebind(Arc::clone(&candidate)).unwrap(), 0);
    let bob = RebindableSessionStores::new_in_tenant(
        &registry,
        Arc::clone(&candidate),
        Scope::tenant("bob").unwrap(),
        false,
    )
    .unwrap();
    let _bob_live = bob.for_agent("agent");
    assert!(
        registry
            .rebind(Arc::new(RejectBobBackend {
                inner: MemoryStorage::new(),
            }))
            .is_err()
    );
    registry.register(&provider, Arc::clone(&initial)).unwrap();

    assert_eq!(
        provider
            .for_agent("agent")
            .kv
            .get("marker", "backend")
            .await
            .unwrap(),
        Some(serde_json::json!("candidate")),
        "a provider registered after rebind must adopt the registry's current backend"
    );
}

#[async_trait::async_trait]
impl StorageBackend for GateBackend {
    fn driver(&self) -> &'static str {
        self.inner.driver()
    }
    fn capabilities(&self) -> tinystoragedrivers_core::Capabilities {
        self.inner.capabilities()
    }
    fn for_scope(
        &self,
        scope: &tinystoragedrivers_core::Scope,
    ) -> std::result::Result<tinystoragedrivers_core::ScopedStorage, StorageError> {
        if let Some((entered, release)) = self.gate.lock().unwrap().take() {
            let _ = entered.send(());
            let _ = release.recv();
        }
        self.inner.for_scope(scope)
    }
}

#[async_trait::async_trait]
impl Store for BlockingStore {
    async fn get(&self, _: &str, _: &str) -> HarnessResult<Option<Value>> {
        if let Some(entered) = self.entered.lock().unwrap().take() {
            let _ = entered.send(());
        }
        let release = { self.release.lock().unwrap().take() };
        if let Some(release) = release {
            let _ = release.await;
        }
        Ok(Some(serde_json::json!("old-operation")))
    }
    async fn put(&self, _: &str, _: &str, _: Value) -> HarnessResult<()> {
        Ok(())
    }
    async fn delete(&self, _: &str, _: &str) -> HarnessResult<()> {
        Ok(())
    }
    async fn list(&self, _: &str) -> HarnessResult<Vec<String>> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn rebind_during_a_blocked_store_call_keeps_that_call_old_and_routes_the_next_new() {
    let base_provider =
        DriverSessionStores::new(Arc::new(MemoryStorage::new()), Scope::local()).unwrap();
    let mut old = base_provider.for_agent("agent");
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    old.kv = Arc::new(BlockingStore {
        entered: Mutex::new(Some(entered_tx)),
        release: Mutex::new(Some(release_rx)),
    });
    let binding = Arc::new(LiveAgentStores {
        current: RwLock::new(old),
    });
    let operation = {
        let store = LiveStore(Arc::clone(&binding));
        tokio::spawn(async move { store.get("n", "k").await.unwrap() })
    };
    entered_rx.await.unwrap();

    let mut next = base_provider.for_agent("agent");
    next.kv = Arc::new(tinyagents_harness::store::InMemoryStore::new());
    next.kv
        .put("n", "k", serde_json::json!("new-operation"))
        .await
        .unwrap();
    *binding.current.write().unwrap() = next;
    release_tx.send(()).unwrap();

    assert_eq!(
        operation.await.unwrap(),
        Some(serde_json::json!("old-operation"))
    );
    assert_eq!(
        LiveStore(binding).get("n", "k").await.unwrap(),
        Some(serde_json::json!("new-operation"))
    );
}

#[test]
fn registration_waits_for_rebind_snapshot_and_then_binds_to_the_new_backend() {
    let registry = SessionStoreRegistry::default();
    let provider =
        RebindableSessionStores::new_in(&registry, Arc::new(MemoryStorage::new()), false).unwrap();
    let _already_live = provider.for_agent("already-live");
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let new_backend = Arc::new(GateBackend {
        inner: MemoryStorage::new(),
        gate: Mutex::new(Some((entered_tx, release_rx))),
    });
    let rebind_provider = Arc::clone(&provider);
    let rebind_backend: Arc<dyn StorageBackend> = new_backend.clone();
    let rebind = std::thread::spawn(move || rebind_provider.rebind(rebind_backend).unwrap());

    // The gate is reached while rebinding the already-live agent, after the
    // provider has captured its live registry. A registration at this point
    // must not be able to slip onto the old provider.
    entered_rx.recv().unwrap();
    assert!(provider.live.try_lock().is_err());
    release_tx.send(()).unwrap();
    rebind.join().unwrap();

    let late = provider.for_agent("late-agent");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        late.kv
            .put("n", "k", serde_json::json!("new"))
            .await
            .unwrap();
        let direct = DriverSessionStores::new(new_backend, Scope::local()).unwrap();
        assert_eq!(
            direct
                .for_agent("late-agent")
                .kv
                .get("n", "k")
                .await
                .unwrap(),
            Some(serde_json::json!("new"))
        );
    });
}

#[test]
fn concurrent_registration_and_rebind_leave_all_retained_handles_on_current_backend() {
    let registry = SessionStoreRegistry::default();
    let provider =
        RebindableSessionStores::new_in(&registry, Arc::new(MemoryStorage::new()), false).unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let registration_provider = Arc::clone(&provider);
    let registration_barrier = Arc::clone(&barrier);
    let registration = std::thread::spawn(move || {
        registration_barrier.wait();
        registration_provider.for_agent("racing-agent")
    });
    let rebind_provider = Arc::clone(&provider);
    let rebind_barrier = Arc::clone(&barrier);
    let new_backend = Arc::new(MemoryStorage::new());
    let expected_backend = Arc::clone(&new_backend);
    let rebind = std::thread::spawn(move || {
        rebind_barrier.wait();
        rebind_provider.rebind(new_backend).unwrap();
    });
    barrier.wait();
    let registered = registration.join().unwrap();
    rebind.join().unwrap();

    let fresh = provider.for_agent("racing-agent");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        registered
            .kv
            .put("n", "k", serde_json::json!("retained"))
            .await
            .unwrap();
        let independent = DriverSessionStores::new(expected_backend, Scope::local())
            .unwrap()
            .for_agent("racing-agent");
        assert_eq!(
            independent.kv.get("n", "k").await.unwrap(),
            Some(serde_json::json!("retained"))
        );
        assert_eq!(
            fresh.kv.get("n", "k").await.unwrap(),
            Some(serde_json::json!("retained"))
        );
    });
}

#[test]
fn concurrent_rebinds_publish_one_consistent_backend_for_existing_and_new_handles() {
    let registry = SessionStoreRegistry::default();
    let provider =
        RebindableSessionStores::new_in(&registry, Arc::new(MemoryStorage::new()), false).unwrap();
    let retained = provider.for_agent("agent");
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let backends = [
        Arc::new(MemoryStorage::new()),
        Arc::new(MemoryStorage::new()),
    ];
    let candidate_backends = backends.clone();
    let mut workers = Vec::new();
    for backend in &backends {
        let provider = Arc::clone(&provider);
        let barrier = Arc::clone(&barrier);
        let backend = Arc::clone(backend);
        workers.push(std::thread::spawn(move || {
            barrier.wait();
            provider.rebind(backend).unwrap();
        }));
    }
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    let fresh = provider.for_agent("agent");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        retained
            .kv
            .put("n", "k", serde_json::json!("same-backend"))
            .await
            .unwrap();
        let mut winner = None;
        for (index, backend) in candidate_backends.into_iter().enumerate() {
            let direct = DriverSessionStores::new(backend, Scope::local()).unwrap();
            if direct
                .for_agent("agent")
                .kv
                .get("n", "k")
                .await
                .unwrap()
                .is_some()
            {
                winner = Some(index);
            }
        }
        let winner = winner.expect("one concurrent rebind backend has the retained write");
        fresh
            .kv
            .put("n", "fresh", serde_json::json!("current"))
            .await
            .unwrap();
        let direct = DriverSessionStores::new(backends[winner].clone(), Scope::local()).unwrap();
        assert_eq!(
            direct.for_agent("agent").kv.get("n", "k").await.unwrap(),
            Some(serde_json::json!("same-backend"))
        );
        assert_eq!(
            direct
                .for_agent("agent")
                .kv
                .get("n", "fresh")
                .await
                .unwrap(),
            Some(serde_json::json!("current"))
        );
    });
}

#[tokio::test]
async fn tenant_is_preserved_across_rebind_without_crossing_other_tenants() {
    let registry = SessionStoreRegistry::default();
    let tenant = Scope::tenant("alice").unwrap();
    let provider = RebindableSessionStores::new_in_tenant(
        &registry,
        Arc::new(MemoryStorage::new()),
        tenant.clone(),
        false,
    )
    .unwrap();
    let retained = provider.for_agent("agent");
    let next = Arc::new(MemoryStorage::new());
    registry.rebind(next.clone()).unwrap();
    retained
        .kv
        .put("n", "k", serde_json::json!("alice"))
        .await
        .unwrap();
    let alice = DriverSessionStores::new(next.clone(), tenant)
        .unwrap()
        .for_agent("agent");
    let bob = DriverSessionStores::new(next.clone(), Scope::tenant("bob").unwrap())
        .unwrap()
        .for_agent("agent");
    let local = DriverSessionStores::new(next, Scope::local())
        .unwrap()
        .for_agent("agent");
    assert_eq!(
        alice.kv.get("n", "k").await.unwrap(),
        Some(serde_json::json!("alice"))
    );
    assert_eq!(bob.kv.get("n", "k").await.unwrap(), None);
    assert_eq!(local.kv.get("n", "k").await.unwrap(), None);
}
