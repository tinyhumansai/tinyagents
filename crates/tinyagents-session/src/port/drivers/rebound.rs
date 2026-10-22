//! Rebindable session-store providers with stable handles for live agents.
//!
//! A handle operation snapshots its current store immediately before the
//! operation and releases the binding lock before doing I/O. Operations that
//! already captured a backend finish there; later calls through retained or
//! cloned `AgentStores` follow the new backend after `rebind` completes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};

use serde_json::Value;
use tinyagents_harness::Result as HarnessResult;
use tinyagents_harness::store::{AppendStore, Store};
use tinystoragedrivers_core::{Scope, StorageBackend, StorageError};

use super::{AgentStores, SessionStoreLease, SessionStoreLeaseError, SessionStoreProvider};
use crate::DriverSessionStores;
use crate::port::{TurnState, TurnStates};
use crate::transcript::{
    SessionAdoption, SessionRef, ThreadTranscripts, TranscriptHistory, TranscriptLocator,
    TranscriptMeta, TranscriptPartial, TranscriptRead,
};
use crate::turn_state::TurnLifecycle;

/// Scopes live provider handles so one test or runtime can rebind its own
/// stores without moving unrelated providers.
#[derive(Default)]
pub struct SessionStoreRegistry {
    state: Mutex<SessionStoreRegistryState>,
}

#[derive(Default)]
struct SessionStoreRegistryState {
    live: Vec<Weak<RebindableSessionStores>>,
    current_backend: Option<Arc<dyn StorageBackend>>,
}

impl SessionStoreRegistry {
    fn register(
        &self,
        stores: &Arc<RebindableSessionStores>,
        initial_backend: Arc<dyn StorageBackend>,
    ) -> Result<(), StorageError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.live.retain(|weak| weak.strong_count() > 0);
        if let Some(current_backend) = &state.current_backend
            && !Arc::ptr_eq(current_backend, &initial_backend)
        {
            stores.rebind(Arc::clone(current_backend))?;
        }
        state.live.push(Arc::downgrade(stores));
        Ok(())
    }

    /// Rebinds every provider registered here, returning the number moved.
    ///
    /// Every provider prepares its new backend and live agent handles before
    /// any provider publishes them. If preparation fails, all providers keep
    /// their current backend and live handles.
    pub fn rebind(
        &self,
        backend: Arc<dyn StorageBackend>,
    ) -> std::result::Result<usize, StorageError> {
        // Keep registrations stable across the batch: otherwise a provider
        // could be registered after the snapshot and remain on the old
        // backend when this call returns.
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.live.retain(|weak| weak.strong_count() > 0);
        let stores = state
            .live
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        let prepared = stores
            .iter()
            .map(|store| store.prepare_rebind(Arc::clone(&backend)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for provider in prepared {
            provider.commit();
        }
        state.current_backend = Some(backend);
        Ok(stores.len())
    }
}

/// Driver-backed stores whose retained agent handles follow backend rebinds.
pub struct RebindableSessionStores {
    current: RwLock<Arc<DriverSessionStores>>,
    live: Mutex<HashMap<String, Weak<LiveAgentStores>>>,
    recover_on_open: bool,
    tenant: Scope,
}

impl RebindableSessionStores {
    /// Creates a provider registered in the process-wide registry.
    pub fn new(
        backend: Arc<dyn StorageBackend>,
        recover_on_open: bool,
    ) -> std::result::Result<Arc<Self>, StorageError> {
        Self::new_for_tenant(backend, Scope::local(), recover_on_open)
    }

    /// Creates a process-registered provider whose agents remain under `tenant`.
    pub fn new_for_tenant(
        backend: Arc<dyn StorageBackend>,
        tenant: Scope,
        recover_on_open: bool,
    ) -> std::result::Result<Arc<Self>, StorageError> {
        static PROCESS: std::sync::LazyLock<SessionStoreRegistry> =
            std::sync::LazyLock::new(SessionStoreRegistry::default);
        Self::new_in_tenant(&PROCESS, backend, tenant, recover_on_open)
    }

    /// Creates a provider registered only in `registry`.
    pub fn new_in(
        registry: &SessionStoreRegistry,
        backend: Arc<dyn StorageBackend>,
        recover_on_open: bool,
    ) -> std::result::Result<Arc<Self>, StorageError> {
        Self::new_in_tenant(registry, backend, Scope::local(), recover_on_open)
    }

    /// Creates a registry-local provider whose tenant survives backend rebinds.
    pub fn new_in_tenant(
        registry: &SessionStoreRegistry,
        backend: Arc<dyn StorageBackend>,
        tenant: Scope,
        recover_on_open: bool,
    ) -> std::result::Result<Arc<Self>, StorageError> {
        let initial_backend = Arc::clone(&backend);
        let this = Self::new_unregistered(backend, tenant, recover_on_open)?;
        registry.register(&this, initial_backend)?;
        Ok(this)
    }

    fn new_unregistered(
        backend: Arc<dyn StorageBackend>,
        tenant: Scope,
        recover_on_open: bool,
    ) -> std::result::Result<Arc<Self>, StorageError> {
        let driver =
            DriverSessionStores::new(backend, tenant.clone())?.recover_on_open(recover_on_open);
        Ok(Arc::new(Self {
            current: RwLock::new(Arc::new(driver)),
            live: Mutex::new(HashMap::new()),
            recover_on_open,
            tenant,
        }))
    }

    fn current(&self) -> Arc<DriverSessionStores> {
        Arc::clone(&self.current.read().unwrap_or_else(PoisonError::into_inner))
    }

    /// Replaces the backend and updates every retained agent handle.
    ///
    /// Existing method calls retain the provider snapshot they started with;
    /// subsequent calls through a live handle resolve against the new backend.
    pub fn rebind(
        &self,
        backend: Arc<dyn StorageBackend>,
    ) -> std::result::Result<(), StorageError> {
        self.prepare_rebind(backend)?.commit();
        Ok(())
    }

    fn prepare_rebind(
        &self,
        backend: Arc<dyn StorageBackend>,
    ) -> Result<PreparedRebind<'_>, StorageError> {
        // Serialize publication with agent registration so a concurrent
        // `for_agent` cannot attach to the old provider after the batch
        // snapshot. The guard remains held through commit. This synchronous
        // lock is never held over async I/O.
        let mut live_registry = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        live_registry.retain(|_, weak| weak.strong_count() > 0);
        let next = Arc::new(
            DriverSessionStores::new(backend, self.tenant.clone())?
                .recover_on_open(self.recover_on_open),
        );
        let live = live_registry
            .iter()
            .filter_map(|(agent_id, weak)| {
                weak.upgrade().map(|binding| (agent_id.clone(), binding))
            })
            .collect::<Vec<_>>();

        // Prepare every replacement before publishing any, so a failed scope
        // bind cannot strand some live handles on each side of the rebind.
        let replacements = live
            .iter()
            .map(|(agent_id, binding)| {
                next.try_for_agent(agent_id)
                    .map(|stores| (Arc::clone(binding), stores))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(PreparedRebind {
            provider: self,
            _live_registry: live_registry,
            next,
            replacements,
        })
    }
}

struct PreparedRebind<'a> {
    provider: &'a RebindableSessionStores,
    _live_registry: MutexGuard<'a, HashMap<String, Weak<LiveAgentStores>>>,
    next: Arc<DriverSessionStores>,
    replacements: Vec<(Arc<LiveAgentStores>, AgentStores)>,
}

impl PreparedRebind<'_> {
    fn commit(self) {
        for (binding, stores) in self.replacements {
            *binding
                .current
                .write()
                .unwrap_or_else(PoisonError::into_inner) = stores;
        }
        *self
            .provider
            .current
            .write()
            .unwrap_or_else(PoisonError::into_inner) = self.next;
    }
}

struct LiveAgentStores {
    current: RwLock<AgentStores>,
}

impl LiveAgentStores {
    fn snapshot(&self) -> AgentStores {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn handles(self: &Arc<Self>) -> AgentStores {
        AgentStores {
            transcripts: Arc::new(LiveTranscriptLocator(Arc::clone(self))),
            turn_states: Arc::new(LiveTurnStates(Arc::clone(self))),
            kv: Arc::new(LiveStore(Arc::clone(self))),
            journal: Arc::new(LiveAppendStore(Arc::clone(self))),
        }
    }
}

#[async_trait::async_trait]
impl Store for LiveStore {
    async fn get(&self, namespace: &str, key: &str) -> HarnessResult<Option<Value>> {
        let store = self.0.snapshot().kv;
        store.get(namespace, key).await
    }
    async fn put(&self, namespace: &str, key: &str, value: Value) -> HarnessResult<()> {
        let store = self.0.snapshot().kv;
        store.put(namespace, key, value).await
    }
    async fn delete(&self, namespace: &str, key: &str) -> HarnessResult<()> {
        let store = self.0.snapshot().kv;
        store.delete(namespace, key).await
    }
    async fn list(&self, namespace: &str) -> HarnessResult<Vec<String>> {
        let store = self.0.snapshot().kv;
        store.list(namespace).await
    }
}

#[async_trait::async_trait]
impl AppendStore for LiveAppendStore {
    async fn append(&self, stream: &str, value: Value) -> HarnessResult<u64> {
        let store = self.0.snapshot().journal;
        store.append(stream, value).await
    }
    async fn read_from(&self, stream: &str, offset: u64) -> HarnessResult<Vec<(u64, Value)>> {
        let store = self.0.snapshot().journal;
        store.read_from(stream, offset).await
    }
    async fn read_window(
        &self,
        stream: &str,
        offset: u64,
        limit: usize,
    ) -> HarnessResult<Vec<(u64, Value)>> {
        let store = self.0.snapshot().journal;
        store.read_window(stream, offset, limit).await
    }
    async fn len(&self, stream: &str) -> HarnessResult<u64> {
        let store = self.0.snapshot().journal;
        store.len(stream).await
    }
}

struct LiveStore(Arc<LiveAgentStores>);
struct LiveAppendStore(Arc<LiveAgentStores>);
struct LiveTurnStates(Arc<LiveAgentStores>);
struct LiveTranscriptLocator(Arc<LiveAgentStores>);

impl TurnStates for LiveTurnStates {
    fn put(&self, state: &TurnState) -> std::result::Result<(), String> {
        self.0.snapshot().turn_states.put(state)
    }
    fn put_unless_completed(&self, state: &TurnState) -> std::result::Result<bool, String> {
        self.0.snapshot().turn_states.put_unless_completed(state)
    }
    fn get(&self, thread_id: &str) -> std::result::Result<Option<TurnState>, String> {
        self.0.snapshot().turn_states.get(thread_id)
    }
    fn get_turn(
        &self,
        thread_id: &str,
        request_id: &str,
    ) -> std::result::Result<Option<TurnState>, String> {
        self.0
            .snapshot()
            .turn_states
            .get_turn(thread_id, request_id)
    }
    fn delete(&self, thread_id: &str) -> std::result::Result<bool, String> {
        self.0.snapshot().turn_states.delete(thread_id)
    }
    fn delete_turn(&self, thread_id: &str, request_id: &str) -> std::result::Result<bool, String> {
        self.0
            .snapshot()
            .turn_states
            .delete_turn(thread_id, request_id)
    }
    fn list(&self) -> std::result::Result<Vec<TurnState>, String> {
        self.0.snapshot().turn_states.list()
    }
    fn list_thread(&self, thread_id: &str) -> std::result::Result<Vec<TurnState>, String> {
        self.0.snapshot().turn_states.list_thread(thread_id)
    }
    fn clear_all(&self) -> std::result::Result<usize, String> {
        self.0.snapshot().turn_states.clear_all()
    }
    fn mark_all_interrupted(&self, now: &str) -> std::result::Result<usize, String> {
        self.0.snapshot().turn_states.mark_all_interrupted(now)
    }
    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now: &str,
    ) -> std::result::Result<bool, String> {
        self.0
            .snapshot()
            .turn_states
            .settle_turn(thread_id, request_id, lifecycle, now)
    }
}

impl TranscriptLocator for LiveTranscriptLocator {
    fn destination_key(&self) -> Option<String> {
        self.0.snapshot().transcripts.destination_key()
    }
    fn latest_for_agent(&self, agent: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.0.snapshot().transcripts.latest_for_agent(agent)
    }
    fn root_for_thread(&self, thread: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.0.snapshot().transcripts.root_for_thread(thread)
    }
    fn root_for_thread_scoped(
        &self,
        thread: &str,
        agent: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        self.0
            .snapshot()
            .transcripts
            .root_for_thread_scoped(thread, agent)
    }
    fn thread_transcripts(&self, thread: &str, agent: Option<&str>) -> Option<ThreadTranscripts> {
        self.0
            .snapshot()
            .transcripts
            .thread_transcripts(thread, agent)
    }
    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        // A bound history handle is an operation/session snapshot. The
        // locator remains live for the next lookup, while this handle finishes
        // against the backend it was opened on.
        self.0.snapshot().transcripts.open_stem(stem, seed)
    }
    fn head_generation(&self, session: &SessionRef) -> SessionRef {
        self.0.snapshot().transcripts.head_generation(session)
    }
    fn session_exists(&self, session: &SessionRef) -> bool {
        self.0.snapshot().transcripts.session_exists(session)
    }
    fn session_chain(&self, session: &SessionRef) -> Vec<SessionRef> {
        self.0.snapshot().transcripts.session_chain(session)
    }
    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        self.0
            .snapshot()
            .transcripts
            .read_session_transcript(session)
    }
    fn open_session(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.0.snapshot().transcripts.open_session(session, seed)
    }
    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<SessionAdoption>> {
        self.0
            .snapshot()
            .transcripts
            .adopt_legacy(session, thread, seed)
    }
    fn append_interrupted_partial(
        &self,
        thread: &str,
        agent: Option<&str>,
        partial: &TranscriptPartial,
        request: Option<&str>,
    ) -> anyhow::Result<bool> {
        self.0
            .snapshot()
            .transcripts
            .append_interrupted_partial(thread, agent, partial, request)
    }
    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        self.0
            .snapshot()
            .transcripts
            .begin_generation(session, seed)
    }
    fn begin_generation_from_baseline(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: &[crate::transcript::TranscriptMessage],
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        self.0
            .snapshot()
            .transcripts
            .begin_generation_from_baseline(session, seed, baseline)
    }
    fn truncate_into_next_generation(
        &self,
        session: &SessionRef,
        cut: crate::transcript::TruncateCut,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(
        SessionRef,
        Arc<dyn TranscriptHistory>,
        Vec<crate::transcript::TranscriptMessage>,
    )> {
        self.0
            .snapshot()
            .transcripts
            .truncate_into_next_generation(session, cut, seed)
    }
}

/// Each resolved trait object is stable, while its backing stores are replaced
/// atomically when the owning provider rebinds.
impl SessionStoreProvider for RebindableSessionStores {
    fn lease_for_agent(&self, agent_id: &str) -> Result<SessionStoreLease, SessionStoreLeaseError> {
        self.current().lease_for_agent(agent_id)
    }

    fn for_agent(&self, agent_id: &str) -> AgentStores {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        live.retain(|_, weak| weak.strong_count() > 0);
        if let Some(binding) = live.get(agent_id).and_then(Weak::upgrade) {
            return binding.handles();
        }
        let stores = self.current().for_agent(agent_id);
        let binding = Arc::new(LiveAgentStores {
            current: RwLock::new(stores),
        });
        live.insert(agent_id.to_owned(), Arc::downgrade(&binding));
        binding.handles()
    }

    fn try_for_agent(&self, agent_id: &str) -> std::result::Result<AgentStores, StorageError> {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        live.retain(|_, weak| weak.strong_count() > 0);
        if let Some(binding) = live.get(agent_id).and_then(Weak::upgrade) {
            return Ok(binding.handles());
        }
        let stores = self.current().try_for_agent(agent_id)?;
        let binding = Arc::new(LiveAgentStores {
            current: RwLock::new(stores),
        });
        live.insert(agent_id.to_owned(), Arc::downgrade(&binding));
        Ok(binding.handles())
    }

    fn bind_actor(
        &self,
        backend: Arc<dyn StorageBackend>,
    ) -> std::result::Result<Option<Arc<dyn SessionStoreProvider>>, StorageError> {
        self.current().bind_actor(backend)
    }
    fn bind_actor_with_schema_admission(
        &self,
        backend: Arc<dyn StorageBackend>,
        admission: tinyagents_harness::store::SchemaAdmission,
    ) -> std::result::Result<Option<Arc<dyn SessionStoreProvider>>, StorageError> {
        self.current()
            .bind_actor_with_schema_admission(backend, admission)
    }
    fn run_ledger_for_agent(
        &self,
        agent_id: &str,
    ) -> crate::run_ledger::LedgerResult<Option<Arc<dyn crate::run_ledger::RunLedgerStore>>> {
        self.current().run_ledger_for_agent(agent_id)
    }
    fn conversations_for_agent(
        &self,
        agent_id: &str,
    ) -> crate::threads::ConversationResult<Option<Arc<dyn crate::threads::ConversationPersistence>>>
    {
        self.current().conversations_for_agent(agent_id)
    }
    fn recover(&self) -> anyhow::Result<()> {
        self.current().recover()
    }
}

#[cfg(test)]
#[path = "rebound_tests.rs"]
mod tests;
