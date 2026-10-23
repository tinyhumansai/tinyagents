use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tinyagents_storage_kit::{
    Capabilities, Page, Query, ReadCheck, Sort, StreamEntry, Version, WriteOp, WriteResult,
};
use tinystoragedrivers_api::StorageBackend;
use tokio::sync::{Barrier, Notify};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Probe,
    Reserve,
    Append,
    Finalize,
    ReadWindow,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Action {
    Pause,
    PauseLoseAck,
    LoseAck,
    FailBefore,
    Unsupported,
    Barrier,
}
#[derive(Debug)]
pub(super) struct Fault {
    phase: Phase,
    action: Action,
    hits: AtomicUsize,
    skip: usize,
    pub conflicts: AtomicUsize,
    barrier: Barrier,
    pub reached: Notify,
    pub release: Notify,
}
impl Fault {
    pub fn new(phase: Phase, action: Action) -> Arc<Self> {
        Self::new_after(phase, action, 0)
    }
    pub fn new_after(phase: Phase, action: Action, skip: usize) -> Arc<Self> {
        Arc::new(Self {
            phase,
            action,
            hits: AtomicUsize::new(0),
            skip,
            conflicts: AtomicUsize::new(0),
            barrier: Barrier::new(2),
            reached: Notify::new(),
            release: Notify::new(),
        })
    }
    fn take(&self, phase: Phase) -> bool {
        if self.phase != phase {
            return false;
        }
        let hit = self.hits.fetch_add(1, Ordering::SeqCst);
        hit >= self.skip
            && hit - self.skip
                < if matches!(self.action, Action::Barrier) {
                    2
                } else {
                    1
                }
    }
    async fn after(&self, active: bool) -> tinystoragedrivers_types::Result<()> {
        if !active {
            return Ok(());
        }
        match self.action {
            Action::Pause | Action::PauseLoseAck => {
                self.reached.notify_one();
                self.release.notified().await;
                if matches!(self.action, Action::PauseLoseAck) {
                    Err(Self::failure())
                } else {
                    Ok(())
                }
            }
            Action::LoseAck => Err(Self::failure()),
            _ => Ok(()),
        }
    }
    fn failure() -> StorageError {
        StorageError::unavailable("injected native acknowledgment loss")
            .with_source(std::io::Error::other("original native cause"))
    }
}
#[derive(Debug)]
struct Documents {
    inner: Arc<dyn DocumentStore>,
    fault: Arc<Fault>,
    capabilities: Option<Capabilities>,
}
#[async_trait]
impl DocumentStore for Documents {
    fn capabilities(&self) -> Capabilities {
        self.capabilities
            .unwrap_or_else(|| self.inner.capabilities())
    }
    async fn ensure_collection(&self, _: &CollectionSpec) -> tinystoragedrivers_types::Result<()> {
        panic!("actor attempted DDL")
    }
    async fn validate_collection_spec(
        &self,
        spec: &CollectionSpec,
    ) -> tinystoragedrivers_types::Result<()> {
        self.inner.validate_collection_spec(spec).await
    }
    async fn get(
        &self,
        c: &str,
        id: &str,
    ) -> tinystoragedrivers_types::Result<Option<Versioned<Value>>> {
        self.inner.get(c, id).await
    }
    async fn put(
        &self,
        c: &str,
        id: &str,
        v: Value,
        p: Precondition,
    ) -> tinystoragedrivers_types::Result<Version> {
        self.inner.put(c, id, v, p).await
    }
    async fn put_with(
        &self,
        c: &str,
        id: &str,
        v: Value,
        p: Precondition,
        o: tinystoragedrivers_types::WriteOptions,
    ) -> tinystoragedrivers_types::Result<Version> {
        self.inner.put_with(c, id, v, p, o).await
    }
    async fn delete(
        &self,
        c: &str,
        id: &str,
        p: Precondition,
    ) -> tinystoragedrivers_types::Result<bool> {
        self.inner.delete(c, id, p).await
    }
    async fn query(
        &self,
        c: &str,
        q: &Query,
    ) -> tinystoragedrivers_types::Result<Page<Versioned<Value>>> {
        self.inner.query(c, q).await
    }
    async fn count(&self, c: &str, f: &Filter) -> tinystoragedrivers_types::Result<u64> {
        self.inner.count(c, f).await
    }
    async fn delete_where(&self, c: &str, f: &Filter) -> tinystoragedrivers_types::Result<u64> {
        self.inner.delete_where(c, f).await
    }
    async fn claim(
        &self,
        c: &str,
        f: &Filter,
        s: &[Sort],
        p: &Value,
    ) -> tinystoragedrivers_types::Result<Option<Versioned<Value>>> {
        self.inner.claim(c, f, s, p).await
    }
    async fn drop_collection(&self, c: &str) -> tinystoragedrivers_types::Result<()> {
        self.inner.drop_collection(c).await
    }
    async fn atomic_batch_checked(
        &self,
        checks: Vec<ReadCheck>,
        ops: Vec<WriteOp>,
    ) -> tinystoragedrivers_types::Result<Vec<WriteResult>> {
        let phase = if ops.len() == 1 {
            Phase::Reserve
        } else {
            Phase::Finalize
        };
        let active = self.fault.take(phase);
        if active && matches!(self.fault.action, Action::FailBefore) {
            return Err(Fault::failure());
        }
        if active && matches!(self.fault.action, Action::Barrier) {
            self.fault.barrier.wait().await;
        }
        let result = self.inner.atomic_batch_checked(checks, ops).await;
        if matches!(&result, Err(error) if error.kind() == ErrorKind::Conflict) {
            self.fault.conflicts.fetch_add(1, Ordering::SeqCst);
        }
        let result = result?;
        self.fault.after(active).await?;
        Ok(result)
    }
}
#[derive(Debug)]
struct Streams {
    inner: Arc<dyn StreamStore>,
    fault: Arc<Fault>,
}
#[async_trait]
impl StreamStore for Streams {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }

    async fn append(&self, s: &str, v: Value) -> tinystoragedrivers_types::Result<u64> {
        self.inner.append(s, v).await
    }
    async fn append_batch(&self, s: &str, v: Vec<Value>) -> tinystoragedrivers_types::Result<u64> {
        self.inner.append_batch(s, v).await
    }
    async fn append_batch_if_offset(
        &self,
        s: &str,
        o: u64,
        v: Vec<Value>,
    ) -> tinystoragedrivers_types::Result<u64> {
        let phase = if v.is_empty() {
            Phase::Probe
        } else {
            Phase::Append
        };
        let active = self.fault.take(phase);
        if active && matches!(self.fault.action, Action::Unsupported) {
            return Err(StorageError::unsupported(
                tinystoragedrivers_types::Capability::ConditionalStreamAppend,
                "paired stream refuses actual probe",
            ));
        }
        if active && matches!(self.fault.action, Action::FailBefore) {
            return Err(Fault::failure());
        }
        let result = self.inner.append_batch_if_offset(s, o, v).await?;
        self.fault.after(active).await?;
        Ok(result)
    }
    async fn read_window(
        &self,
        s: &str,
        o: u64,
        n: usize,
    ) -> tinystoragedrivers_types::Result<Vec<StreamEntry>> {
        let active = self.fault.take(Phase::ReadWindow);
        if active && matches!(self.fault.action, Action::FailBefore) {
            return Err(Fault::failure());
        }
        let result = self.inner.read_window(s, o, n).await?;
        self.fault.after(active).await?;
        Ok(result)
    }
    async fn len(&self, s: &str) -> tinystoragedrivers_types::Result<u64> {
        self.inner.len(s).await
    }
    async fn truncate_before(&self, s: &str, o: u64) -> tinystoragedrivers_types::Result<u64> {
        self.inner.truncate_before(s, o).await
    }
    async fn delete_stream(&self, s: &str) -> tinystoragedrivers_types::Result<bool> {
        self.inner.delete_stream(s).await
    }
    async fn streams(&self, p: &str) -> tinystoragedrivers_types::Result<Vec<String>> {
        self.inner.streams(p).await
    }
}
pub(super) fn faulted(
    store: &DriverConversationStore,
    fault: Arc<Fault>,
) -> DriverConversationStore {
    DriverConversationStore {
        docs: Arc::new(Documents {
            inner: store.docs.clone(),
            fault: fault.clone(),
            capabilities: None,
        }),
        streams: Arc::new(Streams {
            inner: store.streams.clone(),
            fault,
        }),
        validate_only: true,
    }
}
pub(super) fn without_document_capability(
    store: &DriverConversationStore,
    capability: tinystoragedrivers_types::Capability,
) -> DriverConversationStore {
    DriverConversationStore {
        docs: Arc::new(Documents {
            inner: store.docs.clone(),
            fault: Fault::new(Phase::Probe, Action::LoseAck),
            capabilities: Some(store.docs.capabilities().without(capability)),
        }),
        streams: store.streams.clone(),
        validate_only: true,
    }
}
pub(super) async fn actor(backend: &dyn StorageBackend) -> DriverConversationStore {
    let scope = Scope::new("alice").unwrap();
    let raw = backend.for_scope(&scope).unwrap();
    DriverConversationStore::prepare_append_schema(raw.documents().as_ref())
        .await
        .unwrap();
    let store = DriverConversationStore::new_validate_only(
        tinyagents_storage_kit::ScopedStorage::new(raw, scope),
    );
    store.ensure_thread(request("same")).await.unwrap();
    store
}
