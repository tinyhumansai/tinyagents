//! A [`CompletionStore`] over `tinystoragedrivers` document ports.
//!
//! [`DriverCompletionStore`] keeps the latest [`CompletionRecord`] per task on
//! whatever backend the host opened, so a completion the router reported as
//! stored survives a restart. The document handle is already bound to a tenant
//! scope.
//!
//! # Layout
//!
//! One collection, `tasks.completions` (see [`storage_schema`](crate::storage_schema)),
//! with one document per task id under `Key::of([task_id])`: a bare snake_case
//! body (`task_id`, `parent_key`, `agent_id`, `state`, `result`, `finished_ms`,
//! ...). `parent_key` and `state` are indexed for [`CompletionStore::list`] and
//! [`CompletionStore::compact`]. The record's last-update time is the document's
//! `updated_ms` metadata, which the driver stamps on every write. Ordinary
//! `put` is an upsert; claim, reclaim, and settlement operations use a row
//! version precondition so concurrent routers cannot overwrite one another.
//!
//! # Sync trait
//!
//! [`CompletionStore`] is synchronous, so each call runs its port futures on a
//! [`Blocking`] bridge thread; this works from any caller, inside a tokio
//! runtime or not.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_storage_kit::primitives::LEASES;
use tinyagents_storage_kit::{
    Blocking, CollectionSpec, DocumentStore, DocumentStoreExt, ErrorKind, Filter, IndexSpec,
    Leases, Precondition, Query, Scope, StorageBackend, StorageError, UPDATED_FIELD, Versioned,
    WriteCondition, WriteOptions,
};
use tokio::sync::OnceCell;

use super::store::{
    ClaimResolution, CompletionStore, ParentGateClaim, PendingClaimRequest, RenewClaimRequest,
    ResolveClaimRequest, TaskClaimRequest,
};
use super::types::{CompletionRecord, CompletionState};
use crate::storage::{
    COMPLETIONS, CompletionDoc, SCHEMA_VERSION, completions_spec, task_id_key, to_ms,
};

const LOG_PREFIX: &str = "[completion-store]";
const PARENT_GATE_OWNER: &str = "tinyagents.tasks.completions";

fn map_error(error: StorageError) -> TinyAgentsError {
    TinyAgentsError::Storage(format!("completion store: {error}"))
}

fn decode(row: Versioned<serde_json::Value>) -> Option<CompletionRecord> {
    let updated_ms = row.meta.updated_ms;
    match serde_json::from_value::<CompletionDoc>(row.doc) {
        Ok(doc) => Some(doc.into_record(updated_ms)),
        Err(error) => {
            // One unreadable record must not block every other completion.
            tracing::warn!(error = %error, "{LOG_PREFIX} skipping unreadable record");
            None
        }
    }
}

fn decode_strict(row: Versioned<serde_json::Value>) -> Result<CompletionRecord> {
    let updated_ms = row.meta.updated_ms;
    serde_json::from_value::<CompletionDoc>(row.doc)
        .map(|doc| doc.into_record(updated_ms))
        .map_err(|error| {
            TinyAgentsError::Storage(format!(
                "completion store contains unreadable record: {error}"
            ))
        })
}

/// A durable [`CompletionStore`] stored in a driver [`DocumentStore`].
#[derive(Clone)]
pub struct DriverCompletionStore {
    backend: Arc<dyn StorageBackend>,
    scope: Scope,
    coordination_domain: String,
    docs: Arc<dyn DocumentStore>,
    bridge: Blocking,
    declared: Arc<OnceCell<()>>,
}

impl std::fmt::Debug for DriverCompletionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverCompletionStore")
            .field("scope", &self.scope.as_str())
            .field("coordination_domain", &self.coordination_domain)
            .finish_non_exhaustive()
    }
}

impl DriverCompletionStore {
    /// Store completions in `tasks.completions`, with its own bridge.
    ///
    /// The default coordination identity is tied to this exact `Arc`
    /// handle. Use `new_with_coordination_domain` when distinct handles may
    /// point at the same shared database; all processes using that database
    /// and scope must be configured with the same trusted domain.
    ///
    /// # Errors
    ///
    /// When the bridge's runtime thread cannot start.
    pub fn new(
        backend: Arc<dyn StorageBackend>,
        scope: Scope,
    ) -> std::result::Result<Self, StorageError> {
        let coordination_domain = format!("backend-handle:{:p}", Arc::as_ptr(&backend));
        Self::new_with_coordination_domain(backend, scope, coordination_domain)
    }

    /// Creates a driver-backed store with an explicit identity for a shared
    /// backend domain. Hosts opening distinct handles to the same database
    /// must pass the same opaque domain id; unrelated databases must use
    /// different ids. Scope remains part of the resulting identity.
    pub fn new_with_coordination_domain(
        backend: Arc<dyn StorageBackend>,
        scope: Scope,
        coordination_domain: impl Into<String>,
    ) -> std::result::Result<Self, StorageError> {
        Self::with_bridge_and_coordination_domain(
            backend,
            scope,
            Blocking::new()?,
            coordination_domain,
        )
    }

    /// Store completions in `tasks.completions`, running the synchronous trait
    /// on `bridge` so several stores share one thread.
    ///
    /// The default coordination identity is tied to this exact `Arc`
    /// handle. Use `with_bridge_and_coordination_domain` when distinct
    /// handles may point at the same shared database.
    ///
    /// # Panics
    ///
    /// Never: the collection name is a valid constant.
    #[allow(clippy::expect_used, reason = "the collection name is a constant")]
    pub fn with_bridge(
        backend: Arc<dyn StorageBackend>,
        scope: Scope,
        bridge: Blocking,
    ) -> std::result::Result<Self, StorageError> {
        let coordination_domain = format!("backend-handle:{:p}", Arc::as_ptr(&backend));
        Self::with_bridge_and_coordination_domain(backend, scope, bridge, coordination_domain)
    }

    /// Like [`Self::with_bridge`], with the caller's stable shared-backend
    /// coordination identity.
    pub fn with_bridge_and_coordination_domain(
        backend: Arc<dyn StorageBackend>,
        scope: Scope,
        bridge: Blocking,
        coordination_domain: impl Into<String>,
    ) -> std::result::Result<Self, StorageError> {
        // Fail during setup when the backend cannot issue durable handles
        // whose writes are atomically guarded by the lease document.
        let probe = WriteCondition::new(
            scope.clone(),
            "tinyagents.condition_probe",
            "completion-store",
            Filter::eq("holder", "probe"),
        )?;
        backend.for_scope_conditioned(&scope, &probe)?;
        let scoped = backend.for_scope(&scope)?;
        let docs = scoped.documents().clone();
        Ok(Self {
            docs,
            backend,
            scope,
            coordination_domain: coordination_domain.into(),
            bridge,
            declared: Arc::new(OnceCell::new()),
        })
    }

    /// The declaration a store using `collection` needs.
    pub fn collection_spec(collection: impl Into<String>) -> CollectionSpec {
        CollectionSpec::new(collection)
            .index(IndexSpec::new("by_parent", ["parent_key"]))
            .index(IndexSpec::new("by_state", ["state"]))
    }

    /// Declares the backing collection eagerly so a host can decide whether to
    /// keep this durable store or fall back to memory before accepting work.
    pub fn declare_collections(&self) -> Result<()> {
        let this = self.clone();
        self.run(async move { this.declare().await })
    }

    fn run<T: Send + 'static>(
        &self,
        work: impl std::future::Future<Output = Result<T>> + Send + 'static,
    ) -> Result<T> {
        self.bridge
            .run(work)
            .map_err(|e| TinyAgentsError::Storage(format!("completion store bridge: {e}")))?
    }

    async fn declare(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                self.docs
                    .ensure_collection(&completions_spec())
                    .await
                    .map_err(map_error)?;
                self.docs
                    .ensure_collection(&CollectionSpec::new(LEASES))
                    .await
                    .map_err(map_error)
            })
            .await
            .map(|_| ())
    }

    async fn fenced_docs(
        &self,
        parent_key: &str,
        claim: &ParentGateClaim,
    ) -> Result<Arc<dyn DocumentStore>> {
        let lease = Leases::new(&self.docs)
            .current(PARENT_GATE_OWNER, parent_key)
            .await
            .map_err(map_error)?
            .filter(|lease| lease.holder == claim.holder && lease.epoch == claim.epoch)
            .ok_or_else(|| {
                TinyAgentsError::Storage("completion store: parent admission lease was lost".into())
            })?;
        let condition = lease
            .write_condition(self.scope.clone())
            .map_err(map_error)?;
        let fenced = self
            .backend
            .for_scope_conditioned(&self.scope, &condition)
            .map_err(map_error)?;
        Ok(Arc::clone(fenced.documents()))
    }
}

impl CompletionStore for DriverCompletionStore {
    fn coordination_identity(&self) -> Option<String> {
        Some(format!(
            "driver:{}",
            serde_json::to_string(&(self.coordination_domain.as_str(), self.scope.as_str()))
                .expect("driver coordination identity is serializable")
        ))
    }

    fn acquire_parent_gate(
        &self,
        parent_key: &str,
        holder: &str,
        ttl: Duration,
    ) -> Result<Option<ParentGateClaim>> {
        let this = self.clone();
        let parent_key = parent_key.to_owned();
        let holder = holder.to_owned();
        let ttl_ms = duration_ms(ttl)?;
        self.run(async move {
            this.declare().await?;
            Leases::new(&this.docs)
                .acquire(PARENT_GATE_OWNER, &parent_key, &holder, ttl_ms)
                .await
                .map(|lease| {
                    lease.map(|lease| ParentGateClaim {
                        holder: lease.holder,
                        epoch: lease.epoch,
                    })
                })
                .map_err(map_error)
        })
    }

    fn renew_parent_gate(
        &self,
        parent_key: &str,
        claim: &ParentGateClaim,
        ttl: Duration,
    ) -> Result<bool> {
        let this = self.clone();
        let parent_key = parent_key.to_owned();
        let claim = claim.clone();
        let ttl_ms = duration_ms(ttl)?;
        self.run(async move {
            this.declare().await?;
            Leases::new(&this.docs)
                .renew(
                    PARENT_GATE_OWNER,
                    &parent_key,
                    &claim.holder,
                    claim.epoch,
                    ttl_ms,
                )
                .await
                .map(|lease| lease.is_some())
                .map_err(map_error)
        })
    }

    fn release_parent_gate(&self, parent_key: &str, claim: &ParentGateClaim) -> Result<bool> {
        let this = self.clone();
        let parent_key = parent_key.to_owned();
        let claim = claim.clone();
        self.run(async move {
            this.declare().await?;
            Leases::new(&this.docs)
                .release(PARENT_GATE_OWNER, &parent_key, &claim.holder, claim.epoch)
                .await
                .map_err(map_error)
        })
    }

    fn get(&self, task_id: &str) -> Option<CompletionRecord> {
        let this = self.clone();
        let task_id = task_id.to_owned();
        self.run(async move {
            this.declare().await?;
            let id = task_id_key(&task_id).map_err(map_error)?;
            let found = this.docs.get(COMPLETIONS, &id).await.map_err(map_error)?;
            Ok(found.and_then(decode))
        })
        .inspect_err(|error| tracing::warn!(error = %error, "{LOG_PREFIX} read failed"))
        .ok()
        .flatten()
    }

    fn try_get(&self, task_id: &str) -> Result<Option<CompletionRecord>> {
        let this = self.clone();
        let task_id = task_id.to_owned();
        self.run(async move {
            this.declare().await?;
            let id = task_id_key(&task_id).map_err(map_error)?;
            let found = this.docs.get(COMPLETIONS, &id).await.map_err(map_error)?;
            found.map(decode_strict).transpose()
        })
    }

    fn put(&self, record: &CompletionRecord) -> Result<()> {
        let _ = record;
        Err(TinyAgentsError::Storage(
            "completion store: durable writes require a parent gate claim".into(),
        ))
    }

    fn put_fenced(
        &self,
        record: &CompletionRecord,
        parent_key: &str,
        gate: &ParentGateClaim,
    ) -> Result<()> {
        let this = self.clone();
        let record = record.clone();
        let parent_key = parent_key.to_owned();
        let gate = gate.clone();
        self.run(async move {
            this.declare().await?;
            let docs = this.fenced_docs(&parent_key, &gate).await?;
            let id = task_id_key(&record.task_id).map_err(map_error)?;
            let current = docs.get(COMPLETIONS, &id).await.map_err(map_error)?;
            let precondition = current.as_ref().map_or(Precondition::Absent, |row| {
                Precondition::Version(row.version)
            });
            docs.put_with(
                COMPLETIONS,
                &id,
                serde_json::to_value(CompletionDoc::from_record(&record))?,
                precondition,
                WriteOptions::new().schema(SCHEMA_VERSION),
            )
            .await
            .map(|_| ())
            .map_err(map_error)
        })
    }

    fn insert_if_absent(&self, record: &CompletionRecord, gate: &ParentGateClaim) -> Result<bool> {
        let this = self.clone();
        let record = record.clone();
        let gate = gate.clone();
        self.run(async move {
            this.declare().await?;
            let docs = this.fenced_docs(&record.parent_key, &gate).await?;
            let id = task_id_key(&record.task_id).map_err(map_error)?;
            let body = serde_json::to_value(CompletionDoc::from_record(&record))?;
            match docs
                .put_with(
                    COMPLETIONS,
                    &id,
                    body,
                    Precondition::Absent,
                    WriteOptions::new().schema(SCHEMA_VERSION),
                )
                .await
            {
                Ok(_) => Ok(true),
                Err(error) if error.kind() == ErrorKind::Conflict => Ok(false),
                Err(error) => Err(map_error(error)),
            }
        })
    }

    fn claim_pending(&self, request: PendingClaimRequest) -> Result<Vec<CompletionRecord>> {
        let this = self.clone();
        let PendingClaimRequest {
            parent_key,
            parent_epoch,
            gate,
            mode,
            max,
            token,
            visible_after,
        } = request;
        self.run(async move {
            this.declare().await?;
            let docs = this.fenced_docs(&parent_key, &gate).await?;
            let rows = docs
                .query_all(
                    COMPLETIONS,
                    &Query::filter(Filter::eq("parent_key", parent_key.as_str())),
                )
                .await
                .map_err(map_error)?;
            let now = SystemTime::now();
            let mut rows = rows
                .into_iter()
                .filter_map(|row| decode_strict(row.clone()).ok().map(|record| (row, record)))
                .filter(|(_, record)| {
                    record.parent_epoch == parent_epoch
                        && record.state == CompletionState::Pending
                        && mode.accepts(record.notify_mode)
                        && record
                            .claim
                            .as_ref()
                            .is_none_or(|claim| claim.visible_after <= now)
                })
                .collect::<Vec<_>>();
            rows.sort_by(|(_, left), (_, right)| {
                left.finished_at
                    .cmp(&right.finished_at)
                    .then_with(|| left.task_id.cmp(&right.task_id))
            });
            let mut claimed = Vec::new();
            for (row, mut record) in rows.into_iter().take(max) {
                record.attempts = record.attempts.saturating_add(1);
                record.updated_at = now;
                record.claim = Some(crate::completions::CompletionClaim {
                    token: format!("{token}:{}", record.task_id),
                    visible_after,
                });
                let body = serde_json::to_value(CompletionDoc::from_record(&record))?;
                match docs
                    .put_with(
                        COMPLETIONS,
                        &row.id,
                        body,
                        Precondition::Version(row.version),
                        WriteOptions::new().schema(SCHEMA_VERSION),
                    )
                    .await
                {
                    Ok(_) => claimed.push(record),
                    Err(error) if error.kind() == ErrorKind::Conflict => {}
                    Err(error) => return Err(map_error(error)),
                }
            }
            Ok(claimed)
        })
    }

    fn claim_task(&self, request: TaskClaimRequest) -> Result<Option<CompletionRecord>> {
        let this = self.clone();
        let TaskClaimRequest {
            task_id,
            parent_key,
            parent_epoch,
            gate,
            mode,
            token,
            visible_after,
        } = request;
        self.run(async move {
            this.declare().await?;
            let docs = this.fenced_docs(&parent_key, &gate).await?;
            let id = task_id_key(&task_id).map_err(map_error)?;
            let Some(row) = docs.get(COMPLETIONS, &id).await.map_err(map_error)? else {
                return Ok(None);
            };
            let mut record = decode_strict(row.clone())?;
            let now = SystemTime::now();
            if record.parent_key != parent_key
                || record.parent_epoch != parent_epoch
                || record.state != CompletionState::Pending
                || !mode.accepts(record.notify_mode)
                || record
                    .claim
                    .as_ref()
                    .is_some_and(|claim| claim.visible_after > now)
            {
                return Ok(None);
            }
            record.attempts = record.attempts.saturating_add(1);
            record.updated_at = now;
            record.claim = Some(crate::completions::CompletionClaim {
                token: format!("{token}:{task_id}"),
                visible_after,
            });
            let body = serde_json::to_value(CompletionDoc::from_record(&record))?;
            match docs
                .put_with(
                    COMPLETIONS,
                    &id,
                    body,
                    Precondition::Version(row.version),
                    WriteOptions::new().schema(SCHEMA_VERSION),
                )
                .await
            {
                Ok(_) => Ok(Some(record)),
                Err(error) if error.kind() == ErrorKind::Conflict => Ok(None),
                Err(error) => Err(map_error(error)),
            }
        })
    }

    fn resolve_claim(&self, request: ResolveClaimRequest) -> Result<Option<CompletionRecord>> {
        let this = self.clone();
        let ResolveClaimRequest {
            task_id,
            parent_key,
            parent_epoch,
            gate,
            token,
            resolution,
        } = request;
        self.run(async move {
            this.declare().await?;
            let docs = this.fenced_docs(&parent_key, &gate).await?;
            let id = task_id_key(&task_id).map_err(map_error)?;
            let Some(row) = docs.get(COMPLETIONS, &id).await.map_err(map_error)? else {
                return Ok(None);
            };
            let mut record = decode_strict(row.clone())?;
            let now = SystemTime::now();
            if record.parent_key != parent_key
                || record.parent_epoch != parent_epoch
                || record.state != CompletionState::Pending
                || record
                    .claim
                    .as_ref()
                    .is_none_or(|claim| claim.token != token || claim.visible_after <= now)
            {
                return Ok(None);
            }
            record.claim = None;
            record.updated_at = SystemTime::now();
            match resolution {
                ClaimResolution::Delivered => record.state = CompletionState::Delivered,
                ClaimResolution::Failed { max_attempts } if record.attempts >= max_attempts => {
                    record.state = CompletionState::GaveUp;
                }
                ClaimResolution::Failed { .. } | ClaimResolution::Release => {}
            }
            let body = serde_json::to_value(CompletionDoc::from_record(&record))?;
            match docs
                .put_with(
                    COMPLETIONS,
                    &id,
                    body,
                    Precondition::Version(row.version),
                    WriteOptions::new().schema(SCHEMA_VERSION),
                )
                .await
            {
                Ok(_) => Ok(Some(record)),
                Err(error) if error.kind() == ErrorKind::Conflict => Ok(None),
                Err(error) => Err(map_error(error)),
            }
        })
    }

    fn renew_claim(&self, request: RenewClaimRequest) -> Result<bool> {
        let this = self.clone();
        let RenewClaimRequest {
            task_id,
            parent_key,
            parent_epoch,
            gate,
            token,
            visible_after,
        } = request;
        self.run(async move {
            this.declare().await?;
            let docs = this.fenced_docs(&parent_key, &gate).await?;
            let id = task_id_key(&task_id).map_err(map_error)?;
            let Some(row) = docs.get(COMPLETIONS, &id).await.map_err(map_error)? else {
                return Ok(false);
            };
            let mut record = decode_strict(row.clone())?;
            if record.parent_key != parent_key
                || record.parent_epoch != parent_epoch
                || record.state != CompletionState::Pending
            {
                return Ok(false);
            }
            let Some(claim) = record.claim.as_mut().filter(|claim| claim.token == token) else {
                return Ok(false);
            };
            claim.visible_after = visible_after;
            let body = serde_json::to_value(CompletionDoc::from_record(&record))?;
            match docs
                .put_with(
                    COMPLETIONS,
                    &id,
                    body,
                    Precondition::Version(row.version),
                    WriteOptions::new().schema(SCHEMA_VERSION),
                )
                .await
            {
                Ok(_) => Ok(true),
                Err(error) if error.kind() == ErrorKind::Conflict => Ok(false),
                Err(error) => Err(map_error(error)),
            }
        })
    }

    fn list(&self, parent_key: Option<&str>) -> Vec<CompletionRecord> {
        let this = self.clone();
        let parent_key = parent_key.map(str::to_owned);
        self.run(async move {
            this.declare().await?;
            let filter = match &parent_key {
                Some(parent) => Filter::eq("parent_key", parent.as_str()),
                None => Filter::default(),
            };
            let rows = this
                .docs
                .query_all(COMPLETIONS, &Query::filter(filter))
                .await
                .map_err(map_error)?;
            Ok(rows.into_iter().filter_map(decode).collect())
        })
        .inspect_err(|error| tracing::warn!(error = %error, "{LOG_PREFIX} list failed"))
        .unwrap_or_default()
    }

    fn try_list(&self, parent_key: Option<&str>) -> Result<Vec<CompletionRecord>> {
        let this = self.clone();
        let parent_key = parent_key.map(str::to_owned);
        self.run(async move {
            this.declare().await?;
            let filter = match &parent_key {
                Some(parent) => Filter::eq("parent_key", parent.as_str()),
                None => Filter::default(),
            };
            let rows = this
                .docs
                .query_all(COMPLETIONS, &Query::filter(filter))
                .await
                .map_err(map_error)?;
            rows.into_iter().map(decode_strict).collect()
        })
    }

    fn compact(&self, retain: Duration) -> Result<usize> {
        let this = self.clone();
        let cutoff = to_ms(SystemTime::now())
            .saturating_sub(i64::try_from(retain.as_millis()).unwrap_or(i64::MAX));
        self.run(async move {
            this.declare().await?;
            // Settled means anything but pending, last written more than
            // `retain` ago: the document's own `updated_ms`.
            let filter = Filter::ne("state", serde_json::to_value(CompletionState::Pending)?)
                .and(Filter::lt(UPDATED_FIELD, cutoff));
            let rows = this
                .docs
                .query_all(COMPLETIONS, &Query::filter(filter))
                .await
                .map_err(map_error)?;
            let mut dropped = 0;
            for row in rows {
                let is_parent_marker = decode_strict(row.clone())
                    .is_ok_and(|record| record.task_id.starts_with("\u{1}parent-cancelled:"));
                if is_parent_marker {
                    continue;
                }
                // Only the version we saw: a record refreshed since is live again.
                match this
                    .docs
                    .delete(COMPLETIONS, &row.id, Precondition::Version(row.version))
                    .await
                {
                    Ok(removed) => dropped += usize::from(removed),
                    Err(error) if error.kind() == ErrorKind::Conflict => {}
                    Err(error) => return Err(map_error(error)),
                }
            }
            tracing::debug!(dropped, "{LOG_PREFIX} compacted");
            Ok(dropped)
        })
    }
}

fn duration_ms(duration: Duration) -> Result<i64> {
    i64::try_from(duration.as_millis()).map_err(|_| {
        TinyAgentsError::Storage("completion store: parent gate ttl out of range".into())
    })
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;
