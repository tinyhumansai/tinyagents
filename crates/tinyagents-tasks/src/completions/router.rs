//! The completion router.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::run_queue::{QueueLane, RunQueueHandle};

use super::format::{CompletionFormatter, NeutralCompletionFormatter};
use super::lifecycle::{ParentLifecycle, ParentLifecycleGuard, ParentLifecycleStatus};
use super::store::{
    ClaimResolution, CompletionClaimMode, CompletionStore, ParentGateClaim, PendingClaimRequest,
    RenewClaimRequest, ResolveClaimRequest, TaskClaimRequest,
};
use super::types::{CompletionReceipt, CompletionRecord, CompletionState, NotifyMode};

const LOG_PREFIX: &str = "[completion-router]";
const CLAIM_VISIBILITY: Duration = Duration::from_secs(30);
const PARENT_GATE_TTL: Duration = Duration::from_secs(60);
const PARENT_GATE_WAIT: Duration = Duration::from_secs(65);

/// Delivery attempts before a record gives up, unless overridden.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 5;

/// What [`CompletionRouter::record`] did with a completion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordOutcome {
    /// Stored. `lane` is set when it was pushed onto a live parent's queue; the
    /// record is then pending and leased until the host calls
    /// [`CompletionRouter::mark_delivered`].
    Recorded {
        /// The lane the completion was pushed onto, if any.
        lane: Option<QueueLane>,
        /// The receipt authorizing settlement of a live push.
        receipt: Option<CompletionReceipt>,
    },
    /// A record for this task id already exists; nothing changed.
    Duplicate,
    /// Dropped: the task was tombstoned, or its parent was cancelled.
    Suppressed,
    /// The task id is already recorded for a *different* parent. Completion ids
    /// are the dedupe key and must be unique across parents (the host's
    /// `sub-<uuid>` ids are); a host that reuses ids across roots must make them
    /// unique before recording. Nothing was stored.
    IdCollision,
}

/// What [`CompletionRouter::tombstone`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TombstoneOutcome {
    /// A pending completion was withdrawn.
    Suppressed,
    /// Nothing was recorded yet; a completion arriving later will be dropped.
    Reserved,
    /// The record was already delivered, given up, or tombstoned.
    AlreadyFinal,
}

/// Result of an identity-guarded backing-store migration.
#[derive(Clone)]
pub enum CompletionStoreReplaceOutcome {
    /// The replacement was copied and installed on this router.
    Replaced(Arc<dyn CompletionStore>),
    /// The expected store was stale; this is the router's current store.
    Current(Arc<dyn CompletionStore>),
}

struct RouterState {
    /// Store operations and in-process leases share this lock so a migration
    /// cannot detach a claim from the state that owns its acknowledgement.
    store: Arc<dyn CompletionStore>,
    /// Process-local lifecycle authority retained by the owning scope pool.
    lifecycle: Arc<ParentLifecycle>,
    /// Task ids handed out by a claim and not yet resolved. Process-local on
    /// purpose: after a restart nobody holds a lease, so everything still
    /// pending is claimable again.
    leased: HashSet<String>,
    /// Durable claim tokens owned by this router, used to reject stale settles.
    active_receipts: HashMap<String, CompletionReceipt>,
    /// Live parents and their steering queues.
    parents: HashMap<String, RunQueueHandle>,
}

/// Durable, deduplicating hand-off of finished children to their parents.
///
/// See the `completions` module docs for the division of labour with the host.
/// Router locks protect local bookkeeping; shared stores coordinate delivery
/// ownership through atomic claims, so independent routers may safely share a
/// backend that implements those claims.
pub struct CompletionRouter {
    formatter: Arc<dyn CompletionFormatter>,
    max_attempts: u32,
    claim_visibility: Duration,
    state: Mutex<RouterState>,
    admission_store: Arc<Mutex<AdmissionStoreSlot>>,
    lifecycle: Mutex<Arc<ParentLifecycle>>,
}

pub(super) struct AdmissionStoreSlot {
    pub(super) store: Arc<dyn CompletionStore>,
    pub(super) migrating: bool,
}

struct StoreMigration<'a> {
    slot: &'a Mutex<AdmissionStoreSlot>,
    replacement: Option<Arc<dyn CompletionStore>>,
}

impl StoreMigration<'_> {
    fn publish(mut self, replacement: Arc<dyn CompletionStore>) {
        self.replacement = Some(replacement);
    }
}

impl Drop for StoreMigration<'_> {
    fn drop(&mut self) {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(replacement) = self.replacement.take() {
            slot.store = replacement;
        }
        slot.migrating = false;
    }
}

struct ParentGateGuard {
    store: Arc<dyn CompletionStore>,
    _local: Option<ParentLifecycleGuard>,
    parent_key: String,
    claim: ParentGateClaim,
}

/// Short-lived durable authority held while a queued completion crosses into
/// its parent turn. Dropping it releases the parent lifecycle gate.
pub struct ParentAdmissionPermit {
    _guard: ParentGateGuard,
}

impl tinyagents_harness::run_queue::QueueAdmissionPermit for ParentAdmissionPermit {}

impl std::fmt::Debug for ParentAdmissionPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParentAdmissionPermit")
            .finish_non_exhaustive()
    }
}

impl ParentAdmissionPermit {
    pub(crate) fn acquire(
        store: Arc<dyn CompletionStore>,
        lifecycle: Arc<ParentLifecycle>,
        parent_key: &str,
        parent_epoch: u64,
    ) -> Result<Option<Self>> {
        let gate = ParentGateGuard::acquire(Arc::clone(&store), lifecycle.clone(), parent_key)?;
        let current =
            CompletionRouter::parent_lifecycle_status(&store, &lifecycle, gate.local_guard())?;
        let admitted = current == Some(ParentLifecycleStatus::Open(parent_epoch));
        Ok(admitted.then_some(Self { _guard: gate }))
    }
}

impl ParentGateGuard {
    pub(crate) fn acquire(
        store: Arc<dyn CompletionStore>,
        lifecycle: Arc<ParentLifecycle>,
        parent_key: &str,
    ) -> Result<Self> {
        let mut gate = Self::acquire_native(store, parent_key)?;
        gate._local = Some(lifecycle.acquire(parent_key)?);
        if !gate.renew()? {
            return Err(TinyAgentsError::Graph(format!(
                "completion router: parent gate expired while waiting for local lifecycle `{parent_key}`"
            )));
        }
        Ok(gate)
    }

    fn acquire_native(store: Arc<dyn CompletionStore>, parent_key: &str) -> Result<Self> {
        let holder = uuid::Uuid::new_v4().to_string();
        let started = std::time::Instant::now();
        loop {
            if let Some(claim) = store.acquire_parent_gate(parent_key, &holder, PARENT_GATE_TTL)? {
                return Ok(Self {
                    store,
                    _local: None,
                    parent_key: parent_key.to_owned(),
                    claim,
                });
            }
            if started.elapsed() >= PARENT_GATE_WAIT {
                return Err(TinyAgentsError::Graph(format!(
                    "completion store: timed out waiting for parent gate {parent_key}"
                )));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn renew(&self) -> Result<bool> {
        self.store
            .renew_parent_gate(&self.parent_key, &self.claim, PARENT_GATE_TTL)
    }

    fn claim(&self) -> &ParentGateClaim {
        &self.claim
    }

    pub(super) fn local_guard(&self) -> &ParentLifecycleGuard {
        self._local
            .as_ref()
            .expect("router gates retain local lifecycle")
    }
}

impl Drop for ParentGateGuard {
    fn drop(&mut self) {
        if let Err(error) = self
            .store
            .release_parent_gate(&self.parent_key, &self.claim)
        {
            tracing::warn!(parent_key = %self.parent_key, error = %error, "{LOG_PREFIX} parent gate release failed");
        }
    }
}

impl CompletionRouter {
    /// A router over `store` with the neutral formatter and
    /// [`DEFAULT_MAX_ATTEMPTS`].
    pub fn new(store: Arc<dyn CompletionStore>) -> Self {
        let lifecycle = if store.is_process_local() {
            store
                .parent_lifecycle()
                .unwrap_or_else(|| ParentLifecycle::for_store(store.as_ref()))
        } else {
            ParentLifecycle::new()
        };
        let admission_store = Arc::new(Mutex::new(AdmissionStoreSlot {
            store: Arc::clone(&store),
            migrating: false,
        }));
        Self {
            formatter: Arc::new(NeutralCompletionFormatter),
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            claim_visibility: CLAIM_VISIBILITY,
            state: Mutex::new(RouterState {
                store,
                lifecycle: Arc::clone(&lifecycle),
                leased: HashSet::new(),
                active_receipts: HashMap::new(),
                parents: HashMap::new(),
            }),
            admission_store,
            lifecycle: Mutex::new(lifecycle),
        }
    }

    pub(super) fn attach_parent_lifecycle(&self, lifecycle: Arc<ParentLifecycle>) -> Result<()> {
        let mut state = self.lock()?;
        if state.store.is_process_local() {
            state.store.bind_parent_lifecycle(Arc::clone(&lifecycle))?;
        }
        state.lifecycle = Arc::clone(&lifecycle);
        *self
            .lifecycle
            .lock()
            .map_err(|_| TinyAgentsError::Graph("router lifecycle lock poisoned".into()))? =
            lifecycle;
        Ok(())
    }

    pub(super) fn parent_lifecycle_handle(&self) -> Arc<ParentLifecycle> {
        Arc::clone(
            &self
                .lifecycle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    pub(super) fn parent_lifecycle_status(
        store: &Arc<dyn CompletionStore>,
        lifecycle: &Arc<ParentLifecycle>,
        gate: &ParentLifecycleGuard,
    ) -> Result<Option<ParentLifecycleStatus>> {
        let parent_key = gate.parent_key();
        let marker = store.try_get(&Self::parent_marker_id(parent_key))?;
        let marker_status = marker.map(|marker| match marker.state {
            CompletionState::Delivered => ParentLifecycleStatus::Open(marker.parent_epoch),
            CompletionState::ParentDeleted => ParentLifecycleStatus::Deleted(marker.parent_epoch),
            _ => ParentLifecycleStatus::Stopped(marker.parent_epoch),
        });

        if store.is_process_local() {
            let cached = lifecycle.status(parent_key)?;
            let status = match (marker_status, cached) {
                (Some(ParentLifecycleStatus::Deleted(marker_epoch)), snapshot) => {
                    Some(ParentLifecycleStatus::Deleted(
                        marker_epoch.max(snapshot.map_or(0, status_epoch)),
                    ))
                }
                (Some(marker), Some(ParentLifecycleStatus::Deleted(snapshot_epoch))) => Some(
                    ParentLifecycleStatus::Deleted(snapshot_epoch.max(status_epoch(marker))),
                ),
                (None, Some(ParentLifecycleStatus::Deleted(epoch))) => {
                    Some(ParentLifecycleStatus::Deleted(epoch))
                }
                (Some(marker), Some(snapshot)) => {
                    Some(if status_epoch(marker) >= status_epoch(snapshot) {
                        marker
                    } else {
                        snapshot
                    })
                }
                (Some(marker), None) => Some(marker),
                (None, snapshot) => snapshot,
            };
            if let Some(status) = status {
                lifecycle.set_status(gate, status)?;
            }
            return Ok(status);
        }
        if let Some(status) = marker_status {
            lifecycle.set_status(gate, status)?;
            return Ok(Some(status));
        }
        lifecycle.status(parent_key)
    }

    pub(super) fn admission_store_handle(&self) -> Arc<Mutex<AdmissionStoreSlot>> {
        Arc::clone(&self.admission_store)
    }

    pub(super) fn admission_store_snapshot(
        slot: &Mutex<AdmissionStoreSlot>,
    ) -> std::result::Result<
        Arc<dyn CompletionStore>,
        tinyagents_harness::run_queue::QueueAdmissionError,
    > {
        let slot = slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.migrating {
            Err(tinyagents_harness::run_queue::QueueAdmissionError::Retry)
        } else {
            Ok(Arc::clone(&slot.store))
        }
    }

    pub(super) fn admission_store_is_current(
        slot: &Mutex<AdmissionStoreSlot>,
        store: &Arc<dyn CompletionStore>,
    ) -> std::result::Result<(), tinyagents_harness::run_queue::QueueAdmissionError> {
        let slot = slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.migrating || !Arc::ptr_eq(&slot.store, store) {
            Err(tinyagents_harness::run_queue::QueueAdmissionError::Retry)
        } else {
            Ok(())
        }
    }

    /// Returns the persistence store backing this router.
    pub fn store(&self) -> Arc<dyn CompletionStore> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .store
            .clone()
    }

    /// Atomically migrates this router to `replacement`, preserving its
    /// in-flight leases, parent queues, formatter, and attempt policy. The
    /// current store is enumerated and copied before publication; on any
    /// source-read or target-write error the old store remains canonical. The
    /// replacement must be private to this router while the copy is made.
    pub fn replace_store_if_current(
        &self,
        expected: &Arc<dyn CompletionStore>,
        replacement: Arc<dyn CompletionStore>,
    ) -> Result<CompletionStoreReplaceOutcome> {
        let mut state = self.lock()?;
        if !Arc::ptr_eq(&state.store, expected) {
            return Ok(CompletionStoreReplaceOutcome::Current(Arc::clone(
                &state.store,
            )));
        }
        if Arc::ptr_eq(&state.store, &replacement) {
            return Ok(CompletionStoreReplaceOutcome::Current(Arc::clone(
                &state.store,
            )));
        }
        // Stop new queue admissions without holding a slot lock while waiting
        // on durable parent gates. Existing permits finish first; admissions
        // racing the migration fail closed and remain recoverable in the store.
        {
            let mut slot = self
                .admission_store
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.migrating = true;
        }
        let migration = StoreMigration {
            slot: &self.admission_store,
            replacement: None,
        };
        if replacement.is_process_local() {
            replacement.bind_parent_lifecycle(Arc::clone(&state.lifecycle))?;
        }
        // The first snapshot discovers which parent gates must be acquired.
        // It is deliberately not copied: another router may close or update a
        // parent while this migration is waiting for those gates.
        let discovered = state.store.try_list(None)?;
        let mut parents = discovered
            .iter()
            .map(|record| {
                if record.parent_key.is_empty() {
                    Self::parent_from_marker_id(&record.task_id)
                        .unwrap_or_else(|| record.task_id.clone())
                } else {
                    record.parent_key.clone()
                }
            })
            .collect::<Vec<_>>();
        parents.sort();
        parents.dedup();
        let _source_gates = parents
            .iter()
            .map(|parent| ParentGateGuard::acquire_native(Arc::clone(&state.store), parent))
            .collect::<Result<Vec<_>>>()?;
        let _local_gates = parents
            .iter()
            .map(|parent| state.lifecycle.acquire(parent))
            .collect::<Result<Vec<_>>>()?;
        for gate in &_source_gates {
            if !gate.renew()? {
                return Err(TinyAgentsError::Graph(
                    "completion store: source parent gate expired while waiting for local lifecycle".into(),
                ));
            }
        }
        let records = state.store.try_list(None)?;
        let gated_parents = parents.iter().cloned().collect::<HashSet<_>>();
        if records.iter().any(|record| {
            let parent = if record.parent_key.is_empty() {
                Self::parent_from_marker_id(&record.task_id)
                    .unwrap_or_else(|| record.task_id.clone())
            } else {
                record.parent_key.clone()
            };
            !gated_parents.contains(&parent)
        }) {
            return Err(TinyAgentsError::Graph(
                "completion store: migration snapshot gained a parent while acquiring gates; retry migration".into(),
            ));
        }
        for record in records {
            let parent_key = if record.parent_key.is_empty() {
                Self::parent_from_marker_id(&record.task_id)
                    .unwrap_or_else(|| record.task_id.clone())
            } else {
                record.parent_key.clone()
            };
            let gate = ParentGateGuard::acquire_native(Arc::clone(&replacement), &parent_key)?;
            replacement.put_fenced(&record, &parent_key, gate.claim())?;
        }
        state.store = Arc::clone(&replacement);
        migration.publish(Arc::clone(&replacement));
        Ok(CompletionStoreReplaceOutcome::Replaced(replacement))
    }

    /// Replaces the formatter (the host's own wording).
    pub fn with_formatter(mut self, formatter: Arc<dyn CompletionFormatter>) -> Self {
        self.formatter = formatter;
        self
    }

    /// Sets how many failed attempts a record survives. At least one.
    pub fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    /// Sets how long a delivery claim lasts before another consumer may take
    /// the record (30 s by default). [`Self::renew_claims`] extends it.
    pub fn with_claim_visibility(mut self, visibility: Duration) -> Self {
        self.claim_visibility = visibility;
        self
    }

    /// How long a delivery claim lasts.
    pub fn claim_visibility(&self) -> Duration {
        self.claim_visibility
    }

    /// The formatter in use.
    pub fn formatter(&self) -> &Arc<dyn CompletionFormatter> {
        &self.formatter
    }

    /// The attempt limit.
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    fn lock(&self) -> Result<MutexGuard<'_, RouterState>> {
        self.state
            .lock()
            .map_err(|_| TinyAgentsError::Graph("completion router lock poisoned".into()))
    }

    /// Marks `parent_key` live: [`NotifyMode::Followup`] and
    /// [`NotifyMode::Collect`] completions for it are pushed onto `queue`.
    pub fn attach_parent(&self, parent_key: impl Into<String>, queue: RunQueueHandle) {
        let parent_key = parent_key.into();
        tracing::debug!(parent_key = %parent_key, "{LOG_PREFIX} parent attached");
        if let Ok(mut state) = self.lock() {
            state.parents.insert(parent_key, queue);
        }
    }

    /// Marks `parent_key` no longer live. Later completions stay pending.
    ///
    /// The parent's queue is presumed gone, so leases on its in-flight pushes
    /// are released and those records can be claimed again. Acknowledge what the
    /// parent already received with [`Self::mark_delivered`] first.
    pub fn detach_parent(&self, parent_key: &str) {
        tracing::debug!(parent_key = %parent_key, "{LOG_PREFIX} parent detached");
        if let Ok(mut state) = self.lock() {
            state.parents.remove(parent_key);
            self.release_parent_locked(&mut state, parent_key);
        }
    }

    fn release_parent_locked(&self, state: &mut RouterState, parent_key: &str) {
        let Ok(gate) = ParentGateGuard::acquire(
            Arc::clone(&state.store),
            Arc::clone(&state.lifecycle),
            parent_key,
        ) else {
            return;
        };
        for record in state.store.list(Some(parent_key)) {
            state.leased.remove(&record.task_id);
            if let Some(receipt) = state.active_receipts.remove(&record.task_id) {
                let _ = state.store.resolve_claim(ResolveClaimRequest {
                    task_id: record.task_id,
                    parent_key: parent_key.to_owned(),
                    parent_epoch: receipt.parent_epoch,
                    gate: gate.claim().clone(),
                    token: receipt.token,
                    resolution: ClaimResolution::Release,
                });
            }
        }
    }

    /// Releases the lease on `task_ids` without counting a failure, so they can
    /// be claimed again. Use it when a claim or push was abandoned (a dropped
    /// delivery future, a cleared queue). Attempts already counted stay counted.
    pub fn release(&self, receipts: &[CompletionReceipt]) {
        if let Ok(mut state) = self.lock() {
            for receipt in receipts {
                if state.active_receipts.get(&receipt.task_id) == Some(receipt) {
                    state.leased.remove(&receipt.task_id);
                    state.active_receipts.remove(&receipt.task_id);
                }
                let Ok(gate) = ParentGateGuard::acquire(
                    Arc::clone(&state.store),
                    Arc::clone(&state.lifecycle),
                    &receipt.parent_key,
                ) else {
                    continue;
                };
                let _ = state.store.resolve_claim(ResolveClaimRequest {
                    task_id: receipt.task_id.clone(),
                    parent_key: receipt.parent_key.clone(),
                    parent_epoch: receipt.parent_epoch,
                    gate: gate.claim().clone(),
                    token: receipt.token.clone(),
                    resolution: ClaimResolution::Release,
                });
            }
        }
    }

    /// Releases every receipt currently owned by this router so another
    /// router can retry the pending records immediately. Store failures leave
    /// the affected receipt locally owned and are returned to the caller.
    pub fn release_all(&self) -> Result<usize> {
        let mut state = self.lock()?;
        let receipts: Vec<_> = state.active_receipts.values().cloned().collect();
        let mut released = 0;
        for receipt in receipts {
            let gate = ParentGateGuard::acquire(
                Arc::clone(&state.store),
                Arc::clone(&state.lifecycle),
                &receipt.parent_key,
            )?;
            let resolved = state.store.resolve_claim(ResolveClaimRequest {
                task_id: receipt.task_id.clone(),
                parent_key: receipt.parent_key.clone(),
                parent_epoch: receipt.parent_epoch,
                gate: gate.claim().clone(),
                token: receipt.token.clone(),
                resolution: ClaimResolution::Release,
            })?;
            if state.active_receipts.get(&receipt.task_id) == Some(&receipt) {
                state.leased.remove(&receipt.task_id);
                state.active_receipts.remove(&receipt.task_id);
            }
            if resolved.is_some() {
                released += 1;
            }
        }
        Ok(released)
    }

    /// Records pushed onto `parent_key`'s live queue (or claimed) and not yet
    /// acknowledged, oldest first.
    pub fn in_flight_for(&self, parent_key: &str) -> Vec<CompletionRecord> {
        let Ok(state) = self.lock() else {
            return Vec::new();
        };
        let mut records: Vec<_> = state
            .store
            .list(Some(parent_key))
            .into_iter()
            .filter(|r| r.state == CompletionState::Pending && state.leased.contains(&r.task_id))
            .collect();
        sort_oldest_first(&mut records);
        records
    }

    fn classify_existing(
        existing: &CompletionRecord,
        incoming: &CompletionRecord,
    ) -> RecordOutcome {
        if existing.state == CompletionState::Tombstoned {
            tracing::debug!(task_id = %incoming.task_id, "{LOG_PREFIX} dropped: tombstoned");
            RecordOutcome::Suppressed
        } else if !existing.parent_key.is_empty() && existing.parent_key != incoming.parent_key {
            tracing::warn!(task_id = %incoming.task_id, "{LOG_PREFIX} task id already recorded for another parent");
            RecordOutcome::IdCollision
        } else {
            tracing::debug!(task_id = %incoming.task_id, state = existing.state.as_str(), "{LOG_PREFIX} duplicate completion ignored");
            RecordOutcome::Duplicate
        }
    }

    /// Records a finished child, idempotent per task id.
    ///
    /// A new record is stored `Pending`. If its [`NotifyMode`] is `Followup` or
    /// `Collect` and the parent is attached, it is also pushed onto that lane,
    /// counted as attempt one and leased. The queue is in memory and has no
    /// acknowledgement, so the record settles only when the host calls
    /// [`Self::mark_delivered`] after the parent has the message; if the queue
    /// is lost first ([`Self::detach_parent`], [`Self::release`], or a restart)
    /// the record is claimable again. Delivery is at-least-once.
    pub async fn record(&self, mut record: CompletionRecord) -> Result<RecordOutcome> {
        let (push, gate) = {
            let mut state = self.lock()?;
            let gate = ParentGateGuard::acquire(
                Arc::clone(&state.store),
                Arc::clone(&state.lifecycle),
                &record.parent_key,
            )?;
            let marker_id = Self::parent_marker_id(&record.parent_key);
            let marker_before = state.store.try_get(&marker_id)?;
            let Some(parent_epoch) = Self::admit_child_locked(&state, &record.parent_key, &gate)?
            else {
                tracing::debug!(
                    task_id = %record.task_id,
                    parent_key = %record.parent_key,
                    "{LOG_PREFIX} dropped: parent is closed"
                );
                return Ok(RecordOutcome::Suppressed);
            };
            if let Some(existing) = state.store.get(&record.task_id) {
                return Ok(Self::classify_existing(&existing, &record));
            }
            if (record.parent_epoch != 0 && record.parent_epoch != parent_epoch)
                || (record.parent_epoch == 0
                    && marker_before.is_some_and(|marker| marker.parent_epoch > 1))
            {
                tracing::debug!(
                    task_id = %record.task_id,
                    parent_key = %record.parent_key,
                    supplied_epoch = record.parent_epoch,
                    current_epoch = parent_epoch,
                    "{LOG_PREFIX} suppressed completion from a stale or unversioned child"
                );
                return Ok(RecordOutcome::Suppressed);
            }
            record.state = CompletionState::Pending;
            record.parent_epoch = parent_epoch;
            record.attempts = 0;
            record.updated_at = SystemTime::now();
            if !state.store.insert_if_absent(&record, gate.claim())? {
                let existing = state.store.try_get(&record.task_id)?.ok_or_else(|| {
                    TinyAgentsError::Graph(format!(
                        "completion store: conditional create conflicted but `{}` was not readable",
                        record.task_id
                    ))
                })?;
                return Ok(Self::classify_existing(&existing, &record));
            }
            tracing::debug!(
                task_id = %record.task_id,
                parent_key = %record.parent_key,
                status = record.status.as_str(),
                notify_mode = record.notify_mode.as_str(),
                "{LOG_PREFIX} recorded"
            );
            let lane = match record.notify_mode {
                NotifyMode::Followup => Some(QueueLane::Followup),
                NotifyMode::Collect => Some(QueueLane::Collect),
                NotifyMode::HoldForNextTurn | NotifyMode::Off => None,
            };
            // A live push is a delivery attempt, not a delivery: the record
            // stays pending and leased until the host acknowledges it with
            // `mark_delivered`, so a crash, a cleared queue or a detached parent
            // never loses it (it is redelivered through `claim_pending`).
            let push = match lane.and_then(|lane| {
                state
                    .parents
                    .get(&record.parent_key)
                    .map(|queue| (lane, queue.clone()))
            }) {
                Some((lane, queue)) => {
                    let token = uuid::Uuid::new_v4().to_string();
                    let visible_after = SystemTime::now() + self.claim_visibility;
                    let Some(claimed) = state.store.claim_task(TaskClaimRequest {
                        task_id: record.task_id.clone(),
                        parent_key: record.parent_key.clone(),
                        parent_epoch: record.parent_epoch,
                        gate: gate.claim().clone(),
                        mode: CompletionClaimMode::Followup,
                        token,
                        visible_after,
                    })?
                    else {
                        return Ok(RecordOutcome::Recorded {
                            lane: None,
                            receipt: None,
                        });
                    };
                    record = claimed;
                    let receipt = record.receipt().expect("claimed record has a receipt");
                    state.leased.insert(record.task_id.clone());
                    state
                        .active_receipts
                        .insert(record.task_id.clone(), receipt.clone());
                    Some((lane, queue, receipt))
                }
                None => None,
            };
            (push, gate)
        };
        let Some((lane, queue, push_receipt)) = push else {
            return Ok(RecordOutcome::Recorded {
                lane: None,
                receipt: None,
            });
        };
        if !gate.renew()? {
            self.release(std::slice::from_ref(&push_receipt));
            return Err(TinyAgentsError::Graph(format!(
                "completion router: parent admission gate lost for `{}`",
                record.parent_key
            )));
        }
        let message = self.formatter.to_message(std::slice::from_ref(&record));
        let admission = Arc::new(super::CompletionAdmission::for_router(self, &record));
        queue.push_guarded(lane, message, admission).await;
        tracing::debug!(
            task_id = %record.task_id,
            parent_key = %record.parent_key,
            lane = lane.as_str(),
            "{LOG_PREFIX} pushed to live parent"
        );
        Ok(RecordOutcome::Recorded {
            lane: Some(lane),
            receipt: Some(push_receipt),
        })
    }

    /// [`Self::record`], retried up to `attempts` times with a short backoff
    /// when the store fails. Safe because `record` is idempotent per task id
    /// (a retry after a write that actually landed reports `Duplicate`).
    pub async fn record_with_retries(
        &self,
        record: CompletionRecord,
        attempts: u32,
    ) -> Result<RecordOutcome> {
        let attempts = attempts.max(1);
        let mut attempt = 1;
        loop {
            match self.record(record.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(error) if attempt >= attempts => return Err(error),
                Err(error) => {
                    tracing::warn!(
                        task_id = %record.task_id,
                        attempt,
                        error = %error,
                        "{LOG_PREFIX} record failed, retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(50 * u64::from(attempt))).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Records with retries and, only when the caller explicitly permits a
    /// process-local scope, migrates this router to memory after durable
    /// writes fail. Existing records must be readable and copyable before the
    /// replacement is published; otherwise the durable store remains active
    /// and the migration error is returned.
    pub async fn record_resilient(
        &self,
        record: CompletionRecord,
        attempts: u32,
        fallback: super::registry::CompletionStoreFallback,
    ) -> Result<RecordOutcome> {
        match self.record_with_retries(record.clone(), attempts).await {
            Ok(outcome) => Ok(outcome),
            Err(error) if fallback == super::registry::CompletionStoreFallback::FailClosed => {
                Err(error)
            }
            Err(_) => {
                let current = self.store();
                let replacement: Arc<dyn CompletionStore> =
                    Arc::new(super::InMemoryCompletionStore::new());
                self.replace_store_if_current(&current, replacement)?;
                self.record_with_retries(record, attempts).await
            }
        }
    }

    /// Withdraws a child's completion because the parent collected it itself
    /// (waited on it, or read its result). Suppresses routing from now on; a
    /// message already pushed onto a live queue is not retracted. A completion that has not arrived
    /// yet is dropped when it does.
    pub fn tombstone(&self, task_id: &str) -> Result<TombstoneOutcome> {
        let mut state = self.lock()?;
        let outcome = match state.store.get(task_id) {
            Some(mut existing) if existing.state == CompletionState::Pending => {
                let gate_key = existing.parent_key.clone();
                let gate = ParentGateGuard::acquire(
                    Arc::clone(&state.store),
                    Arc::clone(&state.lifecycle),
                    &gate_key,
                )?;
                existing.state = CompletionState::Tombstoned;
                existing.updated_at = SystemTime::now();
                state.store.put_fenced(&existing, &gate_key, gate.claim())?;
                state.leased.remove(task_id);
                TombstoneOutcome::Suppressed
            }
            Some(_) => TombstoneOutcome::AlreadyFinal,
            None => {
                let gate_key = format!("completion-task:{task_id}");
                let gate = ParentGateGuard::acquire(
                    Arc::clone(&state.store),
                    Arc::clone(&state.lifecycle),
                    &gate_key,
                )?;
                state.store.put_fenced(
                    &CompletionRecord::tombstone_stub(task_id),
                    &gate_key,
                    gate.claim(),
                )?;
                TombstoneOutcome::Reserved
            }
        };
        tracing::debug!(task_id = %task_id, outcome = ?outcome, "{LOG_PREFIX} tombstone");
        Ok(outcome)
    }

    /// Drops every pending completion for `parent_key` and everything that
    /// finishes for it later, until [`Self::resume_parent`]. For a deleted or
    /// stopped parent. A message already pushed onto the parent's queue is not
    /// retracted; queued generations must be rejected when the parent turn is
    /// admitted. The flag is durable and survives completion compaction and
    /// process restarts. Returns how many pending records were withdrawn.
    pub fn cancel_parent(&self, parent_key: &str) -> Result<usize> {
        self.close_parent(parent_key, CompletionState::Tombstoned)
    }

    /// Permanently closes a deleted parent thread. The durable marker is
    /// retained through completion compaction so late children remain dropped
    /// after a restart.
    pub fn delete_parent(&self, parent_key: &str) -> Result<usize> {
        self.close_parent(parent_key, CompletionState::ParentDeleted)
    }

    fn close_parent(&self, parent_key: &str, marker_state: CompletionState) -> Result<usize> {
        let mut state = self.lock()?;
        let gate = ParentGateGuard::acquire(
            Arc::clone(&state.store),
            Arc::clone(&state.lifecycle),
            parent_key,
        )?;
        let current =
            Self::parent_lifecycle_status(&state.store, &state.lifecycle, gate.local_guard())?;
        if marker_state != CompletionState::ParentDeleted
            && matches!(current, Some(ParentLifecycleStatus::Deleted(_)))
        {
            state.parents.remove(parent_key);
            return Ok(0);
        }
        if marker_state == CompletionState::ParentDeleted
            && matches!(current, Some(ParentLifecycleStatus::Deleted(_)))
        {
            state.parents.remove(parent_key);
            return Ok(0);
        }
        Self::set_parent_marker(&state, parent_key, marker_state, &gate)?;
        state.parents.remove(parent_key);
        let mut withdrawn = 0;
        for mut record in state.store.try_list(Some(parent_key))? {
            if record.state == CompletionState::Pending {
                record.state = CompletionState::Tombstoned;
                record.updated_at = SystemTime::now();
                state.store.put_fenced(&record, parent_key, gate.claim())?;
                state.leased.remove(&record.task_id);
                withdrawn += 1;
            }
        }
        tracing::debug!(parent_key = %parent_key, withdrawn, "{LOG_PREFIX} parent cancelled");
        Ok(withdrawn)
    }

    /// Reopens a parent stopped with [`Self::cancel_parent`]. Calling this on
    /// an already open parent is idempotent. A parent closed with
    /// [`Self::delete_parent`] cannot be reopened.
    pub fn resume_parent(&self, parent_key: &str) {
        let result = self.lock().and_then(|state| {
            let gate = ParentGateGuard::acquire(
                Arc::clone(&state.store),
                Arc::clone(&state.lifecycle),
                parent_key,
            )?;
            match Self::parent_lifecycle_status(&state.store, &state.lifecycle, gate.local_guard())?
            {
                Some(ParentLifecycleStatus::Open(_)) => Ok(()),
                Some(ParentLifecycleStatus::Deleted(_)) => Err(TinyAgentsError::Graph(format!(
                    "completion router: deleted parent {parent_key:?} cannot be resumed"
                ))),
                Some(ParentLifecycleStatus::Stopped(_)) | None => {
                    Self::set_parent_marker(&state, parent_key, CompletionState::Delivered, &gate)
                }
            }
        });
        if let Err(error) = result {
            tracing::warn!(parent_key = %parent_key, error = %error, "{LOG_PREFIX} could not resume parent");
        }
    }

    /// The cancelled-parent flag lives in the store as a marker record, so it
    /// survives a restart with the rest of the log. `Tombstoned` means
    /// cancelled; `Delivered` means resumed.
    fn parent_marker_id(parent_key: &str) -> String {
        format!("\u{1}parent-cancelled:{parent_key}")
    }

    fn parent_from_marker_id(task_id: &str) -> Option<String> {
        task_id
            .strip_prefix("\u{1}parent-cancelled:")
            .map(str::to_owned)
    }

    /// Opens a parent on first sight, or reports whether its durable lifecycle
    /// marker currently admits child completions. Closed parents are never
    /// reopened implicitly.
    pub fn admit_child(&self, parent_key: &str) -> Result<bool> {
        Ok(self.admit_child_generation(parent_key)?.is_some())
    }

    /// Admits a child into the currently open durable lifecycle and returns
    /// the generation it must carry when it later records its completion.
    /// The generation changes only at an explicit parent stop/delete/resume,
    /// not at ordinary router or lease handoff.
    pub fn admit_child_generation(&self, parent_key: &str) -> Result<Option<u64>> {
        let state = self.lock()?;
        let gate = ParentGateGuard::acquire(
            Arc::clone(&state.store),
            Arc::clone(&state.lifecycle),
            parent_key,
        )?;
        Self::admit_child_locked(&state, parent_key, &gate)
    }

    /// Whether a generation carried by a queued completion still belongs to
    /// the currently open parent lifecycle. Hosts use this at turn admission
    /// to discard queue entries that raced with stop, delete, or reopen.
    pub fn parent_generation_admitted(&self, parent_key: &str, parent_epoch: u64) -> Result<bool> {
        let state = self.lock()?;
        let gate = ParentGateGuard::acquire(
            Arc::clone(&state.store),
            Arc::clone(&state.lifecycle),
            parent_key,
        )?;
        Ok(
            Self::parent_lifecycle_status(&state.store, &state.lifecycle, gate.local_guard())?
                == Some(ParentLifecycleStatus::Open(parent_epoch)),
        )
    }

    fn admit_child_locked(
        router: &RouterState,
        parent_key: &str,
        gate: &ParentGateGuard,
    ) -> Result<Option<u64>> {
        let marker_id = Self::parent_marker_id(parent_key);
        match Self::parent_lifecycle_status(&router.store, &router.lifecycle, gate.local_guard())? {
            Some(ParentLifecycleStatus::Open(epoch)) => {
                if !router.store.is_process_local() || router.store.try_get(&marker_id)?.is_none() {
                    Self::write_parent_marker(
                        router,
                        parent_key,
                        ParentLifecycleStatus::Open(epoch),
                        gate,
                    )?;
                }
                Ok(Some(epoch))
            }
            Some(ParentLifecycleStatus::Stopped(_)) | Some(ParentLifecycleStatus::Deleted(_)) => {
                if !router.store.is_process_local() && router.store.try_get(&marker_id)?.is_none() {
                    let status = router.lifecycle.status(parent_key)?.ok_or_else(|| {
                        TinyAgentsError::Graph(
                            "completion router: lifecycle marker disappeared".into(),
                        )
                    })?;
                    Self::write_parent_marker(router, parent_key, status, gate)?;
                }
                Ok(None)
            }
            None => {
                // Records written before lifecycle markers existed carry
                // generation zero. First sight admits those durable records
                // into the new open generation under the same parent gate.
                for mut legacy in router.store.try_list(Some(parent_key))? {
                    if legacy.parent_epoch == 0 {
                        legacy.parent_epoch = 1;
                        router.store.put_fenced(&legacy, parent_key, gate.claim())?;
                    }
                }
                Self::set_parent_marker(router, parent_key, CompletionState::Delivered, gate)?;
                let marker = router.store.try_get(&marker_id)?.ok_or_else(|| {
                    TinyAgentsError::Graph(
                        "completion router: parent marker disappeared after admission".into(),
                    )
                })?;
                Ok(Some(marker.parent_epoch))
            }
        }
    }

    fn set_parent_marker(
        router: &RouterState,
        parent_key: &str,
        state: CompletionState,
        gate: &ParentGateGuard,
    ) -> Result<()> {
        let mut marker = CompletionRecord::tombstone_stub(&Self::parent_marker_id(parent_key));
        marker.parent_key = parent_key.to_owned();
        marker.state = state;
        let current = router.store.try_get(&marker.task_id)?;
        let local = router.lifecycle.status(parent_key)?;
        let epoch = if router.store.is_process_local() {
            local
                .map(status_epoch)
                .or_else(|| current.map(|record| record.parent_epoch))
        } else {
            current
                .map(|record| record.parent_epoch)
                .or_else(|| local.map(status_epoch))
        };
        marker.parent_epoch = epoch.map_or(1, |epoch| epoch.saturating_add(1));
        router.store.put_fenced(&marker, parent_key, gate.claim())?;
        router.lifecycle.set_status(
            gate.local_guard(),
            status_from_marker(state, marker.parent_epoch),
        )
    }

    fn write_parent_marker(
        router: &RouterState,
        parent_key: &str,
        status: ParentLifecycleStatus,
        gate: &ParentGateGuard,
    ) -> Result<()> {
        let mut marker = CompletionRecord::tombstone_stub(&Self::parent_marker_id(parent_key));
        marker.parent_key = parent_key.to_owned();
        marker.parent_epoch = status_epoch(status);
        marker.state = marker_state(status);
        router.store.put_fenced(&marker, parent_key, gate.claim())
    }

    /// Every undelivered completion for `parent_key`, oldest first, in every
    /// [`NotifyMode`] and whether or not a claim currently holds it. Read-only:
    /// this is what surfaces after a restart.
    pub fn pending_for(&self, parent_key: &str) -> Vec<CompletionRecord> {
        let Ok(state) = self.lock() else {
            return Vec::new();
        };
        let mut pending: Vec<_> = state
            .store
            .list(Some(parent_key))
            .into_iter()
            .filter(|r| r.state == CompletionState::Pending)
            .collect();
        sort_oldest_first(&mut pending);
        pending
    }

    /// Parent keys with at least one pending completion in this store's scope.
    /// Tenant isolation is provided by the scoped completion store.
    pub fn parents_with_pending(&self) -> Result<Vec<String>> {
        let state = self.lock()?;
        Ok(state
            .store
            .try_list(None)?
            .into_iter()
            .filter(|record| record.state == CompletionState::Pending)
            .map(|record| record.parent_key)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect())
    }

    /// Closes every parent with pending completions and returns the number of
    /// records withdrawn. Each marker uses the same durable fenced path as
    /// [`Self::cancel_parent`].
    pub fn cancel_all_parents(&self) -> Result<usize> {
        let (mut parents, records) = {
            let state = self.lock()?;
            let records = state.store.try_list(None)?;
            let mut parents = state
                .parents
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>();
            for record in &records {
                if let Some(parent) = Self::parent_from_marker_id(&record.task_id) {
                    if record.state == CompletionState::Delivered {
                        parents.insert(parent);
                    }
                } else if record.state == CompletionState::Pending && !record.parent_key.is_empty()
                {
                    parents.insert(record.parent_key.clone());
                }
            }
            (parents, records)
        };
        // Pending rows may predate lifecycle markers, so retain their parent
        // keys even when they have no open marker yet.
        parents.extend(
            records
                .into_iter()
                .filter(|record| record.state == CompletionState::Pending)
                .map(|record| record.parent_key),
        );
        parents.into_iter().try_fold(0usize, |total, parent| {
            self.cancel_parent(&parent)
                .map(|withdrawn| total + withdrawn)
        })
    }

    /// Claims up to `max` `Followup`/`Collect` completions for the idle-time
    /// batch, oldest first, counting an attempt on each. A claimed record is
    /// leased until [`Self::mark_delivered`] or [`Self::mark_failed`].
    pub fn claim_pending(&self, parent_key: &str, max: usize) -> Result<Vec<CompletionRecord>> {
        self.claim(parent_key, max, CompletionClaimMode::Followup)
    }

    /// Claims the `HoldForNextTurn` completions for the turn that is starting.
    pub fn begin_turn(&self, parent_key: &str) -> Result<Vec<CompletionRecord>> {
        self.claim(parent_key, usize::MAX, CompletionClaimMode::HoldForNextTurn)
    }

    /// Claims up to `max` `Off` completions the parent asked to pull.
    pub fn pull(&self, parent_key: &str, max: usize) -> Result<Vec<CompletionRecord>> {
        self.claim(parent_key, max, CompletionClaimMode::Pull)
    }

    fn claim(
        &self,
        parent_key: &str,
        max: usize,
        mode: CompletionClaimMode,
    ) -> Result<Vec<CompletionRecord>> {
        let mut state = self.lock()?;
        let _gate = ParentGateGuard::acquire(
            Arc::clone(&state.store),
            Arc::clone(&state.lifecycle),
            parent_key,
        )?;
        let Some(parent_epoch) = Self::admit_child_locked(&state, parent_key, &_gate)? else {
            return Ok(Vec::new());
        };
        let token = uuid::Uuid::new_v4().to_string();
        let visible_after = SystemTime::now() + self.claim_visibility;
        let mut candidates = state.store.claim_pending(PendingClaimRequest {
            parent_key: parent_key.to_owned(),
            parent_epoch,
            gate: _gate.claim().clone(),
            mode,
            max,
            token,
            visible_after,
        })?;
        sort_oldest_first(&mut candidates);
        for record in &candidates {
            state.leased.insert(record.task_id.clone());
            if let Some(receipt) = record.receipt() {
                state
                    .active_receipts
                    .insert(record.task_id.clone(), receipt);
            }
        }
        if !candidates.is_empty() {
            tracing::debug!(
                parent_key = %parent_key,
                claimed = candidates.len(),
                "{LOG_PREFIX} claimed"
            );
        }
        Ok(candidates)
    }

    /// Settles claimed (or pulled) records as delivered. Returns how many
    /// changed; records that are not pending are left alone.
    pub fn mark_delivered(&self, receipts: &[CompletionReceipt]) -> Result<usize> {
        let mut state = self.lock()?;
        let mut changed = 0;
        for receipt in receipts {
            let gate = ParentGateGuard::acquire(
                Arc::clone(&state.store),
                Arc::clone(&state.lifecycle),
                &receipt.parent_key,
            )?;
            let resolved = state.store.resolve_claim(ResolveClaimRequest {
                task_id: receipt.task_id.clone(),
                parent_key: receipt.parent_key.clone(),
                parent_epoch: receipt.parent_epoch,
                gate: gate.claim().clone(),
                token: receipt.token.clone(),
                resolution: ClaimResolution::Delivered,
            })?;
            if state.active_receipts.get(&receipt.task_id) == Some(receipt) {
                state.active_receipts.remove(&receipt.task_id);
                state.leased.remove(&receipt.task_id);
            }
            if resolved.is_some() {
                changed += 1;
            }
        }
        tracing::debug!(changed, "{LOG_PREFIX} delivered");
        Ok(changed)
    }

    /// Extends claimed records for another [`Self::claim_visibility`] while a
    /// delivery is still running. Returns how many claims were still owned;
    /// a reclaimed or settled record is left alone.
    pub fn renew_claims(&self, receipts: &[CompletionReceipt]) -> Result<usize> {
        let state = self.lock()?;
        let visible_after = SystemTime::now() + self.claim_visibility;
        let mut renewed = 0;
        for receipt in receipts {
            let gate = ParentGateGuard::acquire(
                Arc::clone(&state.store),
                Arc::clone(&state.lifecycle),
                &receipt.parent_key,
            )?;
            if state.store.renew_claim(RenewClaimRequest {
                task_id: receipt.task_id.clone(),
                parent_key: receipt.parent_key.clone(),
                parent_epoch: receipt.parent_epoch,
                gate: gate.claim().clone(),
                token: receipt.token.clone(),
                visible_after,
            })? {
                renewed += 1;
            }
        }
        Ok(renewed)
    }

    /// Reports a failed delivery. The lease is released so the record can be
    /// claimed again, unless its attempts have reached [`Self::max_attempts`]:
    /// then it becomes [`CompletionState::GaveUp`] and is returned so the host
    /// can apply its own give-up policy (the router never delivers it again).
    /// A failure resolving one receipt does not hide give-ups already resolved
    /// in this batch; the failed receipt can be retried after its store claim
    /// expires.
    pub fn mark_failed(&self, receipts: &[CompletionReceipt]) -> Result<Vec<CompletionRecord>> {
        let mut state = self.lock()?;
        let mut gave_up = Vec::new();
        for receipt in receipts {
            let resolved = ParentGateGuard::acquire(
                Arc::clone(&state.store),
                Arc::clone(&state.lifecycle),
                &receipt.parent_key,
            )
            .and_then(|gate| {
                state.store.resolve_claim(ResolveClaimRequest {
                    task_id: receipt.task_id.clone(),
                    parent_key: receipt.parent_key.clone(),
                    parent_epoch: receipt.parent_epoch,
                    gate: gate.claim().clone(),
                    token: receipt.token.clone(),
                    resolution: ClaimResolution::Failed {
                        max_attempts: self.max_attempts,
                    },
                })
            });
            if state.active_receipts.get(&receipt.task_id) == Some(receipt) {
                state.active_receipts.remove(&receipt.task_id);
                state.leased.remove(&receipt.task_id);
            }
            match resolved {
                Ok(Some(record)) if record.state == CompletionState::GaveUp => {
                    tracing::warn!(task_id = %record.task_id, parent_key = %record.parent_key, attempts = record.attempts, "{LOG_PREFIX} gave up");
                    gave_up.push(record);
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(task_id = %receipt.task_id, %error, "{LOG_PREFIX} could not resolve failed delivery");
                }
            }
        }
        Ok(gave_up)
    }

    /// Drops settled records older than `retain` from the store (and, for the
    /// JSONL store, the log). See [`CompletionStore::compact`].
    pub fn compact(&self, retain: Duration) -> Result<usize> {
        let state = self.lock()?;
        let dropped = state.store.compact(retain)?;
        tracing::debug!(dropped, "{LOG_PREFIX} compacted");
        Ok(dropped)
    }
}

fn sort_oldest_first(records: &mut [CompletionRecord]) {
    records.sort_by(|a, b| {
        a.finished_at
            .cmp(&b.finished_at)
            .then_with(|| a.task_id.cmp(&b.task_id))
    });
}

fn status_epoch(status: ParentLifecycleStatus) -> u64 {
    match status {
        ParentLifecycleStatus::Open(epoch)
        | ParentLifecycleStatus::Stopped(epoch)
        | ParentLifecycleStatus::Deleted(epoch) => epoch,
    }
}

fn marker_state(status: ParentLifecycleStatus) -> CompletionState {
    match status {
        ParentLifecycleStatus::Open(_) => CompletionState::Delivered,
        ParentLifecycleStatus::Stopped(_) => CompletionState::Tombstoned,
        ParentLifecycleStatus::Deleted(_) => CompletionState::ParentDeleted,
    }
}

fn status_from_marker(state: CompletionState, epoch: u64) -> ParentLifecycleStatus {
    match state {
        CompletionState::Delivered => ParentLifecycleStatus::Open(epoch),
        CompletionState::ParentDeleted => ParentLifecycleStatus::Deleted(epoch),
        _ => ParentLifecycleStatus::Stopped(epoch),
    }
}
