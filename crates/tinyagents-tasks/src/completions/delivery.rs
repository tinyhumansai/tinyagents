//! Host-driven delivery timing around durable completion claims.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::{CompletionReceipt, CompletionRecord, CompletionRouter};

/// Timing and batch bounds for one delivery drain.
#[derive(Clone, Copy, Debug)]
pub struct CompletionDeliveryConfig {
    /// Wait for a burst of completions to settle before claiming a batch.
    pub debounce: Duration,
    /// Delay recovered completions after boot while the host opens resources.
    pub recovery_delay: Duration,
    /// Delay between failed delivery turns.
    pub retry_backoff: Duration,
    /// Poll interval while the parent is busy.
    pub busy_poll: Duration,
    /// Maximum busy polls before leaving records pending.
    pub max_busy_polls: u32,
    /// Maximum completions in one delivery turn.
    pub batch_size: usize,
}

impl Default for CompletionDeliveryConfig {
    fn default() -> Self {
        Self {
            debounce: Duration::from_secs(3),
            recovery_delay: Duration::from_secs(15),
            retry_backoff: Duration::from_secs(2),
            busy_poll: Duration::from_secs(5),
            max_busy_polls: 360,
            batch_size: 32,
        }
    }
}

/// Why a drain ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryOutcome {
    /// There were no eligible completions.
    Empty,
    /// The delivery turn was acknowledged.
    Delivered { count: usize },
    /// Delivery failed and the records remain retryable or gave up.
    Failed { count: usize, gave_up: usize },
    /// Another drain already owns the tenant-scoped thread delivery lease.
    LeaseHeld,
    /// The parent remained busy beyond the configured polling window.
    ParentBusy,
    /// The host's ownership fence stopped this drain.
    Fenced,
}

/// Parent and thread selected for one delivery drain.
#[derive(Clone, Copy)]
pub struct DeliveryTarget<'a> {
    /// Durable completion router for the owning scope.
    pub router: &'a CompletionRouter,
    /// Stable key recorded on child completions.
    pub parent_key: &'a str,
    /// Tenant/profile namespace for thread-level coordination.
    pub tenant_key: &'a str,
    /// Thread key used for one-delivery-at-a-time coordination.
    pub thread_key: &'a str,
}

/// Timing policy owned by the host's background worker; delivery state lives
/// in [`CompletionRouter`]. Callbacks are seams for host policy and execution.
pub struct CompletionDelivery {
    config: CompletionDeliveryConfig,
    thread_leases: ThreadDeliveryLeases,
}

impl CompletionDelivery {
    /// Creates a delivery drain with explicit timing bounds.
    pub fn new(config: CompletionDeliveryConfig, thread_leases: ThreadDeliveryLeases) -> Self {
        Self {
            config,
            thread_leases,
        }
    }

    /// Delivers one batch, retrying failed turns with backoff up to the
    /// router's attempt limit. The host supplies busy/fence checks, the actual
    /// parent turn, and a durable writer for records which give up.
    pub async fn deliver_once<B, F, D, Fut, E, U>(
        &self,
        target: DeliveryTarget<'_>,
        is_busy: B,
        admitted: F,
        deliver: D,
        mut write_undelivered: U,
    ) -> DeliveryOutcome
    where
        B: FnMut() -> bool,
        F: DeliveryFence,
        D: FnMut(&[CompletionRecord]) -> Fut,
        Fut: Future<Output = std::result::Result<(), E>>,
        E: std::fmt::Display,
        U: FnMut(&[CompletionRecord]),
    {
        self.deliver_once_with_error(target, is_busy, admitted, deliver, move |records, _| {
            write_undelivered(records)
        })
        .await
    }

    /// Delivers one batch with a call-specific initial delay. The override
    /// applies only to this drain; recovery and later calls keep their own
    /// configured delays.
    pub async fn deliver_once_with_initial_delay<B, F, D, Fut, E, U>(
        &self,
        target: DeliveryTarget<'_>,
        initial_delay: Duration,
        is_busy: B,
        admitted: F,
        deliver: D,
        write_undelivered: U,
    ) -> DeliveryOutcome
    where
        B: FnMut() -> bool,
        F: DeliveryFence,
        D: FnMut(&[CompletionRecord]) -> Fut,
        Fut: Future<Output = std::result::Result<(), E>>,
        E: std::fmt::Display,
        U: FnMut(&[CompletionRecord]),
    {
        let mut config = self.config;
        config.debounce = initial_delay;
        Self::new(config, self.thread_leases.clone())
            .deliver_once(target, is_busy, admitted, deliver, write_undelivered)
            .await
    }

    /// Error-aware form of [`Self::deliver_once`]. The callback receives the
    /// final delivery error for records that give up, synchronously under the
    /// thread lease after the durable give-up transition.
    pub async fn deliver_once_with_error<B, F, D, Fut, E, U>(
        &self,
        target: DeliveryTarget<'_>,
        mut is_busy: B,
        admitted: F,
        mut deliver: D,
        mut write_undelivered: U,
    ) -> DeliveryOutcome
    where
        B: FnMut() -> bool,
        F: DeliveryFence,
        D: FnMut(&[CompletionRecord]) -> Fut,
        Fut: Future<Output = std::result::Result<(), E>>,
        E: std::fmt::Display,
        U: FnMut(&[CompletionRecord], &str),
    {
        let Some(_lease) = self
            .thread_leases
            .try_acquire(TenantThreadKey::new(target.tenant_key, target.thread_key))
        else {
            return DeliveryOutcome::LeaseHeld;
        };
        self.drain(
            target,
            self.config.debounce,
            &mut is_busy,
            &admitted,
            &mut deliver,
            &mut write_undelivered,
        )
        .await
    }

    /// Delivers records recovered from durable storage after process startup.
    /// It uses the same claim/retry path as a live completion and is fenced
    /// before any wait or claim.
    pub async fn recover_on_boot<B, F, D, Fut, E, U>(
        &self,
        target: DeliveryTarget<'_>,
        is_busy: B,
        admitted: F,
        deliver: D,
        mut write_undelivered: U,
    ) -> DeliveryOutcome
    where
        B: FnMut() -> bool,
        F: DeliveryFence,
        D: FnMut(&[CompletionRecord]) -> Fut,
        Fut: Future<Output = std::result::Result<(), E>>,
        E: std::fmt::Display,
        U: FnMut(&[CompletionRecord]),
    {
        self.recover_on_boot_with_error(target, is_busy, admitted, deliver, move |records, _| {
            write_undelivered(records)
        })
        .await
    }

    /// Error-aware form of [`Self::recover_on_boot`]. The callback receives
    /// the final error for records that give up while the thread lease is held.
    pub async fn recover_on_boot_with_error<B, F, D, Fut, E, U>(
        &self,
        target: DeliveryTarget<'_>,
        mut is_busy: B,
        admitted: F,
        mut deliver: D,
        mut write_undelivered: U,
    ) -> DeliveryOutcome
    where
        B: FnMut() -> bool,
        F: DeliveryFence,
        D: FnMut(&[CompletionRecord]) -> Fut,
        Fut: Future<Output = std::result::Result<(), E>>,
        E: std::fmt::Display,
        U: FnMut(&[CompletionRecord], &str),
    {
        let Some(_lease) = self
            .thread_leases
            .try_acquire(TenantThreadKey::new(target.tenant_key, target.thread_key))
        else {
            return DeliveryOutcome::LeaseHeld;
        };
        self.drain(
            target,
            self.config.recovery_delay,
            &mut is_busy,
            &admitted,
            &mut deliver,
            &mut write_undelivered,
        )
        .await
    }

    async fn drain<B, F, D, Fut, E, U>(
        &self,
        target: DeliveryTarget<'_>,
        initial_delay: Duration,
        is_busy: &mut B,
        admitted: &F,
        deliver: &mut D,
        write_undelivered: &mut U,
    ) -> DeliveryOutcome
    where
        B: FnMut() -> bool,
        F: DeliveryFence,
        D: FnMut(&[CompletionRecord]) -> Fut,
        Fut: Future<Output = std::result::Result<(), E>>,
        E: std::fmt::Display,
        U: FnMut(&[CompletionRecord], &str),
    {
        if !sleep_if_admitted(initial_delay, admitted).await {
            return DeliveryOutcome::Fenced;
        }
        for _ in 0..self.config.max_busy_polls {
            if !admitted.admitted() {
                return DeliveryOutcome::Fenced;
            }
            if !is_busy() {
                break;
            }
            if !sleep_if_admitted(self.config.busy_poll, admitted).await {
                return DeliveryOutcome::Fenced;
            }
        }
        if !admitted.admitted() {
            return DeliveryOutcome::Fenced;
        }
        if is_busy() {
            return DeliveryOutcome::ParentBusy;
        }

        let mut batch = match target
            .router
            .claim_pending(target.parent_key, self.config.batch_size)
        {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(parent_key = target.parent_key, error = %error, "completion claim failed");
                return DeliveryOutcome::Failed {
                    count: 0,
                    gave_up: 0,
                };
            }
        };
        if batch.is_empty() {
            return DeliveryOutcome::Empty;
        }
        let mut claim = BatchClaim::new(target.router, &batch);
        let mut delivered_total = 0;
        let mut gave_up_total = 0;

        loop {
            if !admitted.admitted() {
                return DeliveryOutcome::Fenced;
            }
            let delivery_result = renewing(target.router, &batch, deliver(&batch)).await;
            if !admitted.admitted() {
                return DeliveryOutcome::Fenced;
            }
            match delivery_result {
                Ok(()) => {
                    let receipts: Vec<_> =
                        batch.iter().filter_map(CompletionRecord::receipt).collect();
                    match target.router.mark_delivered(&receipts) {
                        Ok(settled) => {
                            claim.clear();
                            delivered_total += settled;
                        }
                        Err(error) => {
                            tracing::warn!(parent_key = target.parent_key, error = %error, "completion acknowledgement failed");
                            return DeliveryOutcome::Failed {
                                count: delivered_total + batch.len(),
                                gave_up: gave_up_total,
                            };
                        }
                    }
                    if !admitted.admitted() {
                        return DeliveryOutcome::Fenced;
                    }
                    batch = match target
                        .router
                        .claim_pending(target.parent_key, self.config.batch_size)
                    {
                        Ok(batch) if !batch.is_empty() => {
                            claim.replace(&batch);
                            batch
                        }
                        Ok(_) => {
                            return if delivered_total > 0 {
                                DeliveryOutcome::Delivered {
                                    count: delivered_total,
                                }
                            } else {
                                DeliveryOutcome::Empty
                            };
                        }
                        Err(error) => {
                            tracing::warn!(parent_key = target.parent_key, error = %error, "completion claim failed");
                            return DeliveryOutcome::Failed {
                                count: delivered_total,
                                gave_up: gave_up_total,
                            };
                        }
                    };
                }
                Err(error) => {
                    tracing::warn!(parent_key = target.parent_key, error = %error, "completion delivery turn failed");
                    let receipts: Vec<_> =
                        batch.iter().filter_map(CompletionRecord::receipt).collect();
                    let gave_up = match target.router.mark_failed(&receipts) {
                        Ok(records) => records,
                        Err(error) => {
                            tracing::warn!(parent_key = target.parent_key, error = %error, "completion failure update failed");
                            return DeliveryOutcome::Failed {
                                count: delivered_total + batch.len(),
                                gave_up: gave_up_total,
                            };
                        }
                    };
                    claim.clear();
                    gave_up_total += gave_up.len();
                    if !gave_up.is_empty() {
                        (*write_undelivered)(&gave_up, &error.to_string());
                    }
                    if gave_up.len() == batch.len() || !admitted.admitted() {
                        return DeliveryOutcome::Failed {
                            count: delivered_total + batch.len(),
                            gave_up: gave_up_total,
                        };
                    }
                    if !sleep_if_admitted(self.config.retry_backoff, admitted).await {
                        return DeliveryOutcome::Failed {
                            count: delivered_total + batch.len(),
                            gave_up: gave_up_total,
                        };
                    }
                    batch = match target
                        .router
                        .claim_pending(target.parent_key, self.config.batch_size)
                    {
                        Ok(batch) if !batch.is_empty() => batch,
                        Ok(_) => {
                            return DeliveryOutcome::Failed {
                                count: delivered_total + batch.len(),
                                gave_up: gave_up_total,
                            };
                        }
                        Err(error) => {
                            tracing::warn!(parent_key = target.parent_key, error = %error, "completion retry claim failed");
                            return DeliveryOutcome::Failed {
                                count: delivered_total + batch.len(),
                                gave_up: gave_up_total,
                            };
                        }
                    };
                    claim.replace(&batch);
                }
            }
        }
    }
}

/// Runs one delivery turn, renewing the batch's claims every half visibility,
/// so a turn that outlasts one claim is neither reclaimed nor repeated.
async fn renewing<T>(
    router: &CompletionRouter,
    batch: &[CompletionRecord],
    turn: impl Future<Output = T>,
) -> T {
    let receipts: Vec<_> = batch.iter().filter_map(CompletionRecord::receipt).collect();
    let every = (router.claim_visibility() / 2).max(Duration::from_millis(1));
    tokio::pin!(turn);
    loop {
        tokio::select! {
            result = &mut turn => return result,
            () = tokio::time::sleep(every) => {
                if let Err(error) = router.renew_claims(&receipts) {
                    tracing::warn!(error = %error, "completion claim renewal failed");
                }
            }
        }
    }
}

async fn sleep_if_admitted<F: DeliveryFence>(duration: Duration, admitted: &F) -> bool {
    if !admitted.admitted() {
        return false;
    }
    if duration.is_zero() {
        return admitted.admitted();
    }
    tokio::select! {
        _ = tokio::time::sleep(duration) => admitted.admitted(),
        _ = admitted.lost() => false,
    }
}

struct BatchClaim<'a> {
    router: &'a CompletionRouter,
    receipts: Vec<CompletionReceipt>,
}
impl<'a> BatchClaim<'a> {
    fn new(router: &'a CompletionRouter, records: &[CompletionRecord]) -> Self {
        Self {
            router,
            receipts: records
                .iter()
                .filter_map(CompletionRecord::receipt)
                .collect(),
        }
    }
    fn replace(&mut self, records: &[CompletionRecord]) {
        self.receipts = records
            .iter()
            .filter_map(CompletionRecord::receipt)
            .collect();
    }
    fn clear(&mut self) {
        self.receipts.clear();
    }
}
impl Drop for BatchClaim<'_> {
    fn drop(&mut self) {
        self.router.release(&self.receipts);
    }
}

/// Tenant-scoped identity used by parent and thread delivery gates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TenantThreadKey {
    tenant: String,
    key: String,
}

impl TenantThreadKey {
    /// Names a thread or session within a tenant/profile.
    pub fn new(tenant: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            tenant: tenant.into(),
            key: key.into(),
        }
    }

    /// Tenant/profile component.
    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    /// Thread or session component.
    pub fn key(&self) -> &str {
        &self.key
    }
}

impl From<String> for TenantThreadKey {
    fn from(key: String) -> Self {
        Self::new("", key)
    }
}

impl From<&str> for TenantThreadKey {
    fn from(key: &str) -> Self {
        Self::new("", key)
    }
}

#[derive(Default)]
struct ParentBusyState {
    leases: HashSet<(TenantThreadKey, u64)>,
    /// Maps an active session id to the corresponding stable thread id.
    session_threads: HashMap<TenantThreadKey, TenantThreadKey>,
}

/// Parent turn activity gate. Dropping a lease clears only the guard it owns.
#[derive(Clone, Default)]
pub struct ParentBusyGate(Arc<Mutex<ParentBusyState>>);

/// RAII guard marking one parent turn busy.
pub struct ParentBusyLease {
    gate: ParentBusyGate,
    key: TenantThreadKey,
    id: u64,
}

impl ParentBusyGate {
    /// Marks one user turn busy. Overlapping leases keep the parent busy until
    /// every guard is dropped; `None` indicates a poisoned lock.
    pub fn enter(&self, parent_key: impl Into<TenantThreadKey>) -> Option<ParentBusyLease> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let key = parent_key.into();
        let mut entries = self.0.lock().ok()?;
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        entries.leases.insert((key.clone(), id));
        Some(ParentBusyLease {
            gate: self.clone(),
            key,
            id,
        })
    }

    /// Whether a parent has an active busy lease.
    pub fn is_busy(&self, parent_key: &str) -> bool {
        self.is_busy_scoped(&TenantThreadKey::from(parent_key))
    }

    /// Maps a session alias to its stable thread key for busy lookups.
    pub fn alias_session_to_thread(
        &self,
        session: TenantThreadKey,
        thread: TenantThreadKey,
    ) -> bool {
        if session.tenant != thread.tenant {
            return false;
        }
        self.0
            .lock()
            .is_ok_and(|mut state| state.session_threads.insert(session, thread).is_none())
    }

    /// Whether a tenant-scoped thread or one of its session aliases is busy.
    pub fn is_busy_scoped(&self, key: &TenantThreadKey) -> bool {
        self.0.lock().is_ok_and(|state| {
            state.leases.iter().any(|(busy, _)| {
                busy == key
                    || state.session_threads.get(busy) == Some(key)
                    || state.session_threads.get(key) == Some(busy)
            })
        })
    }

    /// Clears active leases for a parent and removes its session aliases.
    pub fn clear(&self, parent_key: &TenantThreadKey) -> usize {
        let Ok(mut state) = self.0.lock() else {
            return 0;
        };
        let mut related = HashSet::from([parent_key.clone()]);
        for (session, thread) in &state.session_threads {
            if session == parent_key || thread == parent_key {
                related.insert(session.clone());
                related.insert(thread.clone());
            }
        }
        state
            .session_threads
            .retain(|session, thread| !related.contains(session) && !related.contains(thread));
        let before = state.leases.len();
        state.leases.retain(|(key, _)| !related.contains(key));
        before - state.leases.len()
    }
}

impl Drop for ParentBusyLease {
    fn drop(&mut self) {
        if let Ok(mut entries) = self.gate.0.lock() {
            entries.leases.remove(&(self.key.clone(), self.id));
        }
    }
}

/// At-most-one live delivery turn per thread, released automatically on drop.
#[derive(Clone, Default)]
pub struct ThreadDeliveryLeases(Arc<Mutex<HashMap<TenantThreadKey, u64>>>);

/// RAII exclusive delivery lease for one thread.
pub struct ThreadDeliveryLease {
    leases: ThreadDeliveryLeases,
    key: TenantThreadKey,
    id: u64,
}

impl ThreadDeliveryLeases {
    /// Acquires the per-thread delivery slot, if currently free.
    pub fn try_acquire(
        &self,
        thread_key: impl Into<TenantThreadKey>,
    ) -> Option<ThreadDeliveryLease> {
        let key = thread_key.into();
        let mut leases = self.0.lock().ok()?;
        if leases.contains_key(&key) {
            return None;
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        leases.insert(key.clone(), id);
        Some(ThreadDeliveryLease {
            leases: self.clone(),
            key,
            id,
        })
    }

    /// Releases every process-local delivery lease and returns the count.
    /// Existing guards become inert and cannot remove a subsequently acquired
    /// lease for the same thread when they are dropped.
    pub fn release_all(&self) -> usize {
        self.0.lock().map_or(0, |mut leases| {
            let count = leases.len();
            leases.clear();
            count
        })
    }
}

impl Drop for ThreadDeliveryLease {
    fn drop(&mut self) {
        if let Ok(mut leases) = self.leases.0.lock()
            && leases.get(&self.key) == Some(&self.id)
        {
            leases.remove(&self.key);
        }
    }
}

/// Host fence check. Kept as a closure in the drain API so lease/fencing policy
/// stays outside the provider-neutral task crate.
pub trait DeliveryFence: Send + Sync {
    /// Whether this worker still owns the right to deliver.
    fn admitted(&self) -> bool;
    /// Completes when this worker loses its ownership fence. Implementations
    /// with no notification channel may keep the default pending future.
    fn lost(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(std::future::pending())
    }
}

impl<F> DeliveryFence for F
where
    F: Fn() -> bool + Send + Sync,
{
    fn admitted(&self) -> bool {
        self()
    }
}
