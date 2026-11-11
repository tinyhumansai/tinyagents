use super::*;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

use crate::completions::{
    CompletionRecord, CompletionResult, CompletionStatus, DeliveryTarget, InMemoryCompletionStore,
};

fn target(router: &CompletionRouter) -> DeliveryTarget<'_> {
    DeliveryTarget {
        router,
        parent_key: "parent",
        tenant_key: "",
        thread_key: "thread",
    }
}

#[tokio::test]
async fn delivery_claims_a_batch_and_acknowledges_it() {
    let router = CompletionRouter::new(Arc::new(InMemoryCompletionStore::new()));
    router
        .record(CompletionRecord::new(
            "a",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::text("A"),
        ))
        .await
        .unwrap();
    router
        .record(CompletionRecord::new(
            "b",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::text("B"),
        ))
        .await
        .unwrap();
    let leases = ThreadDeliveryLeases::default();
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        leases.clone(),
    );
    let mut delivered = Vec::new();
    let outcome = delivery
        .deliver_once(
            target(&router),
            || false,
            || true,
            |batch| {
                delivered = batch.iter().map(|record| record.task_id.clone()).collect();
                async { Ok::<(), &'static str>(()) }
            },
            |_| {},
        )
        .await;
    assert_eq!(outcome, DeliveryOutcome::Delivered { count: 2 });
    assert_eq!(delivered.len(), 2);
    assert!(router.pending_for("parent").is_empty());
}

#[tokio::test]
async fn failed_delivery_releases_for_retry_and_reports_give_up() {
    let router =
        CompletionRouter::new(Arc::new(InMemoryCompletionStore::new())).with_max_attempts(1);
    router
        .record(CompletionRecord::new(
            "a",
            "parent",
            "worker",
            CompletionStatus::Failed,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        ThreadDeliveryLeases::default(),
    );
    let mut gave_up = Vec::new();
    let outcome = delivery
        .deliver_once(
            target(&router),
            || false,
            || true,
            |_| async { Err::<(), _>("turn failed") },
            |records| gave_up = records.to_vec(),
        )
        .await;
    assert_eq!(
        outcome,
        DeliveryOutcome::Failed {
            count: 1,
            gave_up: 1
        }
    );
    assert_eq!(gave_up[0].task_id, "a");
}

#[tokio::test]
async fn error_aware_delivery_reports_the_final_error_while_holding_the_lease() {
    let router =
        CompletionRouter::new(Arc::new(InMemoryCompletionStore::new())).with_max_attempts(2);
    router
        .record(CompletionRecord::new(
            "retry-final-error",
            "parent",
            "worker",
            CompletionStatus::Failed,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let leases = ThreadDeliveryLeases::default();
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        leases.clone(),
    );
    let mut attempts = 0;
    let mut notification = None;
    let outcome = delivery
        .deliver_once_with_error(
            target(&router),
            || false,
            || true,
            |_| {
                attempts += 1;
                let error = if attempts == 1 {
                    "transient"
                } else {
                    "final reason"
                };
                async move { Err::<(), _>(error) }
            },
            |records, reason| {
                assert_eq!(records.len(), 1);
                assert_eq!(records[0].task_id, "retry-final-error");
                assert_eq!(reason, "final reason");
                assert!(leases.try_acquire("thread").is_none());
                notification = Some(reason.to_owned());
            },
        )
        .await;

    assert_eq!(attempts, 2);
    assert_eq!(notification.as_deref(), Some("final reason"));
    assert!(matches!(
        outcome,
        DeliveryOutcome::Failed { gave_up: 1, .. }
    ));
}

#[tokio::test]
async fn error_aware_boot_recovery_reports_the_last_failure() {
    let router =
        CompletionRouter::new(Arc::new(InMemoryCompletionStore::new())).with_max_attempts(1);
    router
        .record(CompletionRecord::new(
            "boot-final-error",
            "parent",
            "worker",
            CompletionStatus::Failed,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        ThreadDeliveryLeases::default(),
    );
    let mut reason = None;
    let outcome = delivery
        .recover_on_boot_with_error(
            target(&router),
            || false,
            || true,
            |_| async { Err::<(), _>("boot delivery failed") },
            |records, error| {
                assert_eq!(records[0].task_id, "boot-final-error");
                reason = Some(error.to_owned());
            },
        )
        .await;

    assert!(matches!(
        outcome,
        DeliveryOutcome::Failed { gave_up: 1, .. }
    ));
    assert_eq!(reason.as_deref(), Some("boot delivery failed"));
}

#[test]
fn busy_gate_and_delivery_lease_are_raii_and_exclusive() {
    let busy = ParentBusyGate::default();
    let first = busy.enter("parent").unwrap();
    let second = busy.enter("parent").unwrap();
    assert!(busy.is_busy("parent"));
    drop(first);
    assert!(busy.is_busy("parent"));
    drop(second);
    assert!(!busy.is_busy("parent"));

    let leases = ThreadDeliveryLeases::default();
    let lease = leases.try_acquire("thread").unwrap();
    assert!(leases.try_acquire("thread").is_none());
    drop(lease);
    assert!(leases.try_acquire("thread").is_some());
}

#[tokio::test]
async fn recovery_is_fenced_before_claiming_records() {
    let router = CompletionRouter::new(Arc::new(InMemoryCompletionStore::new()));
    router
        .record(CompletionRecord::new(
            "a",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        ThreadDeliveryLeases::default(),
    );
    let outcome = delivery
        .recover_on_boot(
            target(&router),
            || false,
            || false,
            |_| async { Ok::<(), &'static str>(()) },
            |_| {},
        )
        .await;
    assert_eq!(outcome, DeliveryOutcome::Fenced);
    assert_eq!(router.pending_for("parent").len(), 1);
}

#[tokio::test]
async fn losing_fence_during_successful_delivery_skips_ack_and_releases_claim() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let router = CompletionRouter::new(Arc::new(InMemoryCompletionStore::new()));
    router
        .record(CompletionRecord::new(
            "success-fenced",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let admitted = Arc::new(AtomicBool::new(true));
    let delivered = Arc::clone(&admitted);
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        ThreadDeliveryLeases::default(),
    );

    let outcome = delivery
        .deliver_once(
            target(&router),
            || false,
            {
                let admitted = Arc::clone(&admitted);
                move || admitted.load(Ordering::SeqCst)
            },
            move |_| {
                let delivered = Arc::clone(&delivered);
                async move {
                    delivered.store(false, Ordering::SeqCst);
                    Ok::<(), &'static str>(())
                }
            },
            |_| panic!("successful delivery does not write undelivered records"),
        )
        .await;

    assert_eq!(outcome, DeliveryOutcome::Fenced);
    let record = router.store().get("success-fenced").unwrap();
    assert_eq!(record.state, super::CompletionState::Pending);
    assert_eq!(record.attempts, 1, "only the original claim was counted");
    let reclaimed = router.claim_pending("parent", 1).unwrap();
    assert_eq!(reclaimed.len(), 1, "the lost-fence claim was released");
}

#[tokio::test]
async fn losing_fence_during_failed_delivery_skips_failure_and_final_writer() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let router =
        CompletionRouter::new(Arc::new(InMemoryCompletionStore::new())).with_max_attempts(1);
    router
        .record(CompletionRecord::new(
            "error-fenced",
            "parent",
            "worker",
            CompletionStatus::Failed,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let admitted = Arc::new(AtomicBool::new(true));
    let delivered = Arc::clone(&admitted);
    let writes = Arc::new(AtomicUsize::new(0));
    let writer = Arc::clone(&writes);
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        ThreadDeliveryLeases::default(),
    );

    let outcome = delivery
        .deliver_once_with_error(
            target(&router),
            || false,
            {
                let admitted = Arc::clone(&admitted);
                move || admitted.load(Ordering::SeqCst)
            },
            move |_| {
                let delivered = Arc::clone(&delivered);
                async move {
                    delivered.store(false, Ordering::SeqCst);
                    Err::<(), _>("delivery failed")
                }
            },
            move |_, _| {
                writer.fetch_add(1, Ordering::SeqCst);
            },
        )
        .await;

    assert_eq!(outcome, DeliveryOutcome::Fenced);
    assert_eq!(writes.load(Ordering::SeqCst), 0);
    let record = router.store().get("error-fenced").unwrap();
    assert_eq!(record.state, super::CompletionState::Pending);
    assert_eq!(record.attempts, 1, "failure must not add a retry attempt");
    let reclaimed = router.claim_pending("parent", 1).unwrap();
    assert_eq!(reclaimed.len(), 1, "the lost-fence claim was released");
}

#[tokio::test]
async fn shared_thread_lease_prevents_overlapping_delivery_turns() {
    let router = Arc::new(CompletionRouter::new(Arc::new(
        InMemoryCompletionStore::new(),
    )));
    router
        .record(CompletionRecord::new(
            "a",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let leases = ThreadDeliveryLeases::default();
    let delivery = Arc::new(CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        leases,
    ));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let task_router = Arc::clone(&router);
    let task_delivery = Arc::clone(&delivery);
    let mut started_tx = Some(started_tx);
    let mut release_rx = Some(release_rx);
    let first = tokio::spawn(async move {
        task_delivery
            .deliver_once(
                target(&task_router),
                || false,
                || true,
                |_| {
                    let started = started_tx.take().unwrap();
                    let release = release_rx.take().unwrap();
                    async move {
                        let _ = started.send(());
                        release.await.unwrap();
                        Ok::<(), &'static str>(())
                    }
                },
                |_| {},
            )
            .await
    });
    started_rx.await.unwrap();
    let second = delivery
        .deliver_once(
            target(&router),
            || false,
            || true,
            |_| async { Err::<(), &'static str>("overlapping delivery callback must not run") },
            |_| {},
        )
        .await;
    assert_eq!(second, DeliveryOutcome::LeaseHeld);
    release_tx.send(()).unwrap();
    assert_eq!(
        first.await.unwrap(),
        DeliveryOutcome::Delivered { count: 1 }
    );
}

#[test]
fn same_thread_id_is_isolated_between_tenants_for_delivery_and_busy_gates() {
    let delivery = ThreadDeliveryLeases::default();
    let thread_a = TenantThreadKey::new("tenant-a", "thread");
    let thread_b = TenantThreadKey::new("tenant-b", "thread");
    let lease_a = delivery.try_acquire(thread_a.clone()).unwrap();
    let lease_b = delivery.try_acquire(thread_b.clone()).unwrap();
    assert!(delivery.try_acquire(thread_a.clone()).is_none());

    let busy = ParentBusyGate::default();
    let session_a = TenantThreadKey::new("tenant-a", "session");
    assert!(busy.alias_session_to_thread(session_a.clone(), thread_a.clone()));
    let _busy_a = busy.enter(session_a).unwrap();
    assert!(busy.is_busy_scoped(&thread_a));
    assert!(!busy.is_busy_scoped(&thread_b));
    assert_eq!(busy.clear(&thread_a), 1);
    assert!(!busy.is_busy_scoped(&thread_a));

    drop((lease_a, lease_b));
}

#[tokio::test]
async fn delivery_accepts_a_call_specific_initial_delay() {
    let router = CompletionRouter::new(Arc::new(InMemoryCompletionStore::new()));
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig::default(),
        ThreadDeliveryLeases::default(),
    );
    let outcome = delivery
        .deliver_once_with_initial_delay(
            target(&router),
            Duration::ZERO,
            || false,
            || true,
            |_| async { Ok::<(), &'static str>(()) },
            |_| {},
        )
        .await;
    assert_eq!(outcome, DeliveryOutcome::Empty);
}

#[test]
fn release_all_clears_delivery_leases_without_touching_busy_state() {
    let delivery = ThreadDeliveryLeases::default();
    let thread_a = TenantThreadKey::new("tenant-a", "thread-a");
    let thread_b = TenantThreadKey::new("tenant-b", "thread-b");
    let busy = ParentBusyGate::default();
    let lease_a = delivery.try_acquire(thread_a.clone()).unwrap();
    let lease_b = delivery.try_acquire(thread_b.clone()).unwrap();
    let _busy = busy.enter(thread_a.clone()).unwrap();

    assert_eq!(delivery.release_all(), 2);
    let renewed_a = delivery.try_acquire(thread_a.clone()).unwrap();
    let renewed_b = delivery.try_acquire(thread_b.clone()).unwrap();
    drop((lease_a, lease_b));
    assert!(delivery.try_acquire(thread_a.clone()).is_none());
    assert!(delivery.try_acquire(thread_b).is_none());
    assert!(busy.is_busy_scoped(&thread_a));
    drop((renewed_a, renewed_b));
}

#[tokio::test]
async fn aborted_delivery_releases_its_claimed_batch() {
    let router = Arc::new(CompletionRouter::new(Arc::new(
        InMemoryCompletionStore::new(),
    )));
    router
        .record(CompletionRecord::new(
            "cancelled",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let delivery = Arc::new(CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 1,
        },
        ThreadDeliveryLeases::default(),
    ));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let task_router = Arc::clone(&router);
    let task_delivery = Arc::clone(&delivery);
    let task = tokio::spawn(async move {
        let mut started_tx = Some(started_tx);
        task_delivery
            .deliver_once(
                target(&task_router),
                || false,
                || true,
                |_| {
                    let started = started_tx.take().unwrap();
                    async move {
                        let _ = started.send(());
                        std::future::pending::<Result<(), &'static str>>().await
                    }
                },
                |_| {},
            )
            .await
    });
    started_rx.await.unwrap();
    assert_eq!(router.in_flight_for("parent").len(), 1);
    task.abort();
    let _ = task.await;
    assert_eq!(router.claim_pending("parent", 1).unwrap().len(), 1);
}

#[tokio::test]
async fn boot_recovery_drains_more_than_one_batch() {
    let router = CompletionRouter::new(Arc::new(InMemoryCompletionStore::new()));
    for id in ["a", "b", "c"] {
        router
            .record(CompletionRecord::new(
                id,
                "parent",
                "worker",
                CompletionStatus::Success,
                CompletionResult::default(),
            ))
            .await
            .unwrap();
    }
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 1,
        },
        ThreadDeliveryLeases::default(),
    );
    let mut batches = 0;
    let outcome = delivery
        .recover_on_boot(
            target(&router),
            || false,
            || true,
            |_| {
                batches += 1;
                async { Ok::<(), &'static str>(()) }
            },
            |_| {},
        )
        .await;
    assert_eq!(outcome, DeliveryOutcome::Delivered { count: 3 });
    assert_eq!(batches, 3);
    assert!(router.pending_for("parent").is_empty());
}

#[derive(Clone)]
struct NotifiedFence {
    notify: Arc<Notify>,
    admitted: Arc<std::sync::atomic::AtomicBool>,
}
impl DeliveryFence for NotifiedFence {
    fn admitted(&self) -> bool {
        self.admitted.load(std::sync::atomic::Ordering::SeqCst)
    }
    fn lost(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(self.notify.notified())
    }
}

#[tokio::test]
async fn fence_notification_interrupts_a_long_delivery_delay() {
    let fence = NotifiedFence {
        notify: Arc::new(Notify::new()),
        admitted: Arc::new(std::sync::atomic::AtomicBool::new(true)),
    };
    let signal = fence.clone();
    let router = Arc::new(CompletionRouter::new(Arc::new(
        InMemoryCompletionStore::new(),
    )));
    router
        .record(CompletionRecord::new(
            "fenced",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::default(),
        ))
        .await
        .unwrap();
    let delivery = Arc::new(CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::from_secs(60),
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 1,
        },
        ThreadDeliveryLeases::default(),
    ));
    let task = tokio::spawn(async move {
        delivery
            .deliver_once(
                target(&router),
                || false,
                fence,
                |_| async { Ok::<(), &'static str>(()) },
                |_| {},
            )
            .await
    });
    tokio::task::yield_now().await;
    signal
        .admitted
        .store(false, std::sync::atomic::Ordering::SeqCst);
    signal.notify.notify_one();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap(),
        DeliveryOutcome::Fenced
    );
}

/// How a delivery turn relates to the record's claim. The router's claim is
/// shortened to [`CLAIM`] so a turn can outlast it in real time.
#[derive(Clone, Copy)]
enum SlowTurn {
    WithinClaim,
    OutlivesClaim,
    OutlivesClaimAndIsReclaimed,
}

const CLAIM: Duration = Duration::from_secs(1);

/// Every hand-off of the record to a consumer (the drain's `deliver` and, for
/// the reclaim case, a concurrent `claim_pending` such as `begin_turn`/`pull`
/// would make), plus the drain's outcome.
async fn handoffs(turn: SlowTurn) -> (Vec<String>, DeliveryOutcome) {
    let store = Arc::new(InMemoryCompletionStore::new());
    let router = CompletionRouter::new(store).with_claim_visibility(CLAIM);
    router
        .record(CompletionRecord::new(
            "slow",
            "parent",
            "worker",
            CompletionStatus::Success,
            CompletionResult::text("child result"),
        ))
        .await
        .unwrap();
    let delivery = CompletionDelivery::new(
        CompletionDeliveryConfig {
            debounce: Duration::ZERO,
            recovery_delay: Duration::ZERO,
            retry_backoff: Duration::ZERO,
            busy_poll: Duration::ZERO,
            max_busy_polls: 0,
            batch_size: 8,
        },
        ThreadDeliveryLeases::default(),
    );
    let handed = Mutex::new(Vec::new());
    let turns = Mutex::new(0);
    let outcome = delivery
        .deliver_once(
            target(&router),
            || false,
            || true,
            |batch| {
                *turns.lock().unwrap() += 1;
                let first = *turns.lock().unwrap() == 1;
                handed.lock().unwrap().extend(
                    batch
                        .iter()
                        .map(|record| format!("deliver:{}", record.task_id)),
                );
                let (handed, router) = (&handed, &router);
                async move {
                    if first && !matches!(turn, SlowTurn::WithinClaim) {
                        tokio::time::sleep(CLAIM * 9 / 5).await;
                        if matches!(turn, SlowTurn::OutlivesClaimAndIsReclaimed) {
                            let other = router.claim_pending("parent", 8).unwrap();
                            handed.lock().unwrap().extend(
                                other
                                    .iter()
                                    .map(|record| format!("claim:{}", record.task_id)),
                            );
                        }
                        tokio::time::sleep(CLAIM * 7 / 10).await;
                    }
                    Ok::<(), &'static str>(())
                }
            },
            |_| {},
        )
        .await;
    (handed.into_inner().unwrap(), outcome)
}

#[tokio::test]
async fn a_delivery_within_its_claim_hands_the_record_over_once() {
    let (handed, outcome) = handoffs(SlowTurn::WithinClaim).await;
    assert_eq!(handed, ["deliver:slow"]);
    assert_eq!(outcome, DeliveryOutcome::Delivered { count: 1 });
}

#[tokio::test]
async fn a_delivery_that_outlives_its_claim_hands_the_record_over_once() {
    let (handed, outcome) = handoffs(SlowTurn::OutlivesClaim).await;
    assert_eq!(
        handed,
        ["deliver:slow"],
        "redelivered after the claim expired"
    );
    assert_eq!(outcome, DeliveryOutcome::Delivered { count: 1 });
}

#[tokio::test]
async fn a_reclaimed_slow_delivery_hands_the_record_over_once() {
    let (handed, outcome) = handoffs(SlowTurn::OutlivesClaimAndIsReclaimed).await;
    assert_eq!(
        handed,
        ["deliver:slow"],
        "handed to a second consumer after the claim expired"
    );
    assert_eq!(outcome, DeliveryOutcome::Delivered { count: 1 });
}
