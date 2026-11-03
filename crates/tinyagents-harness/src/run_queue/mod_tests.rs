//! Tests for [`RunQueue`]: per-lane push/drain FIFO ordering, status
//! snapshots, `clear`, and lane independence.

use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
struct MutableAdmission(AtomicBool);

impl QueueAdmission for MutableAdmission {
    fn acquire(&self) -> Result<Arc<dyn QueueAdmissionPermit>, QueueAdmissionError> {
        self.0
            .load(Ordering::SeqCst)
            .then(|| Arc::new(()) as Arc<dyn QueueAdmissionPermit>)
            .ok_or(QueueAdmissionError::Rejected)
    }
}

#[derive(Debug)]
struct RetryOnce(AtomicBool);

impl QueueAdmission for RetryOnce {
    fn acquire(&self) -> Result<Arc<dyn QueueAdmissionPermit>, QueueAdmissionError> {
        if self.0.swap(false, Ordering::SeqCst) {
            Err(QueueAdmissionError::Retry)
        } else {
            Ok(Arc::new(()))
        }
    }
}

#[tokio::test]
async fn transient_admission_result_retains_item_for_the_next_boundary() {
    let queue = RunQueue::new();
    queue
        .push_guarded(
            QueueLane::Followup,
            "recover after migration",
            Arc::new(RetryOnce(AtomicBool::new(true))),
        )
        .await;

    assert!(
        queue
            .take_admitted(QueueLane::Followup, QueueMode::All)
            .await
            .is_empty()
    );
    assert_eq!(queue.status().await.followups, 1);
    let retried = queue
        .take_admitted(QueueLane::Followup, QueueMode::All)
        .await;
    assert_eq!(retried.len(), 1);
    assert_eq!(retried[0].value, "recover after migration");
}

#[tokio::test]
async fn new_queue_is_empty() {
    let queue = RunQueue::<String>::new();
    assert_eq!(
        queue.status().await,
        QueueStatus {
            steers: 0,
            followups: 0,
            collects: 0,
            total: 0,
        }
    );
}

#[tokio::test]
async fn push_routes_items_to_the_requested_lane() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, "steer").await;
    queue.push(QueueLane::Followup, "followup").await;
    queue.push(QueueLane::Collect, "collect").await;

    assert_eq!(
        queue.status().await,
        QueueStatus {
            steers: 1,
            followups: 1,
            collects: 1,
            total: 3,
        }
    );
}

#[tokio::test]
async fn drain_is_fifo_and_does_not_affect_other_lanes() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, "first").await;
    queue.push(QueueLane::Steer, "second").await;
    queue.push(QueueLane::Followup, "later").await;

    assert_eq!(queue.drain(QueueLane::Steer).await, vec!["first", "second"]);
    assert_eq!(queue.status().await.followups, 1);
    assert_eq!(queue.status().await.steers, 0);
}

#[tokio::test]
async fn clear_empties_every_lane_and_reports_the_drop_count() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, 1).await;
    queue.push(QueueLane::Followup, 2).await;
    queue.push(QueueLane::Collect, 3).await;

    assert_eq!(queue.clear().await, 3);
    assert_eq!(queue.status().await.total, 0);
}

#[tokio::test]
async fn take_one_at_a_time_pops_only_the_oldest_item() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, "first").await;
    queue.push(QueueLane::Steer, "second").await;

    assert_eq!(
        queue.take(QueueLane::Steer, QueueMode::OneAtATime).await,
        vec!["first"]
    );
    assert_eq!(queue.status().await.steers, 1);
    assert_eq!(
        queue.take(QueueLane::Steer, QueueMode::OneAtATime).await,
        vec!["second"]
    );
    assert!(
        queue
            .take(QueueLane::Steer, QueueMode::OneAtATime)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn take_discards_items_whose_retained_admission_was_revoked() {
    let queue = RunQueue::new();
    let admission = Arc::new(MutableAdmission(AtomicBool::new(true)));
    queue
        .push_guarded(QueueLane::Followup, "child result", admission.clone())
        .await;
    queue.push(QueueLane::Followup, "ordinary follow-up").await;
    admission.0.store(false, Ordering::SeqCst);

    assert_eq!(
        queue.take(QueueLane::Followup, QueueMode::All).await,
        vec!["ordinary follow-up"],
        "legacy extraction must not transfer a guarded payload without its permit"
    );
    assert_eq!(
        queue
            .take_admitted(QueueLane::Followup, QueueMode::All)
            .await
            .into_iter()
            .map(|item| item.value)
            .collect::<Vec<_>>(),
        Vec::<&str>::new(),
        "a revoked item is rejected at the lifecycle admission boundary"
    );
}

#[tokio::test]
async fn one_at_a_time_skips_a_revoked_oldest_item_without_blocking_the_lane() {
    let queue = RunQueue::new();
    let admission = Arc::new(MutableAdmission(AtomicBool::new(false)));
    queue
        .push_guarded(QueueLane::Steer, "revoked", admission)
        .await;
    queue.push(QueueLane::Steer, "current").await;

    assert_eq!(
        queue
            .take_admitted(QueueLane::Steer, QueueMode::OneAtATime)
            .await
            .into_iter()
            .map(|item| item.value)
            .collect::<Vec<_>>(),
        vec!["current"]
    );
}

#[tokio::test]
async fn take_all_drains_the_whole_lane() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Followup, "first").await;
    queue.push(QueueLane::Followup, "second").await;

    assert_eq!(
        queue.take(QueueLane::Followup, QueueMode::All).await,
        vec!["first", "second"]
    );
    assert_eq!(queue.status().await.followups, 0);
}

#[tokio::test]
async fn snapshot_returns_every_item_in_lane_and_fifo_order_without_draining() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, "steer-1").await;
    queue.push(QueueLane::Followup, "followup-1").await;
    queue.push(QueueLane::Steer, "steer-2").await;
    queue.push(QueueLane::Collect, "collect-1").await;
    queue.push(QueueLane::Followup, "followup-2").await;

    assert_eq!(
        queue.snapshot().await,
        vec![
            (QueueLane::Steer, "steer-1"),
            (QueueLane::Steer, "steer-2"),
            (QueueLane::Followup, "followup-1"),
            (QueueLane::Followup, "followup-2"),
            (QueueLane::Collect, "collect-1"),
        ]
    );
    // Non-destructive: the queue is unchanged.
    assert_eq!(queue.status().await.total, 5);
}

#[tokio::test]
async fn snapshot_of_an_empty_queue_is_empty() {
    let queue = RunQueue::<String>::new();
    assert_eq!(queue.snapshot().await, Vec::new());
}

#[tokio::test]
async fn remove_where_removes_matching_items_across_every_lane() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, "keep").await;
    queue.push(QueueLane::Steer, "drop").await;
    queue.push(QueueLane::Followup, "drop").await;
    queue.push(QueueLane::Collect, "keep").await;

    let removed = queue.remove_where(|item| *item == "drop").await;

    assert_eq!(removed, 2);
    assert_eq!(
        queue.snapshot().await,
        vec![(QueueLane::Steer, "keep"), (QueueLane::Collect, "keep"),]
    );
}

#[tokio::test]
async fn remove_where_preserves_relative_order_of_survivors() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, 1).await;
    queue.push(QueueLane::Steer, 2).await;
    queue.push(QueueLane::Steer, 3).await;

    let removed = queue.remove_where(|item| *item == 2).await;

    assert_eq!(removed, 1);
    assert_eq!(queue.drain(QueueLane::Steer).await, vec![1, 3]);
}

#[tokio::test]
async fn remove_where_with_no_match_removes_nothing() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, "a").await;
    queue.push(QueueLane::Followup, "b").await;

    let removed = queue.remove_where(|item| *item == "nonexistent").await;

    assert_eq!(removed, 0);
    assert_eq!(queue.status().await.total, 2);
}

#[tokio::test]
async fn editing_changes_only_the_first_pending_match_in_delivery_order() {
    let queue = RunQueue::new();
    queue.push(QueueLane::Steer, 1).await;
    queue.push(QueueLane::Followup, 1).await;
    assert!(
        queue
            .edit_first_where(|item| *item == 1, |item| *item = 2)
            .await
    );
    assert_eq!(
        queue.snapshot().await,
        vec![(QueueLane::Steer, 2), (QueueLane::Followup, 1)]
    );
    queue.drain(QueueLane::Steer).await;
    assert!(
        !queue
            .edit_first_where(
                |item| *item == 2,
                |_| panic!("consumed item must not be edited")
            )
            .await
    );
}

#[tokio::test]
async fn moving_can_reorder_and_promote_a_followup_without_dropping_other_items() {
    let queue = RunQueue::new();
    for item in [1, 2, 3] {
        queue.push(QueueLane::Followup, item).await;
    }
    assert!(
        queue
            .move_first_where(|item| *item == 3, QueueLane::Followup, 0)
            .await
    );
    assert!(
        queue
            .move_first_where(|item| *item == 2, QueueLane::Steer, usize::MAX)
            .await
    );
    assert_eq!(
        queue.snapshot().await,
        vec![
            (QueueLane::Steer, 2),
            (QueueLane::Followup, 3),
            (QueueLane::Followup, 1)
        ]
    );
    assert!(
        !queue
            .move_first_where(|item| *item == 99, QueueLane::Collect, 0)
            .await
    );
    assert_eq!(queue.status().await.total, 3);
}
