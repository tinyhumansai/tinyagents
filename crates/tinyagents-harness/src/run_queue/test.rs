use super::*;

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
