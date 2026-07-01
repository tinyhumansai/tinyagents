//! Tests for the parallel map/reduce helper.

use super::*;
use crate::error::TinyAgentsError;

#[tokio::test]
async fn preserves_input_order_despite_completion_order() {
    // Items complete out of order (larger sleeps first) but results stay ordered.
    let out = map_reduce(
        vec![3u64, 1, 2],
        ParallelOptions::default(),
        |_i, n| async move {
            tokio::time::sleep(std::time::Duration::from_millis(n)).await;
            Ok::<_, TinyAgentsError>(n * 10)
        },
    )
    .await
    .unwrap();

    let values: Vec<u64> = out.into_successes();
    assert_eq!(values, vec![30, 10, 20]);
}

#[tokio::test]
async fn collect_all_records_per_item_failure() {
    let out = map_reduce(
        vec![1, 2, 3, 4],
        ParallelOptions::default().with_failure_policy(FailurePolicy::CollectAll),
        |_i, n| async move {
            if n % 2 == 0 {
                Err(TinyAgentsError::Graph(format!("even {n}")))
            } else {
                Ok(n)
            }
        },
    )
    .await
    .unwrap();

    assert_eq!(out.success_count(), 2);
    assert_eq!(out.failure_count(), 2);
    // Outcomes preserve input order and index.
    assert_eq!(out.outcomes[0].index, 0);
    assert!(out.outcomes[1].result.is_err());
}

#[tokio::test]
async fn fail_fast_returns_first_error() {
    let result = map_reduce(
        vec![1, 2, 3],
        ParallelOptions::default().with_failure_policy(FailurePolicy::FailFast),
        |_i, n| async move {
            if n == 2 {
                Err(TinyAgentsError::Graph("boom".to_string()))
            } else {
                Ok(n)
            }
        },
    )
    .await;
    assert!(matches!(result, Err(TinyAgentsError::Graph(_))));
}

#[tokio::test]
async fn quorum_requires_minimum_successes() {
    let opts = ParallelOptions::default().with_failure_policy(FailurePolicy::Quorum(3));
    let items = vec![1, 2, 3, 4];
    let f = |_i: usize, n: i32| async move {
        if n <= 2 {
            Ok(n)
        } else {
            Err(TinyAgentsError::Graph("nope".to_string()))
        }
    };
    // Only 2 succeed, quorum of 3 not met.
    let result = map_reduce(items.clone(), opts, f).await;
    assert!(matches!(result, Err(TinyAgentsError::Graph(_))));

    // Quorum of 2 is met.
    let opts2 = ParallelOptions::default().with_failure_policy(FailurePolicy::Quorum(2));
    let ok = map_reduce(items, opts2, |_i, n| async move {
        if n <= 2 {
            Ok(n)
        } else {
            Err(TinyAgentsError::Graph("nope".to_string()))
        }
    })
    .await
    .unwrap();
    assert_eq!(ok.success_count(), 2);
}

#[tokio::test]
async fn best_effort_keeps_only_successes() {
    let out = map_reduce(
        vec![1, 2, 3],
        ParallelOptions::default().with_failure_policy(FailurePolicy::BestEffort),
        |_i, n| async move {
            if n == 2 {
                Err(TinyAgentsError::Graph("skip".to_string()))
            } else {
                Ok(n)
            }
        },
    )
    .await
    .unwrap();
    assert_eq!(out.into_successes(), vec![1, 3]);
}

#[tokio::test]
async fn concurrency_cap_bounds_simultaneous_work() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let opts = ParallelOptions::default().with_max_concurrency(2);

    map_reduce(vec![0u64; 6], opts, |_i, _n| {
        let live = live.clone();
        let peak = peak.clone();
        async move {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            live.fetch_sub(1, Ordering::SeqCst);
            Ok::<_, TinyAgentsError>(())
        }
    })
    .await
    .unwrap();

    assert!(peak.load(Ordering::SeqCst) <= 2, "concurrency exceeded cap");
}
