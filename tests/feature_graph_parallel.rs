//! Feature tests for `graph::parallel::map_reduce` — the reusable, bounded
//! concurrency map/reduce helper independent of the graph executor.
//!
//! This helper had no integration coverage: these tests pin down input-order
//! result determinism despite out-of-order completion, the concurrency cap, the
//! per-item timeout, and every [`FailurePolicy`] variant.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tinyagents::{FailurePolicy, ParallelOptions, TinyAgentsError, map_reduce};

#[tokio::test]
async fn results_are_returned_in_input_order_despite_out_of_order_completion() {
    // Larger inputs finish *sooner* (shorter sleep), so completion order is the
    // reverse of input order; the outcome must still be input-ordered.
    let items = vec![1_u64, 2, 3, 4, 5];
    let outcome = map_reduce(
        items,
        ParallelOptions::default().with_failure_policy(FailurePolicy::CollectAll),
        |_index, item| async move {
            let delay = 50u64.saturating_sub(item * 5);
            tokio::time::sleep(Duration::from_millis(delay)).await;
            Ok::<u64, TinyAgentsError>(item * item)
        },
    )
    .await
    .expect("collect-all never errors");

    let indices: Vec<usize> = outcome.outcomes.iter().map(|o| o.index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    assert_eq!(outcome.into_successes(), vec![1, 4, 9, 16, 25]);
}

#[tokio::test]
async fn bounded_concurrency_never_exceeds_the_configured_cap() {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));

    let items: Vec<u64> = (0..8).collect();
    let outcome = map_reduce(
        items,
        ParallelOptions::default().with_max_concurrency(3),
        |_index, _item| {
            let in_flight = in_flight.clone();
            let peak = peak.clone();
            async move {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok::<u64, TinyAgentsError>(0)
            }
        },
    )
    .await
    .expect("collect-all never errors");

    assert_eq!(outcome.success_count(), 8);
    assert!(
        peak.load(Ordering::SeqCst) <= 3,
        "observed {} concurrent items, cap was 3",
        peak.load(Ordering::SeqCst)
    );
}

#[tokio::test]
async fn fail_fast_returns_the_first_error_in_input_order() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let items = vec![1_u64, 2, 3, 4];
    let err = map_reduce(
        items,
        ParallelOptions::default()
            .with_max_concurrency(1)
            .with_failure_policy(FailurePolicy::FailFast),
        {
            let attempts = attempts.clone();
            move |_index, item| {
                let attempts = attempts.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    if item == 2 {
                        Err(TinyAgentsError::Graph("item two blew up".to_string()))
                    } else {
                        Ok(item)
                    }
                }
            }
        },
    )
    .await
    .expect_err("fail-fast surfaces the first item error");

    assert!(err.to_string().contains("item two blew up"), "got: {err}");
    // With concurrency 1 the batch stops after the failing item, never reaching 4.
    assert!(attempts.load(Ordering::SeqCst) <= 2);
}

#[tokio::test]
async fn collect_all_records_each_success_and_failure() {
    let items = vec![1_u64, 2, 3, 4, 5];
    let outcome = map_reduce(
        items,
        ParallelOptions::default().with_failure_policy(FailurePolicy::CollectAll),
        |_index, item| async move {
            if item % 2 == 0 {
                Err(TinyAgentsError::Graph(format!("even {item}")))
            } else {
                Ok(item)
            }
        },
    )
    .await
    .expect("collect-all always returns Ok");

    assert_eq!(outcome.outcomes.len(), 5);
    assert_eq!(outcome.success_count(), 3);
    assert_eq!(outcome.failure_count(), 2);
    assert_eq!(outcome.successes(), vec![&1, &3, &5]);
    // Failures preserve their input index for correlation.
    let failed_indices: Vec<usize> = outcome
        .outcomes
        .iter()
        .filter(|o| !o.is_ok())
        .map(|o| o.index)
        .collect();
    assert_eq!(failed_indices, vec![1, 3]);
}

#[tokio::test]
async fn quorum_succeeds_at_threshold_and_errors_below_it() {
    let run = |threshold: usize, fail_from: u64| async move {
        let items = vec![1_u64, 2, 3, 4];
        map_reduce(
            items,
            ParallelOptions::default().with_failure_policy(FailurePolicy::Quorum(threshold)),
            move |_index, item| async move {
                if item >= fail_from {
                    Err(TinyAgentsError::Graph(format!("fail {item}")))
                } else {
                    Ok(item)
                }
            },
        )
        .await
    };

    // 3 of 4 succeed (items 1,2,3); a quorum of 3 is met.
    let met = run(3, 4).await.expect("quorum of 3 is satisfied");
    assert_eq!(met.success_count(), 3);

    // Only 1 of 4 succeeds (item 1); a quorum of 3 is not met.
    let unmet = run(3, 2).await.expect_err("quorum of 3 is not satisfied");
    assert!(matches!(unmet, TinyAgentsError::Graph(_)), "got: {unmet:?}");
}

#[tokio::test]
async fn best_effort_keeps_only_the_successful_outputs() {
    let items = vec![1_u64, 2, 3, 4, 5, 6];
    let outcome = map_reduce(
        items,
        ParallelOptions::default().with_failure_policy(FailurePolicy::BestEffort),
        |_index, item| async move {
            if item % 3 == 0 {
                Err(TinyAgentsError::Graph(format!("mult of three {item}")))
            } else {
                Ok(item)
            }
        },
    )
    .await
    .expect("best-effort always returns Ok");

    // Only successes are retained, in input order; failures are dropped entirely.
    assert_eq!(outcome.failure_count(), 0);
    assert_eq!(outcome.into_successes(), vec![1, 2, 4, 5]);
}

#[tokio::test]
async fn per_item_timeout_turns_a_slow_item_into_a_recoverable_failure() {
    let items = vec![10_u64, 200, 30];
    let outcome = map_reduce(
        items,
        ParallelOptions::default()
            .with_failure_policy(FailurePolicy::CollectAll)
            .with_item_timeout(Duration::from_millis(60)),
        |_index, item| async move {
            tokio::time::sleep(Duration::from_millis(item)).await;
            Ok::<u64, TinyAgentsError>(item)
        },
    )
    .await
    .expect("collect-all absorbs the timeout as a per-item failure");

    // The 200ms item exceeds the 60ms bound and is recorded as a failure; the
    // fast items still succeed.
    assert_eq!(outcome.success_count(), 2);
    assert_eq!(outcome.failure_count(), 1);
    let failed = outcome.outcomes.iter().find(|o| !o.is_ok()).unwrap();
    assert_eq!(failed.index, 1);
}
