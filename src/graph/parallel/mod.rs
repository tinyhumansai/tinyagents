//! Ordered, bounded-concurrency parallel map/reduce with a configurable failure
//! policy.
//!
//! Graph `Send` fanout is the low-level primitive for parallel supersteps, but
//! callers frequently want a *reusable* "run these N items concurrently and
//! reduce the results" helper with deterministic input-order results, a
//! concurrency cap, per-item success/failure isolation, and a policy for what to
//! do when some items fail (fail-fast, collect-all, quorum, best-effort). That
//! is what [`map_reduce`] provides, independent of the graph executor.
//!
//! Results are always returned in **input order** even though items complete out
//! of order, because the concurrency is driven by
//! [`buffered`](futures::stream::StreamExt::buffered), which preserves order.

mod types;

pub use types::*;

use futures::stream::StreamExt;
use std::future::Future;

use crate::error::{Result, TinyAgentsError};

/// Runs `f` over `items` concurrently (bounded by
/// [`ParallelOptions::max_concurrency`]), collecting per-item outcomes in input
/// order and applying the configured [`FailurePolicy`].
///
/// `f` receives each item's input index and the item, and returns a future
/// yielding `Result<T>`. The mapping is deterministic in input order regardless
/// of completion order.
///
/// # Errors
///
/// - [`FailurePolicy::FailFast`] returns the first item error (in input order),
///   cancelling remaining in-flight work.
/// - [`FailurePolicy::Quorum`] returns [`TinyAgentsError::Graph`] when fewer than
///   the required number of items succeeded.
/// - [`FailurePolicy::CollectAll`] and [`FailurePolicy::BestEffort`] never error.
pub async fn map_reduce<I, T, F, Fut>(
    items: Vec<I>,
    options: ParallelOptions,
    f: F,
) -> Result<ParallelOutcome<T>>
where
    F: Fn(usize, I) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let total = items.len();
    let concurrency = if options.max_concurrency == 0 {
        total.max(1)
    } else {
        options.max_concurrency
    };

    // Each future carries its input index so results can be re-ordered.
    // `buffer_unordered` bounds concurrency to `concurrency` and yields items as
    // they complete; we re-order into `slots` by index. Dropping the stream on a
    // fail-fast break cancels any remaining in-flight work.
    let mut stream = futures::stream::iter(items.into_iter().enumerate().map(|(index, item)| {
        let fut = f(index, item);
        async move { (index, fut.await) }
    }))
    .buffer_unordered(concurrency);

    let mut slots: Vec<Option<std::result::Result<T, String>>> = (0..total).map(|_| None).collect();
    let mut fail_fast_error: Option<TinyAgentsError> = None;

    while let Some((index, result)) = stream.next().await {
        match result {
            Ok(value) => slots[index] = Some(Ok(value)),
            Err(err) => {
                if options.failure_policy == FailurePolicy::FailFast {
                    fail_fast_error = Some(err);
                    break; // dropping `pending` cancels remaining work
                }
                slots[index] = Some(Err(err.to_string()));
            }
        }
    }

    if let Some(err) = fail_fast_error {
        return Err(err);
    }

    let mut outcomes: Vec<ItemOutcome<T>> = Vec::with_capacity(total);
    for (index, slot) in slots.into_iter().enumerate() {
        if let Some(result) = slot {
            outcomes.push(ItemOutcome { index, result });
        }
    }

    let success_count = outcomes.iter().filter(|o| o.is_ok()).count();

    match options.failure_policy {
        FailurePolicy::Quorum(required) if success_count < required => Err(TinyAgentsError::Graph(
            format!("parallel quorum not met: {success_count}/{required} items succeeded"),
        )),
        FailurePolicy::BestEffort => {
            outcomes.retain(|o| o.is_ok());
            Ok(ParallelOutcome { outcomes })
        }
        _ => Ok(ParallelOutcome { outcomes }),
    }
}

#[cfg(test)]
mod test;
