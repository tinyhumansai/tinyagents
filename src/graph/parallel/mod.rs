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
//! of order: concurrency is driven by
//! [`buffer_unordered`](futures::stream::StreamExt::buffer_unordered), which
//! yields items as they finish, and each future carries its input index so the
//! completed results are re-sorted into input-order slots before being returned.

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

    let item_timeout = options.item_timeout;
    // Each future carries its input index so results can be re-ordered.
    // `buffer_unordered` bounds concurrency to `concurrency` and yields items as
    // they complete; we re-order into `slots` by index. Dropping the stream on a
    // fail-fast break cancels any remaining in-flight work.
    let mut stream = futures::stream::iter(items.into_iter().enumerate().map(|(index, item)| {
        let fut = f(index, item);
        async move {
            // A per-item timeout turns a slow item into a recoverable failure
            // rather than stalling the whole batch.
            let result = match item_timeout {
                Some(limit) => match tokio::time::timeout(limit, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(TinyAgentsError::Timeout(format!(
                        "parallel item {index} exceeded {} ms",
                        limit.as_millis()
                    ))),
                },
                None => fut.await,
            };
            (index, result)
        }
    }))
    .buffer_unordered(concurrency);

    let mut slots: Vec<Option<std::result::Result<T, String>>> = (0..total).map(|_| None).collect();
    // Under FailFast we return the first failure in *input* order, not the first
    // to complete (items finish out of order under `buffer_unordered`). Track the
    // lowest-index error seen; once every earlier item has also resolved, no
    // smaller-index error can still appear, so that error is final and we can
    // drop the stream to cancel the remaining in-flight work.
    let mut fail_fast_error: Option<(usize, TinyAgentsError)> = None;

    // The collection loop, wrapped so it can be raced against an overall timeout
    // and a cancellation token without duplicating the drain logic.
    let collect = async {
        while let Some((index, result)) = stream.next().await {
            match result {
                Ok(value) => slots[index] = Some(Ok(value)),
                Err(err) => {
                    if options.failure_policy == FailurePolicy::FailFast {
                        // Mark this index resolved so the "all earlier items
                        // done" check below can see it.
                        slots[index] = Some(Err(String::new()));
                        if fail_fast_error
                            .as_ref()
                            .is_none_or(|(seen, _)| index < *seen)
                        {
                            fail_fast_error = Some((index, err));
                        }
                        let min_index = fail_fast_error.as_ref().map(|(i, _)| *i).unwrap_or(index);
                        if slots[..min_index].iter().all(Option::is_some) {
                            break; // dropping the stream cancels remaining work
                        }
                        continue;
                    }
                    slots[index] = Some(Err(err.to_string()));
                }
            }
        }
    };

    let cancelled = async {
        match &options.cancellation {
            Some(token) => token.cancelled().await,
            None => std::future::pending::<()>().await,
        }
    };

    // Race collection against cancellation, all under an optional total timeout.
    let raced = async {
        tokio::select! {
            biased;
            _ = cancelled => Err(TinyAgentsError::Cancelled),
            () = collect => Ok(()),
        }
    };
    match options.total_timeout {
        Some(limit) => match tokio::time::timeout(limit, raced).await {
            Ok(inner) => inner?,
            Err(_) => {
                return Err(TinyAgentsError::Timeout(format!(
                    "parallel map/reduce exceeded {} ms",
                    limit.as_millis()
                )));
            }
        },
        None => raced.await?,
    }

    if let Some((_, err)) = fail_fast_error {
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
