//! Types for the ordered, bounded parallel map/reduce helper.

/// What [`map_reduce`](super::map_reduce) does when some items fail.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FailurePolicy {
    /// Stop at the first failing item (in input order) and return its error;
    /// remaining in-flight work is dropped/cancelled.
    FailFast,
    /// Run every item to completion and always return `Ok`, recording per-item
    /// success or failure for the caller to inspect.
    #[default]
    CollectAll,
    /// Run every item to completion; return `Ok` when at least `n` items
    /// succeeded, otherwise return an error.
    Quorum(usize),
    /// Run every item to completion and always return `Ok`, silently keeping
    /// only the successful outputs.
    BestEffort,
}

/// Options controlling parallel execution.
#[derive(Clone, Debug, Default)]
pub struct ParallelOptions {
    /// Maximum number of items to run concurrently. `0` means unbounded.
    pub max_concurrency: usize,
    /// How to react to per-item failures.
    pub failure_policy: FailurePolicy,
    /// Per-item wall-clock timeout. An item exceeding it fails with a timeout
    /// message (then handled by the [`FailurePolicy`]); `None` means no per-item
    /// bound.
    pub item_timeout: Option<std::time::Duration>,
    /// Overall wall-clock timeout for the whole map/reduce. On elapse the call
    /// returns [`TinyAgentsError::Timeout`][crate::error::TinyAgentsError::Timeout]
    /// and remaining in-flight work is dropped; `None` means no overall bound.
    pub total_timeout: Option<std::time::Duration>,
    /// Cooperative cancellation token: when cancelled, the call stops collecting
    /// and returns [`TinyAgentsError::Cancelled`][crate::error::TinyAgentsError::Cancelled],
    /// dropping remaining in-flight work.
    pub cancellation: Option<crate::harness::cancel::CancellationToken>,
}

impl ParallelOptions {
    /// Bounds concurrency to `n` simultaneous items.
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }

    /// Sets the failure policy.
    pub fn with_failure_policy(mut self, policy: FailurePolicy) -> Self {
        self.failure_policy = policy;
        self
    }

    /// Bounds each item to `timeout` of wall-clock execution.
    pub fn with_item_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.item_timeout = Some(timeout);
        self
    }

    /// Bounds the whole map/reduce to `timeout` of wall-clock execution.
    pub fn with_total_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }

    /// Attaches a cooperative cancellation token.
    pub fn with_cancellation(mut self, token: crate::harness::cancel::CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }
}

/// The outcome of one item in a parallel map/reduce, tagged with its input
/// index so callers can correlate results back to inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemOutcome<T> {
    /// The input index (0-based) this outcome corresponds to.
    pub index: usize,
    /// `Ok(value)` on success, `Err(message)` on failure.
    pub result: std::result::Result<T, String>,
}

impl<T> ItemOutcome<T> {
    /// Returns `true` when the item succeeded.
    pub fn is_ok(&self) -> bool {
        self.result.is_ok()
    }
}

/// The collected result of a parallel map/reduce, preserving input order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParallelOutcome<T> {
    /// Per-item outcomes, in input order. Under [`FailurePolicy::BestEffort`]
    /// only successful items are present.
    pub outcomes: Vec<ItemOutcome<T>>,
}

impl<T> ParallelOutcome<T> {
    /// Number of successful items.
    pub fn success_count(&self) -> usize {
        self.outcomes.iter().filter(|o| o.is_ok()).count()
    }

    /// Number of failed items.
    pub fn failure_count(&self) -> usize {
        self.outcomes.iter().filter(|o| !o.is_ok()).count()
    }

    /// Borrows every successful value, in input order.
    pub fn successes(&self) -> Vec<&T> {
        self.outcomes
            .iter()
            .filter_map(|o| o.result.as_ref().ok())
            .collect()
    }

    /// Consumes the outcome and returns only the successful values, in input
    /// order.
    pub fn into_successes(self) -> Vec<T> {
        self.outcomes
            .into_iter()
            .filter_map(|o| o.result.ok())
            .collect()
    }
}
