//! **Task runs**: the claim, heartbeat, and reclaim layer over a task board.
//!
//! A [`TaskBoardCard`](super::TaskBoardCard) records what needs doing;
//! [`TaskRun`] records an attempt at doing it. A worker claims a card, writes a
//! run, ticks a heartbeat while it works, and closes the run with an outcome.
//! If it dies without closing, the run goes stale and
//! [`store::reclaim_stale`] hands the card back to the queue — bounded by
//! [`RunLimits::max_reclaim_count`] so a card that keeps killing its workers
//! parks at `Blocked` rather than cycling forever.
//!
//! Runs are addressed exactly like boards — `(Store, thread_id)` — and share
//! the board's timestamp format (unix-epoch milliseconds as a string). The
//! staleness policy itself is a pure, clock-injected function
//! ([`staleness_reason`]) so it can be tested without waiting.
//!
//! ```no_run
//! use std::sync::Arc;
//! use tinyagents::graph::todos::runs::{RunLimits, RunOutcome, store as runs};
//! use tinyagents::harness::store::{InMemoryStore, Store};
//!
//! # async fn demo() -> tinyagents::error::Result<()> {
//! let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
//! let run = runs::create_run(&store, "thread-1", None, "task-1", "worker-a").await?;
//! runs::update_heartbeat(&store, "thread-1", &run.run_id).await?;
//! runs::complete_run(&store, "thread-1", &run.run_id, RunOutcome::Success, None, vec![]).await?;
//! // A sweep finds nothing to reclaim: the run reached a terminal state.
//! let swept = runs::reclaim_stale(&store, "thread-1", &RunLimits::default()).await?;
//! assert_eq!(swept.reclaimed_count, 0);
//! # Ok(())
//! # }
//! ```

pub mod store;
mod types;

pub use store::{
    DEFAULT_HEARTBEAT_TICK, RUNS_NAMESPACE, complete_run, count_reclaims_for_card, create_run,
    find_stale_runs, get_run, import_if_absent, list_runs, reclaim_stale, spawn_heartbeat_task,
    update_heartbeat,
};
pub use types::{
    DEFAULT_CLAIM_TTL_SECS, DEFAULT_HEARTBEAT_STALE_SECS, DEFAULT_MAX_RECLAIM_COUNT, ReclaimDetail,
    ReclaimResult, RunLimits, RunOutcome, TaskRun, staleness_reason,
};

#[cfg(test)]
mod test;
