//! Persistence for the thread-level goal, on the harness
//! [`Store`](crate::harness::store::Store).
//!
//! Each thread's goal is a single serialized [`ThreadGoal`] value under the
//! [`GOALS_NAMESPACE`] namespace, keyed by the hex-encoded thread id. There is
//! at most one goal per thread.
//!
//! The [`Store`] trait offers no compare-and-set and no cross-key transaction,
//! so every mutation runs `load → mutate → put` under a **per-thread async
//! mutex** ([`thread_lock`]) — the process-local analogue of OpenHuman's
//! file-rename atomicity. Inside that lock the `goal_id` compare-and-set guard
//! (see [`account_usage`] / [`set_continuation_suppressed_if`]) still rejects
//! stale accounting from a replaced goal.
//!
//! # Single-process only
//!
//! `thread_lock` serializes writers **within one process**. Across multiple
//! processes sharing the same [`FileStore`](crate::harness::store::FileStore)
//! there is no atomic CAS, so two concurrent read-modify-writes can lose an
//! update. The `goal_id` guard still prevents *logical corruption* from stale
//! accounting, but not lost updates. For multi-writer deployments funnel goal
//! mutations through a single process (a clean fix would be a
//! `Store::compare_and_swap` extension, out of scope here).

use std::sync::{Arc, OnceLock};

use tokio::sync::Mutex;

use super::types::{ThreadGoal, ThreadGoalStatus};
use crate::error::{Result, TinyAgentsError};
use crate::graph::thread_locks::ThreadLockMap;
use crate::harness::ids::{next_seq, now_ms};
use crate::harness::store::Store;

/// The [`Store`] namespace holding one [`ThreadGoal`] per thread.
pub const GOALS_NAMESPACE: &str = "graph.goals";

/// Serialises `load → mutate → put` per thread so a read-modify-write is atomic
/// within the process. Returns the thread's dedicated async mutex, creating it
/// on first use; unused mutexes are reclaimed (see
/// [`ThreadLockMap`](crate::graph::thread_locks::ThreadLockMap)) so the map
/// does not grow with every thread id ever seen.
fn thread_lock(thread_id: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<ThreadLockMap> = OnceLock::new();
    LOCKS
        .get_or_init(|| ThreadLockMap::new("goal lock map"))
        .lock_for(thread_id)
}

/// Hex-encodes the thread id into a [`Store`]-safe key. Required because
/// [`FileStore`](crate::harness::store::FileStore) rejects key bytes outside
/// `[A-Za-z0-9._-]`; hex is uniform across every backend.
fn key(thread_id: &str) -> String {
    thread_id
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Mints a fresh, process-unique goal id (`goal-<n>`).
fn new_goal_id() -> String {
    format!("goal-{}", next_seq())
}

fn validate_thread_id(thread_id: &str) -> Result<String> {
    let trimmed = thread_id.trim();
    if trimmed.is_empty() {
        return Err(TinyAgentsError::Validation(
            "thread goal thread_id must not be empty or whitespace".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

/// Reads the raw goal for `thread_id` (no validation, no lock). Returns `None`
/// when the thread has no goal.
async fn load(store: &Arc<dyn Store>, thread_id: &str) -> Result<Option<ThreadGoal>> {
    match store.get(GOALS_NAMESPACE, &key(thread_id)).await? {
        Some(value) => Ok(Some(serde_json::from_value(value)?)),
        None => Ok(None),
    }
}

/// Serializes and persists `goal`.
async fn save(store: &Arc<dyn Store>, goal: &ThreadGoal) -> Result<()> {
    let value = serde_json::to_value(goal)?;
    store
        .put(GOALS_NAMESPACE, &key(&goal.thread_id), value)
        .await
}

/// Load the goal for `thread_id` (read-only), or `None`.
pub async fn get(store: &Arc<dyn Store>, thread_id: &str) -> Result<Option<ThreadGoal>> {
    let thread_id = validate_thread_id(thread_id)?;
    load(store, &thread_id).await
}

/// Load every persisted thread goal (read-only). Skips values that fail to
/// deserialize so one corrupt entry can't hide the rest.
pub async fn list_all(store: &Arc<dyn Store>) -> Result<Vec<ThreadGoal>> {
    let mut goals = Vec::new();
    for k in store.list(GOALS_NAMESPACE).await? {
        if let Some(value) = store.get(GOALS_NAMESPACE, &k).await?
            && let Ok(goal) = serde_json::from_value::<ThreadGoal>(value)
        {
            goals.push(goal);
        }
    }
    Ok(goals)
}

/// Build + persist the goal for a `set`. The caller MUST hold the thread lock so
/// the read-modify-write is atomic.
///
/// If `objective` **differs** from the current one a fresh `goal_id` is minted
/// and counters reset (status → `Active`). If the objective is **unchanged**,
/// counters and `goal_id` are preserved and only the budget / `updated_at` are
/// refreshed; the goal re-opens to `Active` unless it is still over budget.
async fn compute_and_put_set(
    store: &Arc<dyn Store>,
    thread_id: &str,
    objective: &str,
    token_budget: Option<u64>,
) -> Result<ThreadGoal> {
    let now = now_ms();
    let goal = match load(store, thread_id).await? {
        Some(mut existing) if existing.objective == objective => {
            existing.token_budget = token_budget;
            existing.continuation_suppressed = false;
            existing.updated_at_ms = now;
            existing.status = if existing.over_budget() {
                ThreadGoalStatus::BudgetLimited
            } else {
                ThreadGoalStatus::Active
            };
            existing
        }
        existing => {
            let created_at_ms = existing.as_ref().map(|g| g.created_at_ms).unwrap_or(now);
            ThreadGoal {
                thread_id: thread_id.to_string(),
                goal_id: new_goal_id(),
                objective: objective.to_string(),
                status: ThreadGoalStatus::Active,
                token_budget,
                tokens_used: 0,
                time_used_seconds: 0,
                created_at_ms,
                updated_at_ms: now,
                continuation_suppressed: false,
            }
        }
    };
    save(store, &goal).await?;
    Ok(goal)
}

/// Create or replace the goal for `thread_id`.
pub async fn set(
    store: &Arc<dyn Store>,
    thread_id: &str,
    objective: &str,
    token_budget: Option<u64>,
) -> Result<ThreadGoal> {
    let objective = objective.trim();
    if objective.is_empty() {
        return Err(TinyAgentsError::Validation(
            "thread goal objective must not be empty".to_string(),
        ));
    }
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    compute_and_put_set(store, &thread_id, objective, token_budget).await
}

/// Set the goal **only if the thread has none yet**. Returns `Some(goal)` when a
/// new goal was created, or `None` when a goal already existed (left untouched).
/// The check and the write run under one lock so a concurrent writer can't slip
/// into the gap.
pub async fn set_if_absent(
    store: &Arc<dyn Store>,
    thread_id: &str,
    objective: &str,
    token_budget: Option<u64>,
) -> Result<Option<ThreadGoal>> {
    let objective = objective.trim();
    if objective.is_empty() {
        return Err(TinyAgentsError::Validation(
            "thread goal objective must not be empty".to_string(),
        ));
    }
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    if load(store, &thread_id).await?.is_some() {
        return Ok(None);
    }
    Ok(Some(
        compute_and_put_set(store, &thread_id, objective, token_budget).await?,
    ))
}

/// Delete the goal for `thread_id`. Returns whether one existed.
pub async fn clear(store: &Arc<dyn Store>, thread_id: &str) -> Result<bool> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let existed = load(store, &thread_id).await?.is_some();
    store.delete(GOALS_NAMESPACE, &key(&thread_id)).await?;
    Ok(existed)
}

/// Generic guarded mutator: load, apply `f`, persist. Returns the updated goal,
/// or a [`Validation`](TinyAgentsError::Validation) error if the thread has none.
async fn mutate<F>(store: &Arc<dyn Store>, thread_id: &str, f: F) -> Result<ThreadGoal>
where
    F: FnOnce(&mut ThreadGoal),
{
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut goal = load(store, &thread_id).await?.ok_or_else(|| {
        TinyAgentsError::Validation(format!("no thread goal for thread '{thread_id}'"))
    })?;
    f(&mut goal);
    goal.updated_at_ms = now_ms();
    save(store, &goal).await?;
    Ok(goal)
}

/// Mark the goal `Complete` (model-driven success). Suppresses further
/// continuation so a completed goal never re-drives.
pub async fn complete(store: &Arc<dyn Store>, thread_id: &str) -> Result<ThreadGoal> {
    mutate(store, thread_id, |g| {
        g.status = ThreadGoalStatus::Complete;
        g.continuation_suppressed = true;
    })
    .await
}

/// Pause an `Active` goal (host-driven). A no-op for goals that aren't active.
pub async fn pause(store: &Arc<dyn Store>, thread_id: &str) -> Result<ThreadGoal> {
    mutate(store, thread_id, |g| {
        if g.status.is_active() {
            g.status = ThreadGoalStatus::Paused;
        }
    })
    .await
}

/// Resume a `Paused` goal (host-driven). A no-op for goals that aren't paused —
/// a completed/budget-limited goal is not reactivated.
pub async fn resume(store: &Arc<dyn Store>, thread_id: &str) -> Result<ThreadGoal> {
    mutate(store, thread_id, |g| {
        if matches!(g.status, ThreadGoalStatus::Paused) {
            g.status = ThreadGoalStatus::Active;
            g.continuation_suppressed = false;
        }
    })
    .await
}

/// Set `continuation_suppressed` only when the thread's current goal still
/// matches `expected_goal_id` and is still active (compare-and-set). Returns the
/// goal as it stands after the (possibly skipped) write, or `None` when the
/// thread has no goal.
///
/// The guard means a goal completed or replaced during a continuation iteration
/// is never suppressed by a late post-iteration write.
pub async fn set_continuation_suppressed_if(
    store: &Arc<dyn Store>,
    thread_id: &str,
    expected_goal_id: &str,
    suppressed: bool,
) -> Result<Option<ThreadGoal>> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let Some(mut goal) = load(store, &thread_id).await? else {
        return Ok(None);
    };
    if goal.goal_id != expected_goal_id
        || !goal.status.is_active()
        || goal.continuation_suppressed == suppressed
    {
        return Ok(Some(goal));
    }
    goal.continuation_suppressed = suppressed;
    goal.updated_at_ms = now_ms();
    save(store, &goal).await?;
    Ok(Some(goal))
}

/// Account token + time usage against the goal, applying the budget constraint.
///
/// **Stale-write guard:** the delta is **silently ignored** when
/// `expected_goal_id` doesn't match the current goal — an in-flight accounting
/// call from a now-replaced goal must not corrupt the new one. An active goal
/// that crosses its budget becomes [`BudgetLimited`](ThreadGoalStatus::BudgetLimited).
/// Returns the goal as it stands after the (possibly skipped) update, or `None`
/// if there is no goal for the thread.
pub async fn account_usage(
    store: &Arc<dyn Store>,
    thread_id: &str,
    expected_goal_id: &str,
    token_delta: u64,
    secs_delta: u64,
) -> Result<Option<ThreadGoal>> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let Some(mut goal) = load(store, &thread_id).await? else {
        return Ok(None);
    };
    if goal.goal_id != expected_goal_id {
        return Ok(Some(goal));
    }
    if token_delta == 0 && secs_delta == 0 {
        return Ok(Some(goal));
    }
    goal.tokens_used = goal.tokens_used.saturating_add(token_delta);
    goal.time_used_seconds = goal.time_used_seconds.saturating_add(secs_delta);
    if goal.status.is_active() && goal.over_budget() {
        goal.status = ThreadGoalStatus::BudgetLimited;
    }
    goal.updated_at_ms = now_ms();
    save(store, &goal).await?;
    Ok(Some(goal))
}
