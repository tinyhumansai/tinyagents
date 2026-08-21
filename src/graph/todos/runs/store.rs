//! Persistence and lifecycle for [`TaskRun`]s, on the harness
//! [`Store`](crate::harness::store::Store).
//!
//! Each thread's runs are a single serialized `Vec<TaskRun>` under the
//! [`RUNS_NAMESPACE`] namespace, keyed by the hex-encoded thread id — the same
//! addressing the board itself uses in
//! [`graph::todos::store`](crate::graph::todos::store), so a board and its run
//! log live side by side in one store. Every mutation runs
//! `load → mutate → put` under a **per-thread async mutex**, so the
//! read-modify-write is atomic within the process (the same single-process
//! caveat as the board store).
//!
//! The run log is what makes a wedged worker recoverable: [`reclaim_stale`]
//! sweeps runs whose heartbeat or claim has aged out, closes them as
//! [`RunOutcome::Reclaimed`], and moves their card back to `Todo` — or parks it
//! at `Blocked` once the card has burned through
//! [`RunLimits::max_reclaim_count`] attempts, so a permanently poisonous card
//! stops cycling through workers.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::Mutex;

use super::types::{
    ReclaimDetail, ReclaimResult, RunLimits, RunOutcome, TaskRun, new_claim_token, new_run_id,
    staleness_reason,
};
use crate::error::{Result, TinyAgentsError};
use crate::graph::thread_locks::ThreadLockMap;
use crate::graph::todos::store as board;
use crate::graph::todos::types::{CardPatch, TaskCardStatus, now_millis, now_stamp};
use crate::harness::store::Store;

/// The [`Store`] namespace holding one `Vec<TaskRun>` per thread.
pub const RUNS_NAMESPACE: &str = "graph.todos.runs";

/// Default cadence of the background heartbeat spawned by
/// [`spawn_heartbeat_task`].
pub const DEFAULT_HEARTBEAT_TICK: Duration = Duration::from_secs(30);

/// Serialises `load → mutate → put` per thread. Kept separate from the board's
/// lock map so a run write never blocks a card write (and vice versa) — the
/// reclaim path deliberately takes them one after the other, never nested.
fn runs_lock(thread_id: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<ThreadLockMap> = OnceLock::new();
    LOCKS
        .get_or_init(|| ThreadLockMap::new("task run lock map"))
        .lock_for(thread_id)
}

/// Hex-encodes the thread id into a [`Store`]-safe key.
fn key(thread_id: &str) -> String {
    thread_id
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn validate_thread_id(thread_id: &str) -> Result<String> {
    let trimmed = thread_id.trim();
    if trimmed.is_empty() {
        return Err(TinyAgentsError::Validation(
            "task run thread_id must not be empty or whitespace".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

async fn load(store: &Arc<dyn Store>, thread_id: &str) -> Result<Vec<TaskRun>> {
    match store.get(RUNS_NAMESPACE, &key(thread_id)).await? {
        Some(value) => Ok(serde_json::from_value(value)?),
        None => Ok(Vec::new()),
    }
}

async fn save(store: &Arc<dyn Store>, thread_id: &str, runs: &[TaskRun]) -> Result<()> {
    store
        .put(RUNS_NAMESPACE, &key(thread_id), serde_json::to_value(runs)?)
        .await
}

/// Claim `card_id` for `claimed_by` and record the claim as a fresh active run.
///
/// `run_id` lets a caller supply its own id (to correlate the run with an
/// external session); `None` mints one. Claiming does **not** touch the card —
/// the caller moves it to `InProgress` through the board store, which is what
/// enforces the single-`InProgress` invariant.
pub async fn create_run(
    store: &Arc<dyn Store>,
    thread_id: &str,
    run_id: Option<&str>,
    card_id: &str,
    claimed_by: &str,
) -> Result<TaskRun> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = runs_lock(&thread_id);
    let _guard = lock.lock().await;

    let now = now_stamp();
    let run = TaskRun {
        run_id: run_id.map(str::to_string).unwrap_or_else(new_run_id),
        card_id: card_id.to_string(),
        claimed_by: claimed_by.to_string(),
        claim_token: new_claim_token(),
        started_at: now.clone(),
        last_heartbeat_at: now,
        completed_at: None,
        outcome: None,
        error: None,
        evidence: Vec::new(),
    };

    let mut runs = load(store, &thread_id).await?;
    if runs.iter().any(|existing| existing.run_id == run.run_id) {
        return Err(TinyAgentsError::Validation(format!(
            "task run '{}' already exists on thread '{thread_id}'",
            run.run_id
        )));
    }
    runs.push(run.clone());
    save(store, &thread_id, &runs).await?;

    tracing::info!(
        thread_id = %thread_id,
        run_id = %run.run_id,
        card_id = %card_id,
        claimed_by = %claimed_by,
        "[graph:todos:runs] claim created"
    );
    Ok(run)
}

/// Refresh the liveness tick of an **active** run.
///
/// Errors when the run is unknown or already terminal — the heartbeat loop
/// treats that as its stop signal rather than resurrecting a finished run.
pub async fn update_heartbeat(store: &Arc<dyn Store>, thread_id: &str, run_id: &str) -> Result<()> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = runs_lock(&thread_id);
    let _guard = lock.lock().await;

    let mut runs = load(store, &thread_id).await?;
    let run = runs
        .iter_mut()
        .find(|run| run.run_id == run_id && run.is_active())
        .ok_or_else(|| {
            TinyAgentsError::Validation(format!(
                "active task run '{run_id}' not found on thread '{thread_id}'"
            ))
        })?;
    run.last_heartbeat_at = now_stamp();
    save(store, &thread_id, &runs).await
}

/// Close an **active** run with a terminal outcome, returning the closed record.
pub async fn complete_run(
    store: &Arc<dyn Store>,
    thread_id: &str,
    run_id: &str,
    outcome: RunOutcome,
    error: Option<String>,
    evidence: Vec<String>,
) -> Result<TaskRun> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = runs_lock(&thread_id);
    let _guard = lock.lock().await;

    let mut runs = load(store, &thread_id).await?;
    let run = runs
        .iter_mut()
        .find(|run| run.run_id == run_id && run.is_active())
        .ok_or_else(|| {
            TinyAgentsError::Validation(format!(
                "active task run '{run_id}' not found on thread '{thread_id}'"
            ))
        })?;
    run.completed_at = Some(now_stamp());
    run.outcome = Some(outcome);
    run.error = error;
    run.evidence = evidence;
    let completed = run.clone();
    save(store, &thread_id, &runs).await?;

    tracing::info!(
        thread_id = %thread_id,
        run_id = %run_id,
        outcome = ?completed.outcome,
        "[graph:todos:runs] run completed"
    );
    Ok(completed)
}

/// Every run recorded for the thread, oldest first; filtered to one card when
/// `card_id` is given.
pub async fn list_runs(
    store: &Arc<dyn Store>,
    thread_id: &str,
    card_id: Option<&str>,
) -> Result<Vec<TaskRun>> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = runs_lock(&thread_id);
    let _guard = lock.lock().await;

    let runs = load(store, &thread_id).await?;
    Ok(match card_id {
        Some(card_id) => runs.into_iter().filter(|r| r.card_id == card_id).collect(),
        None => runs,
    })
}

/// One run by id, or `None` when the thread has never recorded it.
pub async fn get_run(
    store: &Arc<dyn Store>,
    thread_id: &str,
    run_id: &str,
) -> Result<Option<TaskRun>> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = runs_lock(&thread_id);
    let _guard = lock.lock().await;

    Ok(load(store, &thread_id)
        .await?
        .into_iter()
        .find(|run| run.run_id == run_id))
}

/// Active runs judged stale under `limits`, each paired with the reason.
pub async fn find_stale_runs(
    store: &Arc<dyn Store>,
    thread_id: &str,
    limits: &RunLimits,
) -> Result<Vec<(TaskRun, String)>> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = runs_lock(&thread_id);
    let _guard = lock.lock().await;

    let now = now_millis();
    Ok(load(store, &thread_id)
        .await?
        .into_iter()
        .filter(TaskRun::is_active)
        .filter_map(|run| staleness_reason(&run, now, limits).map(|reason| (run, reason)))
        .collect())
}

/// How many times `card_id` has already been reclaimed on this thread.
pub async fn count_reclaims_for_card(
    store: &Arc<dyn Store>,
    thread_id: &str,
    card_id: &str,
) -> Result<u32> {
    Ok(list_runs(store, thread_id, Some(card_id))
        .await?
        .iter()
        .filter(|run| run.outcome == Some(RunOutcome::Reclaimed))
        .count() as u32)
}

/// Sweep the thread's stale runs: close each as [`RunOutcome::Reclaimed`], then
/// move its card back to `Todo` so a later dispatch can pick it up — or park it
/// at `Blocked` with a diagnostic blocker once the card has exceeded
/// [`RunLimits::max_reclaim_count`] reclaims.
///
/// Best-effort per run: a card write that fails is logged and skipped, leaving
/// the rest of the sweep to proceed. The returned [`ReclaimResult`] is the
/// crate's only report — it publishes no events of its own, so a host that
/// needs them derives them from `details`.
pub async fn reclaim_stale(
    store: &Arc<dyn Store>,
    thread_id: &str,
    limits: &RunLimits,
) -> Result<ReclaimResult> {
    let thread_id = validate_thread_id(thread_id)?;
    let stale = find_stale_runs(store, &thread_id, limits).await?;
    if stale.is_empty() {
        return Ok(ReclaimResult::default());
    }

    let mut result = ReclaimResult::default();
    for (run, reason) in &stale {
        if let Err(error) = complete_run(
            store,
            &thread_id,
            &run.run_id,
            RunOutcome::Reclaimed,
            Some(reason.clone()),
            Vec::new(),
        )
        .await
        {
            tracing::warn!(
                thread_id = %thread_id,
                run_id = %run.run_id,
                %error,
                "[graph:todos:runs] could not close stale run"
            );
            continue;
        }

        // Counted *after* closing this run, so the current reclaim is included:
        // the card parks at `Blocked` on the reclaim that reaches the limit.
        let reclaims = count_reclaims_for_card(store, &thread_id, &run.card_id).await?;
        let park = reclaims >= limits.max_reclaim_count;
        let status = if park {
            TaskCardStatus::Blocked
        } else {
            TaskCardStatus::Todo
        };
        let patch = CardPatch {
            status: Some(status),
            blocker: park.then(|| {
                format!(
                    "Reclaimed {reclaims} time(s), exceeding limit of {}. Last reclaim reason: {reason}",
                    limits.max_reclaim_count
                )
            }),
            ..Default::default()
        };

        match board::edit(store, &thread_id, &run.card_id, patch).await {
            Ok(_) => {
                if park {
                    result.blocked_count += 1;
                } else {
                    result.reclaimed_count += 1;
                }
                result.details.push(ReclaimDetail {
                    run_id: run.run_id.clone(),
                    card_id: run.card_id.clone(),
                    reason: reason.clone(),
                    new_card_status: status.as_str().to_string(),
                });
                tracing::info!(
                    thread_id = %thread_id,
                    run_id = %run.run_id,
                    card_id = %run.card_id,
                    new_status = status.as_str(),
                    reclaims,
                    %reason,
                    "[graph:todos:runs] card reclaimed"
                );
            }
            Err(error) => tracing::warn!(
                thread_id = %thread_id,
                run_id = %run.run_id,
                card_id = %run.card_id,
                %error,
                "[graph:todos:runs] could not move card after reclaim"
            ),
        }
    }
    Ok(result)
}

/// Spawn the background heartbeat for an in-flight run.
///
/// Ticks [`update_heartbeat`] every `tick` until either the run stops being
/// active (the tick errors, which is the normal end after
/// [`complete_run`]) or `cancel` fires. The first immediate tick is skipped so
/// the freshly-created run is not written twice.
pub fn spawn_heartbeat_task(
    store: Arc<dyn Store>,
    thread_id: String,
    run_id: String,
    mut cancel: tokio::sync::watch::Receiver<bool>,
    tick: Duration,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(tick);
        ticker.tick().await; // `interval` fires immediately; skip that one.
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    if let Err(error) = update_heartbeat(&store, &thread_id, &run_id).await {
                        tracing::debug!(
                            thread_id = %thread_id,
                            run_id = %run_id,
                            %error,
                            "[graph:todos:runs] heartbeat stopped (run is no longer active)"
                        );
                        break;
                    }
                }
                _ = cancel.changed() => {
                    tracing::debug!(
                        thread_id = %thread_id,
                        run_id = %run_id,
                        "[graph:todos:runs] heartbeat cancelled"
                    );
                    break;
                }
            }
        }
    });
}
