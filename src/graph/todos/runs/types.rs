//! Domain types for **task runs**: the claim/heartbeat/outcome record a worker
//! writes while it executes one [`TaskBoardCard`](super::super::TaskBoardCard).
//!
//! A board card says *what* to do; a [`TaskRun`] says *who is doing it right
//! now, since when, and whether they are still alive*. Ported from OpenHuman's
//! `threads::todos::runs`, minus the app coupling (domain-event bus,
//! workspace-file layout): a run is always `(Store, thread_id)`-addressed like
//! the board it belongs to.

use serde::{Deserialize, Serialize};

use crate::harness::ids::{next_seq, process_nonce};

/// Default staleness threshold for a run's heartbeat, in seconds.
///
/// A healthy worker ticks [`update_heartbeat`](super::store::update_heartbeat)
/// well inside this window; one that has not is presumed wedged.
pub const DEFAULT_HEARTBEAT_STALE_SECS: u64 = 300;

/// Default ceiling on a single claim's total age, in seconds. A run older than
/// this is reclaimed even if it is still heartbeating.
pub const DEFAULT_CLAIM_TTL_SECS: u64 = 3600;

/// Default number of reclaims a card tolerates before it parks as
/// [`Blocked`](super::super::TaskCardStatus::Blocked) instead of returning to
/// the queue.
pub const DEFAULT_MAX_RECLAIM_COUNT: u32 = 3;

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    /// The worker finished the card's work.
    Success,
    /// The worker ran and failed; `error` carries the reason.
    Failed,
    /// The run went stale and was reclaimed by a sweep; the worker never
    /// reported an outcome of its own.
    Reclaimed,
}

/// One claim on a card: who took it, when, and how it ended.
///
/// Timestamps are unix-epoch **milliseconds** rendered as strings, matching the
/// board's [`updated_at`](super::super::TaskBoardCard::updated_at).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskRun {
    /// Stable run id, unique within the thread.
    pub run_id: String,
    /// The card this run is executing.
    pub card_id: String,
    /// Opaque worker identity (an agent id, a host label, …).
    pub claimed_by: String,
    /// Freshly minted per claim, so a reclaimed worker's late write-back can be
    /// told apart from the current claim's.
    pub claim_token: String,
    /// When the claim was taken.
    pub started_at: String,
    /// Last liveness tick.
    pub last_heartbeat_at: String,
    /// When the run reached a terminal state; `None` while it is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    /// Terminal outcome; `None` while the run is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<RunOutcome>,
    /// Failure or reclaim reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Evidence the run gathered toward the card's acceptance criteria.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<String>,
}

impl TaskRun {
    /// Whether the run has not yet reached a terminal state.
    pub fn is_active(&self) -> bool {
        self.completed_at.is_none()
    }
}

/// Staleness and reclaim policy applied by a sweep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunLimits {
    /// A run whose last heartbeat is older than this is stale.
    pub heartbeat_stale_secs: u64,
    /// A run older than this is stale regardless of its heartbeat.
    pub claim_ttl_secs: u64,
    /// Reclaims a card tolerates before it parks as `Blocked`.
    pub max_reclaim_count: u32,
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            heartbeat_stale_secs: DEFAULT_HEARTBEAT_STALE_SECS,
            claim_ttl_secs: DEFAULT_CLAIM_TTL_SECS,
            max_reclaim_count: DEFAULT_MAX_RECLAIM_COUNT,
        }
    }
}

/// What one [`reclaim_stale`](super::store::reclaim_stale) sweep did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReclaimResult {
    /// Cards returned to `Todo` and re-dispatchable.
    pub reclaimed_count: usize,
    /// Cards parked at `Blocked` for exceeding
    /// [`RunLimits::max_reclaim_count`].
    pub blocked_count: usize,
    /// One entry per reclaimed run, in sweep order. Hosts that publish domain
    /// events read them from here — the crate emits none of its own.
    pub details: Vec<ReclaimDetail>,
}

/// One reclaimed run within a [`ReclaimResult`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReclaimDetail {
    /// The run that was reclaimed.
    pub run_id: String,
    /// The card it had claimed.
    pub card_id: String,
    /// Why it was judged stale.
    pub reason: String,
    /// The status the card was moved to (`todo` or `blocked`).
    pub new_card_status: String,
}

/// Why `run` is stale at `now_ms` under `limits`, or `None` when it is healthy.
///
/// Pure and clock-injected so the policy is testable without sleeping. The TTL
/// check is evaluated first, so a run that is both too old *and* silent reports
/// the more fundamental reason. An unparsable timestamp is treated as healthy:
/// a corrupt record must not cause a live worker's card to be yanked away.
pub fn staleness_reason(run: &TaskRun, now_ms: u64, limits: &RunLimits) -> Option<String> {
    let started = run.started_at.parse::<u64>().ok()?;
    let last_heartbeat = run.last_heartbeat_at.parse::<u64>().ok()?;

    let age_secs = now_ms.saturating_sub(started) / 1000;
    let heartbeat_age_secs = now_ms.saturating_sub(last_heartbeat) / 1000;

    if age_secs > limits.claim_ttl_secs {
        return Some(format!(
            "claim TTL expired (age {age_secs}s > limit {}s)",
            limits.claim_ttl_secs
        ));
    }
    if heartbeat_age_secs > limits.heartbeat_stale_secs {
        return Some(format!(
            "heartbeat stale (last heartbeat {heartbeat_age_secs}s ago > limit {}s)",
            limits.heartbeat_stale_secs
        ));
    }
    None
}

/// Mints a fresh, process-unique run id (`run-<nonce>-<n>`).
pub(crate) fn new_run_id() -> String {
    format!("run-{:x}-{}", process_nonce(), next_seq())
}

/// Mints a fresh claim token, unique across processes and restarts.
pub(crate) fn new_claim_token() -> String {
    format!("claim-{:x}-{}", process_nonce(), next_seq())
}
