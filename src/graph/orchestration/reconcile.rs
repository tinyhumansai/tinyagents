//! Settling orphaned tasks left behind by a dead executor.
//!
//! A detached task runs on an executor owned by the process that spawned it.
//! When that process dies, the executor, its abort handle and its cancellation
//! token die with it — but a durable [`TaskStore`] still holds a non-terminal
//! record. Nothing will ever complete that record, so without reconciliation it
//! reads as perpetually live: a supervisor waits forever, and a ledger renders a
//! run that has not existed since the last restart.
//!
//! [`reconcile_orphaned_tasks`] settles those records. It owns the state machine
//! — which statuses count as live, and which terminal state each becomes — and
//! nothing else. The failure *reason* is supplied by the caller, because the
//! sentence a user reads about their own product is not this crate's to write.
//!
//! Reconciliation is best-effort by construction: a record that races to
//! terminal between the listing and the transition is expected, not exceptional,
//! so per-task failures are captured in the report and the sweep continues.

use crate::harness::ids::TaskId;

use super::store::TaskStore;
use super::types::{OrchestrationTaskFilter, OrchestrationTaskRecord, OrchestrationTaskStatus};

/// What reconciliation did to one task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileOutcome {
    /// The task had a cancellation pending and was settled as cancelled.
    Cancelled,
    /// The task was live with no cancellation pending and was settled as failed.
    Failed,
    /// The transition itself failed; the task may still be non-terminal.
    Error(String),
}

impl ReconcileOutcome {
    /// `true` when the task reached a terminal state.
    pub fn is_settled(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Failed)
    }
}

/// One task considered by a reconciliation sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconciledTask {
    /// The task that was swept.
    pub task_id: TaskId,
    /// The non-terminal status it held before the sweep.
    pub prior_status: OrchestrationTaskStatus,
    /// What the sweep did to it.
    pub outcome: ReconcileOutcome,
    /// The record as it stood before the transition, so callers can read their
    /// own metadata off it when emitting lifecycle events.
    pub record: OrchestrationTaskRecord,
}

/// The result of one reconciliation sweep.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReconcileReport {
    /// Every orphan considered, in store order.
    pub tasks: Vec<ReconciledTask>,
}

impl ReconcileReport {
    /// How many tasks reached a terminal state.
    pub fn reconciled_count(&self) -> usize {
        self.settled().count()
    }

    /// How many tasks could not be transitioned.
    pub fn error_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|task| matches!(task.outcome, ReconcileOutcome::Error(_)))
            .count()
    }

    /// The tasks that reached a terminal state.
    pub fn settled(&self) -> impl Iterator<Item = &ReconciledTask> {
        self.tasks.iter().filter(|task| task.outcome.is_settled())
    }

    /// `true` when the sweep found no orphans at all.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

/// Stable lowercase label for a task status, for logs and audit records.
pub fn task_status_label(status: OrchestrationTaskStatus) -> &'static str {
    match status {
        OrchestrationTaskStatus::Pending => "pending",
        OrchestrationTaskStatus::Running => "running",
        OrchestrationTaskStatus::Awaiting => "awaiting",
        OrchestrationTaskStatus::CancelRequested => "cancel_requested",
        OrchestrationTaskStatus::Completed => "completed",
        OrchestrationTaskStatus::Failed => "failed",
        OrchestrationTaskStatus::Cancelled => "cancelled",
        OrchestrationTaskStatus::TimedOut => "timed_out",
        OrchestrationTaskStatus::Abandoned => "abandoned",
    }
}

/// Settles every live task matching `filter` into a terminal state.
///
/// A task with a cancellation already requested settles as cancelled — honouring
/// the intent that was recorded before the executor died. Every other live task
/// settles as failed with the caller's `reason`, because its driver went away
/// without producing a terminal event.
///
/// Already-terminal records are left untouched and do not appear in the report.
/// Per-task transition failures are recorded as [`ReconcileOutcome::Error`] and
/// never abort the sweep.
pub fn reconcile_orphaned_tasks(
    store: &dyn TaskStore,
    filter: OrchestrationTaskFilter,
    reason: &dyn Fn(&OrchestrationTaskRecord) -> String,
) -> ReconcileReport {
    let orphans: Vec<OrchestrationTaskRecord> = store
        .list(filter)
        .into_iter()
        .filter(|record| record.status.is_live())
        .collect();

    let mut report = ReconcileReport {
        tasks: Vec::with_capacity(orphans.len()),
    };

    for record in orphans {
        let task_id = record.spec.task_id.clone();
        let prior_status = record.status;

        let outcome = match prior_status {
            OrchestrationTaskStatus::CancelRequested => match store.mark_cancelled(&task_id) {
                Ok(_) => ReconcileOutcome::Cancelled,
                Err(err) => ReconcileOutcome::Error(err.to_string()),
            },
            _ => match store.fail(&task_id, reason(&record)) {
                Ok(_) => ReconcileOutcome::Failed,
                Err(err) => ReconcileOutcome::Error(err.to_string()),
            },
        };

        if let ReconcileOutcome::Error(detail) = &outcome {
            tracing::warn!(
                task_id = %task_id.as_str(),
                prior_status = task_status_label(prior_status),
                error = %detail,
                "[orchestration] failed to reconcile orphaned task"
            );
        }

        report.tasks.push(ReconciledTask {
            task_id,
            prior_status,
            outcome,
            record,
        });
    }

    report
}
