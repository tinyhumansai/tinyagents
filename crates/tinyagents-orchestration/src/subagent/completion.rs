//! Recording a finished subagent with the durable completion router.
//!
//! Opt-in: the driver only touches any of this when a
//! [`CompletionRouter`](tinyagents_tasks::CompletionRouter) was configured with
//! [`SubagentDriver::with_completion_router`](super::SubagentDriver::with_completion_router).

use std::sync::Arc;

use tinyagents_tasks::{
    CompletionArtifact, CompletionRecord, CompletionResult, CompletionRouter, CompletionStatus,
    NotifyMode,
};

use super::{
    AppliedResult, ArtifactReference, PreparedSubagent, SubagentOutcome, SubagentOutcomeKind,
    SubagentTaskKey,
};

const LOG_PREFIX: &str = "[subagent-completion]";

impl From<&ArtifactReference> for CompletionArtifact {
    fn from(artifact: &ArtifactReference) -> Self {
        Self {
            id: artifact.id.clone(),
            media_type: artifact.media_type.clone(),
            metadata: artifact.metadata.clone(),
        }
    }
}

impl From<&AppliedResult> for CompletionResult {
    /// The result policy's output as the parent will be shown it.
    fn from(applied: &AppliedResult) -> Self {
        Self {
            text: applied.text.clone(),
            omitted_chars: applied.omitted_chars,
            artifact: applied.artifact.as_ref().map(CompletionArtifact::from),
        }
    }
}

/// Who a finished child reports to, captured when the child is launched.
pub(crate) struct CompletionOrigin {
    task_id: String,
    parent_key: String,
    agent_id: String,
    notify_mode: NotifyMode,
}

impl CompletionOrigin {
    /// `None` when the spawn did not ask to be recorded.
    pub(crate) fn new<C>(
        task_key: &SubagentTaskKey,
        prepared: &PreparedSubagent<C>,
    ) -> Option<Self> {
        let notify_mode = prepared.notify_mode?;
        // The thread outlives a restart; the parent run id does not, so it is
        // only the fallback for a thread-less parent.
        let parent_key = prepared
            .completion_parent
            .clone()
            .or_else(|| task_key.thread_id.clone())
            .unwrap_or_else(|| task_key.parent_run_id.clone());
        Some(Self {
            task_id: task_key.task_id.clone(),
            parent_key,
            agent_id: prepared.agent_key.clone(),
            notify_mode,
        })
    }

    fn record(&self, status: CompletionStatus, result: CompletionResult) -> CompletionRecord {
        CompletionRecord::new(
            self.task_id.clone(),
            self.parent_key.clone(),
            self.agent_id.clone(),
            status,
            result,
        )
        .with_notify_mode(self.notify_mode)
    }

    /// The record for a finished lifecycle, or `None` when the outcome is not a
    /// completion: a cancellation is the parent's own doing, and a pause is not
    /// final (the same task id completes later).
    pub(crate) fn record_for_outcome(
        &self,
        outcome: &SubagentOutcome,
        omitted_chars: usize,
        overflow_artifact: Option<&ArtifactReference>,
    ) -> Option<CompletionRecord> {
        let (status, text) = match &outcome.status {
            SubagentOutcomeKind::Completed => (CompletionStatus::Success, outcome.output.clone()),
            SubagentOutcomeKind::Incomplete(incomplete) => (
                CompletionStatus::Incomplete,
                if outcome.output.is_empty() {
                    incomplete.reason.clone()
                } else {
                    outcome.output.clone()
                },
            ),
            SubagentOutcomeKind::AwaitingInput(_) | SubagentOutcomeKind::Cancelled => return None,
        };
        let result = CompletionResult {
            text,
            omitted_chars,
            // Only the artifact the result policy stored the full output in; the
            // executor's own artifacts are not the omitted output.
            artifact: overflow_artifact.map(CompletionArtifact::from),
        };
        Some(self.record(status, result))
    }
}

/// Hands `record` to the router. A router failure is logged, never raised: the
/// child's own result is already persisted and must not be lost to a
/// notification problem.
pub(crate) async fn deliver(router: &Arc<CompletionRouter>, record: CompletionRecord) {
    let task_id = record.task_id.clone();
    match router.record_with_retries(record, 3).await {
        Ok(outcome) => {
            tracing::debug!("{LOG_PREFIX} task_id={task_id} outcome={outcome:?}");
        }
        Err(error) => {
            tracing::warn!("{LOG_PREFIX} task_id={task_id} could not record completion: {error}");
        }
    }
}
