//! Driving the delegation graph: fresh runs, durable resumes, and the
//! checkpoint classification that decides between the two.

use std::future::Future;

use serde_json::{Value, json};

use super::graph::build_delegation_graph;
use super::types::{
    CURRENT_SCHEMA_VERSION, DelegationConfig, DelegationOutcome, DelegationStage,
    DelegationStageOutput, DelegationState, DelegationUpdate, PendingApproval,
};
use crate::graph::checkpoint::{Checkpoint, Checkpointer};
use crate::graph::{Command, END};

/// Run the plan→execute⇄review→finalize delegation graph, invoking `run_stage`
/// for each stage. Returns the final [`DelegationState`].
///
/// `run_stage` is the seam to the agent harness: production passes a closure that
/// dispatches each [`DelegationStage`] to `run_subagent`; tests pass a mock.
///
/// This is the non-gated convenience wrapper: with the default config
/// (`require_review_approval = false`) the graph never interrupts, so the
/// returned state is always terminal.
pub async fn run_delegation<F, Fut>(
    config: DelegationConfig,
    run_stage: F,
) -> Result<DelegationState, String>
where
    F: Fn(DelegationStage, DelegationState) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<DelegationStageOutput, String>> + Send + 'static,
{
    Ok(run_delegation_durable(config, run_stage).await?.state)
}

/// Run the delegation graph and report whether it finalized or parked on a
/// durable human-approval interrupt.
///
/// When [`DelegationConfig::require_review_approval`] is set and the reviewer
/// approves, the `approval` node emits [`NodeResult::Interrupt`]; the executor
/// persists a checkpoint (Sync durability — the crate default) and returns
/// control here with the interrupt in [`DelegationOutcome::pending`]. Deliver the
/// approver's decision later with [`resume_delegation`] — it may run after a
/// process restart, since the pause lives entirely in the checkpointer.
pub async fn run_delegation_durable<F, Fut>(
    config: DelegationConfig,
    run_stage: F,
) -> Result<DelegationOutcome, String>
where
    F: Fn(DelegationStage, DelegationState) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<DelegationStageOutput, String>> + Send + 'static,
{
    let thread_id = config.thread_id.clone();
    let mut graph = build_delegation_graph(
        config.max_revisions,
        config.cancel.clone(),
        config.require_review_approval,
        run_stage,
    )?;
    if let Some(sink) = config.event_sink.clone() {
        graph = graph.with_event_sink(sink);
    }

    if let Some(cp) = config.checkpointer {
        graph = graph.with_checkpointer(cp);
    }

    tracing::info!(
        max_revisions = config.max_revisions,
        durable = thread_id.is_some(),
        human_gated = config.require_review_approval,
        "[delegation] running sub-agent delegation graph"
    );

    let initial = DelegationState::new_run();
    let execution = match thread_id.clone() {
        Some(tid) => graph.run_with_thread(tid, initial).await,
        None => graph.run(initial).await,
    }
    .map_err(|e| format!("delegation graph run failed: {e}"))?;

    Ok(into_outcome(execution, thread_id))
}

/// Resume a delegation graph parked on a durable human-approval interrupt,
/// delivering the approver's `decision` through `Command { resume: .. }`.
///
/// The graph is rebuilt (its node closures are not serializable — only the typed
/// state is checkpointed) with the same checkpointer + `thread_id`, then
/// re-entered at the interrupted node via [`CompiledGraph::resume`] (the
/// `ResumeTarget::Latest` checkpoint). `decision` maps to approve/deny via
/// [`decision_is_approve`], so passing the approval RPC's `ApprovalDecision`
/// (serialized with its stable `as_str()` wire value — `approve_once` /
/// `approve_always_for_tool` / `deny`) routes the existing decision contract
/// into the resume **without changing that contract**.
///
/// TTL expiry → resume-with-deny: call this with [`deny_decision`] to preserve
/// the existing timeout-deny behavior for a pause that was never answered.
pub async fn resume_delegation<F, Fut>(
    config: DelegationConfig,
    decision: Value,
    run_stage: F,
) -> Result<DelegationOutcome, String>
where
    F: Fn(DelegationStage, DelegationState) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<DelegationStageOutput, String>> + Send + 'static,
{
    let approved = decision_is_approve(&decision);
    tracing::info!(
        approved,
        "[interrupt] resuming durable delegation graph with approval decision"
    );
    let command = Command {
        resume: Some(decision),
        ..Command::default()
    };
    resume_graph(config, command, run_stage).await
}

/// Run the delegation graph, resuming from the last checkpoint boundary when the
/// configured thread has a live, compatible, non-terminal checkpoint, else
/// starting fresh (issue #3884 — node-level checkpoint & resume).
///
/// Classifies the thread's latest checkpoint and routes accordingly:
/// - **resumable** (a crash/failure left a mid-run boundary) → re-run only the
///   not-yet-completed nodes from that boundary via [`CompiledGraph::resume`]
///   with an empty command — never restarting from `plan`, and never re-running
///   an already-completed step (its [`StepRecord`] is restored from the state);
/// - **terminal** (already finalized/cancelled) → return the stored final state
///   without re-running (idempotent re-invocation of a stable thread);
/// - **absent** (no checkpoint) → a fresh durable run;
/// - **incompatible** (an undecodable record — e.g. a pre-#3884 `Vec<String>`
///   `executions` shape) → log, best-effort prune, and start fresh. The decode
///   error is swallowed into a fresh-run decision, never propagated, never a panic.
///
/// Callers that mint a unique `thread_id` per run (today's default) always take
/// the fresh path unchanged; the resume paths activate only when a caller reuses
/// a stable `thread_id`, so this is byte-compatible for existing callers while
/// wiring the resume seam that #3881 (plan-edit resume) builds on.
pub async fn run_or_resume_delegation<F, Fut>(
    config: DelegationConfig,
    run_stage: F,
) -> Result<DelegationOutcome, String>
where
    F: Fn(DelegationStage, DelegationState) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<DelegationStageOutput, String>> + Send + 'static,
{
    // Without a checkpointer + thread there is nothing to resume from.
    let (Some(cp), Some(tid)) = (config.checkpointer.clone(), config.thread_id.clone()) else {
        return run_delegation_durable(config, run_stage).await;
    };

    match cp.get(tid.as_str(), None).await {
        // A checkpoint written under an older state schema (e.g. a pre-#3884
        // record whose `executions` happened to be empty and so still decoded
        // into `Vec<StepRecord>`) is expired rather than resumed/returned — its
        // semantics may not match the current graph. This is what makes
        // `schema_version` an actual guard, not just documentation, and closes
        // the empty-`executions` gap that a decode failure alone cannot catch.
        Ok(Some(checkpoint)) if checkpoint.state.schema_version < CURRENT_SCHEMA_VERSION => {
            tracing::warn!(
                thread_id = %tid,
                schema_version = checkpoint.state.schema_version,
                current = CURRENT_SCHEMA_VERSION,
                "[delegation] checkpoint predates the current state schema; pruning and starting fresh"
            );
            prune_thread(cp.as_ref(), &tid).await;
            run_delegation_durable(config, run_stage).await
        }
        Ok(Some(checkpoint)) if checkpoint_is_resumable(&checkpoint) => {
            tracing::info!(
                thread_id = %tid,
                "[delegation] resuming durable delegation from its last checkpoint boundary"
            );
            // A crash/failure resume carries no decision value — an empty command
            // simply re-runs the pending node(s) (the crate's `retry` semantics).
            resume_graph(config, Command::default(), run_stage).await
        }
        Ok(Some(checkpoint)) => {
            // Terminal: return the finalized state without re-running. Defensive:
            // if this checkpoint still carried an unconsumed interrupt, surface it
            // instead of silently dropping it. The current routing never produces
            // this (an interrupt boundary schedules its node and is classified
            // resumable above), but a future schedule could, and a dropped pause
            // would strand a run.
            let pending = checkpoint.interrupts.first().map(|i| PendingApproval {
                interrupt_id: i.id.clone(),
                node: i.node.as_str().to_string(),
                payload: i.payload.clone(),
                thread_id: tid.clone(),
            });
            if pending.is_some() {
                tracing::warn!(
                    thread_id = %tid,
                    "[delegation] terminal-classified checkpoint carried a pending interrupt; surfacing it"
                );
            } else {
                tracing::info!(
                    thread_id = %tid,
                    "[delegation] thread already terminal; returning finalized state without re-running"
                );
            }
            Ok(DelegationOutcome {
                state: checkpoint.state,
                pending,
            })
        }
        Ok(None) => {
            tracing::debug!(
                thread_id = %tid,
                "[delegation] no checkpoint for thread; starting a fresh durable run"
            );
            run_delegation_durable(config, run_stage).await
        }
        // Only a *decode / shape-incompatibility* read error expires the
        // checkpoint. An operational error (SQLite busy / I/O / poisoned lock)
        // must NOT silently restart a valid resumable run — it is propagated so
        // durable work is retried by the caller, not dropped.
        Err(e) if is_incompatible_checkpoint_error(&e) => {
            tracing::warn!(
                thread_id = %tid,
                error = %e,
                "[delegation] undecodable/incompatible checkpoint; pruning and starting fresh"
            );
            prune_thread(cp.as_ref(), &tid).await;
            run_delegation_durable(config, run_stage).await
        }
        Err(e) => {
            tracing::error!(
                thread_id = %tid,
                error = %e,
                "[delegation] checkpoint read failed (operational); not restarting — propagating error"
            );
            Err(format!(
                "delegation checkpoint read failed for thread {tid}: {e}"
            ))
        }
    }
}

/// Best-effort prune of a dead/expired checkpoint thread so it is not re-probed
/// forever. Failure to prune is non-fatal (logged at debug).
async fn prune_thread(cp: &dyn Checkpointer<DelegationState>, thread_id: &str) {
    if let Err(e) = cp.delete_thread(thread_id).await {
        tracing::debug!(
            thread_id = %thread_id,
            error = %e,
            "[delegation] could not prune checkpoint thread (non-fatal)"
        );
    }
}

/// Whether a `Checkpointer::get` error is a decode / shape-incompatibility (safe
/// to expire the checkpoint) rather than an operational failure (SQLite busy /
/// I/O / poisoned lock — must not silently restart durable work). The vendored
/// `SqliteCheckpointer` reports both as `TinyAgentsError::Checkpoint(String)` but
/// tags decode failures with a `"decode …"` context (`sqlite.rs`:
/// `decode record` / `decode namespace` / `decode next_nodes`) — the only stable
/// discriminator it exposes.
pub(super) fn is_incompatible_checkpoint_error(e: &crate::TinyAgentsError) -> bool {
    matches!(e, crate::TinyAgentsError::Checkpoint(msg) if msg.contains("decode"))
}

/// Whether a loaded checkpoint still has work to resume: a non-finalized,
/// non-cancelled run that still schedules a real (non-`END`) node.
fn checkpoint_is_resumable(checkpoint: &Checkpoint<DelegationState>) -> bool {
    if checkpoint.state.final_output.is_some() || checkpoint.state.cancelled {
        return false;
    }
    checkpoint.next_nodes.iter().any(|n| n.as_str() != END)
}

/// Rebuild the delegation graph (its node closures are not serializable — only
/// the typed state is checkpointed) with the same checkpointer + `thread_id`, and
/// re-enter it at the latest checkpoint's pending node(s) via
/// [`CompiledGraph::resume`]. `command` carries an approver's decision for the
/// human-approval interrupt path, or is empty for a plain crash/failure resume.
async fn resume_graph<F, Fut>(
    config: DelegationConfig,
    command: Command<DelegationUpdate>,
    run_stage: F,
) -> Result<DelegationOutcome, String>
where
    F: Fn(DelegationStage, DelegationState) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<DelegationStageOutput, String>> + Send + 'static,
{
    let thread_id = config
        .thread_id
        .clone()
        .ok_or_else(|| "delegation resume requires a thread_id".to_string())?;
    let cp = config
        .checkpointer
        .clone()
        .ok_or_else(|| "delegation resume requires a checkpointer".to_string())?;

    let mut graph = build_delegation_graph(
        config.max_revisions,
        config.cancel.clone(),
        config.require_review_approval,
        run_stage,
    )?
    .with_checkpointer(cp);
    if let Some(sink) = config.event_sink.clone() {
        graph = graph.with_event_sink(sink);
    }

    let execution = graph
        .resume(thread_id.clone(), command)
        .await
        .map_err(|e| format!("delegation graph resume failed: {e}"))?;

    Ok(into_outcome(execution, Some(thread_id)))
}

/// The canonical deny decision used for TTL-expiry resume (resume-with-deny),
/// serialized to the approval RPC's stable `deny` wire value.
pub fn deny_decision() -> Value {
    json!("deny")
}

/// Fold a finished/paused graph execution into a [`DelegationOutcome`],
/// surfacing the first pending interrupt (if the run parked on one).
fn into_outcome(
    execution: crate::graph::GraphExecution<DelegationState>,
    thread_id: Option<String>,
) -> DelegationOutcome {
    let pending = execution.interrupts.first().map(|i| {
        tracing::info!(
            interrupt_id = %i.id,
            node = %i.node.as_str(),
            "[interrupt] delegation run parked on durable human-approval interrupt"
        );
        PendingApproval {
            interrupt_id: i.id.clone(),
            node: i.node.as_str().to_string(),
            payload: i.payload.clone(),
            thread_id: thread_id.clone().unwrap_or_default(),
        }
    });
    DelegationOutcome {
        state: execution.state,
        pending,
    }
}

/// Map an approval decision value onto approve/deny. Accepts the approval RPC's
/// stable string forms (`approve_once`, `approve_always_for_tool`, `deny`), a
/// bare bool, or an object carrying `approved`/`decision` — so the existing
/// decision contract routes into `Command::resume` unchanged.
pub(super) fn decision_is_approve(decision: &Value) -> bool {
    match decision {
        Value::Bool(b) => *b,
        Value::String(s) => matches!(
            s.as_str(),
            "approve_once" | "approve_always_for_tool" | "approve" | "approved"
        ),
        Value::Object(m) => {
            if let Some(b) = m.get("approved").and_then(Value::as_bool) {
                return b;
            }
            m.get("decision")
                .and_then(Value::as_str)
                .map(|d| d.starts_with("approve"))
                .unwrap_or(false)
        }
        _ => false,
    }
}
