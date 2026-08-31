//! Module-local unit tests for [`crate::run_ledger`].
//!
//! Consolidated here per AGENTS.md: one `test.rs` per module directory.

use super::ops::*;
use super::types::*;
use chrono::Utc;
use serde_json::json;
use std::path::Path;
use tempfile::TempDir;

/// Workspace root for a test: the ledger derives its database path from
/// this, so a fresh `TempDir` per test gives a fresh database.
fn test_workspace(dir: &TempDir) -> &Path {
    dir.path()
}

// ── Regressions for the review findings on PR #90 ─────────────────────

/// An upsert that moves a claimed task off `in_progress` must drop the
/// claim, or the row is stranded: a new claim sees AlreadyClaimed,
/// completion sees NotClaimed, and release/shutdown skip it entirely.
#[test]
fn upsert_clears_the_claim_when_a_task_leaves_in_progress() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-strand");
    seed_member(workspace_dir, "team-strand", "m1");
    seed_task(workspace_dir, "team-strand", "task-strand", vec![]);

    let claimed =
        claim_agent_team_task(workspace_dir, "team-strand", "task-strand", "m1", "tok").unwrap();
    assert!(matches!(claimed, ClaimOutcome::Claimed(_)));

    // Recovery-style edit that resets status back to todo.
    upsert_agent_team_task(
        workspace_dir,
        AgentTeamTaskUpsert {
            id: "task-strand".into(),
            team_id: "team-strand".into(),
            title: "task task-strand".into(),
            objective: None,
            status: AgentTeamTaskStatus::Todo,
            owner_member_id: None,
            depends_on: vec![],
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        },
    )
    .unwrap();

    let after = get_agent_team_task(workspace_dir, "task-strand")
        .unwrap()
        .expect("task present");
    assert_eq!(after.claimed_by_member_id, None, "claim must be dropped");
    assert_eq!(after.claim_token, None);

    // ...and the task is claimable again rather than stranded.
    let reclaimed =
        claim_agent_team_task(workspace_dir, "team-strand", "task-strand", "m1", "tok2").unwrap();
    assert!(
        matches!(reclaimed, ClaimOutcome::Claimed(_)),
        "a released task must be re-claimable, got {reclaimed:?}"
    );
}

/// A live claim survives an unrelated edit that keeps the task in progress.
#[test]
fn upsert_preserves_a_live_claim_while_in_progress() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-keep");
    seed_member(workspace_dir, "team-keep", "m1");
    seed_task(workspace_dir, "team-keep", "task-keep", vec![]);
    claim_agent_team_task(workspace_dir, "team-keep", "task-keep", "m1", "tok").unwrap();

    upsert_agent_team_task(
        workspace_dir,
        AgentTeamTaskUpsert {
            id: "task-keep".into(),
            team_id: "team-keep".into(),
            title: "retitled".into(),
            objective: None,
            status: AgentTeamTaskStatus::InProgress,
            owner_member_id: None,
            depends_on: vec![],
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        },
    )
    .unwrap();

    let after = get_agent_team_task(workspace_dir, "task-keep")
        .unwrap()
        .expect("task present");
    assert_eq!(after.title, "retitled");
    assert_eq!(
        after.claimed_by_member_id.as_deref(),
        Some("m1"),
        "an in-progress edit must not steal the claim"
    );
}

/// A partial telemetry upsert must not fail the NOT NULL constraints.
///
/// The counters are `Option` so one field can be updated alone, but the
/// columns are `NOT NULL DEFAULT` and SQLite does not apply a column
/// default to an explicitly supplied NULL. Recording only `model` used to
/// fail outright on the first write for a run.
#[test]
fn partial_telemetry_upsert_applies_column_defaults() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);

    let telemetry = upsert_run_telemetry(
        workspace_dir,
        RunTelemetryUpsert {
            run_id: "run-partial".into(),
            model: Some("claude-x".into()),
            ..Default::default()
        },
    )
    .expect("a model-only upsert must succeed");

    assert_eq!(telemetry.model.as_deref(), Some("claude-x"));
    assert_eq!(telemetry.input_tokens, 0, "counters default, not NULL");
    assert_eq!(telemetry.output_tokens, 0);
    assert_eq!(telemetry.cost_usd, 0.0);
}

/// A later partial upsert must not clobber fields it does not carry.
#[test]
fn partial_telemetry_upsert_preserves_untouched_fields() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);

    upsert_run_telemetry(
        workspace_dir,
        RunTelemetryUpsert {
            run_id: "run-merge".into(),
            input_tokens: Some(120),
            model: Some("claude-x".into()),
            ..Default::default()
        },
    )
    .unwrap();

    // Second write carries only the error; everything else must survive.
    let merged = upsert_run_telemetry(
        workspace_dir,
        RunTelemetryUpsert {
            run_id: "run-merge".into(),
            error: Some("boom".into()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(merged.error.as_deref(), Some("boom"));
    assert_eq!(merged.input_tokens, 120, "prior counter must survive");
    assert_eq!(merged.model.as_deref(), Some("claude-x"));
}

/// Sequences are allocated by the INSERT itself, so appends stay dense and
/// ordered rather than racing on a read-then-write.
#[test]
fn run_event_sequences_are_allocated_by_the_insert() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);

    let seqs: Vec<u64> = (0..5)
        .map(|i| {
            append_run_event(
                workspace_dir,
                RunEventAppend {
                    run_id: "run-seq".into(),
                    event_type: format!("evt-{i}"),
                    payload: json!({ "i": i }),
                },
            )
            .unwrap()
            .sequence
        })
        .collect();

    assert_eq!(seqs, vec![1, 2, 3, 4, 5]);

    // A second run numbers independently from 1.
    let other = append_run_event(
        workspace_dir,
        RunEventAppend {
            run_id: "run-other".into(),
            event_type: "evt".into(),
            payload: json!({}),
        },
    )
    .unwrap();
    assert_eq!(other.sequence, 1);
}

#[test]
fn agent_run_append_list_get_and_events_are_ordered() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);

    let run = upsert_agent_run(
        workspace_dir,
        AgentRunUpsert {
            id: "run-1".into(),
            kind: AgentRunKind::Subagent,
            parent_run_id: Some("parent".into()),
            parent_thread_id: Some("thread-1".into()),
            agent_id: Some("researcher".into()),
            status: AgentRunStatus::Running,
            prompt_ref: Some("worker-1:user:seed".into()),
            worker_thread_id: Some("worker-1".into()),
            task_board_id: None,
            task_card_id: None,
            checkpoint_path: None,
            checkpoint: None,
            summary: None,
            error: None,
            metadata: json!({"source": "test"}),
            started_at: None,
            completed_at: None,
        },
    )
    .unwrap();
    assert_eq!(run.status, AgentRunStatus::Running);

    append_run_event(
        workspace_dir,
        RunEventAppend {
            run_id: "run-1".into(),
            event_type: "spawned".into(),
            payload: json!({"agentId": "researcher"}),
        },
    )
    .unwrap();
    append_run_event(
        workspace_dir,
        RunEventAppend {
            run_id: "run-1".into(),
            event_type: "completed".into(),
            payload: json!({"elapsedMs": 12}),
        },
    )
    .unwrap();

    let events = list_recent_run_events(
        workspace_dir,
        &RunEventListRequest {
            run_id: "run-1".into(),
            after_sequence: Some(0),
            limit: None,
        },
    )
    .unwrap();
    assert_eq!(events.events.len(), 2);
    assert_eq!(events.events[0].sequence, 1);
    assert_eq!(events.events[1].sequence, 2);

    let list = list_agent_runs(
        workspace_dir,
        &AgentRunListRequest {
            parent_thread_id: Some("thread-1".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(list.count, 1);
    assert_eq!(list.runs[0].worker_thread_id.as_deref(), Some("worker-1"));
}

#[test]
fn transition_sets_status_and_clears_error_and_completed_at() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);

    // Seed a failed run carrying an error + completion time.
    let completed_at = Utc::now();
    upsert_agent_run(
        workspace_dir,
        AgentRunUpsert {
            id: "run-1".into(),
            kind: AgentRunKind::Subagent,
            parent_run_id: None,
            parent_thread_id: Some("thread-1".into()),
            agent_id: Some("researcher".into()),
            status: AgentRunStatus::Failed,
            prompt_ref: None,
            worker_thread_id: None,
            task_board_id: None,
            task_card_id: None,
            checkpoint_path: None,
            checkpoint: None,
            summary: None,
            error: Some("boom".into()),
            metadata: json!({}),
            started_at: None,
            completed_at: Some(completed_at),
        },
    )
    .unwrap();

    // Re-queue: passing None for both columns must CLEAR them (the upsert
    // path's COALESCE cannot do this — that is the whole reason this op
    // exists).
    let updated =
        transition_agent_run_status(workspace_dir, "run-1", AgentRunStatus::Pending, None, None)
            .unwrap()
            .expect("run present");
    assert_eq!(updated.status, AgentRunStatus::Pending);
    assert_eq!(updated.error, None);
    assert_eq!(updated.completed_at, None);

    // Stopping: status + error + completion are all set verbatim.
    let stopped_at = Utc::now();
    let updated = transition_agent_run_status(
        workspace_dir,
        "run-1",
        AgentRunStatus::Cancelled,
        Some("manual"),
        Some(stopped_at),
    )
    .unwrap()
    .expect("run present");
    assert_eq!(updated.status, AgentRunStatus::Cancelled);
    assert_eq!(updated.error.as_deref(), Some("manual"));
    assert!(updated.completed_at.is_some());
}

#[test]
fn transition_unknown_run_returns_none() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    let result =
        transition_agent_run_status(workspace_dir, "ghost", AgentRunStatus::Pending, None, None)
            .unwrap();
    assert!(result.is_none());
}

fn seed_team(workspace_dir: &Path, team_id: &str) {
    upsert_agent_team(
        workspace_dir,
        AgentTeamUpsert {
            id: team_id.into(),
            parent_thread_id: Some("thread-team".into()),
            lead_agent_id: "lead".into(),
            status: AgentTeamStatus::Active,
            summary: None,
            created_at: None,
            closed_at: None,
        },
    )
    .unwrap();
}

fn seed_task(workspace_dir: &Path, team_id: &str, task_id: &str, depends_on: Vec<String>) {
    upsert_agent_team_task(
        workspace_dir,
        AgentTeamTaskUpsert {
            id: task_id.into(),
            team_id: team_id.into(),
            title: format!("task {task_id}"),
            objective: None,
            status: AgentTeamTaskStatus::Todo,
            owner_member_id: None,
            depends_on,
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        },
    )
    .unwrap();
}

#[test]
fn claim_is_atomic_first_wins_then_already_claimed() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-1");
    seed_task(workspace_dir, "team-1", "task-a", vec![]);

    let first = claim_agent_team_task(workspace_dir, "team-1", "task-a", "m1", "tok-1").unwrap();
    match first {
        ClaimOutcome::Claimed(task) => {
            assert_eq!(task.claimed_by_member_id.as_deref(), Some("m1"));
            assert_eq!(task.status, AgentTeamTaskStatus::InProgress);
        }
        other => panic!("expected Claimed, got {other:?}"),
    }

    let second = claim_agent_team_task(workspace_dir, "team-1", "task-a", "m2", "tok-2").unwrap();
    assert_eq!(second, ClaimOutcome::AlreadyClaimed);
}

#[test]
fn claim_unknown_task_returns_unknown() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-1");
    let outcome = claim_agent_team_task(workspace_dir, "team-1", "ghost", "m1", "tok").unwrap();
    assert_eq!(outcome, ClaimOutcome::UnknownTask);
}

fn seed_member(workspace_dir: &Path, team_id: &str, member_id: &str) {
    upsert_agent_team_member(
        workspace_dir,
        AgentTeamMemberUpsert {
            id: member_id.into(),
            team_id: team_id.into(),
            name: member_id.into(),
            agent_id: None,
            member_status: AgentTeamMemberStatus::Pending,
            current_task_id: None,
            worker_thread_id: None,
            run_id: None,
            created_at: None,
        },
    )
    .unwrap();
}

#[test]
fn mark_member_running_then_idle_keeps_run_pointer() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-1");
    seed_member(workspace_dir, "team-1", "m1");
    seed_task(workspace_dir, "team-1", "task-a", vec![]);
    claim_agent_team_task(workspace_dir, "team-1", "task-a", "m1", "tok-1").unwrap();

    let running = mark_agent_team_member_running(
        workspace_dir,
        "team-1",
        "m1",
        "task-a",
        "worker-x",
        "run-x",
    )
    .unwrap()
    .expect("member updated");
    assert_eq!(running.member_status, AgentTeamMemberStatus::Active);
    assert_eq!(running.current_task_id.as_deref(), Some("task-a"));
    assert_eq!(running.worker_thread_id.as_deref(), Some("worker-x"));
    assert_eq!(running.run_id.as_deref(), Some("run-x"));

    let idle = mark_agent_team_member_idle(workspace_dir, "team-1", "m1")
        .unwrap()
        .expect("member updated");
    assert_eq!(idle.member_status, AgentTeamMemberStatus::Idle);
    assert_eq!(idle.current_task_id, None);
    // worker/run pointer retained as last-run history.
    assert_eq!(idle.worker_thread_id.as_deref(), Some("worker-x"));
    assert_eq!(idle.run_id.as_deref(), Some("run-x"));

    // Unknown member → None, no-op.
    assert!(
        mark_agent_team_member_running(workspace_dir, "team-1", "ghost", "task-a", "w", "r")
            .unwrap()
            .is_none()
    );
    assert!(
        mark_agent_team_member_idle(workspace_dir, "team-1", "ghost")
            .unwrap()
            .is_none()
    );
}

#[test]
fn release_task_frees_in_progress_only() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-1");
    seed_member(workspace_dir, "team-1", "m1");
    seed_task(workspace_dir, "team-1", "task-a", vec![]);
    claim_agent_team_task(workspace_dir, "team-1", "task-a", "m1", "tok-1").unwrap();

    // In progress → released back to todo, claim cleared, gate reset.
    assert!(release_agent_team_task(workspace_dir, "team-1", "task-a").unwrap());
    let task = get_agent_team_task(workspace_dir, "task-a")
        .unwrap()
        .unwrap();
    assert_eq!(task.status, AgentTeamTaskStatus::Todo);
    assert_eq!(task.claimed_by_member_id, None);
    assert_eq!(task.claim_token, None);
    assert_eq!(task.gate_status, "pending");

    // Already todo (not in_progress) → no-op, returns false.
    assert!(!release_agent_team_task(workspace_dir, "team-1", "task-a").unwrap());
    // Unknown task → false.
    assert!(!release_agent_team_task(workspace_dir, "team-1", "ghost").unwrap());
}

#[test]
fn claim_blocked_until_dependency_done() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-1");
    seed_task(workspace_dir, "team-1", "task-a", vec![]);
    seed_task(workspace_dir, "team-1", "task-b", vec!["task-a".into()]);

    // B is blocked while A is still todo.
    let blocked = claim_agent_team_task(workspace_dir, "team-1", "task-b", "m1", "tok").unwrap();
    assert_eq!(
        blocked,
        ClaimOutcome::Blocked {
            unmet: vec!["task-a".into()]
        }
    );

    // Mark A done, then B claims fine.
    upsert_agent_team_task(
        workspace_dir,
        AgentTeamTaskUpsert {
            id: "task-a".into(),
            team_id: "team-1".into(),
            title: "task task-a".into(),
            objective: None,
            status: AgentTeamTaskStatus::Done,
            owner_member_id: None,
            depends_on: vec![],
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        },
    )
    .unwrap();

    let ok = claim_agent_team_task(workspace_dir, "team-1", "task-b", "m1", "tok").unwrap();
    assert!(matches!(ok, ClaimOutcome::Claimed(_)));
}

#[test]
fn team_members_and_tasks_list_back() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);
    seed_team(workspace_dir, "team-1");
    upsert_agent_team_member(
        workspace_dir,
        AgentTeamMemberUpsert {
            id: "mem-1".into(),
            team_id: "team-1".into(),
            name: "alice".into(),
            agent_id: Some("researcher".into()),
            member_status: AgentTeamMemberStatus::Active,
            current_task_id: None,
            worker_thread_id: None,
            run_id: None,
            created_at: None,
        },
    )
    .unwrap();
    seed_task(workspace_dir, "team-1", "task-a", vec![]);

    let members = list_agent_team_members(workspace_dir, "team-1").unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "alice");

    let tasks = list_agent_team_tasks(workspace_dir, "team-1").unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, "task-a");

    let teams = list_agent_teams(workspace_dir, &AgentTeamListRequest::default()).unwrap();
    assert_eq!(teams.count, 1);
}

fn seed_run(workspace_dir: &Path, id: &str, status: AgentRunStatus) {
    upsert_agent_run(
        workspace_dir,
        AgentRunUpsert {
            id: id.into(),
            kind: AgentRunKind::Subagent,
            parent_run_id: None,
            parent_thread_id: Some("thread-1".into()),
            agent_id: Some("tinyplace_agent".into()),
            status,
            prompt_ref: None,
            worker_thread_id: None,
            task_board_id: None,
            task_card_id: None,
            checkpoint_path: None,
            checkpoint: None,
            summary: None,
            error: None,
            metadata: json!({}),
            started_at: None,
            completed_at: None,
        },
    )
    .unwrap();
}

#[test]
fn interrupt_orphaned_runs_settles_only_non_terminal_inflight_rows() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = test_workspace(&dir);

    seed_run(workspace_dir, "run-running", AgentRunStatus::Running);
    seed_run(workspace_dir, "run-pending", AgentRunStatus::Pending);
    seed_run(workspace_dir, "run-completed", AgentRunStatus::Completed);
    seed_run(workspace_dir, "run-awaiting", AgentRunStatus::AwaitingUser);

    let settled = interrupt_orphaned_agent_runs(workspace_dir).unwrap();
    assert_eq!(settled, 2, "only running + pending are orphaned at boot");

    let get = |id: &str| {
        get_agent_run(workspace_dir, id)
            .unwrap()
            .expect("run present")
    };
    // Orphaned in-flight rows become terminal `interrupted` with a completion time…
    let running = get("run-running");
    assert_eq!(running.status, AgentRunStatus::Interrupted);
    assert!(running.completed_at.is_some());
    assert_eq!(get("run-pending").status, AgentRunStatus::Interrupted);
    // …already-terminal and resumable rows are untouched.
    assert_eq!(get("run-completed").status, AgentRunStatus::Completed);
    assert_eq!(get("run-awaiting").status, AgentRunStatus::AwaitingUser);

    // Idempotent: a second sweep finds nothing left to settle.
    assert_eq!(interrupt_orphaned_agent_runs(workspace_dir).unwrap(), 0);
}
