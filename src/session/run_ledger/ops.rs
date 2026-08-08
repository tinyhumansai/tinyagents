use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};

use crate::error::Result;

use super::super::context::StorageContext;
use super::store::init_run_ledger_schema;
use super::types::{
    AgentRun, AgentRunListRequest, AgentRunListResponse, AgentRunStatus, AgentRunUpsert, AgentTeam,
    AgentTeamListRequest, AgentTeamListResponse, AgentTeamMember, AgentTeamMemberStatus,
    AgentTeamMemberUpsert, AgentTeamStatus, AgentTeamTask, AgentTeamTaskStatus,
    AgentTeamTaskUpsert, AgentTeamUpsert, ClaimOutcome, CompletionOutcome, RunEvent,
    RunEventAppend, RunEventListRequest, RunEventListResponse, RunTelemetry, RunTelemetryUpsert,
    WorkflowRun, WorkflowRunListRequest, WorkflowRunListResponse, WorkflowRunUpsert,
};

const LOG_PREFIX: &str = "[session_db:run_ledger]";

pub fn upsert_agent_run(workspace_dir: &Path, upsert: AgentRunUpsert) -> Result<AgentRun> {
    let now = Utc::now();
    let started_at = upsert.started_at.unwrap_or(now);
    let updated_at = now;
    let metadata_json =
        serde_json::to_string(&upsert.metadata).storage_context("serialize agent run metadata")?;
    let checkpoint_json = upsert
        .checkpoint
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .storage_context("serialize agent run checkpoint")?;

    tracing::debug!(
        "{LOG_PREFIX} upsert_agent_run id={} kind={} status={} parent={} thread={}",
        upsert.id,
        upsert.kind.as_str(),
        upsert.status.as_str(),
        upsert.parent_run_id.as_deref().unwrap_or("-"),
        upsert.parent_thread_id.as_deref().unwrap_or("-")
    );

    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_runs (
                id, kind, parent_run_id, parent_thread_id, agent_id, status,
                prompt_ref, worker_thread_id, task_board_id, task_card_id,
                checkpoint_path, checkpoint_json, summary, error, metadata_json,
                started_at, updated_at, completed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
             ON CONFLICT(id) DO UPDATE SET
                kind = CASE
                    WHEN agent_runs.kind = 'worker_thread' AND excluded.kind = 'subagent' THEN agent_runs.kind
                    ELSE excluded.kind
                END,
                parent_run_id = COALESCE(excluded.parent_run_id, agent_runs.parent_run_id),
                parent_thread_id = COALESCE(excluded.parent_thread_id, agent_runs.parent_thread_id),
                agent_id = COALESCE(excluded.agent_id, agent_runs.agent_id),
                status = excluded.status,
                prompt_ref = COALESCE(excluded.prompt_ref, agent_runs.prompt_ref),
                worker_thread_id = COALESCE(excluded.worker_thread_id, agent_runs.worker_thread_id),
                task_board_id = COALESCE(excluded.task_board_id, agent_runs.task_board_id),
                task_card_id = COALESCE(excluded.task_card_id, agent_runs.task_card_id),
                checkpoint_path = COALESCE(excluded.checkpoint_path, agent_runs.checkpoint_path),
                checkpoint_json = COALESCE(excluded.checkpoint_json, agent_runs.checkpoint_json),
                summary = COALESCE(excluded.summary, agent_runs.summary),
                error = COALESCE(excluded.error, agent_runs.error),
                metadata_json = CASE
                    WHEN excluded.metadata_json = '{}' THEN agent_runs.metadata_json
                    ELSE excluded.metadata_json
                END,
                updated_at = excluded.updated_at,
                completed_at = COALESCE(excluded.completed_at, agent_runs.completed_at)",
            params![
                upsert.id,
                upsert.kind.as_str(),
                upsert.parent_run_id,
                upsert.parent_thread_id,
                upsert.agent_id,
                upsert.status.as_str(),
                upsert.prompt_ref,
                upsert.worker_thread_id,
                upsert.task_board_id,
                upsert.task_card_id,
                upsert.checkpoint_path,
                checkpoint_json,
                upsert.summary,
                upsert.error,
                metadata_json,
                started_at.to_rfc3339(),
                updated_at.to_rfc3339(),
                upsert.completed_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .storage_context("upsert agent run")?;
        Ok(())
    })?;

    get_agent_run(workspace_dir, &upsert.id)?.storage_context("agent run missing after upsert")
}

pub fn upsert_workflow_run(workspace_dir: &Path, upsert: WorkflowRunUpsert) -> Result<WorkflowRun> {
    let now = Utc::now();
    let started_at = upsert.started_at.unwrap_or(now);
    let input_json =
        serde_json::to_string(&upsert.input).storage_context("serialize workflow input")?;
    let phase_states_json = serde_json::to_string(&upsert.phase_states)
        .storage_context("serialize workflow phase states")?;
    let child_run_ids_json =
        serde_json::to_string(&upsert.child_run_ids).storage_context("serialize child run ids")?;

    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO workflow_runs (
                id, definition_id, parent_thread_id, input_json, phase_states_json,
                child_run_ids_json, status, summary, started_at, updated_at, completed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(id) DO UPDATE SET
                definition_id = excluded.definition_id,
                parent_thread_id = COALESCE(excluded.parent_thread_id, workflow_runs.parent_thread_id),
                input_json = excluded.input_json,
                phase_states_json = excluded.phase_states_json,
                child_run_ids_json = excluded.child_run_ids_json,
                status = excluded.status,
                summary = COALESCE(excluded.summary, workflow_runs.summary),
                updated_at = excluded.updated_at,
                completed_at = COALESCE(excluded.completed_at, workflow_runs.completed_at)",
            params![
                upsert.id,
                upsert.definition_id,
                upsert.parent_thread_id,
                input_json,
                phase_states_json,
                child_run_ids_json,
                upsert.status.as_str(),
                upsert.summary,
                started_at.to_rfc3339(),
                now.to_rfc3339(),
                upsert.completed_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .storage_context("upsert workflow run")?;
        Ok(())
    })?;

    get_workflow_run(workspace_dir, &upsert.id)?
        .storage_context("workflow run missing after upsert")
}

pub fn append_run_event(workspace_dir: &Path, event: RunEventAppend) -> Result<RunEvent> {
    let now = Utc::now();
    let payload_json =
        serde_json::to_string(&event.payload).storage_context("serialize run event")?;
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        // Allocate and insert the sequence in ONE statement. Reading
        // `MAX(sequence) + 1` and then inserting is a read-modify-write race:
        // two connections appending for the same run can read the same next
        // value, and the loser fails the `(run_id, sequence)` primary key —
        // silently dropping a real run event unless every caller implements an
        // undocumented retry. The sub-select is evaluated inside the same
        // statement, so SQLite's write lock serializes the whole allocation.
        let next_sequence: i64 = conn
            .query_row(
                "INSERT INTO run_events (run_id, sequence, event_type, payload_json, timestamp)
             VALUES (
                ?1,
                (SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?1),
                ?2, ?3, ?4
             )
             RETURNING sequence",
                params![
                    event.run_id,
                    event.event_type,
                    payload_json,
                    now.to_rfc3339(),
                ],
                |row| row.get(0),
            )
            .storage_context("append run event")?;
        Ok(RunEvent {
            run_id: event.run_id,
            sequence: next_sequence as u64,
            event_type: event.event_type,
            payload: serde_json::from_str(&payload_json).unwrap_or_else(|_| json!({})),
            timestamp: now,
        })
    })
}

pub fn upsert_run_telemetry(
    workspace_dir: &Path,
    upsert: RunTelemetryUpsert,
) -> Result<RunTelemetry> {
    let now = Utc::now();
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            // The counters are `Option` so a caller can update one field without
            // clobbering the rest, but the columns are `NOT NULL DEFAULT`, and
            // SQLite does NOT apply a column default to an explicitly supplied
            // NULL. Binding the raw `None` therefore made every partial upsert
            // (say, recording only `model` or only `error`) fail a NOT NULL
            // constraint on first write. The insert side coalesces to the
            // column default; the update side re-reads the SAME parameter and
            // coalesces to the stored value, which keeps per-field optionality.
            // `excluded.*` cannot serve the update side here — it observes the
            // already-coalesced insert row, so a `None` would read as 0 and
            // overwrite the stored counter.
            "INSERT INTO run_telemetry (
                run_id, input_tokens, output_tokens, cached_input_tokens, cost_usd,
                elapsed_ms, tool_count, model, provider, error, updated_at
             ) VALUES (
                ?1,
                COALESCE(?2, 0), COALESCE(?3, 0), COALESCE(?4, 0), COALESCE(?5, 0.0),
                ?6, COALESCE(?7, 0), ?8, ?9, ?10, ?11
             )
             ON CONFLICT(run_id) DO UPDATE SET
                input_tokens = COALESCE(?2, run_telemetry.input_tokens),
                output_tokens = COALESCE(?3, run_telemetry.output_tokens),
                cached_input_tokens = COALESCE(?4, run_telemetry.cached_input_tokens),
                cost_usd = COALESCE(?5, run_telemetry.cost_usd),
                elapsed_ms = COALESCE(?6, run_telemetry.elapsed_ms),
                tool_count = COALESCE(?7, run_telemetry.tool_count),
                model = COALESCE(?8, run_telemetry.model),
                provider = COALESCE(?9, run_telemetry.provider),
                error = COALESCE(?10, run_telemetry.error),
                updated_at = ?11",
            params![
                upsert.run_id,
                upsert.input_tokens.map(|v| v as i64),
                upsert.output_tokens.map(|v| v as i64),
                upsert.cached_input_tokens.map(|v| v as i64),
                upsert.cost_usd,
                upsert.elapsed_ms.map(|v| v as i64),
                upsert.tool_count.map(|v| v as i64),
                upsert.model,
                upsert.provider,
                upsert.error,
                now.to_rfc3339(),
            ],
        )
        .storage_context("upsert run telemetry")?;
        get_run_telemetry_inner(conn, &upsert.run_id)
    })
}

pub fn get_agent_run(workspace_dir: &Path, id: &str) -> Result<Option<AgentRun>> {
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        get_agent_run_inner(conn, id)
    })
}

/// Apply a durable status transition to a single agent run.
///
/// Unlike [`upsert_agent_run`] — whose `ON CONFLICT` clause `COALESCE`s the
/// `error` and `completed_at` columns and can therefore only ever *set* them —
/// this is a direct `UPDATE` that can both set and *clear* both columns. That
/// is required by control verbs such as "retry", which moves a failed run back
/// to `pending` and must drop the stale failure reason and completion time.
///
/// `status` is always written. `error` and `completed_at` are written verbatim,
/// so passing `None` clears the column. `updated_at` is bumped to now. Returns
/// the freshly-read run, or `None` when no row matched `id` (e.g. it was
/// deleted between a prior read and this write).
pub fn transition_agent_run_status(
    workspace_dir: &Path,
    id: &str,
    status: AgentRunStatus,
    error: Option<&str>,
    completed_at: Option<DateTime<Utc>>,
) -> Result<Option<AgentRun>> {
    let now = Utc::now();
    tracing::debug!(
        "{LOG_PREFIX} transition_agent_run_status id={id} status={} has_error={} has_completed_at={}",
        status.as_str(),
        error.is_some(),
        completed_at.is_some()
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let rows_affected = conn
            .execute(
                "UPDATE agent_runs
                 SET status = ?1, error = ?2, completed_at = ?3, updated_at = ?4
                 WHERE id = ?5",
                params![
                    status.as_str(),
                    error,
                    completed_at.map(|dt| dt.to_rfc3339()),
                    now.to_rfc3339(),
                    id,
                ],
            )
            .storage_context("transition agent run status")?;
        if rows_affected == 0 {
            tracing::debug!("{LOG_PREFIX} transition_agent_run_status.miss id={id}");
            return Ok(None);
        }
        get_agent_run_inner(conn, id)
    })
}

/// Settle non-terminal `agent_runs` rows left behind by a previous process.
///
/// A freshly-booted core has no in-flight subagents — any detached run task
/// from a prior process is gone with that process. So a row still marked
/// `running` (or `pending`) at startup is, by definition, orphaned: its driver
/// died without firing a terminal `DomainEvent::Subagent{Completed,Failed}`, so
/// the [`register_run_ledger_finalize_subscriber`] never settled it. Without
/// this sweep those rows render as perpetual "running" timeline entries on every
/// thread reopen.
///
/// We stamp them `interrupted` (outcome unknown — mirrors the turn-state
/// `mark_all_interrupted` recovery) and set `completed_at`. `awaiting_user` /
/// `paused` are intentionally left untouched: those are resumable states a user
/// may still continue.
///
/// [`register_run_ledger_finalize_subscriber`]: crate::openhuman::agent::orchestration::run_ledger_finalize::register_run_ledger_finalize_subscriber
pub fn interrupt_orphaned_agent_runs(workspace_dir: &Path) -> Result<usize> {
    let now = Utc::now();
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let rows_affected = conn
            .execute(
                "UPDATE agent_runs
                 SET status = ?1, completed_at = COALESCE(completed_at, ?2), updated_at = ?2
                 WHERE status IN ('running', 'pending')",
                params![AgentRunStatus::Interrupted.as_str(), now.to_rfc3339()],
            )
            .storage_context("interrupt orphaned agent runs")?;
        if rows_affected > 0 {
            tracing::info!(
                "{LOG_PREFIX} interrupted {rows_affected} orphaned agent run(s) on startup"
            );
        }
        Ok(rows_affected)
    })
}

pub fn list_agent_runs(
    workspace_dir: &Path,
    request: &AgentRunListRequest,
) -> Result<AgentRunListResponse> {
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut where_clauses = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(status) = request.status.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(status.to_string()));
            where_clauses.push(format!("status = ?{}", values.len()));
        }
        if let Some(kind) = request.kind.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(kind.to_string()));
            where_clauses.push(format!("kind = ?{}", values.len()));
        }
        if let Some(parent) = request
            .parent_run_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(parent.to_string()));
            where_clauses.push(format!("parent_run_id = ?{}", values.len()));
        }
        if let Some(thread) = request
            .parent_thread_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(thread.to_string()));
            where_clauses.push(format!("parent_thread_id = ?{}", values.len()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let count_sql = format!("SELECT COUNT(*) FROM agent_runs {where_sql}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let count = conn.query_row(&count_sql, params_ref.as_slice(), |row| {
            row.get::<_, i64>(0)
        })? as usize;

        let limit = request.limit.unwrap_or(50).min(500) as i64;
        let offset = request.offset.unwrap_or(0) as i64;
        values.push(Box::new(limit));
        let limit_idx = values.len();
        values.push(Box::new(offset));
        let offset_idx = values.len();

        let query_sql = format!(
            "SELECT id, kind, parent_run_id, parent_thread_id, agent_id, status,
                    prompt_ref, worker_thread_id, task_board_id, task_card_id,
                    checkpoint_path, checkpoint_json, summary, error, metadata_json,
                    started_at, updated_at, completed_at
             FROM agent_runs {where_sql}
             ORDER BY updated_at DESC
             LIMIT ?{limit_idx} OFFSET ?{offset_idx}"
        );
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), |row| map_agent_run_row(conn, row))?;
        let mut runs = Vec::new();
        for row in rows {
            runs.push(row?);
        }
        Ok(AgentRunListResponse { runs, count })
    })
}

pub fn list_recent_run_events(
    workspace_dir: &Path,
    request: &RunEventListRequest,
) -> Result<RunEventListResponse> {
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let limit = request.limit.unwrap_or(100).min(1000) as i64;
        let after = request.after_sequence.unwrap_or(0) as i64;
        let mut stmt = conn.prepare(
            "SELECT run_id, sequence, event_type, payload_json, timestamp
             FROM run_events
             WHERE run_id = ?1 AND sequence > ?2
             ORDER BY sequence ASC
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![request.run_id, after, limit], map_run_event_row)?;
        let mut events = Vec::new();
        for row in rows {
            events.push(row?);
        }
        Ok(RunEventListResponse {
            count: events.len(),
            events,
        })
    })
}

pub fn get_workflow_run(workspace_dir: &Path, id: &str) -> Result<Option<WorkflowRun>> {
    tracing::debug!("{LOG_PREFIX} get_workflow_run.entry id={id}");
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut stmt = conn.prepare(
            "SELECT id, definition_id, parent_thread_id, input_json, phase_states_json,
                    child_run_ids_json, status, summary, started_at, updated_at, completed_at
             FROM workflow_runs WHERE id = ?1",
        )?;
        let run = stmt
            .query_row(params![id], map_workflow_run_row)
            .optional()?;
        tracing::debug!(
            "{LOG_PREFIX} get_workflow_run.exit id={id} found={}",
            run.is_some()
        );
        Ok(run)
    })
}

/// List durable workflow runs, most-recently-updated first, with optional
/// filters (definition id, status, parent thread) and pagination. Mirrors
/// [`list_agent_runs`] for the workflow_runs table.
pub fn list_workflow_runs(
    workspace_dir: &Path,
    request: &WorkflowRunListRequest,
) -> Result<WorkflowRunListResponse> {
    tracing::debug!(
        "{LOG_PREFIX} list_workflow_runs.entry definition={:?} status={:?} parent_thread={:?} limit={:?} offset={:?}",
        request.definition_id,
        request.status,
        request.parent_thread_id,
        request.limit,
        request.offset
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut where_clauses = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(definition) = request
            .definition_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(definition.to_string()));
            where_clauses.push(format!("definition_id = ?{}", values.len()));
        }
        if let Some(status) = request.status.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(status.to_string()));
            where_clauses.push(format!("status = ?{}", values.len()));
        }
        if let Some(thread) = request
            .parent_thread_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(thread.to_string()));
            where_clauses.push(format!("parent_thread_id = ?{}", values.len()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let count_sql = format!("SELECT COUNT(*) FROM workflow_runs {where_sql}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let count = conn.query_row(&count_sql, params_ref.as_slice(), |row| {
            row.get::<_, i64>(0)
        })? as usize;

        let limit = request.limit.unwrap_or(50).min(500) as i64;
        // `offset` is `u64`; convert checked so a value > i64::MAX surfaces a
        // clear error instead of wrapping negative and corrupting pagination.
        let offset = i64::try_from(request.offset.unwrap_or(0))
            .storage_context("workflow run list offset exceeds i64::MAX")?;
        values.push(Box::new(limit));
        let limit_idx = values.len();
        values.push(Box::new(offset));
        let offset_idx = values.len();

        let query_sql = format!(
            "SELECT id, definition_id, parent_thread_id, input_json, phase_states_json,
                    child_run_ids_json, status, summary, started_at, updated_at, completed_at
             FROM workflow_runs {where_sql}
             ORDER BY updated_at DESC
             LIMIT ?{limit_idx} OFFSET ?{offset_idx}"
        );
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), map_workflow_run_row)?;
        let mut runs = Vec::new();
        for row in rows {
            runs.push(row?);
        }
        tracing::debug!(
            "{LOG_PREFIX} list_workflow_runs.exit count={count} returned={}",
            runs.len()
        );
        Ok(WorkflowRunListResponse { runs, count })
    })
}

// ---------------------------------------------------------------------------
// Agent-team coordination (issue #3374)
// ---------------------------------------------------------------------------

/// Insert or update a team row.
pub fn upsert_agent_team(workspace_dir: &Path, upsert: AgentTeamUpsert) -> Result<AgentTeam> {
    let now = Utc::now();
    let created_at = upsert.created_at.unwrap_or(now);
    tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team.entry id={} lead={} status={}",
        upsert.id,
        upsert.lead_agent_id,
        upsert.status.as_str()
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_teams (
                id, parent_thread_id, lead_agent_id, status, summary,
                created_at, updated_at, closed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(id) DO UPDATE SET
                parent_thread_id = COALESCE(excluded.parent_thread_id, agent_teams.parent_thread_id),
                lead_agent_id = excluded.lead_agent_id,
                status = excluded.status,
                summary = COALESCE(excluded.summary, agent_teams.summary),
                updated_at = excluded.updated_at,
                closed_at = COALESCE(excluded.closed_at, agent_teams.closed_at)",
            params![
                upsert.id,
                upsert.parent_thread_id,
                upsert.lead_agent_id,
                upsert.status.as_str(),
                upsert.summary,
                created_at.to_rfc3339(),
                now.to_rfc3339(),
                upsert.closed_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .storage_context("upsert agent team")?;
        Ok(())
    })?;
    let team = get_agent_team(workspace_dir, &upsert.id)?
        .storage_context("agent team missing after upsert")?;
    tracing::debug!("{LOG_PREFIX} upsert_agent_team.exit id={}", team.id);
    Ok(team)
}

/// Fetch a single team by id.
pub fn get_agent_team(workspace_dir: &Path, id: &str) -> Result<Option<AgentTeam>> {
    tracing::debug!("{LOG_PREFIX} get_agent_team.entry id={id}");
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let team = get_agent_team_inner(conn, id)?;
        tracing::debug!(
            "{LOG_PREFIX} get_agent_team.exit id={id} found={}",
            team.is_some()
        );
        Ok(team)
    })
}

/// List teams, most-recently-updated first, with optional thread/status filters.
pub fn list_agent_teams(
    workspace_dir: &Path,
    request: &AgentTeamListRequest,
) -> Result<AgentTeamListResponse> {
    tracing::debug!(
        "{LOG_PREFIX} list_agent_teams.entry parent_thread={:?} status={:?} limit={:?} offset={:?}",
        request.parent_thread_id,
        request.status,
        request.limit,
        request.offset
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut where_clauses = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(thread) = request
            .parent_thread_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(thread.to_string()));
            where_clauses.push(format!("parent_thread_id = ?{}", values.len()));
        }
        if let Some(status) = request.status.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(status.to_string()));
            where_clauses.push(format!("status = ?{}", values.len()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let count_sql = format!("SELECT COUNT(*) FROM agent_teams {where_sql}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let count = conn.query_row(&count_sql, params_ref.as_slice(), |row| {
            row.get::<_, i64>(0)
        })? as usize;

        let limit = request.limit.unwrap_or(50).min(500) as i64;
        // `offset` is `u64`; convert checked so a value > i64::MAX surfaces a
        // clear error instead of wrapping negative and corrupting pagination.
        let offset = i64::try_from(request.offset.unwrap_or(0))
            .storage_context("agent team list offset exceeds i64::MAX")?;
        values.push(Box::new(limit));
        let limit_idx = values.len();
        values.push(Box::new(offset));
        let offset_idx = values.len();

        let query_sql = format!(
            "SELECT id, parent_thread_id, lead_agent_id, status, summary,
                    created_at, updated_at, closed_at
             FROM agent_teams {where_sql}
             ORDER BY updated_at DESC
             LIMIT ?{limit_idx} OFFSET ?{offset_idx}"
        );
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), map_agent_team_row)?;
        let mut teams = Vec::new();
        for row in rows {
            teams.push(row?);
        }
        tracing::debug!(
            "{LOG_PREFIX} list_agent_teams.exit count={count} returned={}",
            teams.len()
        );
        Ok(AgentTeamListResponse { teams, count })
    })
}

/// Insert or update a team member. `UNIQUE(team_id, name)` enforces unique names.
pub fn upsert_agent_team_member(
    workspace_dir: &Path,
    upsert: AgentTeamMemberUpsert,
) -> Result<AgentTeamMember> {
    let now = Utc::now();
    let created_at = upsert.created_at.unwrap_or(now);
    tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team_member.entry id={} team={} name={} status={}",
        upsert.id,
        upsert.team_id,
        upsert.name,
        upsert.member_status.as_str()
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_team_members (
                id, team_id, name, agent_id, member_status,
                current_task_id, worker_thread_id, run_id, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                agent_id = COALESCE(excluded.agent_id, agent_team_members.agent_id),
                member_status = excluded.member_status,
                current_task_id = COALESCE(excluded.current_task_id, agent_team_members.current_task_id),
                worker_thread_id = COALESCE(excluded.worker_thread_id, agent_team_members.worker_thread_id),
                run_id = COALESCE(excluded.run_id, agent_team_members.run_id),
                updated_at = excluded.updated_at",
            params![
                upsert.id,
                upsert.team_id,
                upsert.name,
                upsert.agent_id,
                upsert.member_status.as_str(),
                upsert.current_task_id,
                upsert.worker_thread_id,
                upsert.run_id,
                created_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )
        .storage_context("upsert agent team member")?;
        Ok(())
    })?;
    let member = get_agent_team_member(workspace_dir, &upsert.id)?
        .storage_context("agent team member missing after upsert")?;
    tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team_member.exit id={}",
        member.id
    );
    Ok(member)
}

/// Fetch a single member by id.
pub fn get_agent_team_member(workspace_dir: &Path, id: &str) -> Result<Option<AgentTeamMember>> {
    tracing::debug!("{LOG_PREFIX} get_agent_team_member.entry id={id}");
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let member = get_agent_team_member_inner(conn, id)?;
        tracing::debug!(
            "{LOG_PREFIX} get_agent_team_member.exit id={id} found={}",
            member.is_some()
        );
        Ok(member)
    })
}

/// List all members of a team, by creation order.
pub fn list_agent_team_members(
    workspace_dir: &Path,
    team_id: &str,
) -> Result<Vec<AgentTeamMember>> {
    tracing::debug!("{LOG_PREFIX} list_agent_team_members.entry team={team_id}");
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut stmt = conn.prepare(
            "SELECT id, team_id, name, agent_id, member_status,
                    current_task_id, worker_thread_id, run_id, created_at, updated_at
             FROM agent_team_members WHERE team_id = ?1
             ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map(params![team_id], map_agent_team_member_row)?;
        let mut members = Vec::new();
        for row in rows {
            members.push(row?);
        }
        tracing::debug!(
            "{LOG_PREFIX} list_agent_team_members.exit team={team_id} count={}",
            members.len()
        );
        Ok(members)
    })
}

/// Insert or update a team task.
pub fn upsert_agent_team_task(
    workspace_dir: &Path,
    upsert: AgentTeamTaskUpsert,
) -> Result<AgentTeamTask> {
    let now = Utc::now();
    let created_at = upsert.created_at.unwrap_or(now);
    let depends_on_json =
        serde_json::to_string(&upsert.depends_on).storage_context("serialize task depends_on")?;
    let evidence_json =
        serde_json::to_string(&upsert.evidence).storage_context("serialize task evidence")?;
    let gate_status = upsert.gate_status.unwrap_or_else(|| "pending".to_string());
    tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team_task.entry id={} team={} status={} deps={}",
        upsert.id,
        upsert.team_id,
        upsert.status.as_str(),
        upsert.depends_on.len()
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_team_tasks (
                id, team_id, title, objective, status, owner_member_id,
                claimed_by_member_id, claim_token, depends_on_json, gate_status,
                gate_reason, evidence_json, source_run_id, order_index,
                created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
             ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                objective = COALESCE(excluded.objective, agent_team_tasks.objective),
                status = excluded.status,
                -- A claim only means anything while the task is in_progress.
                -- Editing a claimed task through this upsert (to change
                -- dependencies, or to reset status during recovery) used to
                -- leave claimed_by_member_id/claim_token set on a todo row,
                -- which strands it: a fresh claim returns AlreadyClaimed,
                -- completion returns NotClaimed, and release/shutdown skip it
                -- because they only match in_progress. Drop the claim whenever
                -- the new status is not in_progress; preserve it otherwise so
                -- an unrelated edit does not steal a live claim.
                claimed_by_member_id = CASE
                    WHEN excluded.status = 'in_progress'
                    THEN agent_team_tasks.claimed_by_member_id ELSE NULL END,
                claim_token = CASE
                    WHEN excluded.status = 'in_progress'
                    THEN agent_team_tasks.claim_token ELSE NULL END,
                owner_member_id = COALESCE(excluded.owner_member_id, agent_team_tasks.owner_member_id),
                depends_on_json = excluded.depends_on_json,
                gate_status = excluded.gate_status,
                gate_reason = COALESCE(excluded.gate_reason, agent_team_tasks.gate_reason),
                evidence_json = excluded.evidence_json,
                source_run_id = COALESCE(excluded.source_run_id, agent_team_tasks.source_run_id),
                order_index = excluded.order_index,
                updated_at = excluded.updated_at",
            params![
                upsert.id,
                upsert.team_id,
                upsert.title,
                upsert.objective,
                upsert.status.as_str(),
                upsert.owner_member_id,
                depends_on_json,
                gate_status,
                upsert.gate_reason,
                evidence_json,
                upsert.source_run_id,
                upsert.order_index,
                created_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )
        .storage_context("upsert agent team task")?;
        Ok(())
    })?;
    let task = get_agent_team_task(workspace_dir, &upsert.id)?
        .storage_context("agent team task missing after upsert")?;
    tracing::debug!("{LOG_PREFIX} upsert_agent_team_task.exit id={}", task.id);
    Ok(task)
}

/// Fetch a single task by id.
pub fn get_agent_team_task(workspace_dir: &Path, id: &str) -> Result<Option<AgentTeamTask>> {
    tracing::debug!("{LOG_PREFIX} get_agent_team_task.entry id={id}");
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let task = get_agent_team_task_inner(conn, id)?;
        tracing::debug!(
            "{LOG_PREFIX} get_agent_team_task.exit id={id} found={}",
            task.is_some()
        );
        Ok(task)
    })
}

/// List all tasks of a team, by `order_index` then creation order.
pub fn list_agent_team_tasks(workspace_dir: &Path, team_id: &str) -> Result<Vec<AgentTeamTask>> {
    tracing::debug!("{LOG_PREFIX} list_agent_team_tasks.entry team={team_id}");
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut stmt = conn.prepare(
            "SELECT id, team_id, title, objective, status, owner_member_id,
                    claimed_by_member_id, claim_token, depends_on_json, gate_status,
                    gate_reason, evidence_json, source_run_id, order_index,
                    created_at, updated_at
             FROM agent_team_tasks WHERE team_id = ?1
             ORDER BY order_index ASC, created_at ASC",
        )?;
        let rows = stmt.query_map(params![team_id], map_agent_team_task_row)?;
        let mut tasks = Vec::new();
        for row in rows {
            tasks.push(row?);
        }
        tracing::debug!(
            "{LOG_PREFIX} list_agent_team_tasks.exit team={team_id} count={}",
            tasks.len()
        );
        Ok(tasks)
    })
}

/// Atomically claim a task for a member.
///
/// All steps run inside a single `with_connection` transaction so that the
/// dependency check and the compare-and-swap observe a consistent snapshot:
/// 1. Resolve the task by `(id, team_id)`; absent → [`ClaimOutcome::UnknownTask`].
/// 2. For every dependency id, look up its status; collect those not `done`
///    into `unmet`. Non-empty → [`ClaimOutcome::Blocked`].
/// 3. WHERE-guarded `UPDATE ... WHERE claimed_by_member_id IS NULL`: SQLite
///    serializes writers, so exactly one concurrent claimer flips the row from
///    unclaimed to claimed. `rows_affected == 0` → already taken
///    ([`ClaimOutcome::AlreadyClaimed`]); otherwise re-fetch and return
///    [`ClaimOutcome::Claimed`].
pub fn claim_agent_team_task(
    workspace_dir: &Path,
    team_id: &str,
    task_id: &str,
    member_id: &str,
    claim_token: &str,
) -> Result<ClaimOutcome> {
    tracing::debug!(
        "{LOG_PREFIX} claim_agent_team_task.entry team={team_id} task={task_id} member={member_id}"
    );
    let outcome = crate::session::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;

        // 1. Resolve the task within this team.
        let task = match get_agent_team_task_inner(conn, task_id)? {
            Some(task) if task.team_id == team_id => task,
            _ => {
                tracing::debug!(
                    "{LOG_PREFIX} claim_agent_team_task.unknown team={team_id} task={task_id}"
                );
                return Ok(ClaimOutcome::UnknownTask);
            }
        };

        // 2. Dependency gate: every dep must be `done`.
        let mut unmet = Vec::new();
        for dep_id in &task.depends_on {
            let dep_status: Option<String> = conn
                .query_row(
                    "SELECT status FROM agent_team_tasks WHERE id = ?1 AND team_id = ?2",
                    params![dep_id, team_id],
                    |row| row.get(0),
                )
                .optional()?;
            let is_done = dep_status.as_deref() == Some(AgentTeamTaskStatus::Done.as_str());
            if !is_done {
                unmet.push(dep_id.clone());
            }
        }
        if !unmet.is_empty() {
            tracing::debug!(
                "{LOG_PREFIX} claim_agent_team_task.blocked team={team_id} task={task_id} unmet={}",
                unmet.len()
            );
            return Ok(ClaimOutcome::Blocked { unmet });
        }

        // 3. Compare-and-swap on the unclaimed guard.
        let now = Utc::now();
        let rows_affected = conn
            .execute(
                "UPDATE agent_team_tasks
                 SET claimed_by_member_id = ?1, claim_token = ?2, status = 'in_progress', updated_at = ?3
                 WHERE id = ?4 AND team_id = ?5 AND claimed_by_member_id IS NULL",
                params![member_id, claim_token, now.to_rfc3339(), task_id, team_id],
            )
            .storage_context("compare-and-swap claim agent team task")?;
        if rows_affected == 0 {
            tracing::debug!(
                "{LOG_PREFIX} claim_agent_team_task.already_claimed team={team_id} task={task_id}"
            );
            return Ok(ClaimOutcome::AlreadyClaimed);
        }

        let claimed = get_agent_team_task_inner(conn, task_id)?
            .storage_context("claimed task missing after compare-and-swap")?;
        Ok(ClaimOutcome::Claimed(Box::new(claimed)))
    })?;
    tracing::debug!(
        "{LOG_PREFIX} claim_agent_team_task.exit team={team_id} task={task_id} outcome={}",
        match &outcome {
            ClaimOutcome::Claimed(_) => "claimed",
            ClaimOutcome::AlreadyClaimed => "already_claimed",
            ClaimOutcome::Blocked { .. } => "blocked",
            ClaimOutcome::UnknownTask => "unknown",
        }
    );
    Ok(outcome)
}

/// Quality-gate a task's completion and, on pass, transition it to `done`.
///
/// Runs inside a single transaction so the gate evaluation and the status flip
/// observe one consistent snapshot:
/// 1. Resolve the task by `(id, team_id)`; absent → [`CompletionOutcome::UnknownTask`].
/// 2. The completer must be the current claimant and the task must be
///    `in_progress`; otherwise [`CompletionOutcome::NotClaimed`].
/// 3. Evaluate the quality gate (every dependency `done`, claimant matches any
///    pre-assigned owner, evidence present when `require_evidence`). Any unmet
///    invariant records `gate_status = "failed"` + the joined reasons and leaves
///    the task `in_progress` → [`CompletionOutcome::GateFailed`].
/// 4. On pass, merge `evidence`, set `status = "done"`, `gate_status = "passed"`,
///    clear `gate_reason`, re-fetch → [`CompletionOutcome::Completed`].
pub fn complete_agent_team_task(
    workspace_dir: &Path,
    team_id: &str,
    task_id: &str,
    member_id: &str,
    evidence: &[String],
    require_evidence: bool,
) -> Result<CompletionOutcome> {
    tracing::debug!(
        "{LOG_PREFIX} complete_agent_team_task.entry team={team_id} task={task_id} member={member_id}"
    );
    let outcome = crate::session::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;

        // 1. Resolve the task within this team.
        let task = match get_agent_team_task_inner(conn, task_id)? {
            Some(task) if task.team_id == team_id => task,
            _ => {
                tracing::debug!(
                    "{LOG_PREFIX} complete_agent_team_task.unknown team={team_id} task={task_id}"
                );
                return Ok(CompletionOutcome::UnknownTask);
            }
        };

        // 2. Only the current claimant may complete, and only while in progress.
        let is_claimant = task.claimed_by_member_id.as_deref() == Some(member_id);
        let in_progress = task.status == AgentTeamTaskStatus::InProgress;
        if !is_claimant || !in_progress {
            tracing::debug!(
                "{LOG_PREFIX} complete_agent_team_task.not_claimed team={team_id} task={task_id} claimant={is_claimant} in_progress={in_progress}"
            );
            return Ok(CompletionOutcome::NotClaimed);
        }

        // Merge prior evidence with the newly-supplied links (de-duplicated,
        // order-preserving) so a retry that adds evidence accumulates it.
        let mut merged_evidence = task.evidence.clone();
        for link in evidence {
            if !merged_evidence.iter().any(|e| e == link) {
                merged_evidence.push(link.clone());
            }
        }

        // 3. Quality gate.
        let reasons =
            evaluate_completion_gate(conn, team_id, &task, &merged_evidence, require_evidence)?;
        let now = Utc::now();
        if !reasons.is_empty() {
            let joined = reasons.join("; ");
            // Persist the merged evidence even though the gate failed. Evidence
            // accumulates across attempts (see `merged_evidence` above), so
            // dropping it here punished a caller for an unrelated gate failure:
            // after a dependency was fixed, a retry that did not resend the same
            // links would fail `require_evidence` on evidence it had already
            // submitted. Only the gate verdict is a failure; the submission is
            // still real.
            let evidence_json = serde_json::to_string(&merged_evidence)
                .storage_context("serialize completion evidence")?;
            conn.execute(
                "UPDATE agent_team_tasks
                 SET gate_status = 'failed', gate_reason = ?1,
                     evidence_json = ?2, updated_at = ?3
                 WHERE id = ?4 AND team_id = ?5",
                params![joined, evidence_json, now.to_rfc3339(), task_id, team_id],
            )
            .storage_context("record failed completion gate")?;
            tracing::debug!(
                "{LOG_PREFIX} complete_agent_team_task.gate_failed team={team_id} task={task_id} reasons={}",
                reasons.len()
            );
            return Ok(CompletionOutcome::GateFailed { reasons });
        }

        // 4. Gate passed — flip to done. The WHERE clause is the real CAS: the
        // `claimed_by_member_id` guard stops a concurrent shutdown/unclaim from
        // completing a task it no longer holds, and the `status = 'in_progress'`
        // guard stops a concurrent double-complete by the same member (the
        // snapshot check above is a read, not part of the swap — only one of two
        // racing UPDATEs flips `in_progress -> done`).
        let evidence_json = serde_json::to_string(&merged_evidence)
            .storage_context("serialize completion evidence")?;
        let rows_affected = conn
            .execute(
                "UPDATE agent_team_tasks
                 SET status = 'done', gate_status = 'passed', gate_reason = NULL,
                     evidence_json = ?1, updated_at = ?2
                 WHERE id = ?3 AND team_id = ?4 AND claimed_by_member_id = ?5
                   AND status = 'in_progress'",
                params![evidence_json, now.to_rfc3339(), task_id, team_id, member_id],
            )
            .storage_context("complete agent team task")?;
        if rows_affected == 0 {
            tracing::debug!(
                "{LOG_PREFIX} complete_agent_team_task.lost_claim team={team_id} task={task_id}"
            );
            return Ok(CompletionOutcome::NotClaimed);
        }

        let done = get_agent_team_task_inner(conn, task_id)?
            .storage_context("completed task missing after update")?;
        Ok(CompletionOutcome::Completed(Box::new(done)))
    })?;
    tracing::debug!(
        "{LOG_PREFIX} complete_agent_team_task.exit team={team_id} task={task_id} outcome={}",
        match &outcome {
            CompletionOutcome::Completed(_) => "completed",
            CompletionOutcome::GateFailed { .. } => "gate_failed",
            CompletionOutcome::NotClaimed => "not_claimed",
            CompletionOutcome::UnknownTask => "unknown",
        }
    );
    Ok(outcome)
}

/// Evaluate the quality-gate invariants for a completing task. Returns one
/// human-readable reason per unmet invariant (empty = gate passes).
fn evaluate_completion_gate(
    conn: &Connection,
    team_id: &str,
    task: &AgentTeamTask,
    merged_evidence: &[String],
    require_evidence: bool,
) -> Result<Vec<String>> {
    let mut reasons = Vec::new();

    // Every dependency must still be `done` (defends against a dependency that
    // regressed after this task was claimed).
    for dep_id in &task.depends_on {
        let dep_status: Option<String> = conn
            .query_row(
                "SELECT status FROM agent_team_tasks WHERE id = ?1 AND team_id = ?2",
                params![dep_id, team_id],
                |row| row.get(0),
            )
            .optional()?;
        if dep_status.as_deref() != Some(AgentTeamTaskStatus::Done.as_str()) {
            reasons.push(format!("dependency {dep_id} is not done"));
        }
    }

    // No overlapping ownership: a pre-assigned owner must be the one completing.
    if let Some(owner) = task
        .owner_member_id
        .as_ref()
        .filter(|owner| Some(owner.as_str()) != task.claimed_by_member_id.as_deref())
    {
        reasons.push(format!(
            "task is owned by {owner} but claimed by {}",
            task.claimed_by_member_id.as_deref().unwrap_or("nobody")
        ));
    }

    // Evidence gate.
    if require_evidence && merged_evidence.is_empty() {
        reasons.push("completion requires at least one evidence link".to_string());
    }

    Ok(reasons)
}

/// Stop a team member and release any task it is actively working on.
///
/// In one transaction: unclaim the member's `in_progress` tasks back to `todo`
/// (clearing claimant + token so another teammate can pick them up), then mark
/// the member `stopped` and clear its `current_task_id`. Returns the updated
/// member plus the ids of the tasks that were released, or `None` if the member
/// is not part of the team.
pub fn shutdown_agent_team_member(
    workspace_dir: &Path,
    team_id: &str,
    member_id: &str,
) -> Result<Option<(AgentTeamMember, Vec<String>)>> {
    tracing::debug!(
        "{LOG_PREFIX} shutdown_agent_team_member.entry team={team_id} member={member_id}"
    );
    let result = crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;

        // Existence + team-membership check only; the row is intentionally not
        // reused — the caller-facing member is re-read after the UPDATEs below so
        // it reflects the stopped state.
        match get_agent_team_member_inner(conn, member_id)? {
            Some(found) if found.team_id == team_id => {}
            _ => {
                tracing::debug!(
                    "{LOG_PREFIX} shutdown_agent_team_member.unknown team={team_id} member={member_id}"
                );
                return Ok(None);
            }
        }

        // Collect the ids first so the caller can report exactly what was freed.
        let released: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT id FROM agent_team_tasks
                 WHERE team_id = ?1 AND claimed_by_member_id = ?2 AND status = 'in_progress'",
            )?;
            let ids = stmt.query_map(params![team_id, member_id], |row| row.get::<_, String>(0))?;
            let mut out = Vec::new();
            for id in ids {
                out.push(id?);
            }
            out
        };

        let now = Utc::now();
        conn.execute(
            "UPDATE agent_team_tasks
             SET claimed_by_member_id = NULL, claim_token = NULL, status = 'todo', updated_at = ?1
             WHERE team_id = ?2 AND claimed_by_member_id = ?3 AND status = 'in_progress'",
            params![now.to_rfc3339(), team_id, member_id],
        )
        .storage_context("release tasks on member shutdown")?;
        conn.execute(
            "UPDATE agent_team_members
             SET member_status = 'stopped', current_task_id = NULL, updated_at = ?1
             WHERE id = ?2 AND team_id = ?3",
            params![now.to_rfc3339(), member_id, team_id],
        )
        .storage_context("stop agent team member")?;

        let member = get_agent_team_member_inner(conn, member_id)?
            .storage_context("member missing after shutdown")?;
        Ok(Some((member, released)))
    })?;
    tracing::debug!(
        "{LOG_PREFIX} shutdown_agent_team_member.exit team={team_id} member={member_id} released={}",
        result.as_ref().map(|(_, r)| r.len()).unwrap_or(0)
    );
    Ok(result)
}

/// Mark a member as actively running a task: status → `active`, with the
/// current task id and the worker/run identifiers of the spawned agent. Used by
/// the live runtime right after it claims a task and dispatches a worker.
/// Returns the updated member, or `None` if the member is not in the team.
pub fn mark_agent_team_member_running(
    workspace_dir: &Path,
    team_id: &str,
    member_id: &str,
    task_id: &str,
    worker_thread_id: &str,
    run_id: &str,
) -> Result<Option<AgentTeamMember>> {
    tracing::debug!(
        "{LOG_PREFIX} mark_agent_team_member_running.entry team={team_id} member={member_id} task={task_id} run={run_id}"
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let now = Utc::now();
        let changed = conn
            .execute(
                "UPDATE agent_team_members
                 SET member_status = 'active', current_task_id = ?1,
                     worker_thread_id = ?2, run_id = ?3, updated_at = ?4
                 WHERE id = ?5 AND team_id = ?6",
                params![
                    task_id,
                    worker_thread_id,
                    run_id,
                    now.to_rfc3339(),
                    member_id,
                    team_id
                ],
            )
            .storage_context("mark agent team member running")?;
        if changed == 0 {
            return Ok(None);
        }
        get_agent_team_member_inner(conn, member_id)
    })
}

/// Mark a member idle: status → `idle`, clearing `current_task_id`. The
/// `worker_thread_id` / `run_id` are intentionally retained as a pointer to the
/// member's last run for history. Returns the updated member, or `None` if the
/// member is not in the team. Used when a worker run finishes (completed,
/// gate-failed, or failed) so the member is free to pick up new work.
pub fn mark_agent_team_member_idle(
    workspace_dir: &Path,
    team_id: &str,
    member_id: &str,
) -> Result<Option<AgentTeamMember>> {
    tracing::debug!(
        "{LOG_PREFIX} mark_agent_team_member_idle.entry team={team_id} member={member_id}"
    );
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let now = Utc::now();
        let changed = conn
            .execute(
                "UPDATE agent_team_members
                 SET member_status = 'idle', current_task_id = NULL, updated_at = ?1
                 WHERE id = ?2 AND team_id = ?3",
                params![now.to_rfc3339(), member_id, team_id],
            )
            .storage_context("mark agent team member idle")?;
        if changed == 0 {
            return Ok(None);
        }
        get_agent_team_member_inner(conn, member_id)
    })
}

/// Release a single `in_progress` task back to `todo`, clearing its claim and
/// resetting the quality gate. Returns `true` if a row was actually released
/// (the task existed, belonged to the team, and was `in_progress`). Used by the
/// live runtime when a worker run fails or is aborted, so the task is free for
/// another teammate — the per-task analogue of the bulk release in
/// `shutdown_agent_team_member`.
pub fn release_agent_team_task(workspace_dir: &Path, team_id: &str, task_id: &str) -> Result<bool> {
    tracing::debug!("{LOG_PREFIX} release_agent_team_task.entry team={team_id} task={task_id}");
    crate::session::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let now = Utc::now();
        let changed = conn
            .execute(
                "UPDATE agent_team_tasks
                 SET status = 'todo', claimed_by_member_id = NULL, claim_token = NULL,
                     gate_status = 'pending', gate_reason = NULL, updated_at = ?1
                 WHERE id = ?2 AND team_id = ?3 AND status = 'in_progress'",
                params![now.to_rfc3339(), task_id, team_id],
            )
            .storage_context("release agent team task")?;
        tracing::debug!(
            "{LOG_PREFIX} release_agent_team_task.exit team={team_id} task={task_id} released={}",
            changed > 0
        );
        Ok(changed > 0)
    })
}

fn get_agent_team_inner(conn: &Connection, id: &str) -> Result<Option<AgentTeam>> {
    let mut stmt = conn.prepare(
        "SELECT id, parent_thread_id, lead_agent_id, status, summary,
                created_at, updated_at, closed_at
         FROM agent_teams WHERE id = ?1",
    )?;
    stmt.query_row(params![id], map_agent_team_row)
        .optional()
        .map_err(Into::into)
}

fn get_agent_team_member_inner(conn: &Connection, id: &str) -> Result<Option<AgentTeamMember>> {
    let mut stmt = conn.prepare(
        "SELECT id, team_id, name, agent_id, member_status,
                current_task_id, worker_thread_id, run_id, created_at, updated_at
         FROM agent_team_members WHERE id = ?1",
    )?;
    stmt.query_row(params![id], map_agent_team_member_row)
        .optional()
        .map_err(Into::into)
}

fn get_agent_team_task_inner(conn: &Connection, id: &str) -> Result<Option<AgentTeamTask>> {
    let mut stmt = conn.prepare(
        "SELECT id, team_id, title, objective, status, owner_member_id,
                claimed_by_member_id, claim_token, depends_on_json, gate_status,
                gate_reason, evidence_json, source_run_id, order_index,
                created_at, updated_at
         FROM agent_team_tasks WHERE id = ?1",
    )?;
    stmt.query_row(params![id], map_agent_team_task_row)
        .optional()
        .map_err(Into::into)
}

fn map_agent_team_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentTeam> {
    Ok(AgentTeam {
        id: row.get(0)?,
        parent_thread_id: row.get(1)?,
        lead_agent_id: row.get(2)?,
        status: AgentTeamStatus::parse(&row.get::<_, String>(3)?),
        summary: row.get(4)?,
        created_at: parse_rfc3339(&row.get::<_, String>(5)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(6)?)?,
        closed_at: parse_rfc3339_opt(row.get(7)?)?,
    })
}

fn map_agent_team_member_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentTeamMember> {
    Ok(AgentTeamMember {
        id: row.get(0)?,
        team_id: row.get(1)?,
        name: row.get(2)?,
        agent_id: row.get(3)?,
        member_status: AgentTeamMemberStatus::parse(&row.get::<_, String>(4)?),
        current_task_id: row.get(5)?,
        worker_thread_id: row.get(6)?,
        run_id: row.get(7)?,
        created_at: parse_rfc3339(&row.get::<_, String>(8)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(9)?)?,
    })
}

fn map_agent_team_task_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentTeamTask> {
    Ok(AgentTeamTask {
        id: row.get(0)?,
        team_id: row.get(1)?,
        title: row.get(2)?,
        objective: row.get(3)?,
        status: AgentTeamTaskStatus::parse(&row.get::<_, String>(4)?),
        owner_member_id: row.get(5)?,
        claimed_by_member_id: row.get(6)?,
        claim_token: row.get(7)?,
        depends_on: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
        gate_status: row.get(9)?,
        gate_reason: row.get(10)?,
        evidence: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
        source_run_id: row.get(12)?,
        order_index: row.get(13)?,
        created_at: parse_rfc3339(&row.get::<_, String>(14)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(15)?)?,
    })
}

fn get_agent_run_inner(conn: &Connection, id: &str) -> Result<Option<AgentRun>> {
    let mut stmt = conn.prepare(
        "SELECT id, kind, parent_run_id, parent_thread_id, agent_id, status,
                prompt_ref, worker_thread_id, task_board_id, task_card_id,
                checkpoint_path, checkpoint_json, summary, error, metadata_json,
                started_at, updated_at, completed_at
         FROM agent_runs WHERE id = ?1",
    )?;
    stmt.query_row(params![id], |row| map_agent_run_row(conn, row))
        .optional()
        .map_err(Into::into)
}

fn get_run_telemetry_inner(conn: &Connection, run_id: &str) -> Result<RunTelemetry> {
    let mut stmt = conn.prepare(
        "SELECT run_id, input_tokens, output_tokens, cached_input_tokens, cost_usd,
                elapsed_ms, tool_count, model, provider, error, updated_at
         FROM run_telemetry WHERE run_id = ?1",
    )?;
    stmt.query_row(params![run_id], map_run_telemetry_row)
        .storage_context("run telemetry missing after upsert")
}

fn get_optional_run_telemetry(
    conn: &Connection,
    run_id: &str,
) -> rusqlite::Result<Option<RunTelemetry>> {
    let mut stmt = conn.prepare(
        "SELECT run_id, input_tokens, output_tokens, cached_input_tokens, cost_usd,
                elapsed_ms, tool_count, model, provider, error, updated_at
         FROM run_telemetry WHERE run_id = ?1",
    )?;
    stmt.query_row(params![run_id], map_run_telemetry_row)
        .optional()
}

fn map_agent_run_row(conn: &Connection, row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentRun> {
    let id: String = row.get(0)?;
    let checkpoint_json: Option<String> = row.get(11)?;
    let metadata_json: String = row.get(14)?;
    Ok(AgentRun {
        id: id.clone(),
        kind: super::types::AgentRunKind::parse(&row.get::<_, String>(1)?),
        parent_run_id: row.get(2)?,
        parent_thread_id: row.get(3)?,
        agent_id: row.get(4)?,
        status: AgentRunStatus::parse(&row.get::<_, String>(5)?),
        prompt_ref: row.get(6)?,
        worker_thread_id: row.get(7)?,
        task_board_id: row.get(8)?,
        task_card_id: row.get(9)?,
        checkpoint_path: row.get(10)?,
        checkpoint: parse_json_opt(checkpoint_json),
        summary: row.get(12)?,
        error: row.get(13)?,
        metadata: parse_json(metadata_json),
        telemetry: get_optional_run_telemetry(conn, &id)?,
        started_at: parse_rfc3339(&row.get::<_, String>(15)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(16)?)?,
        completed_at: parse_rfc3339_opt(row.get(17)?)?,
    })
}

fn map_workflow_run_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<WorkflowRun> {
    Ok(WorkflowRun {
        id: row.get(0)?,
        definition_id: row.get(1)?,
        parent_thread_id: row.get(2)?,
        input: parse_json(row.get(3)?),
        phase_states: parse_json(row.get(4)?),
        child_run_ids: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
        status: super::types::WorkflowRunStatus::parse(&row.get::<_, String>(6)?),
        summary: row.get(7)?,
        started_at: parse_rfc3339(&row.get::<_, String>(8)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(9)?)?,
        completed_at: parse_rfc3339_opt(row.get(10)?)?,
    })
}

fn map_run_event_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunEvent> {
    Ok(RunEvent {
        run_id: row.get(0)?,
        sequence: row.get::<_, i64>(1)? as u64,
        event_type: row.get(2)?,
        payload: parse_json(row.get(3)?),
        timestamp: parse_rfc3339(&row.get::<_, String>(4)?)?,
    })
}

fn map_run_telemetry_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunTelemetry> {
    Ok(RunTelemetry {
        run_id: row.get(0)?,
        input_tokens: row.get::<_, i64>(1)? as u64,
        output_tokens: row.get::<_, i64>(2)? as u64,
        cached_input_tokens: row.get::<_, i64>(3)? as u64,
        cost_usd: row.get(4)?,
        elapsed_ms: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
        tool_count: row.get::<_, i64>(6)? as u64,
        model: row.get(7)?,
        provider: row.get(8)?,
        error: row.get(9)?,
        updated_at: Some(parse_rfc3339(&row.get::<_, String>(10)?)?),
    })
}

fn parse_json(raw: String) -> Value {
    serde_json::from_str(&raw).unwrap_or_else(|_| json!({}))
}

fn parse_json_opt(raw: Option<String>) -> Option<Value> {
    raw.and_then(|value| serde_json::from_str(&value).ok())
}

fn parse_rfc3339(raw: &str) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(err))
        })
}

fn parse_rfc3339_opt(raw: Option<String>) -> rusqlite::Result<Option<DateTime<Utc>>> {
    match raw {
        Some(value) => parse_rfc3339(&value).map(Some),
        None => Ok(None),
    }
}
