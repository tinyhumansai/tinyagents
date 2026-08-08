//! Retention, pruning and search-index repair for the session database.
//!
//! # Why this module exists
//!
//! Until it did, the session database had **no delete path at all**: sessions,
//! messages, tool calls, run events and telemetry accumulated forever, and the
//! only `DELETE FROM` anywhere in the crate lived in the graph checkpointer. A
//! long-lived workspace therefore grew without bound, and its full-text index
//! grew with it.
//!
//! It also had no way to *repair* the search index. `record_message` and its
//! siblings used to insert the row and its FTS entry as two separate autocommit
//! statements, so a failure in between left a row that could never be found by
//! search — permanently, because nothing re-derived the index. Those pairs are
//! transactional now, but databases written before that change still carry the
//! gaps, which is what [`reindex_fts`] is for.
//!
//! # Shape
//!
//! Every entry point takes an explicit bound (`older_than` / `keep_last`) and
//! returns the number of rows removed. Nothing here runs on a schedule or on
//! its own: retention is a policy decision, so the host chooses when and how
//! much. All of them run inside one transaction, so a partial prune is never
//! observable.

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::params;

use super::context::StorageContext;
use super::ops::fts_snippet_bytes;
use super::store::with_transaction;
use crate::error::Result;

/// Grep prefix for retention logging.
const LOG_PREFIX: &str = "[session_db:retention]";

/// What a single retention pass removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RetentionReport {
    /// Sessions deleted (their messages and tool calls cascade).
    pub sessions: usize,
    /// Messages deleted directly (not via a session cascade).
    pub messages: usize,
    /// Tool calls deleted directly (not via a session cascade).
    pub tool_calls: usize,
    /// Run-ledger events deleted.
    pub run_events: usize,
    /// Run-telemetry rows deleted.
    pub run_telemetry: usize,
}

impl RetentionReport {
    /// Total rows removed across every table.
    pub fn total(&self) -> usize {
        self.sessions + self.messages + self.tool_calls + self.run_events + self.run_telemetry
    }
}

/// Deletes every **finished** session that ended before `older_than`, together
/// with its messages, tool calls and search-index entries.
///
/// Running sessions are never touched, whatever their start time: a long-lived
/// session is not stale data. `session_messages` and `session_tool_calls`
/// cascade through their foreign keys (the connection sets
/// `PRAGMA foreign_keys = ON`), but the FTS entries do not — the virtual table
/// has no foreign keys — so they are removed explicitly.
pub fn prune_sessions_before(workspace_dir: &Path, older_than: DateTime<Utc>) -> Result<usize> {
    let cutoff = older_than.to_rfc3339();
    tracing::debug!("{LOG_PREFIX} prune_sessions_before.entry cutoff={cutoff}");
    let removed = with_transaction(workspace_dir, |conn| {
        // Collect first so the FTS rows can be removed by session id.
        let ids: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT id FROM sessions
                 WHERE status != 'running' AND ended_at IS NOT NULL AND ended_at < ?1",
            )?;
            let rows = stmt.query_map(params![cutoff], |row| row.get::<_, String>(0))?;
            let mut ids = Vec::new();
            for row in rows {
                ids.push(row?);
            }
            ids
        };
        let mut removed = 0usize;
        for id in &ids {
            conn.execute(
                "DELETE FROM sessions_fts WHERE session_id = ?1",
                params![id],
            )
            .storage_context("delete session FTS rows")?;
            removed += conn
                .execute("DELETE FROM sessions WHERE id = ?1", params![id])
                .storage_context("delete session")?;
        }
        Ok(removed)
    })?;
    tracing::debug!("{LOG_PREFIX} prune_sessions_before.exit removed={removed}");
    Ok(removed)
}

/// Trims a session's transcript to its most recent `keep_last` messages,
/// returning how many were removed.
///
/// The oldest messages go first. Their FTS entries are content-addressed by
/// session rather than by message id, so trimming a session's messages
/// necessarily leaves its index entries stale; call [`reindex_fts`] afterwards
/// if search accuracy for trimmed sessions matters.
pub fn trim_session_messages(
    workspace_dir: &Path,
    session_id: &str,
    keep_last: usize,
) -> Result<usize> {
    tracing::debug!(
        "{LOG_PREFIX} trim_session_messages.entry session={session_id} keep_last={keep_last}"
    );
    let removed = with_transaction(workspace_dir, |conn| {
        let removed = conn
            .execute(
                "DELETE FROM session_messages
                 WHERE session_id = ?1 AND id NOT IN (
                    SELECT id FROM session_messages WHERE session_id = ?1
                    ORDER BY id DESC LIMIT ?2
                 )",
                params![session_id, keep_last as i64],
            )
            .storage_context("trim session messages")?;
        Ok(removed)
    })?;
    tracing::debug!("{LOG_PREFIX} trim_session_messages.exit removed={removed}");
    Ok(removed)
}

/// Deletes tool-call rows created before `older_than`, returning how many.
pub fn prune_tool_calls_before(workspace_dir: &Path, older_than: DateTime<Utc>) -> Result<usize> {
    let cutoff = older_than.to_rfc3339();
    tracing::debug!("{LOG_PREFIX} prune_tool_calls_before.entry cutoff={cutoff}");
    let removed = with_transaction(workspace_dir, |conn| {
        conn.execute(
            "DELETE FROM session_tool_calls WHERE created_at < ?1",
            params![cutoff],
        )
        .storage_context("prune tool calls")
    })?;
    tracing::debug!("{LOG_PREFIX} prune_tool_calls_before.exit removed={removed}");
    Ok(removed)
}

/// Deletes run-ledger events stamped before `older_than`, returning how many.
///
/// Event sequences are per run and allocated as `MAX(sequence) + 1`, so pruning
/// the head of a run's log does not renumber or collide with later appends.
pub fn prune_run_events_before(workspace_dir: &Path, older_than: DateTime<Utc>) -> Result<usize> {
    let cutoff = older_than.to_rfc3339();
    tracing::debug!("{LOG_PREFIX} prune_run_events_before.entry cutoff={cutoff}");
    let removed = with_transaction(workspace_dir, |conn| {
        conn.execute(
            "DELETE FROM run_events WHERE timestamp < ?1",
            params![cutoff],
        )
        .storage_context("prune run events")
    })?;
    tracing::debug!("{LOG_PREFIX} prune_run_events_before.exit removed={removed}");
    Ok(removed)
}

/// Deletes run-telemetry rows last updated before `older_than`, returning how
/// many.
pub fn prune_run_telemetry_before(
    workspace_dir: &Path,
    older_than: DateTime<Utc>,
) -> Result<usize> {
    let cutoff = older_than.to_rfc3339();
    tracing::debug!("{LOG_PREFIX} prune_run_telemetry_before.entry cutoff={cutoff}");
    let removed = with_transaction(workspace_dir, |conn| {
        conn.execute(
            "DELETE FROM run_telemetry WHERE updated_at < ?1",
            params![cutoff],
        )
        .storage_context("prune run telemetry")
    })?;
    tracing::debug!("{LOG_PREFIX} prune_run_telemetry_before.exit removed={removed}");
    Ok(removed)
}

/// Applies one age-based retention pass across every table, returning what it
/// removed.
///
/// Sessions are pruned first so their messages and tool calls cascade rather
/// than being deleted twice; the remaining passes then catch orphaned rows that
/// outlived their session or belong to the run ledger.
pub fn apply_retention(workspace_dir: &Path, older_than: DateTime<Utc>) -> Result<RetentionReport> {
    tracing::debug!(
        "{LOG_PREFIX} apply_retention.entry cutoff={}",
        older_than.to_rfc3339()
    );
    let report = RetentionReport {
        sessions: prune_sessions_before(workspace_dir, older_than)?,
        messages: 0,
        tool_calls: prune_tool_calls_before(workspace_dir, older_than)?,
        run_events: prune_run_events_before(workspace_dir, older_than)?,
        run_telemetry: prune_run_telemetry_before(workspace_dir, older_than)?,
    };
    tracing::info!(
        "{LOG_PREFIX} apply_retention.exit removed total={} sessions={} tool_calls={} \
         run_events={} run_telemetry={}",
        report.total(),
        report.sessions,
        report.tool_calls,
        report.run_events,
        report.run_telemetry
    );
    Ok(report)
}

/// Rebuilds `sessions_fts` from the authoritative tables, returning the number
/// of index rows written.
///
/// The repair path the database never had. Any row whose FTS entry was lost —
/// to a failure between the two old autocommit statements, to a retention pass
/// that trimmed messages, or to a change of the
/// [snippet cap](super::ops::set_fts_snippet_bytes) — becomes searchable again.
/// The whole rebuild runs in one transaction, so search is never left with a
/// half-built index.
pub fn reindex_fts(workspace_dir: &Path) -> Result<usize> {
    tracing::debug!("{LOG_PREFIX} reindex_fts.entry");
    let limit = fts_snippet_bytes();
    let written = with_transaction(workspace_dir, |conn| {
        conn.execute("DELETE FROM sessions_fts", [])
            .storage_context("clear sessions_fts")?;
        let mut written = 0usize;

        // One row per session, carrying the agent name.
        {
            let mut stmt =
                conn.prepare("SELECT id, agent_definition_name FROM sessions ORDER BY rowid ASC")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut insert = conn.prepare(
                "INSERT INTO sessions_fts (session_id, agent_definition_name, content, tool_name)
                 VALUES (?1, ?2, '', '')",
            )?;
            for row in rows {
                let (id, name) = row?;
                insert.execute(params![id, name])?;
                written += 1;
            }
        }

        // One row per message, carrying the (capped) content snippet.
        {
            let mut stmt =
                conn.prepare("SELECT session_id, content FROM session_messages ORDER BY id ASC")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut insert = conn.prepare(
                "INSERT INTO sessions_fts (session_id, agent_definition_name, content, tool_name)
                 VALUES (?1, '', ?2, '')",
            )?;
            for row in rows {
                let (session_id, content) = row?;
                insert.execute(params![session_id, snippet(&content, limit)])?;
                written += 1;
            }
        }

        // One row per tool call, carrying the tool name.
        {
            let mut stmt = conn
                .prepare("SELECT session_id, tool_name FROM session_tool_calls ORDER BY id ASC")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut insert = conn.prepare(
                "INSERT INTO sessions_fts (session_id, agent_definition_name, content, tool_name)
                 VALUES (?1, '', '', ?2)",
            )?;
            for row in rows {
                let (session_id, tool_name) = row?;
                insert.execute(params![session_id, tool_name])?;
                written += 1;
            }
        }
        Ok(written)
    })?;
    tracing::info!("{LOG_PREFIX} reindex_fts.exit rows={written}");
    Ok(written)
}

/// Truncates `content` to at most `limit` bytes on a character boundary.
fn snippet(content: &str, limit: usize) -> &str {
    if content.len() <= limit {
        return content;
    }
    let mut cutoff = limit;
    while cutoff > 0 && !content.is_char_boundary(cutoff) {
        cutoff -= 1;
    }
    &content[..cutoff]
}
