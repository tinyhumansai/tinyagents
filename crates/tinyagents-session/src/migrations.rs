//! Versioned, idempotent schema migrations for the session database.
//!
//! # Why a version marker at all
//!
//! The session database used to (re-)execute its full `CREATE TABLE IF NOT
//! EXISTS` DDL on *every* operation. That is idempotent, but it is also a dead
//! end: `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
//! exists, so no column could ever be added to a workspace database that had
//! already been created. There was no way to express "and now also do this".
//!
//! # Shape
//!
//! [`MIGRATIONS`] is an ordered list whose **index is the version**. A
//! `schema_version` table records the highest index applied. On connection open
//! every migration with an index greater than the recorded version runs, in
//! order, each inside its own transaction, and the version is bumped after each
//! one. The version is re-read after taking the write lock so two connections
//! opening the same newly-upgraded workspace cannot repeat non-idempotent DDL.
//!
//! Two rules keep this sound:
//!
//! 1. **Append only.** Never reorder, insert into the middle of, or delete from
//!    the list — the index *is* the version, so any of those silently re-number
//!    every later migration.
//! 2. **Retire in place.** A migration that must stop running is replaced by the
//!    deliberate no-op `"SELECT 1;"` rather than removed, preserving the
//!    numbering of everything after it. (This is the convention LangGraph's
//!    Postgres checkpointer uses for the same reason.)
//!
//! # Pre-existing databases
//!
//! A workspace database created before this module existed has the tables but no
//! `schema_version` row, so it reads as version `-1` and every migration runs.
//! Migration 0–2 are exactly the DDL those databases already had, and every
//! statement is `IF NOT EXISTS`, so replaying them is a no-op that ends with the
//! version marker correctly stamped. No special-case detection is needed.

use rusqlite::{Connection, params};

use super::context::StorageContext;
use tinyagents_harness::error::Result;

/// Grep prefix for migration logging.
const LOG_PREFIX: &str = "[session_db:migrations]";

/// The ordered migration list. **Index == schema version.**
///
/// See the module docs before touching this: append only, and retire a
/// migration by replacing its body with `"SELECT 1;"` rather than deleting it.
pub(super) const MIGRATIONS: &[&str] = &[
    // ---- 0: base session tables ------------------------------------------
    "CREATE TABLE IF NOT EXISTS sessions (
        id                    TEXT PRIMARY KEY,
        agent_definition_id   TEXT NOT NULL,
        agent_definition_name TEXT NOT NULL,
        session_key           TEXT NOT NULL,
        parent_session_id     TEXT,
        thread_id             TEXT,
        source_channel        TEXT,
        status                TEXT NOT NULL DEFAULT 'running',
        model                 TEXT,
        turn_count            INTEGER NOT NULL DEFAULT 0,
        input_tokens          INTEGER NOT NULL DEFAULT 0,
        output_tokens         INTEGER NOT NULL DEFAULT 0,
        cached_input_tokens   INTEGER NOT NULL DEFAULT 0,
        cost_usd              REAL NOT NULL DEFAULT 0.0,
        transcript_path       TEXT,
        started_at            TEXT NOT NULL,
        ended_at              TEXT,
        FOREIGN KEY (parent_session_id) REFERENCES sessions(id) ON DELETE SET NULL
     );
     CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_definition_id);
     CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
     CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
     CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
     CREATE INDEX IF NOT EXISTS idx_sessions_thread ON sessions(thread_id);
     CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(source_channel);
     CREATE INDEX IF NOT EXISTS idx_sessions_key ON sessions(session_key);

     CREATE TABLE IF NOT EXISTS session_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        model       TEXT,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        cost_usd    REAL,
        created_at  TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
     );
     CREATE INDEX IF NOT EXISTS idx_messages_session ON session_messages(session_id);

     CREATE TABLE IF NOT EXISTS session_tool_calls (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        message_id  INTEGER,
        tool_name   TEXT NOT NULL,
        tool_input  TEXT,
        tool_output TEXT,
        status      TEXT NOT NULL DEFAULT 'pending',
        duration_ms INTEGER,
        created_at  TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (message_id) REFERENCES session_messages(id) ON DELETE SET NULL
     );
     CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON session_tool_calls(session_id);
     CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON session_tool_calls(tool_name);",
    // ---- 1: full-text search index ---------------------------------------
    "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
        session_id,
        agent_definition_name,
        content,
        tool_name
     );",
    // ---- 2: run ledger ---------------------------------------------------
    "CREATE TABLE IF NOT EXISTS agent_runs (
        id                 TEXT PRIMARY KEY,
        kind               TEXT NOT NULL,
        parent_run_id      TEXT,
        parent_thread_id   TEXT,
        agent_id           TEXT,
        status             TEXT NOT NULL,
        prompt_ref         TEXT,
        worker_thread_id   TEXT,
        task_board_id      TEXT,
        task_card_id       TEXT,
        checkpoint_path    TEXT,
        checkpoint_json    TEXT,
        summary            TEXT,
        error              TEXT,
        metadata_json      TEXT NOT NULL DEFAULT '{}',
        started_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        completed_at       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_kind ON agent_runs(kind);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_parent ON agent_runs(parent_run_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_thread ON agent_runs(parent_thread_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_updated ON agent_runs(updated_at);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_worker_thread ON agent_runs(worker_thread_id);

    CREATE TABLE IF NOT EXISTS workflow_runs (
        id                 TEXT PRIMARY KEY,
        definition_id      TEXT NOT NULL,
        parent_thread_id   TEXT,
        input_json         TEXT NOT NULL DEFAULT '{}',
        phase_states_json  TEXT NOT NULL DEFAULT '{}',
        child_run_ids_json TEXT NOT NULL DEFAULT '[]',
        status             TEXT NOT NULL,
        summary            TEXT,
        started_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        completed_at       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_workflow_runs_definition ON workflow_runs(definition_id);
    CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status);
    CREATE INDEX IF NOT EXISTS idx_workflow_runs_thread ON workflow_runs(parent_thread_id);

    CREATE TABLE IF NOT EXISTS run_events (
        run_id      TEXT NOT NULL,
        sequence    INTEGER NOT NULL,
        event_type  TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        timestamp   TEXT NOT NULL,
        PRIMARY KEY (run_id, sequence)
    );
    CREATE INDEX IF NOT EXISTS idx_run_events_timestamp ON run_events(timestamp);

    CREATE TABLE IF NOT EXISTS run_telemetry (
        run_id              TEXT PRIMARY KEY,
        input_tokens        INTEGER NOT NULL DEFAULT 0,
        output_tokens       INTEGER NOT NULL DEFAULT 0,
        cached_input_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd            REAL NOT NULL DEFAULT 0.0,
        elapsed_ms          INTEGER,
        tool_count          INTEGER NOT NULL DEFAULT 0,
        model               TEXT,
        provider            TEXT,
        error               TEXT,
        updated_at          TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS agent_teams (
        id                 TEXT PRIMARY KEY,
        parent_thread_id   TEXT,
        lead_agent_id      TEXT NOT NULL,
        status             TEXT NOT NULL,
        summary            TEXT,
        created_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        closed_at          TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_agent_teams_thread ON agent_teams(parent_thread_id);
    CREATE INDEX IF NOT EXISTS idx_agent_teams_status ON agent_teams(status);

    CREATE TABLE IF NOT EXISTS agent_team_members (
        id                 TEXT PRIMARY KEY,
        team_id            TEXT NOT NULL,
        name               TEXT NOT NULL,
        agent_id           TEXT,
        member_status      TEXT NOT NULL,
        current_task_id    TEXT,
        worker_thread_id   TEXT,
        run_id             TEXT,
        created_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        UNIQUE(team_id, name)
    );
    CREATE INDEX IF NOT EXISTS idx_agent_team_members_team ON agent_team_members(team_id);

    CREATE TABLE IF NOT EXISTS agent_team_tasks (
        id                  TEXT PRIMARY KEY,
        team_id             TEXT NOT NULL,
        title               TEXT NOT NULL,
        objective           TEXT,
        status              TEXT NOT NULL,
        owner_member_id     TEXT,
        claimed_by_member_id TEXT,
        claim_token         TEXT,
        depends_on_json     TEXT NOT NULL DEFAULT '[]',
        gate_status         TEXT NOT NULL DEFAULT 'pending',
        gate_reason         TEXT,
        evidence_json       TEXT NOT NULL DEFAULT '[]',
        source_run_id       TEXT,
        order_index         INTEGER NOT NULL DEFAULT 0,
        created_at          TEXT NOT NULL,
        updated_at          TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_team ON agent_team_tasks(team_id);
    CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_status ON agent_team_tasks(status);
    CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_claimed ON agent_team_tasks(claimed_by_member_id);",
    // ---- 3: index coverage for the ORDER BY / sort columns ---------------
    //
    // `list_workflow_runs` and `list_agent_teams` both order by `updated_at
    // DESC` and `list_agent_team_tasks` by `(order_index, created_at)`, none of
    // which had an index — every listing was a full scan plus a sort. The
    // `agent_runs` table already had its `updated_at` index; these bring the
    // rest up to parity.
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_updated ON workflow_runs(updated_at);
     CREATE INDEX IF NOT EXISTS idx_agent_teams_updated ON agent_teams(updated_at);
     CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_order
        ON agent_team_tasks(team_id, order_index, created_at);",
    // ---- 4: retain hidden assistant reasoning ---------------------------
    "ALTER TABLE session_messages ADD COLUMN reasoning_content TEXT;",
];

/// Applies every migration newer than the database's recorded schema version.
///
/// Idempotent: running it against an up-to-date database reads one row and
/// returns. Each migration runs inside its own transaction together with the
/// version bump, so a crash mid-migration leaves the version pointing at the
/// last fully applied step rather than at a half-applied one.
pub(super) fn apply(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            id      INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL
         );",
    )
    .storage_context("failed to create schema_version table")?;

    // `-1` means "nothing applied yet", which is also what a pre-migration
    // workspace database reads as. Every migration is `IF NOT EXISTS`, so
    // replaying 0..=2 over such a database is a no-op that stamps the marker.
    let current: i64 = conn
        .query_row(
            "SELECT COALESCE((SELECT version FROM schema_version WHERE id = 1), -1)",
            [],
            |row| row.get(0),
        )
        .storage_context("failed to read schema_version")?;

    let latest = MIGRATIONS.len() as i64 - 1;
    if current >= latest {
        return Ok(());
    }
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} applying migrations from version {current} to {latest}"
    );

    for (version, sql) in MIGRATIONS.iter().enumerate() {
        let version = version as i64;
        if version <= current {
            continue;
        }
        apply_one(conn, version, sql)?;
    }
    Ok(())
}

/// Apply one migration after acquiring the database write lock.
///
/// The schema version is deliberately re-read *after* `BEGIN IMMEDIATE`:
/// another connection may have completed this migration after our optimistic
/// read in [`apply`] but before this connection acquired the lock.
pub(super) fn apply_one(conn: &Connection, version: i64, sql: &str) -> Result<bool> {
    conn.execute_batch("BEGIN IMMEDIATE")
        .storage_context("begin migration transaction")?;
    let applied = (|| -> Result<bool> {
        let locked_current: i64 = conn
            .query_row(
                "SELECT COALESCE((SELECT version FROM schema_version WHERE id = 1), -1)",
                [],
                |row| row.get(0),
            )
            .storage_context("re-read schema_version under migration lock")?;
        if version <= locked_current {
            return Ok(false);
        }
        conn.execute_batch(sql)
            .storage_context(&format!("failed to apply session DB migration {version}"))?;
        conn.execute(
            "INSERT INTO schema_version (id, version) VALUES (1, ?1)
             ON CONFLICT(id) DO UPDATE SET version = excluded.version",
            params![version],
        )
        .storage_context("failed to record schema version")?;
        Ok(true)
    })();
    match applied {
        Ok(did_apply) => {
            conn.execute_batch("COMMIT")
                .storage_context("commit migration transaction")?;
            if did_apply {
                tinyagents_tracing::debug!("{LOG_PREFIX} applied migration {version}");
            }
            Ok(did_apply)
        }
        Err(err) => {
            if let Err(rollback) = conn.execute_batch("ROLLBACK") {
                tinyagents_tracing::warn!(
                    "{LOG_PREFIX} rollback of migration {version} failed: {rollback} (original: {err})"
                );
            }
            Err(err)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// The index of a migration **is** its version, so the list may only ever
    /// be appended to. This pins the current length: bumping it is the moment
    /// to re-read the module docs and confirm nothing was reordered.
    #[test]
    fn migration_list_is_append_only() {
        assert_eq!(
            MIGRATIONS.len(),
            5,
            "MIGRATIONS is append-only — adding one is fine, reordering or \
             deleting one silently re-numbers every later migration"
        );
    }

    #[test]
    fn apply_is_idempotent_and_records_the_version() {
        let conn = Connection::open_in_memory().expect("open");
        apply(&conn).expect("first apply");
        let version: i64 = conn
            .query_row("SELECT version FROM schema_version WHERE id = 1", [], |r| {
                r.get(0)
            })
            .expect("read version");
        assert_eq!(version, MIGRATIONS.len() as i64 - 1);
        // Second run is a no-op and must not error.
        apply(&conn).expect("second apply");
    }

    /// A database created by the pre-migration DDL has the tables but no
    /// version marker. Applying migrations must bring it forward rather than
    /// failing on already-existing objects.
    #[test]
    fn apply_upgrades_a_pre_migration_database() {
        let conn = Connection::open_in_memory().expect("open");
        // Simulate the old world: migrations 0..=2 executed with no marker.
        for sql in &MIGRATIONS[..3] {
            conn.execute_batch(sql).expect("legacy ddl");
        }
        apply(&conn).expect("upgrade");
        // The version-3 index only exists because the migration ran.
        let exists: bool = conn
            .prepare(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_agent_teams_updated'",
            )
            .expect("prepare")
            .exists([])
            .expect("exists");
        assert!(
            exists,
            "migration 3 added the agent_teams(updated_at) index"
        );
    }
}
