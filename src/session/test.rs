//! Module-local unit tests for [`crate::session`].
//!
//! Consolidated here per AGENTS.md: one `test.rs` per module directory rather
//! than per-file inline `mod tests` blocks. Sections mirror the source files.

use super::context::StorageContext;
use super::migrations::apply as init_schema;
use super::ops::*;
use super::store::with_memory_connection;
use super::types::*;
use crate::error::TinyAgentsError;
use chrono::Utc;
use rusqlite::{Connection, params};

// ── ops.rs ──────────────────────────────────────────────────────────────
fn insert_test_session(conn: &Connection, id: &str, agent_id: &str, key: &str) {
    let now = Utc::now();
    conn.execute(
        "INSERT INTO sessions (
            id, agent_definition_id, agent_definition_name, session_key,
            status, started_at
         ) VALUES (?1, ?2, ?3, ?4, 'running', ?5)",
        params![id, agent_id, agent_id, key, now.to_rfc3339()],
    )
    .unwrap();
    index_fts_session(conn, id, agent_id).unwrap();
}

fn insert_test_session_with_parent(
    conn: &Connection,
    id: &str,
    agent_id: &str,
    key: &str,
    parent_id: &str,
) {
    let now = Utc::now();
    conn.execute(
        "INSERT INTO sessions (
            id, agent_definition_id, agent_definition_name, session_key,
            parent_session_id, status, started_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, 'running', ?6)",
        params![id, agent_id, agent_id, key, parent_id, now.to_rfc3339()],
    )
    .unwrap();
    index_fts_session(conn, id, agent_id).unwrap();
}

#[test]
fn map_session_row_roundtrip() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "sess-1", "orchestrator", "1700000000_orchestrator");

        let mut stmt = conn.prepare(
            "SELECT id, agent_definition_id, agent_definition_name, session_key,
                    parent_session_id, thread_id, source_channel, status, model,
                    turn_count, input_tokens, output_tokens, cached_input_tokens,
                    cost_usd, transcript_path, started_at, ended_at
             FROM sessions WHERE id = 'sess-1'",
        )?;
        let session = stmt.query_row([], map_session_row)?;

        assert_eq!(session.id, "sess-1");
        assert_eq!(session.agent_definition_id, "orchestrator");
        assert_eq!(session.session_key, "1700000000_orchestrator");
        assert_eq!(session.status, SessionStatus::Running);
        assert!(session.parent_session_id.is_none());
        assert!(session.ended_at.is_none());
        Ok(())
    })
    .unwrap();
}

#[test]
fn search_by_agent_id() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "a1", "orchestrator", "key1");
        insert_test_session(conn, "a2", "researcher", "key2");
        insert_test_session(conn, "a3", "orchestrator", "key3");

        let params = SessionSearchParams {
            agent_id: Some("orchestrator".to_string()),
            ..Default::default()
        };

        let result = search_sessions_inner(conn, &params)?;
        assert_eq!(result.total, 2);
        assert_eq!(result.sessions.len(), 2);
        assert!(
            result
                .sessions
                .iter()
                .all(|s| s.agent_definition_id == "orchestrator")
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn search_by_fts_query() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "b1", "orchestrator", "key1");
        insert_test_session(conn, "b2", "researcher", "key2");

        conn.execute(
            "INSERT INTO session_messages (session_id, role, content, created_at)
             VALUES ('b1', 'user', 'Fix the login bug in authentication', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;
        index_fts_content(conn, "b1", "Fix the login bug in authentication")?;

        conn.execute(
            "INSERT INTO session_messages (session_id, role, content, created_at)
             VALUES ('b2', 'user', 'Deploy the new feature to production', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;
        index_fts_content(conn, "b2", "Deploy the new feature to production")?;

        let params = SessionSearchParams {
            query: Some("login".to_string()),
            ..Default::default()
        };

        let result = search_sessions_inner(conn, &params)?;
        assert_eq!(result.total, 1);
        assert_eq!(result.sessions[0].id, "b1");
        Ok(())
    })
    .unwrap();
}

#[test]
fn search_by_tool_name() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "c1", "orchestrator", "key1");
        insert_test_session(conn, "c2", "researcher", "key2");

        conn.execute(
            "INSERT INTO session_tool_calls (session_id, tool_name, status, created_at)
             VALUES ('c1', 'shell', 'ok', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;
        conn.execute(
            "INSERT INTO session_tool_calls (session_id, tool_name, status, created_at)
             VALUES ('c2', 'file_read', 'ok', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;

        let params = SessionSearchParams {
            tool_name: Some("shell".to_string()),
            ..Default::default()
        };

        let result = search_sessions_inner(conn, &params)?;
        assert_eq!(result.total, 1);
        assert_eq!(result.sessions[0].id, "c1");
        Ok(())
    })
    .unwrap();
}

#[test]
fn search_by_parent_session() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "parent-1", "orchestrator", "key1");
        insert_test_session_with_parent(conn, "child-1", "researcher", "key2", "parent-1");
        insert_test_session_with_parent(conn, "child-2", "coder", "key3", "parent-1");
        insert_test_session(conn, "unrelated", "other", "key4");

        let params = SessionSearchParams {
            parent_session_id: Some("parent-1".to_string()),
            ..Default::default()
        };

        let result = search_sessions_inner(conn, &params)?;
        assert_eq!(result.total, 2);
        Ok(())
    })
    .unwrap();
}

#[test]
fn search_pagination() {
    with_memory_connection(|conn| {
        for i in 0..10 {
            insert_test_session(conn, &format!("p{i}"), "agent", &format!("key{i}"));
        }

        let params = SessionSearchParams {
            limit: Some(3),
            offset: Some(0),
            ..Default::default()
        };
        let result = search_sessions_inner(conn, &params)?;
        assert_eq!(result.total, 10);
        assert_eq!(result.sessions.len(), 3);

        let params2 = SessionSearchParams {
            limit: Some(3),
            offset: Some(3),
            ..Default::default()
        };
        let result2 = search_sessions_inner(conn, &params2)?;
        assert_eq!(result2.total, 10);
        assert_eq!(result2.sessions.len(), 3);
        assert_ne!(result.sessions[0].id, result2.sessions[0].id);

        Ok(())
    })
    .unwrap();
}

#[test]
fn search_empty_results() {
    with_memory_connection(|conn| {
        let params = SessionSearchParams {
            agent_id: Some("nonexistent".to_string()),
            ..Default::default()
        };
        let result = search_sessions_inner(conn, &params)?;
        assert_eq!(result.total, 0);
        assert!(result.sessions.is_empty());
        Ok(())
    })
    .unwrap();
}

#[test]
fn tool_output_truncation() {
    with_memory_connection(|conn| {
        let session_id = "trunc-sess";
        insert_test_session(conn, session_id, "agent", "key");

        let large_output = "x".repeat(MAX_TOOL_OUTPUT_BYTES + 1000);
        let bounded = if large_output.len() <= MAX_TOOL_OUTPUT_BYTES {
            large_output.clone()
        } else {
            let mut cutoff = MAX_TOOL_OUTPUT_BYTES;
            while cutoff > 0 && !large_output.is_char_boundary(cutoff) {
                cutoff -= 1;
            }
            let mut truncated = large_output[..cutoff].to_string();
            truncated.push_str("\n...[truncated]");
            truncated
        };

        conn.execute(
            "INSERT INTO session_tool_calls (session_id, tool_name, tool_output, status, created_at)
             VALUES (?1, 'test', ?2, 'ok', ?3)",
            params![session_id, bounded, Utc::now().to_rfc3339()],
        )?;

        let stored: String = conn.query_row(
            "SELECT tool_output FROM session_tool_calls WHERE session_id = ?1",
            params![session_id],
            |r| r.get(0),
        )?;
        assert!(stored.len() <= MAX_TOOL_OUTPUT_BYTES + 20);
        assert!(stored.ends_with("[truncated]"));

        Ok(())
    })
    .unwrap();
}

#[test]
fn mark_interrupted_updates_running() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "run1", "agent", "key1");
        insert_test_session(conn, "run2", "agent", "key2");
        conn.execute(
            "UPDATE sessions SET status = 'completed' WHERE id = 'run2'",
            [],
        )?;

        let now = Utc::now();
        let changed = conn.execute(
            "UPDATE sessions SET status = 'interrupted', ended_at = ?1
             WHERE status = 'running'",
            params![now.to_rfc3339()],
        )?;
        assert_eq!(changed, 1);

        let status: String =
            conn.query_row("SELECT status FROM sessions WHERE id = 'run1'", [], |r| {
                r.get(0)
            })?;
        assert_eq!(status, "interrupted");

        let status2: String =
            conn.query_row("SELECT status FROM sessions WHERE id = 'run2'", [], |r| {
                r.get(0)
            })?;
        assert_eq!(status2, "completed");

        Ok(())
    })
    .unwrap();
}

#[test]
fn session_end_updates_cost_fields() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "cost-sess", "agent", "key");

        let now = Utc::now();
        conn.execute(
            "UPDATE sessions SET
                status = 'completed', turn_count = 5, input_tokens = 10000,
                output_tokens = 2000, cached_input_tokens = 8000,
                cost_usd = 0.0345, ended_at = ?1
             WHERE id = 'cost-sess'",
            params![now.to_rfc3339()],
        )?;

        let mut stmt = conn.prepare(
            "SELECT id, agent_definition_id, agent_definition_name, session_key,
                    parent_session_id, thread_id, source_channel, status, model,
                    turn_count, input_tokens, output_tokens, cached_input_tokens,
                    cost_usd, transcript_path, started_at, ended_at
             FROM sessions WHERE id = 'cost-sess'",
        )?;
        let session = stmt.query_row([], map_session_row)?;

        assert_eq!(session.status, SessionStatus::Completed);
        assert_eq!(session.turn_count, 5);
        assert_eq!(session.input_tokens, 10000);
        assert_eq!(session.output_tokens, 2000);
        assert_eq!(session.cached_input_tokens, 8000);
        assert!((session.cost_usd - 0.0345).abs() < f64::EPSILON);
        assert!(session.ended_at.is_some());

        Ok(())
    })
    .unwrap();
}

#[test]
fn combined_filters() {
    with_memory_connection(|conn| {
        insert_test_session(conn, "cf1", "orchestrator", "key1");
        insert_test_session(conn, "cf2", "orchestrator", "key2");
        insert_test_session(conn, "cf3", "researcher", "key3");

        conn.execute(
            "UPDATE sessions SET status = 'completed' WHERE id = 'cf1'",
            [],
        )?;

        let params = SessionSearchParams {
            agent_id: Some("orchestrator".to_string()),
            status: Some("completed".to_string()),
            ..Default::default()
        };

        let result = search_sessions_inner(conn, &params)?;
        assert_eq!(result.total, 1);
        assert_eq!(result.sessions[0].id, "cf1");
        Ok(())
    })
    .unwrap();
}

// ── Regressions for the review findings on PR #90 ─────────────────────────

/// A long non-ASCII message must not panic while being indexed.
///
/// `&content[..2000]` panicked whenever byte 2000 landed inside a multi-byte
/// character. The message INSERT autocommits before the FTS write, so the panic
/// left a stored message with no FTS row — permanently unsearchable.
#[test]
fn long_multibyte_message_is_indexed_without_panicking() {
    with_memory_connection(|conn| {
        // '€' is 3 bytes and 2000 is not a multiple of 3, so byte 2000 lands
        // strictly inside a character — the case that panicked. A 2-byte char
        // would leave 2000 on a boundary and pass vacuously.
        let content = "€".repeat(1500);
        assert!(content.len() > 2000);
        assert!(!content.is_char_boundary(2000));

        index_fts_content(conn, "sess-utf8", &content)?;

        let indexed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sessions_fts WHERE session_id = 'sess-utf8'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(indexed, 1, "the message must still get an FTS row");
        Ok(())
    })
    .unwrap();
}

/// An ASCII message longer than the cap still truncates to exactly the cap.
#[test]
fn long_ascii_message_truncates_at_the_byte_cap() {
    with_memory_connection(|conn| {
        let content = "a".repeat(5000);
        index_fts_content(conn, "sess-ascii", &content)?;
        let stored: String = conn.query_row(
            "SELECT content FROM sessions_fts WHERE session_id = 'sess-ascii'",
            [],
            |r| r.get(0),
        )?;
        assert_eq!(stored.len(), 2000);
        Ok(())
    })
    .unwrap();
}

/// Punctuation that is meaningful to FTS5 must be searched as literal text.
///
/// `SessionSearchParams::query` is plain text, not an FTS5 expression. Binding
/// it raw made ordinary input (`C++`, `foo-bar`, `file.rs`, a stray quote)
/// return a syntax or `no such column` error instead of results.
#[test]
fn fts_query_treats_punctuation_as_literal_text() {
    for raw in ["C++", "foo-bar", "file.rs", "a\"b", "NOT", "*"] {
        with_memory_connection(|conn| {
            conn.execute(
                "INSERT INTO sessions_fts (session_id, agent_definition_name, content, tool_name)
                 VALUES ('s1', '', ?1, '')",
                params![raw],
            )?;
            let params = SessionSearchParams {
                query: Some(raw.to_string()),
                ..Default::default()
            };
            // The assertion is that this does not error; FTS tokenization
            // decides whether a given punctuation string is recallable.
            search_sessions_inner(conn, &params)
                .unwrap_or_else(|e| panic!("query {raw:?} must not error: {e}"));
            Ok(())
        })
        .unwrap();
    }
}

/// Multi-term queries stay conjunctive, as a search box implies.
#[test]
fn fts_query_joins_terms_with_and() {
    assert_eq!(fts_match_query("alpha beta"), "\"alpha\" AND \"beta\"");
    assert_eq!(fts_match_query("C++"), "\"C++\"");
    // Embedded quotes are escaped by doubling, per the FTS5 string grammar.
    assert_eq!(fts_match_query("a\"b"), "\"a\"\"b\"");
}

// ── types.rs ───────────────────────────────────────────────────────────

#[test]
fn session_status_roundtrip() {
    for status in [
        SessionStatus::Running,
        SessionStatus::Completed,
        SessionStatus::Failed,
        SessionStatus::Interrupted,
    ] {
        assert_eq!(SessionStatus::parse(status.as_str()), status);
    }
}

#[test]
fn session_status_parse_unknown_defaults_to_running() {
    assert_eq!(SessionStatus::parse("bogus"), SessionStatus::Running);
    assert_eq!(SessionStatus::parse(""), SessionStatus::Running);
}

#[test]
fn session_status_serde_roundtrip() {
    let status = SessionStatus::Completed;
    let json = serde_json::to_string(&status).unwrap();
    assert_eq!(json, "\"completed\"");
    let parsed: SessionStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, status);
}

#[test]
fn session_search_params_defaults() {
    let params = SessionSearchParams::default();
    assert!(params.query.is_none());
    assert!(params.agent_id.is_none());
    assert!(params.limit.is_none());
    assert!(params.offset.is_none());
}

// ── store.rs ───────────────────────────────────────────────────────────

#[test]
fn schema_initializes_without_error() {
    with_memory_connection(|conn| {
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        assert_eq!(count, 0);
        Ok(())
    })
    .unwrap();
}

#[test]
fn schema_is_idempotent() {
    let conn = Connection::open_in_memory().unwrap();
    init_schema(&conn).unwrap();
    init_schema(&conn).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn wal_mode_is_set() {
    with_memory_connection(|conn| {
        let mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0))?;
        // In-memory DBs may report "memory" instead of "wal"
        assert!(mode == "wal" || mode == "memory");
        Ok(())
    })
    .unwrap();
}

#[test]
fn fts_table_exists_after_init() {
    with_memory_connection(|conn| {
        let exists: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions_fts'")?
            .exists([])?;
        assert!(exists);
        Ok(())
    })
    .unwrap();
}

#[test]
fn foreign_keys_are_enabled() {
    with_memory_connection(|conn| {
        let fk: i64 = conn.query_row("PRAGMA foreign_keys", [], |r| r.get(0))?;
        assert_eq!(fk, 1);
        Ok(())
    })
    .unwrap();
}

// ── context.rs ───────────────────────────────────────────────────────────

#[test]
fn result_error_is_prefixed_with_context() {
    let failed: std::result::Result<(), _> = Err("disk full");
    let err = failed.storage_context("write session").unwrap_err();
    assert!(matches!(err, TinyAgentsError::Storage(_)));
    assert_eq!(err.to_string(), "storage error: write session: disk full");
}

#[test]
fn result_ok_passes_through() {
    let ok: std::result::Result<u8, &str> = Ok(7);
    assert_eq!(ok.storage_context("read").unwrap(), 7);
}

#[test]
fn none_becomes_storage_error_without_a_source_suffix() {
    let absent: Option<u8> = None;
    let err = absent
        .storage_context("run missing after upsert")
        .unwrap_err();
    assert_eq!(err.to_string(), "storage error: run missing after upsert");
}

#[test]
fn some_passes_through() {
    assert_eq!(Some(3).storage_context("read").unwrap(), 3);
}

/// `record_tool_call` must return the `session_tool_calls` row id, not the
/// rowid of the FTS row written immediately afterwards.
///
/// The FTS insert moves `last_insert_rowid()`, so reading it after indexing
/// handed callers an id for a tool call that does not exist. The session row is
/// already in `sessions_fts` in any real session, which is what makes the two
/// counters diverge.
#[test]
fn record_tool_call_returns_the_tool_call_row_id() {
    with_memory_connection(|conn| {
        // Seed an FTS row first, as a real session always would, so the FTS
        // rowid counter is ahead of the tool-call one.
        index_fts_session(conn, "s1", "agent")?;

        conn.execute(
            "INSERT INTO sessions (
                id, agent_definition_id, agent_definition_name, session_key, started_at
             ) VALUES ('s1', 'a', 'agent', 's1', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;
        conn.execute(
            "INSERT INTO session_tool_calls (session_id, tool_name, status, created_at)
             VALUES ('s1', 'echo', 'ok', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;
        let expected = conn.last_insert_rowid();
        index_fts_tool(conn, "s1", "echo")?;

        // The FTS insert must have moved the connection's rowid...
        assert_ne!(
            conn.last_insert_rowid(),
            expected,
            "test is vacuous unless the FTS insert moves last_insert_rowid()"
        );
        // ...and the id we hand back must still address a real tool call.
        let exists: i64 = conn.query_row(
            "SELECT COUNT(*) FROM session_tool_calls WHERE id = ?1",
            params![expected],
            |r| r.get(0),
        )?;
        assert_eq!(
            exists, 1,
            "returned id must address a session_tool_calls row"
        );
        Ok(())
    })
    .unwrap();
}

/// Every session-DB connection waits out a competing writer.
///
/// SQLite's own default `busy_timeout` is 0: with no handler installed a
/// `BEGIN IMMEDIATE` that meets another writer fails instantly with
/// `SQLITE_BUSY` rather than waiting, which would contradict the
/// serialize-at-BEGIN rationale that `with_transaction`, the task claim CAS and
/// the run-event sequence allocation are all written against.
///
/// This pins the property, not a fix. It passes against the pre-change code as
/// well, because `rusqlite::Connection::open` installs a 5s busy timeout of its
/// own accord — a fact nothing in this crate stated, and nothing enforced.
/// `store::BUSY_TIMEOUT` now sets it explicitly and this test is what catches
/// it disappearing.
///
/// The test holds a real write lock from a second connection for a beat, then
/// asserts a `with_transaction` issued concurrently waits and succeeds.
#[test]
fn with_transaction_waits_out_a_competing_writer() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().to_path_buf();

    // Create the DB (and run migrations) before contending for it.
    super::store::with_connection(&workspace, |_| Ok(())).unwrap();

    let blocker = Connection::open(super::store::db_path(&workspace)).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();

    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        blocker.execute_batch("ROLLBACK").unwrap();
    });

    let result = super::store::with_transaction(&workspace, |conn| {
        conn.execute(
            "INSERT INTO sessions (
                id, agent_definition_id, agent_definition_name, session_key,
                status, started_at
             ) VALUES ('waited', 'a', 'a', 'k', 'running', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;
        Ok(())
    });

    releaser.join().unwrap();
    assert!(
        result.is_ok(),
        "BEGIN IMMEDIATE must wait for the competing writer, not fail instantly: {result:?}"
    );
}
