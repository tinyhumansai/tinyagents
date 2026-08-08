//! Regression tests for the session database's durability and correctness
//! fixes: schema migrations, the busy handler, the task-claim guard, atomic
//! FTS indexing, and the retention/reindex entry points.
//!
//! These live as integration tests rather than module-local ones so they
//! exercise the same public surface a host uses.
#![cfg(feature = "sqlite")]

use chrono::{Duration, Utc};
use tinyagents::session;
use tinyagents::session::run_ledger::{
    self,
    types::{
        AgentTeamStatus, AgentTeamTaskStatus, AgentTeamTaskUpsert, AgentTeamUpsert, ClaimOutcome,
    },
};

fn team(id: &str) -> AgentTeamUpsert {
    AgentTeamUpsert {
        id: id.into(),
        parent_thread_id: None,
        lead_agent_id: "lead".into(),
        status: AgentTeamStatus::Active,
        summary: None,
        created_at: None,
        closed_at: None,
    }
}

fn task(id: &str, status: AgentTeamTaskStatus) -> AgentTeamTaskUpsert {
    AgentTeamTaskUpsert {
        id: id.into(),
        team_id: "team".into(),
        title: format!("work {id}"),
        objective: None,
        status,
        owner_member_id: None,
        depends_on: Vec::new(),
        gate_status: None,
        gate_reason: None,
        evidence: Vec::new(),
        source_run_id: None,
        order_index: 0,
        created_at: None,
    }
}

fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// A competing writer must be waited out, not failed on.
///
/// This pins a property rather than a fix: it passes against the pre-change
/// code too, because `rusqlite` installs a 5s busy timeout on every
/// `Connection::open` of its own accord. The explicit `store::BUSY_TIMEOUT` is
/// about owning that guarantee instead of inheriting it from an undocumented
/// dependency default — and this test is what would catch it going away.
#[test]
fn a_competing_writer_is_waited_out_rather_than_failed_on() {
    let dir = workspace();
    session::with_connection(dir.path(), |_| Ok(())).unwrap();

    let blocker = rusqlite::Connection::open(session::db_path(dir.path())).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        blocker.execute_batch("ROLLBACK").unwrap();
    });

    let result = session::record_session_start(
        dir.path(),
        "s-waited",
        "agent",
        "Agent",
        "key",
        None,
        None,
        None,
        None,
        None,
    );
    releaser.join().unwrap();
    assert!(
        result.is_ok(),
        "the write must wait for the lock, not fail instantly: {result:?}"
    );
}

/// C1: the schema carries a version marker, so a column could actually be
/// added to an existing workspace database.
#[test]
fn the_schema_records_a_version() {
    let dir = workspace();
    let version: i64 = session::with_connection(dir.path(), |conn| {
        Ok(conn
            .query_row("SELECT version FROM schema_version WHERE id = 1", [], |r| {
                r.get(0)
            })
            .unwrap())
    })
    .unwrap();
    assert!(
        version >= 0,
        "a freshly created database records the schema version it was built at"
    );
}

/// SESS-11: the listing sort columns are indexed.
#[test]
fn listing_sort_columns_are_indexed() {
    let dir = workspace();
    session::with_connection(dir.path(), |conn| {
        for name in [
            "idx_workflow_runs_updated",
            "idx_agent_teams_updated",
            "idx_agent_team_tasks_order",
        ] {
            let exists: bool = conn
                .prepare("SELECT 1 FROM sqlite_master WHERE type='index' AND name=?1")
                .unwrap()
                .exists([name])
                .unwrap();
            assert!(exists, "missing index {name}");
        }
        Ok(())
    })
    .unwrap();
}

/// SESS-2: a finished task must not be re-claimable.
///
/// `upsert_agent_team_task` NULLs `claimed_by_member_id` whenever the status is
/// not `in_progress`, so the old `claimed_by_member_id IS NULL` guard was
/// satisfied by every `done` task. A stale worker could flip a completed task
/// back to `in_progress` and strand everything that depended on it.
#[test]
fn a_done_task_cannot_be_reclaimed() {
    let dir = workspace();
    run_ledger::upsert_agent_team(dir.path(), team("team")).unwrap();
    run_ledger::upsert_agent_team_task(dir.path(), task("task", AgentTeamTaskStatus::Done))
        .unwrap();

    let outcome =
        run_ledger::claim_agent_team_task(dir.path(), "team", "task", "stale-worker", "tok")
            .unwrap();
    assert!(
        matches!(outcome, ClaimOutcome::AlreadyClaimed),
        "a terminal task is not claimable, whatever its claim column says: {outcome:?}"
    );

    let task = run_ledger::get_agent_team_task(dir.path(), "task")
        .unwrap()
        .unwrap();
    assert_eq!(
        task.status,
        AgentTeamTaskStatus::Done,
        "the rejected claim must not have flipped the task back to in_progress"
    );
}

/// A claimable task still claims — the guard must not be so tight that it
/// breaks the happy path.
#[test]
fn a_todo_task_is_still_claimable() {
    let dir = workspace();
    run_ledger::upsert_agent_team(dir.path(), team("team")).unwrap();
    run_ledger::upsert_agent_team_task(dir.path(), task("task", AgentTeamTaskStatus::Todo))
        .unwrap();
    let outcome =
        run_ledger::claim_agent_team_task(dir.path(), "team", "task", "worker", "tok").unwrap();
    assert!(matches!(outcome, ClaimOutcome::Claimed(_)), "{outcome:?}");
}

/// SESS-6: retention actually deletes, and reindexing restores searchability.
#[test]
fn retention_prunes_finished_sessions_and_reindex_restores_search() {
    let dir = workspace();
    session::record_session_start(
        dir.path(),
        "old",
        "agent",
        "Agent",
        "key",
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    session::record_message(
        dir.path(),
        "old",
        "user",
        "findable haystack",
        None,
        None,
        None,
        None,
    )
    .unwrap();
    session::record_session_end(
        dir.path(),
        "old",
        session::SessionStatus::Completed,
        1,
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    // Searchable to begin with.
    let found = session::search_sessions(
        dir.path(),
        &session::SessionSearchParams {
            query: Some("haystack".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(found.total, 1);

    // Simulate the pre-fix damage: an FTS entry lost between the two old
    // autocommit statements. Before `reindex_fts` existed there was no way back.
    session::with_connection(dir.path(), |conn| {
        conn.execute("DELETE FROM sessions_fts", []).unwrap();
        Ok(())
    })
    .unwrap();
    let lost = session::search_sessions(
        dir.path(),
        &session::SessionSearchParams {
            query: Some("haystack".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(lost.total, 0, "the index really is gone");

    let rows = session::reindex_fts(dir.path()).unwrap();
    assert!(rows >= 2, "reindex writes a row per session and message");
    let recovered = session::search_sessions(
        dir.path(),
        &session::SessionSearchParams {
            query: Some("haystack".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        recovered.total, 1,
        "reindexing makes the row findable again"
    );

    // Retention removes the finished session and everything hanging off it.
    let report = session::apply_retention(dir.path(), Utc::now() + Duration::seconds(1)).unwrap();
    assert_eq!(report.sessions, 1, "the finished session was pruned");
    assert!(session::get_session(dir.path(), "old").is_err());
    let messages = session::with_connection(dir.path(), |conn| {
        Ok(conn
            .query_row("SELECT COUNT(*) FROM session_messages", [], |r| {
                r.get::<_, i64>(0)
            })
            .unwrap())
    })
    .unwrap();
    assert_eq!(messages, 0, "messages cascade with their session");
}

/// A running session is never pruned, however old.
#[test]
fn retention_never_prunes_a_running_session() {
    let dir = workspace();
    session::record_session_start(
        dir.path(),
        "live",
        "agent",
        "Agent",
        "key",
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let report = session::apply_retention(dir.path(), Utc::now() + Duration::days(365)).unwrap();
    assert_eq!(report.sessions, 0);
    assert!(session::get_session(dir.path(), "live").is_ok());
}
