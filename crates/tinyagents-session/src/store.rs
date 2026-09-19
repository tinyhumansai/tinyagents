//! Connection lifecycle for the session database: path resolution, pragma
//! setup, and the autocommit/transaction entry points every other module in
//! this crate opens the database through.
//!
//! Every caller in `super::ops`, `super::retention`, and `super::run_ledger`
//! goes through [`with_connection`] or [`with_transaction`] rather than
//! opening a [`Connection`] directly, so schema migrations
//! ([`super::migrations`]) and the busy-timeout/WAL/foreign-key pragmas
//! ([`prepare_connection`]) are applied uniformly on every path into the
//! database.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use rusqlite::Connection;

use super::context::StorageContext;
use super::migrations;
use tinyagents_harness::error::Result;

/// A connection handle shared by every caller for one database path.
///
/// `rusqlite::Connection` is `Send` but not `Sync`, so a `Mutex` is the
/// minimum needed to hand the same handle to concurrent callers; it also
/// gives operations on one database path the same autocommit serialization
/// they had before, when each call opened (and implicitly serialized behind)
/// its own file handle.
type ConnectionHandle = Arc<Mutex<Connection>>;

/// Process-wide cache of open session-database connections, keyed by the
/// resolved database file path.
///
/// A `Connection::open` per operation was measured as the dominant cost of
/// session-store calls under load: each open re-parses pragmas, re-checks
/// migrations, and pays SQLite's own connection setup. Caching by path
/// reuses one connection for the lifetime of the process (or until nothing
/// references it — entries are never evicted, matching the small, bounded
/// number of distinct workspaces a single process actually opens).
fn connection_cache() -> &'static Mutex<HashMap<PathBuf, ConnectionHandle>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, ConnectionHandle>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Subdirectory of the workspace holding the session database.
const DB_SUBDIR: &str = "session_db";
/// Database filename inside [`DB_SUBDIR`].
const DB_FILE: &str = "sessions.db";

/// How long a statement waits for a competing writer's lock before giving up
/// with `SQLITE_BUSY`.
///
/// # This is a guarantee we own, not a bug fix
///
/// SQLite's own default is zero — a `BEGIN IMMEDIATE` that finds the write lock
/// held would fail instantly rather than wait — and every claim, gate and
/// sequence allocation in this module is written on the assumption that racing
/// writers *serialize* at `BEGIN`.
///
/// That assumption was, as it happens, already satisfied: `rusqlite`'s
/// `Connection::open` calls `sqlite3_busy_timeout(db, 5000)` unconditionally,
/// so the connections here have never actually had a zero timeout. Setting it
/// explicitly changes no behaviour today. It is worth doing anyway, because the
/// alternative is that a load-bearing correctness property of this module is
/// supplied by an undocumented default of a transitive dependency, invisible at
/// every call site and free to change in a patch release. Stating it here makes
/// the dependency deliberate and greppable.
///
/// Five seconds is long enough to ride out any transaction this module takes
/// (all of them are a handful of small statements) and short enough to surface
/// a genuine deadlock rather than hang.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Resolves the session database path for a workspace root.
///
/// Kept public so hosts can locate the file for backup, inspection, or
/// migration without reproducing the layout.
pub fn db_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(DB_SUBDIR).join(DB_FILE)
}

/// Returns the cached connection for `db_path`, opening and preparing one
/// (pragmas, then migrations) the first time this path is seen.
///
/// Pragma setup and migrations run exactly once per path, when the
/// connection is created — not on every call — since both are properties of
/// the connection/database, not of an individual operation.
fn cached_connection(db_path: &Path) -> Result<ConnectionHandle> {
    let mut cache = connection_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(existing) = cache.get(db_path) {
        return Ok(existing.clone());
    }

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).storage_context(&format!(
            "failed to create session_db directory: {}",
            parent.display()
        ))?;
    }

    let conn = Connection::open(db_path)
        .storage_context(&format!("failed to open session DB: {}", db_path.display()))?;
    prepare_connection(&conn)?;
    migrations::apply(&conn)?;

    let handle: ConnectionHandle = Arc::new(Mutex::new(conn));
    cache.insert(db_path.to_path_buf(), handle.clone());
    Ok(handle)
}

/// Opens (or reuses) the workspace's session database connection, applying
/// schema migrations on first use, and runs `f` against the connection.
///
/// A single connection per database path is cached for the process and
/// reused across calls, guarded by a `Mutex` so operations on the same path
/// still serialize the way they did when every call opened its own file
/// handle. Note that because the connection is cached rather than reopened,
/// a database file atomically replaced at this same path after the first
/// call will *not* be picked up — the process keeps its original handle.
pub fn with_connection<T>(
    workspace_dir: &Path,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    let db_path = db_path(workspace_dir);
    let handle = cached_connection(&db_path)?;
    let conn = handle
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&conn)
}

/// Applies the per-connection pragmas every session-DB handle needs.
///
/// `journal_mode = WAL` is persistent (stored in the file header) but is set
/// here so a freshly created database gets it; `foreign_keys` and
/// `busy_timeout` are **per connection** and must be set on every open.
fn prepare_connection(conn: &Connection) -> Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)
        .storage_context("failed to set session DB busy_timeout")?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA foreign_keys = ON;",
    )
    .storage_context("failed to apply session DB pragmas")?;
    Ok(())
}

/// Opens the session database and runs `f` inside a single **immediate**
/// write transaction, committing on `Ok` and rolling back on `Err`.
///
/// [`with_connection`] hands out an autocommit connection: each statement
/// commits on its own, so a multi-statement read-then-write sequence has no
/// isolation at all. Any operation whose correctness depends on the state it
/// read still holding when it writes — a compare-and-swap claim, a gate that
/// checks dependencies before acting — must use this instead.
///
/// `BEGIN IMMEDIATE` rather than the default deferred begin: it takes the
/// write lock up front, so two racing claims serialize at `BEGIN` instead of
/// discovering the conflict at COMMIT time and failing with `SQLITE_BUSY`
/// after one of them has already decided it won.
pub fn with_transaction<T>(
    workspace_dir: &Path,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    with_connection(workspace_dir, |conn| {
        conn.execute_batch("BEGIN IMMEDIATE")
            .storage_context("begin session DB transaction")?;
        match f(conn) {
            Ok(value) => {
                conn.execute_batch("COMMIT")
                    .storage_context("commit session DB transaction")?;
                Ok(value)
            }
            Err(err) => {
                // Roll back best-effort: the caller's error is the one worth
                // reporting, and a failed rollback (connection already gone)
                // must not mask it.
                if let Err(rollback_err) = conn.execute_batch("ROLLBACK") {
                    tracing::warn!(
                        "[session] rollback after error failed: {rollback_err} (original: {err})"
                    );
                }
                Err(err)
            }
        }
    })
}

#[cfg(test)]
pub fn with_memory_connection<T>(f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
    let conn =
        Connection::open_in_memory().storage_context("failed to open in-memory session DB")?;
    prepare_connection(&conn)?;
    migrations::apply(&conn)?;
    f(&conn)
}
