use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use rusqlite::Connection;

use super::context::StorageContext;
use super::migrations;
use crate::error::Result;

/// Subdirectory of the workspace holding the session database.
const DB_SUBDIR: &str = "session_db";
/// Database filename inside [`DB_SUBDIR`].
const DB_FILE: &str = "sessions.db";

/// How long a statement waits for a competing writer's lock before giving up
/// with `SQLITE_BUSY`.
///
/// SQLite's default is **zero**: a `BEGIN IMMEDIATE` that finds the write lock
/// held fails instantly rather than waiting. Every claim/gate/sequence
/// allocation in this module is written on the assumption that racing writers
/// *serialize* at `BEGIN` — with no busy handler installed they do not, they
/// just fail, and the caller sees a spurious storage error under ordinary
/// concurrency. Five seconds is long enough to ride out any transaction this
/// module takes (all of them are a handful of small statements) and short
/// enough to surface a genuine deadlock rather than hang.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Databases whose migrations have already been applied **in this process**.
///
/// [`migrations::apply`] is idempotent and cheap when up to date (one indexed
/// row read), but a connection is opened per operation, so even that read is
/// worth skipping once we know the file is current. Keyed by resolved path;
/// entries are only inserted after a successful migration run.
fn migrated_paths() -> &'static Mutex<HashSet<PathBuf>> {
    static MIGRATED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    MIGRATED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Resolves the session database path for a workspace root.
///
/// Kept public so hosts can locate the file for backup, inspection, or
/// migration without reproducing the layout.
pub fn db_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(DB_SUBDIR).join(DB_FILE)
}

/// Opens the workspace's session database, applying schema migrations, and
/// runs `f` against the connection.
///
/// A connection is opened per call rather than pooled: these operations are
/// short, infrequent relative to a run's model calls, and SQLite in WAL mode
/// handles concurrent readers without a shared handle to synchronize.
pub fn with_connection<T>(
    workspace_dir: &Path,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    let db_path = db_path(workspace_dir);
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).storage_context(&format!(
            "failed to create session_db directory: {}",
            parent.display()
        ))?;
    }

    let conn = Connection::open(&db_path)
        .storage_context(&format!("failed to open session DB: {}", db_path.display()))?;
    prepare_connection(&conn)?;

    // Migrations run once per database per process; see `migrated_paths`.
    let already_migrated = {
        let guard = migrated_paths()
            .lock()
            .map_err(|e| poisoned("migration cache", e))?;
        guard.contains(&db_path)
    };
    if !already_migrated {
        migrations::apply(&conn)?;
        migrated_paths()
            .lock()
            .map_err(|e| poisoned("migration cache", e))?
            .insert(db_path.clone());
    }

    f(&conn)
}

fn poisoned(what: &str, err: impl std::fmt::Display) -> crate::error::TinyAgentsError {
    crate::error::TinyAgentsError::Storage(format!("session DB {what} lock poisoned: {err}"))
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
