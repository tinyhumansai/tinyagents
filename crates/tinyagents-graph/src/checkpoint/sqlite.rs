//! SQLite-backed [`Checkpointer`] — a durable, queryable backend behind the
//! optional `sqlite` cargo feature.
//!
//! Every checkpoint is one row in a `checkpoints` table keyed by
//! `(thread_id, checkpoint_id)`. The full [`Checkpoint`] is stored serialized as
//! JSON in the `record` column, while the lineage/listing fields (parent id,
//! namespace, next nodes, source, step, run id, and an interrupts flag) are
//! projected into their own columns so thread listing and parent-chain walks are
//! served by indexes without deserializing whole graph states.
//!
//! A monotonic `seq` primary key records insertion order, so the backend
//! reproduces the in-memory/file semantics exactly: `get(None)` returns the most
//! recently written checkpoint, `get(Some(id))` the latest row with that id, and
//! `list` walks rows in insertion order. `put` always appends a row (it never
//! updates in place), matching the append-only history the other backends keep.
//!
//! The backend opens either a file path or an in-memory database (`":memory:"`).
//! In-memory databases live for as long as the connection, so clones share the
//! single underlying connection (`Arc<Mutex<Connection>>`) and therefore the same
//! data — exactly like the in-memory map backend.
//!
//! Like [`FileCheckpointer`](super::FileCheckpointer), the [`Checkpointer`] impl
//! is bound by `State: Serialize + DeserializeOwned`; the trait itself stays
//! bound-free so non-serializable states keep using the in-memory path.

use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSource, CheckpointTuple,
    Checkpointer, PendingWrite, decode_json_err, merge_writes,
};
use crate::{Result, TinyAgentsError};
use tinyagents_harness::ids::{CheckpointId, NodeId, TaskId};

/// A [`Checkpointer`] that persists checkpoints in a SQLite database.
///
/// Cheap to clone; clones share the same underlying connection (and therefore
/// the same data, including for in-memory databases). Generic over `State`; the
/// [`Checkpointer`] impl requires `State: Serialize + DeserializeOwned`.
pub struct SqliteCheckpointer<State> {
    conn: Arc<Mutex<Connection>>,
    _marker: PhantomData<fn() -> State>,
}

impl<State> Clone for SqliteCheckpointer<State> {
    fn clone(&self) -> Self {
        Self {
            conn: self.conn.clone(),
            _marker: PhantomData,
        }
    }
}

fn sqlite_err(context: &str, err: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Checkpoint(format!("sqlite checkpointer: {context}: {err}"))
}

/// How long a statement waits for a competing writer's lock before giving up
/// with `SQLITE_BUSY`, mirroring `tinyagents-session`'s `store.rs` (see its
/// `BUSY_TIMEOUT` doc comment for why this is set explicitly rather than
/// relied on as an undocumented `rusqlite` default).
const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Applies the per-connection pragmas every checkpointer handle needs:
/// `journal_mode = WAL` for concurrent-reader-friendly durability,
/// `synchronous = NORMAL` (safe under WAL — only a whole-OS crash can lose a
/// commit, not a process crash) instead of the slower `FULL` default, and an
/// explicit `busy_timeout` so a writer contending with another connection
/// waits rather than failing immediately with `SQLITE_BUSY`.
fn prepare_connection(conn: &Connection) -> Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)
        .map_err(|e| sqlite_err("set busy_timeout", e))?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;",
    )
    .map_err(|e| sqlite_err("apply pragmas", e))?;
    Ok(())
}

impl<State> SqliteCheckpointer<State> {
    /// Opens (creating if needed) a SQLite-backed checkpointer at `path`.
    ///
    /// Pass `":memory:"` for an ephemeral in-memory database (see
    /// [`SqliteCheckpointer::in_memory`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref()).map_err(|e| sqlite_err("open database", e))?;
        Self::from_connection(conn)
    }

    /// Opens an ephemeral in-memory checkpointer (`":memory:"`).
    ///
    /// The database lives only as long as this handle and its clones, which share
    /// the single underlying connection.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| sqlite_err("open in-memory", e))?;
        Self::from_connection(conn)
    }

    /// Wraps a caller-owned open [`Connection`], ensuring the checkpoint schema
    /// exists.
    ///
    /// Use this to share a connection from your own pool or an existing
    /// application database instead of letting the checkpointer own its handle.
    /// The schema is idempotent (`CREATE TABLE IF NOT EXISTS`), so it is safe to
    /// call on a database that already has the tables.
    ///
    /// If your application depends on a *different* `rusqlite`/`libsqlite3-sys`
    /// version (a native-link conflict that prevents passing a `Connection`
    /// across the boundary), apply [`SqliteCheckpointer::schema_sql`] to your own
    /// connection instead and drive the tables directly.
    pub fn from_connection(conn: Connection) -> Result<Self> {
        prepare_connection(&conn)?;
        conn.execute_batch(SCHEMA)
            .map_err(|e| sqlite_err("create schema", e))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            _marker: PhantomData,
        })
    }

    /// Returns the checkpoint table + index DDL as a reusable, dependency-free
    /// SQL string.
    ///
    /// This is the schema-helper escape hatch for applications that own their
    /// own SQLite connection (possibly at an incompatible native-link version):
    /// execute this DDL on your connection to create the tables the checkpoint
    /// projection expects, without linking this crate's `rusqlite`.
    pub fn schema_sql() -> &'static str {
        SCHEMA
    }

    /// Test-only: reads back the live `journal_mode` pragma from this
    /// checkpointer's own connection.
    ///
    /// Exists so the pragma regression test observes exactly what
    /// [`prepare_connection`] set on `self`'s handle, rather than a second,
    /// freshly opened connection (whose own pragmas default independently —
    /// `synchronous` is per-connection, not persisted in the file, though
    /// `journal_mode` is).
    #[cfg(test)]
    pub(crate) fn journal_mode(&self) -> Result<String> {
        let conn = lock_conn(&self.conn)?;
        conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .map_err(|e| sqlite_err("read journal_mode pragma", e))
    }

    /// Test-only: reads back the live `synchronous` pragma from this
    /// checkpointer's own connection. SQLite reports it as an integer
    /// (`0` = OFF, `1` = NORMAL, `2` = FULL, `3` = EXTRA).
    #[cfg(test)]
    pub(crate) fn synchronous(&self) -> Result<i64> {
        let conn = lock_conn(&self.conn)?;
        conn.query_row("PRAGMA synchronous", [], |row| row.get(0))
            .map_err(|e| sqlite_err("read synchronous pragma", e))
    }
}

/// Locks a checkpointer's shared connection, mapping a poisoned mutex to a
/// [`TinyAgentsError`]. Free function (rather than a method) so it can be
/// called from inside a `spawn_blocking` closure that only holds the cloned
/// `Arc<Mutex<Connection>>`, not `&self`.
fn lock_conn(conn: &Arc<Mutex<Connection>>) -> Result<std::sync::MutexGuard<'_, Connection>> {
    conn.lock().map_err(|_| {
        TinyAgentsError::Checkpoint("sqlite checkpointer: connection lock poisoned".to_string())
    })
}

/// Table + indexes. `seq` preserves insertion order; the indexes serve thread
/// listing, `(thread_id, checkpoint_id)` parent-chain lookups, and — since the
/// namespace-scoped overrides landed — `(thread_id, namespace, …)` scoped
/// lookups.
///
/// # `namespace` is a first-class, indexed column
///
/// It holds the canonical JSON encoding of the namespace vector, which
/// `serde_json` emits deterministically, so equality on the column is exactly
/// equality on the namespace. It was already stored this way; what was missing
/// were the indexes, and therefore the ability to *push the scope down into
/// SQL* at all. Without them `get_scoped`, `get_tuple` and `state_history` all
/// fell back to the trait defaults, which scan the whole thread once per
/// lineage hop — O(H²) per namespaced read on the one backend that had no
/// business being in that class. Both are `CREATE INDEX IF NOT EXISTS`, so an
/// existing database picks them up on the next open with no migration step.
///
/// `checkpoint_writes` is the partial-failure ledger. Its primary key
/// `(thread_id, namespace, checkpoint_id, task_id, idx)` mirrors LangGraph's
/// writes table and is what makes `put_writes` idempotent in SQL rather than in
/// application code.
const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS checkpoints (
    seq                  INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id            TEXT    NOT NULL,
    checkpoint_id        TEXT    NOT NULL,
    parent_checkpoint_id TEXT,
    run_id               TEXT,
    namespace            TEXT    NOT NULL,
    next_nodes           TEXT    NOT NULL,
    source               TEXT    NOT NULL,
    step                 INTEGER NOT NULL,
    has_interrupts       INTEGER NOT NULL,
    record               TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints (thread_id, seq);
CREATE INDEX IF NOT EXISTS idx_checkpoints_lookup ON checkpoints (thread_id, checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_scoped ON checkpoints (thread_id, namespace, seq);
CREATE INDEX IF NOT EXISTS idx_checkpoints_scoped_lookup
    ON checkpoints (thread_id, namespace, checkpoint_id, seq);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id     TEXT    NOT NULL,
    namespace     TEXT    NOT NULL,
    checkpoint_id TEXT    NOT NULL,
    task_id       TEXT    NOT NULL,
    idx           INTEGER NOT NULL,
    node          TEXT    NOT NULL,
    channel       TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    PRIMARY KEY (thread_id, namespace, checkpoint_id, task_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread
    ON checkpoint_writes (thread_id, checkpoint_id);

-- C3/R4: the durable half of the per-thread execution lease. The executor
-- holds an in-process lock for the run's lifetime (see
-- `compiled::executor::execute`) AND claims this row, so a lease surviving a
-- crashed owner past its TTL is reclaimable by a different process instead of
-- stranding the thread forever.
CREATE TABLE IF NOT EXISTS thread_leases (
    thread_id  TEXT    PRIMARY KEY,
    owner      TEXT    NOT NULL,
    expires_at INTEGER NOT NULL
);
";

/// Recursive parent-chain walk for [`Checkpointer::state_history`], newest
/// first, capped by `?3` rows — a caller-supplied `limit` (already clamped to
/// the namespace's total row count) rather than a truncation applied in Rust
/// after decoding everything.
///
/// `latest` first dedups: `put` never updates a row in place (see the module
/// doc), so a reused `checkpoint_id` — from `copy_thread`, a fork, or a
/// hand-written record — can have more than one row. Keeping only the
/// highest-`seq` row per id makes `checkpoint_id` unique within `latest`,
/// which is what makes the following recursive join well-defined: a plain
/// `JOIN` on `parent_checkpoint_id = checkpoint_id` over non-unique ids could
/// fan out.
///
/// `chain` walks from the head (the namespace's own highest-`seq` row) along
/// `parent_checkpoint_id`, capped by `depth < ?3`. Termination is guaranteed
/// even over a corrupted lineage with a parent cycle: `latest` has at most one
/// row per distinct id, so after at most that many hops the depth cap (itself
/// bounded by the namespace's total row count, see the caller) stops the
/// recursion regardless of what the pointers do.
const STATE_HISTORY_CTE: &str = "\
WITH RECURSIVE latest AS (
    SELECT c1.seq, c1.checkpoint_id, c1.parent_checkpoint_id, c1.record
    FROM checkpoints c1
    WHERE c1.thread_id = ?1 AND c1.namespace = ?2
      AND c1.seq = (
        SELECT MAX(c2.seq) FROM checkpoints c2
        WHERE c2.thread_id = c1.thread_id AND c2.namespace = c1.namespace
          AND c2.checkpoint_id = c1.checkpoint_id
      )
),
chain(seq, checkpoint_id, parent_checkpoint_id, record, depth) AS (
    SELECT seq, checkpoint_id, parent_checkpoint_id, record, 1
    FROM latest
    WHERE seq = (SELECT MAX(seq) FROM latest)
    UNION ALL
    SELECT l.seq, l.checkpoint_id, l.parent_checkpoint_id, l.record, chain.depth + 1
    FROM latest l
    JOIN chain ON l.checkpoint_id = chain.parent_checkpoint_id
    WHERE chain.depth < ?3
)
SELECT record FROM chain ORDER BY depth ASC LIMIT ?3;
";

/// The projected listing columns read from one `checkpoints` row.
struct MetaRow {
    thread_id: String,
    checkpoint_id: String,
    run_id: Option<String>,
    parent_checkpoint_id: Option<String>,
    namespace_json: String,
    next_nodes_json: String,
    source: String,
    step: i64,
    has_interrupts: i64,
}

/// Reconstructs a [`CheckpointMetadata`] from the projected listing columns,
/// without touching the full serialized record.
fn row_metadata(row: MetaRow) -> Result<CheckpointMetadata> {
    let namespace: Vec<String> = serde_json::from_str(&row.namespace_json)
        .map_err(|e| decode_json_err("sqlite checkpointer", "namespace", e))?;
    let next_nodes: Vec<NodeId> = serde_json::from_str(&row.next_nodes_json)
        .map_err(|e| decode_json_err("sqlite checkpointer", "next_nodes", e))?;
    Ok(CheckpointMetadata {
        thread_id: row.thread_id,
        checkpoint_id: row.checkpoint_id,
        run_id: row.run_id,
        parent_checkpoint_id: row.parent_checkpoint_id,
        namespace,
        next_nodes,
        has_interrupts: row.has_interrupts != 0,
        source: CheckpointSource::parse(&row.source).unwrap_or(CheckpointSource::Loop),
        step: row.step as usize,
    })
}

/// Inserts one `checkpoints` row for `checkpoint`.
///
/// Takes `&Connection` rather than `&SqliteCheckpointer` so it can run either
/// directly against a locked connection ([`Checkpointer::put`]) or against a
/// [`rusqlite::Transaction`] (which derefs to `Connection`) shared with a
/// `put_writes` insert in the same commit
/// ([`SqliteCheckpointer`]'s `put_with_writes` override).
fn insert_checkpoint_row<State: Serialize>(
    conn: &Connection,
    checkpoint: &Checkpoint<State>,
) -> Result<()> {
    let meta = checkpoint.to_metadata();
    let namespace = serde_json::to_string(&checkpoint.namespace)
        .map_err(|e| sqlite_err("encode namespace", e))?;
    let next_nodes = serde_json::to_string(&checkpoint.next_nodes)
        .map_err(|e| sqlite_err("encode next_nodes", e))?;
    let record = serde_json::to_string(checkpoint).map_err(|e| sqlite_err("encode record", e))?;
    conn.execute(
        "INSERT INTO checkpoints (
            thread_id, checkpoint_id, parent_checkpoint_id, run_id,
            namespace, next_nodes, source, step, has_interrupts, record
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            checkpoint.thread_id,
            checkpoint.checkpoint_id,
            checkpoint.parent_checkpoint_id,
            checkpoint.run_id,
            namespace,
            next_nodes,
            meta.source.as_str(),
            meta.step as i64,
            i64::from(meta.has_interrupts),
            record,
        ],
    )
    .map_err(|e| sqlite_err("insert checkpoint", e))?;
    Ok(())
}

/// Inserts `writes` into `checkpoint_writes` for the checkpoint addressed by
/// `config`, returning how many rows were actually stored (a control-plane
/// write always stores; a data write with an already-seen `(task_id, idx)` is
/// ignored — see [`Checkpointer::put_writes`]'s doc comment for the rule).
///
/// Takes `&Connection` for the same reason as [`insert_checkpoint_row`]: it
/// runs standalone under [`Checkpointer::put_writes`] and shares a
/// transaction with [`insert_checkpoint_row`] under `put_with_writes`.
fn insert_checkpoint_writes(
    conn: &Connection,
    config: &CheckpointConfig,
    checkpoint_id: &str,
    writes: &[PendingWrite],
) -> Result<usize> {
    let namespace_json =
        serde_json::to_string(&config.namespace).map_err(|e| sqlite_err("encode namespace", e))?;
    let mut stored = 0usize;
    for write in writes {
        // The replace-vs-ignore rule pushed into SQL: a control-plane write
        // (`idx < 0`) legitimately changes on a retry and upserts, while a
        // data write is append-once so a retried `put_writes` is a no-op.
        // Doing it with two conflict clauses rather than a read-then-write
        // keeps it correct under concurrent writers.
        let sql = if write.is_control_plane() {
            "INSERT INTO checkpoint_writes
                (thread_id, namespace, checkpoint_id, task_id, idx, node, channel, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(thread_id, namespace, checkpoint_id, task_id, idx) DO UPDATE SET
                node = excluded.node,
                channel = excluded.channel,
                payload = excluded.payload"
        } else {
            "INSERT INTO checkpoint_writes
                (thread_id, namespace, checkpoint_id, task_id, idx, node, channel, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(thread_id, namespace, checkpoint_id, task_id, idx) DO NOTHING"
        };
        let payload = serde_json::to_string(&write.payload)
            .map_err(|e| sqlite_err("encode write payload", e))?;
        stored += conn
            .execute(
                sql,
                params![
                    config.thread_id,
                    namespace_json,
                    checkpoint_id,
                    write.task_id.as_str(),
                    write.idx,
                    write.node.as_str(),
                    write.channel,
                    payload,
                ],
            )
            .map_err(|e| sqlite_err("insert checkpoint write", e))?;
    }
    Ok(stored)
}

#[async_trait]
impl<State> Checkpointer<State> for SqliteCheckpointer<State>
where
    State: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId> {
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        // Serialize + the synchronous rusqlite insert (which also blocks on the
        // connection mutex) is blocking work; run it on the blocking pool so it
        // never stalls a tokio worker on the step-critical path.
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = lock_conn(&conn)?;
            insert_checkpoint_row(&conn, &checkpoint)
        })
        .await
        .map_err(|e| sqlite_err("join blocking put task", e))??;
        Ok(id)
    }

    async fn put_with_writes(
        &self,
        checkpoint: Checkpoint<State>,
        writes: &[PendingWrite],
    ) -> Result<CheckpointId> {
        // One transaction covering both the checkpoint row and its writes —
        // the boundary the executor commits at should never observe the
        // checkpoint durable but its writes lost (or vice versa) to a crash
        // between two separate autocommit statements.
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        if writes.is_empty() {
            // Nothing to share a transaction with; `put` alone is already one
            // statement.
            self.put(checkpoint).await?;
            return Ok(id);
        }
        let config = CheckpointConfig {
            thread_id: checkpoint.thread_id.clone(),
            checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
            namespace: checkpoint.namespace.clone(),
        };
        let checkpoint_id = checkpoint.checkpoint_id.clone();
        let writes = writes.to_vec();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut conn = lock_conn(&conn)?;
            let tx = conn
                .transaction()
                .map_err(|e| sqlite_err("begin put_with_writes", e))?;
            insert_checkpoint_row(&tx, &checkpoint)?;
            insert_checkpoint_writes(&tx, &config, &checkpoint_id, &writes)?;
            tx.commit()
                .map_err(|e| sqlite_err("commit put_with_writes", e))?;
            Ok(())
        })
        .await
        .map_err(|e| sqlite_err("join blocking put_with_writes task", e))??;
        Ok(id)
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint<State>>> {
        let conn = self.conn.clone();
        let thread_id = thread_id.to_string();
        let checkpoint_id = checkpoint_id.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || -> Result<Option<Checkpoint<State>>> {
            let conn = lock_conn(&conn)?;
            // Latest matching row (highest seq) for either the whole thread or
            // a specific id, mirroring the append-only history of the other
            // backends.
            let record: Option<String> = match &checkpoint_id {
                Some(id) => conn
                    .query_row(
                        "SELECT record FROM checkpoints
                         WHERE thread_id = ?1 AND checkpoint_id = ?2
                         ORDER BY seq DESC LIMIT 1",
                        params![thread_id, id],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|e| sqlite_err("query checkpoint", e))?,
                None => conn
                    .query_row(
                        "SELECT record FROM checkpoints
                         WHERE thread_id = ?1
                         ORDER BY seq DESC LIMIT 1",
                        params![thread_id],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|e| sqlite_err("query latest checkpoint", e))?,
            };
            match record {
                Some(json) => {
                    Ok(Some(serde_json::from_str(&json).map_err(|e| {
                        decode_json_err("sqlite checkpointer", "record", e)
                    })?))
                }
                None => Ok(None),
            }
        })
        .await
        .map_err(|e| sqlite_err("join blocking get task", e))?
    }

    async fn get_scoped(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &[String],
    ) -> Result<Option<Checkpoint<State>>> {
        // Pushed down to one indexed query. The trait default lists the whole
        // thread and then re-`get`s the winner, which costs a full thread scan
        // per call — and `state_history` calls it once per lineage hop.
        let conn = self.conn.clone();
        let thread_id = thread_id.to_string();
        let checkpoint_id = checkpoint_id.map(|s| s.to_string());
        let namespace_json =
            serde_json::to_string(namespace).map_err(|e| sqlite_err("encode namespace", e))?;
        tokio::task::spawn_blocking(move || -> Result<Option<Checkpoint<State>>> {
            let conn = lock_conn(&conn)?;
            let record: Option<String> = match &checkpoint_id {
                Some(id) => conn
                    .query_row(
                        "SELECT record FROM checkpoints
                         WHERE thread_id = ?1 AND namespace = ?2 AND checkpoint_id = ?3
                         ORDER BY seq DESC LIMIT 1",
                        params![thread_id, namespace_json, id],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|e| sqlite_err("query scoped checkpoint", e))?,
                None => conn
                    .query_row(
                        "SELECT record FROM checkpoints
                         WHERE thread_id = ?1 AND namespace = ?2
                         ORDER BY seq DESC LIMIT 1",
                        params![thread_id, namespace_json],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|e| sqlite_err("query latest scoped checkpoint", e))?,
            };
            match record {
                Some(json) => {
                    Ok(Some(serde_json::from_str(&json).map_err(|e| {
                        decode_json_err("sqlite checkpointer", "record", e)
                    })?))
                }
                None => Ok(None),
            }
        })
        .await
        .map_err(|e| sqlite_err("join blocking get_scoped task", e))?
    }

    async fn state_history(
        &self,
        thread_id: &str,
        namespace: &[String],
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple<State>>> {
        // Walks the parent chain with a recursive SQL CTE so `LIMIT` is applied
        // in SQL: a `state_history(Some(1))` call decodes exactly one record
        // instead of every record in the namespace. See `STATE_HISTORY_CTE`'s
        // doc comment for the query shape and the cycle-termination argument.
        let conn = self.conn.clone();
        let thread_id_owned = thread_id.to_string();
        let namespace_json =
            serde_json::to_string(namespace).map_err(|e| sqlite_err("encode namespace", e))?;
        tokio::task::spawn_blocking(move || -> Result<Vec<CheckpointTuple<State>>> {
            let conn = lock_conn(&conn)?;

            // Total row count in the namespace both answers "is there
            // anything at all" and, more importantly, bounds the CTE's
            // recursion depth: it is a safe upper bound on the number of
            // *distinct* checkpoint ids reachable, so capping recursion there
            // guarantees termination even over a hand-corrupted or forked
            // lineage with a parent cycle (the trait's documented hazard —
            // `parent_checkpoint_id` is caller-set data, not a structurally
            // acyclic pointer).
            let total: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?1 AND namespace = ?2",
                    params![thread_id_owned, namespace_json],
                    |row| row.get(0),
                )
                .map_err(|e| sqlite_err("count state_history rows", e))?;
            if total == 0 {
                return Ok(Vec::new());
            }
            let cap: i64 = match limit {
                Some(limit) => (limit as i64).min(total),
                None => total,
            };
            if cap <= 0 {
                return Ok(Vec::new());
            }

            let mut stmt = conn
                .prepare(STATE_HISTORY_CTE)
                .map_err(|e| sqlite_err("prepare state_history", e))?;
            let rows = stmt
                .query_map(params![thread_id_owned, namespace_json, cap], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(|e| sqlite_err("query state_history", e))?;
            let mut records: Vec<Checkpoint<State>> = Vec::new();
            for row in rows {
                let json = row.map_err(|e| sqlite_err("read record row", e))?;
                records.push(
                    serde_json::from_str(&json)
                        .map_err(|e| decode_json_err("sqlite checkpointer", "record", e))?,
                );
            }
            if records.is_empty() {
                return Ok(Vec::new());
            }
            let writes = read_writes_by_checkpoint(&conn, &thread_id_owned, &namespace_json)?;

            // The CTE already returns newest-first (depth ascending from the
            // head), so no further sorting or in-memory lineage walk is
            // needed here.
            let mut out = Vec::with_capacity(records.len());
            for checkpoint in records {
                let config = CheckpointConfig {
                    thread_id: checkpoint.thread_id.clone(),
                    checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
                    namespace: checkpoint.namespace.clone(),
                };
                let parent_config =
                    checkpoint
                        .parent_checkpoint_id
                        .as_ref()
                        .map(|parent| CheckpointConfig {
                            thread_id: checkpoint.thread_id.clone(),
                            checkpoint_id: Some(parent.clone()),
                            namespace: checkpoint.namespace.clone(),
                        });
                let pending_writes = writes
                    .get(&checkpoint.checkpoint_id)
                    .cloned()
                    .unwrap_or_else(|| checkpoint.pending_writes.clone());
                out.push(CheckpointTuple {
                    config,
                    checkpoint,
                    parent_config,
                    pending_writes,
                });
            }
            Ok(out)
        })
        .await
        .map_err(|e| sqlite_err("join blocking state_history task", e))?
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        let conn = self.conn.clone();
        let thread_id = thread_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<Vec<CheckpointMetadata>> {
            let conn = lock_conn(&conn)?;
            let mut stmt = conn
                .prepare(
                    "SELECT thread_id, checkpoint_id, run_id, parent_checkpoint_id,
                            namespace, next_nodes, source, step, has_interrupts
                     FROM checkpoints WHERE thread_id = ?1 ORDER BY seq ASC",
                )
                .map_err(|e| sqlite_err("prepare list", e))?;
            let rows = stmt
                .query_map(params![thread_id], |row| {
                    Ok(MetaRow {
                        thread_id: row.get(0)?,
                        checkpoint_id: row.get(1)?,
                        run_id: row.get(2)?,
                        parent_checkpoint_id: row.get(3)?,
                        namespace_json: row.get(4)?,
                        next_nodes_json: row.get(5)?,
                        source: row.get(6)?,
                        step: row.get(7)?,
                        has_interrupts: row.get(8)?,
                    })
                })
                .map_err(|e| sqlite_err("query list", e))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row_metadata(
                    row.map_err(|e| sqlite_err("read list row", e))?,
                )?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| sqlite_err("join blocking list task", e))?
    }

    async fn get_thread(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        // Single-pass bulk read: one indexed range query over the thread's
        // rows in insertion order, instead of the default's one point query
        // per listed id.
        let conn = self.conn.clone();
        let thread_id = thread_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<Vec<Checkpoint<State>>> {
            let conn = lock_conn(&conn)?;
            let mut stmt = conn
                .prepare("SELECT record FROM checkpoints WHERE thread_id = ?1 ORDER BY seq ASC")
                .map_err(|e| sqlite_err("prepare get_thread", e))?;
            let rows = stmt
                .query_map(params![thread_id], |row| row.get::<_, String>(0))
                .map_err(|e| sqlite_err("query get_thread", e))?;
            let mut out = Vec::new();
            for row in rows {
                let json = row.map_err(|e| sqlite_err("read record row", e))?;
                out.push(
                    serde_json::from_str(&json)
                        .map_err(|e| decode_json_err("sqlite checkpointer", "record", e))?,
                );
            }
            Ok(out)
        })
        .await
        .map_err(|e| sqlite_err("join blocking get_thread task", e))?
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
            let conn = lock_conn(&conn)?;
            let mut stmt = conn
                .prepare("SELECT DISTINCT thread_id FROM checkpoints")
                .map_err(|e| sqlite_err("prepare list_threads", e))?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| sqlite_err("query list_threads", e))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(|e| sqlite_err("read thread row", e))?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| sqlite_err("join blocking list_threads task", e))?
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let conn = self.conn.clone();
        let thread_id = thread_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut conn = lock_conn(&conn)?;
            let tx = conn
                .transaction()
                .map_err(|e| sqlite_err("begin delete_thread", e))?;
            tx.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?1",
                params![thread_id],
            )
            .map_err(|e| sqlite_err("delete thread", e))?;
            // Writes go with the thread — and across *every* namespace, not
            // just the root one, or an embedded subgraph's ledger outlives
            // its thread.
            tx.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = ?1",
                params![thread_id],
            )
            .map_err(|e| sqlite_err("delete thread writes", e))?;
            tx.commit()
                .map_err(|e| sqlite_err("commit delete_thread", e))?;
            Ok(())
        })
        .await
        .map_err(|e| sqlite_err("join blocking delete_thread task", e))?
    }

    async fn delete_checkpoints(&self, thread_id: &str, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let conn = self.conn.clone();
        let thread_id = thread_id.to_string();
        let ids = ids.to_vec();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let mut conn = lock_conn(&conn)?;
            let tx = conn
                .transaction()
                .map_err(|e| sqlite_err("begin transaction", e))?;
            let mut removed = 0usize;
            for id in &ids {
                removed += tx
                    .execute(
                        "DELETE FROM checkpoints WHERE thread_id = ?1 AND checkpoint_id = ?2",
                        params![thread_id, id],
                    )
                    .map_err(|e| sqlite_err("delete checkpoint", e))?;
                tx.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = ?1 AND checkpoint_id = ?2",
                    params![thread_id, id],
                )
                .map_err(|e| sqlite_err("delete checkpoint writes", e))?;
            }
            tx.commit().map_err(|e| sqlite_err("commit delete", e))?;
            Ok(removed)
        })
        .await
        .map_err(|e| sqlite_err("join blocking delete_checkpoints task", e))?
    }

    async fn put_writes(&self, config: &CheckpointConfig, writes: &[PendingWrite]) -> Result<()> {
        let checkpoint_id = super::require_checkpoint_id(config)?;
        if writes.is_empty() {
            return Ok(());
        }
        let config = config.clone();
        let writes = writes.to_vec();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut conn = lock_conn(&conn)?;
            let tx = conn
                .transaction()
                .map_err(|e| sqlite_err("begin put_writes", e))?;
            let stored = insert_checkpoint_writes(&tx, &config, &checkpoint_id, &writes)?;
            tx.commit()
                .map_err(|e| sqlite_err("commit put_writes", e))?;
            tracing::debug!(
                "[checkpoint:sqlite] put_writes thread={} checkpoint={checkpoint_id} offered={} stored={stored}",
                config.thread_id,
                writes.len()
            );
            Ok(())
        })
        .await
        .map_err(|e| sqlite_err("join blocking put_writes task", e))?
    }

    async fn get_writes(&self, config: &CheckpointConfig) -> Result<Vec<PendingWrite>> {
        let Some(checkpoint_id) = self.resolve_write_target(config).await? else {
            return Ok(Vec::new());
        };
        let config = config.clone();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<PendingWrite>> {
            let namespace_json = serde_json::to_string(&config.namespace)
                .map_err(|e| sqlite_err("encode namespace", e))?;
            let conn = lock_conn(&conn)?;
            let mut stmt = conn
                .prepare(
                    "SELECT node, task_id, idx, channel, payload FROM checkpoint_writes
                     WHERE thread_id = ?1 AND namespace = ?2 AND checkpoint_id = ?3
                     ORDER BY rowid ASC",
                )
                .map_err(|e| sqlite_err("prepare get_writes", e))?;
            let rows = stmt
                .query_map(
                    params![config.thread_id, namespace_json, checkpoint_id],
                    map_write_row,
                )
                .map_err(|e| sqlite_err("query get_writes", e))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(|e| sqlite_err("read write row", e))??);
            }
            Ok(out)
        })
        .await
        .map_err(|e| sqlite_err("join blocking get_writes task", e))?
    }

    async fn try_claim(&self, thread: &str, owner: &str, ttl: std::time::Duration) -> Result<bool> {
        let conn = self.conn.clone();
        let thread = thread.to_string();
        let owner = owner.to_string();
        let ttl_ms = ttl.as_millis() as i64;
        tokio::task::spawn_blocking(move || -> Result<bool> {
            let now = tinyagents_harness::ids::now_ms() as i64;
            let expires_at = now.saturating_add(ttl_ms);
            let conn = conn.lock().map_err(|_| {
                TinyAgentsError::Checkpoint(
                    "sqlite checkpointer: connection lock poisoned".to_string(),
                )
            })?;
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| sqlite_err("begin try_claim tx", e))?;
            let existing: Option<(String, i64)> = tx
                .query_row(
                    "SELECT owner, expires_at FROM thread_leases WHERE thread_id = ?1",
                    params![thread],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(|e| sqlite_err("read thread_leases", e))?;
            let claimable = match &existing {
                None => true,
                Some((existing_owner, _)) if existing_owner == &owner => true,
                Some((_, existing_expires)) => *existing_expires <= now,
            };
            if !claimable {
                tx.commit().map_err(|e| sqlite_err("commit try_claim", e))?;
                return Ok(false);
            }
            tx.execute(
                "INSERT INTO thread_leases (thread_id, owner, expires_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(thread_id) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at",
                params![thread, owner, expires_at],
            )
            .map_err(|e| sqlite_err("upsert thread_leases", e))?;
            tx.commit().map_err(|e| sqlite_err("commit try_claim", e))?;
            Ok(true)
        })
        .await
        .map_err(|e| sqlite_err("join blocking try_claim task", e))?
    }

    async fn renew(&self, thread: &str, owner: &str, ttl: std::time::Duration) -> Result<bool> {
        let conn = self.conn.clone();
        let thread = thread.to_string();
        let owner = owner.to_string();
        let ttl_ms = ttl.as_millis() as i64;
        tokio::task::spawn_blocking(move || -> Result<bool> {
            let now = tinyagents_harness::ids::now_ms() as i64;
            let expires_at = now.saturating_add(ttl_ms);
            let conn = conn.lock().map_err(|_| {
                TinyAgentsError::Checkpoint(
                    "sqlite checkpointer: connection lock poisoned".to_string(),
                )
            })?;
            let updated = conn
                .execute(
                    "UPDATE thread_leases SET expires_at = ?1
                     WHERE thread_id = ?2 AND owner = ?3",
                    params![expires_at, thread, owner],
                )
                .map_err(|e| sqlite_err("renew thread_leases", e))?;
            Ok(updated > 0)
        })
        .await
        .map_err(|e| sqlite_err("join blocking renew task", e))?
    }

    async fn release(&self, thread: &str, owner: &str) -> Result<()> {
        let conn = self.conn.clone();
        let thread = thread.to_string();
        let owner = owner.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = conn.lock().map_err(|_| {
                TinyAgentsError::Checkpoint(
                    "sqlite checkpointer: connection lock poisoned".to_string(),
                )
            })?;
            conn.execute(
                "DELETE FROM thread_leases WHERE thread_id = ?1 AND owner = ?2",
                params![thread, owner],
            )
            .map_err(|e| sqlite_err("release thread_leases", e))?;
            Ok(())
        })
        .await
        .map_err(|e| sqlite_err("join blocking release task", e))?
    }
}

/// Decodes one `checkpoint_writes` row into a [`PendingWrite`].
///
/// Returns a nested `Result` because the payload decode can fail with a serde
/// error that `rusqlite`'s row-mapper signature has no room for.
#[allow(clippy::type_complexity)]
fn map_write_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Result<PendingWrite>> {
    let node: String = row.get(0)?;
    let task_id: String = row.get(1)?;
    let idx: i64 = row.get(2)?;
    let channel: String = row.get(3)?;
    let payload_json: String = row.get(4)?;
    Ok(
        match serde_json::from_str::<serde_json::Value>(&payload_json) {
            Ok(payload) => Ok(PendingWrite {
                node: NodeId::from(node),
                task_id: TaskId::from(task_id),
                idx,
                channel,
                payload,
            }),
            Err(e) => Err(decode_json_err("sqlite checkpointer", "write payload", e)),
        },
    )
}

/// Reads every write in `thread_id`/`namespace`, grouped by checkpoint id.
///
/// One query for the whole lineage, so `state_history` does not issue a
/// `get_writes` per hop.
fn read_writes_by_checkpoint(
    conn: &Connection,
    thread_id: &str,
    namespace_json: &str,
) -> Result<std::collections::HashMap<String, Vec<PendingWrite>>> {
    let mut stmt = conn
        .prepare(
            "SELECT checkpoint_id, node, task_id, idx, channel, payload FROM checkpoint_writes
             WHERE thread_id = ?1 AND namespace = ?2 ORDER BY rowid ASC",
        )
        .map_err(|e| sqlite_err("prepare writes-by-checkpoint", e))?;
    let rows = stmt
        .query_map(params![thread_id, namespace_json], |row| {
            let checkpoint_id: String = row.get(0)?;
            let node: String = row.get(1)?;
            let task_id: String = row.get(2)?;
            let idx: i64 = row.get(3)?;
            let channel: String = row.get(4)?;
            let payload_json: String = row.get(5)?;
            Ok((checkpoint_id, node, task_id, idx, channel, payload_json))
        })
        .map_err(|e| sqlite_err("query writes-by-checkpoint", e))?;
    let mut out: std::collections::HashMap<String, Vec<PendingWrite>> =
        std::collections::HashMap::new();
    for row in rows {
        let (checkpoint_id, node, task_id, idx, channel, payload_json) =
            row.map_err(|e| sqlite_err("read write row", e))?;
        let payload = serde_json::from_str(&payload_json)
            .map_err(|e| decode_json_err("sqlite checkpointer", "write payload", e))?;
        let write = PendingWrite {
            node: NodeId::from(node),
            task_id: TaskId::from(task_id),
            idx,
            channel,
            payload,
        };
        let slot = out.entry(checkpoint_id).or_default();
        // `merge_writes` keeps the shared dedupe semantics even here, where the
        // primary key already guarantees uniqueness — one rule, one place.
        merge_writes(slot, std::slice::from_ref(&write));
    }
    Ok(out)
}
