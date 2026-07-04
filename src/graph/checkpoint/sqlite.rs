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

use super::{Checkpoint, CheckpointMetadata, CheckpointSource, Checkpointer};
use crate::harness::ids::{CheckpointId, NodeId};
use crate::{Result, TinyAgentsError};

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

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn.lock().map_err(|_| {
            TinyAgentsError::Checkpoint("sqlite checkpointer: connection lock poisoned".to_string())
        })
    }
}

/// Table + indexes. `seq` preserves insertion order; the indexes serve thread
/// listing and `(thread_id, checkpoint_id)` parent-chain lookups.
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
    let namespace: Vec<String> =
        serde_json::from_str(&row.namespace_json).map_err(|e| sqlite_err("decode namespace", e))?;
    let next_nodes: Vec<NodeId> = serde_json::from_str(&row.next_nodes_json)
        .map_err(|e| sqlite_err("decode next_nodes", e))?;
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
            let meta = checkpoint.to_metadata();
            let namespace = serde_json::to_string(&checkpoint.namespace)
                .map_err(|e| sqlite_err("encode namespace", e))?;
            let next_nodes = serde_json::to_string(&checkpoint.next_nodes)
                .map_err(|e| sqlite_err("encode next_nodes", e))?;
            let record =
                serde_json::to_string(&checkpoint).map_err(|e| sqlite_err("encode record", e))?;

            let conn = conn.lock().map_err(|_| {
                TinyAgentsError::Checkpoint(
                    "sqlite checkpointer: connection lock poisoned".to_string(),
                )
            })?;
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
        })
        .await
        .map_err(|e| sqlite_err("join blocking put task", e))??;
        Ok(id)
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint<State>>> {
        let conn = self.lock()?;
        // Latest matching row (highest seq) for either the whole thread or a
        // specific id, mirroring the append-only history of the other backends.
        let record: Option<String> = match checkpoint_id {
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
            Some(json) => Ok(Some(
                serde_json::from_str(&json).map_err(|e| sqlite_err("decode record", e))?,
            )),
            None => Ok(None),
        }
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        let conn = self.lock()?;
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
    }

    async fn get_thread(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        // Single-pass bulk read: one indexed range query over the thread's
        // rows in insertion order, instead of the default's one point query
        // per listed id.
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare("SELECT record FROM checkpoints WHERE thread_id = ?1 ORDER BY seq ASC")
            .map_err(|e| sqlite_err("prepare get_thread", e))?;
        let rows = stmt
            .query_map(params![thread_id], |row| row.get::<_, String>(0))
            .map_err(|e| sqlite_err("query get_thread", e))?;
        let mut out = Vec::new();
        for row in rows {
            let json = row.map_err(|e| sqlite_err("read record row", e))?;
            out.push(serde_json::from_str(&json).map_err(|e| sqlite_err("decode record", e))?);
        }
        Ok(out)
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        let conn = self.lock()?;
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
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?1",
            params![thread_id],
        )
        .map_err(|e| sqlite_err("delete thread", e))?;
        Ok(())
    }

    async fn delete_checkpoints(&self, thread_id: &str, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let mut conn = self.lock()?;
        let tx = conn
            .transaction()
            .map_err(|e| sqlite_err("begin transaction", e))?;
        let mut removed = 0usize;
        for id in ids {
            removed += tx
                .execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?1 AND checkpoint_id = ?2",
                    params![thread_id, id],
                )
                .map_err(|e| sqlite_err("delete checkpoint", e))?;
        }
        tx.commit().map_err(|e| sqlite_err("commit delete", e))?;
        Ok(removed)
    }
}
