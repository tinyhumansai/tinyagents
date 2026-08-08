//! SQLite-backed [`ResponseCache`] — a durable response cache behind the
//! optional `sqlite` cargo feature.
//!
//! [`InMemoryResponseCache`][super::InMemoryResponseCache] loses everything when
//! the process exits, so every restart pays the whole provider bill again. This
//! backend keeps the same entries in a `response_cache` table keyed by
//! `(ns, key)`, mirroring LangGraph's `SqliteCache`: WAL journalling, an
//! `expiry` column, a lazy expiry purge on read, and `INSERT OR REPLACE` on
//! write.
//!
//! Expiry is stored as an absolute **Unix epoch millisecond** timestamp rather
//! than a monotonic instant, because the value has to survive a restart — a
//! `std::time::Instant` is meaningless in the next process.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};

use super::types::{CacheStats, ResponseCache};
use crate::harness::model::ModelResponse;
use crate::{Result, TinyAgentsError};

/// Table + index DDL. `(ns, key)` is the primary key so a namespaced population
/// can be dropped wholesale, and `expiry` is indexed so the periodic purge does
/// not table-scan.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS response_cache (
    ns      TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    expiry  INTEGER,
    PRIMARY KEY (ns, key)
);
CREATE INDEX IF NOT EXISTS response_cache_expiry ON response_cache (expiry);
";

/// A durable [`ResponseCache`] backed by SQLite.
///
/// Cheap to clone; clones share the same underlying connection (and therefore
/// the same data, including for in-memory databases).
///
/// # Namespacing
/// Every handle carries a namespace (default `"default"`). Two harnesses that
/// must not cross-serve — different tenants, a control and an experiment arm —
/// point at the same file with different namespaces and
/// [`clear`][ResponseCache::clear] then drops only their own population.
#[derive(Clone)]
pub struct SqliteResponseCache {
    conn: Arc<Mutex<Connection>>,
    namespace: String,
}

fn sqlite_err(context: &str, err: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Validation(format!("sqlite response cache: {context}: {err}"))
}

/// Current wall-clock time in Unix epoch milliseconds.
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl SqliteResponseCache {
    /// The namespace used when none is given.
    pub const DEFAULT_NAMESPACE: &'static str = "default";

    /// Opens (creating if needed) a SQLite-backed response cache at `path`.
    ///
    /// Pass `":memory:"` for an ephemeral in-memory database.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref()).map_err(|e| sqlite_err("open database", e))?;
        Self::from_connection(conn)
    }

    /// Opens an ephemeral in-memory cache (`":memory:"`).
    ///
    /// The database lives only as long as this handle and its clones, which
    /// share the single underlying connection.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| sqlite_err("open in-memory", e))?;
        Self::from_connection(conn)
    }

    /// Wraps a caller-owned open [`Connection`], ensuring the schema exists.
    ///
    /// WAL is requested but not required: an in-memory database rejects it, and
    /// a read-only mount may too, so a failure here is logged and ignored
    /// rather than failing the open — journalling mode is a performance knob,
    /// not a correctness one.
    pub fn from_connection(conn: Connection) -> Result<Self> {
        if let Err(error) = conn.pragma_update(None, "journal_mode", "WAL") {
            tracing::debug!(%error, "[cache] sqlite WAL unavailable; continuing with the default journal mode");
        }
        conn.execute_batch(SCHEMA)
            .map_err(|e| sqlite_err("create schema", e))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            namespace: Self::DEFAULT_NAMESPACE.to_string(),
        })
    }

    /// Returns the table + index DDL as a reusable, dependency-free SQL string,
    /// for applications that own their own SQLite connection at a possibly
    /// incompatible native-link version.
    pub fn schema_sql() -> &'static str {
        SCHEMA
    }

    /// Returns a handle scoped to `namespace`, sharing this handle's
    /// connection.
    pub fn with_namespace(&self, namespace: impl Into<String>) -> Self {
        Self {
            conn: Arc::clone(&self.conn),
            namespace: namespace.into(),
        }
    }

    /// Returns this handle's namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| sqlite_err("connection lock", "poisoned"))
    }
}

#[async_trait]
impl ResponseCache for SqliteResponseCache {
    async fn get(&self, key: &str) -> Result<Option<ModelResponse>> {
        let conn = self.lock()?;
        let row: Option<(String, Option<i64>)> = conn
            .query_row(
                "SELECT value, expiry FROM response_cache WHERE ns = ?1 AND key = ?2",
                params![self.namespace, key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| sqlite_err("read entry", e))?;
        let Some((value, expiry)) = row else {
            return Ok(None);
        };
        // Lazy expiry purge: a stale row is deleted on the way past, so a cache
        // that is read but never written still sheds expired entries.
        if expiry.is_some_and(|at| at <= now_ms()) {
            conn.execute(
                "DELETE FROM response_cache WHERE ns = ?1 AND key = ?2",
                params![self.namespace, key],
            )
            .map_err(|e| sqlite_err("purge expired entry", e))?;
            tracing::debug!(key = %key, "[cache] sqlite entry expired; treating as miss");
            return Ok(None);
        }
        let response: ModelResponse =
            serde_json::from_str(&value).map_err(|e| sqlite_err("decode entry", e))?;
        Ok(Some(response))
    }

    async fn put(&self, key: &str, value: ModelResponse) -> Result<()> {
        self.put_with_ttl(key, value, None).await
    }

    async fn put_with_ttl(
        &self,
        key: &str,
        value: ModelResponse,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let encoded = serde_json::to_string(&value).map_err(|e| sqlite_err("encode entry", e))?;
        let expiry = ttl.map(|ttl| now_ms().saturating_add(ttl.as_millis() as i64));
        let conn = self.lock()?;
        conn.execute(
            "INSERT OR REPLACE INTO response_cache (ns, key, value, expiry) \
             VALUES (?1, ?2, ?3, ?4)",
            params![self.namespace, key, encoded, expiry],
        )
        .map_err(|e| sqlite_err("write entry", e))?;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.lock()?;
        let dropped = conn
            .execute(
                "DELETE FROM response_cache WHERE ns = ?1",
                params![self.namespace],
            )
            .map_err(|e| sqlite_err("clear namespace", e))?;
        tracing::debug!(
            namespace = %self.namespace,
            dropped,
            "[cache] cleared the sqlite response cache namespace"
        );
        Ok(())
    }

    fn stats(&self) -> CacheStats {
        let Ok(conn) = self.lock() else {
            return CacheStats::default();
        };
        let row: rusqlite::Result<(i64, i64)> = conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(value)), 0) FROM response_cache WHERE ns = ?1",
            params![self.namespace],
            |row| Ok((row.get(0)?, row.get(1)?)),
        );
        match row {
            Ok((entries, bytes)) => CacheStats {
                entries: entries.max(0) as u64,
                bytes: bytes.max(0) as u64,
                ..CacheStats::default()
            },
            Err(_) => CacheStats::default(),
        }
    }
}
