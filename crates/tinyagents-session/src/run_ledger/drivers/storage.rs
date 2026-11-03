//! v2 names, scoped declarations, storage-body conversion, conditional updates and paging.
//!
//! Records are stored as the snake_case DTOs of [`super::dto`] and
//! [`super::dto_history`] under `Key::of` ids; a single-record read-modify-write
//! is one [`Repo::update`] loop. Operations that must commit several documents
//! together (a status change with its event, a claim with its dependency checks)
//! keep their own validated-batch loops.

use super::DriverRunLedger;
use super::dto::Stored;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tinyagents_storage_kit::primitives::{self, Repo};

use tinyagents_storage_kit::{
    CollectionSpec, DocumentStoreExt, Filter, IndexSpec, Key, MIGRATIONS_COLLECTION, Query, Sort,
    StorageError, Versioned, WriteOp, WriteOptions,
};

/// The owner of the run ledger's collections and stream.
pub(super) const LEDGER_OWNER: &str = "ledger";
/// The owner of the session history's collections.
pub(super) const HISTORY_OWNER: &str = "history";
/// The body version every v1 -> v2 step produces.
pub(super) const SCHEMA_VERSION: u32 = 2;

pub(super) const WORKFLOWS: &str = "ledger.workflow_runs";
pub(super) const TELEMETRY: &str = "ledger.telemetry";
pub(super) const AGENTS: &str = "ledger.agent_runs";
pub(super) const TEAMS: &str = "ledger.teams";
pub(super) const MEMBERS: &str = "ledger.members";
pub(super) const EFFECTS: &str = "ledger.tool_effects";
pub(super) const EVENTS: &str = "ledger.event_outbox";
pub(super) const EVENT_HEADS: &str = "ledger.event_heads";
pub(super) const TASKS: &str = "ledger.tasks";
pub(super) const INPUTS: &str = "ledger.inputs";
/// The per-run event stream family: `ledger.events/<run>`.
pub(super) const EVENT_STREAM: &str = "ledger.events";
pub(super) const HIST_SESSIONS: &str = "history.sessions";
pub(super) const HIST_MESSAGES: &str = "history.messages";
pub(super) const HIST_TOOL_CALLS: &str = "history.tool_calls";
pub(super) const HIST_FTS: &str = "history.search";
/// Attempts of the validated-batch loops (the single-record loops use
/// [`Repo::update`]'s own budget).
pub(super) const CAS_ATTEMPTS: usize = 64;

/// The id of a record: its parts joined and escaped by [`Key::of`].
pub(super) fn key(parts: &[&str]) -> Result<String, StorageError> {
    Ok(Key::of(parts)?.as_str().to_owned())
}

/// The name of `run_id`'s event stream, `ledger.events/<run>`.
pub(super) fn event_stream(run_id: &str) -> Result<String, StorageError> {
    Ok(format!("{EVENT_STREAM}/{}", key(&[run_id])?))
}

/// The id of a workflow run's driver lease in `tsd.leases`: owner `ledger`,
/// key `Key::of(["workflow", run_id])`.
pub(super) fn workflow_lease_key(run_id: &str) -> Result<String, StorageError> {
    key(&["workflow", run_id])
}

/// The current time at the precision the ledger stores: whole milliseconds.
pub(super) fn now() -> DateTime<Utc> {
    DateTime::from_timestamp_millis(Utc::now().timestamp_millis()).unwrap_or_else(Utc::now)
}

/// A caller-supplied time at the precision the ledger stores.
pub(super) fn trunc(time: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(time.timestamp_millis()).unwrap_or(time)
}

/// Ignore trim-empty filter inputs, retaining nonblank values verbatim as SQL does.
pub(super) fn eq_nonblank(filter: Filter, field: &str, value: Option<&str>) -> Filter {
    match value.filter(|value| !value.trim().is_empty()) {
        Some(value) => filter.and(Filter::eq(field, value)),
        None => filter,
    }
}

/// The run ledger's `ledger.*` declarations. Expiry never deletes durable run
/// metadata, so none declares a TTL.
pub(super) fn ledger_specs() -> Vec<CollectionSpec> {
    vec![
        CollectionSpec::new(INPUTS)
            .index(IndexSpec::new("recipient_sequence", ["recipient_run_id", "sequence"]).unique())
            .index(IndexSpec::new("root", ["root_thread_id"])),
        CollectionSpec::new(EVENT_HEADS),
        CollectionSpec::new(EVENTS)
            .index(IndexSpec::new("run_sequence", ["run_id", "sequence"]).unique()),
        CollectionSpec::new(EFFECTS)
            .index(IndexSpec::new("run", ["run_id"]))
            .index(IndexSpec::new("status", ["status"]))
            .index(IndexSpec::new("started", ["started_ms"])),
        CollectionSpec::new(WORKFLOWS)
            .index(IndexSpec::new("definition", ["definition_id"]))
            .index(IndexSpec::new("status", ["status"]))
            .index(IndexSpec::new("thread", ["parent_thread_id"]))
            .index(IndexSpec::new("updated", ["updated_ms"])),
        CollectionSpec::new(TELEMETRY).index(IndexSpec::new("updated", ["updated_ms"])),
        CollectionSpec::new(AGENTS)
            .index(IndexSpec::new("status", ["status"]))
            .index(IndexSpec::new("kind", ["kind"]))
            .index(IndexSpec::new("parent", ["parent_run_id"]))
            .index(IndexSpec::new("thread", ["parent_thread_id"]))
            .index(IndexSpec::new("updated", ["updated_ms"]))
            .index(IndexSpec::new("worker", ["worker_thread_id"])),
        CollectionSpec::new(TEAMS)
            .index(IndexSpec::new("thread", ["parent_thread_id"]))
            .index(IndexSpec::new("status", ["status"]))
            .index(IndexSpec::new("updated", ["updated_ms"])),
        CollectionSpec::new(MEMBERS)
            .index(IndexSpec::new("team", ["team_id"]))
            .index(IndexSpec::new("team_name", ["team_id", "name"]).unique())
            .index(IndexSpec::new("created", ["created_ms"])),
        CollectionSpec::new(TASKS)
            .index(IndexSpec::new("team", ["team_id"]))
            .index(IndexSpec::new("status", ["status"]))
            .index(IndexSpec::new("claimed", ["claimed_by_member_id"]))
            .index(IndexSpec::new(
                "order",
                ["team_id", "order_index", "created_ms"],
            )),
    ]
}

/// The session history's `history.*` declarations.
pub(super) fn history_specs() -> Vec<CollectionSpec> {
    vec![
        CollectionSpec::new(HIST_SESSIONS)
            .index(IndexSpec::new("status_started", ["status", "started_ms"]))
            .index(IndexSpec::new(
                "parent",
                ["parent_session_id", "started_ms"],
            ))
            .index(IndexSpec::new("agent", ["agent_definition_id"]))
            .index(IndexSpec::new("thread", ["thread_id"]))
            .index(IndexSpec::new("channel", ["source_channel"]))
            .index(IndexSpec::new("started", ["started_ms"])),
        CollectionSpec::new(HIST_MESSAGES)
            .index(IndexSpec::new("session_id", ["session_id", "id"])),
        CollectionSpec::new(HIST_TOOL_CALLS)
            .index(IndexSpec::new("session_id", ["session_id", "id"]))
            .index(IndexSpec::new("tool", ["tool_name"]))
            .index(IndexSpec::new("created", ["created_ms"])),
        CollectionSpec::new(HIST_FTS)
            .index(IndexSpec::new("session", ["session_id"]))
            .searchable(["agent_name", "content", "tool_name"]),
    ]
}

/// Every collection of both owners, as one declaration list.
pub(super) fn specs() -> Vec<CollectionSpec> {
    let mut all = ledger_specs();
    all.extend(history_specs());
    all
}

/// The shared `tsd.*` collections this ledger reads: leases, counters and the
/// migration receipts.
pub(super) fn primitive_specs() -> Vec<CollectionSpec> {
    let mut all = primitives::collection_specs();
    all.push(CollectionSpec::new(MIGRATIONS_COLLECTION));
    all
}

/// A record's body, refusing one the same domain type cannot read back (JSON
/// silently encodes nonfinite floats as null) before any native write occurs.
pub(super) fn dto_of<T: Stored>(record: &T) -> Result<T::Dto, StorageError> {
    let dto = record.to_dto();
    let value = serde_json::to_value(&dto)?;
    let _: T::Dto = serde_json::from_value(value)?;
    Ok(dto)
}

/// A record as the document body the ledger stores.
pub(super) fn encode<T: Stored>(record: &T) -> Result<Value, StorageError> {
    Ok(serde_json::to_value(dto_of(record)?)?)
}

pub(super) fn decode<T: Stored>(value: Value) -> Result<T, StorageError> {
    T::from_dto(serde_json::from_value(value)?)
}

/// A conditional put of `record` at `id`.
pub(super) fn put_op<T: Stored>(
    collection: &str,
    id: String,
    record: &T,
    precondition: tinystoragedrivers_types::Precondition,
) -> Result<WriteOp, StorageError> {
    Ok(WriteOp::Put {
        collection: collection.into(),
        id,
        doc: encode(record)?,
        precondition,
    })
}

impl DriverRunLedger {
    /// Retry failed declarations on a later operation instead of silently using an undeclared layout.
    pub(super) async fn declare(&self) -> Result<(), StorageError> {
        self.declared
            .get_or_try_init(|| async {
                for spec in specs().into_iter().chain(primitive_specs()) {
                    if self.validate_collections {
                        self.scoped
                            .documents()
                            .validate_collection_spec(&spec)
                            .await?;
                    } else {
                        self.scoped.documents().ensure_collection(&spec).await?;
                    }
                }
                Ok::<(), StorageError>(())
            })
            .await
            .map(|_| ())
    }

    /// A typed handle on `collection` whose writes declare the v2 body version.
    pub(super) fn repo<T: Stored>(&self, collection: &str) -> Result<Repo<T::Dto>, StorageError> {
        Ok(Repo::new(self.scoped.documents(), collection)?
            .with_options(WriteOptions::new().schema(SCHEMA_VERSION)))
    }

    pub(super) async fn read<T: Stored + Send>(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<Versioned<T>>, StorageError> {
        self.declare().await?;
        self.scoped
            .documents()
            .get(collection, &key(&[id])?)
            .await?
            .map(|found| {
                Ok(Versioned {
                    id: found.id,
                    version: found.version,
                    doc: decode(found.doc)?,
                    meta: found.meta,
                })
            })
            .transpose()
    }

    /// A merge returns precisely its committed body; a concurrent successor cannot change its read-back.
    pub(super) async fn merge<T, F>(
        &self,
        collection: &str,
        id: &str,
        merge: F,
    ) -> Result<T, StorageError>
    where
        T: Stored + Clone + Send + Sync,
        F: Fn(Option<T>) -> T + Send + Sync,
    {
        self.merge_at(collection, &key(&[id])?, merge).await
    }

    /// [`Self::merge`] for a record whose document id is already built (a composite key).
    pub(super) async fn merge_at<T, F>(
        &self,
        collection: &str,
        doc_id: &str,
        merge: F,
    ) -> Result<T, StorageError>
    where
        T: Stored + Clone + Send + Sync,
        F: Fn(Option<T>) -> T + Send + Sync,
    {
        self.change_at(collection, doc_id, |previous| {
            let record = merge(previous);
            Ok((Some(record.clone()), record))
        })
        .await
    }

    /// Evaluate eligibility against the read version, re-evaluating only native CAS conflicts.
    ///
    /// One [`Repo::update`] loop: `change` returns the record to store (or
    /// `None` to leave it) and a result, and runs again on a lost swap, so it
    /// must be free of side effects.
    pub(super) async fn change<T, R, F>(
        &self,
        collection: &str,
        id: &str,
        change: F,
    ) -> Result<R, StorageError>
    where
        T: Stored + Send + Sync,
        R: Send,
        F: Fn(Option<T>) -> Result<(Option<T>, R), StorageError> + Send + Sync,
    {
        self.change_at(collection, &key(&[id])?, change).await
    }

    /// [`Self::change`] for a record whose document id is already built.
    pub(super) async fn change_at<T, R, F>(
        &self,
        collection: &str,
        doc_id: &str,
        change: F,
    ) -> Result<R, StorageError>
    where
        T: Stored + Send + Sync,
        R: Send,
        F: Fn(Option<T>) -> Result<(Option<T>, R), StorageError> + Send + Sync,
    {
        self.declare().await?;
        let repo = self.repo::<T>(collection)?;
        let mut outcome = None;
        repo.update(doc_id, |previous| {
            let previous = previous.map(T::from_dto).transpose()?;
            let (record, result) = change(previous)?;
            outcome = Some(result);
            record.as_ref().map(dto_of).transpose()
        })
        .await?;
        outcome.ok_or_else(|| StorageError::backend("run-ledger update ran no attempt"))
    }

    pub(super) async fn records<T: Stored + Send>(
        &self,
        collection: &str,
        query: Query,
    ) -> Result<Vec<T>, StorageError> {
        self.declare().await?;
        self.scoped
            .documents()
            .query_all(collection, &query)
            .await?
            .into_iter()
            .map(|row| decode(row.doc))
            .collect()
    }

    /// Count all matches, then follow driver cursors for the requested SQL-style offset/page.
    pub(super) async fn page<T: Stored + Send>(
        &self,
        collection: &str,
        filter: Filter,
        limit: u64,
        offset: u64,
    ) -> Result<(Vec<T>, usize), StorageError> {
        if offset > i64::MAX as u64 {
            return Err(StorageError::invalid_input(
                "run-ledger offset exceeds i64::MAX",
            ));
        }
        self.declare().await?;
        let count = usize::try_from(self.scoped.documents().count(collection, &filter).await?)
            .map_err(|_| StorageError::invalid_input("run-ledger count exceeds usize"))?;
        let mut records = Vec::new();
        if limit == 0 {
            return Ok((records, count));
        }
        let mut skip = offset;
        let mut query = Query::filter(filter)
            .sort(Sort::desc("updated_ms"))
            .limit(256);
        loop {
            let page = self.scoped.documents().query(collection, &query).await?;
            for row in page.items {
                if skip > 0 {
                    skip -= 1;
                    continue;
                }
                records.push(decode(row.doc)?);
                if records.len() as u64 == limit {
                    return Ok((records, count));
                }
            }
            match page.next {
                Some(cursor) => query.cursor = Some(cursor),
                None => return Ok((records, count)),
            }
        }
    }
}
