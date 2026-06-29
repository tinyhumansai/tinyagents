//! Durable observability for the harness — journals, status stores, and sinks.
//!
//! The live [`crate::harness::events`] layer fans typed [`AgentEvent`]s out to
//! in-process listeners. This module makes that history **durable and
//! correlatable** so a UI, supervisor, or test can reconstruct a recursive run
//! tree after the fact:
//!
//! - [`AgentObservation`] — a durable envelope pairing an event with its run
//!   lineage (`run_id` / `parent_run_id` / `root_run_id`), stream `offset`, and
//!   timestamp.
//! - [`HarnessEventJournal`] — an append-only, offset-addressable journal of
//!   observations, with an [`InMemoryEventJournal`] and a store-backed
//!   [`StoreEventJournal`] (stream key = run id).
//! - [`HarnessStatusStore`] — a compact "what is running now?" surface, with an
//!   [`InMemoryStatusStore`].
//! - Sinks that implement [`EventListener`]: [`FanOutSink`] (broadcast),
//!   [`RedactingSink`] (mask secrets before forwarding), [`JournalSink`]
//!   (persist observations into a journal), and [`JsonlSink`] (append records
//!   to a JSONL stream).
//!
//! Persisting sinks bridge the synchronous [`EventListener::on_event`] hook to
//! the async journal/store APIs with `futures::executor::block_on` and treat
//! persistence as best-effort: a backend error never aborts the run.

mod types;

pub use types::*;

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::harness::events::{AgentEvent, EventListener, EventRecord, HarnessRunStatus};
use crate::harness::ids::RunId;
use crate::harness::store::{AppendStore, JsonlAppendStore};

// ---------------------------------------------------------------------------
// InMemoryEventJournal
// ---------------------------------------------------------------------------

impl InMemoryEventJournal {
    /// Creates a new, empty in-memory journal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of observations stored for `run_id`.
    pub fn len(&self, run_id: &str) -> usize {
        self.runs
            .lock()
            .expect("InMemoryEventJournal lock poisoned")
            .get(run_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Returns `true` when no observations are stored for `run_id`.
    pub fn is_empty(&self, run_id: &str) -> bool {
        self.len(run_id) == 0
    }
}

#[async_trait]
impl HarnessEventJournal for InMemoryEventJournal {
    async fn append(&self, obs: AgentObservation) -> Result<u64> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|e| poisoned("InMemoryEventJournal", e))?;
        let entries = runs.entry(obs.run_id.as_str().to_string()).or_default();
        let offset = entries.len() as u64;
        entries.push(obs);
        Ok(offset)
    }

    async fn read_from(&self, run_id: &str, offset: u64) -> Result<Vec<AgentObservation>> {
        let runs = self
            .runs
            .lock()
            .map_err(|e| poisoned("InMemoryEventJournal", e))?;
        let Some(entries) = runs.get(run_id) else {
            return Ok(Vec::new());
        };
        Ok(entries.iter().skip(offset as usize).cloned().collect())
    }
}

// ---------------------------------------------------------------------------
// StoreEventJournal
// ---------------------------------------------------------------------------

impl<A: AppendStore> StoreEventJournal<A> {
    /// Wraps `store` as an event journal whose stream key is the run id.
    pub fn new(store: A) -> Self {
        Self { store }
    }

    /// Returns a reference to the backing store.
    pub fn store(&self) -> &A {
        &self.store
    }
}

#[async_trait]
impl<A: AppendStore + 'static> HarnessEventJournal for StoreEventJournal<A> {
    async fn append(&self, obs: AgentObservation) -> Result<u64> {
        let stream = obs.run_id.as_str().to_string();
        let value = serde_json::to_value(&obs)?;
        self.store.append(&stream, value).await
    }

    async fn read_from(&self, run_id: &str, offset: u64) -> Result<Vec<AgentObservation>> {
        let raw = self.store.read_from(run_id, offset).await?;
        let mut out = Vec::with_capacity(raw.len());
        for (_offset, value) in raw {
            out.push(serde_json::from_value(value)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// InMemoryStatusStore
// ---------------------------------------------------------------------------

impl InMemoryStatusStore {
    /// Creates a new, empty in-memory status store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of distinct runs with a recorded status.
    pub fn len(&self) -> usize {
        self.statuses
            .lock()
            .expect("InMemoryStatusStore lock poisoned")
            .len()
    }

    /// Returns `true` when no statuses have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl HarnessStatusStore for InMemoryStatusStore {
    async fn put_status(&self, status: HarnessRunStatus) -> Result<()> {
        let mut statuses = self
            .statuses
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        statuses.insert(status.run_id.as_str().to_string(), status);
        Ok(())
    }

    async fn get_status(&self, run_id: &str) -> Result<Option<HarnessRunStatus>> {
        let statuses = self
            .statuses
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        Ok(statuses.get(run_id).cloned())
    }

    async fn list_by_thread(&self, thread_id: &str) -> Result<Vec<HarnessRunStatus>> {
        let statuses = self
            .statuses
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        Ok(statuses
            .values()
            .filter(|s| {
                s.thread_id
                    .as_ref()
                    .is_some_and(|t| t.as_str() == thread_id)
            })
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// FanOutSink
// ---------------------------------------------------------------------------

impl FanOutSink {
    /// Creates an empty fan-out sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `listener` and returns `self` for builder-style chaining.
    pub fn with(mut self, listener: Arc<dyn EventListener>) -> Self {
        self.listeners.push(listener);
        self
    }

    /// Adds `listener` in place.
    pub fn add(&mut self, listener: Arc<dyn EventListener>) -> &mut Self {
        self.listeners.push(listener);
        self
    }

    /// Returns the number of downstream listeners.
    pub fn len(&self) -> usize {
        self.listeners.len()
    }

    /// Returns `true` when no listeners are registered.
    pub fn is_empty(&self) -> bool {
        self.listeners.is_empty()
    }
}

impl EventListener for FanOutSink {
    fn on_event(&self, record: &EventRecord) {
        for listener in &self.listeners {
            listener.on_event(record);
        }
    }
}

// ---------------------------------------------------------------------------
// RedactingSink
// ---------------------------------------------------------------------------

impl RedactingSink {
    /// Default mask substituted for each secret occurrence.
    pub const DEFAULT_MASK: &'static str = "[REDACTED]";

    /// Wraps `inner`, masking each substring in `secrets` with the default
    /// mask before forwarding.
    pub fn new(inner: Arc<dyn EventListener>, secrets: Vec<String>) -> Self {
        Self {
            inner,
            secrets,
            mask: Self::DEFAULT_MASK.to_string(),
        }
    }

    /// Overrides the replacement mask.
    pub fn with_mask(mut self, mask: impl Into<String>) -> Self {
        self.mask = mask.into();
        self
    }
}

impl EventListener for RedactingSink {
    fn on_event(&self, record: &EventRecord) {
        // Serialize the event, mask secrets in every string field, and rebuild
        // it. On any (de)serialization failure forward the original unchanged
        // so observability is never silently dropped.
        let Ok(mut value) = serde_json::to_value(&record.event) else {
            self.inner.on_event(record);
            return;
        };
        redact_value(&mut value, &self.secrets, &self.mask);
        let Ok(event) = serde_json::from_value::<AgentEvent>(value) else {
            self.inner.on_event(record);
            return;
        };
        let redacted = EventRecord {
            id: record.id.clone(),
            offset: record.offset,
            event,
        };
        self.inner.on_event(&redacted);
    }
}

/// Recursively replaces every occurrence of each secret substring in every
/// JSON string value with `mask`.
fn redact_value(value: &mut serde_json::Value, secrets: &[String], mask: &str) {
    match value {
        serde_json::Value::String(s) => {
            for secret in secrets {
                if !secret.is_empty() && s.contains(secret.as_str()) {
                    *s = s.replace(secret.as_str(), mask);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                redact_value(item, secrets, mask);
            }
        }
        serde_json::Value::Object(map) => {
            for entry in map.values_mut() {
                redact_value(entry, secrets, mask);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// JournalSink
// ---------------------------------------------------------------------------

impl JournalSink {
    /// Builds a journal sink that stamps every observation with `run_id`'s
    /// lineage. `root_run_id` defaults to `run_id` for a top-level run; use
    /// [`Self::with_lineage`] to set a parent and a different root.
    pub fn new(journal: Arc<dyn HarnessEventJournal>, run_id: RunId) -> Self {
        Self {
            root_run_id: run_id.clone(),
            run_id,
            parent_run_id: None,
            journal,
        }
    }

    /// Sets the parent and root run ids stamped onto every observation.
    pub fn with_lineage(mut self, parent_run_id: Option<RunId>, root_run_id: RunId) -> Self {
        self.parent_run_id = parent_run_id;
        self.root_run_id = root_run_id;
        self
    }
}

impl EventListener for JournalSink {
    fn on_event(&self, record: &EventRecord) {
        let obs = AgentObservation::from_record(
            record,
            self.run_id.clone(),
            self.parent_run_id.clone(),
            self.root_run_id.clone(),
        );
        // Best-effort durable append; never abort the run on a journal error.
        let _ = futures::executor::block_on(self.journal.append(obs));
    }
}

// ---------------------------------------------------------------------------
// JsonlSink
// ---------------------------------------------------------------------------

impl JsonlSink {
    /// Builds a sink that appends each [`EventRecord`] as a JSON line into the
    /// `stream` of `store`.
    ///
    /// [`EventRecord`]: crate::harness::events::EventRecord
    pub fn new(store: JsonlAppendStore, stream: impl Into<String>) -> Self {
        Self {
            store,
            stream: stream.into(),
        }
    }
}

impl EventListener for JsonlSink {
    fn on_event(&self, record: &EventRecord) {
        let Ok(value) = serde_json::to_value(record) else {
            return;
        };
        // Best-effort durable append; never abort the run on a store error.
        let _ = futures::executor::block_on(self.store.append(&self.stream, value));
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Builds a uniform poisoned-lock validation error for the in-memory backends.
fn poisoned<E: std::fmt::Display>(what: &str, err: E) -> crate::error::TinyAgentsError {
    crate::error::TinyAgentsError::Validation(format!("{what} lock poisoned: {err}"))
}

#[cfg(test)]
mod test;
