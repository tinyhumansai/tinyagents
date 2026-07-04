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
//! the async journal/store APIs through a background [`worker::AppendWorker`]:
//! `on_event` never blocks the run on I/O, persistence is best-effort (a full
//! bounded queue drops rather than stalls; backend errors are reported, not
//! propagated), and `flush` blocks until the durable log has caught up.

mod langfuse;
mod types;
mod worker;

pub(crate) use worker::{AppendWorker, DEFAULT_DRAIN_CAPACITY};

pub use langfuse::{LangfuseAuth, LangfuseClient, LangfuseTraceConfig};
// Shared Langfuse payload helpers reused by the graph observability exporter so
// ISO-8601 timestamp formatting and null-field pruning live in one place.
pub(crate) use langfuse::{clean_nulls, iso_ms};
pub use types::*;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use async_trait::async_trait;

use crate::error::Result;
use crate::harness::events::{AgentEvent, EventListener, EventRecord, HarnessRunStatus};
use crate::harness::ids::{CallId, RunId};
use crate::harness::store::{AppendStore, JsonlAppendStore};

// ---------------------------------------------------------------------------
// AgentLatencyMetrics
// ---------------------------------------------------------------------------

impl AgentLatencyMetrics {
    /// Builds latency rollups from durable observations for one agent run.
    ///
    /// Observations can contain redacted payload strings, but structural ids
    /// must be preserved. Incomplete calls are ignored because there is no
    /// terminal timestamp to measure against.
    pub fn from_observations(observations: &[AgentObservation]) -> Self {
        let mut metrics = Self::default();
        let mut run_start: Option<u64> = None;
        let mut model_starts: HashMap<CallId, (String, u64)> = HashMap::new();
        let mut tool_starts: HashMap<CallId, (String, u64)> = HashMap::new();

        for obs in observations {
            match &obs.event {
                AgentEvent::RunStarted { .. } if run_start.is_none() => {
                    run_start = Some(obs.ts_ms);
                }
                AgentEvent::RunStarted { .. } => {}
                AgentEvent::RunCompleted { .. } | AgentEvent::RunFailed { .. } => {
                    if metrics.run_elapsed_ms.is_none()
                        && let Some(start) = run_start
                    {
                        metrics.run_elapsed_ms = Some(obs.ts_ms.saturating_sub(start));
                    }
                }
                AgentEvent::ModelStarted { call_id, model } => {
                    model_starts.insert(call_id.clone(), (model.clone(), obs.ts_ms));
                }
                AgentEvent::ModelCompleted { call_id, .. } => {
                    if let Some((name, start)) = model_starts.remove(call_id) {
                        metrics.record_model_call(AgentCallLatency {
                            call_id: call_id.clone(),
                            kind: "model".to_string(),
                            name,
                            elapsed_ms: obs.ts_ms.saturating_sub(start),
                        });
                    }
                }
                AgentEvent::ToolStarted { call_id, tool_name } => {
                    tool_starts.insert(call_id.clone(), (tool_name.clone(), obs.ts_ms));
                }
                AgentEvent::ToolCompleted { call_id, .. } => {
                    if let Some((name, start)) = tool_starts.remove(call_id) {
                        metrics.record_tool_call(AgentCallLatency {
                            call_id: call_id.clone(),
                            kind: "tool".to_string(),
                            name,
                            elapsed_ms: obs.ts_ms.saturating_sub(start),
                        });
                    }
                }
                _ => {}
            }
        }

        metrics
    }

    /// Builds a run-level latency summary from a compact status snapshot.
    ///
    /// Status snapshots do not contain per-call timings, but they do carry
    /// started/updated/ended timestamps for end-to-end elapsed time.
    pub fn from_status(status: &HarnessRunStatus) -> Self {
        let end = status.ended_at.unwrap_or(status.updated_at);
        Self {
            run_elapsed_ms: duration_ms(status.started_at, end),
            ..Self::default()
        }
    }

    /// Average model-call latency for completed calls.
    pub fn average_model_ms(&self) -> Option<u64> {
        average(self.total_model_ms, self.model_calls.len())
    }

    /// Average tool-call latency for completed calls.
    pub fn average_tool_ms(&self) -> Option<u64> {
        average(self.total_tool_ms, self.tool_calls.len())
    }

    fn record_model_call(&mut self, latency: AgentCallLatency) {
        self.total_model_ms = self.total_model_ms.saturating_add(latency.elapsed_ms);
        self.max_model_ms = self.max_model_ms.max(latency.elapsed_ms);
        self.model_calls.push(latency);
    }

    fn record_tool_call(&mut self, latency: AgentCallLatency) {
        self.total_tool_ms = self.total_tool_ms.saturating_add(latency.elapsed_ms);
        self.max_tool_ms = self.max_tool_ms.max(latency.elapsed_ms);
        self.tool_calls.push(latency);
    }
}

// ---------------------------------------------------------------------------
// InMemoryEventJournal
// ---------------------------------------------------------------------------

impl InMemoryEventJournal {
    /// Creates a new, empty in-memory journal with the default run-retention
    /// cap ([`DEFAULT_JOURNAL_MAX_RUNS`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new, empty in-memory journal that retains at most
    /// `max_runs` distinct `run_id` streams, evicting the oldest (by first
    /// append) once exceeded. `0` means unbounded.
    pub fn with_max_runs(max_runs: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(EventJournalState::default())),
            max_runs,
        }
    }

    /// Returns the number of observations stored for `run_id`.
    pub fn len(&self, run_id: &str) -> usize {
        self.state
            .lock()
            .expect("InMemoryEventJournal lock poisoned")
            .streams
            .get(run_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Returns `true` when no observations are stored for `run_id`.
    pub fn is_empty(&self, run_id: &str) -> bool {
        self.len(run_id) == 0
    }

    /// Returns the number of distinct `run_id` streams currently retained.
    pub fn run_count(&self) -> usize {
        self.state
            .lock()
            .expect("InMemoryEventJournal lock poisoned")
            .streams
            .len()
    }
}

#[async_trait]
impl HarnessEventJournal for InMemoryEventJournal {
    async fn append(&self, obs: AgentObservation) -> Result<u64> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| poisoned("InMemoryEventJournal", e))?;
        let run_id = obs.run_id.as_str().to_string();
        if !state.streams.contains_key(&run_id) {
            state.order.push_back(run_id.clone());
            // Evict the oldest run(s) once the cap is exceeded so a
            // long-lived process journaling many runs doesn't grow this
            // map without bound. `max_runs == 0` disables the cap.
            if self.max_runs > 0 {
                while state.order.len() > self.max_runs {
                    if let Some(oldest) = state.order.pop_front() {
                        state.streams.remove(&oldest);
                    } else {
                        break;
                    }
                }
            }
        }
        let entries = state.streams.entry(run_id).or_default();
        let offset = entries.len() as u64;
        entries.push(obs);
        Ok(offset)
    }

    async fn read_from(&self, run_id: &str, offset: u64) -> Result<Vec<AgentObservation>> {
        let state = self
            .state
            .lock()
            .map_err(|e| poisoned("InMemoryEventJournal", e))?;
        let Some(entries) = state.streams.get(run_id) else {
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

/// Returns `true` for statuses that are still in flight and must never be
/// evicted to make room for new runs.
fn is_active_status(status: &HarnessRunStatus) -> bool {
    use crate::harness::ids::ExecutionStatus;
    matches!(
        status.status,
        ExecutionStatus::Pending | ExecutionStatus::Running | ExecutionStatus::Interrupted
    )
}

impl InMemoryStatusStore {
    /// Creates a new, empty in-memory status store with the default
    /// run-retention cap ([`DEFAULT_STATUS_STORE_MAX_RUNS`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new, empty in-memory status store that retains at most
    /// `max_runs` distinct runs, evicting the oldest terminal run once
    /// exceeded. `0` means unbounded.
    pub fn with_max_runs(max_runs: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(StatusStoreState::default())),
            max_runs,
        }
    }

    /// Returns the number of distinct runs with a recorded status.
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .expect("InMemoryStatusStore lock poisoned")
            .statuses
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
        let mut state = self
            .state
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        let run_id = status.run_id.as_str().to_string();
        if !state.statuses.contains_key(&run_id) {
            state.order.push_back(run_id.clone());
        }
        state.statuses.insert(run_id, status);

        // Evict the oldest terminal runs once the cap is exceeded so a
        // supervisor tracking many short-lived runs doesn't grow this map
        // without bound. Active runs are never evicted; if every retained
        // run is still active we simply exceed the cap rather than drop
        // in-flight state. `max_runs == 0` disables the cap.
        if self.max_runs > 0 && state.statuses.len() > self.max_runs {
            let mut requeue = Vec::new();
            while state.statuses.len() > self.max_runs {
                let Some(candidate) = state.order.pop_front() else {
                    break;
                };
                match state.statuses.get(&candidate) {
                    Some(s) if is_active_status(s) => requeue.push(candidate),
                    Some(_) => {
                        state.statuses.remove(&candidate);
                    }
                    None => {}
                }
            }
            for id in requeue.into_iter().rev() {
                state.order.push_front(id);
            }
        }
        Ok(())
    }

    async fn get_status(&self, run_id: &str) -> Result<Option<HarnessRunStatus>> {
        let state = self
            .state
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        Ok(state.statuses.get(run_id).cloned())
    }

    async fn list_by_thread(&self, thread_id: &str) -> Result<Vec<HarnessRunStatus>> {
        let state = self
            .state
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        Ok(state
            .statuses
            .values()
            .filter(|s| {
                s.thread_id
                    .as_ref()
                    .is_some_and(|t| t.as_str() == thread_id)
            })
            .cloned()
            .collect())
    }

    async fn list_by_root(&self, root_run_id: &str) -> Result<Vec<HarnessRunStatus>> {
        let state = self
            .state
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        Ok(state
            .statuses
            .values()
            .filter(|s| s.root_run_id.as_str() == root_run_id)
            .cloned()
            .collect())
    }

    async fn list_active(&self) -> Result<Vec<HarnessRunStatus>> {
        let state = self
            .state
            .lock()
            .map_err(|e| poisoned("InMemoryStatusStore", e))?;
        Ok(state
            .statuses
            .values()
            .filter(|s| is_active_status(s))
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
        // Fast path: with no secrets configured there is nothing to redact, so
        // forward the original record unchanged.
        if self.secrets.is_empty() {
            self.inner.on_event(record);
            return;
        }
        // Serialize the event, mask secrets in every string field, and rebuild
        // it. This is a security boundary, so it must fail closed: if we cannot
        // serialize, redact, and rebuild the event, we drop it rather than
        // forward a record that may still contain unredacted secrets.
        let Ok(mut value) = serde_json::to_value(&record.event) else {
            return;
        };
        redact_value(&mut value, &self.secrets, &self.mask);
        let Ok(event) = serde_json::from_value::<AgentEvent>(value) else {
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
        let worker = Arc::new(AppendWorker::spawn(
            "journal-sink",
            DEFAULT_DRAIN_CAPACITY,
            move |obs: AgentObservation| {
                let journal = Arc::clone(&journal);
                async move { journal.append(obs).await.map(|_| ()) }
            },
        ));
        Self {
            root_run_id: run_id.clone(),
            run_id,
            parent_run_id: None,
            worker,
        }
    }

    /// Sets the parent and root run ids stamped onto every observation.
    pub fn with_lineage(mut self, parent_run_id: Option<RunId>, root_run_id: RunId) -> Self {
        self.parent_run_id = parent_run_id;
        self.root_run_id = root_run_id;
        self
    }

    /// Blocks until every observation submitted so far has been persisted.
    ///
    /// Persistence is otherwise asynchronous and best-effort; call this before
    /// reading the journal back or shutting down to guarantee the durable log
    /// has caught up with the events emitted so far.
    pub fn flush(&self) {
        self.worker.flush();
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
        // Hand off to the background drain; never block the run on I/O.
        self.worker.submit(obs);
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
        let stream = stream.into();
        let worker = Arc::new(AppendWorker::spawn(
            "jsonl-sink",
            DEFAULT_DRAIN_CAPACITY,
            move |value: serde_json::Value| {
                let store = store.clone();
                let stream = stream.clone();
                async move { store.append(&stream, value).await.map(|_| ()) }
            },
        ));
        Self { worker }
    }

    /// Blocks until every record submitted so far has been appended.
    ///
    /// Persistence is otherwise asynchronous and best-effort; call this before
    /// reading the stream back or shutting down to guarantee the durable log has
    /// caught up with the events emitted so far.
    pub fn flush(&self) {
        self.worker.flush();
    }
}

impl EventListener for JsonlSink {
    fn on_event(&self, record: &EventRecord) {
        let Ok(value) = serde_json::to_value(record) else {
            return;
        };
        // Hand off to the background drain; never block the run on I/O.
        self.worker.submit(value);
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn average(total: u64, count: usize) -> Option<u64> {
    (count > 0).then_some(total / count as u64)
}

fn duration_ms(start: SystemTime, end: SystemTime) -> Option<u64> {
    end.duration_since(start)
        .ok()
        .map(|duration| duration.as_millis() as u64)
}

/// Builds a uniform poisoned-lock validation error for the in-memory backends.
fn poisoned<E: std::fmt::Display>(what: &str, err: E) -> crate::error::TinyAgentsError {
    crate::error::TinyAgentsError::Validation(format!("{what} lock poisoned: {err}"))
}

#[cfg(test)]
mod test;
