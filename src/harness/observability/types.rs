//! Type definitions for the durable observability layer.
//!
//! These types build on the live event vocabulary in
//! [`crate::harness::events`] to add **durability**: an envelope
//! ([`AgentObservation`]) that carries the run lineage and a timestamp so a
//! single event can be journaled, replayed, and correlated across a recursive
//! run tree; pluggable journal and status traits; and a set of
//! [`EventListener`] sinks that fan out, redact, and persist events.
//!
//! All public items here are re-exported through [`super`]. Trait
//! implementations, sink logic, and tests live in the sibling `mod.rs` and
//! `test.rs` files.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::harness::events::{AgentEvent, EventListener, HarnessRunStatus};
use crate::harness::ids::{CallId, EventId, RunId};
use crate::harness::store::AppendStore;

use super::worker::AppendWorker;

// ---------------------------------------------------------------------------
// AgentObservation
// ---------------------------------------------------------------------------

/// A durable observability envelope around an [`AgentEvent`].
///
/// Where [`crate::harness::events::EventRecord`] is the lightweight,
/// in-process fan-out record (just id, offset, and event), an
/// `AgentObservation` adds everything a durable journal or external trace
/// needs to correlate the event without an in-memory broadcast: the run's
/// `run_id`, its `parent_run_id` / `root_run_id` lineage, the stream `offset`,
/// and a wall-clock `ts_ms` timestamp.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentObservation {
    /// Stable, unique identifier for the underlying event.
    pub event_id: EventId,

    /// The run that emitted the event.
    pub run_id: RunId,

    /// Parent run id when this run was spawned by another run (a sub-agent or
    /// graph node). `None` for top-level runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<RunId>,

    /// Root ancestor run, equal to `run_id` for top-level runs.
    pub root_run_id: RunId,

    /// Monotonic position of the event within its run's stream.
    pub offset: u64,

    /// Wall-clock time the observation was created, in Unix-epoch milliseconds.
    pub ts_ms: u64,

    /// The typed event payload.
    pub event: AgentEvent,
}

// ---------------------------------------------------------------------------
// Agent latency metrics
// ---------------------------------------------------------------------------

/// Latency for a completed model or tool call within an agent run.
///
/// These records are derived from durable [`AgentObservation`] timestamps by
/// correlating `*.started` and `*.completed` events. They intentionally carry
/// ids and short names only, never prompt text, tool arguments, or provider
/// payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentCallLatency {
    /// The model/tool call id being measured.
    pub call_id: CallId,

    /// Stable call family, currently `"model"` or `"tool"`.
    pub kind: String,

    /// Model id or tool name associated with the call.
    pub name: String,

    /// Wall-clock elapsed time between start and completion, in milliseconds.
    pub elapsed_ms: u64,
}

/// Summarized latency metrics for a single agent run.
///
/// `run_elapsed_ms` is measured between `run.started` and the first terminal
/// `run.completed` / `run.failed` observation when both are available. Model
/// and tool latencies are measured by call id from the same observation stream.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentLatencyMetrics {
    /// End-to-end run latency, when both run start and terminal events exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_elapsed_ms: Option<u64>,

    /// Per-model-call latencies in completion order.
    #[serde(default)]
    pub model_calls: Vec<AgentCallLatency>,

    /// Per-tool-call latencies in completion order.
    #[serde(default)]
    pub tool_calls: Vec<AgentCallLatency>,

    /// Sum of completed model-call latency.
    pub total_model_ms: u64,

    /// Slowest completed model call.
    pub max_model_ms: u64,

    /// Sum of completed tool-call latency.
    pub total_tool_ms: u64,

    /// Slowest completed tool call.
    pub max_tool_ms: u64,
}

impl AgentObservation {
    /// Builds an observation from a live [`EventRecord`] and the emitting run's
    /// lineage, stamping it with the current wall-clock time.
    ///
    /// [`EventRecord`]: crate::harness::events::EventRecord
    pub fn from_record(
        record: &crate::harness::events::EventRecord,
        run_id: RunId,
        parent_run_id: Option<RunId>,
        root_run_id: RunId,
    ) -> Self {
        Self {
            event_id: record.id.clone(),
            run_id,
            parent_run_id,
            root_run_id,
            offset: record.offset,
            ts_ms: now_ms(),
            event: record.event.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// HarnessEventJournal
// ---------------------------------------------------------------------------

/// A durable, append-only journal of [`AgentObservation`]s keyed by run id.
///
/// Journals decouple durable replay from live broadcast: a UI or supervisor
/// can attach after a run has started and reconstruct history by reading from
/// a known offset rather than relying on having subscribed to an in-memory
/// [`crate::harness::events::EventSink`].
#[async_trait]
pub trait HarnessEventJournal: Send + Sync {
    /// Appends `obs` to the journal and returns the offset it was stored at
    /// within its run's stream.
    async fn append(&self, obs: AgentObservation) -> Result<u64>;

    /// Returns every observation for `run_id` whose stream offset is `>=
    /// offset`, in offset order. Reading from `0` replays the whole run;
    /// reading an unknown run returns an empty `Vec`.
    async fn read_from(&self, run_id: &str, offset: u64) -> Result<Vec<AgentObservation>>;

    /// Returns at most `limit` observations for `run_id` starting at `offset`
    /// (a bounded replay window). The default reads from `offset` and
    /// truncates; durable backends may override for a server-side limit.
    async fn read_window(
        &self,
        run_id: &str,
        offset: u64,
        limit: usize,
    ) -> Result<Vec<AgentObservation>> {
        let mut all = self.read_from(run_id, offset).await?;
        all.truncate(limit);
        Ok(all)
    }

    /// Returns observations for `run_id` from `offset` whose
    /// [`AgentEvent::kind`][crate::harness::events::AgentEvent::kind] is in
    /// `kinds`. An empty `kinds` slice matches everything. This is the
    /// UI-surface filter (text-only, tool timeline, cost updates, errors, …).
    async fn read_filtered(
        &self,
        run_id: &str,
        offset: u64,
        kinds: &[&str],
    ) -> Result<Vec<AgentObservation>> {
        let all = self.read_from(run_id, offset).await?;
        Ok(all
            .into_iter()
            .filter(|obs| kinds.is_empty() || kinds.contains(&obs.event.kind()))
            .collect())
    }
}

/// In-memory [`HarnessEventJournal`] backed by a per-run `Vec`.
///
/// Cheaply clonable through an inner [`Arc`]; clones share the same streams.
/// There is no durability — entries are lost when the last clone drops.
#[derive(Clone, Debug, Default)]
pub struct InMemoryEventJournal {
    /// `run_id → ordered observations`.
    pub(crate) runs: Arc<Mutex<HashMap<String, Vec<AgentObservation>>>>,
}

/// [`HarnessEventJournal`] backed by any [`AppendStore`].
///
/// Each run's observations are appended to the store under a stream named by
/// the run id, so `read_from` resumes from a durable offset. Pair with
/// [`crate::harness::store::JsonlAppendStore`] for a local durable journal or
/// [`crate::harness::store::InMemoryAppendStore`] for deterministic tests.
#[derive(Clone, Debug)]
pub struct StoreEventJournal<A: AppendStore> {
    /// The backing append store; stream key is the run id.
    pub(crate) store: A,
}

// ---------------------------------------------------------------------------
// HarnessStatusStore
// ---------------------------------------------------------------------------

/// A readable status surface for harness runs.
///
/// Status records are overwritten by `run_id` ("what is running now?") in
/// contrast to the append-only journal ("what happened?"). Writes must stay
/// compact: counters, ids, phase, error summaries, and timestamps — never full
/// prompts or provider payloads (see [`HarnessRunStatus`]).
#[async_trait]
pub trait HarnessStatusStore: Send + Sync {
    /// Inserts or overwrites the status for its `run_id`.
    async fn put_status(&self, status: HarnessRunStatus) -> Result<()>;

    /// Returns the latest status for `run_id`, or `None` if unknown.
    async fn get_status(&self, run_id: &str) -> Result<Option<HarnessRunStatus>>;

    /// Returns all known statuses whose `thread_id` matches `thread_id`, in
    /// unspecified order.
    async fn list_by_thread(&self, thread_id: &str) -> Result<Vec<HarnessRunStatus>>;

    /// Returns all known statuses whose `root_run_id` matches `root_run_id`,
    /// letting a supervisor walk every descendant of a run tree.
    ///
    /// The default returns an empty `Vec` (backends without enumeration cannot
    /// answer lineage queries); enumerable backends such as
    /// [`InMemoryStatusStore`] override it.
    async fn list_by_root(&self, root_run_id: &str) -> Result<Vec<HarnessRunStatus>> {
        let _ = root_run_id;
        Ok(Vec::new())
    }

    /// Returns every non-terminal (active) run status. The default returns an
    /// empty `Vec`; enumerable backends override it.
    async fn list_active(&self) -> Result<Vec<HarnessRunStatus>> {
        Ok(Vec::new())
    }
}

/// In-memory [`HarnessStatusStore`] backed by a `run_id → status` map.
///
/// Cheaply clonable through an inner [`Arc`]; clones share the same map.
#[derive(Clone, Debug, Default)]
pub struct InMemoryStatusStore {
    /// `run_id → latest status`.
    pub(crate) statuses: Arc<Mutex<HashMap<String, HarnessRunStatus>>>,
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

/// An [`EventListener`] that broadcasts every record to N inner listeners.
///
/// Listeners are notified in registration order. A failure or panic in one
/// listener is not isolated, so listeners should themselves be best-effort.
#[derive(Clone, Default)]
pub struct FanOutSink {
    /// The downstream listeners, notified in order.
    pub(crate) listeners: Vec<Arc<dyn EventListener>>,
}

/// An [`EventListener`] that masks configured secret substrings in an event's
/// string fields before forwarding to an inner listener.
///
/// Redaction is generic: the event is serialized to JSON, every string value
/// (at any depth) has each secret substring replaced by the mask, and the
/// result is deserialized back into an [`AgentEvent`]. If (de)serialization
/// fails the original record is forwarded unchanged so observability is never
/// silently dropped.
#[derive(Clone)]
pub struct RedactingSink {
    /// The downstream listener that receives the redacted record.
    pub(crate) inner: Arc<dyn EventListener>,
    /// Secret substrings to mask wherever they appear in string fields.
    pub(crate) secrets: Vec<String>,
    /// Replacement text substituted for each secret occurrence.
    pub(crate) mask: String,
}

/// An [`EventListener`] that writes each event as an [`AgentObservation`] into
/// a [`HarnessEventJournal`].
///
/// The sink is configured with the emitting run's lineage; each received
/// [`EventRecord`] is wrapped into an [`AgentObservation`] and handed to a
/// background [`AppendWorker`] that persists it off the emitting thread. The
/// append is best-effort: see [`AppendWorker`] for the backpressure/drop and
/// error policy, and use [`JournalSink::flush`] to block until the durable log
/// has caught up.
///
/// [`EventRecord`]: crate::harness::events::EventRecord
#[derive(Clone)]
pub struct JournalSink {
    /// The run that owns events delivered to this sink.
    pub(crate) run_id: RunId,
    /// Parent run id stamped onto every observation.
    pub(crate) parent_run_id: Option<RunId>,
    /// Root run id stamped onto every observation.
    pub(crate) root_run_id: RunId,
    /// Background drain that persists observations without blocking the run.
    pub(crate) worker: Arc<AppendWorker<AgentObservation>>,
}

/// An [`EventListener`] that appends each [`EventRecord`] as a JSON line into a
/// [`JsonlAppendStore`](crate::harness::store::JsonlAppendStore) stream.
///
/// This is the lightweight durable sink: it persists the live record (id,
/// offset, event) under a fixed stream name. Each record is handed to a
/// background [`AppendWorker`] that appends it off the emitting thread
/// (best-effort — see [`AppendWorker`] for the drop/error policy). Use
/// [`JsonlSink::flush`] to block until the durable log has caught up.
///
/// [`EventRecord`]: crate::harness::events::EventRecord
#[derive(Clone, Debug)]
pub struct JsonlSink {
    /// Background drain that appends records without blocking the run.
    pub(crate) worker: Arc<AppendWorker<serde_json::Value>>,
}

/// Returns the current time in Unix-epoch milliseconds, saturating at `0`.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
