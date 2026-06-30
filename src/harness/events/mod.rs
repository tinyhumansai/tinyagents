//! Typed observability layer for the harness.
//!
//! Because TinyAgents is recursive — agents call agents, graphs run graphs — a
//! single user request fans out into a *tree* of runs. This module makes that
//! tree observable: every model call, tool call, and (crucially) sub-agent
//! boundary is a typed [`AgentEvent`], and child runs surface through dedicated
//! variants ([`AgentEvent::SubAgentStarted`], [`AgentEvent::SubAgentReused`],
//! [`AgentEvent::SubAgentCompleted`]) carrying recursion `depth`, so an
//! orchestrator can watch the whole recursion fold and unfold in one event
//! stream. [`HarnessRunStatus`] then summarizes a run with its `root_run_id` /
//! `parent_run_id` lineage so usage and cost roll up from leaf runs to the root.
//!
//! This module provides:
//!
//! - [`AgentEvent`] — a typed enum covering every significant harness lifecycle
//!   transition (run boundaries, model calls, tool invocations, middleware,
//!   routing, retries, and state updates).
//! - [`EventRecord`] — a monotonically-offset-keyed wrapper that pairs an
//!   [`EventId`] with the raw event.
//! - [`EventListener`] — a `Send + Sync` trait for pluggable event observers.
//! - [`EventSink`] — a cloneable, thread-safe fan-out bus that assigns ids and
//!   offsets and notifies registered listeners.
//! - [`RecordingListener`] — a collector that buffers events for tests and
//!   inspection.
//! - [`EventJournal`] — an append-only in-memory journal with offset-based
//!   replay.
//! - [`HarnessRunStatus`] — a compact run status snapshot readable without
//!   holding an in-process stream.

mod types;

pub use types::*;
// EventSinkInner is pub(crate) so bring it into scope explicitly for impls
// below; it is not re-exported by `pub use types::*`.
use types::EventSinkInner;

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::harness::cost::CostTotals;
use crate::harness::ids::{ComponentId, ExecutionStatus, HarnessPhase, RunId, ThreadId};
use crate::harness::usage::UsageTotals;

// ---------------------------------------------------------------------------
// EventSink impls
// ---------------------------------------------------------------------------

impl EventSink {
    /// Creates a new, empty event sink with no registered listeners.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventSinkInner {
                next_offset: 0,
                listeners: Vec::new(),
            })),
        }
    }

    /// Subscribes a new listener. The listener will receive every subsequent
    /// [`AgentEvent`] emitted through this sink (or any of its clones).
    pub fn subscribe(&self, listener: Arc<dyn EventListener>) {
        let mut inner = self.inner.lock().expect("EventSink lock poisoned");
        inner.listeners.push(listener);
    }

    /// Emits an event, assigning a monotonic [`EventId`] and offset, then
    /// notifying all registered listeners in insertion order.
    ///
    /// Returns the [`EventRecord`] that was dispatched so the caller can
    /// record the assigned id or offset.
    ///
    /// Listener invocations are synchronous. The sink lock is held only while
    /// assigning the record id and cloning the listener list, so callbacks may
    /// safely emit to the same sink when they guard against event recursion.
    pub fn emit(&self, event: AgentEvent) -> EventRecord {
        let (record, listeners) = {
            let mut inner = self.inner.lock().expect("EventSink lock poisoned");
            let offset = inner.next_offset;
            inner.next_offset += 1;
            let id = crate::harness::ids::EventId::new(format!("evt-{offset}"));
            let record = EventRecord { id, offset, event };
            (record, inner.listeners.clone())
        };
        for listener in &listeners {
            listener.on_event(&record);
        }
        record
    }

    /// Returns the number of currently registered listeners.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("EventSink lock poisoned")
            .listeners
            .len()
    }

    /// Returns `true` when no listeners are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for EventSink {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RecordingListener impls
// ---------------------------------------------------------------------------

impl RecordingListener {
    /// Creates a new, empty recording listener.
    pub fn new() -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a snapshot of all collected [`EventRecord`]s in arrival order.
    pub fn events(&self) -> Vec<EventRecord> {
        self.records
            .lock()
            .expect("RecordingListener lock poisoned")
            .clone()
    }

    /// Returns the number of events collected so far.
    pub fn len(&self) -> usize {
        self.records
            .lock()
            .expect("RecordingListener lock poisoned")
            .len()
    }

    /// Returns `true` when no events have been collected yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl EventListener for RecordingListener {
    fn on_event(&self, record: &EventRecord) {
        self.records
            .lock()
            .expect("RecordingListener lock poisoned")
            .push(record.clone());
    }
}

impl Default for RecordingListener {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EventJournal impls
// ---------------------------------------------------------------------------

impl EventJournal {
    /// Creates a new, empty journal.
    pub fn new() -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
            sink: EventSink::new(),
        }
    }

    /// Appends an event to the journal, assigning a monotonic id and offset.
    ///
    /// Returns the [`EventRecord`] that was stored.
    pub fn append(&self, event: AgentEvent) -> EventRecord {
        let record = self.sink.emit(event);
        self.records
            .lock()
            .expect("EventJournal lock poisoned")
            .push(record.clone());
        record
    }

    /// Returns all records with `offset >= from_offset`, in insertion order.
    ///
    /// Callers can use this to replay run history from any known checkpoint.
    /// A `from_offset` of `0` replays the full journal.
    pub fn replay_from(&self, from_offset: u64) -> Vec<EventRecord> {
        self.records
            .lock()
            .expect("EventJournal lock poisoned")
            .iter()
            .filter(|r| r.offset >= from_offset)
            .cloned()
            .collect()
    }

    /// Returns the total number of events in the journal.
    pub fn len(&self) -> usize {
        self.records
            .lock()
            .expect("EventJournal lock poisoned")
            .len()
    }

    /// Returns `true` when the journal contains no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for EventJournal {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HarnessRunStatus impls
// ---------------------------------------------------------------------------

impl HarnessRunStatus {
    /// Creates a new status record for a top-level run starting now.
    ///
    /// `parent_run_id` and `thread_id` default to `None`; mutate them after
    /// construction when the run is a child.
    pub fn new(run_id: RunId, component: ComponentId) -> Self {
        let now = SystemTime::now();
        Self {
            root_run_id: run_id.clone(),
            run_id,
            parent_run_id: None,
            thread_id: None,
            component,
            status: ExecutionStatus::Pending,
            current_phase: HarnessPhase::Idle,
            model_calls: 0,
            tool_calls: 0,
            active_model_call: None,
            active_tool_calls: Vec::new(),
            last_event_id: None,
            usage: UsageTotals::new(),
            cost: CostTotals::default(),
            started_at: now,
            updated_at: now,
            ended_at: None,
            error: None,
            metadata: serde_json::Value::Null,
        }
    }

    /// Advances the run to [`ExecutionStatus::Running`] and sets the phase.
    pub fn mark_running(&mut self, phase: HarnessPhase) {
        self.status = ExecutionStatus::Running;
        self.current_phase = phase;
        self.touch();
    }

    /// Advances the run to [`ExecutionStatus::Completed`] and records the end
    /// time.
    pub fn mark_completed(&mut self) {
        self.status = ExecutionStatus::Completed;
        self.current_phase = HarnessPhase::Done;
        let now = SystemTime::now();
        self.ended_at = Some(now);
        self.updated_at = now;
    }

    /// Advances the run to [`ExecutionStatus::Failed`], records the error, and
    /// records the end time.
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = ExecutionStatus::Failed;
        self.current_phase = HarnessPhase::Done;
        self.error = Some(error.into());
        let now = SystemTime::now();
        self.ended_at = Some(now);
        self.updated_at = now;
    }

    /// Marks the run as interrupted (waiting for external input).
    pub fn mark_interrupted(&mut self) {
        self.status = ExecutionStatus::Interrupted;
        self.touch();
    }

    /// Records the id of the most recently emitted event.
    pub fn set_last_event(&mut self, id: crate::harness::ids::EventId) {
        self.last_event_id = Some(id);
        self.touch();
    }

    /// Sets the thread id for this run.
    pub fn with_thread(mut self, thread_id: ThreadId) -> Self {
        self.thread_id = Some(thread_id);
        self
    }

    /// Sets the parent and (optionally) overrides the root run id.
    pub fn with_parent(mut self, parent_run_id: RunId, root_run_id: RunId) -> Self {
        self.parent_run_id = Some(parent_run_id);
        self.root_run_id = root_run_id;
        self
    }

    /// Updates `updated_at` to the current wall-clock time.
    fn touch(&mut self) {
        self.updated_at = SystemTime::now();
    }
}

#[cfg(test)]
mod test;
