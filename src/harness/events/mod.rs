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
use types::{EventSinkInner, JournalRecorder};

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
    ///
    /// The sink is given a process-unique stream prefix (`s<n>`), so distinct
    /// sinks never mint colliding [`EventId`]s within one process. For ids that
    /// stay unique *across process restarts* — the case that matters for a
    /// durable journal aggregating many runs — construct the sink with
    /// [`Self::with_stream_id`] seeded from a stable run/thread id instead.
    pub fn new() -> Self {
        Self::with_stream_id(format!("s{}", crate::harness::ids::next_seq()))
    }

    /// Creates a new, empty event sink whose emitted [`EventId`]s are prefixed
    /// with `stream_id`. Passing a stable, unique identifier (typically the
    /// run's or root run's id) makes event ids reproducible and collision-free
    /// across restarts: the same logical event re-emitted for the same
    /// `(stream_id, offset)` gets the same id, and two different runs never
    /// share ids even if both restart their offset counter at zero.
    pub fn with_stream_id(stream_id: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventSinkInner {
                stream_id: stream_id.into(),
                next_offset: 0,
                listeners: Vec::new(),
                pending: std::collections::VecDeque::new(),
                dispatching: false,
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
    /// Returns the [`EventRecord`] that was enqueued so the caller can record
    /// the assigned id or offset.
    ///
    /// Offset assignment and enqueueing happen under one critical section, and
    /// a single emitter at a time drains the queue, so **listeners observe
    /// records in offset order** even when multiple threads emit concurrently.
    /// When another emitter is already draining, this call returns after
    /// enqueueing and that emitter delivers the record; otherwise delivery is
    /// synchronous before this call returns. The sink lock is never held while
    /// a listener runs, so callbacks may safely emit to the same sink (the
    /// re-entrant record is queued and delivered by the active drain loop)
    /// when they guard against unbounded event recursion.
    pub fn emit(&self, event: AgentEvent) -> EventRecord {
        let (record, should_drain) = {
            let mut inner = self.inner.lock().expect("EventSink lock poisoned");
            let offset = inner.next_offset;
            inner.next_offset += 1;
            let id = crate::harness::ids::EventId::new(format!("{}-evt-{offset}", inner.stream_id));
            let record = EventRecord { id, offset, event };
            let listeners = inner.listeners.clone();
            inner.pending.push_back((record.clone(), listeners));
            let should_drain = !inner.dispatching;
            if should_drain {
                inner.dispatching = true;
            }
            (record, should_drain)
        };
        if should_drain {
            loop {
                let next = {
                    let mut inner = self.inner.lock().expect("EventSink lock poisoned");
                    match inner.pending.pop_front() {
                        Some(entry) => entry,
                        None => {
                            inner.dispatching = false;
                            break;
                        }
                    }
                };
                let (queued, listeners) = next;
                for listener in &listeners {
                    listener.on_event(&queued);
                }
            }
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
        let records = Arc::new(Mutex::new(Vec::new()));
        let sink = EventSink::new();
        // Populate the buffer from the sink's ordered dispatch path so records
        // land in offset order even when appends race: pushing after `emit`
        // returned used to store racing appends in completion order.
        sink.subscribe(Arc::new(JournalRecorder {
            records: records.clone(),
        }));
        Self { records, sink }
    }

    /// Appends an event to the journal, assigning a monotonic id and offset.
    ///
    /// Returns the [`EventRecord`] that was stored. Records become visible to
    /// [`Self::replay_from`] in offset order: when appends race, a record may
    /// be published momentarily after this call returns (by the emitter
    /// currently draining the sink's dispatch queue), but never out of order.
    pub fn append(&self, event: AgentEvent) -> EventRecord {
        self.sink.emit(event)
    }

    /// Returns all records with `offset >= from_offset`, in offset order.
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

impl EventListener for JournalRecorder {
    fn on_event(&self, record: &EventRecord) {
        self.records
            .lock()
            .expect("EventJournal lock poisoned")
            .push(record.clone());
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
