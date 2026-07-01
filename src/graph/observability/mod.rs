//! Durable observability for the graph runtime — journals, status stores, and
//! the journaling event sink.
//!
//! The live [`crate::graph::stream`] layer emits transient [`GraphEvent`]s into
//! an in-process [`GraphEventSink`]. This module makes that history **durable
//! and correlatable** so a UI, supervisor, or test can reconstruct a recursive
//! graph run tree after the fact:
//!
//! - [`GraphObservation`] — a durable envelope pairing an event with its run
//!   lineage (`run_id` / `parent_run_id` / `root_run_id`), `graph_id`,
//!   `checkpoint_id`, subgraph `namespace`, `step`, `offset`, and timestamp.
//! - [`GraphEventJournal`] — an append-only, offset-addressable journal of
//!   observations, with an [`InMemoryGraphEventJournal`] and a store-backed
//!   [`StoreGraphEventJournal`] (stream key = run id).
//! - [`GraphStatusStore`] — a compact "what is running now?" surface over
//!   [`crate::graph::GraphRunStatus`], with an [`InMemoryGraphStatusStore`].
//! - [`JournalGraphSink`] — a [`GraphEventSink`] that wraps each emitted event
//!   into a [`GraphObservation`] and appends it to a journal, optionally also
//!   forwarding to a live `inner` sink.
//!
//! [`CompiledGraph`](crate::graph::CompiledGraph) can be wired to write to a
//! status store and a journal through its builder-style
//! [`with_status_store`](crate::graph::CompiledGraph::with_status_store) and
//! [`with_event_journal`](crate::graph::CompiledGraph::with_event_journal)
//! methods; both are opt-in and default off so existing runs are unchanged.
//!
//! The journaling sink bridges the synchronous [`GraphEventSink::emit`] hook to
//! the async journal API with `futures::executor::block_on` and treats
//! persistence as best-effort: a backend error never aborts the run.

mod langfuse;
mod types;

pub use langfuse::GraphLangfuseExporter;
pub use types::*;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::SystemTime;

use async_trait::async_trait;

use crate::error::Result;
use crate::graph::status::GraphRunStatus;
use crate::graph::stream::{GraphEvent, GraphEventSink};
use crate::harness::ids::{CheckpointId, EventId, GraphId, NodeId, RunId, ThreadId};
use crate::harness::store::AppendStore;

// ---------------------------------------------------------------------------
// GraphLatencyMetrics
// ---------------------------------------------------------------------------

impl GraphLatencyMetrics {
    /// Builds latency rollups from durable observations for one graph run.
    ///
    /// Incomplete steps or node executions are ignored because there is no
    /// terminal timestamp to measure against. Duplicate node activations for the
    /// same `(node, step)` are paired in FIFO order.
    pub fn from_observations(observations: &[GraphObservation]) -> Self {
        let mut metrics = Self::default();
        let mut run_start: Option<u64> = None;
        let mut step_starts: HashMap<usize, u64> = HashMap::new();
        let mut node_starts: HashMap<(NodeId, usize), VecDeque<u64>> = HashMap::new();

        for obs in observations {
            match &obs.event {
                GraphEvent::RunStarted { .. } if run_start.is_none() => {
                    run_start = Some(obs.ts_ms);
                }
                GraphEvent::RunStarted { .. } => {}
                GraphEvent::RunCompleted { .. } | GraphEvent::RunFailed { .. } => {
                    if metrics.run_elapsed_ms.is_none()
                        && let Some(start) = run_start
                    {
                        metrics.run_elapsed_ms = Some(obs.ts_ms.saturating_sub(start));
                    }
                }
                GraphEvent::StepStarted { step, .. } => {
                    step_starts.insert(*step, obs.ts_ms);
                }
                GraphEvent::StepCompleted { step } => {
                    if let Some(start) = step_starts.remove(step) {
                        metrics.record_step(GraphStepLatency {
                            step: *step,
                            elapsed_ms: obs.ts_ms.saturating_sub(start),
                        });
                    }
                }
                GraphEvent::NodeStarted { node, step } => {
                    node_starts
                        .entry((node.clone(), *step))
                        .or_default()
                        .push_back(obs.ts_ms);
                }
                GraphEvent::NodeCompleted { node, step } => {
                    if let Some(start) = pop_node_start(&mut node_starts, node, *step) {
                        metrics.record_node(GraphNodeLatency {
                            node: node.clone(),
                            step: *step,
                            elapsed_ms: obs.ts_ms.saturating_sub(start),
                            failed: false,
                        });
                    }
                }
                GraphEvent::NodeFailed { node, step, .. } => {
                    if let Some(start) = pop_node_start(&mut node_starts, node, *step) {
                        metrics.record_node(GraphNodeLatency {
                            node: node.clone(),
                            step: *step,
                            elapsed_ms: obs.ts_ms.saturating_sub(start),
                            failed: true,
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
    /// Status snapshots do not contain per-step or per-node timings, but they
    /// do carry started/updated/ended timestamps for end-to-end elapsed time.
    pub fn from_status(status: &GraphRunStatus) -> Self {
        let end = status.ended_at.unwrap_or(status.updated_at);
        Self {
            run_elapsed_ms: duration_ms(status.started_at, end),
            ..Self::default()
        }
    }

    /// Average completed-step latency.
    pub fn average_step_ms(&self) -> Option<u64> {
        average(self.total_step_ms, self.steps.len())
    }

    /// Average completed-node latency.
    pub fn average_node_ms(&self) -> Option<u64> {
        average(self.total_node_ms, self.nodes.len())
    }

    fn record_step(&mut self, latency: GraphStepLatency) {
        self.total_step_ms = self.total_step_ms.saturating_add(latency.elapsed_ms);
        self.max_step_ms = self.max_step_ms.max(latency.elapsed_ms);
        self.steps.push(latency);
    }

    fn record_node(&mut self, latency: GraphNodeLatency) {
        self.total_node_ms = self.total_node_ms.saturating_add(latency.elapsed_ms);
        self.max_node_ms = self.max_node_ms.max(latency.elapsed_ms);
        self.nodes.push(latency);
    }
}

// ---------------------------------------------------------------------------
// GraphHealthSummary
// ---------------------------------------------------------------------------

impl GraphHealthSummary {
    /// Builds a node/tool health rollup from durable observations for one run.
    ///
    /// Counts every `node.started`, `node.completed`, and `node.failed`
    /// observation per node, plus whether the run itself failed. Per-node
    /// entries are sorted by node id so the summary is deterministic.
    pub fn from_observations(observations: &[GraphObservation]) -> Self {
        let mut per_node: HashMap<NodeId, GraphNodeHealth> = HashMap::new();
        let mut summary = Self::default();

        for obs in observations {
            match &obs.event {
                GraphEvent::NodeStarted { node, .. } => {
                    entry_for(&mut per_node, node).started += 1;
                    summary.total_started += 1;
                }
                GraphEvent::NodeCompleted { node, .. } => {
                    entry_for(&mut per_node, node).completed += 1;
                    summary.total_completed += 1;
                }
                GraphEvent::NodeFailed { node, .. } => {
                    entry_for(&mut per_node, node).failed += 1;
                    summary.total_failed += 1;
                }
                GraphEvent::RunFailed { .. } => {
                    summary.run_failed = true;
                }
                _ => {}
            }
        }

        summary.nodes = per_node.into_values().collect();
        summary
            .nodes
            .sort_by(|a, b| a.node.as_str().cmp(b.node.as_str()));
        summary
    }
}

/// Returns the mutable health entry for `node`, inserting a zeroed one keyed by
/// the node id if absent.
fn entry_for<'a>(
    per_node: &'a mut HashMap<NodeId, GraphNodeHealth>,
    node: &NodeId,
) -> &'a mut GraphNodeHealth {
    per_node
        .entry(node.clone())
        .or_insert_with(|| GraphNodeHealth {
            node: node.clone(),
            started: 0,
            completed: 0,
            failed: 0,
        })
}

// ---------------------------------------------------------------------------
// InMemoryGraphEventJournal
// ---------------------------------------------------------------------------

impl InMemoryGraphEventJournal {
    /// Creates a new, empty in-memory journal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of observations stored for `run_id`.
    pub fn len(&self, run_id: &str) -> usize {
        self.runs
            .lock()
            .expect("InMemoryGraphEventJournal lock poisoned")
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
impl GraphEventJournal for InMemoryGraphEventJournal {
    async fn append(&self, obs: GraphObservation) -> Result<u64> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|e| poisoned("InMemoryGraphEventJournal", e))?;
        let entries = runs.entry(obs.run_id.as_str().to_string()).or_default();
        let offset = entries.len() as u64;
        entries.push(obs);
        Ok(offset)
    }

    async fn read_from(&self, run_id: &str, offset: u64) -> Result<Vec<GraphObservation>> {
        let runs = self
            .runs
            .lock()
            .map_err(|e| poisoned("InMemoryGraphEventJournal", e))?;
        let Some(entries) = runs.get(run_id) else {
            return Ok(Vec::new());
        };
        Ok(entries.iter().skip(offset as usize).cloned().collect())
    }
}

// ---------------------------------------------------------------------------
// StoreGraphEventJournal
// ---------------------------------------------------------------------------

impl<A: AppendStore> StoreGraphEventJournal<A> {
    /// Wraps `store` as a graph event journal whose stream key is the run id.
    pub fn new(store: A) -> Self {
        Self { store }
    }

    /// Returns a reference to the backing store.
    pub fn store(&self) -> &A {
        &self.store
    }
}

#[async_trait]
impl<A: AppendStore + 'static> GraphEventJournal for StoreGraphEventJournal<A> {
    async fn append(&self, obs: GraphObservation) -> Result<u64> {
        let stream = obs.run_id.as_str().to_string();
        let value = serde_json::to_value(&obs)?;
        self.store.append(&stream, value).await
    }

    async fn read_from(&self, run_id: &str, offset: u64) -> Result<Vec<GraphObservation>> {
        let raw = self.store.read_from(run_id, offset).await?;
        let mut out = Vec::with_capacity(raw.len());
        for (_offset, value) in raw {
            out.push(serde_json::from_value(value)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// InMemoryGraphStatusStore
// ---------------------------------------------------------------------------

impl InMemoryGraphStatusStore {
    /// Creates a new, empty in-memory status store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of distinct runs with a recorded status.
    pub fn len(&self) -> usize {
        self.statuses
            .lock()
            .expect("InMemoryGraphStatusStore lock poisoned")
            .len()
    }

    /// Returns `true` when no statuses have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl GraphStatusStore for InMemoryGraphStatusStore {
    async fn put_status(&self, status: GraphRunStatus) -> Result<()> {
        let mut statuses = self
            .statuses
            .lock()
            .map_err(|e| poisoned("InMemoryGraphStatusStore", e))?;
        statuses.insert(status.run_id.as_str().to_string(), status);
        Ok(())
    }

    async fn get_status(&self, run_id: &str) -> Result<Option<GraphRunStatus>> {
        let statuses = self
            .statuses
            .lock()
            .map_err(|e| poisoned("InMemoryGraphStatusStore", e))?;
        Ok(statuses.get(run_id).cloned())
    }

    async fn list_by_thread(&self, thread_id: &str) -> Result<Vec<GraphRunStatus>> {
        let statuses = self
            .statuses
            .lock()
            .map_err(|e| poisoned("InMemoryGraphStatusStore", e))?;
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
// JournalGraphSink
// ---------------------------------------------------------------------------

impl JournalGraphSink {
    /// Builds a journal sink for `run_id` of `graph_id`. `root_run_id` defaults
    /// to `run_id` (a top-level run) and the namespace is empty; use the
    /// builder methods to set a parent, root, thread, namespace, or downstream
    /// sink.
    pub fn new(journal: Arc<dyn GraphEventJournal>, run_id: RunId, graph_id: GraphId) -> Self {
        Self {
            journal,
            inner: None,
            root_run_id: run_id.clone(),
            run_id,
            parent_run_id: None,
            thread_id: None,
            graph_id,
            namespace: Vec::new(),
            offset: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            step: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Sets the parent and root run ids stamped onto every observation.
    pub fn with_lineage(mut self, parent_run_id: Option<RunId>, root_run_id: RunId) -> Self {
        self.parent_run_id = parent_run_id;
        self.root_run_id = root_run_id;
        self
    }

    /// Sets the thread id stamped onto every observation.
    pub fn with_thread(mut self, thread_id: Option<ThreadId>) -> Self {
        self.thread_id = thread_id;
        self
    }

    /// Sets the checkpoint namespace stamped onto every observation.
    ///
    /// A subgraph sink is given the child namespace here so its observations
    /// carry the nested path.
    pub fn with_namespace(mut self, namespace: Vec<String>) -> Self {
        self.namespace = namespace;
        self
    }

    /// Forwards every event to `inner` in addition to journaling it, so one
    /// configured sink can both persist and broadcast.
    pub fn with_inner(mut self, inner: Arc<dyn GraphEventSink>) -> Self {
        self.inner = Some(inner);
        self
    }

    /// Builds the durable observation for `event`, advancing the offset and the
    /// latest-step trackers.
    fn observe(&self, event: &GraphEvent) -> GraphObservation {
        let offset = self.offset.fetch_add(1, Ordering::Relaxed);
        // Track the latest superstep so events without a step (route, run
        // lifecycle) still carry the step they happened during.
        let step = match event.step() {
            Some(step) => {
                self.step.store(step as u64, Ordering::Relaxed);
                step
            }
            None => self.step.load(Ordering::Relaxed) as usize,
        };
        GraphObservation {
            event_id: EventId::new(format!("{}-{offset}", self.run_id.as_str())),
            run_id: self.run_id.clone(),
            root_run_id: self.root_run_id.clone(),
            parent_run_id: self.parent_run_id.clone(),
            thread_id: self.thread_id.clone(),
            graph_id: self.graph_id.clone(),
            checkpoint_id: checkpoint_of(event),
            namespace: self.namespace.clone(),
            step,
            offset,
            ts_ms: now_ms(),
            event: event.clone(),
        }
    }
}

impl GraphEventSink for JournalGraphSink {
    fn emit(&self, event: GraphEvent) {
        let obs = self.observe(&event);
        // Best-effort durable append; never abort the run on a journal error.
        let _ = futures::executor::block_on(self.journal.append(obs));
        if let Some(inner) = &self.inner {
            inner.emit(event);
        }
    }
}

/// Extracts the checkpoint id a [`GraphEvent::CheckpointSaved`] carries, so the
/// observation envelope can record it directly.
fn checkpoint_of(event: &GraphEvent) -> Option<CheckpointId> {
    match event {
        GraphEvent::CheckpointSaved { checkpoint_id } => Some(checkpoint_id.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn pop_node_start(
    starts: &mut HashMap<(NodeId, usize), VecDeque<u64>>,
    node: &NodeId,
    step: usize,
) -> Option<u64> {
    let key = (node.clone(), step);
    let queue = starts.get_mut(&key)?;
    let start = queue.pop_front();
    if queue.is_empty() {
        starts.remove(&key);
    }
    start
}

fn average(total: u64, count: usize) -> Option<u64> {
    (count > 0).then_some(total / count as u64)
}

fn duration_ms(start: SystemTime, end: SystemTime) -> Option<u64> {
    end.duration_since(start)
        .ok()
        .map(|duration| duration.as_millis() as u64)
}

/// Returns the current time in Unix-epoch milliseconds, saturating at `0`.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Builds a uniform poisoned-lock validation error for the in-memory backends.
fn poisoned<E: std::fmt::Display>(what: &str, err: E) -> crate::error::TinyAgentsError {
    crate::error::TinyAgentsError::Validation(format!("{what} lock poisoned: {err}"))
}

#[cfg(test)]
mod test;
