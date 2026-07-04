//! Type definitions for the graph durable observability layer.
//!
//! These types add **durability** to the live [`crate::graph::stream`] event
//! vocabulary: an envelope ([`GraphObservation`]) that carries a run's lineage,
//! checkpoint coordinates, namespace, and a timestamp so a single
//! [`GraphEvent`] can be journaled, replayed, and correlated across a recursive
//! graph run tree; pluggable journal and status traits; and a
//! [`JournalGraphSink`] that wraps emitted events into observations.
//!
//! All public items here are re-exported through [`super`]. Trait
//! implementations, sink logic, and tests live in the sibling `mod.rs` and
//! `test.rs` files.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::graph::status::GraphRunStatus;
use crate::graph::stream::{GraphEvent, GraphEventSink};
use crate::harness::ids::{CheckpointId, EventId, GraphId, NodeId, RunId, ThreadId};
use crate::harness::observability::AppendWorker;
use crate::harness::store::AppendStore;

// ---------------------------------------------------------------------------
// GraphObservation
// ---------------------------------------------------------------------------

/// A durable observability envelope around a [`GraphEvent`].
///
/// Where a raw [`GraphEvent`] is the transient, in-process signal the executor
/// emits at each boundary, a `GraphObservation` adds everything a durable
/// journal or external trace needs to correlate the event without an in-memory
/// broadcast: the run's `run_id`, its `parent_run_id` / `root_run_id` lineage,
/// the owning `graph_id`, the latest `checkpoint_id`, the subgraph `namespace`,
/// the superstep `step`, the stream `offset`, and a wall-clock `ts_ms`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphObservation {
    /// Stable, unique identifier for this observation.
    pub event_id: EventId,

    /// The run that emitted the event.
    pub run_id: RunId,

    /// Root ancestor run, equal to `run_id` for top-level runs.
    pub root_run_id: RunId,

    /// Parent run id when this run was spawned by another run (a subgraph or
    /// sub-agent). `None` for top-level runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<RunId>,

    /// The thread the run executes under, when checkpointing is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<ThreadId>,

    /// The graph this run belongs to.
    pub graph_id: GraphId,

    /// The latest checkpoint id at the time the event was emitted, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<CheckpointId>,

    /// The checkpoint namespace (the child path for nested subgraph runs).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub namespace: Vec<String>,

    /// The superstep number the event is associated with (`0` before the run
    /// has entered a superstep).
    pub step: usize,

    /// Monotonic position of the observation within its run's stream.
    pub offset: u64,

    /// Wall-clock time the observation was created, in Unix-epoch milliseconds.
    pub ts_ms: u64,

    /// The typed event payload.
    pub event: GraphEvent,
}

// ---------------------------------------------------------------------------
// Graph latency metrics
// ---------------------------------------------------------------------------

/// Latency for one graph superstep.
///
/// Derived by correlating `step.started` and `step.completed` observations with
/// the same `step` value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphStepLatency {
    /// 1-based graph superstep.
    pub step: usize,

    /// Wall-clock elapsed time between step start and completion.
    pub elapsed_ms: u64,
}

/// Latency for one node handler execution.
///
/// Derived by correlating `node.started` with either `node.completed` or
/// `node.failed` for the same node and step. The `failed` flag distinguishes
/// successful and failed terminal observations while keeping both in the same
/// latency rollup.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNodeLatency {
    /// Node whose handler ran.
    pub node: NodeId,

    /// 1-based graph superstep.
    pub step: usize,

    /// Wall-clock elapsed time between node start and terminal observation.
    pub elapsed_ms: u64,

    /// True when the node ended with `node.failed`.
    pub failed: bool,
}

/// Summarized latency metrics for a single graph run.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphLatencyMetrics {
    /// End-to-end graph run latency, when both run start and terminal events
    /// exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_elapsed_ms: Option<u64>,

    /// Per-superstep latencies in completion order.
    #[serde(default)]
    pub steps: Vec<GraphStepLatency>,

    /// Per-node handler latencies in terminal-event order.
    #[serde(default)]
    pub nodes: Vec<GraphNodeLatency>,

    /// Sum of completed superstep latency.
    pub total_step_ms: u64,

    /// Slowest completed superstep.
    pub max_step_ms: u64,

    /// Sum of completed node-handler latency.
    pub total_node_ms: u64,

    /// Slowest completed node handler.
    pub max_node_ms: u64,
}

// ---------------------------------------------------------------------------
// Graph health telemetry
// ---------------------------------------------------------------------------

/// Health rollup for a single graph node across one run.
///
/// A graph node is the graph's unit of work — frequently a delegated agent or
/// tool call (see [`crate::graph::SubAgentNode`]) — so per-node success/failure
/// counts double as **tool health** telemetry for the graph. Counts are derived
/// from durable [`GraphObservation`]s by correlating `node.started` with
/// `node.completed` / `node.failed`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNodeHealth {
    /// The node these counts describe.
    pub node: NodeId,

    /// Number of `node.started` observations seen.
    pub started: u64,

    /// Number of `node.completed` observations seen.
    pub completed: u64,

    /// Number of `node.failed` observations seen.
    pub failed: u64,
}

impl GraphNodeHealth {
    /// Terminal attempts (`completed + failed`) — activations with a known
    /// outcome. `started` may exceed this while a node is still running.
    pub fn attempts(&self) -> u64 {
        self.completed.saturating_add(self.failed)
    }

    /// Fraction of terminal attempts that failed, in `0.0..=1.0`. Returns `0.0`
    /// when the node has no terminal attempts yet.
    pub fn failure_rate(&self) -> f64 {
        let attempts = self.attempts();
        if attempts == 0 {
            0.0
        } else {
            self.failed as f64 / attempts as f64
        }
    }

    /// True when the node has never failed.
    pub fn is_healthy(&self) -> bool {
        self.failed == 0
    }
}

/// Aggregate node/tool health for a single graph run.
///
/// Built from the same durable observation stream that feeds
/// [`GraphLatencyMetrics`], this is the compact "is anything unhealthy?" surface
/// a supervisor or dashboard reads, and the telemetry the Langfuse exporter
/// attaches to a trace. Per-node entries are sorted by node id for stable
/// output.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphHealthSummary {
    /// Per-node health entries, sorted by node id.
    #[serde(default)]
    pub nodes: Vec<GraphNodeHealth>,

    /// Total `node.started` observations across all nodes.
    pub total_started: u64,

    /// Total `node.completed` observations across all nodes.
    pub total_completed: u64,

    /// Total `node.failed` observations across all nodes.
    pub total_failed: u64,

    /// True when the run itself emitted `run.failed`.
    pub run_failed: bool,
}

impl GraphHealthSummary {
    /// Terminal node attempts (`total_completed + total_failed`).
    pub fn total_attempts(&self) -> u64 {
        self.total_completed.saturating_add(self.total_failed)
    }

    /// Fraction of terminal node attempts that failed, in `0.0..=1.0`.
    pub fn failure_rate(&self) -> f64 {
        let attempts = self.total_attempts();
        if attempts == 0 {
            0.0
        } else {
            self.total_failed as f64 / attempts as f64
        }
    }

    /// True when no node failed and the run did not fail.
    pub fn is_healthy(&self) -> bool {
        self.total_failed == 0 && !self.run_failed
    }

    /// The nodes that failed at least once, in node-id order.
    pub fn unhealthy_nodes(&self) -> impl Iterator<Item = &GraphNodeHealth> {
        self.nodes.iter().filter(|n| !n.is_healthy())
    }
}

// ---------------------------------------------------------------------------
// GraphEventJournal
// ---------------------------------------------------------------------------

/// A durable, append-only journal of [`GraphObservation`]s keyed by run id.
///
/// Journals decouple durable replay from live broadcast: a UI or supervisor can
/// attach after a graph run has started and reconstruct history by reading from
/// a known offset rather than relying on having subscribed to an in-memory
/// [`GraphEventSink`].
#[async_trait]
pub trait GraphEventJournal: Send + Sync {
    /// Appends `obs` to the journal and returns the offset it was stored at
    /// within its run's stream.
    async fn append(&self, obs: GraphObservation) -> Result<u64>;

    /// Returns every observation for `run_id` whose stream offset is `>=
    /// offset`, in offset order. Reading from `0` replays the whole run;
    /// reading an unknown run returns an empty `Vec`.
    async fn read_from(&self, run_id: &str, offset: u64) -> Result<Vec<GraphObservation>>;
}

/// In-memory [`GraphEventJournal`] backed by a per-run `Vec`.
///
/// Cheaply clonable through an inner [`Arc`]; clones share the same streams.
/// There is no durability — entries are lost when the last clone drops.
#[derive(Clone, Debug, Default)]
pub struct InMemoryGraphEventJournal {
    /// `run_id → ordered observations`.
    pub(crate) runs: Arc<Mutex<HashMap<String, Vec<GraphObservation>>>>,
}

/// [`GraphEventJournal`] backed by any [`AppendStore`].
///
/// Each run's observations are appended to the store under a stream named by
/// the run id, so `read_from` resumes from a durable offset. Pair with
/// [`crate::harness::store::JsonlAppendStore`] for a local durable journal or
/// [`crate::harness::store::InMemoryAppendStore`] for deterministic tests.
#[derive(Clone, Debug)]
pub struct StoreGraphEventJournal<A: AppendStore> {
    /// The backing append store; stream key is the run id.
    pub(crate) store: A,
}

// ---------------------------------------------------------------------------
// GraphStatusStore
// ---------------------------------------------------------------------------

/// A readable status surface for graph runs.
///
/// Status records are overwritten by `run_id` ("what is running now?") in
/// contrast to the append-only journal ("what happened?"). A
/// [`GraphRunStatus`] is compact: ids, step, active nodes, pending interrupts,
/// and timestamps — never full graph state, which belongs to the checkpointer.
#[async_trait]
pub trait GraphStatusStore: Send + Sync {
    /// Inserts or overwrites the status for its `run_id`.
    async fn put_status(&self, status: GraphRunStatus) -> Result<()>;

    /// Returns the latest status for `run_id`, or `None` if unknown.
    async fn get_status(&self, run_id: &str) -> Result<Option<GraphRunStatus>>;

    /// Returns all known statuses whose `thread_id` matches `thread_id`, in
    /// unspecified order.
    async fn list_by_thread(&self, thread_id: &str) -> Result<Vec<GraphRunStatus>>;
}

/// In-memory [`GraphStatusStore`] backed by a `run_id → status` map plus a
/// `thread_id → run ids` secondary index, so
/// [`list_by_thread`](GraphStatusStore::list_by_thread) is proportional to
/// that thread's runs rather than every recorded run.
///
/// # Retention
/// Unbounded by default. Long-lived processes should opt into a run cap with
/// [`InMemoryGraphStatusStore::with_max_runs`]: once the number of distinct
/// runs exceeds the cap, the store evicts the oldest **terminal** run first
/// (completed / failed / cancelled), falling back to the oldest run overall
/// when every run is still live.
///
/// Cheaply clonable through an inner [`Arc`]; clones share the same state
/// *and* the same cap.
#[derive(Clone, Debug, Default)]
pub struct InMemoryGraphStatusStore {
    /// Shared map + indexes, kept coherent under one lock.
    pub(crate) state: Arc<Mutex<StatusStoreState>>,
    /// Maximum retained runs; `None` means unbounded (default).
    pub(crate) max_runs: Option<usize>,
}

/// Internal state of [`InMemoryGraphStatusStore`]: the primary map plus the
/// indexes that keep `list_by_thread` fast and eviction ordered.
#[derive(Debug, Default)]
pub(crate) struct StatusStoreState {
    /// `run_id → latest status`.
    pub(crate) statuses: HashMap<String, GraphRunStatus>,
    /// `thread_id → run ids` secondary index for `list_by_thread`.
    pub(crate) by_thread: HashMap<String, Vec<String>>,
    /// Run ids in first-insertion order, driving oldest-first eviction.
    pub(crate) order: std::collections::VecDeque<String>,
}

// ---------------------------------------------------------------------------
// JournalGraphSink
// ---------------------------------------------------------------------------

/// A [`GraphEventSink`] that wraps each emitted [`GraphEvent`] into a durable
/// [`GraphObservation`] and appends it to a [`GraphEventJournal`].
///
/// The sink is configured with the emitting run's lineage and checkpoint
/// coordinates; each received event is stamped with a monotonically increasing
/// `offset`, the latest observed `step`, and the configured `namespace`. The
/// observation is then handed to a background [`AppendWorker`] that persists it
/// off the executor thread (best-effort — a full bounded queue drops rather
/// than stalling the run, and append errors are reported, not propagated).
/// [`crate::graph::stream::GraphEventSink::flush`] blocks until the durable log
/// has caught up; the executor calls it after the terminal run event.
///
/// An optional `inner` sink lets the journal sink also forward each event to a
/// live transport (for example a [`crate::graph::stream::CollectingSink`]) so a
/// single configured sink can both persist and broadcast.
#[derive(Clone)]
pub struct JournalGraphSink {
    /// Background drain that persists observations without blocking the run.
    pub(crate) worker: Arc<AppendWorker<GraphObservation>>,
    /// Optional downstream sink that also receives every event.
    pub(crate) inner: Option<Arc<dyn GraphEventSink>>,
    /// The run that owns events delivered to this sink.
    pub(crate) run_id: RunId,
    /// Root run id stamped onto every observation.
    pub(crate) root_run_id: RunId,
    /// Parent run id stamped onto every observation.
    pub(crate) parent_run_id: Option<RunId>,
    /// Thread id stamped onto every observation.
    pub(crate) thread_id: Option<ThreadId>,
    /// The graph this run belongs to.
    pub(crate) graph_id: GraphId,
    /// The checkpoint namespace stamped onto every observation.
    pub(crate) namespace: Vec<String>,
    /// Monotonic per-run observation offset counter.
    pub(crate) offset: Arc<AtomicU64>,
    /// The latest observed superstep, used to stamp events with no step.
    pub(crate) step: Arc<AtomicU64>,
}
