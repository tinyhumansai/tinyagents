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
use crate::harness::ids::{CheckpointId, EventId, GraphId, RunId, ThreadId};
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

/// In-memory [`GraphStatusStore`] backed by a `run_id → status` map.
///
/// Cheaply clonable through an inner [`Arc`]; clones share the same map.
#[derive(Clone, Debug, Default)]
pub struct InMemoryGraphStatusStore {
    /// `run_id → latest status`.
    pub(crate) statuses: Arc<Mutex<HashMap<String, GraphRunStatus>>>,
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
/// async append is bridged synchronously with `futures::executor::block_on`,
/// and append errors are swallowed so a failing journal never aborts the run.
///
/// An optional `inner` sink lets the journal sink also forward each event to a
/// live transport (for example a [`crate::graph::stream::CollectingSink`]) so a
/// single configured sink can both persist and broadcast.
#[derive(Clone)]
pub struct JournalGraphSink {
    /// The journal observations are appended to.
    pub(crate) journal: Arc<dyn GraphEventJournal>,
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
