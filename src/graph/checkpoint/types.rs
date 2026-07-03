//! Checkpoint records and metadata — the persisted snapshots that make every
//! level of a recursive graph run resumable and forkable.
//!
//! Checkpoints are graph-runtime persistence, separate from harness memory and
//! long-term stores. They are written at superstep boundaries only — never
//! mid-node — because rerunning a node from its start is far easier to reason
//! about than suspending an async Rust stack, and it matches interrupt/resume
//! semantics exactly.
//!
//! Each record carries a `thread_id` lineage key, a `parent_checkpoint_id`
//! chain (the spine that time-travel and forking walk), and a `namespace` that
//! scopes nested subgraph checkpoints so a parent run and the child graphs it
//! embeds never overwrite each other.

use std::fmt;

use crate::graph::command::Interrupt;
use crate::harness::ids::NodeId;

/// Why a checkpoint was written.
///
/// Mirrors the documented metadata `source` taxonomy: a checkpoint is produced
/// by the initial graph `input`, a normal superstep `loop` boundary, a manual
/// `update` (a state write attributed through the reducers), or a `fork` that
/// branches a thread for time-travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointSource {
    /// The initial state supplied when a run starts.
    Input,
    /// A normal superstep boundary in the execution loop.
    Loop,
    /// A manual state update written through the channel reducers.
    Update,
    /// A fork that branches a thread for time-travel/replay.
    Fork,
}

impl CheckpointSource {
    /// The lowercase wire/string form used in checkpoint metadata.
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckpointSource::Input => "input",
            CheckpointSource::Loop => "loop",
            CheckpointSource::Update => "update",
            CheckpointSource::Fork => "fork",
        }
    }

    /// Parses a source string, returning `None` for unknown values.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "input" => Some(CheckpointSource::Input),
            "loop" => Some(CheckpointSource::Loop),
            "update" => Some(CheckpointSource::Update),
            "fork" => Some(CheckpointSource::Fork),
            _ => None,
        }
    }
}

impl fmt::Display for CheckpointSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// When committed checkpoints are persisted relative to graph execution.
///
/// The default is [`DurabilityMode::Sync`], which preserves today's behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DurabilityMode {
    /// Persist a checkpoint before the next step starts. The boundary state is
    /// durable before any successor node runs — the strongest guarantee.
    #[default]
    Sync,
    /// Persist after the step's state is committed, conceptually while the next
    /// step executes. Currently treated like [`DurabilityMode::Sync`]: the
    /// checkpoint is written at the boundary, but the mode documents the intent
    /// to move persistence off the critical path.
    Async,
    /// Persist only the final checkpoint when the graph exits (or pauses on an
    /// interrupt). Intermediate boundaries are not written, trading
    /// resumability granularity for fewer writes.
    Exit,
}

/// Coordinates that address a checkpoint within a thread.
///
/// `checkpoint_id` of `None` selects the latest checkpoint for the thread;
/// `namespace` scopes nested subgraph checkpoints so a parent run and its
/// embedded child graphs never collide.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CheckpointConfig {
    /// Thread lineage key.
    pub thread_id: String,
    /// Specific checkpoint to address, or `None` for the latest.
    pub checkpoint_id: Option<String>,
    /// Namespace scoping for nested subgraph checkpoints.
    pub namespace: Vec<String>,
}

impl CheckpointConfig {
    /// Builds a config addressing the latest checkpoint of `thread_id` at the
    /// root namespace.
    pub fn latest(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            checkpoint_id: None,
            namespace: Vec::new(),
        }
    }
}

/// The documented core persistence unit: a checkpoint together with its config,
/// the config of its parent, and the per-task pending writes preserved with it.
///
/// Backends compose this from `get` + `list` via
/// [`Checkpointer::get_tuple`](crate::graph::Checkpointer::get_tuple).
#[derive(Clone, Debug)]
pub struct CheckpointTuple<State> {
    /// Config that addresses this checkpoint.
    pub config: CheckpointConfig,
    /// The checkpoint record itself.
    pub checkpoint: Checkpoint<State>,
    /// Config addressing the parent checkpoint, when one exists.
    pub parent_config: Option<CheckpointConfig>,
    /// Pending writes carried by the checkpoint.
    pub pending_writes: Vec<PendingWrite>,
}

/// A persisted snapshot of a graph run at a superstep boundary.
///
/// Derives `Serialize`/`Deserialize` with serde's conditional bounds: a
/// `Checkpoint<State>` is (de)serializable exactly when `State` is, which is
/// what lets file-backed backends such as
/// [`FileCheckpointer`](crate::graph::FileCheckpointer) round-trip whole records
/// through JSON. The in-memory path never needs it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint<State> {
    /// Checkpoint lineage key for a conversation/workflow/tenant run series.
    pub thread_id: String,
    /// This checkpoint's id within the thread.
    pub checkpoint_id: String,
    /// The run that produced this checkpoint, when known.
    ///
    /// Optional and back-compatible: pre-existing records and manual snapshots
    /// may leave it `None`. The executor stamps it so checkpoints can be deleted
    /// by run id via [`Checkpointer::delete_by_run`](crate::graph::Checkpointer::delete_by_run).
    pub run_id: Option<String>,
    /// The previous checkpoint id in the thread lineage.
    pub parent_checkpoint_id: Option<String>,
    /// Namespace scoping for nested subgraph checkpoints.
    pub namespace: Vec<String>,
    /// Committed graph state at this boundary.
    pub state: State,
    /// Nodes that should run when resuming from this checkpoint.
    pub next_nodes: Vec<NodeId>,
    /// Nodes that completed in the step that produced this checkpoint.
    pub completed_tasks: Vec<NodeId>,
    /// Per-task partial writes preserved when a step partially completes.
    pub pending_writes: Vec<PendingWrite>,
    /// Interrupts that paused the run at this boundary.
    pub interrupts: Vec<Interrupt>,
    /// Pending activations to schedule on resume, preserving each pending
    /// node's per-invocation [`Send`](crate::graph::Send) argument.
    ///
    /// A richer superset of [`next_nodes`](Self::next_nodes) (which stays the
    /// node-id projection used for listing and status). `#[serde(default)]`
    /// keeps checkpoints written before this field loadable: they deserialize
    /// to `None`, and resume falls back to `next_nodes` (node-only, no send
    /// arg) — exactly the pre-field behavior.
    #[serde(default)]
    pub pending_activations: Option<Vec<PendingActivation>>,
    /// Barrier (waiting-edge) arrivals accumulated across supersteps, persisted
    /// so a join node's precondition survives an interrupt/failure + resume.
    ///
    /// `#[serde(default)]` for back-compat: older checkpoints load with an
    /// empty set (the pre-field behavior, where arrivals were run-local).
    #[serde(default)]
    pub barrier_arrivals: Vec<BarrierArrivals>,
    /// Free-form metadata (source, step, etc.).
    pub metadata: serde_json::Value,
}

/// One pending node activation persisted in a checkpoint: the node to run on
/// resume plus the optional per-invocation [`Send`](crate::graph::Send)
/// argument that scheduled it.
///
/// The durable counterpart of the executor's in-flight activation. Persisting
/// the `send_arg` is what lets a map-reduce fanout survive an interrupt/failure
/// boundary — without it every pending worker re-runs with no argument.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PendingActivation {
    /// The node scheduled to run on resume.
    pub node: NodeId,
    /// The per-invocation `Send` argument, when the activation was a `Send`
    /// packet (plain edge/goto activations carry `None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub send_arg: Option<serde_json::Value>,
}

/// The persisted arrivals recorded against one barrier (waiting-edge) join node:
/// the predecessors that have already routed to it but whose join has not yet
/// fired.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BarrierArrivals {
    /// The waiting/join node.
    pub node: NodeId,
    /// The predecessor nodes that have arrived so far.
    pub arrived: Vec<NodeId>,
}

impl<State> Checkpoint<State> {
    /// Builds the lightweight [`CheckpointMetadata`] summary for this checkpoint.
    ///
    /// The single source of truth for projecting a stored checkpoint onto its
    /// listing record: it parses the `source`/`step` out of the free-form
    /// `metadata` (falling back to [`CheckpointSource::Loop`]/`0`) and copies the
    /// lineage fields. Both `Checkpointer::list` and the state-inspection API
    /// (`get_state`/`get_state_history`) use it so a snapshot's metadata always
    /// matches what listing reports.
    pub fn to_metadata(&self) -> CheckpointMetadata {
        let source = self
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .and_then(CheckpointSource::parse)
            .unwrap_or(CheckpointSource::Loop);
        let step = self
            .metadata
            .get("step")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        CheckpointMetadata {
            thread_id: self.thread_id.clone(),
            checkpoint_id: self.checkpoint_id.clone(),
            run_id: self.run_id.clone(),
            parent_checkpoint_id: self.parent_checkpoint_id.clone(),
            namespace: self.namespace.clone(),
            next_nodes: self.next_nodes.clone(),
            has_interrupts: !self.interrupts.is_empty(),
            source,
            step,
        }
    }
}

/// A partial write produced by a completed task, preserved across reruns.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PendingWrite {
    /// The node that produced the write.
    pub node: NodeId,
    /// The serialized write payload.
    pub payload: serde_json::Value,
}

/// Lightweight checkpoint summary returned by `Checkpointer::list`.
///
/// Listing must not require deserializing full graph state, so metadata is kept
/// separate from the [`Checkpoint`] state payload.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Thread lineage key.
    pub thread_id: String,
    /// Checkpoint id.
    pub checkpoint_id: String,
    /// The run that produced this checkpoint, when known.
    pub run_id: Option<String>,
    /// Parent checkpoint id.
    pub parent_checkpoint_id: Option<String>,
    /// Namespace scoping.
    pub namespace: Vec<String>,
    /// Nodes to run on resume.
    pub next_nodes: Vec<NodeId>,
    /// Whether the checkpoint carries pending interrupts.
    pub has_interrupts: bool,
    /// Checkpoint source: `input`, `loop`, `update`, or `fork`.
    pub source: CheckpointSource,
    /// The superstep number that produced the checkpoint.
    pub step: usize,
}
