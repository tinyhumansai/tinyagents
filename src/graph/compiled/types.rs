//! Compiled graph and execution-result types.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::graph::builder::{Branch, BuilderNode};
use crate::graph::checkpoint::{
    CheckpointConfig, CheckpointMetadata, Checkpointer, DurabilityMode,
};
use crate::graph::command::Interrupt;
use crate::graph::observability::{GraphEventJournal, GraphStatusStore};
use crate::graph::reducer::StateReducer;
use crate::graph::status::GraphRunStatus;
use crate::graph::stream::GraphEventSink;
use crate::harness::ids::{CheckpointId, GraphId, NodeId};

/// An immutable, validated graph ready to execute.
///
/// Cheap to clone (all heavy fields are `Arc`-shared) and safe to run
/// concurrently. Construct one with [`crate::graph::GraphBuilder::compile`].
pub struct CompiledGraph<State, Update> {
    pub(crate) graph_id: GraphId,
    pub(crate) nodes: Arc<HashMap<NodeId, BuilderNode<State, Update>>>,
    pub(crate) edges: Arc<HashMap<NodeId, NodeId>>,
    pub(crate) branches: Arc<HashMap<NodeId, Branch<State>>>,
    #[allow(dead_code)]
    pub(crate) command_nodes: Arc<HashSet<NodeId>>,
    /// Barrier/waiting edges: target -> the predecessor set that must all
    /// complete (across steps) before the target activates.
    pub(crate) waiting: Arc<HashMap<NodeId, HashSet<NodeId>>>,
    pub(crate) entry: NodeId,
    pub(crate) reducer: Arc<dyn StateReducer<State, Update>>,
    pub(crate) recursion_limit: usize,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer<State>>>,
    pub(crate) event_sink: Option<Arc<dyn GraphEventSink>>,
    /// Optional durable observation journal (opt-in via
    /// [`CompiledGraph::with_event_journal`]).
    pub(crate) journal: Option<Arc<dyn GraphEventJournal>>,
    /// Optional run-status surface (opt-in via
    /// [`CompiledGraph::with_status_store`]).
    pub(crate) status_store: Option<Arc<dyn GraphStatusStore>>,
    pub(crate) namespace: Vec<String>,
    /// When true, the active node set of a superstep is executed concurrently.
    pub(crate) parallel: bool,
    /// Upper bound on concurrently-running branches per step (`None` = unbounded).
    pub(crate) max_concurrency: Option<usize>,
    /// Default per-node handler timeout (`None` = no timeout).
    pub(crate) node_timeout: Option<std::time::Duration>,
    /// When checkpoints are persisted relative to execution (default
    /// [`DurabilityMode::Sync`]).
    pub(crate) durability: DurabilityMode,
}

impl<State, Update> std::fmt::Debug for CompiledGraph<State, Update> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("graph_id", &self.graph_id)
            .field("nodes", &self.nodes.len())
            .field("entry", &self.entry)
            .field("recursion_limit", &self.recursion_limit)
            .field("namespace", &self.namespace)
            .field("parallel", &self.parallel)
            .finish_non_exhaustive()
    }
}

impl<State, Update> Clone for CompiledGraph<State, Update> {
    fn clone(&self) -> Self {
        Self {
            graph_id: self.graph_id.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            branches: self.branches.clone(),
            command_nodes: self.command_nodes.clone(),
            waiting: self.waiting.clone(),
            entry: self.entry.clone(),
            reducer: self.reducer.clone(),
            recursion_limit: self.recursion_limit,
            checkpointer: self.checkpointer.clone(),
            event_sink: self.event_sink.clone(),
            journal: self.journal.clone(),
            status_store: self.status_store.clone(),
            namespace: self.namespace.clone(),
            parallel: self.parallel,
            max_concurrency: self.max_concurrency,
            node_timeout: self.node_timeout,
            durability: self.durability,
        }
    }
}

/// The result of a durable graph run.
///
/// Carries the final committed state, the visited node history, the superstep
/// count, any pending interrupts, the run status snapshot, and the latest
/// checkpoint id (when checkpointing was enabled).
#[derive(Clone, Debug)]
pub struct GraphExecution<State> {
    /// Final committed state.
    pub state: State,
    /// Ordered list of executed nodes (may repeat across supersteps).
    pub visited: Vec<NodeId>,
    /// Number of supersteps executed.
    pub steps: usize,
    /// Interrupts that paused the run, if any.
    pub interrupts: Vec<Interrupt>,
    /// Compact status snapshot at the final boundary.
    pub status: GraphRunStatus,
    /// The latest persisted checkpoint id, if checkpointing was enabled.
    pub checkpoint_id: Option<CheckpointId>,
}

impl<State> GraphExecution<State> {
    /// Returns true when the run paused on an interrupt rather than completing.
    pub fn is_interrupted(&self) -> bool {
        !self.interrupts.is_empty()
    }
}

/// A point-in-time view of a thread's checkpointed state, returned by
/// [`CompiledGraph::get_state`](crate::graph::CompiledGraph::get_state) and
/// [`CompiledGraph::get_state_history`](crate::graph::CompiledGraph::get_state_history).
///
/// This is the state-inspection / time-travel surface: it bundles the committed
/// channel values at a checkpoint with everything a caller needs to reason about
/// or resume from that point — the next nodes that would run, the config that
/// addresses this snapshot, the config of its parent (for walking the lineage),
/// the checkpoint metadata (source/step/run id), and any pending interrupts.
#[derive(Clone, Debug)]
pub struct StateSnapshot<State> {
    /// Committed graph state (channel values) at the snapshot's checkpoint.
    pub values: State,
    /// Nodes that would run if execution resumed from this snapshot.
    pub next_nodes: Vec<NodeId>,
    /// Tasks scheduled for the next superstep (mirrors `next_nodes`).
    pub tasks: Vec<NodeId>,
    /// Config that addresses this snapshot's checkpoint.
    pub config: CheckpointConfig,
    /// Lightweight metadata for the snapshot's checkpoint (source, step, run id).
    pub metadata: CheckpointMetadata,
    /// Config addressing the parent checkpoint, when one exists.
    pub parent_config: Option<CheckpointConfig>,
    /// Interrupts that paused the run at this checkpoint, if any.
    pub pending_interrupts: Vec<Interrupt>,
}

/// Selects which checkpoint a time-travel resume starts from.
///
/// [`CompiledGraph::resume`](crate::graph::CompiledGraph::resume) is shorthand
/// for [`ResumeTarget::Latest`]; [`CompiledGraph::resume_from`](crate::graph::CompiledGraph::resume_from)
/// accepts an explicit target so a run can be replayed from an older checkpoint
/// config (time travel) without mutating the original record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResumeTarget {
    /// Resume from the thread's most recent checkpoint.
    Latest,
    /// Resume from a specific checkpoint id within the thread.
    Checkpoint(String),
}
