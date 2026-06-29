//! Compiled graph and execution-result types.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::graph::builder::{Branch, BuilderNode, NodeMeta};
use crate::graph::checkpoint::{
    CheckpointConfig, CheckpointMetadata, Checkpointer, DurabilityMode,
};
use crate::graph::command::Interrupt;
use crate::graph::observability::{GraphEventJournal, GraphStatusStore};
use crate::graph::recursion::{ChildRun, RunTree};
use crate::graph::reducer::StateReducer;
use crate::graph::status::GraphRunStatus;
use crate::graph::stream::GraphEventSink;
use crate::harness::ids::{CheckpointId, GraphId, NodeId, RunId};

/// An immutable, validated graph ready to execute.
///
/// Cheap to clone (all heavy fields are `Arc`-shared) and safe to run
/// concurrently. Construct one with [`crate::graph::GraphBuilder::compile`].
pub struct CompiledGraph<State, Update> {
    pub(crate) graph_id: GraphId,
    /// Optional human-readable graph name surfaced by the topology export.
    pub(crate) name: Option<String>,
    pub(crate) nodes: Arc<HashMap<NodeId, BuilderNode<State, Update>>>,
    pub(crate) edges: Arc<HashMap<NodeId, NodeId>>,
    pub(crate) branches: Arc<HashMap<NodeId, Branch<State>>>,
    #[allow(dead_code)]
    pub(crate) command_nodes: Arc<HashSet<NodeId>>,
    /// Barrier/waiting edges: target -> the predecessor set that must all
    /// complete (across steps) before the target activates.
    pub(crate) waiting: Arc<HashMap<NodeId, HashSet<NodeId>>>,
    /// Behavior-free per-node markers/metadata surfaced by the topology export.
    pub(crate) node_meta: Arc<HashMap<NodeId, NodeMeta>>,
    pub(crate) entry: NodeId,
    pub(crate) reducer: Arc<dyn StateReducer<State, Update>>,
    pub(crate) recursion_limit: usize,
    /// Explicit recursion caps (run-tree depth, per-node visits, total steps)
    /// enforced by the executor; configured via
    /// [`CompiledGraph::with_recursion_policy`](crate::graph::CompiledGraph::with_recursion_policy).
    pub(crate) recursion_policy: crate::graph::recursion::RecursionPolicy,
    /// Inherited recursion frames of an enclosing run (empty for a top-level
    /// graph). A subgraph/sub-agent wrapper seeds these so a nested run extends
    /// the parent's recursion stack rather than starting a fresh tree.
    pub(crate) recursion_frames: Vec<crate::graph::recursion::RecursionFrame>,
    /// The hosting node this graph runs under when embedded as a subgraph; used
    /// as the `node_id` of this run's root recursion frame so the run tree names
    /// the embedding node. `None` for a top-level run.
    pub(crate) recursion_node: Option<NodeId>,
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
            name: self.name.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            branches: self.branches.clone(),
            command_nodes: self.command_nodes.clone(),
            waiting: self.waiting.clone(),
            node_meta: self.node_meta.clone(),
            entry: self.entry.clone(),
            reducer: self.reducer.clone(),
            recursion_limit: self.recursion_limit,
            recursion_policy: self.recursion_policy,
            recursion_frames: self.recursion_frames.clone(),
            recursion_node: self.recursion_node.clone(),
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
    /// This run's own id.
    pub run_id: RunId,
    /// The graph that produced this run.
    pub graph_id: GraphId,
    /// The root run id of the recursion tree (equals `run_id` for a top-level
    /// run; the shared ancestor for a subgraph/sub-agent child run).
    pub root_run_id: RunId,
    /// The enclosing run's id, when this run was spawned as a child.
    pub parent_run_id: Option<RunId>,
    /// Child runs spawned from subgraph nodes during this run, in completion
    /// order, keyed by the embedding node.
    pub child_runs: Vec<ChildRun>,
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

    /// Builds the parent/child run-lineage view for this run.
    ///
    /// This is the run-id counterpart to the live recursion stack: it reports
    /// this run's id, the shared root, the enclosing parent (when this was a
    /// child run), and every child run spawned from a subgraph node.
    pub fn run_tree(&self) -> RunTree {
        RunTree {
            run_id: self.run_id.clone(),
            root_run_id: self.root_run_id.clone(),
            parent_run_id: self.parent_run_id.clone(),
            children: self.child_runs.clone(),
        }
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
