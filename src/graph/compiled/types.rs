//! Compiled graph and execution-result types.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::graph::builder::{Branch, BuilderNode};
use crate::graph::checkpoint::Checkpointer;
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
            entry: self.entry.clone(),
            reducer: self.reducer.clone(),
            recursion_limit: self.recursion_limit,
            checkpointer: self.checkpointer.clone(),
            event_sink: self.event_sink.clone(),
            journal: self.journal.clone(),
            status_store: self.status_store.clone(),
            namespace: self.namespace.clone(),
            parallel: self.parallel,
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
