//! Builder types for the durable graph.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Result;
use crate::graph::command::NodeResult;
use crate::graph::reducer::StateReducer;
use crate::harness::ids::{GraphId, NodeId, RunId, ThreadId};

/// The reserved virtual entry node.
pub const START: &str = "__start__";
/// The reserved virtual terminal node.
pub const END: &str = "__end__";

/// Boxed future produced by a durable node handler.
pub type NodeFuture<Update> = Pin<Box<dyn Future<Output = Result<NodeResult<Update>>> + Send>>;

/// A durable node handler: receives a state snapshot and per-task context,
/// returns a [`NodeResult`].
pub type NodeHandler<State, Update> =
    dyn Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync;

/// A conditional routing function over committed state. Returns a route label
/// resolved against the node's route table at the step boundary.
pub type RouterFn<State> = dyn Fn(&State) -> String + Send + Sync;

/// Per-task runtime context passed to a durable node handler.
///
/// The context exposes run identity, the current step, and — crucially — an
/// optional `resume` value. On a normal run `resume` is `None`; when a run is
/// resumed after an interrupt, the interrupted node is re-run with `resume` set
/// to the value carried by the resume command.
#[derive(Clone, Debug)]
pub struct NodeContext {
    /// The node being executed.
    pub node_id: NodeId,
    /// The current run id.
    pub run_id: RunId,
    /// The thread id when checkpointing is enabled.
    pub thread_id: Option<ThreadId>,
    /// The 1-based superstep number.
    pub step: usize,
    /// Resume value supplied by `CompiledGraph::resume`, if any.
    pub resume: Option<serde_json::Value>,
}

/// A compiled-in node: id plus its handler.
pub(crate) struct BuilderNode<State, Update> {
    #[allow(dead_code)]
    pub(crate) id: NodeId,
    pub(crate) handler: Arc<NodeHandler<State, Update>>,
}

impl<State, Update> Clone for BuilderNode<State, Update> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            handler: self.handler.clone(),
        }
    }
}

/// Conditional routing for a node: a router function plus its route table.
pub(crate) struct Branch<State> {
    pub(crate) router: Arc<RouterFn<State>>,
    pub(crate) routes: HashMap<String, NodeId>,
}

impl<State> Clone for Branch<State> {
    fn clone(&self) -> Self {
        Self {
            router: self.router.clone(),
            routes: self.routes.clone(),
        }
    }
}

/// A mutable, ergonomic builder for a durable state graph.
///
/// `State` is the committed graph state; `Update` is the partial-update type
/// merged through the configured [`StateReducer`]. For whole-state graphs use
/// `Update == State` together with the overwrite reducer (see
/// [`super::GraphBuilder::overwrite`]).
pub struct GraphBuilder<State, Update> {
    pub(crate) graph_id: GraphId,
    pub(crate) nodes: HashMap<NodeId, BuilderNode<State, Update>>,
    pub(crate) edges: HashMap<NodeId, NodeId>,
    pub(crate) branches: HashMap<NodeId, Branch<State>>,
    pub(crate) command_nodes: HashSet<NodeId>,
    pub(crate) reducer: Option<Arc<dyn StateReducer<State, Update>>>,
    pub(crate) recursion_limit: usize,
}
