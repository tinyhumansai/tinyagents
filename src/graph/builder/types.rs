//! Builder types for the durable graph.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

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

/// Identifies one branch of a concurrent (fan-out) superstep.
///
/// When a graph compiled with [`crate::graph::GraphBuilder::with_parallel`]
/// runs more than one active node in a single superstep, every branch executes
/// against its own cloned `State` snapshot and receives a distinct `ForkId` on
/// its [`NodeContext`]. The `branch_index` is the branch's position in the
/// deterministically-ordered active set, so a handler can tell which fork it is
/// (e.g. to seed per-fork randomness or pick a strategy) and the executor can
/// keep reducer application reproducible regardless of completion order.
///
/// In sequential mode (the default), and in a parallel step that happens to
/// have a single active node, `NodeContext::fork` is `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForkId {
    /// The branch's index in the superstep's active set (0-based, stable).
    pub branch_index: usize,
    /// The node executing on this branch.
    pub node: NodeId,
}

impl ForkId {
    /// Creates a fork id for `node` at `branch_index` within the active set.
    pub fn new(branch_index: usize, node: NodeId) -> Self {
        Self { branch_index, node }
    }
}

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
    /// The branch identity when this node runs as one fork of a concurrent
    /// (fan-out) superstep; `None` in sequential mode or single-node steps.
    pub fork: Option<ForkId>,
    /// The per-invocation argument when this activation was scheduled by a
    /// [`crate::graph::Send`] packet or seeded through
    /// [`crate::graph::GraphInput`]; `None` for normal edge/`goto` activations.
    /// This is how map-reduce / search-fanout branches and external graph
    /// inputs receive custom data that differs from the graph's shared
    /// committed state.
    pub send_arg: Option<serde_json::Value>,
    /// The root run id of the recursion tree this node executes within. For a
    /// top-level run this equals `run_id`; for a subgraph/sub-agent child run it
    /// is the shared ancestor, so a child a node spawns can preserve the root.
    pub root_run_id: Option<RunId>,
    /// The enclosing run's live recursion stack (root-first). A subgraph node
    /// seeds an embedded child graph with these frames so the child extends the
    /// parent's recursion tree instead of starting a fresh one.
    pub recursion_frames: Vec<crate::graph::recursion::RecursionFrame>,
    /// A per-run collector the executor provides so a subgraph node can report
    /// the [`ChildRun`](crate::graph::ChildRun) it spawned back to the enclosing
    /// run; `None` when no executor sink is attached (e.g. a hand-built context).
    pub child_runs: Option<crate::graph::recursion::ChildRunSink>,
}

/// Behavior-free, introspectable metadata attached to a node by the builder.
///
/// Markers and free-form metadata recorded here never affect execution; they
/// exist so the [topology export](crate::graph::export) can describe what a node
/// *is* (a subgraph embedding, an interrupt point, a deferred join, …) without
/// inspecting the node's opaque handler closure. All fields are optional and
/// additive: an unset [`NodeMeta`] (the [`Default`]) contributes nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct NodeMeta {
    /// A human-readable node kind (e.g. `model`, `tool`, `subgraph`).
    pub(crate) kind: Option<String>,
    /// The node pauses the run before/at execution (an interrupt point).
    pub(crate) interrupt: bool,
    /// The node is a deferred join — it activates only after the rest of the
    /// active frontier has drained (a barrier-style synthesis node).
    pub(crate) deferred: bool,
    /// The node embeds and runs a child graph (a subgraph node).
    pub(crate) subgraph: bool,
    /// Declared `goto` destination hints for a command-routing node, in the
    /// order they were registered. Purely advisory: the runtime resolves the
    /// real target from the emitted [`crate::graph::Command`] at runtime.
    pub(crate) command_destinations: Vec<NodeId>,
    /// Arbitrary, sorted key/value annotations carried into the export.
    pub(crate) metadata: BTreeMap<String, String>,
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

/// A small newtype wrapper for a conditional-route label.
///
/// Routers may return any `impl ToString` label (a plain `&str`/`String`, or a
/// user-defined route enum that implements `Display`). `Route` is an optional
/// ergonomic helper for building route tables and for routers that prefer to
/// return a typed value instead of a bare string; it stringifies via
/// [`ToString`] at the route-table boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Route(pub String);

impl Route {
    /// Wraps any `impl ToString` (e.g. a route enum with `Display`) as a label.
    pub fn new(label: impl ToString) -> Self {
        Self(label.to_string())
    }

    /// The label as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Route {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Tunable per-graph defaults applied to a [`GraphBuilder`] in one call via
/// [`GraphBuilder::set_defaults`].
///
/// Every field is optional; only the `Some` fields override the builder's
/// current configuration, so partial defaults compose with explicit
/// `with_*` calls. All fields are opt-in and additive — an unset
/// [`GraphDefaults`] (the [`Default`]) changes nothing.
#[derive(Clone, Debug, Default)]
pub struct GraphDefaults {
    /// Maximum number of supersteps before [`crate::TinyAgentsError::RecursionLimit`].
    pub recursion_limit: Option<usize>,
    /// Whether the active node set of a superstep runs concurrently.
    pub parallel: Option<bool>,
    /// Upper bound on the number of branches run concurrently within one step
    /// (only meaningful when `parallel` is enabled). `None` means unbounded.
    pub max_concurrency: Option<usize>,
    /// Default wall-clock timeout applied to every node handler; on elapse the
    /// run fails with [`crate::TinyAgentsError::Timeout`]. `None` means no
    /// per-node timeout.
    pub node_timeout: Option<Duration>,
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
    /// Optional human-readable graph name carried into the topology export
    /// (the `graph_id` remains the stable identifier).
    pub(crate) name: Option<String>,
    pub(crate) nodes: HashMap<NodeId, BuilderNode<State, Update>>,
    pub(crate) edges: HashMap<NodeId, NodeId>,
    pub(crate) branches: HashMap<NodeId, Branch<State>>,
    pub(crate) command_nodes: HashSet<NodeId>,
    /// Barrier/waiting edges: target node -> set of predecessor nodes that must
    /// all have completed (across steps) before the target activates.
    pub(crate) waiting: HashMap<NodeId, HashSet<NodeId>>,
    /// Mixed fan-in barrier relief registrations; see
    /// [`super::BarrierRelief`].
    pub(crate) barrier_reliefs: Vec<super::BarrierRelief>,
    pub(crate) reducer: Option<Arc<dyn StateReducer<State, Update>>>,
    pub(crate) recursion_limit: usize,
    /// When true, active nodes in a superstep run concurrently (fan-out).
    pub(crate) parallel: bool,
    /// Upper bound on concurrently-running branches per step (`None` = unbounded).
    pub(crate) max_concurrency: Option<usize>,
    /// Default per-node handler timeout (`None` = no timeout).
    pub(crate) node_timeout: Option<Duration>,
    /// Behavior-free per-node markers/metadata surfaced by the topology export.
    pub(crate) node_meta: HashMap<NodeId, NodeMeta>,
}
