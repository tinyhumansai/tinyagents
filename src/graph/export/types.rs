//! Serializable graph-topology types for export and visualization — the single,
//! behavior-free description every graph in the recursive runtime collapses to.
//!
//! Hand-built graphs, `.rag` blueprints, and model-authored graphs all extract
//! to the same [`GraphTopology`] shape, so one type backs JSON export, Mermaid
//! rendering, and test snapshots regardless of how the graph came to exist.
//!
//! These types describe the *structure* of a graph — its nodes, edges, and
//! conditional routes — without referencing any runnable node behavior (handler
//! closures, router closures, reducers). That makes a [`GraphTopology`] cheap to
//! clone, `serde`-serializable, and safe to snapshot in tests or render as a
//! diagram. Extract one from a compiled or built graph with
//! [`crate::graph::CompiledGraph::topology`] /
//! [`crate::graph::GraphBuilder::topology`], or from a `.rag`
//! [`crate::language::Blueprint`] via
//! [`crate::graph::export::blueprint_to_topology`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A serializable, behavior-free description of a graph's structure.
///
/// The topology is the single source of truth shared by JSON export
/// ([`crate::graph::export::to_json`]) and Mermaid rendering
/// ([`crate::graph::export::to_mermaid`]). All collections are stored in a
/// stable, sorted order so exports are deterministic regardless of the
/// underlying `HashMap` iteration order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTopology {
    /// The graph identifier.
    pub graph_id: String,
    /// The optional human-readable graph name (descriptive; `graph_id` remains
    /// the stable identifier).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The entry node (the target of the virtual `START` node), if known.
    pub entry: Option<String>,
    /// Maximum number of supersteps the graph may execute (0 when unknown,
    /// e.g. for a blueprint without an explicit `recursion_limit`).
    pub recursion_limit: usize,
    /// Whether the active node set of a superstep runs concurrently.
    pub parallel: bool,
    /// All declared nodes, sorted by id.
    pub nodes: Vec<NodeInfo>,
    /// Direct (unconditional) edges between nodes, sorted by `(from, to)`.
    /// Excludes the synthetic `START`/`END` edges, which are represented by
    /// [`Self::entry`] and [`Self::finish_nodes`].
    pub edges: Vec<EdgeInfo>,
    /// Conditional (router-driven) edges, sorted by source node id.
    pub conditional_edges: Vec<ConditionalEdgeInfo>,
    /// Barrier / waiting (fan-in) edges, sorted by target node id. A target's
    /// predecessor set must *all* complete before it activates.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub waiting_edges: Vec<WaitingEdgeInfo>,
    /// Nodes that route directly to the virtual `END` node, sorted.
    pub finish_nodes: Vec<String>,
    /// State channel / reducer bindings, when available (populated from a
    /// `.rag` blueprint; empty for compiled whole-state graphs).
    pub channels: Vec<ChannelInfo>,
    /// Graph-level execution policy summary (recursion limit, concurrency,
    /// per-node timeout).
    #[serde(default)]
    pub policy: GraphPolicySummary,
    /// Structural validation report computed over this topology.
    #[serde(default)]
    pub validation: ValidationReport,
}

/// A single graph node's structural metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInfo {
    /// The node id.
    pub id: String,
    /// The node kind, when known (e.g. `model`, `tool` from a blueprint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// True when this node routes exclusively via a `Command` `goto` rather
    /// than static or conditional edges.
    #[serde(default, skip_serializing_if = "is_false")]
    pub command_routing: bool,
    /// True when this node embeds and runs a child graph (a subgraph node).
    #[serde(default, skip_serializing_if = "is_false")]
    pub subgraph: bool,
    /// True when this node is an interrupt point that can pause the run.
    #[serde(default, skip_serializing_if = "is_false")]
    pub interrupt: bool,
    /// True when this node is a deferred join (activates after the frontier
    /// drains).
    #[serde(default, skip_serializing_if = "is_false")]
    pub deferred: bool,
    /// Declared `goto` destination hints for a command-routing node, sorted.
    /// Advisory only — the runtime resolves the real target at runtime.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command_destinations: Vec<String>,
    /// Free-form, sorted key/value annotations carried from the builder.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// A compact, derived summary of this node's routing/behavior policy.
    #[serde(default)]
    pub policy: NodePolicySummary,
}

/// A barrier / waiting (fan-in) edge: `target` only activates once every node
/// in `predecessors` has completed, possibly across supersteps.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WaitingEdgeInfo {
    /// The join node that waits on its predecessors.
    pub target: String,
    /// The predecessor nodes that must all complete first, sorted.
    pub predecessors: Vec<String>,
}

/// Graph-level execution policy, summarized for export.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphPolicySummary {
    /// Maximum number of supersteps the graph may execute (0 when unknown).
    pub recursion_limit: usize,
    /// Whether the active node set of a superstep runs concurrently.
    pub parallel: bool,
    /// Upper bound on branches run concurrently per step (`None` = unbounded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrency: Option<usize>,
    /// Default per-node handler timeout in milliseconds (`None` = no timeout).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_timeout_ms: Option<u128>,
}

/// A derived, per-node summary of how a node routes and what role it plays.
///
/// Every field is computed from the topology itself, so it stays in sync with
/// the structural fields and is safe to snapshot in tests.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodePolicySummary {
    /// The node's routing discipline: one of `static`, `conditional`,
    /// `command`, `terminal`, or `unrouted`.
    pub routing: String,
    /// True when the node is a barrier/waiting (fan-in) join target.
    #[serde(default, skip_serializing_if = "is_false")]
    pub barrier: bool,
    /// True when the node is an interrupt point.
    #[serde(default, skip_serializing_if = "is_false")]
    pub interrupt: bool,
    /// True when the node is a deferred join.
    #[serde(default, skip_serializing_if = "is_false")]
    pub deferred: bool,
    /// True when the node embeds a subgraph.
    #[serde(default, skip_serializing_if = "is_false")]
    pub subgraph: bool,
}

/// A structural validation report computed over a [`GraphTopology`].
///
/// `errors` are structural defects that would make the graph invalid (a missing
/// entry, a dangling edge/route/barrier target). `warnings` are non-fatal
/// observations (unreachable nodes, dead-end nodes). A compiled graph is already
/// validated, so its report is typically clean; a builder-stage topology may
/// surface in-progress issues.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReport {
    /// True when there are no errors.
    pub ok: bool,
    /// Structural defects, sorted and deduplicated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    /// Non-fatal observations, sorted and deduplicated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// A direct, unconditional edge from one node to another.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeInfo {
    /// The source node id.
    pub from: String,
    /// The target node id.
    pub to: String,
}

/// Conditional routing out of a node: a set of labeled targets resolved at
/// runtime by the node's router.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionalEdgeInfo {
    /// The source node id.
    pub from: String,
    /// The labeled routes, sorted by label.
    pub routes: Vec<RouteInfo>,
}

/// One labeled branch of a conditional edge.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RouteInfo {
    /// The route label returned by the router.
    pub label: String,
    /// The target node id (may be the virtual `END`).
    pub target: String,
}

/// A state-channel-to-reducer binding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelInfo {
    /// The channel name.
    pub name: String,
    /// The reducer reference bound to the channel.
    pub reducer: String,
}

/// Serde helper: skip serializing `false` booleans.
fn is_false(value: &bool) -> bool {
    !*value
}
