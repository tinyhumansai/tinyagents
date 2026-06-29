//! Serializable graph-topology types for export and visualization.
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
    /// Nodes that route directly to the virtual `END` node, sorted.
    pub finish_nodes: Vec<String>,
    /// State channel / reducer bindings, when available (populated from a
    /// `.rag` blueprint; empty for compiled whole-state graphs).
    pub channels: Vec<ChannelInfo>,
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
