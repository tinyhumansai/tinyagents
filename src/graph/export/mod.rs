//! Graph export and visualization.
//!
//! This module turns a graph's structure into portable artifacts: a
//! `serde`-serializable [`GraphTopology`], a pretty JSON document, and a
//! deterministic [Mermaid](https://mermaid.js.org/) `flowchart`. It implements
//! the spec's "graph serialization to JSON" and "Mermaid export" future
//! features (see `docs/modules/graph/visualization-testkit.md`).
//!
//! Topology can be extracted from three sources, all yielding the same
//! [`GraphTopology`] shape so visualization and test snapshots share one truth:
//!
//! - [`crate::graph::CompiledGraph::topology`] — a validated, frozen graph.
//! - [`crate::graph::GraphBuilder::topology`] — a graph still under
//!   construction (entry may be unresolved).
//! - [`blueprint_to_topology`] — a `.rag` [`crate::language::Blueprint`].
//!
//! None of these expose runnable behavior (handler/router closures, reducers);
//! only structure is captured.
//!
//! ```
//! use rustagents::graph::{GraphBuilder, NodeResult, START, END};
//! use rustagents::graph::export::{to_json, to_mermaid};
//!
//! let graph = GraphBuilder::<i64, i64>::overwrite()
//!     .add_node("a", |s, _| async move { Ok(NodeResult::Update(s + 1)) })
//!     .add_node("b", |s, _| async move { Ok(NodeResult::Update(s + 1)) })
//!     .add_edge(START, "a")
//!     .add_edge("a", "b")
//!     .add_edge("b", END)
//!     .compile()
//!     .unwrap();
//!
//! let topology = graph.topology();
//! let json = to_json(&topology);
//! let mermaid = to_mermaid(&topology);
//! assert!(mermaid.contains("flowchart TD"));
//! ```

mod types;

pub use types::{ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphTopology, NodeInfo, RouteInfo};

use crate::Result;
use crate::graph::builder::{END, GraphBuilder, START};
use crate::graph::compiled::CompiledGraph;
use crate::language::{Blueprint, Routing};

/// Inputs to [`build_topology`]: a behavior-free view of one graph's structure.
struct TopologyParts {
    graph_id: String,
    recursion_limit: usize,
    parallel: bool,
    /// `(id, kind, command_routing)` for every declared node.
    nodes: Vec<(String, Option<String>, bool)>,
    /// Raw edges, including the synthetic `START`/`END` edges.
    edges: Vec<(String, String)>,
    /// `(from, [(label, target)])` conditional routes.
    conditional: Vec<(String, Vec<(String, String)>)>,
    /// `(channel, reducer)` bindings.
    channels: Vec<(String, String)>,
}

/// Folds raw structural parts into a normalized, deterministically-ordered
/// [`GraphTopology`]. Synthetic `START`/`END` edges are lifted into
/// [`GraphTopology::entry`] and [`GraphTopology::finish_nodes`].
fn build_topology(parts: TopologyParts) -> GraphTopology {
    let TopologyParts {
        graph_id,
        recursion_limit,
        parallel,
        nodes,
        edges,
        conditional,
        channels,
    } = parts;

    let mut entry: Option<String> = None;
    let mut direct: Vec<EdgeInfo> = Vec::new();
    let mut finish_nodes: Vec<String> = Vec::new();

    for (from, to) in edges {
        if from == START {
            entry = Some(to);
        } else if to == END {
            finish_nodes.push(from);
        } else {
            direct.push(EdgeInfo { from, to });
        }
    }

    let mut node_infos: Vec<NodeInfo> = nodes
        .into_iter()
        .map(|(id, kind, command_routing)| NodeInfo {
            id,
            kind,
            command_routing,
        })
        .collect();

    let mut conditional_edges: Vec<ConditionalEdgeInfo> = conditional
        .into_iter()
        .map(|(from, routes)| {
            let mut routes: Vec<RouteInfo> = routes
                .into_iter()
                .map(|(label, target)| RouteInfo { label, target })
                .collect();
            routes.sort();
            ConditionalEdgeInfo { from, routes }
        })
        .collect();

    let channels: Vec<ChannelInfo> = channels
        .into_iter()
        .map(|(name, reducer)| ChannelInfo { name, reducer })
        .collect();

    // Stable ordering so exports are deterministic.
    node_infos.sort_by(|a, b| a.id.cmp(&b.id));
    direct.sort();
    conditional_edges.sort_by(|a, b| a.from.cmp(&b.from));
    finish_nodes.sort();
    finish_nodes.dedup();

    GraphTopology {
        graph_id,
        entry,
        recursion_limit,
        parallel,
        nodes: node_infos,
        edges: direct,
        conditional_edges,
        finish_nodes,
        channels,
    }
}

impl<State, Update> CompiledGraph<State, Update> {
    /// Extracts a behavior-free [`GraphTopology`] describing this graph's
    /// structure (id, nodes, direct edges, conditional routes, entry, finish
    /// nodes). Node handler and router closures are never exposed.
    pub fn topology(&self) -> GraphTopology {
        let nodes = self
            .nodes
            .keys()
            .map(|id| (id.to_string(), None, self.command_nodes.contains(id)))
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|(from, to)| (from.to_string(), to.to_string()))
            .collect();
        let conditional = self
            .branches
            .iter()
            .map(|(from, branch)| {
                let routes = branch
                    .routes
                    .iter()
                    .map(|(label, target)| (label.clone(), target.to_string()))
                    .collect();
                (from.to_string(), routes)
            })
            .collect();
        build_topology(TopologyParts {
            graph_id: self.graph_id().to_string(),
            recursion_limit: self.recursion_limit,
            parallel: self.parallel,
            nodes,
            edges,
            conditional,
            channels: Vec::new(),
        })
    }
}

impl<State, Update> GraphBuilder<State, Update> {
    /// Extracts a [`GraphTopology`] from an in-progress builder.
    ///
    /// Unlike [`CompiledGraph::topology`], the builder has not been validated:
    /// the entry may be unresolved (`None`) if `START` has no edge yet, and
    /// dangling targets are reported as-is.
    pub fn topology(&self) -> GraphTopology {
        let nodes = self
            .nodes
            .keys()
            .map(|id| (id.to_string(), None, self.command_nodes.contains(id)))
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|(from, to)| (from.to_string(), to.to_string()))
            .collect();
        let conditional = self
            .branches
            .iter()
            .map(|(from, branch)| {
                let routes = branch
                    .routes
                    .iter()
                    .map(|(label, target)| (label.clone(), target.to_string()))
                    .collect();
                (from.to_string(), routes)
            })
            .collect();
        build_topology(TopologyParts {
            graph_id: self.graph_id.to_string(),
            recursion_limit: self.recursion_limit,
            parallel: self.parallel,
            nodes,
            edges,
            conditional,
            channels: Vec::new(),
        })
    }
}

/// Builds a [`GraphTopology`] from a `.rag` [`Blueprint`].
///
/// The blueprint already describes topology declaratively, so this is a direct
/// structural mapping: [`Routing::Next`] becomes a direct edge,
/// [`Routing::Conditional`] becomes a conditional edge, and
/// [`Routing::Terminal`] marks a finish node. State channels and their reducer
/// names are carried over. `recursion_limit` is read from the blueprint
/// `defaults` when present (0 otherwise).
pub fn blueprint_to_topology(blueprint: &Blueprint) -> GraphTopology {
    let recursion_limit = blueprint
        .defaults
        .iter()
        .find(|(key, _)| key == "recursion_limit")
        .and_then(|(_, value)| match value {
            crate::language::Literal::Num(n) if *n >= 0.0 => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(0);

    let nodes = blueprint
        .nodes
        .iter()
        .map(|n| (n.name.clone(), Some(n.kind.clone()), false))
        .collect();

    let mut edges: Vec<(String, String)> = blueprint
        .edges
        .iter()
        .map(|e| (e.from.clone(), e.to.clone()))
        .collect();
    edges.push((START.to_string(), blueprint.start.clone()));

    let mut conditional: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for node in &blueprint.nodes {
        match &node.routing {
            Routing::Next(target) => edges.push((node.name.clone(), target.clone())),
            Routing::Terminal => edges.push((node.name.clone(), END.to_string())),
            Routing::Conditional(routes) => {
                conditional.push((node.name.clone(), routes.clone()));
            }
        }
    }

    let channels = blueprint
        .channels
        .iter()
        .map(|c| (c.name.clone(), c.reducer.clone()))
        .collect();

    build_topology(TopologyParts {
        graph_id: blueprint.graph_id.clone(),
        recursion_limit,
        parallel: false,
        nodes,
        edges,
        conditional,
        channels,
    })
}

/// Serializes a [`GraphTopology`] to a pretty-printed JSON document.
pub fn to_json(topology: &GraphTopology) -> String {
    // GraphTopology is a plain data struct of strings/bools/usize; serialization
    // is infallible in practice, but fall back to an empty object defensively.
    serde_json::to_string_pretty(topology).unwrap_or_else(|_| "{}".to_string())
}

/// Deserializes a [`GraphTopology`] from a JSON document produced by
/// [`to_json`]. Returns [`crate::error::RustAgentsError::Serialization`] on
/// malformed input.
pub fn from_json(json: &str) -> Result<GraphTopology> {
    Ok(serde_json::from_str(json)?)
}

/// Renders a [`GraphTopology`] as a Mermaid `flowchart TD`.
///
/// The synthetic `START` and `END` boundaries are drawn as stadium nodes,
/// direct edges use `-->`, and conditional routes are labeled `-- label -->`.
/// Output is deterministic: node declarations and edges follow the topology's
/// already-sorted ordering, so the same graph always renders identically.
pub fn to_mermaid(topology: &GraphTopology) -> String {
    let mut out = String::from("flowchart TD\n");

    // Boundary nodes.
    out.push_str("    START([START])\n");
    out.push_str("    END([END])\n");

    // Node declarations with a quoted label so ids with reserved characters
    // still render. Mermaid ids are sanitized; the label keeps the original.
    for node in &topology.nodes {
        let id = mermaid_id(&node.id);
        out.push_str(&format!("    {id}[\"{}\"]\n", escape_label(&node.id)));
    }

    out.push('\n');

    // Entry edge.
    if let Some(entry) = &topology.entry {
        out.push_str(&format!("    START --> {}\n", mermaid_ref(entry)));
    }

    // Direct edges.
    for edge in &topology.edges {
        out.push_str(&format!(
            "    {} --> {}\n",
            mermaid_ref(&edge.from),
            mermaid_ref(&edge.to)
        ));
    }

    // Conditional edges, labeled.
    for cond in &topology.conditional_edges {
        for route in &cond.routes {
            out.push_str(&format!(
                "    {} -- {} --> {}\n",
                mermaid_ref(&cond.from),
                escape_label(&route.label),
                mermaid_ref(&route.target)
            ));
        }
    }

    // Finish edges.
    for node in &topology.finish_nodes {
        out.push_str(&format!("    {} --> END\n", mermaid_ref(node)));
    }

    out
}

/// Convenience: render a `.rag` [`Blueprint`] directly to Mermaid.
pub fn blueprint_to_mermaid(blueprint: &Blueprint) -> String {
    to_mermaid(&blueprint_to_topology(blueprint))
}

/// Convenience: render a `.rag` [`Blueprint`] directly to pretty JSON.
pub fn blueprint_to_json(blueprint: &Blueprint) -> String {
    to_json(&blueprint_to_topology(blueprint))
}

/// Maps a node id to its Mermaid reference token. The reserved `START`/`END`
/// boundaries map to the literal boundary nodes; everything else is sanitized.
fn mermaid_ref(id: &str) -> String {
    if id == START || id == "START" {
        "START".to_string()
    } else if id == END || id == "END" {
        "END".to_string()
    } else {
        mermaid_id(id)
    }
}

/// Produces a Mermaid-safe identifier from an arbitrary node id by replacing
/// any non-alphanumeric/underscore character with `_` and prefixing `n_` so the
/// result never collides with the reserved `START`/`END` tokens.
fn mermaid_id(id: &str) -> String {
    let mut sanitized = String::with_capacity(id.len() + 2);
    sanitized.push_str("n_");
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    sanitized
}

/// Escapes a label for use inside a Mermaid quoted string or edge label.
fn escape_label(label: &str) -> String {
    label.replace('"', "&quot;")
}

#[cfg(test)]
mod test;
