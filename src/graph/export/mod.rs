//! Graph export and visualization — the introspection surface that lets a
//! recursive harness read back the shape of any graph, including ones a model
//! authored or assembled at runtime.
//!
//! Because graphs in this runtime can be built by hand, compiled from a `.rag`
//! blueprint, or emitted by a model and run on the same runtime, it matters
//! that all three reduce to one inspectable description. Export captures that
//! description as a behavior-free [`GraphTopology`] — never the runnable
//! handler/router closures — so a graph can be diffed, snapshotted in tests, or
//! drawn for a human reviewing what an agent just constructed.
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
//! use tinyagents::graph::{GraphBuilder, NodeResult, START, END};
//! use tinyagents::graph::export::{to_json, to_mermaid};
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

pub use types::{
    ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphPolicySummary, GraphTopology, NodeInfo,
    NodePolicySummary, RouteInfo, ValidationReport, WaitingEdgeInfo,
};

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::Result;
use crate::graph::builder::{END, GraphBuilder, START};
use crate::graph::compiled::CompiledGraph;
use crate::language::{Blueprint, Routing};

/// A behavior-free description of one node, fed into [`build_topology`].
struct NodePart {
    id: String,
    kind: Option<String>,
    command_routing: bool,
    subgraph: bool,
    interrupt: bool,
    deferred: bool,
    command_destinations: Vec<String>,
    metadata: BTreeMap<String, String>,
}

/// Inputs to [`build_topology`]: a behavior-free view of one graph's structure.
struct TopologyParts {
    graph_id: String,
    name: Option<String>,
    recursion_limit: usize,
    parallel: bool,
    max_concurrency: Option<usize>,
    node_timeout_ms: Option<u128>,
    /// Every declared node.
    nodes: Vec<NodePart>,
    /// Raw edges, including the synthetic `START`/`END` edges and any edges that
    /// are also registered as barrier/waiting edges.
    edges: Vec<(String, String)>,
    /// `(from, [(label, target)])` conditional routes.
    conditional: Vec<(String, Vec<(String, String)>)>,
    /// `(target, [predecessors])` barrier/waiting (fan-in) edges.
    waiting: Vec<(String, Vec<String>)>,
    /// `(channel, reducer)` bindings.
    channels: Vec<(String, String)>,
}

/// Folds raw structural parts into a normalized, deterministically-ordered
/// [`GraphTopology`]. Synthetic `START`/`END` edges are lifted into
/// [`GraphTopology::entry`] and [`GraphTopology::finish_nodes`]; barrier edges
/// are lifted out of the direct-edge set into [`GraphTopology::waiting_edges`].
fn build_topology(parts: TopologyParts) -> GraphTopology {
    let TopologyParts {
        graph_id,
        name,
        recursion_limit,
        parallel,
        max_concurrency,
        node_timeout_ms,
        nodes,
        edges,
        conditional,
        waiting,
        channels,
    } = parts;

    // Normalize waiting edges first so direct-edge folding can skip them.
    let mut waiting_edges: Vec<WaitingEdgeInfo> = waiting
        .into_iter()
        .map(|(target, mut predecessors)| {
            predecessors.sort();
            predecessors.dedup();
            WaitingEdgeInfo {
                target,
                predecessors,
            }
        })
        .collect();
    waiting_edges.sort();
    let waiting_pairs: BTreeSet<(String, String)> = waiting_edges
        .iter()
        .flat_map(|w| {
            w.predecessors
                .iter()
                .map(move |p| (p.clone(), w.target.clone()))
        })
        .collect();
    let barrier_targets: BTreeSet<String> =
        waiting_edges.iter().map(|w| w.target.clone()).collect();

    let mut entry: Option<String> = None;
    let mut direct: Vec<EdgeInfo> = Vec::new();
    let mut finish_nodes: Vec<String> = Vec::new();

    for (from, to) in edges {
        if from == START {
            entry = Some(to);
        } else if to == END {
            finish_nodes.push(from);
        } else if waiting_pairs.contains(&(from.clone(), to.clone())) {
            // Represented as a waiting edge, not a direct one.
        } else {
            direct.push(EdgeInfo { from, to });
        }
    }

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
    direct.sort();
    conditional_edges.sort_by(|a, b| a.from.cmp(&b.from));
    finish_nodes.sort();
    finish_nodes.dedup();

    // Index sets used to derive per-node policy summaries.
    let conditional_set: BTreeSet<&str> =
        conditional_edges.iter().map(|c| c.from.as_str()).collect();
    let direct_from_set: BTreeSet<&str> = direct.iter().map(|e| e.from.as_str()).collect();
    let finish_set: BTreeSet<&str> = finish_nodes.iter().map(String::as_str).collect();

    let mut node_infos: Vec<NodeInfo> = nodes
        .into_iter()
        .map(|n| {
            let routing = if n.command_routing {
                "command"
            } else if conditional_set.contains(n.id.as_str()) {
                "conditional"
            } else if finish_set.contains(n.id.as_str()) {
                "terminal"
            } else if direct_from_set.contains(n.id.as_str()) {
                "static"
            } else {
                "unrouted"
            };
            let policy = NodePolicySummary {
                routing: routing.to_string(),
                barrier: barrier_targets.contains(&n.id),
                interrupt: n.interrupt,
                deferred: n.deferred,
                subgraph: n.subgraph,
            };
            let mut command_destinations = n.command_destinations;
            command_destinations.sort();
            command_destinations.dedup();
            NodeInfo {
                id: n.id,
                kind: n.kind,
                command_routing: n.command_routing,
                subgraph: n.subgraph,
                interrupt: n.interrupt,
                deferred: n.deferred,
                command_destinations,
                metadata: n.metadata,
                policy,
            }
        })
        .collect();
    node_infos.sort_by(|a, b| a.id.cmp(&b.id));

    let policy = GraphPolicySummary {
        recursion_limit,
        parallel,
        max_concurrency,
        node_timeout_ms,
    };

    let validation = validate(
        &node_infos,
        entry.as_deref(),
        &direct,
        &conditional_edges,
        &waiting_edges,
        &finish_nodes,
    );

    GraphTopology {
        graph_id,
        name,
        entry,
        recursion_limit,
        parallel,
        nodes: node_infos,
        edges: direct,
        conditional_edges,
        waiting_edges,
        finish_nodes,
        channels,
        policy,
        validation,
    }
}

/// Computes a structural [`ValidationReport`] over the assembled topology.
///
/// Errors flag references to undeclared nodes (a dangling entry, edge, route,
/// or barrier target) which would make the graph invalid; warnings flag
/// non-fatal observations (unreachable nodes, `unrouted` dead-ends). The virtual
/// `START`/`END` boundaries are never treated as undeclared.
fn validate(
    nodes: &[NodeInfo],
    entry: Option<&str>,
    direct: &[EdgeInfo],
    conditional: &[ConditionalEdgeInfo],
    waiting: &[WaitingEdgeInfo],
    finish_nodes: &[String],
) -> ValidationReport {
    let declared: BTreeSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    let known = |id: &str| id == START || id == END || declared.contains(id);

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    match entry {
        None => errors.push("graph has no entry node (no START edge)".to_string()),
        Some(e) if !known(e) => {
            errors.push(format!("entry node `{e}` is not declared"));
        }
        Some(_) => {}
    }

    for edge in direct {
        if !known(&edge.from) {
            errors.push(format!("edge source `{}` is not declared", edge.from));
        }
        if !known(&edge.to) {
            errors.push(format!("edge target `{}` is not declared", edge.to));
        }
    }
    for cond in conditional {
        if !known(&cond.from) {
            errors.push(format!("conditional source `{}` is not declared", cond.from));
        }
        for route in &cond.routes {
            if !known(&route.target) {
                errors.push(format!(
                    "conditional route `{}` of `{}` targets undeclared node `{}`",
                    route.label, cond.from, route.target
                ));
            }
        }
    }
    for w in waiting {
        if !known(&w.target) {
            errors.push(format!("barrier target `{}` is not declared", w.target));
        }
        for pred in &w.predecessors {
            if !known(pred) {
                errors.push(format!(
                    "barrier predecessor `{pred}` of `{}` is not declared",
                    w.target
                ));
            }
        }
    }
    for node in nodes {
        for dest in &node.command_destinations {
            if !known(dest) {
                errors.push(format!(
                    "command destination `{dest}` of `{}` is not declared",
                    node.id
                ));
            }
        }
    }
    for f in finish_nodes {
        if !known(f) {
            errors.push(format!("finish node `{f}` is not declared"));
        }
    }

    // Reachability from the entry across every successor relation.
    if let Some(entry) = entry {
        let mut successors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for edge in direct {
            successors
                .entry(&edge.from)
                .or_default()
                .insert(&edge.to);
        }
        for cond in conditional {
            let entry_set = successors.entry(&cond.from).or_default();
            for route in &cond.routes {
                entry_set.insert(&route.target);
            }
        }
        for w in waiting {
            for pred in &w.predecessors {
                successors.entry(pred).or_default().insert(&w.target);
            }
        }
        for node in nodes {
            for dest in &node.command_destinations {
                successors.entry(&node.id).or_default().insert(dest);
            }
        }

        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        seen.insert(entry);
        queue.push_back(entry);
        while let Some(node) = queue.pop_front() {
            if let Some(next) = successors.get(node) {
                for &n in next {
                    if seen.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
        }
        for node in nodes {
            if !seen.contains(node.id.as_str()) {
                warnings.push(format!("node `{}` is unreachable from the entry", node.id));
            }
        }
    }

    // Dead-end (unrouted) nodes that are not terminal.
    for node in nodes {
        if node.policy.routing == "unrouted" {
            warnings.push(format!(
                "node `{}` has no outgoing route and is not terminal",
                node.id
            ));
        }
    }

    errors.sort();
    errors.dedup();
    warnings.sort();
    warnings.dedup();

    ValidationReport {
        ok: errors.is_empty(),
        errors,
        warnings,
    }
}

/// Builds the per-node [`NodePart`] list from a declared-node iterator, the
/// command-routing set, and the behavior-free node-metadata map.
fn node_parts<'a, I>(
    ids: I,
    is_command: impl Fn(&str) -> bool,
    meta: &std::collections::HashMap<crate::harness::ids::NodeId, crate::graph::builder::NodeMeta>,
) -> Vec<NodePart>
where
    I: IntoIterator<Item = &'a crate::harness::ids::NodeId>,
{
    ids.into_iter()
        .map(|id| {
            let m = meta.get(id);
            NodePart {
                id: id.to_string(),
                kind: m.and_then(|m| m.kind.clone()),
                command_routing: is_command(id.as_str()),
                subgraph: m.is_some_and(|m| m.subgraph),
                interrupt: m.is_some_and(|m| m.interrupt),
                deferred: m.is_some_and(|m| m.deferred),
                command_destinations: m
                    .map(|m| m.command_destinations.iter().map(|d| d.to_string()).collect())
                    .unwrap_or_default(),
                metadata: m.map(|m| m.metadata.clone()).unwrap_or_default(),
            }
        })
        .collect()
}

impl<State, Update> CompiledGraph<State, Update> {
    /// Extracts a behavior-free [`GraphTopology`] describing this graph's
    /// structure (id, nodes, direct edges, conditional routes, entry, finish
    /// nodes). Node handler and router closures are never exposed.
    pub fn topology(&self) -> GraphTopology {
        let nodes = node_parts(
            self.nodes.keys(),
            |id| self.command_nodes.contains(id),
            &self.node_meta,
        );
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
        let waiting = self
            .waiting
            .iter()
            .map(|(target, preds)| {
                (
                    target.to_string(),
                    preds.iter().map(ToString::to_string).collect(),
                )
            })
            .collect();
        build_topology(TopologyParts {
            graph_id: self.graph_id().to_string(),
            name: self.name().map(str::to_string),
            recursion_limit: self.recursion_limit,
            parallel: self.parallel,
            max_concurrency: self.max_concurrency,
            node_timeout_ms: self.node_timeout.map(|d| d.as_millis()),
            nodes,
            edges,
            conditional,
            waiting,
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
        let nodes = node_parts(
            self.nodes.keys(),
            |id| self.command_nodes.contains(id),
            &self.node_meta,
        );
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
        let waiting = self
            .waiting
            .iter()
            .map(|(target, preds)| {
                (
                    target.to_string(),
                    preds.iter().map(ToString::to_string).collect(),
                )
            })
            .collect();
        build_topology(TopologyParts {
            graph_id: self.graph_id.to_string(),
            name: self.name.clone(),
            recursion_limit: self.recursion_limit,
            parallel: self.parallel,
            max_concurrency: self.max_concurrency,
            node_timeout_ms: self.node_timeout.map(|d| d.as_millis()),
            nodes,
            edges,
            conditional,
            waiting,
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
        .map(|n| {
            let subgraph = n.kind == "subgraph";
            NodePart {
                id: n.name.clone(),
                kind: Some(n.kind.clone()),
                command_routing: false,
                subgraph,
                interrupt: false,
                deferred: false,
                command_destinations: Vec::new(),
                metadata: BTreeMap::new(),
            }
        })
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
        name: None,
        recursion_limit,
        parallel: false,
        max_concurrency: None,
        node_timeout_ms: None,
        nodes,
        edges,
        conditional,
        waiting: Vec::new(),
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
/// [`to_json`]. Returns [`crate::error::TinyAgentsError::Serialization`] on
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
    // still render. Subgraph nodes use the Mermaid subroutine shape (`[[...]]`)
    // so they read as embedded graphs; everything else is a plain box.
    for node in &topology.nodes {
        let id = mermaid_id(&node.id);
        let label = escape_label(&node.id);
        if node.subgraph {
            out.push_str(&format!("    {id}[[\"{label}\"]]\n"));
        } else {
            out.push_str(&format!("    {id}[\"{label}\"]\n"));
        }
    }

    // Marker classes. Emitted only when at least one node carries the marker so
    // the output stays minimal and deterministic.
    emit_marker_class(&mut out, topology, "subgraph", |n| n.subgraph);
    emit_marker_class(&mut out, topology, "interrupt", |n| n.interrupt);
    emit_marker_class(&mut out, topology, "deferred", |n| n.deferred);

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

    // Barrier/waiting (fan-in) edges drawn as dotted, labeled joins.
    for w in &topology.waiting_edges {
        for pred in &w.predecessors {
            out.push_str(&format!(
                "    {} -. barrier .-> {}\n",
                mermaid_ref(pred),
                mermaid_ref(&w.target)
            ));
        }
    }

    // Command `goto` destination hints drawn as dotted, labeled edges.
    for node in &topology.nodes {
        for dest in &node.command_destinations {
            out.push_str(&format!(
                "    {} -. goto .-> {}\n",
                mermaid_ref(&node.id),
                mermaid_ref(dest)
            ));
        }
    }

    // Finish edges.
    for node in &topology.finish_nodes {
        out.push_str(&format!("    {} --> END\n", mermaid_ref(node)));
    }

    out
}

/// Emits a Mermaid `classDef` plus `class` assignments for every node matching
/// `pred`, in the topology's already-sorted node order. Nothing is emitted when
/// no node matches, keeping the diagram minimal and deterministic.
fn emit_marker_class(
    out: &mut String,
    topology: &GraphTopology,
    class: &str,
    pred: impl Fn(&NodeInfo) -> bool,
) {
    let matching: Vec<&NodeInfo> = topology.nodes.iter().filter(|n| pred(n)).collect();
    if matching.is_empty() {
        return;
    }
    out.push_str(&format!(
        "    classDef {class} stroke-dasharray: 4 2;\n"
    ));
    for node in matching {
        out.push_str(&format!("    class {} {class}\n", mermaid_id(&node.id)));
    }
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
