//! Compiler: lowers a [`Program`] AST into validated [`Blueprint`]s and wires
//! a blueprint into a runtime [`StateGraph`].
//!
//! The compiler has three responsibilities, each exposed as a free function or
//! trait so callers can stop at the level of safety they need:
//!
//! 1. [`compile`] — semantic validation of the AST and lowering into one
//!    serializable [`Blueprint`] per graph.
//! 2. [`bind_capabilities`] — checks every model/tool reference in a blueprint
//!    against an allowlist ([`CapabilityResolver`]). This is the registry
//!    binding gate: declarative source can only reference capabilities that
//!    Rust has already registered and allowed.
//! 3. [`build_graph`] — materialises a blueprint into a legacy
//!    [`StateGraph`] using a caller-supplied [`NodeFactory`]. The blueprint
//!    describes *topology*; runnable node behaviour comes entirely from the
//!    Rust-side factory, never from the declarative source.

use std::collections::HashSet;

use crate::error::{Result, RustAgentsError};
use crate::graph::{Node, StateGraph};
use crate::language::types::{Blueprint, ChannelSpec, END, EdgeSpec, NodeSpec, Program, Routing};

// ===========================================================================
// Semantic compilation: AST -> Blueprint
// ===========================================================================

/// Compiles a parsed [`Program`] into one [`Blueprint`] per declared graph.
///
/// This performs the semantic validation required by the language spec:
///
/// - duplicate node names within a graph are rejected,
/// - a graph must declare a `start` node,
/// - the `start` node must be defined,
/// - every `next`, `route`, and edge target must be a defined node or the
///   reserved `END`,
/// - a node may use static routing (`next` / incident edges) *or* command
///   routing (`routes`), never both.
///
/// All failures are reported as [`RustAgentsError::Compile`].
pub fn compile(program: &Program) -> Result<Vec<Blueprint>> {
    program.graphs.iter().map(compile_graph).collect()
}

fn compile_graph(graph: &crate::language::types::GraphDecl) -> Result<Blueprint> {
    let compile_err = |msg: String| RustAgentsError::Compile(msg);

    // 1. Collect node names, rejecting duplicates.
    let mut node_names: HashSet<&str> = HashSet::new();
    for node in &graph.nodes {
        if !node_names.insert(node.name.as_str()) {
            return Err(compile_err(format!(
                "duplicate node `{}` in graph `{}`",
                node.name, graph.name
            )));
        }
    }

    // A target is valid if it is a known node or the virtual `END`.
    let target_ok = |target: &str| target == END || node_names.contains(target);

    // 2. Start node must be declared and defined.
    let start = graph
        .start
        .clone()
        .ok_or_else(|| compile_err(format!("graph `{}` has no `start` node", graph.name)))?;
    if !node_names.contains(start.as_str()) {
        return Err(compile_err(format!(
            "start node `{start}` is not defined in graph `{}`",
            graph.name
        )));
    }

    // 3. Validate top-level edges and lower them.
    let mut edges = Vec::new();
    for edge in &graph.edges {
        if !node_names.contains(edge.from.as_str()) {
            return Err(compile_err(format!(
                "edge source `{}` does not exist in graph `{}`",
                edge.from, graph.name
            )));
        }
        if !target_ok(&edge.to) {
            return Err(compile_err(format!(
                "edge target `{}` does not exist in graph `{}`",
                edge.to, graph.name
            )));
        }
        edges.push(EdgeSpec {
            from: edge.from.clone(),
            to: edge.to.clone(),
        });
    }

    // Nodes that already have a static outgoing edge declared at top level.
    let nodes_with_static_edge: HashSet<&str> =
        graph.edges.iter().map(|e| e.from.as_str()).collect();

    // 4. Validate and lower each node.
    let mut nodes = Vec::new();
    for node in &graph.nodes {
        let has_routes = !node.routes.is_empty();
        let has_next = node.next.is_some();
        let has_static_edge = nodes_with_static_edge.contains(node.name.as_str());

        if has_routes && (has_next || has_static_edge) {
            return Err(compile_err(format!(
                "node `{}` mixes static routing (`next`/edge) with command routing (`routes`); use one or the other",
                node.name
            )));
        }

        // Validate routes.
        let mut seen_labels: HashSet<&str> = HashSet::new();
        for route in &node.routes {
            if !seen_labels.insert(route.label.as_str()) {
                return Err(compile_err(format!(
                    "duplicate route label `{}` on node `{}`",
                    route.label, node.name
                )));
            }
            if !target_ok(&route.target) {
                return Err(compile_err(format!(
                    "route target `{}` on node `{}` does not exist",
                    route.target, node.name
                )));
            }
        }

        // Validate `next`.
        if let Some(next) = &node.next
            && !target_ok(next)
        {
            return Err(compile_err(format!(
                "next target `{next}` on node `{}` does not exist",
                node.name
            )));
        }

        // Determine routing. Precedence: explicit `routes` > `next` >
        // top-level edge > terminal.
        let routing = if has_routes {
            Routing::Conditional(
                node.routes
                    .iter()
                    .map(|r| (r.label.clone(), r.target.clone()))
                    .collect(),
            )
        } else if let Some(next) = &node.next {
            if next == END {
                Routing::Terminal
            } else {
                Routing::Next(next.clone())
            }
        } else if let Some(edge) = graph.edges.iter().find(|e| e.from == node.name) {
            if edge.to == END {
                Routing::Terminal
            } else {
                Routing::Next(edge.to.clone())
            }
        } else {
            Routing::Terminal
        };

        nodes.push(NodeSpec {
            name: node.name.clone(),
            kind: node.kind.clone().unwrap_or_else(|| "model".to_string()),
            model: node.model.clone(),
            prompt: node.prompt.clone(),
            tools: node.tools.clone(),
            routing,
        });
    }

    let channels = graph
        .channels
        .iter()
        .map(|c| ChannelSpec {
            name: c.name.clone(),
            reducer: c.reducer.clone(),
        })
        .collect();

    Ok(Blueprint {
        graph_id: graph.name.clone(),
        start,
        channels,
        nodes,
        edges,
        defaults: graph.defaults.clone(),
    })
}

// ===========================================================================
// Capability binding
// ===========================================================================

/// An allowlist of model and tool capability names.
///
/// The expressive language can only *reference* capabilities by name; it can
/// never define them. [`bind_capabilities`] uses a resolver to ensure that
/// every referenced model and tool was already registered and allowed by Rust,
/// which is what makes agent-authored source safe to compile.
#[derive(Clone, Debug, Default)]
pub struct CapabilityResolver {
    models: HashSet<String>,
    tools: HashSet<String>,
}

impl CapabilityResolver {
    /// Creates an empty resolver that allows nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a resolver from iterators of allowed model and tool names.
    pub fn from_lists<M, T>(models: M, tools: T) -> Self
    where
        M: IntoIterator<Item = String>,
        T: IntoIterator<Item = String>,
    {
        Self {
            models: models.into_iter().collect(),
            tools: tools.into_iter().collect(),
        }
    }

    /// Allows an additional model name. Returns `self` for chaining.
    pub fn allow_model(mut self, name: impl Into<String>) -> Self {
        self.models.insert(name.into());
        self
    }

    /// Allows an additional tool name. Returns `self` for chaining.
    pub fn allow_tool(mut self, name: impl Into<String>) -> Self {
        self.tools.insert(name.into());
        self
    }

    /// Returns true if `name` is an allowed model.
    pub fn model_allowed(&self, name: &str) -> bool {
        self.models.contains(name)
    }

    /// Returns true if `name` is an allowed tool.
    pub fn tool_allowed(&self, name: &str) -> bool {
        self.tools.contains(name)
    }
}

/// Verifies that every model and tool referenced by `blueprint` is allowed by
/// `allow`.
///
/// # Errors
///
/// Returns [`RustAgentsError::Capability`] for the first model or tool
/// reference that is not present in the resolver's allowlist.
pub fn bind_capabilities(blueprint: &Blueprint, allow: &CapabilityResolver) -> Result<()> {
    for node in &blueprint.nodes {
        if let Some(model) = &node.model
            && !allow.model_allowed(model)
        {
            return Err(RustAgentsError::Capability(format!(
                "node `{}` references unknown model `{model}`",
                node.name
            )));
        }
        for tool in &node.tools {
            if !allow.tool_allowed(tool) {
                return Err(RustAgentsError::Capability(format!(
                    "node `{}` references unknown tool `{tool}`",
                    node.name
                )));
            }
        }
    }
    Ok(())
}

// ===========================================================================
// Topology materialisation: Blueprint -> StateGraph
// ===========================================================================

/// Builds runtime [`Node`]s from compiled [`NodeSpec`]s.
///
/// This is the boundary between the declarative language and executable Rust:
/// the [`Blueprint`] describes *what* nodes exist and how they are wired, while
/// the factory provides *how* each node behaves. Keeping behaviour on the Rust
/// side is what stops `.rag` source from smuggling in arbitrary code.
pub trait NodeFactory<State> {
    /// Materialises a runnable [`Node`] for the given specification.
    ///
    /// # Errors
    ///
    /// Implementations should return an error (typically
    /// [`RustAgentsError::Compile`] or [`RustAgentsError::Capability`]) when a
    /// node kind is unsupported or a required binding is missing.
    fn make(&self, spec: &NodeSpec) -> Result<Node<State>>;
}

/// Wires a [`Blueprint`] into a legacy [`StateGraph`] using `factory` to
/// materialise each node.
///
/// [`Routing`] is translated into the graph's edge model:
///
/// - [`Routing::Next`] -> [`StateGraph::add_edge`],
/// - [`Routing::Conditional`] -> [`StateGraph::add_conditional_edges`].
///   Route labels whose target is the reserved `END` are *not* added to the
///   edge table (the legacy graph has no `END` node); instead the node's
///   handler is expected to emit a [`crate::graph::NodeOutput::End`] for those
///   labels. Non-terminal labels map to their target node.
/// - [`Routing::Terminal`] -> [`StateGraph::add_end`].
///
/// # Errors
///
/// Propagates any error from `factory.make`.
pub fn build_graph<State, F>(blueprint: &Blueprint, factory: &F) -> Result<StateGraph<State>>
where
    State: Clone + Send + 'static,
    F: NodeFactory<State>,
{
    let mut graph = StateGraph::new().set_start(&blueprint.start);

    for spec in &blueprint.nodes {
        let node = factory.make(spec)?;
        graph = graph.add_node(node);

        graph = match &spec.routing {
            Routing::Next(target) => graph.add_edge(&spec.name, target),
            Routing::Conditional(routes) => graph.add_conditional_edges(
                &spec.name,
                routes
                    .iter()
                    .filter(|(_, target)| target != END)
                    .map(|(label, target)| (label.clone(), target.clone())),
            ),
            Routing::Terminal => graph.add_end(&spec.name),
        };
    }

    Ok(graph)
}
