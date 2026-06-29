//! Compiler: lowers a [`Program`] AST into validated [`Blueprint`]s and wires
//! a blueprint into a durable runtime graph.
//!
//! This is the gate that makes recursive self-authoring safe. A `.rag` plan —
//! whether hand-written or emitted by a model running inside the harness — is
//! semantically validated, then bound *by name* against a live registry through
//! [`CapabilityResolver`]/[`bind_capabilities_with_registry`], so the resulting
//! topology can only reach capabilities Rust has already registered and allowed.
//! Runnable behaviour is supplied entirely by a Rust-side [`NodeFactory`], never
//! by the source, so the same compiler path serves human and model authors alike
//! and the model can re-enter the very runtime it is executing in.
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
//! 3. [`build_graph`] — materialises a blueprint into a durable
//!    [`CompiledGraph`] using a caller-supplied [`NodeFactory`]. The blueprint
//!    describes *topology*; runnable node behaviour comes entirely from the
//!    Rust-side factory, never from the declarative source.

use std::collections::HashSet;
use std::sync::Arc;

use crate::error::{Result, TinyAgentsError};
use crate::graph::{CompiledGraph, GraphBuilder, NodeHandler};
use crate::language::parser::parse_str;
use crate::language::types::{Blueprint, ChannelSpec, END, EdgeSpec, NodeSpec, Program, Routing};
use crate::registry::CapabilityRegistry;

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
/// All failures are reported as [`TinyAgentsError::Compile`].
pub fn compile(program: &Program) -> Result<Vec<Blueprint>> {
    program.graphs.iter().map(compile_graph).collect()
}

fn compile_graph(graph: &crate::language::types::GraphDecl) -> Result<Blueprint> {
    let compile_err = |msg: String| TinyAgentsError::Compile(msg);

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

/// The node `kind` values the registry-backed binding path recognises.
///
/// A `.rag` node may only declare one of these kinds when validated through
/// [`bind_capabilities_with_registry`] (or any resolver built with
/// [`CapabilityResolver::from_registry`]); an unknown kind is a
/// [`TinyAgentsError::Compile`] error. The set deliberately includes `model`,
/// because [`compile`] defaults an unspecified kind to `model`.
///
/// The kinds carry the following capability-reference conventions, applied by
/// the strict binding path:
///
/// - `subgraph` / `graph`: the node's `model` field (when present) names a
///   registered graph [`Blueprint`] — a *subgraph reference*.
/// - `router`: the node's `model` field names a registered router function.
/// - everything else (`agent`, `model`, `tool_executor`, `human`): the
///   `model` field names a registered chat model.
pub const DEFAULT_NODE_KINDS: &[&str] = &[
    "agent",
    "model",
    "tool_executor",
    "subgraph",
    "graph",
    "router",
    "human",
];

/// An allowlist of capability names referenced by the expressive language.
///
/// The expressive language can only *reference* capabilities by name; it can
/// never define them. [`bind_capabilities`] uses a resolver to ensure that
/// every referenced model and tool was already registered and allowed by Rust,
/// which is what makes agent-authored source safe to compile.
///
/// A resolver holds five name allowlists — models, tools, subgraphs (graph
/// blueprints), routers, and reducers — plus an optional set of allowed node
/// `kind` values. The minimal [`new`](Self::new) / [`from_lists`](Self::from_lists)
/// constructors populate only models and tools and leave `node_kinds` empty, so
/// the legacy [`bind_capabilities`] gate keeps its original behaviour (model and
/// tool checks only). The richer checks — subgraph, router, and reducer
/// references plus node-kind validation — are opt-in through the
/// registry-backed path: [`from_registry`](Self::from_registry) and
/// [`bind_capabilities_with_registry`].
#[derive(Clone, Debug, Default)]
pub struct CapabilityResolver {
    models: HashSet<String>,
    tools: HashSet<String>,
    subgraphs: HashSet<String>,
    routers: HashSet<String>,
    reducers: HashSet<String>,
    /// Allowed node kinds. When empty, node-kind validation is skipped (the
    /// legacy, manual behaviour); when non-empty, the strict binding path
    /// rejects any node whose kind is not listed.
    node_kinds: HashSet<String>,
}

impl CapabilityResolver {
    /// Creates an empty resolver that allows nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a resolver from iterators of allowed model and tool names.
    ///
    /// Subgraph, router, and reducer allowlists are left empty and node-kind
    /// validation is disabled; use [`from_registry`](Self::from_registry) for a
    /// fully populated, registry-backed resolver.
    pub fn from_lists<M, T>(models: M, tools: T) -> Self
    where
        M: IntoIterator<Item = String>,
        T: IntoIterator<Item = String>,
    {
        Self {
            models: models.into_iter().collect(),
            tools: tools.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Builds a fully populated resolver from a live [`CapabilityRegistry`].
    ///
    /// Every registered model, tool, graph blueprint, router, and reducer name
    /// — including their aliases — is added to the corresponding allowlist, and
    /// the node-kind allowlist is seeded with [`DEFAULT_NODE_KINDS`]. The
    /// resulting resolver therefore validates `.rag` source against exactly what
    /// Rust has registered, including subgraph/router/reducer references and
    /// node kinds, when used with [`CapabilityResolver::bind_blueprint`] or
    /// [`bind_capabilities_with_registry`].
    pub fn from_registry<State: Send + Sync>(registry: &CapabilityRegistry<State>) -> Self {
        use crate::registry::ComponentKind;

        let collect = |kind| registry.names_including_aliases(kind).into_iter().collect();
        Self {
            models: collect(ComponentKind::Model),
            tools: collect(ComponentKind::Tool),
            subgraphs: collect(ComponentKind::Graph),
            routers: collect(ComponentKind::Router),
            reducers: collect(ComponentKind::Reducer),
            node_kinds: DEFAULT_NODE_KINDS.iter().map(|k| (*k).to_owned()).collect(),
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

    /// Allows an additional subgraph (graph blueprint) name. Returns `self`.
    pub fn allow_subgraph(mut self, name: impl Into<String>) -> Self {
        self.subgraphs.insert(name.into());
        self
    }

    /// Allows an additional router name. Returns `self` for chaining.
    pub fn allow_router(mut self, name: impl Into<String>) -> Self {
        self.routers.insert(name.into());
        self
    }

    /// Allows an additional reducer name. Returns `self` for chaining.
    pub fn allow_reducer(mut self, name: impl Into<String>) -> Self {
        self.reducers.insert(name.into());
        self
    }

    /// Replaces the set of allowed node kinds. Passing a non-empty set enables
    /// node-kind validation in the strict binding path. Returns `self`.
    pub fn with_node_kinds<I, S>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.node_kinds = kinds.into_iter().map(Into::into).collect();
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

    /// Returns true if `name` is an allowed subgraph (graph blueprint).
    pub fn subgraph_allowed(&self, name: &str) -> bool {
        self.subgraphs.contains(name)
    }

    /// Returns true if `name` is an allowed router.
    pub fn router_allowed(&self, name: &str) -> bool {
        self.routers.contains(name)
    }

    /// Returns true if `name` is an allowed reducer.
    pub fn reducer_allowed(&self, name: &str) -> bool {
        self.reducers.contains(name)
    }

    /// Returns true if `kind` is an allowed node kind, or if node-kind
    /// validation is disabled (the allowlist is empty).
    pub fn node_kind_allowed(&self, kind: &str) -> bool {
        self.node_kinds.is_empty() || self.node_kinds.contains(kind)
    }

    /// Runs the full, strict capability binding for `blueprint`.
    ///
    /// In addition to the model/tool checks performed by [`bind_capabilities`],
    /// this validates, per the conventions documented on [`DEFAULT_NODE_KINDS`]:
    ///
    /// - each node `kind` is in the resolver's node-kind allowlist (a
    ///   [`TinyAgentsError::Compile`] error otherwise);
    /// - `subgraph`/`graph` node references resolve to a registered subgraph,
    ///   `router` node references to a registered router, and all other nodes'
    ///   `model` references to a registered model;
    /// - every `channel` reducer reference is registered.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Compile`] for an unknown node kind, and
    /// [`TinyAgentsError::Capability`] for the first unregistered model, tool,
    /// subgraph, router, or reducer reference.
    pub fn bind_blueprint(&self, blueprint: &Blueprint) -> Result<()> {
        for node in &blueprint.nodes {
            if !self.node_kind_allowed(&node.kind) {
                return Err(TinyAgentsError::Compile(format!(
                    "node `{}` has unknown kind `{}`",
                    node.name, node.kind
                )));
            }

            match node.kind.as_str() {
                "subgraph" | "graph" => {
                    if let Some(target) = &node.model
                        && !self.subgraph_allowed(target)
                    {
                        return Err(TinyAgentsError::Capability(format!(
                            "node `{}` references unknown subgraph `{target}`",
                            node.name
                        )));
                    }
                }
                "router" => {
                    if let Some(target) = &node.model
                        && !self.router_allowed(target)
                    {
                        return Err(TinyAgentsError::Capability(format!(
                            "node `{}` references unknown router `{target}`",
                            node.name
                        )));
                    }
                }
                _ => {
                    if let Some(model) = &node.model
                        && !self.model_allowed(model)
                    {
                        return Err(TinyAgentsError::Capability(format!(
                            "node `{}` references unknown model `{model}`",
                            node.name
                        )));
                    }
                }
            }

            for tool in &node.tools {
                if !self.tool_allowed(tool) {
                    return Err(TinyAgentsError::Capability(format!(
                        "node `{}` references unknown tool `{tool}`",
                        node.name
                    )));
                }
            }
        }

        for channel in &blueprint.channels {
            if !self.reducer_allowed(&channel.reducer) {
                return Err(TinyAgentsError::Capability(format!(
                    "channel `{}` references unknown reducer `{}`",
                    channel.name, channel.reducer
                )));
            }
        }

        Ok(())
    }
}

/// Verifies that every model and tool referenced by `blueprint` is allowed by
/// `allow`.
///
/// This is the minimal, manual gate: it checks only `model` and `tool`
/// references on each node and never inspects node kinds, subgraph/router
/// references, or channel reducers. For full registry-backed validation use
/// [`bind_capabilities_with_registry`].
///
/// # Errors
///
/// Returns [`TinyAgentsError::Capability`] for the first model or tool
/// reference that is not present in the resolver's allowlist.
pub fn bind_capabilities(blueprint: &Blueprint, allow: &CapabilityResolver) -> Result<()> {
    for node in &blueprint.nodes {
        if let Some(model) = &node.model
            && !allow.model_allowed(model)
        {
            return Err(TinyAgentsError::Capability(format!(
                "node `{}` references unknown model `{model}`",
                node.name
            )));
        }
        for tool in &node.tools {
            if !allow.tool_allowed(tool) {
                return Err(TinyAgentsError::Capability(format!(
                    "node `{}` references unknown tool `{tool}`",
                    node.name
                )));
            }
        }
    }
    Ok(())
}

/// Validates `blueprint` against a live [`CapabilityRegistry`].
///
/// This is the registry → language binding gate. It builds a fully populated
/// [`CapabilityResolver`] from `registry` (models, tools, subgraphs, routers,
/// reducers, and the default node kinds) and runs
/// [`CapabilityResolver::bind_blueprint`], so declarative source can only
/// reference capabilities that Rust has actually registered.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Compile`] for an unknown node kind, and
/// [`TinyAgentsError::Capability`] for any unregistered model, tool, subgraph,
/// router, or reducer reference.
pub fn bind_capabilities_with_registry<State: Send + Sync>(
    blueprint: &Blueprint,
    registry: &CapabilityRegistry<State>,
) -> Result<()> {
    CapabilityResolver::from_registry(registry).bind_blueprint(blueprint)
}

/// Parses, compiles, and registry-binds `.rag`/`.ragsh` `source` in one call.
///
/// This is the convenience façade for the common path: it runs
/// `parse -> compile -> registry-bind` and returns the validated blueprints.
/// Every produced [`Blueprint`] is checked against `registry` via
/// [`bind_capabilities_with_registry`], so a returned blueprint references only
/// registered capabilities.
///
/// # Errors
///
/// Propagates [`TinyAgentsError::Parse`] from the parser,
/// [`TinyAgentsError::Compile`] from [`compile`] and node-kind validation, and
/// [`TinyAgentsError::Capability`] from capability binding.
pub fn compile_source<State: Send + Sync>(
    source: &str,
    registry: &CapabilityRegistry<State>,
) -> Result<Vec<Blueprint>> {
    let program = parse_str(source)?;
    let blueprints = compile(&program)?;
    for blueprint in &blueprints {
        bind_capabilities_with_registry(blueprint, registry)?;
    }
    Ok(blueprints)
}

// ===========================================================================
// Topology materialisation: Blueprint -> CompiledGraph
// ===========================================================================

/// A durable node handler materialised from a [`NodeSpec`].
///
/// Whole-state graphs use `Update == State`: the handler receives the committed
/// state snapshot plus a [`crate::graph::NodeContext`] and returns a
/// [`crate::graph::NodeResult`]. To continue along a static edge return
/// [`crate::graph::NodeResult::Update`] with the next state; to take a
/// conditional route or stop, return a
/// [`crate::graph::Command`] carrying `goto` (the resolved target node, or the
/// reserved [`crate::graph::END`]) and the next state via `with_update`.
pub type BoxedNode<State> = Arc<NodeHandler<State, State>>;

/// Builds runtime node handlers from compiled [`NodeSpec`]s.
///
/// This is the boundary between the declarative language and executable Rust:
/// the [`Blueprint`] describes *what* nodes exist and how they are wired, while
/// the factory provides *how* each node behaves. Keeping behaviour on the Rust
/// side is what stops `.rag` source from smuggling in arbitrary code.
pub trait NodeFactory<State> {
    /// Materialises a runnable durable node handler for the given
    /// specification.
    ///
    /// For a [`Routing::Conditional`] node the returned handler is responsible
    /// for choosing a route: resolve the chosen label against `spec.routing` to
    /// a target node id and return
    /// `NodeResult::Command(Command::goto([target]).with_update(state))`. The
    /// target may be the reserved [`crate::graph::END`] to stop the run.
    ///
    /// # Errors
    ///
    /// Implementations should return an error (typically
    /// [`TinyAgentsError::Compile`] or [`TinyAgentsError::Capability`]) when a
    /// node kind is unsupported or a required binding is missing.
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<State>>;
}

/// Wires a [`Blueprint`] into a durable, whole-state [`CompiledGraph`] (overwrite
/// reducer) using `factory` to materialise each node.
///
/// [`Routing`] is translated into the durable graph topology:
///
/// - [`Routing::Next`] -> [`GraphBuilder::add_edge`] (a static successor),
/// - [`Routing::Conditional`] -> [`GraphBuilder::mark_command_routing`]. The
///   node decides its own route at runtime by returning a
///   [`crate::graph::Command`] `goto` (the legacy whole-state semantics, where
///   the route label is chosen by the node, not by committed state). The
///   factory resolves the label to a target node id — or the reserved
///   [`crate::graph::END`] — from `spec.routing`.
/// - [`Routing::Terminal`] -> [`GraphBuilder::set_finish`] (route to `END`).
///
/// The blueprint's `start` node becomes the graph entry.
///
/// # Errors
///
/// Propagates any error from `factory.make`, and any topology
/// [`TinyAgentsError::Validation`] raised by [`GraphBuilder::compile`].
pub fn build_graph<State, F>(
    blueprint: &Blueprint,
    factory: &F,
) -> Result<CompiledGraph<State, State>>
where
    State: Clone + Send + Sync + 'static,
    F: NodeFactory<State>,
{
    let mut builder = GraphBuilder::<State, State>::overwrite().set_entry(blueprint.start.as_str());

    for spec in &blueprint.nodes {
        let handler = factory.make(spec)?;
        builder = builder.add_node(spec.name.as_str(), move |state, ctx| {
            (handler.clone())(state, ctx)
        });

        builder = match &spec.routing {
            Routing::Next(target) => builder.add_edge(spec.name.as_str(), target.as_str()),
            Routing::Conditional(_) => builder.mark_command_routing(spec.name.as_str()),
            Routing::Terminal => builder.set_finish(spec.name.as_str()),
        };
    }

    builder.compile()
}
