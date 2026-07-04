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
use crate::language::types::{
    Blueprint, BlueprintProvenance, ChannelSpec, CommandSpec, END, EdgeSpan, EdgeSpec, IoFieldSpec,
    JoinSpec, NamedSpan, NodeSpec, Origin, Program, Routing, SendSpec,
};
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
    let mut graph_ids: HashSet<&str> = HashSet::new();
    for graph in &program.graphs {
        if !graph_ids.insert(graph.name.as_str()) {
            return Err(TinyAgentsError::Compile(format!(
                "duplicate graph `{}`",
                graph.name
            )));
        }
    }
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

    // Reject duplicate channel declarations up front.
    let mut channel_names: HashSet<&str> = HashSet::new();
    for channel in &graph.channels {
        if !channel_names.insert(channel.name.as_str()) {
            return Err(compile_err(format!(
                "duplicate channel `{}` in graph `{}`",
                channel.name, graph.name
            )));
        }
    }

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

    // Reject contradictory multiple top-level edges from the same source: only
    // one static successor can ever be routed to, so a second edge from the
    // same node is silent data loss (the first edge wins, the rest are dead
    // weight in `blueprint.edges`) rather than a legitimate multi-successor.
    for name in &nodes_with_static_edge {
        let targets: Vec<&str> = graph
            .edges
            .iter()
            .filter(|e| e.from == *name)
            .map(|e| e.to.as_str())
            .collect();
        if targets.len() > 1 {
            return Err(compile_err(format!(
                "node `{name}` has multiple top-level edges ({}); a node may declare at most one outgoing edge",
                targets.join(", ")
            )));
        }
    }

    // 4. Validate and lower each node.
    let mut nodes = Vec::new();
    for node in &graph.nodes {
        let has_routes = !node.routes.is_empty();
        let has_next = node.next.is_some();
        let has_static_edge = nodes_with_static_edge.contains(node.name.as_str());
        let has_command_goto = node.command.as_ref().is_some_and(|c| c.goto.is_some());

        if has_routes && (has_next || has_static_edge) {
            return Err(compile_err(format!(
                "node `{}` mixes static routing (`next`/edge) with command routing (`routes`); use one or the other",
                node.name
            )));
        }

        // A node may declare at most one of `routes`, `next`, `command { goto
        // … }`, or a top-level edge as its routing source. Silently resolving
        // by precedence hides a real authoring mistake (e.g. a model-authored
        // revision that adds a `command.goto` without removing the old
        // `next`), so any additional combination is a compile error.
        let routing_sources = [
            (has_routes, "routes"),
            (has_next, "`next`"),
            (has_command_goto, "`command { goto … }`"),
            (has_static_edge, "a top-level edge"),
        ];
        let active: Vec<&str> = routing_sources
            .iter()
            .filter(|(present, _)| *present)
            .map(|(_, label)| *label)
            .collect();
        if active.len() > 1 {
            return Err(compile_err(format!(
                "node `{}` declares conflicting routing sources ({}); use exactly one",
                node.name,
                active.join(", ")
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

        // Validate a `command`'s goto target.
        if let Some(cmd) = &node.command
            && let Some(goto) = &cmd.goto
            && !target_ok(goto)
        {
            return Err(compile_err(format!(
                "command goto target `{goto}` on node `{}` does not exist",
                node.name
            )));
        }

        // Validate fanout `send` targets.
        for send in &node.sends {
            if !target_ok(&send.target) {
                return Err(compile_err(format!(
                    "send target `{}` on node `{}` does not exist",
                    send.target, node.name
                )));
            }
        }

        // Validate `join` node sources.
        for source in &node.sources {
            if !node_names.contains(source.as_str()) {
                return Err(compile_err(format!(
                    "join source `{source}` on node `{}` does not exist",
                    node.name
                )));
            }
        }

        // Determine routing. Precedence: explicit `routes` > `next` > command
        // `goto` > top-level edge > terminal.
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
        } else if let Some(goto) = node.command.as_ref().and_then(|c| c.goto.as_ref()) {
            if goto == END {
                Routing::Terminal
            } else {
                Routing::Next(goto.clone())
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

        let command = node.command.as_ref().map(|c| CommandSpec {
            goto: c.goto.clone(),
            update: c.update.clone(),
        });
        let sends = node
            .sends
            .iter()
            .map(|s| SendSpec {
                target: s.target.clone(),
                input: s.input.clone(),
            })
            .collect();

        nodes.push(NodeSpec {
            name: node.name.clone(),
            kind: node.kind.clone().unwrap_or_else(|| "model".to_string()),
            model: node.model.clone(),
            prompt: node.prompt.clone(),
            tools: node.tools.clone(),
            routing,
            agent: node.agent.clone(),
            subgraph: node.graph.clone(),
            script: node.script.clone(),
            input: node.input.clone(),
            command,
            sends,
            join_sources: node.sources.clone(),
            options: node.options.clone(),
            checkpoint: node.checkpoint.clone(),
            timeout: node.timeout.as_ref().map(|t| t.as_display()),
            retry: node.retry.clone(),
            metadata: node.metadata.clone(),
        });
    }

    // Validate top-level join declarations.
    let mut joins = Vec::new();
    for join in &graph.joins {
        for source in &join.sources {
            if !node_names.contains(source.as_str()) {
                return Err(compile_err(format!(
                    "join source `{source}` does not exist in graph `{}`",
                    graph.name
                )));
            }
        }
        if !target_ok(&join.target) {
            return Err(compile_err(format!(
                "join target `{}` does not exist in graph `{}`",
                join.target, graph.name
            )));
        }
        joins.push(JoinSpec {
            sources: join.sources.clone(),
            target: join.target.clone(),
        });
    }

    let channels = graph
        .channels
        .iter()
        .map(|c| ChannelSpec {
            name: c.name.clone(),
            reducer: c.reducer.clone(),
            args: c.args.clone(),
        })
        .collect();

    let input = graph
        .input
        .iter()
        .map(|f| IoFieldSpec {
            name: f.name.clone(),
            ty: f.ty.clone(),
        })
        .collect();
    let output = graph
        .output
        .iter()
        .map(|f| IoFieldSpec {
            name: f.name.clone(),
            ty: f.ty.clone(),
        })
        .collect();

    Ok(Blueprint {
        graph_id: graph.name.clone(),
        start,
        channels,
        nodes,
        edges,
        defaults: graph.defaults.clone(),
        input,
        output,
        checkpoint: graph.checkpoint.clone(),
        interrupt: graph.interrupt.clone(),
        joins,
        provenance: None,
    })
}

/// Compiles a parsed [`Program`] into one [`Blueprint`] per graph, attaching
/// source [`BlueprintProvenance`] tagged with `origin`.
///
/// This runs the same semantic validation and lowering as [`compile`], then
/// records the source [`Span`](crate::language::span::Span) of every node,
/// channel, and edge plus the blueprint's [`Origin`] so a UI, test, or review
/// tool can trace each compiled piece back to the source it came from. Surface
/// the result through [`Blueprint::provenance`].
///
/// Provenance is the difference from [`compile`]: pass [`Origin::file`] for
/// file-backed source and [`Origin::generated`] / [`Origin::generated_by`] for a
/// model-authored plan. Both still flow through the same gate.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Compile`] for the same semantic failures as
/// [`compile`].
pub fn compile_with_provenance(program: &Program, origin: Origin) -> Result<Vec<Blueprint>> {
    program
        .graphs
        .iter()
        .map(|graph| {
            let mut blueprint = compile_graph(graph)?;
            blueprint.provenance = Some(provenance_of(graph, &origin));
            Ok(blueprint)
        })
        .collect()
}

/// Builds the [`BlueprintProvenance`] for one graph declaration.
fn provenance_of(
    graph: &crate::language::types::GraphDecl,
    origin: &Origin,
) -> BlueprintProvenance {
    BlueprintProvenance {
        origin: origin.clone(),
        graph: graph.span,
        nodes: graph
            .nodes
            .iter()
            .map(|n| NamedSpan {
                name: n.name.clone(),
                span: n.span,
            })
            .collect(),
        channels: graph
            .channels
            .iter()
            .map(|c| NamedSpan {
                name: c.name.clone(),
                span: c.span,
            })
            .collect(),
        edges: graph
            .edges
            .iter()
            .map(|e| EdgeSpan {
                from: e.from.clone(),
                to: e.to.clone(),
                span: e.span,
            })
            .collect(),
    }
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
    "subagent",
    "repl_agent",
    "router",
    "interrupt",
    "join",
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
    /// Registered agent names (and aliases) a `subagent` node may reference.
    agents: HashSet<String>,
    /// Registered REPL script names (and aliases) a `repl_agent` node may
    /// reference.
    scripts: HashSet<String>,
    /// Allowed node kinds. When empty, node-kind validation is skipped (the
    /// legacy, manual behaviour); when non-empty, the strict binding path
    /// rejects any node whose kind is not listed.
    node_kinds: HashSet<String>,
}

/// The class of the primary, kind-specific reference a node carries.
///
/// This is the shared vocabulary of [`CapabilityResolver::classify_reference`],
/// the one policy that maps a node `kind` to the reference that must resolve and
/// the allowlist it resolves against. Every binding gate — the compiler's
/// [`CapabilityResolver::bind_blueprint`] and both
/// [`crate::language::resolver::Resolver`] paths — routes through it so they
/// cannot drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceClass {
    /// A chat model reference (the `_` default and `router`'s pre-classification).
    Model,
    /// A subgraph (graph blueprint) reference.
    Subgraph,
    /// A router-function reference.
    Router,
    /// A sub-agent reference.
    Agent,
    /// A REPL script reference.
    Script,
}

impl ReferenceClass {
    /// The lowercase noun used in "unknown {word}" diagnostics.
    pub fn word(self) -> &'static str {
        match self {
            ReferenceClass::Model => "model",
            ReferenceClass::Subgraph => "subgraph",
            ReferenceClass::Router => "router",
            ReferenceClass::Agent => "agent",
            ReferenceClass::Script => "script",
        }
    }
}

/// The primary reference a node carries, resolved by the shared policy.
#[derive(Clone, Copy, Debug)]
pub struct PrimaryReference<'a> {
    /// Which allowlist the reference must resolve against.
    pub class: ReferenceClass,
    /// The referenced name.
    pub target: &'a str,
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
            agents: collect(ComponentKind::Agent),
            scripts: collect(ComponentKind::Script),
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

    /// Allows an additional agent name (for `subagent` nodes). Returns `self`.
    pub fn allow_agent(mut self, name: impl Into<String>) -> Self {
        self.agents.insert(name.into());
        self
    }

    /// Allows an additional REPL script name (for `repl_agent` nodes). Returns
    /// `self`.
    pub fn allow_script(mut self, name: impl Into<String>) -> Self {
        self.scripts.insert(name.into());
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

    /// Returns true if `name` is an allowed agent (for `subagent` nodes).
    pub fn agent_allowed(&self, name: &str) -> bool {
        self.agents.contains(name)
    }

    /// Returns true if `name` is an allowed REPL script (for `repl_agent`
    /// nodes).
    pub fn script_allowed(&self, name: &str) -> bool {
        self.scripts.contains(name)
    }

    /// The single kind-to-reference policy every binding gate shares.
    ///
    /// Given a node `kind` and the reference fields it carries, returns the
    /// primary reference that must resolve and the allowlist class it resolves
    /// against — or `None` when the node declares no primary reference. The
    /// `subgraph` argument is the caller's already-resolved subgraph target
    /// (the dedicated graph field falling back to the legacy `model` field).
    ///
    /// Centralising this mapping is what keeps
    /// [`bind_blueprint`](Self::bind_blueprint) and both
    /// [`crate::language::resolver::Resolver`] paths from drifting: a new node
    /// kind or a changed reference convention is edited here once.
    pub fn classify_reference<'a>(
        kind: &str,
        model: Option<&'a str>,
        subgraph: Option<&'a str>,
        agent: Option<&'a str>,
        script: Option<&'a str>,
    ) -> Option<PrimaryReference<'a>> {
        let (class, target) = match kind {
            "subgraph" | "graph" => (ReferenceClass::Subgraph, subgraph?),
            "router" => (ReferenceClass::Router, model?),
            "subagent" => (ReferenceClass::Agent, agent?),
            "repl_agent" => (ReferenceClass::Script, script?),
            // Unknown kinds fall through to a model check, mirroring the
            // compiler default of an unspecified kind being `model`.
            _ => (ReferenceClass::Model, model?),
        };
        Some(PrimaryReference { class, target })
    }

    /// Returns true when `target` is allowed for the given reference `class`.
    pub fn reference_allowed(&self, class: ReferenceClass, target: &str) -> bool {
        match class {
            ReferenceClass::Model => self.model_allowed(target),
            ReferenceClass::Subgraph => self.subgraph_allowed(target),
            ReferenceClass::Router => self.router_allowed(target),
            ReferenceClass::Agent => self.agent_allowed(target),
            ReferenceClass::Script => self.script_allowed(target),
        }
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
    ///   `router` node references to a registered router, `subagent` node
    ///   references to a registered agent, `repl_agent` node references to a
    ///   registered script, and all other nodes' `model` references to a
    ///   registered model (via the shared [`classify_reference`](Self::classify_reference) policy);
    /// - every `channel` reducer reference is registered.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Compile`] for an unknown node kind, and
    /// [`TinyAgentsError::Capability`] for the first unregistered model, tool,
    /// subgraph, router, agent, script, or reducer reference.
    pub fn bind_blueprint(&self, blueprint: &Blueprint) -> Result<()> {
        for node in &blueprint.nodes {
            if !self.node_kind_allowed(&node.kind) {
                return Err(TinyAgentsError::Compile(format!(
                    "node `{}` has unknown kind `{}`",
                    node.name, node.kind
                )));
            }

            // Prefer the dedicated `graph "name"` reference, falling back to the
            // legacy `model` field for back-compatibility.
            let subgraph_target = node.subgraph.as_deref().or(node.model.as_deref());
            if let Some(reference) = Self::classify_reference(
                &node.kind,
                node.model.as_deref(),
                subgraph_target,
                node.agent.as_deref(),
                node.script.as_deref(),
            ) && !self.reference_allowed(reference.class, reference.target)
            {
                return Err(TinyAgentsError::Capability(format!(
                    "node `{}` references unknown {} `{}`",
                    node.name,
                    reference.class.word(),
                    reference.target
                )));
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
