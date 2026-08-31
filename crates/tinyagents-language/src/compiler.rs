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

use crate::capability_resolver::{CapabilitySource, bind_capabilities_with_registry};
use crate::parser::parse_str;
use crate::types::{
    Blueprint, BlueprintProvenance, ChannelSpec, CommandSpec, END, EdgeSpan, EdgeSpec, IoFieldSpec,
    JoinSpec, NamedSpan, NodeSpec, Origin, Program, Routing, SendSpec,
};
use std::collections::{HashMap, HashSet};
use tinyagents_harness::error::{Result, TinyAgentsError};

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
    reject_duplicate_graphs(program)?;
    program.graphs.iter().map(compile_graph).collect()
}

/// Rejects a [`Program`] that declares the same graph name more than once.
///
/// Shared by [`compile`] and [`compile_with_provenance`] so both entry points
/// enforce the same semantic guard — two blueprints sharing one `graph_id`
/// would otherwise collide when registered (`DuplicateComponent` mid-way
/// through, or a silent overwrite via `replace_graph_blueprint`).
fn reject_duplicate_graphs(program: &Program) -> Result<()> {
    let mut graph_ids: HashSet<&str> = HashSet::new();
    for graph in &program.graphs {
        if !graph_ids.insert(graph.name.as_str()) {
            return Err(TinyAgentsError::Compile(format!(
                "duplicate graph `{}`",
                graph.name
            )));
        }
    }
    Ok(())
}

fn compile_graph(graph: &crate::types::GraphDecl) -> Result<Blueprint> {
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

    // Group top-level edges by source node (targets in declaration order) so
    // the multi-edge check and per-node routing lookups below are O(1) per
    // node instead of a linear rescan of `graph.edges` each time.
    let mut edge_targets: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &graph.edges {
        edge_targets
            .entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
    }

    // Reject contradictory multiple top-level edges from the same source: only
    // one static successor can ever be routed to, so a second edge from the
    // same node is silent data loss (the first edge wins, the rest are dead
    // weight in `blueprint.edges`) rather than a legitimate multi-successor.
    // Walking `graph.edges` in declaration order keeps the reported node (the
    // first source that repeats) deterministic.
    let mut seen_edge_sources: HashSet<&str> = HashSet::new();
    for edge in &graph.edges {
        let name = edge.from.as_str();
        if !seen_edge_sources.insert(name) {
            let targets = edge_targets
                .get(name)
                .map(|targets| targets.join(", "))
                .unwrap_or_default();
            return Err(compile_err(format!(
                "node `{name}` has multiple top-level edges ({targets}); a node may declare at most one outgoing edge",
            )));
        }
    }

    // 4. Validate and lower each node.
    let mut nodes = Vec::new();
    for node in &graph.nodes {
        let has_routes = !node.routes.is_empty();
        let has_next = node.next.is_some();
        let has_static_edge = edge_targets.contains_key(node.name.as_str());
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

        // Reject a `steering { … }` declaration rather than dropping it on the
        // floor. The grammar accepts the documented shape (parser.rs), but
        // there is no faithful lowering onto the runtime today:
        // `harness::steering::SteeringPolicy` is a single flat allowlist of
        // `SteeringCommandKind`s with no `parent`/`human` actor separation, no
        // delivery policy, and no `add_instruction` / `request_status` kinds.
        // Any partial lowering would have to widen or invent semantics, and a
        // silent no-op is worse still: an operator would deploy a blueprint
        // believing the declared restrictions are enforced while the runtime
        // receives none. Fail loudly until declarative steering is implemented
        // end to end.
        if node.steering.is_some() {
            return Err(compile_err(format!(
                "node `{}` declares a `steering {{ … }}` block, but declarative steering is parsed and not yet enforced: \
the runtime policy (`harness::steering::SteeringPolicy`) is one flat command allowlist with no `parent`/`human` actor \
separation, no delivery policy, and no `add_instruction`/`request_status` commands, so the declaration cannot be lowered \
faithfully. Remove the block and apply a `SteeringPolicy` from the Rust `NodeFactory` that builds this node until \
declarative steering lowering lands.",
                node.name
            )));
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
        } else if let Some(to) = edge_targets
            .get(node.name.as_str())
            .and_then(|targets| targets.first())
        {
            // First declared edge wins, matching the previous linear `find`
            // (after the multi-edge check above there is exactly one).
            if *to == END {
                Routing::Terminal
            } else {
                Routing::Next((*to).to_string())
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
/// records the source [`Span`](crate::span::Span) of every node,
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
    reject_duplicate_graphs(program)?;
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
fn provenance_of(graph: &crate::types::GraphDecl, origin: &Origin) -> BlueprintProvenance {
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

/// Parses, compiles, and registry-binds `.rag` `source` in one call.
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
pub fn compile_source(source: &str, registry: &impl CapabilitySource) -> Result<Vec<Blueprint>> {
    let program = parse_str(source)?;
    let blueprints = compile(&program)?;
    for blueprint in &blueprints {
        bind_capabilities_with_registry(blueprint, registry)?;
    }
    Ok(blueprints)
}
