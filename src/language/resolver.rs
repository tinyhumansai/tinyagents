//! Registry-backed reference resolution for the expressive language (`.rag`).
//!
//! The [`Resolver`] is the single binding gate every `.rag` plan passes through
//! before it can run — whether the source was hand-written or emitted by a model
//! standing inside the harness. It walks a parsed [`Program`] (or a compiled
//! [`Blueprint`]) and binds **every** reference — models, tools, agents,
//! subgraphs, route functions, reducers, and node kinds — by name against a live
//! [`CapabilityRegistry`]. A reference may only resolve to a capability that Rust
//! has already registered and allowed; anything unknown or disallowed is
//! reported as a [`Diagnostic`] pointing at the offending source span with a
//! clear "not registered / not allowed" message.
//!
//! This makes recursive self-authoring safe: a generated topology cannot smuggle
//! in a capability the host never sanctioned, because the *same* registry-derived
//! allowlists validate generated and file-backed source alike. No path lowers a
//! plan into the runtime without first clearing this gate.
//!
//! Two faces, one policy:
//!
//! - [`Resolver::resolve_program`] resolves the AST and collects a spanned
//!   [`Diagnostic`] for every offending reference (rich, source-aware errors).
//!   [`Resolver::check_program`] folds the first diagnostic into a
//!   [`TinyAgentsError`].
//! - [`Resolver::resolve_blueprint`] resolves a compiled [`Blueprint`] that no
//!   longer carries spans, returning the same [`TinyAgentsError`] variants and
//!   messages as the legacy [`crate::language::compiler::CapabilityResolver`]
//!   blueprint gate.
//!
//! [`resolve_source`] is the recommended façade: it parses, resolves against the
//! registry with full source spans, and lowers to validated blueprints in one
//! call, so model-generated source is validated on exactly the same path as a
//! checked-in `.rag` file.

use std::collections::HashSet;

use crate::error::{Result, TinyAgentsError};
use crate::language::ast::{ChannelDecl, GraphDecl, NodeDecl, Program};
use crate::language::compiler::{CapabilityResolver, DEFAULT_NODE_KINDS, compile};
use crate::language::diagnostic::Diagnostic;
use crate::language::parser::parse_str;
use crate::language::source::SourceFile;
use crate::language::span::Span;
use crate::language::types::Blueprint;
use crate::registry::CapabilityRegistry;

// Stable diagnostic codes for resolution failures.
const CODE_UNKNOWN_MODEL: &str = "E-rag-unknown-model";
const CODE_UNKNOWN_TOOL: &str = "E-rag-unknown-tool";
const CODE_UNKNOWN_SUBGRAPH: &str = "E-rag-unknown-subgraph";
const CODE_UNKNOWN_ROUTER: &str = "E-rag-unknown-router";
const CODE_UNKNOWN_AGENT: &str = "E-rag-unknown-agent";
const CODE_UNKNOWN_REDUCER: &str = "E-rag-unknown-reducer";
const CODE_INVALID_NODE_KIND: &str = "E-rag-invalid-node-kind";

/// The single registry-backed binding gate for `.rag` source.
///
/// A `Resolver` holds the set of capability names the host has registered and
/// allowed, keyed by kind. It is built from a live [`CapabilityRegistry`] with
/// [`Resolver::from_registry`] (or from an existing
/// [`CapabilityResolver`]/allowlist via [`Resolver::from_capabilities`]), then
/// asked to resolve a [`Program`] or [`Blueprint`]. Resolution never mutates the
/// plan; it only reports references that fall outside the allowlists.
#[derive(Clone, Debug)]
pub struct Resolver {
    /// The overlapping model/tool/subgraph/router/reducer/node-kind allowlists,
    /// reused from the compiler's [`CapabilityResolver`] so the two share one
    /// policy.
    caps: CapabilityResolver,
    /// Registered agent names (and aliases) a `subagent` node may reference.
    agents: HashSet<String>,
}

impl Resolver {
    /// Builds a resolver from a live [`CapabilityRegistry`].
    ///
    /// Every registered model, tool, graph blueprint, router, reducer, and agent
    /// name — including aliases — populates the corresponding allowlist, and the
    /// node-kind allowlist is seeded with [`DEFAULT_NODE_KINDS`]. The resolver
    /// therefore validates `.rag` source against exactly what Rust has
    /// registered.
    pub fn from_registry<State: Send + Sync>(registry: &CapabilityRegistry<State>) -> Self {
        use crate::registry::ComponentKind;
        let agents = registry
            .names_including_aliases(ComponentKind::Agent)
            .into_iter()
            .collect();
        Self {
            caps: CapabilityResolver::from_registry(registry),
            agents,
        }
    }

    /// Builds a resolver from an existing [`CapabilityResolver`] allowlist, with
    /// no extra agent names. Node-kind validation follows the supplied
    /// resolver's configuration.
    pub fn from_capabilities(caps: CapabilityResolver) -> Self {
        Self {
            caps,
            agents: HashSet::new(),
        }
    }

    /// Returns the underlying capability allowlist.
    pub fn capabilities(&self) -> &CapabilityResolver {
        &self.caps
    }

    /// Allows an additional agent name. Returns `self` for chaining.
    pub fn allow_agent(mut self, name: impl Into<String>) -> Self {
        self.agents.insert(name.into());
        self
    }

    /// Returns true if `name` is a registered/allowed agent.
    pub fn agent_allowed(&self, name: &str) -> bool {
        self.agents.contains(name)
    }

    /// Resolves every reference in `program` against the allowlists, returning a
    /// spanned [`Diagnostic`] for each offending reference (in source order).
    ///
    /// An empty result means every model, tool, agent, subgraph, router, and
    /// reducer reference is registered and every node kind is allowed. Unlike the
    /// fail-fast [`check_program`](Self::check_program), this collects *all*
    /// problems so a caller can surface them together.
    pub fn resolve_program(&self, program: &Program) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        for graph in &program.graphs {
            self.resolve_graph(graph, &mut diagnostics);
        }
        diagnostics
    }

    fn resolve_graph(&self, graph: &GraphDecl, out: &mut Vec<Diagnostic>) {
        for node in &graph.nodes {
            self.resolve_node(node, out);
        }
        for channel in &graph.channels {
            self.resolve_channel(channel, out);
        }
    }

    fn resolve_node(&self, node: &NodeDecl, out: &mut Vec<Diagnostic>) {
        let kind = node.kind.as_deref().unwrap_or("model");

        // 1. Node kind must be allowlisted.
        if !self.caps.node_kind_allowed(kind) {
            out.push(
                Diagnostic::error(
                    format!("node `{}` has unknown kind `{kind}`", node.name),
                    node.span,
                )
                .with_code(CODE_INVALID_NODE_KIND)
                .with_primary_label("not an allowed node kind")
                .with_help(format!("allowed kinds: {}", DEFAULT_NODE_KINDS.join(", "))),
            );
            // The kind drives which reference is checked below; an unknown kind
            // falls through to a model check, which is the compiler default.
        }

        // 2. The kind-specific primary reference.
        match kind {
            "subgraph" | "graph" => {
                if let Some(target) = node.graph.as_deref().or(node.model.as_deref()) {
                    self.check_ref(
                        self.caps.subgraph_allowed(target),
                        &node.name,
                        "subgraph",
                        target,
                        node.span,
                        CODE_UNKNOWN_SUBGRAPH,
                        out,
                    );
                }
            }
            "router" => {
                if let Some(target) = node.model.as_deref() {
                    self.check_ref(
                        self.caps.router_allowed(target),
                        &node.name,
                        "router",
                        target,
                        node.span,
                        CODE_UNKNOWN_ROUTER,
                        out,
                    );
                }
            }
            "subagent" => {
                if let Some(target) = node.agent.as_deref() {
                    self.check_ref(
                        self.agent_allowed(target),
                        &node.name,
                        "agent",
                        target,
                        node.span,
                        CODE_UNKNOWN_AGENT,
                        out,
                    );
                }
            }
            _ => {
                if let Some(target) = node.model.as_deref() {
                    self.check_ref(
                        self.caps.model_allowed(target),
                        &node.name,
                        "model",
                        target,
                        node.span,
                        CODE_UNKNOWN_MODEL,
                        out,
                    );
                }
            }
        }

        // 3. Every referenced tool must be registered.
        for tool in &node.tools {
            self.check_ref(
                self.caps.tool_allowed(tool),
                &node.name,
                "tool",
                tool,
                node.span,
                CODE_UNKNOWN_TOOL,
                out,
            );
        }
    }

    fn resolve_channel(&self, channel: &ChannelDecl, out: &mut Vec<Diagnostic>) {
        if !self.caps.reducer_allowed(&channel.reducer) {
            out.push(
                Diagnostic::error(
                    format!(
                        "channel `{}` references unknown reducer `{}`",
                        channel.name, channel.reducer
                    ),
                    channel.span,
                )
                .with_code(CODE_UNKNOWN_REDUCER)
                .with_primary_label("reducer not registered or not allowed")
                .with_help("register the reducer before referencing it from `.rag`"),
            );
        }
    }

    /// Pushes a "not registered / not allowed" diagnostic when `allowed` is
    /// false. `what` is the reference kind word (`model`, `tool`, …).
    #[allow(clippy::too_many_arguments)]
    fn check_ref(
        &self,
        allowed: bool,
        node: &str,
        what: &str,
        target: &str,
        span: Span,
        code: &str,
        out: &mut Vec<Diagnostic>,
    ) {
        if allowed {
            return;
        }
        out.push(
            Diagnostic::error(
                format!("node `{node}` references unknown {what} `{target}`"),
                span,
            )
            .with_code(code)
            .with_primary_label(format!("{what} not registered or not allowed"))
            .with_help(format!(
                "register `{target}` as a {what} before referencing it from `.rag`"
            )),
        );
    }

    /// Resolves `program` and folds the first diagnostic into a
    /// [`TinyAgentsError`].
    ///
    /// When `source` is provided the error message is the caret-underline
    /// rendering of the diagnostic; otherwise it is the source-free rendering.
    /// An unknown node kind folds into [`TinyAgentsError::Compile`] (mirroring
    /// the compiler's node-kind gate); every other unresolved reference folds
    /// into [`TinyAgentsError::Capability`].
    ///
    /// # Errors
    ///
    /// Returns the first resolution failure, or `Ok(())` if every reference
    /// resolves.
    pub fn check_program(&self, program: &Program, source: Option<&SourceFile>) -> Result<()> {
        match self.resolve_program(program).into_iter().next() {
            Some(diagnostic) => Err(fold_diagnostic(diagnostic, source)),
            None => Ok(()),
        }
    }

    /// Resolves a compiled [`Blueprint`] that no longer carries source spans.
    ///
    /// This is the span-less counterpart to [`resolve_program`](Self::resolve_program):
    /// it returns the same [`TinyAgentsError`] variants and messages as the
    /// legacy [`CapabilityResolver::bind_blueprint`] gate — [`TinyAgentsError::Compile`]
    /// for an unknown node kind, [`TinyAgentsError::Capability`] for the first
    /// unregistered model, tool, agent, subgraph, router, or reducer — extended
    /// with the agent reference check.
    ///
    /// # Errors
    ///
    /// Returns the first resolution failure.
    pub fn resolve_blueprint(&self, blueprint: &Blueprint) -> Result<()> {
        for node in &blueprint.nodes {
            if !self.caps.node_kind_allowed(&node.kind) {
                return Err(TinyAgentsError::Compile(format!(
                    "node `{}` has unknown kind `{}`",
                    node.name, node.kind
                )));
            }
            match node.kind.as_str() {
                "subgraph" | "graph" => {
                    if let Some(target) = node.subgraph.as_deref().or(node.model.as_deref())
                        && !self.caps.subgraph_allowed(target)
                    {
                        return Err(unregistered("subgraph", &node.name, target));
                    }
                }
                "router" => {
                    if let Some(target) = node.model.as_deref()
                        && !self.caps.router_allowed(target)
                    {
                        return Err(unregistered("router", &node.name, target));
                    }
                }
                "subagent" => {
                    if let Some(target) = node.agent.as_deref()
                        && !self.agent_allowed(target)
                    {
                        return Err(unregistered("agent", &node.name, target));
                    }
                }
                _ => {
                    if let Some(target) = node.model.as_deref()
                        && !self.caps.model_allowed(target)
                    {
                        return Err(unregistered("model", &node.name, target));
                    }
                }
            }
            for tool in &node.tools {
                if !self.caps.tool_allowed(tool) {
                    return Err(unregistered("tool", &node.name, tool));
                }
            }
        }
        for channel in &blueprint.channels {
            if !self.caps.reducer_allowed(&channel.reducer) {
                return Err(TinyAgentsError::Capability(format!(
                    "channel `{}` references unknown reducer `{}`",
                    channel.name, channel.reducer
                )));
            }
        }
        Ok(())
    }
}

/// Folds a resolution diagnostic into the appropriate [`TinyAgentsError`].
///
/// The rendered diagnostic (with a caret when `source` is present) becomes the
/// error payload, so callers keep the spanned presentation. An invalid node kind
/// maps to [`TinyAgentsError::Compile`]; every other code maps to
/// [`TinyAgentsError::Capability`].
fn fold_diagnostic(diagnostic: Diagnostic, source: Option<&SourceFile>) -> TinyAgentsError {
    let is_kind = diagnostic.code.as_deref() == Some(CODE_INVALID_NODE_KIND);
    let rendered = match source {
        Some(file) => diagnostic.render(file),
        None => diagnostic.render_plain(),
    };
    if is_kind {
        TinyAgentsError::Compile(rendered)
    } else {
        TinyAgentsError::Capability(rendered)
    }
}

/// Builds the span-less "unknown {what}" [`TinyAgentsError::Capability`] used by
/// [`Resolver::resolve_blueprint`].
fn unregistered(what: &str, node: &str, target: &str) -> TinyAgentsError {
    TinyAgentsError::Capability(format!(
        "node `{node}` references unknown {what} `{target}`"
    ))
}

/// Parses, registry-resolves (with full source spans), and lowers `.rag`/`.ragsh`
/// `source` into validated blueprints in one call.
///
/// This is the recommended single entry point: it routes generated and
/// file-backed source through exactly the same [`Resolver`] gate, so no topology
/// reaches the runtime without binding every reference against `registry`. A
/// resolution failure carries the caret-underline rendering of the first
/// offending reference.
///
/// # Errors
///
/// Propagates [`TinyAgentsError::Parse`] from the parser,
/// [`TinyAgentsError::Capability`]/[`TinyAgentsError::Compile`] from resolution,
/// and any [`TinyAgentsError::Compile`] from lowering.
pub fn resolve_source<State: Send + Sync>(
    source: &str,
    registry: &CapabilityRegistry<State>,
) -> Result<Vec<Blueprint>> {
    let program = parse_str(source)?;
    let file = SourceFile::anonymous(source);
    Resolver::from_registry(registry).check_program(&program, Some(&file))?;
    compile(&program)
}
