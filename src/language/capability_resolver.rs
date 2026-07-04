//! Capability binding: resolves and validates every capability reference
//! (model/tool/subgraph/router/reducer/agent/script) a [`Blueprint`] makes
//! against an allowlist, so declarative `.rag` source can only reach
//! capabilities Rust has already registered and allowed.
//!
//! Lives beside [`crate::language::resolver`] (the spanned-diagnostic
//! resolution path) since both implement the same binding policy;
//! [`crate::language::resolver::Resolver`] wraps a [`CapabilityResolver`]
//! internally. Split out of `compiler.rs`; see that module's doc comment
//! for how binding fits into the overall compile pipeline.

use std::collections::HashSet;

use crate::error::{Result, TinyAgentsError};
use crate::language::types::Blueprint;
use crate::registry::CapabilityRegistry;
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
