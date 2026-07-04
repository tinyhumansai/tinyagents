//! Implementation of the [`CapabilityRegistry`] — the name-resolution engine
//! behind recursion.
//!
//! This is where a name like `"researcher"` or `"summarize"` becomes a real,
//! callable handle. By registering capabilities here and then handing the
//! registry to the language layer, a parent run lets a `.rag` blueprint or
//! `.ragsh` line spawn sub-models, sub-agents, and sub-graphs it never
//! hardcoded — while the registry's allowlist guarantees those references can
//! only resolve to capabilities a human actually registered.
//!
//! See [`types`] for the data definitions. This module provides registration,
//! lookup, aliasing, duplicate validation, and conveniences for handing the
//! catalog's models and tools to a harness ([`to_model_registry`] /
//! [`to_tool_registry`]) or to the `.rag`/`.ragsh` capability resolver
//! ([`capability_resolver`]).
//!
//! [`to_model_registry`]: CapabilityRegistry::to_model_registry
//! [`to_tool_registry`]: CapabilityRegistry::to_tool_registry
//! [`capability_resolver`]: CapabilityRegistry::capability_resolver

mod types;

use std::sync::Arc;

use crate::error::{Result, TinyAgentsError};
use crate::harness::model::{ChatModel, ModelRegistry};
use crate::harness::tool::{Tool, ToolRegistry};
use crate::language::Blueprint;
use crate::language::compiler::CapabilityResolver;
use crate::registry::component::{ComponentKind, ComponentMetadata};

pub use types::*;

impl<State: Send + Sync> CapabilityRegistry<State> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
            tools: std::collections::HashMap::new(),
            graphs: std::collections::HashMap::new(),
            agents: std::collections::HashMap::new(),
            meta: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Internal metadata bookkeeping
    // -----------------------------------------------------------------------

    /// Returns an error if `(kind, name)` is already registered.
    fn ensure_absent(&self, kind: ComponentKind, name: &str) -> Result<()> {
        if self.meta.contains_key(&(kind, name.to_owned())) {
            return Err(TinyAgentsError::DuplicateComponent(format!(
                "{kind} `{name}` is already registered"
            )));
        }
        Ok(())
    }

    /// Records default metadata for `(kind, name)` if no metadata exists yet.
    /// Replacing a value preserves any richer metadata already attached.
    fn record_meta(&mut self, kind: ComponentKind, name: &str) {
        self.meta
            .entry((kind, name.to_owned()))
            .or_insert_with(|| ComponentMetadata::new(name, kind));
    }

    // -----------------------------------------------------------------------
    // Registration: models
    // -----------------------------------------------------------------------

    /// Registers a model under `name`.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if a model is already
    /// registered under `name`. Use [`replace_model`](Self::replace_model) to
    /// overwrite intentionally.
    pub fn register_model(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
    ) -> Result<&mut Self> {
        let name = name.into();
        self.ensure_absent(ComponentKind::Model, &name)?;
        self.record_meta(ComponentKind::Model, &name);
        self.models.insert(name, model);
        Ok(self)
    }

    /// Registers or overwrites a model under `name`, preserving any existing
    /// metadata.
    pub fn replace_model(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
    ) -> &mut Self {
        let name = name.into();
        self.record_meta(ComponentKind::Model, &name);
        self.models.insert(name, model);
        self
    }

    // -----------------------------------------------------------------------
    // Registration: tools
    // -----------------------------------------------------------------------

    /// Registers a tool under its [`Tool::name`].
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if a tool with the same
    /// name is already registered. Use [`replace_tool`](Self::replace_tool) to
    /// overwrite intentionally.
    pub fn register_tool(&mut self, tool: Arc<dyn Tool<State>>) -> Result<&mut Self> {
        let name = tool.name().to_owned();
        self.ensure_absent(ComponentKind::Tool, &name)?;
        self.record_meta(ComponentKind::Tool, &name);
        self.tools.insert(name, tool);
        Ok(self)
    }

    /// Registers or overwrites a tool under its [`Tool::name`], preserving any
    /// existing metadata.
    pub fn replace_tool(&mut self, tool: Arc<dyn Tool<State>>) -> &mut Self {
        let name = tool.name().to_owned();
        self.record_meta(ComponentKind::Tool, &name);
        self.tools.insert(name, tool);
        self
    }

    // -----------------------------------------------------------------------
    // Registration: graph blueprints
    // -----------------------------------------------------------------------

    /// Registers a compiled graph [`Blueprint`] under `name`.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if a blueprint is already
    /// registered under `name`. Use
    /// [`replace_graph_blueprint`](Self::replace_graph_blueprint) to overwrite.
    pub fn register_graph_blueprint(
        &mut self,
        name: impl Into<String>,
        blueprint: Blueprint,
    ) -> Result<&mut Self> {
        let name = name.into();
        self.ensure_absent(ComponentKind::Graph, &name)?;
        self.record_meta(ComponentKind::Graph, &name);
        self.graphs.insert(name, blueprint);
        Ok(self)
    }

    /// Registers or overwrites a graph [`Blueprint`] under `name`, preserving
    /// any existing metadata.
    pub fn replace_graph_blueprint(
        &mut self,
        name: impl Into<String>,
        blueprint: Blueprint,
    ) -> &mut Self {
        let name = name.into();
        self.record_meta(ComponentKind::Graph, &name);
        self.graphs.insert(name, blueprint);
        self
    }

    // -----------------------------------------------------------------------
    // Registration: executable agents
    // -----------------------------------------------------------------------

    /// Registers an executable harness `agent` under its
    /// [`HarnessAgent::name`](crate::graph::subagent_node::HarnessAgent::name).
    ///
    /// Resolved by a
    /// [`SubAgentNode`](crate::graph::subagent_node::SubAgentNode) to delegate a
    /// graph step to the agent.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if an agent with the same
    /// name is already registered. Use [`replace_agent`](Self::replace_agent) to
    /// overwrite intentionally.
    pub fn register_agent(
        &mut self,
        agent: Arc<dyn crate::graph::subagent_node::HarnessAgent>,
    ) -> Result<&mut Self> {
        let name = agent.name().to_owned();
        self.ensure_absent(ComponentKind::Agent, &name)?;
        self.record_meta(ComponentKind::Agent, &name);
        self.agents.insert(name, agent);
        Ok(self)
    }

    /// Registers or overwrites an executable agent under its
    /// [`HarnessAgent::name`](crate::graph::subagent_node::HarnessAgent::name),
    /// preserving any existing metadata.
    pub fn replace_agent(
        &mut self,
        agent: Arc<dyn crate::graph::subagent_node::HarnessAgent>,
    ) -> &mut Self {
        let name = agent.name().to_owned();
        self.record_meta(ComponentKind::Agent, &name);
        self.agents.insert(name, agent);
        self
    }

    /// Looks up a registered executable agent by name or alias.
    pub fn agent(&self, name: &str) -> Option<Arc<dyn crate::graph::subagent_node::HarnessAgent>> {
        let canonical = self.resolve_name(ComponentKind::Agent, name)?;
        self.agents.get(&canonical).cloned()
    }

    // -----------------------------------------------------------------------
    // Registration: name-only descriptors (routers, reducers, stores)
    // -----------------------------------------------------------------------

    /// Registers a router (conditional-routing function) by name.
    ///
    /// Routers are name-only descriptors for now: the registry records that the
    /// name is an allowed router so `.rag` sources can bind to it, but the
    /// executable routing logic lives in Rust.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if already registered.
    pub fn register_router(&mut self, name: impl Into<String>) -> Result<&mut Self> {
        self.register_descriptor(ComponentKind::Router, name.into())
    }

    /// Registers a reducer (state-channel reducer) by name.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if already registered.
    pub fn register_reducer(&mut self, name: impl Into<String>) -> Result<&mut Self> {
        self.register_descriptor(ComponentKind::Reducer, name.into())
    }

    /// Registers a name-only descriptor of an arbitrary [`ComponentKind`]. This
    /// backs [`register_router`](Self::register_router) and
    /// [`register_reducer`](Self::register_reducer), and is the general
    /// public fallback for every other kind that has no dedicated typed
    /// registration method — [`ComponentKind::Store`],
    /// [`ComponentKind::Script`], [`ComponentKind::Middleware`],
    /// [`ComponentKind::Checkpointer`], [`ComponentKind::TaskStore`], and
    /// [`ComponentKind::Listener`]. [`ComponentKind::Model`],
    /// [`ComponentKind::Tool`], [`ComponentKind::Graph`], and
    /// [`ComponentKind::Agent`] have their own dedicated `register_*` methods
    /// instead.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if already registered.
    pub fn register_descriptor(
        &mut self,
        kind: ComponentKind,
        name: impl Into<String>,
    ) -> Result<&mut Self> {
        let name = name.into();
        self.ensure_absent(kind, &name)?;
        self.record_meta(kind, &name);
        Ok(self)
    }

    // -----------------------------------------------------------------------
    // Aliases
    // -----------------------------------------------------------------------

    /// Declares `alias` as an alternate name for `target` within `kind`.
    ///
    /// Subsequent lookups, [`has`](Self::has), and [`metadata`](Self::metadata)
    /// calls resolve the alias to `target`. The alias is also recorded on the
    /// target's [`ComponentMetadata::aliases`] list for discovery.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Capability`] if `target` is not a registered
    /// component of `kind`, and [`TinyAgentsError::DuplicateComponent`] if
    /// `alias` already names a registered component or an existing alias of that
    /// kind.
    pub fn alias(
        &mut self,
        kind: ComponentKind,
        alias: impl Into<String>,
        target: impl Into<String>,
    ) -> Result<&mut Self> {
        let alias = alias.into();
        let target = target.into();

        if !self.meta.contains_key(&(kind, target.clone())) {
            return Err(TinyAgentsError::Capability(format!(
                "cannot alias {kind} `{alias}` -> `{target}`: target is not registered"
            )));
        }
        if self.meta.contains_key(&(kind, alias.clone())) {
            return Err(TinyAgentsError::DuplicateComponent(format!(
                "{kind} `{alias}` is already a registered component"
            )));
        }
        if self.aliases.contains_key(&(kind, alias.clone())) {
            return Err(TinyAgentsError::DuplicateComponent(format!(
                "{kind} alias `{alias}` is already defined"
            )));
        }

        self.aliases.insert((kind, alias.clone()), target.clone());
        if let Some(meta) = self.meta.get_mut(&(kind, target))
            && !meta.aliases.contains(&alias)
        {
            meta.aliases.push(alias);
        }
        Ok(self)
    }

    /// Resolves `name` to a canonical registered name for `kind`, following one
    /// alias hop. Returns `None` when neither a direct registration nor an alias
    /// matches.
    pub fn resolve_name(&self, kind: ComponentKind, name: &str) -> Option<String> {
        if self.meta.contains_key(&(kind, name.to_owned())) {
            return Some(name.to_owned());
        }
        let target = self.aliases.get(&(kind, name.to_owned()))?;
        if self.meta.contains_key(&(kind, target.clone())) {
            Some(target.clone())
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Looks up a registered model by name or alias.
    pub fn model(&self, name: &str) -> Option<Arc<dyn ChatModel<State>>> {
        let canonical = self.resolve_name(ComponentKind::Model, name)?;
        self.models.get(&canonical).cloned()
    }

    /// Looks up a registered tool by name or alias.
    pub fn tool(&self, name: &str) -> Option<Arc<dyn Tool<State>>> {
        let canonical = self.resolve_name(ComponentKind::Tool, name)?;
        self.tools.get(&canonical).cloned()
    }

    /// Looks up a registered graph blueprint by name or alias.
    pub fn graph_blueprint(&self, name: &str) -> Option<&Blueprint> {
        let canonical = self.resolve_name(ComponentKind::Graph, name)?;
        self.graphs.get(&canonical)
    }

    /// Returns `true` when `name` (or an alias of it) is registered for `kind`.
    pub fn has(&self, kind: ComponentKind, name: &str) -> bool {
        self.resolve_name(kind, name).is_some()
    }

    /// Returns the canonical registered names for `kind`, in sorted order.
    /// Aliases are not included.
    pub fn names(&self, kind: ComponentKind) -> Vec<String> {
        let mut names: Vec<String> = self
            .meta
            .keys()
            .filter(|(k, _)| *k == kind)
            .map(|(_, name)| name.clone())
            .collect();
        names.sort();
        names
    }

    /// Returns the canonical registered names for `kind` *and* every alias of
    /// that kind, in sorted, de-duplicated order.
    ///
    /// This is the set of names declarative `.rag`/`.ragsh` source may reference
    /// for `kind`: both the canonical registration and any alias resolve to a
    /// real component, so both are valid references. It backs
    /// [`CapabilityResolver::from_registry`](crate::language::compiler::CapabilityResolver::from_registry).
    pub fn names_including_aliases(&self, kind: ComponentKind) -> Vec<String> {
        let mut names = self.names(kind);
        for (k, alias) in self.aliases.keys() {
            if *k == kind {
                names.push(alias.clone());
            }
        }
        names.sort();
        names.dedup();
        names
    }

    /// Returns the [`ComponentMetadata`] for `name` (or an alias) within `kind`.
    pub fn metadata(&self, kind: ComponentKind, name: &str) -> Option<&ComponentMetadata> {
        let canonical = self.resolve_name(kind, name)?;
        self.meta.get(&(kind, canonical))
    }

    // -----------------------------------------------------------------------
    // Handoff to harness / language layers
    // -----------------------------------------------------------------------

    /// Builds a harness [`ModelRegistry`] from the registered models, including
    /// alias names bound to the same model handle.
    ///
    /// The harness registry's default-model selection follows its own first-
    /// registered rule; since registration order here is unspecified, callers
    /// who need a specific default should set it explicitly on the result.
    pub fn to_model_registry(&self) -> ModelRegistry<State> {
        let mut registry = ModelRegistry::new();
        for (name, model) in &self.models {
            registry.register(name.clone(), model.clone());
        }
        for ((kind, alias), target) in &self.aliases {
            if *kind == ComponentKind::Model
                && let Some(model) = self.models.get(target)
            {
                registry.register(alias.clone(), model.clone());
            }
        }
        registry
    }

    /// Builds a harness [`ToolRegistry`] from the registered tools.
    ///
    /// The harness [`ToolRegistry`] keys tools by their own [`Tool::name`], so
    /// registry-level tool aliases are intentionally not propagated here: a tool
    /// is always invoked at runtime under its canonical schema name.
    pub fn to_tool_registry(&self) -> ToolRegistry<State> {
        let mut registry = ToolRegistry::new();
        for tool in self.tools.values() {
            registry.register(tool.clone());
        }
        registry
    }

    /// Builds a fully populated `.rag`/`.ragsh` [`CapabilityResolver`] from every
    /// registered capability — models, tools, graph blueprints, routers, and
    /// reducers, including their aliases — plus the default node kinds.
    ///
    /// This is the bridge the language layer uses: declarative source may only
    /// reference names that this registry has registered (or aliased), which is
    /// what makes agent-authored `.rag` safe to compile. The returned resolver
    /// is equivalent to [`CapabilityResolver::from_registry`] and enables the
    /// strict checks (subgraph/router/reducer references and node kinds) when
    /// used with [`CapabilityResolver::bind_blueprint`] or
    /// [`bind_capabilities_with_registry`](crate::language::compiler::bind_capabilities_with_registry).
    pub fn capability_resolver(&self) -> CapabilityResolver {
        CapabilityResolver::from_registry(self)
    }

    // -----------------------------------------------------------------------
    // Introspection / diagnostics
    // -----------------------------------------------------------------------

    /// Exports a serializable [`RegistrySnapshot`] of every registered
    /// component's metadata, sorted by `(kind, name)`.
    ///
    /// This is the machine-readable view a CLI or UI renders to show exactly
    /// what capabilities are active, and what an audit log records.
    pub fn snapshot(&self) -> crate::registry::diagnostics::RegistrySnapshot {
        let mut components: Vec<ComponentMetadata> = self.meta.values().cloned().collect();
        components.sort_by(|a, b| (a.kind, &a.id.0).cmp(&(b.kind, &b.id.0)));
        let mut aliases: Vec<crate::registry::diagnostics::AliasBinding> = self
            .aliases
            .iter()
            .map(
                |((kind, alias), canonical)| crate::registry::diagnostics::AliasBinding {
                    kind: *kind,
                    alias: alias.clone(),
                    canonical: canonical.clone(),
                },
            )
            .collect();
        aliases.sort_by(|a, b| (a.kind, &a.alias).cmp(&(b.kind, &b.alias)));
        crate::registry::diagnostics::RegistrySnapshot {
            components,
            aliases,
        }
    }

    /// Returns registry health diagnostics: aliases that shadow a registered
    /// component of the same name (warning) and aliases whose canonical target
    /// is not registered (error).
    pub fn diagnostics(&self) -> Vec<crate::registry::diagnostics::RegistryDiagnostic> {
        use crate::registry::diagnostics::{
            alias_shadows_component, dangling_alias, name_reused_across_kinds,
        };
        let mut out = Vec::new();
        for ((kind, alias), canonical) in &self.aliases {
            if self.meta.contains_key(&(*kind, alias.clone())) {
                out.push(alias_shadows_component(*kind, alias));
            }
            if !self.meta.contains_key(&(*kind, canonical.clone())) {
                out.push(dangling_alias(*kind, alias, canonical));
            }
        }
        // Surface names registered under more than one kind. Registration
        // rejects same-(kind, name) duplicates, but the same name across
        // different kinds is legal and worth flagging for audits.
        let mut kinds_by_name: std::collections::BTreeMap<&str, Vec<ComponentKind>> =
            std::collections::BTreeMap::new();
        for (kind, name) in self.meta.keys() {
            kinds_by_name.entry(name.as_str()).or_default().push(*kind);
        }
        for (name, mut kinds) in kinds_by_name {
            if kinds.len() > 1 {
                kinds.sort();
                out.push(name_reused_across_kinds(name, &kinds));
            }
        }
        out.sort_by(|a, b| (a.kind, &a.name).cmp(&(b.kind, &b.name)));
        out
    }
}

impl<State: Send + Sync> Default for CapabilityRegistry<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State: Send + Sync> std::fmt::Debug for CapabilityRegistry<State> {
    /// Renders the registered names per kind. Executable model/tool handles are
    /// opaque trait objects, so only their names appear.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("CapabilityRegistry");
        for kind in ComponentKind::ALL {
            dbg.field(kind.as_str(), &self.names(kind));
        }
        dbg.field("aliases", &self.aliases).finish()
    }
}

#[cfg(test)]
mod test;
