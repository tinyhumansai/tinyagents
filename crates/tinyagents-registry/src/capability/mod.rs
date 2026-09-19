//! Implementation of the [`CapabilityRegistry`] — the name-resolution engine
//! behind recursion.
//!
//! This is where a name like `"researcher"` or `"summarize"` becomes a real,
//! callable handle. By registering capabilities here and then handing the
//! registry to the language layer, a parent run lets a `.rag` blueprint or a
//! host session spawn sub-models, sub-agents, and sub-graphs it never
//! hardcoded — while the registry's allowlist guarantees those references can
//! only resolve to capabilities a human actually registered.
//!
//! See `types` for the data definitions. This module provides registration,
//! lookup, aliasing, duplicate validation, and conveniences for handing the
//! catalog's models and tools to a harness ([`to_model_registry`] /
//! [`to_tool_registry`]) or to the `.rag` capability resolver
//! ([`capability_resolver`]).
//!
//! [`to_model_registry`]: CapabilityRegistry::to_model_registry
//! [`to_tool_registry`]: CapabilityRegistry::to_tool_registry
//! [`capability_resolver`]: CapabilityRegistry::capability_resolver

mod types;

use std::sync::Arc;

use crate::component::{ComponentKind, ComponentMetadata};
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::model_registry::ModelRegistry;
use tinyagents_harness::tool::ToolRegistry;
use tinyagents_language::Blueprint;
use tinyagents_language::capability_resolver::CapabilityResolver;
use tinyinference_llm::model::ChatModel;
use tinytools::Tool;

pub use types::*;

impl<State: Send + Sync> CapabilityRegistry<State> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
            model_order: Vec::new(),
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

    /// Replaces the [`ComponentMetadata`] recorded for `(kind, name)`.
    ///
    /// Unlike [`record_meta`](Self::record_meta) (which only fills in a
    /// default the first time a name is registered), this overwrites whatever
    /// metadata is already there — so `with_description`/`with_tag` builders
    /// (`crate::component::ComponentMetadata`) actually reach a registered
    /// component instead of being dead on arrival for anything registered
    /// through `register_*`/`replace_*` (see W-I8 in
    /// `docs/runtime-comparison/code-review-workspace.md`).
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Capability`] if `(kind, name)` is not a
    /// registered component: setting metadata on a name nothing registered
    /// would create a "component" with metadata but no backing value.
    pub fn set_metadata(
        &mut self,
        kind: ComponentKind,
        name: &str,
        metadata: ComponentMetadata,
    ) -> Result<&mut Self> {
        if !self.meta.contains_key(&(kind, name.to_owned())) {
            return Err(TinyAgentsError::Capability(format!(
                "cannot set metadata for {kind} `{name}`: not registered"
            )));
        }
        self.meta.insert((kind, name.to_owned()), metadata);
        Ok(self)
    }

    /// Removes a registered component (and its metadata) by `(kind, name)`.
    ///
    /// Removing a component that other names alias makes those aliases
    /// dangling, which [`Self::diagnostics`]'s `dangling_alias` check then
    /// reports — this is the operation that makes that diagnostic reachable
    /// through the public API (see W-I8). Aliases of `name` are left in place
    /// (not cascaded), matching [`Self::alias`]'s "one alias hop" model:
    /// callers that want a clean removal should also drop the alias entries
    /// they know about.
    ///
    /// Returns `true` if a component was present and removed, `false` if
    /// `(kind, name)` was not registered (a no-op, not an error).
    pub fn remove(&mut self, kind: ComponentKind, name: &str) -> bool {
        let key = (kind, name.to_owned());
        if self.meta.remove(&key).is_none() {
            return false;
        }
        match kind {
            ComponentKind::Model => {
                self.models.remove(name);
                self.model_order.retain(|n| n != name);
            }
            ComponentKind::Tool => {
                self.tools.remove(name);
            }
            ComponentKind::Graph => {
                self.graphs.remove(name);
            }
            ComponentKind::Agent => {
                self.agents.remove(name);
            }
            _ => {
                // Router/Reducer/Store/Script/Middleware/Checkpointer/
                // TaskStore/Listener are name-only descriptors: `meta`
                // removal above is the whole registration.
            }
        }
        true
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
        self.remember_model_order(&name);
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
        self.remember_model_order(&name);
        self.models.insert(name, model);
        self
    }

    /// Appends `name` to [`Self::model_order`] the first time it is
    /// registered. Re-registering an existing name (via
    /// [`replace_model`](Self::replace_model)) keeps its original position.
    fn remember_model_order(&mut self, name: &str) {
        if !self.models.contains_key(name) {
            self.model_order.push(name.to_owned());
        }
    }

    /// Registers a model under `name` with explicit [`ComponentMetadata`]
    /// instead of the bare default [`record_meta`](Self::record_meta) would
    /// attach, so a description/tags are attached atomically with
    /// registration rather than needing a follow-up
    /// [`set_metadata`](Self::set_metadata) call.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if a model is already
    /// registered under `name`.
    pub fn register_model_with(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
        metadata: ComponentMetadata,
    ) -> Result<&mut Self> {
        let name = name.into();
        self.ensure_absent(ComponentKind::Model, &name)?;
        self.meta
            .insert((ComponentKind::Model, name.clone()), metadata);
        self.remember_model_order(&name);
        self.models.insert(name, model);
        Ok(self)
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
    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) -> Result<&mut Self> {
        let name = tool.name().to_owned();
        self.ensure_absent(ComponentKind::Tool, &name)?;
        self.record_meta(ComponentKind::Tool, &name);
        self.tools.insert(name, tool);
        Ok(self)
    }

    /// Registers a tool under its [`Tool::name`] with explicit
    /// [`ComponentMetadata`], atomically instead of a follow-up
    /// [`set_metadata`](Self::set_metadata) call.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if a tool with the same
    /// name is already registered.
    pub fn register_tool_with(
        &mut self,
        tool: Arc<dyn Tool>,
        metadata: ComponentMetadata,
    ) -> Result<&mut Self> {
        let name = tool.name().to_owned();
        self.ensure_absent(ComponentKind::Tool, &name)?;
        self.meta
            .insert((ComponentKind::Tool, name.clone()), metadata);
        self.tools.insert(name, tool);
        Ok(self)
    }

    /// Registers or overwrites a tool under its [`Tool::name`], preserving any
    /// existing metadata.
    pub fn replace_tool(&mut self, tool: Arc<dyn Tool>) -> &mut Self {
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

    /// Registers a declarative agent definition under its stable id.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if an agent with the same
    /// name is already registered. Use [`replace_agent`](Self::replace_agent) to
    /// overwrite intentionally.
    pub fn register_agent(
        &mut self,
        agent: tinyagents_definition::AgentDefinition,
    ) -> Result<&mut Self> {
        let name = agent.id.clone();
        self.ensure_absent(ComponentKind::Agent, &name)?;
        self.record_meta(ComponentKind::Agent, &name);
        self.agents.insert(name, agent);
        Ok(self)
    }

    /// Registers or overwrites a declarative agent definition, preserving any
    /// existing metadata.
    pub fn replace_agent(&mut self, agent: tinyagents_definition::AgentDefinition) -> &mut Self {
        let name = agent.id.clone();
        self.record_meta(ComponentKind::Agent, &name);
        self.agents.insert(name, agent);
        self
    }

    /// Looks up a registered declarative agent definition by name or alias.
    pub fn agent(&self, name: &str) -> Option<&tinyagents_definition::AgentDefinition> {
        let canonical = self.resolve_name(ComponentKind::Agent, name)?;
        self.agents.get(&canonical)
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
    pub fn tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
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
    /// This is the set of names declarative `.rag` source may reference
    /// for `kind`: both the canonical registration and any alias resolve to a
    /// real component, so both are valid references. It backs
    /// [`CapabilityResolver::from_registry`].
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
    /// Models are registered onto the result in [`Self::model_order`] — the
    /// order `register_model`/`replace_model` first saw each name in — so the
    /// harness registry's "first-registered model becomes the default" rule
    /// ([`ModelRegistry::register`]) is deterministic and reproducible across
    /// runs, rather than following `HashMap` iteration order. That default is
    /// still whichever model happened to be registered first; callers who
    /// need a specific default regardless of registration order should use
    /// [`Self::to_model_registry_with_default`] or call `set_default`
    /// explicitly on the result.
    pub fn to_model_registry(&self) -> ModelRegistry<State> {
        let mut registry = ModelRegistry::new();
        for name in &self.model_order {
            if let Some(model) = self.models.get(name) {
                registry.register(name.clone(), model.clone());
            }
        }
        let mut aliases: Vec<(&String, &String)> = self
            .aliases
            .iter()
            .filter(|((kind, _), _)| *kind == ComponentKind::Model)
            .map(|((_, alias), target)| (alias, target))
            .collect();
        aliases.sort();
        for (alias, target) in aliases {
            if let Some(model) = self.models.get(target) {
                registry.register(alias.clone(), model.clone());
            }
        }
        registry
    }

    /// Builds a harness [`ModelRegistry`] exactly like [`Self::to_model_registry`],
    /// but with the default model explicitly set to `name` instead of
    /// whichever model was registered first.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::ModelNotFound`] if `name` (or an alias of
    /// it) is not a registered model.
    pub fn to_model_registry_with_default(&self, name: &str) -> Result<ModelRegistry<State>> {
        if self.model(name).is_none() {
            return Err(TinyAgentsError::ModelNotFound(name.to_string()));
        }
        let mut registry = self.to_model_registry();
        registry.set_default(name);
        Ok(registry)
    }

    /// Builds a harness [`ToolRegistry`] from the registered tools.
    ///
    /// The harness [`ToolRegistry`] keys tools by their own [`Tool::name`], so
    /// registry-level tool aliases are intentionally not propagated here: a tool
    /// is always invoked at runtime under its canonical schema name.
    pub fn to_tool_registry<Ctx: Send + Sync>(&self) -> ToolRegistry<State, Ctx> {
        let mut registry = ToolRegistry::new();
        for tool in self.tools.values() {
            registry.register(tool.clone());
        }
        registry
    }

    /// Builds a fully populated `.rag` [`CapabilityResolver`] from every
    /// registered capability — models, tools, graph blueprints, routers, and
    /// reducers, including their aliases — plus the default node kinds.
    ///
    /// This is the bridge the language layer uses: declarative source may only
    /// reference names that this registry has registered (or aliased), which is
    /// what makes agent-authored `.rag` safe to compile. The returned resolver
    /// is equivalent to [`CapabilityResolver::from_registry`] and enables the
    /// strict checks (subgraph/router/reducer references and node kinds) when
    /// used with [`CapabilityResolver::bind_blueprint`] or
    /// [`bind_capabilities_with_registry`](tinyagents_language::bind_capabilities_with_registry).
    pub fn capability_resolver(&self) -> CapabilityResolver {
        CapabilityResolver::from_registry(self)
    }

    // -----------------------------------------------------------------------
    // Introspection / diagnostics
    // -----------------------------------------------------------------------

    /// Exports a serializable [`RegistrySnapshot`](crate::RegistrySnapshot) of every registered
    /// component's metadata, sorted by `(kind, name)`.
    ///
    /// This is the machine-readable view a CLI or UI renders to show exactly
    /// what capabilities are active, and what an audit log records.
    pub fn snapshot(&self) -> crate::diagnostics::RegistrySnapshot {
        let mut components: Vec<ComponentMetadata> = self.meta.values().cloned().collect();
        components.sort_by(|a, b| (a.kind, &a.id.0).cmp(&(b.kind, &b.id.0)));
        let mut aliases: Vec<crate::diagnostics::AliasBinding> = self
            .aliases
            .iter()
            .map(
                |((kind, alias), canonical)| crate::diagnostics::AliasBinding {
                    kind: *kind,
                    alias: alias.clone(),
                    canonical: canonical.clone(),
                },
            )
            .collect();
        aliases.sort_by(|a, b| (a.kind, &a.alias).cmp(&(b.kind, &b.alias)));
        crate::diagnostics::RegistrySnapshot {
            components,
            aliases,
        }
    }

    /// Returns registry health diagnostics: aliases that shadow a registered
    /// component of the same name (warning) and aliases whose canonical target
    /// is not registered (error).
    pub fn diagnostics(&self) -> Vec<crate::diagnostics::RegistryDiagnostic> {
        use crate::diagnostics::{
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

// ===========================================================================
// tinyagents_definition::DefinitionRegistry bridge
// ===========================================================================
//
// `HostCapabilities.definitions: Arc<dyn DefinitionRegistry>` is a required
// async host capability, while `CapabilityRegistry::register_agent` stores
// the same `AgentDefinition` synchronously — before this bridge, nothing
// implemented `DefinitionRegistry` for `CapabilityRegistry`, so a host that
// registered agents in the registry had to build a second, separately
// populated `InMemoryDefinitionRegistry` by hand (see W-I9 in
// `docs/runtime-comparison/code-review-workspace.md`).
//
// This is written out by hand, matching the exact signature the
// `#[async_trait]` macro in `tinyagents-definition` expands
// `DefinitionRegistry`'s methods to, instead of applying `#[async_trait]`
// here: `tinyagents-registry` only has `async-trait` as a *dev*-dependency
// (used by its own tests), so the macro is unavailable to non-test library
// code without adding it as a normal dependency — a `Cargo.toml` edit outside
// this change's file boundary while other in-flight work owns the manifests.
impl<State: Send + Sync> tinyagents_definition::DefinitionRegistry for CapabilityRegistry<State> {
    fn resolve<'life0, 'life1, 'async_trait>(
        &'life0 self,
        id: &'life1 str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = tinyagents_definition::Result<
                        Option<tinyagents_definition::AgentDefinition>,
                    >,
                > + Send
                + 'async_trait,
        >,
    >
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { Ok(self.agent(id).cloned()) })
    }

    fn list<'life0, 'async_trait>(
        &'life0 self,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = tinyagents_definition::Result<
                        Vec<tinyagents_definition::AgentDefinition>,
                    >,
                > + Send
                + 'async_trait,
        >,
    >
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { Ok(self.agents.values().cloned().collect()) })
    }

    fn delegates_for<'life0, 'life1, 'async_trait>(
        &'life0 self,
        id: &'life1 str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = tinyagents_definition::Result<Vec<String>>>
                + Send
                + 'async_trait,
        >,
    >
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            Ok(self
                .agent(id)
                .map(|definition| definition.subagents.clone())
                .unwrap_or_default())
        })
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
