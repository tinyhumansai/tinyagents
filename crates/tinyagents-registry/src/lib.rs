//! Registry coordination and discovery primitives — the **named capability
//! catalog** that makes TinyAgents recursive.
//!
//! In the recursive architecture, a model, agent, or graph can reach for
//! capabilities it never hardcoded: a `.rag` blueprint (or a host orchestration
//! session) references a model/tool/agent/graph *by name*, and the registry is
//! what resolves that name to a real, Rust-registered handle. By owning the set
//! of legal names, the registry is also the boundary that makes agent-authored
//! plans safe to compile — a self-authored workflow can only bind to
//! capabilities a human explicitly registered and allowed.
//!
//! The registry owns named runtime components and local metadata catalogs, in
//! two complementary pieces:
//!
//! - [`CapabilityRegistry`] ([`capability`]) — the name-addressable catalog of
//!   models, tools, graph blueprints, routers, and reducers that `.rag`
//!   sources bind against, plus the discovery [`component`] types
//!   ([`ComponentKind`]/[`ComponentId`]/[`ComponentMetadata`]) that describe
//!   what is registered.
//! - [`ModelCatalog`] ([`catalog`]) — a checked-in snapshot of provider model
//!   prices, context windows, and capabilities for deterministic, offline
//!   lookup (cost estimation, model selection, capability gating).
//! - [`ModelRouter`] ([`router`]) — the declarative workload-tier layer over the
//!   named model registry: maps host workload aliases (`chat-v1`, `vision-v1`, …)
//!   onto concrete registered models with per-tier capability gates and ordered
//!   same-family fallback chains (registry component kind
//!   [`Router`](ComponentKind::Router)).

pub mod capability;
pub mod catalog;
pub mod component;
pub mod diagnostics;
pub mod router;

pub use tinyagents_harness::error::{Result, TinyAgentsError};

pub use capability::CapabilityRegistry;
pub use catalog::{
    ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelCatalogSnapshot, ModelCatalogSource,
    ModelPricing,
};
pub use component::{ComponentId, ComponentKind, ComponentMetadata};
pub use diagnostics::{AliasBinding, DiagnosticSeverity, RegistryDiagnostic, RegistrySnapshot};
pub use router::{ModelRouter, WorkloadRoute};

impl<State: Send + Sync> tinyagents_language::capability_resolver::CapabilitySource
    for CapabilityRegistry<State>
{
    fn names(&self, kind: tinyagents_language::capability_resolver::CapabilityKind) -> Vec<String> {
        use tinyagents_language::capability_resolver::CapabilityKind;

        let kind = match kind {
            CapabilityKind::Model => ComponentKind::Model,
            CapabilityKind::Tool => ComponentKind::Tool,
            CapabilityKind::Graph => ComponentKind::Graph,
            CapabilityKind::Router => ComponentKind::Router,
            CapabilityKind::Reducer => ComponentKind::Reducer,
            CapabilityKind::Agent => ComponentKind::Agent,
            CapabilityKind::Script => ComponentKind::Script,
        };
        self.names_including_aliases(kind)
    }
}

impl<State: Send + Sync> tinyagents_graph::subagent_node::AgentRegistry
    for CapabilityRegistry<State>
{
    fn agent(
        &self,
        name: &str,
    ) -> Option<std::sync::Arc<dyn tinyagents_graph::subagent_node::HarnessAgent>> {
        CapabilityRegistry::agent(self, name)
    }
}
