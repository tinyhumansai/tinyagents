//! Registry coordination and discovery primitives — the **named capability
//! catalog** that makes TinyAgents recursive.
//!
//! In the recursive (RLM-style) architecture, a model, agent, or graph can
//! reach for capabilities it never hardcoded: a `.rag` blueprint or `.ragsh`
//! REPL line references a model/tool/agent/graph *by name*, and the registry is
//! what resolves that name to a real, Rust-registered handle. By owning the set
//! of legal names, the registry is also the boundary that makes agent-authored
//! plans safe to compile — a self-authored workflow can only bind to
//! capabilities a human explicitly registered and allowed.
//!
//! The registry owns named runtime components and local metadata catalogs, in
//! two complementary pieces:
//!
//! - [`CapabilityRegistry`] ([`capability`]) — the name-addressable catalog of
//!   models, tools, graph blueprints, routers, and reducers that `.rag`/`.ragsh`
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

pub use capability::CapabilityRegistry;
pub use catalog::{
    ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelCatalogSnapshot, ModelCatalogSource,
    ModelPricing,
};
pub use component::{ComponentId, ComponentKind, ComponentMetadata};
pub use diagnostics::{AliasBinding, DiagnosticSeverity, RegistryDiagnostic, RegistrySnapshot};
pub use router::{ModelRouter, WorkloadRoute};
