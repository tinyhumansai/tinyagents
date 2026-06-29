//! The named capability registry: a higher-level catalog of capabilities
//! addressable by name.
//!
//! This layer is deliberately distinct from the harness'
//! [`crate::harness::model::ModelRegistry`] and
//! [`crate::harness::tool::ToolRegistry`], which are per-run executable stores.
//! The [`CapabilityRegistry`] is a *capability catalog*: it owns named models,
//! tools, graph blueprints, routers, and reducers so declarative `.rag`/`.ragsh`
//! sources can be bound by name, then validated against what Rust has actually
//! registered and allowed.

use std::collections::HashMap;
use std::sync::Arc;

use crate::harness::model::ChatModel;
use crate::harness::tool::Tool;
use crate::language::Blueprint;
use crate::registry::component::{ComponentKind, ComponentMetadata};

/// A name-addressable catalog of registered capabilities.
///
/// The registry is generic over the application `State` because models and
/// tools are generic over it. The default `State = ()` matches the common case
/// of stateless capabilities.
///
/// Storage is partitioned by [`ComponentKind`]:
///
/// - **Models, tools, graphs** keep an executable/serializable value.
/// - **Routers, reducers** (and the reserved store/agent kinds) are name-only
///   descriptors for now: enough for the `.rag` resolver to answer "is this
///   name registered?".
///
/// The [`metadata`](CapabilityRegistry::metadata) map is the source of truth for
/// *presence*: every successful registration records a
/// [`ComponentMetadata`] entry keyed by `(kind, name)`, so
/// [`has`](CapabilityRegistry::has) and [`names`](CapabilityRegistry::names)
/// work uniformly across kinds.
pub struct CapabilityRegistry<State = ()>
where
    State: Send + Sync,
{
    pub(crate) models: HashMap<String, Arc<dyn ChatModel<State>>>,
    pub(crate) tools: HashMap<String, Arc<dyn Tool<State>>>,
    pub(crate) graphs: HashMap<String, Blueprint>,
    /// Presence + discovery metadata, keyed by `(kind, canonical name)`.
    pub(crate) meta: HashMap<(ComponentKind, String), ComponentMetadata>,
    /// Alias map, keyed by `(kind, alias)` -> canonical name.
    pub(crate) aliases: HashMap<(ComponentKind, String), String>,
}
