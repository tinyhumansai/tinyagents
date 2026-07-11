//! High-level **model router** — the declarative workload-tier layer over the
//! named model registry (registry component kind
//! [`Router`](crate::registry::ComponentKind::Router)).
//!
//! # Why this exists
//!
//! [`CapabilityRegistry`](crate::registry::CapabilityRegistry) and the harness
//! [`ModelRegistry`](crate::harness::runtime::ModelRegistry) resolve a model *by
//! name* and the agent loop can fail over across a
//! [`FallbackPolicy`](crate::harness::retry::FallbackPolicy) — but neither owns
//! the *policy* that maps a host's **workload tiers** (`chat-v1`, `reasoning-v1`,
//! `vision-v1`, …) onto concrete models, nor the per-tier capability gates and
//! same-family fallback ordering that go with them. Hosts have historically
//! re-implemented that projection by hand (OpenHuman's `RouterProvider` +
//! `routes.rs`): register a model per tier alias, build a `FallbackPolicy`, stamp
//! a required [`CapabilitySet`](crate::harness::model::CapabilitySet) per turn.
//!
//! [`ModelRouter`] is the crate-owned home for exactly that policy. A host
//! *declares* its tier table once — each [`WorkloadRoute`] names the model an
//! alias forwards to, the capabilities a request routed there must satisfy, and
//! the ordered sibling aliases to fall back to — and the router answers the three
//! questions the turn assembly needs:
//!
//! - **resolution**: which registered model does this alias forward to?
//!   ([`target_model`](ModelRouter::target_model))
//! - **fallback**: the [`FallbackPolicy`] for a turn whose primary is this alias
//!   (`[alias, fallbacks…]`) ([`fallback_policy`](ModelRouter::fallback_policy))
//! - **capability gate**: the [`CapabilitySet`] to stamp on requests routed here
//!   ([`required_capabilities`](ModelRouter::required_capabilities))
//!
//! It holds no models and drives no I/O — it is pure, cheap, cloneable policy
//! that a harness assembler reads while wiring a registry + run policy. That
//! keeps the *what routes where* declarative and testable in isolation from the
//! *how a model is built* (which stays with the host provider/factory).
//!
//! # Example
//!
//! ```
//! use tinyagents::harness::model::CapabilitySet;
//! use tinyagents::registry::router::{ModelRouter, WorkloadRoute};
//!
//! let router = ModelRouter::new()
//!     .with_route(WorkloadRoute::new("chat-v1", "chat-v1").with_fallbacks(["burst-v1"]))
//!     .with_route(WorkloadRoute::new("burst-v1", "burst-v1").with_fallbacks(["chat-v1"]))
//!     .with_route(
//!         WorkloadRoute::new("vision-v1", "vision-v1")
//!             .requiring(CapabilitySet { image_in: true, ..CapabilitySet::default() }),
//!     )
//!     .with_default("chat-v1");
//!
//! // A chat turn fails over to its sibling burst tier.
//! let policy = router.fallback_policy("chat-v1").unwrap();
//! assert_eq!(policy.next_after("chat-v1"), Some("burst-v1"));
//!
//! // A vision turn is capability-gated and primary-only (no fallback).
//! assert!(router.required_capabilities("vision-v1").unwrap().image_in);
//! assert!(router.fallback_policy("vision-v1").is_none());
//! ```

mod types;

#[cfg(test)]
mod test;

pub use types::WorkloadRoute;

use crate::error::{Result, TinyAgentsError};
use crate::harness::model::CapabilitySet;
use crate::harness::retry::FallbackPolicy;

/// A declarative, name-addressable router over registered models: it maps named
/// workload tiers to concrete registered model names and owns the per-tier
/// capability gates and same-family fallback ordering.
///
/// Insertion order is preserved so [`routes`](Self::routes) and
/// [`aliases`](Self::aliases) iterate deterministically (registration order),
/// which callers rely on when projecting the table onto a registry.
///
/// See the [module docs](self) for the design rationale and an example.
#[derive(Clone, Debug, Default)]
pub struct ModelRouter {
    /// Insertion-ordered route table. Small (a handful of tiers), so linear scans
    /// are cheaper than a map and keep ordering deterministic.
    routes: Vec<WorkloadRoute>,
    /// The alias dispatched to when a caller names no tier, if set.
    default_alias: Option<String>,
}

impl ModelRouter {
    /// An empty router with no routes and no default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a route, returning `self` for chaining (builder style).
    ///
    /// A later route with the same alias **overwrites** the earlier one (last
    /// write wins), keeping builder chains ergonomic. Use
    /// [`register`](Self::register) for the fallible, duplicate-rejecting form.
    #[must_use]
    pub fn with_route(mut self, route: WorkloadRoute) -> Self {
        self.upsert(route);
        self
    }

    /// Sets the default alias (dispatched to when a caller names no tier),
    /// returning `self` for chaining.
    #[must_use]
    pub fn with_default(mut self, alias: impl Into<String>) -> Self {
        self.default_alias = Some(alias.into());
        self
    }

    /// Registers a route, rejecting a duplicate alias.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::DuplicateComponent`] if a route with the same
    /// alias is already registered. Use [`with_route`](Self::with_route) for the
    /// infallible last-write-wins builder form.
    pub fn register(&mut self, route: WorkloadRoute) -> Result<&mut Self> {
        if self.routes.iter().any(|r| r.alias == route.alias) {
            return Err(TinyAgentsError::DuplicateComponent(format!(
                "router route '{}'",
                route.alias
            )));
        }
        self.routes.push(route);
        Ok(self)
    }

    /// Sets the default alias (dispatched to when a caller names no tier).
    pub fn set_default(&mut self, alias: impl Into<String>) -> &mut Self {
        self.default_alias = Some(alias.into());
        self
    }

    /// The default alias, if one is set.
    pub fn default_alias(&self) -> Option<&str> {
        self.default_alias.as_deref()
    }

    /// The route registered under `alias`, if any.
    pub fn route(&self, alias: &str) -> Option<&WorkloadRoute> {
        self.routes.iter().find(|r| r.alias == alias)
    }

    /// All routes in registration order.
    pub fn routes(&self) -> &[WorkloadRoute] {
        &self.routes
    }

    /// Every registered alias in registration order.
    pub fn aliases(&self) -> impl Iterator<Item = &str> {
        self.routes.iter().map(|r| r.alias.as_str())
    }

    /// Whether any route is registered.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// The concrete registered model name `alias` forwards to, if `alias` is a
    /// known route.
    pub fn target_model(&self, alias: &str) -> Option<&str> {
        self.route(alias).map(|r| r.model.as_str())
    }

    /// The [`FallbackPolicy`] for a turn whose primary/effective tier is `alias`:
    /// the chain `[alias, fallbacks…]`.
    ///
    /// The crate's [`FallbackPolicy::next_after`] traversal expects the current
    /// (primary) name as the first entry and yields each subsequent alternate, so
    /// the primary alias itself heads the returned chain.
    ///
    /// Returns `None` when `alias` is unknown or has no fallbacks (e.g. a
    /// capability-gated primary-only tier), so no fallback policy is installed for
    /// that turn.
    pub fn fallback_policy(&self, alias: &str) -> Option<FallbackPolicy> {
        let route = self.route(alias)?;
        if route.fallbacks.is_empty() {
            return None;
        }
        let mut chain = Vec::with_capacity(route.fallbacks.len() + 1);
        chain.push(route.alias.clone());
        chain.extend(route.fallbacks.iter().cloned());
        Some(FallbackPolicy::new(chain))
    }

    /// The [`CapabilitySet`] to stamp on requests routed to `alias`, or `None`
    /// when the route imposes no requirement (the common text turn) — so the
    /// caller installs no capability gate.
    pub fn required_capabilities(&self, alias: &str) -> Option<CapabilitySet> {
        self.route(alias)
            .filter(|r| r.has_capability_gate())
            .map(|r| r.requires.clone())
    }

    /// Insert-or-overwrite by alias (last write wins), preserving the original
    /// position on overwrite so builder order stays stable.
    fn upsert(&mut self, route: WorkloadRoute) {
        if let Some(slot) = self.routes.iter_mut().find(|r| r.alias == route.alias) {
            *slot = route;
        } else {
            self.routes.push(route);
        }
    }
}
