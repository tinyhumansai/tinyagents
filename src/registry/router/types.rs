//! Declarative types for the high-level [`ModelRouter`](super::ModelRouter): a
//! [`WorkloadRoute`] (one named tier) and the router that owns an ordered set of
//! them.

use serde::{Deserialize, Serialize};

use crate::harness::model::CapabilitySet;

/// One declarative **workload route**: a stable alias (e.g. `reasoning-v1`) that
/// resolves to a concrete registered model, plus the capability gate a request
/// routed here must satisfy and an ordered same-family fallback chain of sibling
/// aliases.
///
/// A route is pure metadata — it names a model rather than owning one, so the
/// same route table can be declared once and projected onto any
/// [`CapabilityRegistry`](crate::registry::CapabilityRegistry) /
/// [`ModelRegistry`](crate::harness::runtime::ModelRegistry) that has registered
/// those model names. This is what lets a host describe its tiered routing
/// (workload aliases → concrete BYOK/managed/local models) declaratively and hand
/// it to the crate, instead of re-implementing alias resolution + fallback
/// ordering + capability gating by hand.
///
/// # Fields
/// - `alias` — the routable name callers dispatch to (the registry model name the
///   route is registered under).
/// - `model` — the concrete provider model id this alias forwards to. Often equal
///   to `alias` when the backend resolves the alias server-side (a managed tier),
///   or a real model id when the router itself is the resolver (BYOK).
/// - `requires` — the [`CapabilitySet`] stamped on every request routed to this
///   alias, so an unfit candidate is rejected/skipped pre-dispatch. [`Default`]
///   (require nothing) is the common case.
/// - `fallbacks` — ordered sibling aliases to try when this route errors. Each
///   entry should itself be a registered route so the crate can resolve it. Kept
///   short (0–2 same-family alternates) to bound cross-route fan-out.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadRoute {
    /// The routable alias (registry model name) callers dispatch to.
    pub alias: String,
    /// The concrete provider model id this alias forwards to.
    pub model: String,
    /// Capabilities a request routed to this alias must satisfy.
    #[serde(default, skip_serializing_if = "is_default_capabilities")]
    pub requires: CapabilitySet,
    /// Ordered sibling aliases to fall back to when this route errors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallbacks: Vec<String>,
}

impl WorkloadRoute {
    /// A route mapping `alias` → `model` with no capability gate and no fallbacks.
    pub fn new(alias: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
            model: model.into(),
            requires: CapabilitySet::default(),
            fallbacks: Vec::new(),
        }
    }

    /// Sets the capability gate for requests routed to this alias.
    #[must_use]
    pub fn requiring(mut self, requires: CapabilitySet) -> Self {
        self.requires = requires;
        self
    }

    /// Sets the ordered same-family fallback aliases for this route.
    #[must_use]
    pub fn with_fallbacks(
        mut self,
        fallbacks: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.fallbacks = fallbacks.into_iter().map(Into::into).collect();
        self
    }

    /// Whether this route imposes any capability requirement.
    pub(super) fn has_capability_gate(&self) -> bool {
        self.requires != CapabilitySet::default()
    }
}

/// Skip serializing a `requires` field that requires nothing (the common case),
/// keeping declared route tables compact.
fn is_default_capabilities(set: &CapabilitySet) -> bool {
    *set == CapabilitySet::default()
}
