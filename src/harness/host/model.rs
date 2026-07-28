//! Host-side model routing and construction.
//!
//! See [`ModelResolver`] for the trait contract and [`StaticModelResolver`] for
//! the single-model default.

use std::sync::Arc;

use crate::error::Result;
use crate::harness::ids::RunId;
use crate::harness::model::{ChatModel, ModelProfile, ModelRegistry};

/// Builds the model set for one run.
///
/// [`ChatModel`] is how a model is *called*; this is how one is *chosen and
/// constructed*, for embedders whose routing depends on configuration the crate
/// cannot see. It returns a populated [`ModelRegistry`], so the crate's own
/// selection, fallback, and capability checks run unchanged on top of the
/// host's routing decision.
///
/// Every method is synchronous. Resolution is a configuration read plus a
/// client construction, and capability/window questions are table lookups; a
/// host that must fetch a catalog over the network does so when it builds its
/// state, not on the hot path. Keeping the seam sync is what lets it be called
/// from synchronous session assembly without turning that whole path `async`.
pub trait ModelResolver<State: Send + Sync>: Send + Sync {
    /// Builds the registry for `request`: the primary model as the registry
    /// default, plus any additional named routes the run may select.
    ///
    /// Called once per run — the returned registry is a fresh allocation, and
    /// rebuilding it per turn is waste, not caching policy.
    fn resolve(&self, state: &State, request: &ModelResolution<'_>)
    -> Result<ModelRegistry<State>>;

    /// Returns a model's capability profile without constructing it.
    ///
    /// Lets a caller answer "does this model do native tool calls / accept
    /// images?" before a registry exists, and lets an embedder override a
    /// provider-reported capability from its own configuration. Defaults to
    /// `None` (unknown).
    fn profile(&self, state: &State, model_id: &str) -> Result<Option<ModelProfile>> {
        let _ = (state, model_id);
        Ok(None)
    }

    /// Returns a model's effective input-token window, when known. Defaults to
    /// `None`, in which case no window-driven compaction is scheduled.
    fn context_window(&self, state: &State, model_id: &str) -> Result<Option<u64>> {
        let _ = (state, model_id);
        Ok(None)
    }
}

/// What the runtime knows about the work a model is being resolved for.
#[derive(Clone, Debug)]
pub struct ModelResolution<'a> {
    /// The run being started.
    pub run_id: &'a RunId,
    /// Host-defined identity of the agent taking the run.
    pub agent_id: &'a str,
    /// Opaque host-defined workload label. The crate never interprets it, so
    /// routing rules stay entirely embedder-side.
    pub workload: &'a str,
    /// Explicit model override, when the caller or configuration pinned one.
    pub pinned_model: Option<&'a str>,
    /// Sampling temperature the caller asked for, when it asked.
    pub temperature: Option<f64>,
}

/// A [`ModelResolver`] that always resolves to one preconfigured model.
///
/// The in-crate analogue of registering a single model on a harness: `resolve`
/// builds a fresh [`ModelRegistry`], registers the model under its name (which
/// makes it the registry default), and returns it. `profile` and
/// `context_window` take the `None` trait defaults, so it needs no host
/// configuration at all.
pub struct StaticModelResolver<State: Send + Sync> {
    /// Name the model is registered under.
    name: String,
    /// The model every resolution returns.
    model: Arc<dyn ChatModel<State>>,
}

impl<State: Send + Sync> StaticModelResolver<State> {
    /// Creates a resolver that always returns `model`, registered as `name`.
    pub fn new(name: impl Into<String>, model: Arc<dyn ChatModel<State>>) -> Self {
        Self {
            name: name.into(),
            model,
        }
    }

    /// The name the model is registered under.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<State: Send + Sync> Clone for StaticModelResolver<State> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            model: Arc::clone(&self.model),
        }
    }
}

impl<State: Send + Sync> std::fmt::Debug for StaticModelResolver<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticModelResolver")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> ModelResolver<State> for StaticModelResolver<State> {
    fn resolve(
        &self,
        _state: &State,
        _request: &ModelResolution<'_>,
    ) -> Result<ModelRegistry<State>> {
        let mut registry = ModelRegistry::new();
        registry.register(self.name.clone(), Arc::clone(&self.model));
        Ok(registry)
    }
}
