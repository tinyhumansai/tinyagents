//! Runtime-owned executable model registry and selection policy.
//!
//! Provider-neutral inference types and calls live in `tinyinference`; this
//! module owns only name registration, fallback ordering, and runtime defaults.

mod types;

use std::sync::Arc;

use tinyinference::model::{
    CapabilitySet, ChatModel, ModelProfile, ModelRequest, ModelResolutionSource, ResolvedModel,
};

pub use types::*;

impl<State: Send + Sync> ModelRegistry<State> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
            default: None,
        }
    }

    /// Registers `model` under `name`.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
    ) -> &mut Self {
        let name = name.into();
        if self.default.is_none() {
            self.default = Some(name.clone());
        }
        self.models.insert(name, model);
        self
    }

    /// Sets the runtime default model name.
    pub fn set_default(&mut self, name: impl Into<String>) -> &mut Self {
        self.default = Some(name.into());
        self
    }

    /// Looks up a model by runtime name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn ChatModel<State>>> {
        self.models.get(name).cloned()
    }

    /// Returns the configured default model.
    pub fn default_model(&self) -> Option<Arc<dyn ChatModel<State>>> {
        self.default.as_deref().and_then(|name| self.get(name))
    }

    /// Returns the configured default name.
    pub fn default_name(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// Returns registered names in sorted order.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.models.keys().cloned().collect();
        names.sort();
        names
    }

    /// Resolves a model using override, previous state, hints, and defaults.
    pub fn resolve(&self, selection: ModelSelection) -> Option<ResolvedModelBinding<State>> {
        let required = selection.required_capabilities.as_ref();
        let allow_retired = selection.allow_retired;
        if let Some(requested) = selection.requested
            && let Some(model) = self.get(&requested)
            && model_eligible(model.as_ref(), required, allow_retired)
        {
            return Some(binding(
                model,
                requested.clone(),
                Some(requested),
                ModelResolutionSource::RequestOverride,
            ));
        }
        if selection.reuse_previous
            && let Some(previous) = selection.previous
            && let Some(model) = self.get(&previous.name)
            && model_eligible(model.as_ref(), required, allow_retired)
        {
            return Some(binding(
                model,
                previous.name,
                previous.requested,
                ModelResolutionSource::StateReuse,
            ));
        }
        let mut hints: Vec<(usize, _)> = selection.hints.into_iter().enumerate().collect();
        hints.sort_by(|(left_index, left), (right_index, right)| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left_index.cmp(right_index))
        });
        for (_, hint) in hints {
            if let Some(model) = self.get(&hint.model)
                && model_eligible(model.as_ref(), required, allow_retired)
            {
                return Some(binding(
                    model,
                    hint.model.clone(),
                    Some(hint.model),
                    ModelResolutionSource::Hint,
                ));
            }
        }
        if let Some(agent_default) = selection.agent_default
            && let Some(model) = self.get(&agent_default)
            && model_eligible(model.as_ref(), required, allow_retired)
        {
            return Some(binding(
                model,
                agent_default.clone(),
                Some(agent_default),
                ModelResolutionSource::AgentDefault,
            ));
        }
        let name = self.default_name()?.to_string();
        self.default_model()
            .filter(|model| model_eligible(model.as_ref(), required, allow_retired))
            .map(|model| binding(model, name, None, ModelResolutionSource::RegistryDefault))
    }

    /// Resolves a model from one direct TinyInference request.
    pub fn resolve_request(
        &self,
        request: &ModelRequest,
        agent_default: Option<&str>,
        previous: Option<ResolvedModel>,
    ) -> Option<ResolvedModelBinding<State>> {
        self.resolve(ModelSelection {
            requested: request.model.clone(),
            previous,
            reuse_previous: request.reuse_previous_model,
            hints: request.model_hints.clone(),
            agent_default: agent_default.map(ToOwned::to_owned),
            required_capabilities: request.required_capabilities.clone(),
            allow_retired: false,
        })
    }
}

fn binding<State: Send + Sync>(
    model: Arc<dyn ChatModel<State>>,
    name: String,
    requested: Option<String>,
    source: ModelResolutionSource,
) -> ResolvedModelBinding<State> {
    ResolvedModelBinding {
        resolved: ResolvedModel {
            name,
            requested,
            source,
        },
        model,
    }
}

fn model_satisfies<State: Send + Sync>(
    model: &dyn ChatModel<State>,
    required: Option<&CapabilitySet>,
) -> bool {
    match required {
        None => true,
        Some(required) if required == &CapabilitySet::default() => true,
        Some(required) => model
            .profile()
            .is_some_and(|profile| profile.satisfies(required)),
    }
}

pub(crate) fn model_eligible<State: Send + Sync>(
    model: &dyn ChatModel<State>,
    required: Option<&CapabilitySet>,
    allow_retired: bool,
) -> bool {
    model_satisfies(model, required)
        && (allow_retired || model.profile().is_none_or(ModelProfile::is_usable))
}

impl<State: Send + Sync> Default for ModelRegistry<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
