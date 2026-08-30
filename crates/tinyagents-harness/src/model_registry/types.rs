//! Runtime-owned model registration and selection types.

use std::collections::HashMap;
use std::sync::Arc;

use tinyinference::model::{CapabilitySet, ChatModel, ModelHint, ResolvedModel};

/// Input policy for resolving one registered model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelSelection {
    /// Explicit model override.
    pub requested: Option<String>,
    /// Previous durable selection.
    pub previous: Option<ResolvedModel>,
    /// Whether the previous selection may be reused.
    pub reuse_previous: bool,
    /// Ordered runtime hints.
    pub hints: Vec<ModelHint>,
    /// Agent-level default.
    pub agent_default: Option<String>,
    /// Required inference capabilities.
    pub required_capabilities: Option<CapabilitySet>,
    /// Whether retired models may be selected for replay.
    pub allow_retired: bool,
}

/// Name-keyed runtime registry of executable models.
pub struct ModelRegistry<State: Send + Sync> {
    pub(crate) models: HashMap<String, Arc<dyn ChatModel<State>>>,
    pub(crate) default: Option<String>,
}

impl<State: Send + Sync> std::fmt::Debug for ModelRegistry<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<&str> = self.models.keys().map(String::as_str).collect();
        names.sort_unstable();
        formatter
            .debug_struct("ModelRegistry")
            .field("models", &names)
            .field("default", &self.default)
            .finish()
    }
}

/// Executable binding plus durable selection metadata.
pub struct ResolvedModelBinding<State: Send + Sync> {
    /// Selected-model metadata.
    pub resolved: ResolvedModel,
    /// Executable model handle.
    pub model: Arc<dyn ChatModel<State>>,
}

impl<State: Send + Sync> std::fmt::Debug for ResolvedModelBinding<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedModelBinding")
            .field("resolved", &self.resolved)
            .finish_non_exhaustive()
    }
}
