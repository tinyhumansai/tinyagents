//! Host capability traits — the seams a host implements to supply product
//! behaviour to a generic agent runtime.
//!
//! # Why these exist
//!
//! The agent runtime is generic over its host. It runs the loop; it does not
//! know what a memory is, whether an action is permitted, which model a role
//! should use, or what a "connected integration" means. Those are product
//! decisions, and every one of them is asked here rather than assumed.
//!
//! This is the same extension-trait pattern the crate already uses for
//! [`ChatModel`](tinyinference::model::ChatModel),
//! [`Tool`](crate::tool::Tool),
//! [`ChatHistory`](crate::memory::ChatHistory) and a dozen others —
//! not a new architecture. See `docs/spec/host-capability-traits-rfc.md`.
//!
//! # Required vs optional
//!
//! Four capabilities are **required**: without a context composer, a definition
//! registry, a security gate, and a model resolver there is no coherent turn to
//! run. The other six are **optional**, expressed as `Option<Arc<dyn …>>` on
//! [`HostCapabilities`].
//!
//! Optionality is deliberately modelled as *absence*, never as a default impl
//! that returns an error. A registered-but-failing capability teaches a model
//! that the capability exists and makes it retry; an absent one is simply not
//! offered. The no-op implementations shipped alongside each trait
//! ([`NoopProgressSink`], [`UnlimitedBudgetGate`], …) exist for **tests and
//! embedding**, not as a way to pretend an unconfigured host has a capability.
//!
//! # Value types are inert
//!
//! Every type in these modules is `serde` + `std` only — no tokio types, no
//! storage engines, no host types. A host can depend on this module to write an
//! adapter without pulling in the rest of the crate, which is the same
//! carve-out that lets whole domains be compiled out elsewhere.

pub mod agent_memory;
pub mod budget_gate;
pub mod context_composer;
pub mod definition_registry;
pub mod experience_store;
pub mod learning_sink;
pub mod model_resolver;
pub mod progress_sink;
pub mod security_gate;
pub mod tool_outcome_classifier;

pub use agent_memory::{
    AgentMemory, InMemoryAgentMemory, MemoryId, MemoryItem, NewMemory, RecallRequest,
};
pub use budget_gate::{
    BudgetGate, CallEstimate, CompressionHint, ContextState, Permit, UnlimitedBudgetGate,
};
pub use context_composer::{ContextComposer, StaticContextComposer, TurnContextRequest};
pub use definition_registry::{AgentDefinition, DefinitionRegistry, InMemoryDefinitionRegistry};
pub use experience_store::{Experience, ExperienceStore, InMemoryExperienceStore};
pub use learning_sink::{LearningSink, NoopLearningSink, TurnSummary};
pub use model_resolver::{FixedModelResolver, ModelResolveRequest, ModelResolver};
pub use progress_sink::{NoopProgressSink, ProgressEvent, ProgressSink, RecordingProgressSink};
pub use security_gate::{
    AllowAllSecurityGate, ContentOrigin, GateDecision, ScreenOutcome, SecurityGate, ToolCallRequest,
};
pub use tool_outcome_classifier::{ErrorFieldClassifier, OutcomeClass, ToolOutcomeClassifier};

use std::sync::Arc;

/// The bundle of host capabilities handed to a session.
///
/// A bundle rather than a pile of builder setters, matching the shape
/// `tinyflows::caps::Capabilities` already uses on the sibling crate: one
/// struct makes "what does this host actually provide?" answerable at a glance,
/// and makes the required/optional split visible in the type rather than
/// discoverable only by reading builder code.
///
/// Generic over `State` solely because [`ModelResolver`] is — it hands back a
/// [`ChatModel<State>`](tinyinference::model::ChatModel), and that generic
/// parameter has to thread through.
pub struct HostCapabilities<State: Send + Sync> {
    /// Builds the system prompt and any turn preamble.
    pub context: Arc<dyn ContextComposer>,
    /// Resolves agent ids to definitions.
    pub definitions: Arc<dyn DefinitionRegistry>,
    /// Authorizes tool calls and screens untrusted input.
    pub security: Arc<dyn SecurityGate>,
    /// Chooses which model answers a given turn.
    pub models: Arc<dyn ModelResolver<State>>,

    /// Durable user memory. `None` on a host without a memory store — recall
    /// and remember are then simply not performed.
    pub memory: Option<Arc<dyn AgentMemory>>,
    /// Capacity back-pressure and spend accounting. `None` means unmetered.
    pub budget: Option<Arc<dyn BudgetGate>>,
    /// Turn progress for a UI. `None` means nothing is observing.
    pub progress: Option<Arc<dyn ProgressSink>>,
    /// Post-turn reflection pipeline. `None` means no learning is recorded.
    pub learning: Option<Arc<dyn LearningSink>>,
    /// Product-specific judgement about whether a tool result is a failure.
    /// `None` means results are not classified — there is no runtime-side
    /// fallback. A host wanting the obvious `error`-field rule supplies
    /// [`ErrorFieldClassifier`] explicitly.
    pub tool_outcomes: Option<Arc<dyn ToolOutcomeClassifier>>,
    /// Procedural memory of how this agent has performed before. `None` means
    /// no experience is recorded or recalled.
    pub experience: Option<Arc<dyn ExperienceStore>>,
}

impl<State: Send + Sync> HostCapabilities<State> {
    /// A bundle with the four required capabilities and every optional one
    /// absent.
    ///
    /// Optional capabilities are added with the `with_*` methods, so a host
    /// that gains one later does not have to restate the others — and a host
    /// that never supplies one never has to name it.
    pub fn new(
        context: Arc<dyn ContextComposer>,
        definitions: Arc<dyn DefinitionRegistry>,
        security: Arc<dyn SecurityGate>,
        models: Arc<dyn ModelResolver<State>>,
    ) -> Self {
        Self {
            context,
            definitions,
            security,
            models,
            memory: None,
            budget: None,
            progress: None,
            learning: None,
            tool_outcomes: None,
            experience: None,
        }
    }

    /// Supplies durable user memory.
    pub fn with_memory(mut self, memory: Arc<dyn AgentMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Supplies capacity back-pressure and spend accounting.
    pub fn with_budget(mut self, budget: Arc<dyn BudgetGate>) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Supplies a turn-progress observer.
    pub fn with_progress(mut self, progress: Arc<dyn ProgressSink>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Supplies a post-turn reflection sink.
    pub fn with_learning(mut self, learning: Arc<dyn LearningSink>) -> Self {
        self.learning = Some(learning);
        self
    }

    /// Supplies product-specific tool-outcome classification.
    pub fn with_tool_outcomes(mut self, classifier: Arc<dyn ToolOutcomeClassifier>) -> Self {
        self.tool_outcomes = Some(classifier);
        self
    }

    /// Supplies a procedural-experience store.
    pub fn with_experience(mut self, experience: Arc<dyn ExperienceStore>) -> Self {
        self.experience = Some(experience);
        self
    }
}

// `Clone` by hand rather than `#[derive]`: the derive would demand
// `State: Clone`, which is wrong — every field is an `Arc`, so cloning the
// bundle shares the capabilities regardless of what `State` is.
impl<State: Send + Sync> Clone for HostCapabilities<State> {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            definitions: Arc::clone(&self.definitions),
            security: Arc::clone(&self.security),
            models: Arc::clone(&self.models),
            memory: self.memory.clone(),
            budget: self.budget.clone(),
            progress: self.progress.clone(),
            learning: self.learning.clone(),
            tool_outcomes: self.tool_outcomes.clone(),
            experience: self.experience.clone(),
        }
    }
}

#[cfg(test)]
mod test;
