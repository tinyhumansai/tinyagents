//! Harness-side registration and execution support for canonical tools.
//!
//! `tinytools` owns the public tool vocabulary. This module owns only the host
//! concerns: name lookup, provider-schema projection, timeout settings, error
//! routing, and the explicit recursive-dispatch handoff.

pub mod discover;
mod prompt;
mod schema;
mod schema_compact;
mod schema_prepare;
pub mod select;
mod signature;
mod timeout;
mod types;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

pub use prompt::*;
pub use schema::*;
pub use schema_compact::*;
pub use schema_prepare::*;
pub use select::*;
pub use signature::*;
pub use timeout::*;
pub use types::ToolExecutionContext;

/// A host-owned dispatch hook for the rare canonical tool that must execute
/// against the *typed* parent run (currently recursive sub-agents).
///
/// Normal registrations use [`ToolRegistry::register`] and dispatch through
/// `tinytools::Tool::execute_with_context`. A recursive registration must be
/// explicit: no downcast, global registry, or hidden argument is involved.
#[async_trait]
pub trait ToolDispatch<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Canonical declaration exposed to the model and policy layer.
    fn tool(&self) -> Arc<dyn tinytools::Tool>;

    /// Classifies output for the host security boundary.
    ///
    /// Normal tools produce tool-origin output. Typed recursive dispatchers
    /// override this so delegated-agent output can receive the host's stricter
    /// agent-output screening policy.
    fn output_origin(&self) -> crate::host::ContentOrigin {
        crate::host::ContentOrigin::Tool
    }

    /// Supplies authoritative values for `ToolInjectedArgumentSource::Host`.
    ///
    /// This is deliberately an explicit registration-time dispatch concern;
    /// model arguments never carry host authority. The default is right for
    /// tools that only declare call-id injection (or no injected values).
    fn injected_arguments(
        &self,
        _call: &tinytools::ToolCall,
    ) -> anyhow::Result<tinytools::InjectedToolArguments> {
        Ok(tinytools::InjectedToolArguments::new())
    }

    /// Returns the output shape requested for one invocation.
    ///
    /// The default asks markdown-capable canonical tools for their compact
    /// rendering. A host that needs a different policy supplies a dispatch
    /// implementation with its own per-call choice; the loop passes this same
    /// value to execution and transcript rendering.
    fn call_options(&self, _arguments: &Value) -> tinytools::ToolCallOptions {
        tinytools::ToolCallOptions {
            prefer_markdown: self.tool().supports_markdown(),
        }
    }

    /// Executes with the full typed parent run when the dispatch needs it.
    async fn execute(
        &self,
        state: &State,
        arguments: Value,
        options: tinytools::ToolCallOptions,
        parent: &crate::context::RunContext<Ctx>,
    ) -> anyhow::Result<tinytools::ToolResult>;
}

struct CanonicalDispatch {
    tool: Arc<dyn tinytools::Tool>,
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for CanonicalDispatch {
    fn tool(&self) -> Arc<dyn tinytools::Tool> {
        self.tool.clone()
    }

    async fn execute(
        &self,
        _state: &State,
        arguments: Value,
        options: tinytools::ToolCallOptions,
        parent: &crate::context::RunContext<Ctx>,
    ) -> anyhow::Result<tinytools::ToolResult> {
        let context = ToolExecutionContext::from_run_context(parent);
        self.tool
            .execute_with_context(arguments, options, Some(&context))
            .await
    }
}

/// A name-keyed canonical tool registry.
pub struct ToolRegistry<State: Send + Sync, Ctx: Send + Sync> {
    tools: HashMap<String, Arc<dyn ToolDispatch<State, Ctx>>>,
}

/// Outcome of a registration that reports whether it replaced an existing
/// entry under the same name.
///
/// Returned by [`ToolRegistry::try_register`]/[`ToolRegistry::try_register_dispatch`]
/// so a caller that cares can detect the collision instead of it silently
/// overwriting the earlier registration (M-5; `docs/sdk-gaps.md` §15 asks for
/// duplicate-registration diagnostics).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegisterOutcome {
    /// No prior registration existed under this name.
    Registered,
    /// A prior registration under this name was replaced. Carries the
    /// replaced name (redundant with the call site's own `tool.name()`, but
    /// convenient for a caller that registers in a loop and wants to report
    /// which names collided without re-deriving them).
    Replaced(String),
}

impl RegisterOutcome {
    /// `true` when this call replaced an existing registration.
    pub fn replaced(&self) -> bool {
        matches!(self, RegisterOutcome::Replaced(_))
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolRegistry<State, Ctx> {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a canonical tool under its declared name.
    ///
    /// A duplicate name silently replaces the earlier registration (except
    /// for logging a `tracing::warn!` diagnostic) so this method keeps its
    /// chaining-friendly `&mut Self` return for existing callers; use
    /// [`Self::try_register`] to detect and react to the collision instead.
    pub fn register(&mut self, tool: Arc<dyn tinytools::Tool>) -> &mut Self {
        let name = tool.name().to_owned();
        if let RegisterOutcome::Replaced(name) =
            self.insert_dispatch(name, Arc::new(CanonicalDispatch { tool }))
        {
            tracing::warn!(
                target: "tinyagents::tool",
                tool = %name,
                "[tool] registration replaced an already-registered tool of the same name"
            );
        }
        self
    }

    /// Like [`Self::register`], but reports whether the name was already
    /// registered instead of only logging it, so a caller can fail fast on a
    /// collision it did not expect (M-5).
    pub fn try_register(&mut self, tool: Arc<dyn tinytools::Tool>) -> RegisterOutcome {
        let name = tool.name().to_owned();
        self.insert_dispatch(name, Arc::new(CanonicalDispatch { tool }))
    }

    /// Registers an explicit typed-parent dispatcher for a canonical tool.
    ///
    /// See [`Self::register`] for the duplicate-name policy; use
    /// [`Self::try_register_dispatch`] to detect it instead.
    pub fn register_dispatch(&mut self, dispatch: Arc<dyn ToolDispatch<State, Ctx>>) -> &mut Self {
        let name = dispatch.tool().name().to_owned();
        if let RegisterOutcome::Replaced(name) = self.insert_dispatch(name, dispatch) {
            tracing::warn!(
                target: "tinyagents::tool",
                tool = %name,
                "[tool] registration replaced an already-registered tool of the same name"
            );
        }
        self
    }

    /// Like [`Self::register_dispatch`], but reports whether the name was
    /// already registered instead of only logging it (M-5).
    pub fn try_register_dispatch(
        &mut self,
        dispatch: Arc<dyn ToolDispatch<State, Ctx>>,
    ) -> RegisterOutcome {
        let name = dispatch.tool().name().to_owned();
        self.insert_dispatch(name, dispatch)
    }

    /// Shared insertion path: inserts `dispatch` under `name`, returning
    /// whether a prior entry under that name was replaced.
    fn insert_dispatch(
        &mut self,
        name: String,
        dispatch: Arc<dyn ToolDispatch<State, Ctx>>,
    ) -> RegisterOutcome {
        match self.tools.insert(name.clone(), dispatch) {
            Some(_) => RegisterOutcome::Replaced(name),
            None => RegisterOutcome::Registered,
        }
    }

    /// Looks up the complete host dispatch entry, whatever its exposure.
    ///
    /// This is the host-side lookup: a `Hidden` tool resolves here so host code
    /// (an unknown-tool rewrite target, a composite capability's inner step)
    /// can still reach it. Model-originated calls go through
    /// [`Self::model_dispatch`].
    pub(crate) fn dispatch(&self, name: &str) -> Option<Arc<dyn ToolDispatch<State, Ctx>>> {
        self.tools.get(name).cloned()
    }

    /// Looks up a dispatch entry the *model* is allowed to name.
    ///
    /// `Direct` and `Deferred` tools resolve; a `Hidden` tool does not, so a
    /// model that guesses (or is told) a hidden name gets the same unknown-tool
    /// answer as for a name that was never registered. The deferred case is
    /// what makes discovery work: a tool `tool_search` revealed is callable by
    /// its own name even though it never appeared in the request's `tools`.
    pub(crate) fn model_dispatch(&self, name: &str) -> Option<Arc<dyn ToolDispatch<State, Ctx>>> {
        self.dispatch(name)
            .filter(|dispatch| dispatch.tool().exposure() != tinytools::ToolExposure::Hidden)
    }

    /// Returns how a registered tool enters the model-visible catalogue.
    #[must_use]
    pub fn exposure(&self, name: &str) -> Option<tinytools::ToolExposure> {
        self.dispatch(name)
            .map(|dispatch| dispatch.tool().exposure())
    }

    /// Looks up a canonical tool declaration.
    pub fn get(&self, name: &str) -> Option<Arc<dyn tinytools::Tool>> {
        self.dispatch(name).map(|dispatch| dispatch.tool())
    }

    /// Returns registered names in sorted order, whatever their exposure.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the names a model may call (`Direct` and `Deferred`), sorted.
    #[must_use]
    pub fn model_callable_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .tools
            .iter()
            .filter(|(_, dispatch)| dispatch.tool().exposure() != tinytools::ToolExposure::Hidden)
            .map(|(name, _)| name.clone())
            .collect();
        names.sort();
        names
    }

    /// Returns the provider request schemas of every **directly advertised**
    /// tool, projected from the canonical declarations and sorted by name.
    ///
    /// Only [`tinytools::ToolExposure::Direct`] tools are included. Deferred
    /// tools are the model's to discover (see [`Self::deferred_schemas`] and
    /// [`crate::tool::discover`]); hidden tools never reach a model. The sort
    /// is what keeps the wire bytes stable across turns, which a provider
    /// prompt cache depends on.
    #[must_use]
    pub fn schemas(&self) -> Vec<tinyinference_llm::tool::ToolSchema> {
        self.schemas_with_exposure(tinytools::ToolExposure::Direct)
    }

    /// Returns the provider request schemas of every
    /// [`tinytools::ToolExposure::Deferred`] tool, sorted by name.
    ///
    /// These are the schemas the agent loop indexes for `tool_search` instead
    /// of sending on every request.
    #[must_use]
    pub fn deferred_schemas(&self) -> Vec<tinyinference_llm::tool::ToolSchema> {
        self.schemas_with_exposure(tinytools::ToolExposure::Deferred)
    }

    fn schemas_with_exposure(
        &self,
        exposure: tinytools::ToolExposure,
    ) -> Vec<tinyinference_llm::tool::ToolSchema> {
        let mut schemas: Vec<_> = self
            .tools
            .values()
            .map(|dispatch| dispatch.tool())
            .filter(|tool| tool.exposure() == exposure)
            .map(|tool| provider_schema(tool.as_ref()))
            .collect();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    /// Returns canonical specs including host-injected fields for introspection.
    #[must_use]
    pub fn declared_specs(&self) -> Vec<tinytools::ToolSpec> {
        let mut specs: Vec<_> = self
            .tools
            .values()
            .map(|dispatch| dispatch.tool().spec())
            .collect();
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        specs
    }

    /// Returns declared policies keyed by tool name.
    #[must_use]
    pub fn policies(&self) -> HashMap<String, tinytools::ToolPolicy> {
        self.tools
            .iter()
            .map(|(name, dispatch)| (name.clone(), dispatch.tool().policy()))
            .collect()
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> Default for ToolRegistry<State, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a canonical spec into the inference provider's request schema.
/// Host-injected values are removed before a model sees the schema.
#[must_use]
pub(crate) fn provider_schema(tool: &dyn tinytools::Tool) -> tinyinference_llm::tool::ToolSchema {
    let spec = tool.spec();
    tinyinference_llm::tool::ToolSchema {
        name: spec.name,
        description: spec.description,
        parameters: tinytools::project_injected_arguments(
            &spec.parameters,
            &tool.injected_arguments(),
        ),
        format: tinyinference_llm::tool::ToolFormat::Json,
    }
}

#[cfg(test)]
mod canonical_test;
#[cfg(test)]
mod timeout_test;
