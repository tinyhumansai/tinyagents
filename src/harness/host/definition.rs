//! The agent definitions a runtime can instantiate.
//!
//! See [`DefinitionRegistry`] for the trait contract, [`AgentDefinition`] for
//! the deliberately minimal crate-owned shape, and
//! [`InMemoryDefinitionRegistry`] for the default.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, TinyAgentsError};

/// A crate-owned, product-neutral description of one instantiable agent.
///
/// This type is deliberately **small**. It carries only what a generic runtime
/// must read to stand an agent up — an identity, some text, a model hint, and a
/// tool list — and pushes everything else into [`extras`][Self::extras], an
/// opaque payload the crate never inspects. Host concepts such as compaction
/// profiles, delegation overrides, prompt-source indirection, and workspace
/// layout live there or in the host's own richer definition type, which maps
/// *into* this one at registration time.
///
/// The narrowness is the design, not a placeholder. A published type that grows
/// a field per host concept becomes a breaking change every time the host
/// learns something new, and it drags host vocabulary onto docs.rs. Adding an
/// entry to `extras` costs the crate nothing and breaks no downstream build.
///
/// [`Default`] is implemented so hosts can construct with struct-update syntax
/// (`AgentDefinition { id, ..Default::default() }`) and keep compiling if the
/// crate ever does add a field.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Host-defined identity, unique within a registry.
    pub id: String,
    /// Short human-readable description of what this agent is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Fully rendered system prompt, when the definition carries one directly.
    ///
    /// Plain text on purpose: a host whose prompt is assembled from layers,
    /// files, or a function does that assembly on its side and registers the
    /// result, or renders it through
    /// [`ContextComposer`][super::ContextComposer] instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Model this agent prefers, when the definition pins one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Names of the tools this agent should be given.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// Opaque host payload carried through untouched.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub extras: Value,
}

impl AgentDefinition {
    /// Creates a definition with only an id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Sets the human-readable description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the fully rendered system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Pins the preferred model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the tool names this agent should be given.
    pub fn with_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools = tools.into_iter().map(Into::into).collect();
        self
    }

    /// Attaches the opaque host payload.
    pub fn with_extras(mut self, extras: Value) -> Self {
        self.extras = extras;
        self
    }
}

/// Supplies the agent definitions a runtime can instantiate.
///
/// The crate owns the [`AgentDefinition`] *type*; this trait owns where
/// instances come from — bundled sets, on-disk files, or host configuration.
/// It is deliberately read-only: mutation, validation, and precedence between
/// sources are the embedder's concern and happen before `list` returns.
#[async_trait]
pub trait DefinitionRegistry<State: Send + Sync>: Send + Sync {
    /// Returns the definition registered under `id`, or `Ok(None)` when no
    /// definition claims it. An unknown id is not an error.
    async fn get(&self, state: &State, id: &str) -> Result<Option<AgentDefinition>>;

    /// Returns every definition, in the embedder's preferred order.
    async fn list(&self, state: &State) -> Result<Vec<AgentDefinition>>;

    /// The id to instantiate when a caller names none.
    ///
    /// Defaults to `None`; the crate ships no default agent identity, so the
    /// embedder names it here rather than the crate hard-coding one.
    async fn default_id(&self, state: &State) -> Result<Option<String>> {
        let _ = state;
        Ok(None)
    }
}

/// The map plus insertion order backing [`InMemoryDefinitionRegistry`].
#[derive(Default)]
struct DefinitionIndex {
    /// `id → definition`.
    by_id: HashMap<String, AgentDefinition>,
    /// Ids in insertion order, so `list` is stable.
    order: Vec<String>,
    /// Id returned by [`DefinitionRegistry::default_id`].
    default_id: Option<String>,
}

/// Ephemeral, in-process [`DefinitionRegistry`].
///
/// Constructed empty, so `list` returns `vec![]`, `get` returns `None`, and
/// `default_id` returns `None` until a host populates it. Clones share the same
/// underlying data through the inner [`Arc`]; there is no durability.
#[derive(Clone, Default)]
pub struct InMemoryDefinitionRegistry {
    inner: Arc<Mutex<DefinitionIndex>>,
}

impl InMemoryDefinitionRegistry {
    /// Creates a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `definition`, replacing any prior entry with the same id and
    /// keeping the original insertion position.
    pub fn insert(&self, definition: AgentDefinition) -> Result<()> {
        let mut index = self.lock()?;
        if !index.by_id.contains_key(&definition.id) {
            index.order.push(definition.id.clone());
        }
        index.by_id.insert(definition.id.clone(), definition);
        Ok(())
    }

    /// Builder form of [`insert`](Self::insert).
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned. Use
    /// [`insert`](Self::insert) where a `Result` is preferred.
    pub fn with_definition(self, definition: AgentDefinition) -> Self {
        self.insert(definition)
            .expect("definition registry lock poisoned");
        self
    }

    /// Sets the id returned by [`DefinitionRegistry::default_id`].
    pub fn set_default_id(&self, id: impl Into<String>) -> Result<()> {
        self.lock()?.default_id = Some(id.into());
        Ok(())
    }

    /// Builder form of [`set_default_id`](Self::set_default_id).
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn with_default_id(self, id: impl Into<String>) -> Self {
        self.set_default_id(id)
            .expect("definition registry lock poisoned");
        self
    }

    /// Returns the number of registered definitions.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.order.len())
    }

    /// Returns `true` when no definitions are registered.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.lock()?.order.is_empty())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, DefinitionIndex>> {
        self.inner.lock().map_err(|e| {
            TinyAgentsError::Validation(format!("definition registry lock poisoned: {e}"))
        })
    }
}

#[async_trait]
impl<State: Send + Sync> DefinitionRegistry<State> for InMemoryDefinitionRegistry {
    async fn get(&self, _state: &State, id: &str) -> Result<Option<AgentDefinition>> {
        Ok(self.lock()?.by_id.get(id).cloned())
    }

    async fn list(&self, _state: &State) -> Result<Vec<AgentDefinition>> {
        let index = self.lock()?;
        Ok(index
            .order
            .iter()
            .filter_map(|id| index.by_id.get(id).cloned())
            .collect())
    }

    async fn default_id(&self, _state: &State) -> Result<Option<String>> {
        Ok(self.lock()?.default_id.clone())
    }
}
