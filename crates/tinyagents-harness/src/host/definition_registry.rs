//! The agent **catalogue** seam: turning an agent id into a definition.
//!
//! The runtime needs three facts about an agent before it can run a turn —
//! which model it prefers, which tools it may reach, and which subagents it may
//! delegate to. Every one of those is *host* knowledge: it lives in the host's
//! profiles, its built-in agent tables, and its tier rules. This module states
//! the question the runtime is allowed to ask, and nothing about how the host
//! answers it.
//!
//! A host supplies a [`DefinitionRegistry`] **always** — this is a required
//! capability, not an optional one. A runtime with no catalogue cannot resolve
//! the agent it was asked to run, so there is no meaningful degraded mode. What
//! *is* optional is the catalogue's contents: an empty registry is a legitimate
//! answer, and the default [`InMemoryDefinitionRegistry`] built from an empty
//! `Vec` is exactly that.
//!
//! **Dependency rule:** the value type here is `serde` + `std` only, so a host
//! can build its mapper against [`AgentDefinition`] without pulling in the
//! runtime. Nothing in this module may name a host config schema, a security
//! policy, or a product type — the registry answers *what an agent is*, never
//! *whether the caller is allowed to have it*.
//!
//! # The absence contract
//!
//! [`DefinitionRegistry::resolve`] returning `Ok(None)` is a **normal outcome**,
//! not a soft error. See the method docs — this is the single most important
//! property of the trait and the easiest one to implement wrongly.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

// ── AgentDefinition ───────────────────────────────────────────────────────────

/// What the runtime is permitted to know about an agent.
///
/// Deliberately thin. The host's own agent record is far richer — prompts,
/// provenance, tier metadata, UI affordances — and none of it belongs here. The
/// fields below are the ones the turn loop actually reads; anything the runtime
/// cannot *act* on is host state and must stay host-side.
///
/// Ids are plain [`String`]s rather than a newtype. The crate has no `AgentId`
/// type, and inventing one here would force every host call site to wrap and
/// unwrap at the boundary for no checking benefit — the id is opaque to the
/// runtime, which only ever compares it or passes it back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AgentDefinition {
    /// Host-assigned identifier. Opaque to the runtime: it is compared for
    /// equality and echoed back, never parsed. Uniqueness within a catalogue is
    /// the host's responsibility.
    pub id: String,
    /// Human-readable name, for display and for prompt text that refers to the
    /// agent by name. Not an identity — never resolve on this.
    pub name: String,
    /// Short description of what the agent is for. Surfaced to a parent agent
    /// when it is choosing a delegate, so it should read as a capability
    /// summary rather than an implementation note.
    pub description: String,
    /// Preferred model id, when the agent pins one.
    ///
    /// `None` means "no preference" and the session default applies — it does
    /// **not** mean "no model". Encoding the fallback as `None` rather than as
    /// a baked-in default string keeps the choice of default with the host,
    /// which is the only party that knows what it has credentials for.
    #[serde(default)]
    pub model: Option<String>,
    /// Subagent ids this agent declares it may delegate to.
    ///
    /// Declared, not authorized: read [`DefinitionRegistry::delegates_for`] for
    /// the authorized set. Entries here may not resolve — see the absence
    /// contract on [`DefinitionRegistry::resolve`].
    #[serde(default)]
    pub subagents: Vec<String>,
    /// Tool names this agent may use. Opaque strings matched against the
    /// runtime's tool registry; an unmatched name is skipped, not fatal, for
    /// the same build-variance reason that unresolvable subagents are.
    #[serde(default)]
    pub tools: Vec<String>,
}

impl AgentDefinition {
    /// A definition with `id`, `name`, and `description` set and every optional
    /// field empty.
    ///
    /// The three required values have no sensible default: an agent with a
    /// blank id is unaddressable, and one with a blank description is invisible
    /// to a delegating parent. Everything else genuinely defaults to "none".
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            model: None,
            subagents: Vec::new(),
            tools: Vec::new(),
        }
    }

    /// Sets the pinned model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the declared subagent ids.
    pub fn with_subagents<I, S>(mut self, subagents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.subagents = subagents.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the tool names.
    pub fn with_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools = tools.into_iter().map(Into::into).collect();
        self
    }
}

// ── DefinitionRegistry ────────────────────────────────────────────────────────

/// Resolves an agent id to its [`AgentDefinition`].
///
/// This is a **required** capability: the runtime asks the host what an agent is
/// rather than reading a catalogue of its own, so there is always a registry
/// even when it is empty.
///
/// The trait is read-only on purpose. A catalogue that the runtime could mutate
/// would let a turn grant itself a delegate or a tool, which is precisely the
/// authority this seam exists to keep host-side.
#[async_trait]
pub trait DefinitionRegistry: Send + Sync {
    /// Looks up the definition for `id`.
    ///
    /// **`Ok(None)` for an unknown id is required behaviour, not an error
    /// condition.** A host's agent catalogue is data, and data legitimately
    /// names agents that are not present in the running build — a compile-time
    /// feature gate can remove an agent while the catalogue entry that lists it
    /// as a subagent remains. Callers are expected to skip an unresolved id and
    /// carry on; a registry that returned `Err` instead would turn an ordinary
    /// build variance into a failed run.
    ///
    /// Reserve `Err` for a genuine failure to *answer* — a broken backing store,
    /// a lost connection. The distinction is the whole point: absence must stay
    /// distinguishable from failure.
    async fn resolve(&self, id: &str) -> Result<Option<AgentDefinition>>;

    /// Every definition in the catalogue.
    ///
    /// Used to describe the available agents to a delegating parent, so the
    /// order should be stable across calls — an unstable order reshuffles the
    /// prompt text between turns and invalidates the provider's cached prefix.
    async fn list(&self) -> Result<Vec<AgentDefinition>>;

    /// Subagent ids `id` may delegate to, **already tier-checked by the host**.
    ///
    /// The runtime must treat the returned ids as authorized and must not
    /// re-derive them from [`AgentDefinition::subagents`], which is only what an
    /// agent *declares*. Tier rules, per-user overrides, and depth policy all
    /// live with the host; duplicating any of that here would create a second
    /// place for the answer to be wrong.
    ///
    /// Returns an empty `Vec` for an unknown `id`, for the same reason
    /// [`resolve`](Self::resolve) returns `Ok(None)`: an id that is not in this
    /// build has no delegates, which is a fact, not a fault. Individual returned
    /// ids may still fail to `resolve` — being authorized to delegate to an
    /// agent and that agent being compiled in are independent.
    async fn delegates_for(&self, id: &str) -> Result<Vec<String>>;
}

// ── InMemoryDefinitionRegistry ────────────────────────────────────────────────

/// A fixed catalogue held in memory.
///
/// The default implementation: enough to run tests, examples, and single-agent
/// embeddings without a host, and nothing more. It has no notion of tiers, so
/// [`delegates_for`](DefinitionRegistry::delegates_for) can only return what an
/// agent declares — a real host must not use it where authorization matters.
///
/// Construction deduplicates by id, **first entry wins**. Later duplicates are
/// dropped rather than overwriting, so that
/// [`list`](DefinitionRegistry::list) and [`resolve`](DefinitionRegistry::resolve)
/// can never disagree about which definition a duplicated id refers to. Silent
/// last-wins shadowing is the more surprising of the two failure modes.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDefinitionRegistry {
    /// Definitions in insertion order, deduplicated by id. Ordering is
    /// preserved because `list` must be stable across calls.
    definitions: Vec<AgentDefinition>,
    /// Index into `definitions` by id, so `resolve` does not scan.
    index: HashMap<String, usize>,
}

impl InMemoryDefinitionRegistry {
    /// Builds a registry from `definitions`, keeping the first entry for any
    /// repeated id and preserving insertion order otherwise.
    pub fn new(definitions: Vec<AgentDefinition>) -> Self {
        let mut kept: Vec<AgentDefinition> = Vec::with_capacity(definitions.len());
        let mut index: HashMap<String, usize> = HashMap::with_capacity(definitions.len());

        for definition in definitions {
            if index.contains_key(&definition.id) {
                continue;
            }
            index.insert(definition.id.clone(), kept.len());
            kept.push(definition);
        }

        Self {
            definitions: kept,
            index,
        }
    }

    /// An empty catalogue. Every lookup misses; nothing errors.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Number of retained definitions (after deduplication).
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Whether the catalogue holds no definitions.
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }
}

impl FromIterator<AgentDefinition> for InMemoryDefinitionRegistry {
    fn from_iter<I: IntoIterator<Item = AgentDefinition>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

#[async_trait]
impl DefinitionRegistry for InMemoryDefinitionRegistry {
    async fn resolve(&self, id: &str) -> Result<Option<AgentDefinition>> {
        Ok(self
            .index
            .get(id)
            .and_then(|position| self.definitions.get(*position))
            .cloned())
    }

    async fn list(&self) -> Result<Vec<AgentDefinition>> {
        Ok(self.definitions.clone())
    }

    async fn delegates_for(&self, id: &str) -> Result<Vec<String>> {
        // No tier model here, so the declared set is the only answer available.
        // Declared ids are returned verbatim, including any that will not
        // resolve — filtering them would hide a catalogue mistake behind a
        // shorter list, and callers already handle an unresolvable delegate.
        Ok(self
            .index
            .get(id)
            .and_then(|position| self.definitions.get(*position))
            .map(|definition| definition.subagents.clone())
            .unwrap_or_default())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lead() -> AgentDefinition {
        AgentDefinition::new("lead", "Lead", "Coordinates the run")
            .with_model("test-model")
            .with_subagents(["researcher", "compiled_out"])
            .with_tools(["read_file", "write_file"])
    }

    fn researcher() -> AgentDefinition {
        AgentDefinition::new("researcher", "Researcher", "Gathers material")
    }

    fn registry() -> InMemoryDefinitionRegistry {
        InMemoryDefinitionRegistry::new(vec![lead(), researcher()])
    }

    #[test]
    fn builder_leaves_optional_fields_empty() {
        let definition = AgentDefinition::new("a", "A", "does a");
        assert_eq!(definition.model, None);
        assert!(definition.subagents.is_empty());
        assert!(definition.tools.is_empty());
    }

    #[test]
    fn builder_setters_are_chainable() {
        let definition = lead();
        assert_eq!(definition.model.as_deref(), Some("test-model"));
        assert_eq!(definition.subagents, vec!["researcher", "compiled_out"]);
        assert_eq!(definition.tools, vec!["read_file", "write_file"]);
    }

    #[test]
    fn definition_round_trips_through_json() {
        let definition = lead();
        let encoded = serde_json::to_string(&definition).expect("serialize");
        let decoded: AgentDefinition = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, definition);
    }

    #[test]
    fn omitted_optional_fields_deserialize_to_empty() {
        let decoded: AgentDefinition =
            serde_json::from_str(r#"{"id":"a","name":"A","description":"does a"}"#)
                .expect("deserialize");
        assert_eq!(decoded, AgentDefinition::new("a", "A", "does a"));
    }

    #[tokio::test]
    async fn resolve_returns_the_definition_for_a_known_id() {
        let resolved = registry().resolve("lead").await.expect("resolve");
        assert_eq!(resolved, Some(lead()));
    }

    #[tokio::test]
    async fn resolve_returns_none_for_an_unknown_id_without_erroring() {
        // The absence contract: a catalogue may name agents this build lacks.
        let resolved = registry().resolve("compiled_out").await.expect("resolve");
        assert_eq!(resolved, None);
    }

    #[tokio::test]
    async fn empty_registry_misses_every_lookup() {
        let registry = InMemoryDefinitionRegistry::empty();
        assert!(registry.is_empty());
        assert_eq!(registry.resolve("lead").await.expect("resolve"), None);
        assert!(registry.list().await.expect("list").is_empty());
        assert!(
            registry
                .delegates_for("lead")
                .await
                .expect("delegates")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn list_preserves_insertion_order_across_calls() {
        let registry = registry();
        let first = registry.list().await.expect("list");
        let second = registry.list().await.expect("list");
        assert_eq!(
            first.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            vec!["lead", "researcher"]
        );
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn duplicate_ids_keep_the_first_entry() {
        let shadow = AgentDefinition::new("lead", "Shadow", "should not win");
        let registry = InMemoryDefinitionRegistry::new(vec![lead(), shadow, researcher()]);

        assert_eq!(registry.len(), 2);
        assert_eq!(
            registry.resolve("lead").await.expect("resolve"),
            Some(lead())
        );
        assert_eq!(
            registry
                .list()
                .await
                .expect("list")
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["lead", "researcher"]
        );
    }

    #[tokio::test]
    async fn delegates_for_returns_declared_ids_including_unresolvable_ones() {
        let registry = registry();
        let delegates = registry.delegates_for("lead").await.expect("delegates");
        assert_eq!(delegates, vec!["researcher", "compiled_out"]);
        // The unresolvable delegate is still listed; resolving it is the
        // caller's problem and yields `None`, not an error.
        assert_eq!(
            registry.resolve("compiled_out").await.expect("resolve"),
            None
        );
    }

    #[tokio::test]
    async fn delegates_for_is_empty_for_an_unknown_id() {
        let delegates = registry().delegates_for("nobody").await.expect("delegates");
        assert!(delegates.is_empty());
    }

    #[tokio::test]
    async fn delegates_for_is_empty_when_none_are_declared() {
        let delegates = registry()
            .delegates_for("researcher")
            .await
            .expect("delegates");
        assert!(delegates.is_empty());
    }

    #[tokio::test]
    async fn is_usable_as_a_trait_object() {
        let registry: Box<dyn DefinitionRegistry> = Box::new(registry());
        assert!(
            registry
                .resolve("researcher")
                .await
                .expect("resolve")
                .is_some()
        );
    }

    #[test]
    fn collects_from_an_iterator_with_the_same_dedupe_rule() {
        let registry: InMemoryDefinitionRegistry =
            vec![lead(), researcher(), lead()].into_iter().collect();
        assert_eq!(registry.len(), 2);
    }
}
