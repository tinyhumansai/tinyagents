//! Retrieval over prior-run outcomes the embedder has chosen to retain.
//!
//! See [`ExperienceStore`] for the trait contract and [`NoopExperienceStore`]
//! for the inert default.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;

/// Retrieval over prior-run outcomes the embedder has chosen to retain.
///
/// Distinct from [`MemoryProvider`][super::MemoryProvider], which serves the
/// user's corpus: this serves the agent's own record of what worked. The crate
/// stores nothing and renders nothing — it hands back `body` verbatim for a
/// [`ContextComposer`][super::ContextComposer] to place.
#[async_trait]
pub trait ExperienceStore<State: Send + Sync>: Send + Sync {
    /// Returns at most `max_hits` prior outcomes relevant to `query`.
    ///
    /// Hits with an empty `match_reasons` carry no evidence for why they were
    /// selected and callers may drop them.
    async fn retrieve(
        &self,
        state: &State,
        query: &ExperienceQuery<'_>,
    ) -> Result<Vec<ExperienceHit>>;

    /// Offers candidate outcomes for retention. The default discards them, so
    /// a read-only embedder implements one method.
    async fn record(&self, state: &State, entries: Vec<ExperienceEntry>) -> Result<()> {
        let _ = (state, entries);
        Ok(())
    }
}

/// A relevance query against retained prior-run outcomes.
#[derive(Clone, Debug, Default)]
pub struct ExperienceQuery<'a> {
    /// Free text the backend scores outcomes against.
    pub text: &'a str,
    /// Restrict to outcomes recorded by one agent.
    pub agent_id: Option<&'a str>,
    /// Host-defined label for how the run was entered.
    pub entrypoint: Option<&'a str>,
    /// Opaque partition key. `None` searches every partition.
    pub partition: Option<&'a str>,
    /// Tools available to this run; backends may weight hits that used them.
    pub tool_names: &'a [String],
    /// Maximum hits to return.
    pub max_hits: usize,
}

/// One retained outcome, already rendered by the host.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceHit {
    /// Backend-assigned identity of the outcome.
    pub id: String,
    /// Host-rendered text, used verbatim.
    pub body: String,
    /// Relevance score, when the backend computes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Why this hit matched, as host-defined labels.
    #[serde(default)]
    pub match_reasons: Vec<String>,
}

/// A candidate outcome offered for retention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceEntry {
    /// Host-defined identity for the outcome.
    pub id: String,
    /// Agent that produced it, when attributed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Opaque partition key to file it under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition: Option<String>,
    /// Opaque host payload; the crate never inspects it.
    pub payload: Value,
}

/// An [`ExperienceStore`] that retains nothing and returns nothing.
///
/// Zero state, zero allocation: `retrieve` returns an empty `Vec` and `record`
/// takes the discarding trait default.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopExperienceStore;

impl NoopExperienceStore {
    /// Creates the store.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State: Send + Sync> ExperienceStore<State> for NoopExperienceStore {
    async fn retrieve(
        &self,
        _state: &State,
        _query: &ExperienceQuery<'_>,
    ) -> Result<Vec<ExperienceHit>> {
        Ok(Vec::new())
    }
}
