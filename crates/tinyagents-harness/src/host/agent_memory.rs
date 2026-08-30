//! Host capability: durable agent memory.
//!
//! This is the largest of the host seams described in the host-capability
//! traits RFC (§3.1). A host that keeps long-lived knowledge about a user — a
//! memory store, a summary tree, a citation index — implements
//! [`AgentMemory`] and hands it to the runtime; a host with no such store
//! simply supplies nothing.
//!
//! **The capability is optional, and absence is expressed by the host passing
//! `None`, never by an implementation that returns an error.** An erroring stub
//! would teach the model that memory exists and make it retry a call that can
//! never succeed, so "this deployment has no memory" and "the memory backend is
//! down" must stay distinguishable: the first is `None` at wiring time, the
//! second is an `Err` from a real implementation.
//!
//! # What the runtime is *not* allowed to do
//!
//! The runtime must not learn what a memory is. Items returned by
//! [`AgentMemory::recall`] have **already** been scope-filtered and redacted by
//! the host, so the runtime injects them as it received them: no re-ranking, no
//! second filtering pass, no truncation-by-relevance of its own. Re-ranking
//! would silently reorder a host's access-control decision, and re-filtering
//! would let runtime heuristics drop text the host had deliberately kept.
//! Symmetrically, [`AgentMemory::remember`] carries no provenance, taint, or
//! trust field — stamping those is the host's job, and a runtime-supplied
//! provenance value would be self-attested and therefore worthless as a
//! security signal.
//!
//! # Value types
//!
//! Everything in this module is inert: `serde` + `std` only, no tokio, no
//! storage engine, no host schema type. A host must be able to build a mapper
//! against these structs without pulling in the runtime's machinery.

use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ids::{ThreadId, next_seq, process_nonce};

// ── MemoryId ──────────────────────────────────────────────────────────────────

/// Opaque host-assigned identifier for one stored memory.
///
/// Defined here rather than in [`crate::ids`] because a memory id is
/// part of *this* capability's contract, not of the runtime's own
/// run/thread/call lineage: only a host that implements [`AgentMemory`] ever
/// mints one, and the runtime never parses, orders, or correlates it. It is a
/// newtype rather than a bare `String` so it cannot be passed where a
/// [`ThreadId`] is expected.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(String);

impl MemoryId {
    /// Creates a memory id from anything convertible into a `String`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for MemoryId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for MemoryId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

// ── RecallRequest ─────────────────────────────────────────────────────────────

/// What the runtime is asking the host to recall for the current turn.
///
/// Every field except `query` is a *hint about the caller*, not an instruction
/// about storage. The host decides what those hints mean — which namespaces
/// they open, which they close — because scoping is an access-control decision
/// and the runtime is not in a position to make it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecallRequest {
    /// The text to find memory relevant to, usually the turn's user input.
    ///
    /// Free-form and untokenized: whether the host treats it as a vector query,
    /// a keyword search, or something else is entirely the host's choice.
    pub query: String,
    /// Id of the agent making the request, if the turn has one.
    ///
    /// A plain `String` because this crate has no `AgentId` newtype — agent
    /// identity is host-defined, and inventing a runtime type for it would
    /// imply the runtime validates ids it cannot validate.
    #[serde(default)]
    pub agent_id: Option<String>,
    /// Thread the turn belongs to, when the request is thread-scoped.
    #[serde(default)]
    pub thread_id: Option<ThreadId>,
    /// Maximum number of items the runtime can make use of.
    ///
    /// An upper bound, not a target: returning fewer is normal, and a host may
    /// return fewer than requested for its own reasons. `None` means the
    /// runtime imposes no cap and the host's own default applies.
    #[serde(default)]
    pub limit: Option<usize>,
}

impl RecallRequest {
    /// An unscoped request for `query` with no agent, thread, or limit hint.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            agent_id: None,
            thread_id: None,
            limit: None,
        }
    }

    /// Scopes the request to `thread`.
    pub fn with_thread(mut self, thread_id: ThreadId) -> Self {
        self.thread_id = Some(thread_id);
        self
    }

    /// Attaches the requesting agent's host-defined id.
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Caps how many items the runtime can use.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

// ── MemoryItem ────────────────────────────────────────────────────────────────

/// One recalled memory, exactly as the host chose to expose it.
///
/// Treat this as final. It has already passed the host's scope filter and
/// redaction pass, so the runtime's only job is to inject `text` and, if a
/// surface wants attribution, carry `citation` through untouched.
// No `Eq`: `score` is an `f32`, which is only `PartialEq`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Host-assigned id of the stored memory.
    pub id: MemoryId,
    /// The memory's text, already redacted by the host and safe to inject.
    pub text: String,
    /// The host's own relevance score, when it computes one.
    ///
    /// Advisory and **not** comparable across hosts: the scale, direction, and
    /// meaning are the host's. The runtime must not sort by it — items arrive
    /// in the order the host intends them to be read. `None` is normal for a
    /// host whose retrieval has no notion of a score.
    #[serde(default)]
    pub score: Option<f32>,
    /// Opaque host string identifying where this memory came from.
    ///
    /// Deliberately a `String` and not a structured citation type: a host's
    /// citation shape is a UI contract owned by its frontend, and pulling that
    /// shape into a redistributed crate would couple the crate's version to a
    /// product's rendering. The runtime passes it through and never parses it.
    #[serde(default)]
    pub citation: Option<String>,
}

impl MemoryItem {
    /// A recalled item with no score and no citation.
    pub fn new(id: impl Into<MemoryId>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            score: None,
            citation: None,
        }
    }

    /// Attaches the host's relevance score.
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }

    /// Attaches the host's opaque citation string.
    pub fn with_citation(mut self, citation: impl Into<String>) -> Self {
        self.citation = Some(citation.into());
        self
    }
}

// ── NewMemory ─────────────────────────────────────────────────────────────────

/// A memory a turn produced and asks the host to store durably.
///
/// There is intentionally no provenance, trust, taint, or timestamp field. The
/// host stamps all of those: it knows the transport the text arrived over and
/// whether the turn touched untrusted content, and a value the runtime supplied
/// would be self-attested and unusable as a security signal.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewMemory {
    /// The text to store. Storing it is the host's decision — a host may
    /// deduplicate, rewrite, split, or reject it outright.
    pub text: String,
    /// Id of the agent that produced the memory, if the turn has one. A plain
    /// `String` for the same reason as [`RecallRequest::agent`].
    #[serde(default)]
    pub agent_id: Option<String>,
    /// Thread the memory came from, when it is thread-scoped.
    #[serde(default)]
    pub thread_id: Option<ThreadId>,
    /// Runtime-suggested labels. Advisory only: the host may keep, remap, or
    /// discard them, and must not treat them as an authorization scope.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl NewMemory {
    /// A memory carrying only `text`.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            agent_id: None,
            thread_id: None,
            tags: Vec::new(),
        }
    }

    /// Scopes the memory to `thread`.
    pub fn with_thread(mut self, thread_id: ThreadId) -> Self {
        self.thread_id = Some(thread_id);
        self
    }

    /// Attributes the memory to the producing agent's host-defined id.
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Adds an advisory label.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

// ── AgentMemory ───────────────────────────────────────────────────────────────

/// Durable memory a host lends to the runtime.
///
/// Optional: a host that has no memory store passes nothing rather than
/// wiring an implementation that always errors (see the module docs for why
/// that distinction is load-bearing).
///
/// Implementations are shared across concurrent turns, hence `Send + Sync`, and
/// every method takes `&self` — a host that needs interior mutability owns that
/// choice, since the runtime cannot know whether the backing store is a local
/// mutex or a remote service.
#[async_trait]
pub trait AgentMemory: Send + Sync {
    /// Retrieves memory relevant to the request for injection into a turn's
    /// context.
    ///
    /// Returned items are already scope-filtered and redacted by the host. The
    /// runtime **must not** re-rank or re-filter them, and must inject them in
    /// the order given.
    ///
    /// An empty vector means "nothing relevant", which is an ordinary result
    /// and not a failure; reserve `Err` for a backend that could not answer.
    async fn recall(&self, req: RecallRequest) -> Result<Vec<MemoryItem>>;

    /// Records a durable memory produced by a turn and returns the host's id
    /// for it.
    ///
    /// The host owns provenance and taint stamping; the runtime never supplies
    /// either. A host that deduplicates may legitimately return the id of an
    /// existing memory rather than a fresh one, so callers must not treat the
    /// returned id as proof that a new record was created.
    async fn remember(&self, item: NewMemory) -> Result<MemoryId>;

    /// Rolled-up prior context for a thread, as opaque prose the runtime
    /// injects verbatim.
    ///
    /// `Ok(None)` means the host has no summary for this thread — a new thread,
    /// or a host that does not summarize at all. That is distinct from `Err`,
    /// which means the host could not tell. The runtime must not synthesize a
    /// substitute from recalled items when this returns `None`: a summary is a
    /// host-authored artifact, and a runtime-assembled one would be
    /// indistinguishable from it downstream.
    async fn thread_summary(&self, thread: &ThreadId) -> Result<Option<String>>;
}

// ── InMemoryAgentMemory ───────────────────────────────────────────────────────

/// One stored record inside [`InMemoryAgentMemory`].
#[derive(Clone, Debug)]
struct StoredMemory {
    id: MemoryId,
    text: String,
    agent_id: Option<String>,
    thread_id: Option<ThreadId>,
}

/// Process-local [`AgentMemory`] backed by a `Mutex<Vec<_>>`.
///
/// Intended for unit tests, examples, and short local runs. Nothing is durable:
/// every record is lost when the value is dropped.
///
/// Recall is a case-insensitive **substring** match, deliberately: a test
/// double should be trivially predictable, and any similarity scoring here
/// would invite tests to assert on ranking behaviour that no real host is
/// obliged to reproduce. For the same reason it never sets
/// [`MemoryItem::score`] or [`MemoryItem::citation`] — a runtime that only
/// works when a score is present would break against a host that supplies none.
///
/// This type is **not** a stand-in for the capability being absent. A host
/// without memory passes `None`; this is a working implementation that happens
/// to forget everything on drop.
#[derive(Debug, Default)]
pub struct InMemoryAgentMemory {
    records: Mutex<Vec<StoredMemory>>,
}

impl InMemoryAgentMemory {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of records currently held.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the store holds no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every record.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// Locks the record list, recovering from a poisoned mutex.
    ///
    /// A panic in another thread must not turn this test double into a
    /// permanently erroring capability — that is exactly the "absence looks
    /// like failure" confusion the trait is designed to avoid — so the guard is
    /// taken from the poison error rather than propagated.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<StoredMemory>> {
        self.records.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether a stored record is visible to a request scoped this way.
    ///
    /// A record stored without a scope is unscoped and visible to every
    /// request; a scoped record is visible only to a request naming the same
    /// scope. A request that names no scope sees everything.
    fn scope_matches<T: PartialEq>(requested: Option<&T>, stored: Option<&T>) -> bool {
        match (requested, stored) {
            (Some(requested), Some(stored)) => requested == stored,
            (Some(_), None) => true,
            (None, _) => true,
        }
    }
}

#[async_trait]
impl AgentMemory for InMemoryAgentMemory {
    /// Returns every in-scope record whose text contains `req.query`
    /// case-insensitively, in insertion order, truncated to `req.limit`.
    ///
    /// A blank query matches every in-scope record, since every string contains
    /// the empty string. That is the useful behaviour for a test double — "give
    /// me what you have" — and callers that want nothing back should not call.
    async fn recall(&self, req: RecallRequest) -> Result<Vec<MemoryItem>> {
        let needle = req.query.to_lowercase();
        let records = self.lock();

        let mut out: Vec<MemoryItem> = records
            .iter()
            .filter(|record| {
                Self::scope_matches(req.thread_id.as_ref(), record.thread_id.as_ref())
                    && Self::scope_matches(req.agent_id.as_ref(), record.agent_id.as_ref())
                    && record.text.to_lowercase().contains(&needle)
            })
            .map(|record| MemoryItem::new(record.id.clone(), record.text.clone()))
            .collect();

        if let Some(limit) = req.limit {
            out.truncate(limit);
        }
        Ok(out)
    }

    /// Appends the memory and returns a freshly minted, process-unique id.
    ///
    /// No deduplication: storing the same text twice yields two records with
    /// distinct ids, so a test can tell repeated writes apart.
    async fn remember(&self, item: NewMemory) -> Result<MemoryId> {
        let id = MemoryId::new(format!("mem-{}-{}", process_nonce(), next_seq()));
        self.lock().push(StoredMemory {
            id: id.clone(),
            text: item.text,
            agent_id: item.agent_id,
            thread_id: item.thread_id,
        });
        Ok(id)
    }

    /// Always `Ok(None)`.
    ///
    /// Summaries are host-authored prose. Concatenating stored records into
    /// something summary-shaped would hand the runtime a synthetic artifact it
    /// could not distinguish from a real host summary, and would make tests
    /// pass against behaviour no host guarantees.
    async fn thread_summary(&self, _thread: &ThreadId) -> Result<Option<String>> {
        Ok(None)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn thread(id: &str) -> ThreadId {
        ThreadId::new(id)
    }

    #[test]
    fn memory_id_round_trips_through_string_conversions() {
        let id = MemoryId::from("mem-1");
        assert_eq!(id.as_str(), "mem-1");
        assert_eq!(id.to_string(), "mem-1");
        assert_eq!(MemoryId::new("mem-1"), id);
        assert_eq!(MemoryId::from(String::from("mem-1")), id);
    }

    #[test]
    fn recall_request_builders_compose() {
        let req = RecallRequest::new("hello")
            .with_thread(thread("t1"))
            .with_agent("planner")
            .with_limit(3);

        assert_eq!(req.query, "hello");
        assert_eq!(req.thread_id, Some(thread("t1")));
        assert_eq!(req.agent_id.as_deref(), Some("planner"));
        assert_eq!(req.limit, Some(3));
    }

    #[test]
    fn new_memory_defaults_carry_no_provenance_fields() {
        let item = NewMemory::new("a fact");
        assert_eq!(item.text, "a fact");
        assert!(item.agent_id.is_none());
        assert!(item.thread_id.is_none());
        assert!(item.tags.is_empty());

        let tagged = NewMemory::new("a fact").with_tag("x").with_tag("y");
        assert_eq!(tagged.tags, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn memory_item_defaults_to_no_score_and_no_citation() {
        let item = MemoryItem::new("mem-1", "text");
        assert!(item.score.is_none());
        assert!(item.citation.is_none());

        let decorated = MemoryItem::new("mem-1", "text")
            .with_score(0.5)
            .with_citation("opaque://host/1");
        assert_eq!(decorated.score, Some(0.5));
        assert_eq!(decorated.citation.as_deref(), Some("opaque://host/1"));
    }

    #[test]
    fn memory_item_serde_round_trips() {
        let item = MemoryItem::new("mem-1", "text").with_citation("opaque://host/1");
        let json = serde_json::to_string(&item).expect("serialize");
        let back: MemoryItem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, item);
    }

    #[tokio::test]
    async fn empty_store_recalls_nothing_without_erroring() {
        let store = InMemoryAgentMemory::new();
        assert!(store.is_empty());

        let items = store.recall(RecallRequest::new("anything")).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn remember_then_recall_matches_case_insensitive_substrings() {
        let store = InMemoryAgentMemory::new();
        let id = store
            .remember(NewMemory::new("The Sky Is Blue"))
            .await
            .unwrap();

        let hit = store.recall(RecallRequest::new("sky is")).await.unwrap();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].id, id);
        assert_eq!(hit[0].text, "The Sky Is Blue");
        assert!(hit[0].score.is_none(), "test double must not invent scores");
        assert!(hit[0].citation.is_none());

        let miss = store.recall(RecallRequest::new("green")).await.unwrap();
        assert!(miss.is_empty());
    }

    #[tokio::test]
    async fn remember_mints_a_distinct_id_per_write() {
        let store = InMemoryAgentMemory::new();
        let first = store.remember(NewMemory::new("same text")).await.unwrap();
        let second = store.remember(NewMemory::new("same text")).await.unwrap();

        assert_ne!(first, second);
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn recall_preserves_insertion_order_and_honours_the_limit() {
        let store = InMemoryAgentMemory::new();
        for text in ["note one", "note two", "note three"] {
            store.remember(NewMemory::new(text)).await.unwrap();
        }

        let all = store.recall(RecallRequest::new("note")).await.unwrap();
        let texts: Vec<&str> = all.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(texts, vec!["note one", "note two", "note three"]);

        let capped = store
            .recall(RecallRequest::new("note").with_limit(2))
            .await
            .unwrap();
        assert_eq!(capped.len(), 2);
        assert_eq!(capped[0].text, "note one");
    }

    #[tokio::test]
    async fn blank_query_returns_every_in_scope_record() {
        let store = InMemoryAgentMemory::new();
        store.remember(NewMemory::new("alpha")).await.unwrap();
        store.remember(NewMemory::new("beta")).await.unwrap();

        let all = store.recall(RecallRequest::new("")).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn thread_scoping_hides_other_threads_but_keeps_unscoped_records() {
        let store = InMemoryAgentMemory::new();
        store
            .remember(NewMemory::new("scoped one").with_thread(thread("t1")))
            .await
            .unwrap();
        store
            .remember(NewMemory::new("scoped two").with_thread(thread("t2")))
            .await
            .unwrap();
        store.remember(NewMemory::new("scoped none")).await.unwrap();

        let scoped = store
            .recall(RecallRequest::new("scoped").with_thread(thread("t1")))
            .await
            .unwrap();
        let texts: Vec<&str> = scoped.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(texts, vec!["scoped one", "scoped none"]);

        let unscoped = store.recall(RecallRequest::new("scoped")).await.unwrap();
        assert_eq!(unscoped.len(), 3);
    }

    #[tokio::test]
    async fn agent_scoping_hides_other_agents_but_keeps_unscoped_records() {
        let store = InMemoryAgentMemory::new();
        store
            .remember(NewMemory::new("fact a").with_agent("planner"))
            .await
            .unwrap();
        store
            .remember(NewMemory::new("fact b").with_agent("researcher"))
            .await
            .unwrap();
        store.remember(NewMemory::new("fact c")).await.unwrap();

        let scoped = store
            .recall(RecallRequest::new("fact").with_agent("planner"))
            .await
            .unwrap();
        let texts: Vec<&str> = scoped.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(texts, vec!["fact a", "fact c"]);
    }

    #[tokio::test]
    async fn thread_summary_is_none_even_after_remembering_into_that_thread() {
        let store = InMemoryAgentMemory::new();
        store
            .remember(NewMemory::new("something").with_thread(thread("t1")))
            .await
            .unwrap();

        assert!(store.thread_summary(&thread("t1")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn clear_drops_every_record() {
        let store = InMemoryAgentMemory::new();
        store.remember(NewMemory::new("gone soon")).await.unwrap();
        assert_eq!(store.len(), 1);

        store.clear();
        assert!(store.is_empty());
        assert!(
            store
                .recall(RecallRequest::new(""))
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn usable_behind_a_trait_object() {
        let store: Box<dyn AgentMemory> = Box::new(InMemoryAgentMemory::new());
        store.remember(NewMemory::new("dyn safe")).await.unwrap();
        assert_eq!(
            store.recall(RecallRequest::new("dyn")).await.unwrap().len(),
            1
        );
    }
}
