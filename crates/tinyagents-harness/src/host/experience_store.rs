//! Procedural experience: what an agent learned about *doing* a kind of task.
//!
//! An [`ExperienceStore`] is the runtime's record of **how it performed** —
//! "the last three times this agent was asked to migrate a schema, two runs
//! succeeded and one failed after the same missing-permission step". The
//! runtime writes an [`Experience`] when a task finishes and reads prior ones
//! back before attempting a similar task, so a repeated task can start from the
//! shape of the previous attempt instead of from zero.
//!
//! # Not `AgentMemory` — the mistake worth stating explicitly
//!
//! These two capabilities look alike and are not interchangeable:
//!
//! | | `AgentMemory` | `ExperienceStore` |
//! |---|---|---|
//! | Whose knowledge | the **user's** | the **agent's** |
//! | Kind | declarative — facts, preferences, prior conversation | procedural — task attempts and their outcomes |
//! | Written by | a turn deciding something is worth durably remembering | the runtime, mechanically, when a task completes |
//! | Lifetime | as long as the fact is true | as long as the strategy is informative |
//! | Wrong to leak | into a task-strategy prompt | into anything user-facing |
//!
//! Conflating them fails in both directions. Recording task outcomes through
//! `AgentMemory` pollutes the user's durable knowledge with runtime bookkeeping
//! ("attempt 4 failed") that will later be recalled and injected as if the user
//! had said it. Recalling user facts through `ExperienceStore` presents personal
//! data as procedural evidence, and — because experience is keyed by agent id
//! rather than by the host's memory scope — bypasses whatever scoping and
//! redaction the host applies to real memory. A host that has only one backing
//! store must still keep the two namespaces separate.
//!
//! # Optional capability
//!
//! This is optional (RFC §3.9). A host that does not track performance history
//! supplies `None` and the runtime skips both calls entirely. It must **not**
//! supply an implementation whose methods return errors: an erroring stub is
//! indistinguishable from a store that is present but broken, and the runtime
//! would report a real failure for a capability the host simply declined.
//! [`InMemoryExperienceStore`] is for tests and short-lived local runs, not a
//! stand-in for "no store" — it *does* retain and return experiences.
//!
//! # Agent identity
//!
//! The RFC writes `recall_for(&self, agent: &AgentId, …)`. The crate has no
//! `AgentId` type, and introducing one here would force every host and every
//! sibling trait to convert at the boundary for no gain, so agent identity is
//! `&str` throughout — opaque to the runtime, matched only for equality.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

// ── Experience ────────────────────────────────────────────────────────────────

/// One agent's record of one attempt at one task.
///
/// Inert by construction: `serde` + `std` only, no ids, no timestamps, no
/// engine types. A host can persist it with any backing store, and a host that
/// wants provenance (when, in which run, under whose scope) stamps that on its
/// own row — the runtime deliberately does not supply it, exactly as it does
/// not supply provenance for memory.
///
/// `outcome` is prose, not a structured result, because the useful part of a
/// prior attempt is rarely machine-shaped: the reason a step failed, the
/// ordering that worked. It is fed back to a model as context, so a host must
/// treat it as untrusted text on the way in and out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Experience {
    /// Opaque host-defined id of the agent that made the attempt. Matched by
    /// equality only; the runtime never parses or namespaces it.
    pub agent_id: String,
    /// What was attempted, in the words the task was posed in. This is the key
    /// recall matches against, so it should describe the task rather than name
    /// a run.
    pub task: String,
    /// What happened, as prose worth reading before a similar attempt. May be
    /// empty when there is nothing informative to say beyond `success`.
    pub outcome: String,
    /// Whether the attempt achieved what was asked. A failed attempt is often
    /// the more valuable record — it is why this flag exists separately from
    /// `outcome` rather than being left for a reader to infer from the prose.
    pub success: bool,
}

impl Experience {
    /// A recorded attempt, defaulting to failure.
    ///
    /// Failure is the default because an attempt that never reached an explicit
    /// success is, from the runtime's point of view, not a success — a
    /// constructor that defaulted the other way would silently record crashed
    /// and abandoned runs as wins.
    pub fn new(
        agent_id: impl Into<String>,
        task: impl Into<String>,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            task: task.into(),
            outcome: outcome.into(),
            success: false,
        }
    }

    /// Marks this attempt as successful (`success = true`).
    pub fn succeeded(mut self) -> Self {
        self.success = true;
        self
    }

    /// Whether this record carries anything worth storing.
    ///
    /// An experience with no agent or no task cannot be recalled by
    /// [`ExperienceStore::recall_for`] — nothing would ever match it — so
    /// storing it only grows the store. Implementations may skip such records;
    /// [`InMemoryExperienceStore`] does, and reports success rather than an
    /// error, since an unrecallable record is not a host failure.
    pub fn is_recallable(&self) -> bool {
        !self.agent_id.trim().is_empty() && !self.task.trim().is_empty()
    }
}

// ── ExperienceStore ───────────────────────────────────────────────────────────

/// Records and recalls an agent's own performance history.
///
/// Both methods are `async` because a host will typically back this with the
/// same durable store it uses for everything else; neither is on a hot path.
#[async_trait]
pub trait ExperienceStore: Send + Sync {
    /// Records one completed attempt.
    ///
    /// Takes `&Experience` rather than an owned value so a caller that also
    /// reports the same outcome elsewhere (a `LearningSink`, an event journal)
    /// does not have to clone it. Implementations should treat a duplicate
    /// record as acceptable — the runtime makes no attempt to deduplicate, and
    /// a retried task genuinely is a second attempt.
    ///
    /// Returning `Err` means the record was *lost*, and callers may log it, but
    /// must not fail the turn over it: the task already finished, and the value
    /// of this capability is advisory.
    async fn record(&self, exp: &Experience) -> Result<()>;

    /// Returns prior attempts by `agent` at tasks resembling `task`.
    ///
    /// "Resembling" is the **host's** judgement, not the runtime's. A host with
    /// embeddings may match semantically; a host with a database may match on a
    /// key; the in-memory default matches on shared words. The runtime does not
    /// re-rank or re-filter what comes back — the same rule as `AgentMemory`
    /// recall — so an implementation should return results in the order it
    /// wants them read and bound the count itself.
    ///
    /// An empty `Vec` is the normal answer for a new agent or a novel task and
    /// must not be reported as an error; only a store failure is an `Err`.
    async fn recall_for(&self, agent_id: &str, task: &str) -> Result<Vec<Experience>>;
}

// ── InMemoryExperienceStore ───────────────────────────────────────────────────

/// Thread-safe, non-durable [`ExperienceStore`] for tests and short local runs.
///
/// Everything is lost when the value is dropped. Retention is bounded by
/// [`DEFAULT_CAPACITY`](Self::DEFAULT_CAPACITY): once full, the **oldest**
/// record is dropped on insert. Age, not relevance, is the eviction key —
/// judging relevance would mean inventing exactly the ranking policy this trait
/// exists to leave with the host.
///
/// This is a working store, not a "capability absent" placeholder; see the
/// module docs.
#[derive(Clone, Debug)]
pub struct InMemoryExperienceStore {
    inner: Arc<Mutex<BoundedExperienceLog>>,
}

/// Insertion-ordered, capacity-bounded log backing [`InMemoryExperienceStore`].
#[derive(Debug)]
struct BoundedExperienceLog {
    /// Records in oldest-to-newest order.
    entries: VecDeque<Experience>,
    /// Maximum records retained before the oldest is dropped.
    capacity: usize,
}

impl InMemoryExperienceStore {
    /// Records retained by [`new`](Self::new) and [`Default`].
    pub const DEFAULT_CAPACITY: usize = 512;

    /// An empty store bounded by [`DEFAULT_CAPACITY`](Self::DEFAULT_CAPACITY).
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// An empty store retaining at most `capacity` records.
    ///
    /// A `capacity` of zero is clamped to `1`, so the store always keeps the
    /// most recent attempt. A zero-capacity store that silently discarded
    /// everything would look identical to a store whose matching is broken.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BoundedExperienceLog {
                entries: VecDeque::new(),
                capacity: capacity.max(1),
            })),
        }
    }

    /// Number of records currently retained.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether the store holds no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Locks the log, recovering from a poisoned mutex.
    ///
    /// Matches [`InMemoryAgentMemory`](super::agent_memory::InMemoryAgentMemory)
    /// and [`RecordingProgressSink`](super::progress_sink::RecordingProgressSink):
    /// a panic in another thread must not turn this test double into a
    /// permanently erroring capability, which is exactly the "absence looks like
    /// failure" confusion these traits are designed to avoid. Poisoning is also
    /// not a *validation* problem, so surfacing it as one would have been the
    /// wrong error variant as well as the wrong policy.
    fn lock(&self) -> std::sync::MutexGuard<'_, BoundedExperienceLog> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for InMemoryExperienceStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Lowercased alphanumeric words of `text`, used for the default matcher.
///
/// Splitting on non-alphanumeric characters (rather than whitespace) keeps
/// punctuation from making `deploy.` and `deploy` different words. This is a
/// deliberately naive heuristic: it is here so the default store is *useful in
/// a test*, not so hosts inherit a ranking policy from it.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Whether two task descriptions share at least one word.
///
/// Identical-after-normalization tasks always match, including when they carry
/// no words at all (e.g. two tasks that are pure punctuation) — otherwise a
/// record would be unreachable by the very string it was stored under.
fn tasks_related(a: &str, b: &str) -> bool {
    let (left, right) = (words(a), words(b));
    if left == right {
        return true;
    }
    left.iter().any(|w| right.contains(w))
}

#[async_trait]
impl ExperienceStore for InMemoryExperienceStore {
    async fn record(&self, exp: &Experience) -> Result<()> {
        if !exp.is_recallable() {
            // Unrecallable records are dropped, not rejected: nothing could
            // ever match them, and failing here would surface as a store error
            // for what is really a no-op.
            return Ok(());
        }
        let mut log = self.lock();
        log.entries.push_back(exp.clone());
        while log.entries.len() > log.capacity {
            log.entries.pop_front();
        }
        Ok(())
    }

    async fn recall_for(&self, agent_id: &str, task: &str) -> Result<Vec<Experience>> {
        let log = self.lock();
        // Newest first: the most recent attempt at a task is the one whose
        // strategy is most likely still valid, and a caller that truncates the
        // list should keep that end.
        Ok(log
            .entries
            .iter()
            .rev()
            .filter(|e| e.agent_id == agent_id && tasks_related(&e.task, task))
            .cloned()
            .collect())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn exp(agent: &str, task: &str, outcome: &str, success: bool) -> Experience {
        let e = Experience::new(agent, task, outcome);
        if success { e.succeeded() } else { e }
    }

    #[test]
    fn new_experience_defaults_to_failure() {
        let e = Experience::new("planner", "migrate schema", "permission denied");
        assert!(!e.success);
        assert!(e.succeeded().success);
    }

    #[test]
    fn recallable_requires_agent_and_task() {
        assert!(Experience::new("planner", "migrate", "").is_recallable());
        assert!(!Experience::new("", "migrate", "ok").is_recallable());
        assert!(!Experience::new("planner", "   ", "ok").is_recallable());
    }

    #[test]
    fn experience_round_trips_through_serde() {
        let e = exp("planner", "migrate schema", "worked on retry", true);
        let json = serde_json::to_string(&e).expect("serialize");
        let back: Experience = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(e, back);
    }

    #[tokio::test]
    async fn empty_store_recalls_nothing() {
        let store = InMemoryExperienceStore::new();
        assert!(store.is_empty());
        let found = store
            .recall_for("planner", "anything")
            .await
            .expect("recall");
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn records_and_recalls_by_agent_and_related_task() {
        let store = InMemoryExperienceStore::new();
        store
            .record(&exp("planner", "migrate the schema", "ok", true))
            .await
            .expect("record");
        store
            .record(&exp("planner", "brew coffee", "no kettle", false))
            .await
            .expect("record");
        store
            .record(&exp("writer", "migrate the schema", "n/a", true))
            .await
            .expect("record");

        let found = store
            .recall_for("planner", "schema migration")
            .await
            .expect("recall");
        assert_eq!(found.len(), 1, "other agent and unrelated task excluded");
        assert_eq!(found[0].task, "migrate the schema");
        assert!(found[0].success);
    }

    #[tokio::test]
    async fn recall_returns_newest_first() {
        let store = InMemoryExperienceStore::new();
        store
            .record(&exp("planner", "deploy service", "first", false))
            .await
            .expect("record");
        store
            .record(&exp("planner", "deploy service", "second", true))
            .await
            .expect("record");

        let found = store.recall_for("planner", "deploy").await.expect("recall");
        let outcomes: Vec<&str> = found.iter().map(|e| e.outcome.as_str()).collect();
        assert_eq!(outcomes, vec!["second", "first"]);
    }

    #[tokio::test]
    async fn punctuation_and_case_do_not_defeat_matching() {
        let store = InMemoryExperienceStore::new();
        store
            .record(&exp("planner", "Deploy the service.", "ok", true))
            .await
            .expect("record");
        let found = store
            .recall_for("planner", "deploy?")
            .await
            .expect("recall");
        assert_eq!(found.len(), 1);
    }

    #[tokio::test]
    async fn unrecallable_records_are_dropped_without_error() {
        let store = InMemoryExperienceStore::new();
        store
            .record(&exp("", "migrate", "ok", true))
            .await
            .expect("record must not fail");
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn oldest_records_are_evicted_at_capacity() {
        let store = InMemoryExperienceStore::with_capacity(2);
        for outcome in ["a", "b", "c"] {
            store
                .record(&exp("planner", "deploy service", outcome, false))
                .await
                .expect("record");
        }
        assert_eq!(store.len(), 2);
        let found = store.recall_for("planner", "deploy").await.expect("recall");
        let outcomes: Vec<&str> = found.iter().map(|e| e.outcome.as_str()).collect();
        assert_eq!(outcomes, vec!["c", "b"], "oldest record evicted");
    }

    #[tokio::test]
    async fn zero_capacity_is_clamped_to_one() {
        let store = InMemoryExperienceStore::with_capacity(0);
        store
            .record(&exp("planner", "deploy service", "only", true))
            .await
            .expect("record");
        assert_eq!(store.len(), 1);
    }
}
