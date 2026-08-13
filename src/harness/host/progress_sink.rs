//! Host capability: coarse, fire-and-forget turn progress.
//!
//! A running turn produces observable milestones — it started, it invoked a
//! tool, it streamed a token, it finished, it failed. Some hosts want to show
//! those to a human while the turn is still running; most (batch jobs, cron
//! runs, tests) do not care. [`ProgressSink`] is the seam for the ones that do.
//!
//! # When a host supplies it
//!
//! Supply a `ProgressSink` when something outside the runtime must react to a
//! turn *before* it completes: a chat UI streaming tokens, a channel bridge
//! echoing "calling `search`…", a progress bar. Hosts that only need the final
//! result should pass `None` and let the runtime skip the calls entirely.
//!
//! This is an **optional** capability. Absence is expressed by the host
//! handing the runtime `None`, never by installing a sink whose `emit` errors:
//! an erroring stub is indistinguishable from a broken sink, and the runtime
//! has no way to tell "nobody is watching" from "the watcher fell over". A
//! host that wants an explicitly inert sink can install [`NoopProgressSink`].
//!
//! # Fire-and-forget is a contract, not an implementation detail
//!
//! [`ProgressSink::emit`] returns **unit**, not `Result<()>`. That is
//! deliberate and load-bearing:
//!
//! - The runtime must never be able to fail a turn because a UI socket
//!   hung up. Progress is a side channel; the turn is the product.
//! - There is no return value to branch on, so no future refactor can
//!   accidentally make turn control flow depend on the sink.
//! - An implementation that needs to do real I/O (a socket write, a queue
//!   push) must **not** block in `emit`. Hand the event to a bounded channel
//!   and drop on backpressure. Dropping progress events is always preferable
//!   to slowing or stalling the turn that produces them.
//!
//! `emit` is nevertheless `async` (per the RFC signature) so an implementation
//! can hand off to an async channel without spawning a task per event. It is
//! still not to be awaited on the critical path in any blocking sense.
//!
//! # Why [`ProgressEvent`] stays coarse
//!
//! Six variants, and the bar for a seventh is high. A host's user-facing
//! progress model — timeline entries, cost footers, citation chips, avatars,
//! retry badges — is a **host contract** owned by that host's UI, and it is
//! *produced from* these events by host-side adapter code. It must not migrate
//! into this enum.
//!
//! The failure mode if it does is nasty: this crate is redistributed, so a
//! UI-shaped field added here becomes a compatibility surface for every
//! consumer, and the originating host's frontend starts depending on the
//! runtime to populate presentation state. That breaks in rendering — wrong
//! chip, missing footer, stale timeline row — which unit tests in *either*
//! repository will not catch, because both sides still compile and both sides
//! still pass. Keep presentation on the host side of the seam.
//!
//! # The admission test: runtime facts, not presentation
//!
//! "Coarse" is not a variant budget, it is a category rule, and the rule is
//! whether **only the runtime can know it**. `Started`, `ToolCall`,
//! `ToolCallFinished`, `Token`, `Finished` and `Error` are all facts about what
//! the engine did; no host can derive any of them from the outside. A chip, a
//! footer or a badge is computed *from* those facts by host code, and belongs
//! there. Ask that question of a proposed variant before counting variants.
//!
//! [`ProgressEvent::ToolCallFinished`] was admitted under exactly that test.
//! This module previously declared that there was intentionally no completion
//! variant, on the reasoning that tool results are large and can carry
//! untrusted text — true of the *payload*, but it left `ToolCall` opening a
//! timeline row that nothing could ever close. `Started`/`Finished` bracket a
//! run; tool calls now bracket the same way. The asymmetry was an oversight
//! rather than a boundary, and the payload concern is answered by policy on the
//! field (see [`ToolCallFinished::output`](Self::ToolCallFinished)) instead of
//! by withholding the outcome.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::harness::ids::{CallId, RunId, ThreadId};
use crate::harness::usage::Usage;

// ── ProgressEvent ─────────────────────────────────────────────────────────────

/// A coarse milestone in a running turn.
///
/// Every variant carries the [`RunId`] it belongs to so a host multiplexing
/// several concurrent turns onto one sink can route without keeping its own
/// side table — see [`ProgressEvent::run_id`], which is total for exactly this
/// reason.
///
/// Inert by construction: `serde` + `std` only, no engine or transport types,
/// so a host can build its own progress model against this enum without
/// depending on the rest of the crate.
///
/// **Not an audit log.** Events may be dropped under backpressure and are not
/// ordered across runs. A host that needs a durable, complete record should
/// use the harness event journal, not this.
///
/// # Relationship to `AgentEvent`
///
/// [`AgentEvent`](crate::harness::events::AgentEvent) is the authoritative,
/// fine-grained event stream, and these six variants are a deliberately coarse
/// projection of it (`Started` ← `RunStarted`, `Token` ← `ModelDelta`, and so
/// on). `AgentEvent` is the source of truth; `ProgressEvent` is derived.
///
/// Keeping it coarse is the point: this enum crosses into host UI code, and a
/// UI pinned to every internal event would break on every internal change. When
/// the wiring lands it should be **one** conversion from `AgentEvent`, not
/// parallel emission from separate call sites — two independent producers of
/// "what is happening now" will drift, and the drift shows up as a UI that
/// disagrees with the journal about what the agent did.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProgressEvent {
    /// The turn began.
    Started {
        /// The run this event belongs to.
        run: RunId,
        /// The conversation thread, when the turn belongs to one. Absent for
        /// one-shot runs that are not part of a thread.
        thread: Option<ThreadId>,
        /// Host-defined agent identifier, as a plain string.
        ///
        /// The crate has no `AgentId` newtype and this module deliberately does
        /// not invent one — a progress event is a leaf, and minting an id type
        /// here would force every host onto it before the registry seam
        /// (RFC §3.3) has settled on a representation.
        agent: String,
    },

    /// A tool invocation started.
    ///
    /// Closed by at most one [`ToolCallFinished`](Self::ToolCallFinished)
    /// carrying the same `call`. **At most**, not exactly one: a run that fails
    /// or is cancelled mid-tool emits `Error` and no closing event, and events
    /// may be dropped under backpressure. A host must therefore tear down open
    /// tool rows when [`is_terminal`](Self::is_terminal) fires, rather than
    /// waiting for a close that may never arrive.
    ///
    /// Call arguments are deliberately absent. They can be large and can carry
    /// untrusted or sensitive text, and a host does not need them here — it
    /// owns the tool being invoked and already has them.
    ToolCall {
        /// The run this event belongs to.
        run: RunId,
        /// The individual call, so a host can correlate with its own records.
        call: CallId,
        /// Tool name as registered with the runtime.
        tool: String,
    },

    /// A tool invocation finished, successfully or not.
    ///
    /// Correlates with the opening [`ToolCall`](Self::ToolCall) by `call`. This
    /// is a runtime fact, not a presentation concern: only the engine knows
    /// whether the tool returned or failed, so no host can derive it. Without
    /// it a row opened by `ToolCall` could never be closed truthfully — see the
    /// admission test in the module docs.
    ///
    /// **Never synthesise this event.** The tempting shortcut is to infer
    /// completion from the arrival of the next event and assume success. That
    /// establishes *that* a tool finished but not *how*, and a fabricated
    /// `success: true` writes wrong data into both the host's timeline and its
    /// trace exporter. A row visibly stuck in a running state is a better
    /// outcome than a confidently incorrect one: the first gets noticed, the
    /// second does not.
    ToolCallFinished {
        /// The run this event belongs to.
        run: RunId,
        /// The individual call, matching the opening [`ToolCall`](Self::ToolCall).
        call: CallId,
        /// Whether the tool returned successfully.
        success: bool,
        /// Raw tool output, or empty when the runtime ran with payload capture
        /// off.
        ///
        /// Deliberately carries **no** truncation or redaction policy: both are
        /// host decisions, and a host must apply its own before rendering or
        /// exporting this. Empty is ambiguous by construction — it means "not
        /// captured" or "genuinely empty", and a host needing to tell those
        /// apart must consult its own capture configuration rather than
        /// inferring from this field.
        output: String,
    },

    /// A chunk of assistant output became available.
    ///
    /// Named `Token` for the RFC's sake, but `text` is whatever the provider
    /// handed over — it may be several tokens, a partial word, or empty.
    /// Concatenating the `text` of every `Token` for a run reconstructs the
    /// streamed reply *only if no event was dropped*, which the sink does not
    /// guarantee. Treat it as a display hint, never as the authoritative reply.
    Token {
        /// The run this event belongs to.
        run: RunId,
        /// The output fragment.
        text: String,
    },

    /// The turn completed normally.
    Finished {
        /// The run this event belongs to.
        run: RunId,
        /// Realised token usage when the runtime knows it.
        ///
        /// `None` means "not reported", not "zero" — a provider that omits
        /// usage and a genuinely free call must stay distinguishable, or a
        /// host's cost display silently under-reports.
        usage: Option<Usage>,
    },

    /// The turn failed.
    ///
    /// Carries a human-readable message rather than a crate error type: the
    /// error enum is an evolving internal surface, whereas a progress consumer
    /// only ever displays this. Hosts must not parse `message` to make
    /// decisions — the authoritative failure is the turn's returned error.
    Error {
        /// The run this event belongs to.
        run: RunId,
        /// Display-oriented description of the failure.
        message: String,
    },
}

impl ProgressEvent {
    /// The run this event belongs to.
    ///
    /// Total on purpose — a sink shared by concurrent turns must always be able
    /// to route an event, so no variant may be added without a `RunId`.
    pub fn run_id(&self) -> &RunId {
        match self {
            Self::Started { run, .. }
            | Self::ToolCall { run, .. }
            | Self::ToolCallFinished { run, .. }
            | Self::Token { run, .. }
            | Self::Finished { run, .. }
            | Self::Error { run, .. } => run,
        }
    }

    /// Whether this event ends the run — [`Finished`](Self::Finished) or
    /// [`Error`](Self::Error).
    ///
    /// Lets a host tear down per-run UI state without matching on every
    /// variant, so adding a mid-turn variant later cannot silently turn into a
    /// leaked progress row.
    ///
    /// [`ToolCallFinished`](Self::ToolCallFinished) is **not** terminal despite
    /// the name — it ends a tool call, not the run, and a turn typically emits
    /// several before `Finished`. Treating it as terminal would tear the run's
    /// UI down at the first tool result.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Finished { .. } | Self::Error { .. })
    }
}

// ── ProgressSink ──────────────────────────────────────────────────────────────

/// Receives coarse turn progress from the runtime.
///
/// Optional: a host that does not supply one gets no calls at all, rather than
/// calls into a stub that fails.
#[async_trait]
pub trait ProgressSink: Send + Sync {
    /// Delivers one progress event.
    ///
    /// Returns nothing, and that is the whole point — see the module docs. An
    /// implementation must not block, must not panic, and must swallow its own
    /// transport failures. Dropping the event is the correct response to
    /// backpressure or a dead consumer.
    ///
    /// The runtime may call this from within a turn's hot loop (once per
    /// streamed chunk), so the body should be cheap: a channel send, not a
    /// write to durable storage.
    async fn emit(&self, ev: ProgressEvent);
}

/// Forwards to the wrapped sink, so `Arc<dyn ProgressSink>` and
/// `Arc<ConcreteSink>` are both usable wherever a `ProgressSink` is wanted.
///
/// Without this, a host holding an `Arc` would have to deref-juggle at every
/// call site or wrap in yet another type — sinks are shared by definition
/// (one UI, many concurrent turns), so sharing must be frictionless.
#[async_trait]
impl<T: ProgressSink + ?Sized> ProgressSink for Arc<T> {
    async fn emit(&self, ev: ProgressEvent) {
        (**self).emit(ev).await;
    }
}

// ── NoopProgressSink ──────────────────────────────────────────────────────────

/// A sink that discards everything.
///
/// The default when a host wants the capability present but inert — for
/// example a builder that requires *some* sink, or a benchmark isolating
/// runtime cost from progress delivery cost.
///
/// Prefer passing `None` when progress genuinely is not wanted: `None` lets the
/// runtime skip building the event at all, whereas this still constructs each
/// event before dropping it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoopProgressSink;

impl NoopProgressSink {
    /// A discarding sink.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ProgressSink for NoopProgressSink {
    async fn emit(&self, _ev: ProgressEvent) {}
}

// ── RecordingProgressSink ─────────────────────────────────────────────────────

/// An in-memory sink that keeps every event it receives, in arrival order.
///
/// For tests and local debugging. Unbounded by design — a test wants to assert
/// on the whole sequence, and silently evicting would turn a real regression
/// into a confusing assertion failure. Do **not** install it in a long-lived
/// process: it grows without limit.
///
/// Uses a plain [`std::sync::Mutex`] rather than an async lock because every
/// critical section is a `push` or a `clone` of the buffer — never an await —
/// so an async lock would add a scheduling hop to the turn's hot path for no
/// benefit.
#[derive(Clone, Debug, Default)]
pub struct RecordingProgressSink {
    events: Arc<Mutex<Vec<ProgressEvent>>>,
}

impl RecordingProgressSink {
    /// An empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// A snapshot of every event received so far, oldest first.
    pub fn events(&self) -> Vec<ProgressEvent> {
        self.lock().clone()
    }

    /// How many events have been received.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no event has been received yet.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Discards everything recorded so far, so one recorder can be reused
    /// across phases of a test without re-plumbing it.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// Locks the buffer, recovering from poisoning instead of panicking.
    ///
    /// A sink must never turn someone else's panic into a second panic on the
    /// turn's thread — the fire-and-forget contract says progress cannot fail a
    /// turn, and that has to hold even when a previous holder unwound. The
    /// recorded buffer is a plain `Vec` with no cross-element invariant, so the
    /// data behind a poisoned lock is still perfectly usable.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<ProgressEvent>> {
        self.events.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[async_trait]
impl ProgressSink for RecordingProgressSink {
    async fn emit(&self, ev: ProgressEvent) {
        self.lock().push(ev);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run() -> RunId {
        RunId::new("run-1")
    }

    fn started() -> ProgressEvent {
        ProgressEvent::Started {
            run: run(),
            thread: Some(ThreadId::new("thread-1")),
            agent: "lead".to_string(),
        }
    }

    fn tool_call() -> ProgressEvent {
        ProgressEvent::ToolCall {
            run: run(),
            call: CallId::new("call-1"),
            tool: "search".to_string(),
        }
    }

    fn tool_call_finished(success: bool, output: &str) -> ProgressEvent {
        ProgressEvent::ToolCallFinished {
            run: run(),
            call: CallId::new("call-1"),
            success,
            output: output.to_string(),
        }
    }

    fn token(text: &str) -> ProgressEvent {
        ProgressEvent::Token {
            run: run(),
            text: text.to_string(),
        }
    }

    fn finished() -> ProgressEvent {
        ProgressEvent::Finished {
            run: run(),
            usage: Some(Usage {
                input_tokens: 12,
                output_tokens: 3,
                total_tokens: 15,
                ..Usage::default()
            }),
        }
    }

    fn error() -> ProgressEvent {
        ProgressEvent::Error {
            run: run(),
            message: "provider unavailable".to_string(),
        }
    }

    fn all_variants() -> Vec<ProgressEvent> {
        vec![
            started(),
            tool_call(),
            tool_call_finished(true, "result"),
            token("hi"),
            finished(),
            error(),
        ]
    }

    // ── Value-type invariants ────────────────────────────────────────────────

    #[test]
    fn every_variant_exposes_its_run_id() {
        for ev in all_variants() {
            assert_eq!(ev.run_id(), &run(), "run id missing for {ev:?}");
        }
    }

    #[test]
    fn only_finished_and_error_are_terminal() {
        assert!(!started().is_terminal());
        assert!(!tool_call().is_terminal());
        assert!(!token("x").is_terminal());
        assert!(finished().is_terminal());
        assert!(error().is_terminal());
    }

    #[test]
    fn tool_completion_is_not_terminal() {
        // A turn emits one of these per tool call, several before `Finished`.
        // If this ever reads as terminal a host tears its run UI down at the
        // first tool result, which looks like a truncated turn rather than a
        // bug in this predicate.
        assert!(!tool_call_finished(true, "ok").is_terminal());
        assert!(!tool_call_finished(false, "boom").is_terminal());
    }

    #[test]
    fn tool_completion_correlates_with_its_opening_call() {
        // The `call` id is the only thing joining the two events. If they ever
        // stop matching, a host closes the wrong timeline row and the bug
        // surfaces as a mis-rendered tool, not as a failure here.
        let (
            ProgressEvent::ToolCall { call: opened, .. },
            ProgressEvent::ToolCallFinished { call: closed, .. },
        ) = (tool_call(), tool_call_finished(true, "ok"))
        else {
            panic!("constructors changed shape");
        };
        assert_eq!(opened, closed);
    }

    #[test]
    fn failure_is_representable_and_distinct_from_success() {
        // The whole point of #88: a host must be able to report that a tool
        // failed. If these ever compare equal, `success` has stopped carrying
        // information and every failed tool renders as a successful one.
        assert_ne!(
            tool_call_finished(true, "same"),
            tool_call_finished(false, "same")
        );
    }

    #[test]
    fn uncaptured_output_is_representable() {
        // Empty output is a legitimate state (payload capture off), not a
        // reason to withhold the outcome — success must still round-trip.
        let ev = tool_call_finished(false, "");
        let json = serde_json::to_string(&ev).expect("serialize");
        let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, ev);
    }

    #[test]
    fn events_round_trip_through_serde() {
        for ev in all_variants() {
            let json = serde_json::to_string(&ev).expect("serialize");
            let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn serialized_events_are_internally_tagged_by_kind() {
        let json = serde_json::to_value(tool_call()).expect("serialize");
        assert_eq!(json["kind"], "tool_call");
        assert_eq!(json["tool"], "search");

        // Distinct tag from `tool_call` — these cross a serde boundary into
        // host code, so a collision would silently merge open and close.
        let json = serde_json::to_value(tool_call_finished(false, "boom")).expect("serialize");
        assert_eq!(json["kind"], "tool_call_finished");
        assert_eq!(json["success"], false);
        assert_eq!(json["output"], "boom");
    }

    #[test]
    fn absent_usage_is_distinct_from_zero_usage() {
        let unreported = ProgressEvent::Finished {
            run: run(),
            usage: None,
        };
        let free_call = ProgressEvent::Finished {
            run: run(),
            usage: Some(Usage::default()),
        };
        assert_ne!(unreported, free_call);
    }

    // ── NoopProgressSink ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn noop_sink_accepts_every_variant_and_keeps_nothing() {
        let sink = NoopProgressSink::new();
        for ev in all_variants() {
            sink.emit(ev).await;
        }
        // Nothing observable to assert beyond "did not panic, returned unit" —
        // which is precisely the contract: emit has no failure channel.
        assert_eq!(sink, NoopProgressSink);
    }

    // ── RecordingProgressSink ────────────────────────────────────────────────

    #[tokio::test]
    async fn recording_sink_starts_empty() {
        let sink = RecordingProgressSink::new();
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
        assert!(sink.events().is_empty());
    }

    #[tokio::test]
    async fn recording_sink_preserves_arrival_order() {
        let sink = RecordingProgressSink::new();
        for ev in all_variants() {
            sink.emit(ev).await;
        }
        assert_eq!(sink.events(), all_variants());
        assert_eq!(sink.len(), all_variants().len());
        assert!(!sink.is_empty());
    }

    #[tokio::test]
    async fn recording_sink_retains_duplicate_events() {
        let sink = RecordingProgressSink::new();
        sink.emit(token("a")).await;
        sink.emit(token("a")).await;
        // Streamed chunks legitimately repeat; a recorder that deduplicated
        // would hide a real double-emit regression.
        assert_eq!(sink.events(), vec![token("a"), token("a")]);
    }

    #[tokio::test]
    async fn recording_sink_clear_resets_the_buffer() {
        let sink = RecordingProgressSink::new();
        sink.emit(started()).await;
        sink.clear();
        assert!(sink.is_empty());
        sink.emit(finished()).await;
        assert_eq!(sink.events(), vec![finished()]);
    }

    #[tokio::test]
    async fn recording_sink_clones_share_one_buffer() {
        let sink = RecordingProgressSink::new();
        let clone = sink.clone();
        clone.emit(started()).await;
        // The runtime hands out clones to concurrent turns; if they did not
        // share, a test would observe an empty recorder and pass vacuously.
        assert_eq!(sink.events(), vec![started()]);
    }

    #[tokio::test]
    async fn recording_sink_records_through_a_trait_object() {
        let sink = RecordingProgressSink::new();
        let dynamic: Arc<dyn ProgressSink> = Arc::new(sink.clone());
        dynamic.emit(token("chunk")).await;
        assert_eq!(sink.events(), vec![token("chunk")]);
    }

    #[tokio::test]
    async fn recording_sink_survives_a_poisoned_lock() {
        let sink = RecordingProgressSink::new();
        sink.emit(started()).await;

        let poisoner = sink.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.events.lock().expect("acquire");
            panic!("poison the mutex");
        })
        .join();

        // Progress must never escalate someone else's panic into a failed turn.
        sink.emit(finished()).await;
        assert_eq!(sink.events(), vec![started(), finished()]);
    }
}
