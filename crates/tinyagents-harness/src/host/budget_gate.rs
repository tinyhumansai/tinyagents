//! Host budget capability: back-pressure and spend accounting around model calls.
//!
//! A host that meters model spend — token budgets, per-tenant quotas, a global
//! concurrency ceiling on in-flight completions — supplies a [`BudgetGate`].
//! The runtime consults it immediately before each model call, reports realised
//! [`Usage`](tinyinference::usage::Usage) afterwards, and asks it (cheaply,
//! synchronously) whether context should be compressed before the next call.
//!
//! **This capability is optional.** A host that does not meter spend passes
//! `None` rather than an erroring stub: absence must stay distinguishable from
//! failure, because a gate that errors teaches the loop the capability exists
//! and invites a retry. When the host supplies nothing the runtime skips the
//! calls entirely; [`UnlimitedBudgetGate`] exists for hosts and tests that want
//! a present-but-permissive gate rather than no gate at all.
//!
//! **Why a permit rather than a boolean.** [`BudgetGate::acquire`] is `async`
//! and hands back a [`Permit`] specifically so the host can *wait* for
//! capacity instead of refusing. That is back-pressure, not admission control:
//! the number of live permits is what bounds concurrency, and the permit
//! releasing on `Drop` is what makes that bound hold even when a turn unwinds
//! through an error, a cancellation, or a panic. A `-> Result<bool>` check
//! could not express any of that.
//!
//! Everything in this module other than [`Permit`] is inert (`serde` + `std`),
//! so a host can build its estimates and read its hints without depending on
//! the runtime's machinery.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ids::ThreadId;
use tinyinference::usage::Usage;

// ── CallEstimate ──────────────────────────────────────────────────────────────

/// What the runtime believes a model call is about to cost, supplied to
/// [`BudgetGate::acquire`] before the call is made.
///
/// Every token figure is an **estimate**, not a measurement — the runtime has
/// not spoken to the provider yet. A host must therefore treat these as a
/// sizing signal for admission, and reconcile against the authoritative numbers
/// that arrive later via [`BudgetGate::record`]. Estimates can be zero when the
/// runtime has no tokenizer for the target model, so a host must not read
/// `0` as "free".
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEstimate {
    /// Provider-facing model id the call will target. An opaque string: model
    /// naming is the host's and the provider's business, never parsed here.
    #[serde(default)]
    pub model: String,
    /// The agent the call belongs to, when it is attributed to one.
    ///
    /// A plain `String` because the crate has no `AgentId` newtype and this
    /// module deliberately does not invent one — an id minted here would not be
    /// the same type the host's registry hands around, so the newtype would buy
    /// no type safety while adding a conversion at every call site.
    ///
    /// `Option`, not an empty-string sentinel: "unattributed" is a real state a
    /// budget gate may want to treat differently, and encoding it as `""` means
    /// every host has to remember the convention and one day won't.
    #[serde(default)]
    pub agent_id: Option<String>,
    /// Thread the call is part of, when the call belongs to one. Lets a host
    /// scope a budget per conversation rather than only globally.
    #[serde(default)]
    pub thread_id: Option<ThreadId>,
    /// Estimated prompt tokens the request will carry.
    #[serde(default)]
    pub estimated_input_tokens: u64,
    /// Estimated completion tokens the reply may produce. Usually derived from
    /// a configured max-output cap, so it is an upper bound rather than a
    /// prediction — reserving against it is the conservative reading.
    #[serde(default)]
    pub estimated_output_tokens: u64,
    /// Number of tool schemas attached to the request. Exposed because tool
    /// schemas are a large, easily-overlooked share of prompt tokens, and a
    /// host tuning its own estimator wants to see it separately.
    #[serde(default)]
    pub tool_count: usize,
}

impl CallEstimate {
    /// An estimate for `model` with input/output token guesses and no other
    /// attribution.
    pub fn new(model: impl Into<String>, input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            model: model.into(),
            estimated_input_tokens: input_tokens,
            estimated_output_tokens: output_tokens,
            ..Self::default()
        }
    }

    /// Attributes this estimate to `agent_id`.
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Attributes this estimate to `thread_id`.
    pub fn with_thread(mut self, thread_id: impl Into<ThreadId>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Total tokens the call is expected to consume, saturating rather than
    /// overflowing — an absurd estimate must not wrap around to a small number
    /// and let an oversized call slip past a budget check.
    pub fn estimated_total_tokens(&self) -> u64 {
        self.estimated_input_tokens
            .saturating_add(self.estimated_output_tokens)
    }
}

// ── Permit ────────────────────────────────────────────────────────────────────

/// A grant of capacity for one model call, released when dropped.
///
/// The runtime holds the permit for exactly as long as the call is in flight
/// and then lets it fall out of scope. It never inspects the permit; the value
/// exists so that *not holding it* is impossible to express while a call runs.
///
/// Release runs from `Drop`, so it happens on every exit path — normal return,
/// `?` propagation, task cancellation, unwind. The release hook therefore must
/// not block, must not panic, and must not assume it is on any particular
/// runtime: it may run on an arbitrary thread while another panic is already
/// unwinding. Hosts that need to do real work on release should hand the
/// closure a channel send or a counter decrement, not a lock-and-flush.
///
/// This is the one type in this module that is not `serde`-inert, and
/// necessarily so: a permit is a live resource handle, not a value. Sending one
/// over a wire would sever it from the capacity it represents.
pub struct Permit {
    /// Host-defined identifier for the grant, for correlating with the host's
    /// own accounting. Opaque here.
    id: Option<String>,
    /// Tokens the host reserved for this call, when it reserved a specific
    /// amount. `None` means the host granted without earmarking.
    reserved_tokens: Option<u64>,
    /// Run on drop. `None` for a permit that costs nothing to release.
    on_release: Option<Box<dyn FnOnce() + Send + Sync>>,
}

impl Permit {
    /// A permit that reserves nothing and does nothing on release.
    ///
    /// Correct for any gate that grants unconditionally; it is what
    /// [`UnlimitedBudgetGate`] returns.
    pub fn unlimited() -> Self {
        Self {
            id: None,
            reserved_tokens: None,
            on_release: None,
        }
    }

    /// A permit that invokes `on_release` exactly once when dropped.
    ///
    /// "Exactly once" is guaranteed by the hook being consumed out of the
    /// permit on drop, and by [`Permit`] being non-`Clone` — capacity that
    /// could be duplicated would not bound anything.
    pub fn with_release<F>(on_release: F) -> Self
    where
        F: FnOnce() + Send + Sync + 'static,
    {
        Self {
            id: None,
            reserved_tokens: None,
            on_release: Some(Box::new(on_release)),
        }
    }

    /// Tags this permit with a host-defined `id`.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Records that the host earmarked `tokens` for this call.
    pub fn with_reserved_tokens(mut self, tokens: u64) -> Self {
        self.reserved_tokens = Some(tokens);
        self
    }

    /// The host-defined grant id, if any.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Tokens earmarked by the host for this call, if any.
    pub fn reserved_tokens(&self) -> Option<u64> {
        self.reserved_tokens
    }

    /// Releases the permit now instead of at end of scope.
    ///
    /// Only a convenience: dropping does the same thing. It exists so a caller
    /// that finishes a call early can return capacity without contorting its
    /// scopes, which matters when one task holds a permit across a long
    /// post-processing tail.
    pub fn release(self) {
        drop(self);
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        if let Some(on_release) = self.on_release.take() {
            on_release();
        }
    }
}

// Hand-written because the release hook is a closure and cannot derive `Debug`.
// Printing the hook's identity would be noise anyway; the grant's identity is
// the useful part.
impl std::fmt::Debug for Permit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Permit")
            .field("id", &self.id)
            .field("reserved_tokens", &self.reserved_tokens)
            .field("has_release_hook", &self.on_release.is_some())
            .finish()
    }
}

// ── ContextState ──────────────────────────────────────────────────────────────

/// A snapshot of how full the turn's context has become, passed to
/// [`BudgetGate::compression_hint`].
///
/// Deliberately small and cheap to build: the hint is consulted on the hot path
/// between calls, so the runtime must be able to produce this without walking
/// the transcript or re-tokenizing it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextState {
    /// Messages currently in the transcript that would be sent.
    #[serde(default)]
    pub message_count: usize,
    /// Best-known prompt token count for the next request. May be an estimate
    /// or zero when no tokenizer is available for the model.
    #[serde(default)]
    pub prompt_tokens: u64,
    /// The model's context window in tokens, when the runtime knows it.
    /// `None` means unknown — a host must not treat that as "unbounded".
    #[serde(default)]
    pub context_window_tokens: Option<u64>,
    /// Tool-call/model round trips already spent in this turn. A turn that has
    /// iterated many times is often worth compressing even well below the
    /// window, which is why this travels alongside the token counts.
    #[serde(default)]
    pub iterations: usize,
}

impl ContextState {
    /// Fraction of the context window currently occupied, in `0.0..`.
    ///
    /// `None` when the window is unknown or reported as zero — a divide there
    /// would be meaningless, and returning `0.0` would read as "plenty of
    /// room", the most dangerous possible wrong answer.
    pub fn utilization(&self) -> Option<f64> {
        match self.context_window_tokens {
            Some(window) if window > 0 => Some(self.prompt_tokens as f64 / window as f64),
            _ => None,
        }
    }
}

// ── CompressionHint ───────────────────────────────────────────────────────────

/// The host's advice on compressing context before the next model call.
///
/// Advice, not a command: the runtime decides how to act (summarize, drop old
/// tool results, offload). Splitting `Soft` from `Hard` lets a host distinguish
/// "cheaper if you do" from "the next call will not fit otherwise" without the
/// crate having to model the host's cost curve.
///
/// Note the [`None`](Self::None) variant shadows `Option::None` inside a `use
/// CompressionHint::*` scope. Match on the qualified path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionHint {
    /// Proceed as-is. The default, so a host that has not thought about
    /// compression never accidentally triggers it.
    #[default]
    None,
    /// Compression is worthwhile but optional — the runtime may skip it, for
    /// example when the remaining work is one short call.
    Soft,
    /// Compression is required before the next call can reasonably be made.
    Hard,
}

impl CompressionHint {
    /// Whether the host is advising any compression at all.
    pub fn is_advised(&self) -> bool {
        !matches!(self, CompressionHint::None)
    }

    /// Whether the host considers compression mandatory before the next call.
    pub fn is_required(&self) -> bool {
        matches!(self, CompressionHint::Hard)
    }
}

// ── BudgetGate ────────────────────────────────────────────────────────────────

/// Host-supplied metering around model calls: admission, accounting, and
/// context-pressure advice.
///
/// Optional. A host that does not meter spend supplies no gate at all rather
/// than an implementation that errors — see the module docs for why absence and
/// failure must stay distinct.
#[async_trait]
pub trait BudgetGate: Send + Sync {
    /// Awaited immediately before each model call; resolves when the host has
    /// capacity for it.
    ///
    /// This is back-pressure. A host **may** park here for as long as it needs
    /// — the caller's own turn timeout and cancellation are what bound the
    /// wait, so a gate must never install its own silent deadline. Reserve
    /// `Err` for a genuine budget refusal or an accounting failure; a
    /// temporarily-full budget is a reason to wait, not to fail.
    ///
    /// The returned [`Permit`] must be held for the duration of the call. Its
    /// `Drop` is what returns capacity, so it holds on every exit path.
    async fn acquire(&self, est: &CallEstimate) -> Result<Permit>;

    /// Reports realised token usage after a call completes.
    ///
    /// Called once per call with the provider's authoritative numbers, which
    /// will differ from the [`CallEstimate`] — reconciling the two is the
    /// host's job. It is also called for **failed** calls that still consumed
    /// tokens, so an implementation must be additive and must not assume a
    /// successful reply followed.
    async fn record(&self, usage: &Usage) -> Result<()>;

    /// Whether context should be compressed before the next call.
    ///
    /// Synchronous on purpose: this sits on the hot path between iterations of
    /// the turn loop and is consulted far more often than [`acquire`](Self::acquire).
    /// An implementation must be cheap and non-blocking — read a counter,
    /// compare against a threshold. Anything that needs I/O belongs in
    /// [`record`](Self::record), whose result this can then consult.
    ///
    /// # Relationship to `SummarizationPolicy`
    ///
    /// The crate already decides "is the context too full?" in
    /// [`SummarizationPolicy`](crate::summarization::SummarizationPolicy),
    /// which owns the context window and threshold fraction. This hint does
    /// **not** replace it and must not be read as a second copy of it: the
    /// policy answers *has the context outgrown the window*, while this answers
    /// *does the host want compression for a reason of its own* — spend,
    /// quota, tenant tier.
    ///
    /// Precedence is **union, not override**: compression happens when the
    /// policy calls for it **or** the host hints for it. A host returning
    /// [`CompressionHint::None`] is declining to *ask* for compression, not
    /// vetoing the policy — otherwise an unmetered host could silently disable
    /// summarization and blow the context window.
    fn compression_hint(&self, state: &ContextState) -> CompressionHint;
}

// ── UnlimitedBudgetGate ───────────────────────────────────────────────────────

/// A gate that meters nothing: grants immediately, records nothing, never
/// advises compression.
///
/// For tests and for hosts that want the code path exercised without a real
/// budget. It is **not** the "no capability" case — that is expressed by
/// supplying no gate, which skips these calls entirely. Reach for this when you
/// specifically want a present, permissive gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnlimitedBudgetGate;

impl UnlimitedBudgetGate {
    /// Creates the gate. Stateless, so all instances are interchangeable.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl BudgetGate for UnlimitedBudgetGate {
    /// Grants without waiting. Returns [`Permit::unlimited`], which reserves
    /// nothing and does no work on drop, so holding it costs nothing.
    async fn acquire(&self, _est: &CallEstimate) -> Result<Permit> {
        Ok(Permit::unlimited())
    }

    /// Discards the usage. Nothing accumulates, so this gate can never refuse a
    /// later call.
    async fn record(&self, _usage: &Usage) -> Result<()> {
        Ok(())
    }

    /// Always [`CompressionHint::None`], regardless of how full the context is:
    /// deciding otherwise would be this gate silently applying a policy it was
    /// chosen to avoid.
    fn compression_hint(&self, _state: &ContextState) -> CompressionHint {
        CompressionHint::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn call_estimate_total_saturates_instead_of_wrapping() {
        let est = CallEstimate::new("m", u64::MAX, 10);
        assert_eq!(est.estimated_total_tokens(), u64::MAX);
    }

    #[test]
    fn call_estimate_builders_attribute_agent_and_thread() {
        let est = CallEstimate::new("m", 100, 20)
            .with_agent("lead")
            .with_thread("t-1");
        assert_eq!(est.agent_id.as_deref(), Some("lead"));
        assert_eq!(est.thread_id, Some(ThreadId::from("t-1")));
        assert_eq!(est.estimated_total_tokens(), 120);
    }

    #[test]
    fn call_estimate_round_trips_through_serde() {
        let est = CallEstimate::new("m", 1, 2).with_agent("a");
        let json = serde_json::to_string(&est).expect("serializes");
        let back: CallEstimate = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, est);
    }

    #[test]
    fn permit_runs_release_hook_exactly_once_on_drop() {
        let released = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&released);
        {
            let _permit = Permit::with_release(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            });
            assert_eq!(released.load(Ordering::SeqCst), 0, "not released early");
        }
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn permit_release_returns_capacity_immediately() {
        let released = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&released);
        let permit = Permit::with_release(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        permit.release();
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unlimited_permit_carries_no_reservation() {
        let permit = Permit::unlimited();
        assert!(permit.id().is_none());
        assert!(permit.reserved_tokens().is_none());
    }

    #[test]
    fn permit_metadata_builders_are_readable() {
        let permit = Permit::unlimited()
            .with_id("grant-7")
            .with_reserved_tokens(512);
        assert_eq!(permit.id(), Some("grant-7"));
        assert_eq!(permit.reserved_tokens(), Some(512));
        assert!(format!("{permit:?}").contains("grant-7"));
    }

    #[test]
    fn context_state_utilization_is_none_without_a_known_window() {
        let state = ContextState {
            prompt_tokens: 1_000,
            ..ContextState::default()
        };
        assert_eq!(state.utilization(), None);

        let zero_window = ContextState {
            prompt_tokens: 1_000,
            context_window_tokens: Some(0),
            ..ContextState::default()
        };
        assert_eq!(zero_window.utilization(), None);
    }

    #[test]
    fn context_state_utilization_is_the_occupied_fraction() {
        let state = ContextState {
            prompt_tokens: 500,
            context_window_tokens: Some(2_000),
            ..ContextState::default()
        };
        let utilization = state.utilization().expect("window is known");
        assert!((utilization - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn compression_hint_defaults_to_none_and_classifies_severity() {
        assert_eq!(CompressionHint::default(), CompressionHint::None);
        assert!(!CompressionHint::None.is_advised());
        assert!(CompressionHint::Soft.is_advised());
        assert!(!CompressionHint::Soft.is_required());
        assert!(CompressionHint::Hard.is_advised());
        assert!(CompressionHint::Hard.is_required());
    }

    #[tokio::test]
    async fn unlimited_gate_grants_a_costless_permit() {
        let gate = UnlimitedBudgetGate::new();
        let permit = gate
            .acquire(&CallEstimate::new("m", u64::MAX, u64::MAX))
            .await
            .expect("unlimited gate never refuses");
        assert!(permit.reserved_tokens().is_none());
    }

    #[tokio::test]
    async fn unlimited_gate_records_usage_without_accumulating() {
        let gate = UnlimitedBudgetGate;
        gate.record(&Usage::new(1_000, 1_000))
            .await
            .expect("recording is infallible");
        // Recording cannot make a later acquire refuse.
        gate.acquire(&CallEstimate::new("m", 1, 1))
            .await
            .expect("still grants after recording");
    }

    #[test]
    fn unlimited_gate_never_advises_compression_even_when_full() {
        let gate = UnlimitedBudgetGate::new();
        let full = ContextState {
            message_count: 10_000,
            prompt_tokens: 999_999,
            context_window_tokens: Some(1_000),
            iterations: 500,
        };
        assert_eq!(gate.compression_hint(&full), CompressionHint::None);
    }

    #[test]
    fn unlimited_gate_is_usable_as_a_trait_object() {
        // Pins object safety: the harness stores this as `Arc<dyn BudgetGate>`.
        let gate: Arc<dyn BudgetGate> = Arc::new(UnlimitedBudgetGate::new());
        assert_eq!(
            gate.compression_hint(&ContextState::default()),
            CompressionHint::None
        );
    }
}
