//! Admission control and cost accounting supplied by the embedder.
//!
//! See [`BudgetGate`] for the trait contract and [`UnmeteredBudgetGate`] for
//! the inert default.

use std::any::Any;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::harness::cost::CostTotals;
use crate::harness::ids::{CallId, RunId, ThreadId};
use crate::harness::usage::Usage;

/// Admission control and cost accounting owned by the embedder.
///
/// The crate already tracks [`Usage`] and [`CostTotals`] and enforces
/// [`RunLimits`][crate::harness::limits::RunLimits]. This trait is the seam for
/// *external* budgets: process-wide concurrency, pricing tables, and durable
/// spend ledgers the crate cannot know about.
///
/// Scope note: this is money and admission, not context. Fitting a
/// conversation into a model's window is
/// [`Summarizer`][crate::harness::summarization::Summarizer] plus
/// [`RunLimits`][crate::harness::limits::RunLimits]; a host's compaction
/// *profile* is host data and rides in
/// [`AgentDefinition::extras`][super::AgentDefinition::extras].
#[async_trait]
pub trait BudgetGate<State: Send + Sync>: Send + Sync {
    /// Waits for permission to start work, returning an opaque lease held for
    /// the duration and released on drop. `Ok(None)` means admission was
    /// refused without an error (for example, a paused scheduler).
    ///
    /// The default admits immediately with an inert lease.
    async fn acquire(
        &self,
        state: &State,
        request: &AdmissionRequest<'_>,
    ) -> Result<Option<BudgetLease>> {
        let _ = (state, request);
        Ok(Some(BudgetLease::unmetered()))
    }

    /// Prices one model call when the provider did not report a charge.
    /// Synchronous: implementations are table lookups, not I/O.
    fn estimate_cost(&self, state: &State, model_id: &str, usage: &Usage) -> CostTotals {
        let _ = (state, model_id, usage);
        CostTotals::default()
    }

    /// Records one completed model call against the host's ledger.
    async fn record_usage(&self, state: &State, entry: &UsageEntry<'_>) -> Result<()>;

    /// Charges a completed turn and reports whether the run may continue.
    ///
    /// A [`BudgetVerdict::Stop`] is a graceful request drained at the next
    /// iteration boundary, not an abort; a bounded overshoot is expected.
    /// The default never stops a run.
    async fn account_turn(&self, state: &State, charge: &TurnCharge<'_>) -> Result<BudgetVerdict> {
        let _ = (state, charge);
        Ok(BudgetVerdict::Continue)
    }
}

/// Work asking permission to start.
#[derive(Clone, Debug)]
pub struct AdmissionRequest<'a> {
    /// The run about to start.
    pub run_id: &'a RunId,
    /// Host-defined identity of the agent taking the run.
    pub agent_id: &'a str,
    /// Host-defined workload label the run will draw against.
    pub workload: &'a str,
    /// `true` when a user is waiting on the result; hosts commonly prioritise
    /// interactive work over scheduled work.
    pub interactive: bool,
}

/// An opaque admission lease. The crate holds it and drops it; it never
/// inspects it, so hosts can carry a semaphore permit, a token bucket handle,
/// or nothing at all.
///
/// The inner value is intentionally write-only from the crate's perspective:
/// its whole purpose is to be dropped at the right moment.
/// [`into_inner`][Self::into_inner] exists for hosts that need it back.
pub struct BudgetLease(Box<dyn Any + Send + Sync>);

impl BudgetLease {
    /// Wraps a host guard whose `Drop` releases the admission.
    pub fn new<T: Send + Sync + 'static>(guard: T) -> Self {
        Self(Box::new(guard))
    }

    /// A lease that grants everything and releases nothing.
    pub fn unmetered() -> Self {
        Self::new(())
    }

    /// Returns the wrapped guard, for a host that needs to downcast it.
    pub fn into_inner(self) -> Box<dyn Any + Send + Sync> {
        self.0
    }
}

impl std::fmt::Debug for BudgetLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BudgetLease(..)")
    }
}

/// One completed model call, ready for the host's ledger.
#[derive(Clone, Debug)]
pub struct UsageEntry<'a> {
    /// The run the call belongs to.
    pub run_id: &'a RunId,
    /// Provider-assigned identity of the call.
    pub call_id: &'a CallId,
    /// Model that was called.
    pub model_id: &'a str,
    /// Token counts for the call.
    pub usage: Usage,
    /// Provider-reported charge when available, otherwise the value returned
    /// by [`BudgetGate::estimate_cost`].
    pub cost: CostTotals,
}

/// One completed turn, ready to be charged.
#[derive(Clone, Debug)]
pub struct TurnCharge<'a> {
    /// The run the turn belongs to.
    pub run_id: &'a RunId,
    /// Thread the run belongs to, when it belongs to one.
    pub thread_id: Option<&'a ThreadId>,
    /// Token counts accumulated over the turn.
    pub usage: Usage,
    /// Charge accumulated over the turn.
    pub cost: CostTotals,
    /// Wall-clock duration of the turn.
    pub elapsed_secs: u64,
}

/// Whether the run may continue after a charge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "verdict")]
pub enum BudgetVerdict {
    /// The run may keep going.
    Continue,
    /// The run should wind down. Host-authored reason, surfaced to the caller
    /// verbatim.
    Stop {
        /// Host-authored explanation.
        reason: String,
    },
}

/// A [`BudgetGate`] with no budget behind it.
///
/// `record_usage` accepts and discards; every other method takes its trait
/// default. Wiring it is observationally identical to running ungated.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnmeteredBudgetGate;

impl UnmeteredBudgetGate {
    /// Creates the gate.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State: Send + Sync> BudgetGate<State> for UnmeteredBudgetGate {
    async fn record_usage(&self, _state: &State, _entry: &UsageEntry<'_>) -> Result<()> {
        Ok(())
    }
}
