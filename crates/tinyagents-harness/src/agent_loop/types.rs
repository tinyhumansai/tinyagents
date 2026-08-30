//! Type definitions for the agent-loop module.
//!
//! The agent loop itself is implemented as inherent methods on
//! [`crate::runtime::AgentHarness`] in the sibling `mod.rs`. The only
//! public type owned here is [`AgentLoopResult`], the richer return value that
//! pairs the accumulated [`AgentRun`] with a compact [`HarnessRunStatus`]
//! snapshot for callers that want lifecycle/status information alongside the
//! transcript.
//!
//! All public items are re-exported through [`super`].

use crate::events::{HarnessRunStatus, LimitKind};
use crate::middleware::AgentRun;
use crate::steering::PauseState;

/// The result of an agent-loop invocation that keeps the partial run even when
/// the loop fails.
///
/// [`crate::runtime::AgentHarness::invoke`] returns `Err` on failure
/// and drops the [`AgentRun`] with it, so a run that tripped a limit or hit a
/// tool failure halfway through loses every message and usage figure it had
/// accumulated. `PartialRunOutcome` keeps both, letting a caller inspect,
/// repair, or resume from the partial conversation.
#[derive(Debug)]
pub struct PartialRunOutcome {
    /// The accumulated transcript, usage, counters, and final response —
    /// populated as far as the run got, whether or not it failed.
    pub run: AgentRun,
    /// A compact lifecycle/status snapshot reflecting how the run ended.
    pub status: HarnessRunStatus,
    /// The error that ended the run, or `None` when it succeeded.
    pub error: Option<crate::error::TinyAgentsError>,
}

/// How the agent loop body stopped iterating.
///
/// Kept separate from the `Result` channel so a *deliberate* stop (a pause, a
/// `StopWithPartial` limit) is never confused with a failure, and so the caller
/// can finalize each kind differently: a finish and a limit-stop complete the
/// run, while a pause is reported as interrupted and leaves the pause latched
/// on the steering handle.
#[derive(Clone, Debug)]
pub(crate) enum LoopExit {
    /// The model produced a final answer, or a middleware requested
    /// [`crate::context::MiddlewareControl::StopWithFinal`].
    Finished,
    /// A call cap was reached under
    /// [`LimitBehavior::StopWithPartial`][crate::limits::LimitBehavior::StopWithPartial]:
    /// stop cleanly and keep everything the run produced.
    LimitStop(LimitKind),
    /// Steering latched a pause; the run is resumable, not finished.
    Paused(PauseState),
}

/// The full result of an agent-loop invocation: the accumulated [`AgentRun`]
/// plus a compact [`HarnessRunStatus`] snapshot.
///
/// [`crate::runtime::AgentHarness::invoke`] returns only the
/// [`AgentRun`]; callers that also want the run's lifecycle status (phase,
/// counters, timing, error summary) can use
/// [`crate::runtime::AgentHarness::invoke_with_status`] and read the
/// `status` field.
#[derive(Clone, Debug)]
pub struct AgentLoopResult {
    /// The accumulated transcript, usage, counters, and final response.
    pub run: AgentRun,
    /// A compact lifecycle/status snapshot reflecting how the run ended.
    pub status: HarnessRunStatus,
}
