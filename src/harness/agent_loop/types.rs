//! Type definitions for the agent-loop module.
//!
//! The agent loop itself is implemented as inherent methods on
//! [`crate::harness::runtime::AgentHarness`] in the sibling `mod.rs`. The only
//! public type owned here is [`AgentLoopResult`], the richer return value that
//! pairs the accumulated [`AgentRun`] with a compact [`HarnessRunStatus`]
//! snapshot for callers that want lifecycle/status information alongside the
//! transcript.
//!
//! All public items are re-exported through [`super`].

use crate::harness::events::HarnessRunStatus;
use crate::harness::middleware::AgentRun;

/// The full result of an agent-loop invocation: the accumulated [`AgentRun`]
/// plus a compact [`HarnessRunStatus`] snapshot.
///
/// [`crate::harness::runtime::AgentHarness::invoke`] returns only the
/// [`AgentRun`]; callers that also want the run's lifecycle status (phase,
/// counters, timing, error summary) can use
/// [`crate::harness::runtime::AgentHarness::invoke_with_status`] and read the
/// `status` field.
#[derive(Clone, Debug)]
pub struct AgentLoopResult {
    /// The accumulated transcript, usage, counters, and final response.
    pub run: AgentRun,
    /// A compact lifecycle/status snapshot reflecting how the run ended.
    pub status: HarnessRunStatus,
}
