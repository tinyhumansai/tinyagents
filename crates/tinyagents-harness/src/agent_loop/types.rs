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

use std::collections::{BTreeMap, BTreeSet};

use super::*;

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
    /// One or more tool calls in the last batch need a human decision or
    /// host-side execution before the run can continue (A2). The transcript
    /// keeps the assistant's tool-call row and every non-deferred sibling's
    /// result; resume with
    /// [`crate::runtime::AgentHarness::resume_deferred`].
    Deferred(crate::tool::DeferredToolRequests),
}

/// The effect of draining a pending [`crate::context::MiddlewareControl`] at
/// one of the loop's safe checkpoints.
///
/// Kept distinct from [`LoopExit`] because not every drained control ends the
/// run: [`crate::context::MiddlewareControl::JumpTo`]`(`[`crate::context::LoopTarget::Model`]`)`
/// must abandon the current turn (skip whatever the checkpoint's caller was
/// about to do next) without exiting the loop body, which a plain
/// `Option<LoopExit>` cannot express.
#[derive(Clone, Debug)]
pub(crate) enum ControlEffect {
    /// Nothing to do; the checkpoint's caller proceeds as it otherwise would.
    None,
    /// Abandon the rest of this turn and restart the loop body from the top.
    ContinueLoop,
    /// The run is done; propagate this [`LoopExit`] to the caller.
    Exit(LoopExit),
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

/// The recovery counters and boosted output cap of the turn in flight.
///
/// Every counter is consecutive-per-turn, not per-run (the output-validation
/// retry budget, which is run-wide, lives outside this struct). The one
/// run-wide member is [`Self::reasoning_fallback`]: its hold-off outlives the
/// turn that set it and no turn-boundary reset touches it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct TurnRecovery {
    /// Retries of a length-truncated empty reply
    /// (`RunPolicy::truncated_empty_retries`).
    pub(super) truncated_empty_retries_used: u32,
    /// "Stop deliberating" re-prompts once the retries above are spent
    /// (`RunPolicy::truncated_empty_nudges`).
    pub(super) truncated_empty_nudges_used: u32,
    /// Retries of a normally-finished reply with no visible answer
    /// (`RunPolicy::empty_response_retries`).
    pub(super) empty_response_retries_used: u32,
    /// Consecutive "you said tool_calls but sent none" re-prompts
    /// (`RunPolicy::dropped_tool_call_nudges`).
    pub(super) dropped_tool_call_nudges_used: u32,
    /// Consecutive re-prompts after a call written on a turn with no callable
    /// tool (bounded by the same `dropped_tool_call_nudges`).
    pub(super) withheld_call_nudges_used: u32,
    /// Consecutive length-truncated tool turns answered with errors
    /// (`RunPolicy::truncated_tool_call_retries`); reset by any tool turn that
    /// was not cut off, never by the truncated turn itself.
    pub(super) truncated_tool_call_retries_used: u32,
    /// Tools whose call was cut off on the most recent truncated turn, so a
    /// repeat of the same oversized shape can be told apart from a new one.
    pub(super) truncated_tool_names: std::collections::BTreeSet<String>,
    /// Tools already answered with the repeat corrective in this logical
    /// turn; one more cut-off call of any of them ends the run.
    pub(super) truncated_repeat_names: std::collections::BTreeSet<String>,
    /// Overrides the next request's output cap after a length truncation.
    pub(super) boosted_max_tokens: Option<u32>,
    /// The original output cap, so growth stays clamped at 4x.
    pub(super) truncation_base: Option<u32>,
    /// Reasoning switched off for the calls after a dead one (see
    /// `RunPolicy::truncated_empty_reasoning_fallback`). Run-wide: the
    /// hold-off outlives the turn that set it.
    pub(super) reasoning_fallback: super::reasoning_fallback::ReasoningFallback,
}

/// The tool schemas a run advertises and the discovery state behind them.
pub(super) struct ToolSurface {
    /// The wire list for the turn in flight: direct tools, promoted tools,
    /// then the discovery bridge schemas. Rebuilt by
    /// [`ToolSurface::assemble_turn_schemas`] each turn.
    pub(super) tool_schemas: Vec<ToolSchema>,
    /// `Direct`-exposure count at run start, captured before the bridge
    /// schemas are appended so `ToolsAdvertised.direct` is not inflated.
    pub(super) direct_schema_count: usize,
    /// The direct set (registry plus toolset chain); replaced when the
    /// toolset's live set changes mid-run.
    pub(super) direct_tool_schemas: Vec<ToolSchema>,
    /// What the transcript has actually been told about the toolset chain's
    /// tools so far (B6). Starts empty so the first-turn diff always declares
    /// the full initial set; bridge schemas are deliberately excluded.
    pub(super) declared_tool_schemas: Vec<ToolSchema>,
    /// The intrinsic discovery-bridge schemas; constant within a run.
    pub(super) bridge_schemas: Vec<ToolSchema>,
    /// Deferred tools reachable through `tool_search`.
    pub(super) deferred_catalog: crate::tool::discover::DeferredCatalog,
    /// Typed declarations of discovered tools already promoted.
    pub(super) promoted_schemas: BTreeMap<String, ToolSchema>,
    /// Names returned by a successful intrinsic search; grown by tool
    /// execution.
    pub(super) promoted_names: BTreeSet<String>,
    /// Promoted names already recorded on the transcript as a patch.
    pub(super) recorded_promotions: BTreeSet<String>,
}

/// What the recovery stages need to know about the response in hand.
pub(super) struct ResponseTurn<'a> {
    pub(super) call_id: &'a CallId,
    pub(super) response: &'a ModelResponse,
    pub(super) tool_calls: &'a [ToolCall],
    /// The output cap actually sent with the request that produced `response`.
    pub(super) attempt_max_tokens: Option<u32>,
    /// When the model call that produced `response` started (`ids::now_ms`),
    /// so a recovery can weigh a retry against how long the call just took.
    pub(super) started_at_ms: u64,
    /// What the dialect layer recovered or withheld from the response text.
    pub(super) recovery: &'a super::dialect::TextRecovery,
    /// Whether the request offered a callable tool this turn.
    pub(super) tools_available: bool,
    /// Whether text-dialect call recovery applied to this response (a forced
    /// text dialect, or `RunPolicy::text_dialect_recovery` enabled for it).
    pub(super) text_dialect_calls_recoverable: bool,
    /// Whether this turn carried a structured-output plan.
    pub(super) has_structured_plan: bool,
    /// Names of the structured-output tool(s) the plan can call.
    pub(super) structured_call_names: &'a [String],
}

/// How [`AgentHarness::reject_truncated_tool_calls`] left the turn.
pub(super) enum TruncationOutcome {
    /// No call was cut off; the turn proceeds normally.
    Clean,
    /// Suspect calls are marked for an error answer at admission; the turn
    /// proceeds (the model retries them next turn).
    CallsRejected,
    /// The turn was failed and settled here. `None` restarts the loop;
    /// `Some(exit)` ends the run.
    EndTurn(Option<LoopExit>),
}

/// What the loop does after a mixed turn has been handled.
pub(super) enum TurnFlow {
    /// Run another turn of the loop.
    NextTurn,
    /// The run is over; propagate this exit to the caller.
    Exit(LoopExit),
}

/// The pieces of a mixed turn, split out of its response.
pub(super) struct MixedStructuredTurn<'a> {
    pub(super) response: ModelResponse,
    pub(super) structured_plan: Option<&'a (StructuredStrategy, String, Value)>,
    /// The structured-output schema call(s).
    pub(super) structured_hits: Vec<ToolCall>,
    /// The genuine tool calls alongside them.
    pub(super) real_tool_calls: Vec<ToolCall>,
    /// A length stop cut one of the real calls off.
    pub(super) turn_had_truncated_calls: bool,
}
