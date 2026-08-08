//! Types for run-scoped limit enforcement.
//!
//! [`RunLimits`] carries the configured policy — including the
//! [`RunLimits::max_depth`] recursion cap that bounds how far the sub-agent /
//! sub-graph run tree may nest; [`LimitTracker`] (in the sibling `mod.rs`)
//! holds the live counters and checks them against the policy.

/// Configures the hard limits applied across a single harness run.
///
/// All limits are checked fail-closed: the first call that exceeds a cap
/// returns an error and the run should be stopped.
///
/// # Examples
///
/// ```
/// use tinyagents::harness::limits::RunLimits;
///
/// let limits = RunLimits::default()
///     .with_max_model_calls(10)
///     .with_max_tool_calls(20);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RunLimits {
    /// Maximum number of model API calls permitted for this run.
    pub max_model_calls: usize,
    /// Maximum number of tool invocations permitted for this run.
    pub max_tool_calls: usize,
    /// Maximum elapsed wall-clock time in milliseconds. `None` means no limit.
    pub max_wall_clock_ms: Option<u64>,
    /// Maximum number of retry *attempts* (not counting the first try)
    /// permitted for an individual model call. Reconciled with
    /// [`crate::harness::retry::RetryPolicy::max_attempts`] by the agent loop
    /// (see [`crate::harness::retry::RetryPolicy::max_attempts_capped_at`]):
    /// whichever of the two is stricter wins, so this is a hard ceiling a
    /// looser `RetryPolicy` cannot exceed.
    pub max_retries_per_call: usize,
    /// Maximum sub-agent / recursion depth allowed for the run tree rooted at
    /// this run. A top-level run is depth `0`; each nested child run increments
    /// the depth. A sub-agent invocation whose child depth would exceed this cap
    /// fails fast (see [`crate::harness::subagent`]). Defaults to
    /// [`RunLimits::DEFAULT_MAX_DEPTH`].
    pub max_depth: usize,
    /// What the run should do when a call cap is reached. Defaults to
    /// [`LimitBehavior::Error`], which is the historical behaviour.
    pub behavior: LimitBehavior,
}

/// What a run does when it reaches a configured call cap.
///
/// Ported from LangChain's `exit_behavior` on `ModelCallLimitMiddleware` /
/// `ToolCallLimitMiddleware`. Every cap here used to be a hard error, which
/// throws away the whole run *and everything it already accomplished* — for a
/// long research run that is often the worst possible outcome, since the
/// partial answer was the valuable part.
///
/// The variant only names the *policy*; the agent loop is what acts on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LimitBehavior {
    /// Fail the run with
    /// [`TinyAgentsError::LimitExceeded`][crate::error::TinyAgentsError::LimitExceeded].
    /// The historical (and still default) behaviour.
    #[default]
    Error,

    /// Stop the loop cleanly and return whatever the run has produced so far,
    /// as if the model had finished its turn.
    ///
    /// # Contract for the agent loop (wave 2)
    ///
    /// When [`LimitTracker::record_model_call`][crate::harness::limits::LimitTracker::record_model_call]
    /// or [`record_tool_call`][crate::harness::limits::LimitTracker::record_tool_call]
    /// reports exhaustion under this behaviour they return
    /// [`LimitOutcome::Stop`] instead of `Err`, and the loop must:
    ///
    /// 1. Emit the existing
    ///    [`AgentEvent::LimitReached`][crate::harness::events::AgentEvent::LimitReached]
    ///    so the stop is still observable and still distinguishable from a
    ///    model that simply finished.
    /// 2. Break out of the loop and finalize normally, keeping the transcript
    ///    accumulated so far as the run result.
    /// 3. For the **tool** cap specifically, mirror LangChain's `"end"` path:
    ///    append a tool result for every remaining requested call saying it was
    ///    stopped before it could run (the provider APIs require every
    ///    `tool_call_id` to be answered, so skipping them corrupts the
    ///    transcript), and **roll the counter back** by the number of calls that
    ///    never executed via
    ///    [`LimitTracker::rollback_tool_calls`][crate::harness::limits::LimitTracker::rollback_tool_calls],
    ///    so the reported count reflects work actually done.
    StopWithPartial,
}

impl LimitBehavior {
    /// Stable, snake_case label for logs and telemetry dimensions.
    pub fn as_str(self) -> &'static str {
        match self {
            LimitBehavior::Error => "error",
            LimitBehavior::StopWithPartial => "stop_with_partial",
        }
    }
}

/// The result of recording a call against a [`RunLimits`] cap.
///
/// Returned by the `try_record_*` methods so the caller sees the cap decision
/// as data rather than only as a `Result`. The `record_*` methods remain for
/// callers that always want the hard-error form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitOutcome {
    /// The call is within the cap; carry on.
    Proceed,
    /// The cap is exhausted and [`RunLimits::behavior`] is
    /// [`LimitBehavior::StopWithPartial`]: stop the loop cleanly and return
    /// what the run has so far. Carries which cap tripped.
    Stop(LimitKind),
}

/// Names which cap a [`LimitOutcome::Stop`] refers to.
///
/// Deliberately mirrors
/// [`crate::harness::events::LimitKind`] rather than reusing it, so the limits
/// module (a leaf with no event dependency) does not have to import the
/// observability layer. Convert with [`LimitKind::as_str`] when emitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitKind {
    /// The per-run model-call cap.
    ModelCalls,
    /// The per-run tool-call cap.
    ToolCalls,
}

impl LimitKind {
    /// Stable, snake_case label matching
    /// [`crate::harness::events::LimitKind::as_str`].
    pub fn as_str(self) -> &'static str {
        match self {
            LimitKind::ModelCalls => "model_calls",
            LimitKind::ToolCalls => "tool_calls",
        }
    }
}

impl RunLimits {
    /// Default sub-agent / recursion depth cap when none is configured.
    pub const DEFAULT_MAX_DEPTH: usize = 8;
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_model_calls: 25,
            max_tool_calls: 50,
            max_wall_clock_ms: None,
            max_retries_per_call: 3,
            max_depth: Self::DEFAULT_MAX_DEPTH,
            behavior: LimitBehavior::Error,
        }
    }
}
