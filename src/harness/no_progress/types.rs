//! Type definitions for the no-progress escalation ladder
//! (`ToolAttempt`, `NoProgress`, `LadderState`, `NoProgressTracker`).
//!
//! Split out of `no_progress/mod.rs`; see that module's doc comment for
//! the full escalation-ladder design.

use std::sync::Mutex;

pub struct ToolAttempt<'a> {
    /// Tool name.
    pub tool: &'a str,
    /// Stable fingerprint of the call arguments (computed by the driver). Folded
    /// into the identical-repeat signature so the "identical arguments" ladder
    /// only trips when the args truly repeat.
    pub arg_fingerprint: &'a str,
    /// `None` on success; otherwise the tool's error text.
    pub error: Option<&'a str>,
    /// `true` when the result is a hard security/approval rejection that can
    /// never succeed re-issued unchanged.
    pub hard_reject: bool,
    /// `true` for the unknown-tool recovery sentinel — a correctable miss that
    /// must not feed the generic any-failure backstop (it still feeds the
    /// identical-repeat counter, so re-issuing the *same* unavailable tool
    /// halts).
    pub recoverable_miss: bool,
}

/// The ladder's verdict for one recorded attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoProgress {
    /// Progress was made, or not enough repetition yet — carry on.
    Continue,
    /// Same-strategy repetition detected below the retry cap: feed this
    /// structured "no progress since step X" corrective back into the loop so
    /// the model picks a *different* next action.
    Nudge(String),
    /// Same-strategy retries exhausted (or the any-failure backstop tripped):
    /// halt with this root-cause summary.
    Halt(String),
}

#[derive(Default)]
pub(super) struct LadderState {
    /// Signature of the previous failing call (tool + args + first error line).
    pub(super) last_sig: Option<String>,
    /// Consecutive repeats of `last_sig`.
    pub(super) same_count: usize,
    /// Consecutive failures of any kind (reset by any success).
    pub(super) consecutive: usize,
    /// Signature we have already nudged on, so a nudge fires at most once per
    /// distinct failing `(tool, args, error)` before escalating to a halt.
    pub(super) nudged_sig: Option<String>,
    /// `true` once the varied-failure nudge fired for the current streak.
    pub(super) nudged_streak: bool,
}

/// Tracks recent tool outcomes and drives the no-progress escalation ladder.
///
/// Cheap to construct and interior-mutable, so a middleware can hold one behind
/// a shared reference for the whole turn. `identical_halt_threshold` is the
/// same-strategy retry cap; it is clamped so a nudge always precedes a halt.
pub struct NoProgressTracker {
    pub(super) identical_halt_threshold: usize,
    pub(super) state: Mutex<LadderState>,
}

/// Verdict returned after recording a successful-repeat signal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuccessfulRepeat {
    /// The signature changed, is exempt, failed, or remains below its threshold.
    Continue,
    /// The same successful action has repeated enough times to be considered
    /// stuck. The message is suitable for steering or a halt summary.
    Halt(String),
}

#[derive(Default)]
pub(super) struct Streak {
    pub(super) last_hash: Option<u64>,
    pub(super) consecutive: u32,
}

/// Tracks identical assistant-output and successful tool-call batches.
///
/// The two streaks are independent, but their verdict timing is coordinated:
/// output is staged before tools execute and can halt only after the matching
/// call batch is recorded as successful and non-exempt. Exempt polling batches
/// and failed batches reset both streaks so the failure ladder remains
/// authoritative.
pub struct SuccessfulRepeatTracker {
    pub(super) output_threshold: u32,
    pub(super) call_threshold: u32,
    pub(super) output: Mutex<Streak>,
    pub(super) calls: Mutex<Streak>,
}
