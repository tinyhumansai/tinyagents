//! Foundational no-progress primitive.
//!
//! Agents that hit an unproductive tool result tend to *retry the same
//! strategy* — the identical tool with the identical arguments — instead of
//! adapting. This module is the reusable detector that breaks that pattern: it
//! tracks recent `(tool, args) → outcome` across a turn and, on each failure,
//! decides whether the loop is still making progress, should be **nudged** to
//! change approach, or has exhausted its same-strategy retries and must
//! **halt**.
//!
//! The escalation ladder caps same-strategy retries *before* giving up: a
//! repeated identical failure first feeds a structured "no progress since step
//! X" signal back into the loop (a nudge) so the model picks a *different* next
//! action, and only halts if it keeps re-issuing the same failing call.
//!
//! It is deliberately free of harness types so it can be unit tested in
//! isolation and reused by higher-level reliability layers. A driver (typically
//! a middleware) feeds each tool outcome in via [`NoProgressTracker::record`]
//! and turns the returned [`NoProgress`] verdict into a steering nudge
//! (`Nudge`) or a halt (`Halt`).

use std::sync::Mutex;

/// Consecutive **identical** (tool + args + error) failures tolerated before the
/// ladder halts the run — a call re-issued unchanged that keeps failing can
/// never succeed.
pub const DEFAULT_IDENTICAL_HALT_THRESHOLD: usize = 3;
/// Identical repeats that trigger the **nudge** — one below the halt threshold,
/// so the model gets exactly one corrective chance to change strategy before the
/// same-strategy retry cap trips.
const IDENTICAL_NUDGE_THRESHOLD: usize = 2;
/// Consecutive **any**-failure no-progress backstop: different commands all
/// failing means the goal is unreachable here.
const NO_PROGRESS_HALT_THRESHOLD: usize = 6;
/// Consecutive varied failures that trigger the **nudge** before the any-failure
/// backstop halts.
const NO_PROGRESS_NUDGE_THRESHOLD: usize = 4;
/// Consecutive identical **hard policy rejections** before halting — a blocked
/// call re-issued unchanged can never succeed.
const HARD_REJECT_HALT_THRESHOLD: usize = 2;

/// One tool call's outcome, reduced to the deterministic parts the ladder
/// compares. Built by the driver from a tool result plus the argument
/// fingerprint captured before execution.
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
struct LadderState {
    /// Signature of the previous failing call (tool + args + first error line).
    last_sig: Option<String>,
    /// Consecutive repeats of `last_sig`.
    same_count: usize,
    /// Consecutive failures of any kind (reset by any success).
    consecutive: usize,
    /// Signature we have already nudged on, so a nudge fires at most once per
    /// distinct failing `(tool, args, error)` before escalating to a halt.
    nudged_sig: Option<String>,
    /// `true` once the varied-failure nudge fired for the current streak.
    nudged_streak: bool,
}

/// Tracks recent tool outcomes and drives the no-progress escalation ladder.
///
/// Cheap to construct and interior-mutable, so a middleware can hold one behind
/// a shared reference for the whole turn. `identical_halt_threshold` is the
/// same-strategy retry cap; it is clamped so a nudge always precedes a halt.
pub struct NoProgressTracker {
    identical_halt_threshold: usize,
    state: Mutex<LadderState>,
}

impl NoProgressTracker {
    /// Build a tracker whose identical-repeat halt threshold is
    /// `identical_halt_threshold`, clamped up so it always sits above the nudge
    /// threshold (a single failure is never a loop, and the nudge must land
    /// before the halt).
    pub fn new(identical_halt_threshold: usize) -> Self {
        Self {
            identical_halt_threshold: identical_halt_threshold.max(IDENTICAL_NUDGE_THRESHOLD + 1),
            state: Mutex::new(LadderState::default()),
        }
    }

    /// Clear every counter. Called after a halt so a resumed run does not
    /// immediately re-trip on the same latched state.
    pub fn reset(&self) {
        *self.state.lock().unwrap() = LadderState::default();
    }

    /// Record one tool outcome observed at loop `step` (the current model-call
    /// count, used only for the "no progress since step X" wording) and return
    /// the ladder's verdict. On a [`NoProgress::Halt`] the internal state is
    /// reset for the caller.
    pub fn record(&self, step: usize, attempt: &ToolAttempt) -> NoProgress {
        let mut state = self.state.lock().unwrap();

        let Some(err) = attempt.error else {
            // Success → progress was made; clear every counter.
            *state = LadderState::default();
            return NoProgress::Continue;
        };

        // Signature: tool name + argument fingerprint + first error line (the
        // deterministic parts; a huge payload tail must not dominate the
        // identical-repeat comparison).
        let err_line = err.lines().next().unwrap_or(err);
        let sig = format!(
            "{}\u{1f}{}\u{1f}{err_line}",
            attempt.tool, attempt.arg_fingerprint
        );

        // The unknown-tool recovery is correctable feedback the model already
        // received, so it must NOT feed the generic any-failure backstop — else a
        // turn that recovers from one bad tool name and then legitimately
        // exhausts its budget would trip the backstop instead of hitting the cap.
        // It still feeds the identical-repeat counter below.
        if !attempt.recoverable_miss {
            state.consecutive += 1;
        }

        let same_count = match &state.last_sig {
            Some(prev) if *prev == sig => {
                state.same_count += 1;
                state.same_count
            }
            _ => {
                state.last_sig = Some(sig.clone());
                state.same_count = 1;
                // A fresh signature is eligible for its own nudge again.
                state.nudged_sig = None;
                1
            }
        };

        // ── Halt: same-strategy retries exhausted ───────────────────────────
        // A hard policy rejection can never succeed re-issued unchanged, so it
        // trips fastest.
        if attempt.hard_reject && same_count >= HARD_REJECT_HALT_THRESHOLD {
            let summary = format!(
                "Stopping: the `{}` call is blocked by the security policy and was re-issued with \
                 identical arguments — it can never succeed this way. Reason:\n{}\n\nDo not repeat \
                 this call; use an allowed alternative or report that it can't be done here.",
                attempt.tool,
                truncate_for_halt(err),
            );
            *state = LadderState::default();
            return NoProgress::Halt(summary);
        }
        if same_count >= self.identical_halt_threshold {
            let summary = format!(
                "Stopping: the `{}` call was retried {same_count} times with identical arguments \
                 and kept failing — repeating it will not help. Last error:\n{}\n\nThis looks \
                 unrecoverable in the current environment. Report this back instead of retrying.",
                attempt.tool,
                truncate_for_halt(err),
            );
            *state = LadderState::default();
            return NoProgress::Halt(summary);
        }
        if state.consecutive >= NO_PROGRESS_HALT_THRESHOLD {
            let summary = format!(
                "Stopping: {} tool calls in a row failed with no progress. Last error (from \
                 `{}`):\n{}\n\nDifferent commands are all failing — the goal looks unreachable in \
                 this environment. Report this back instead of retrying.",
                state.consecutive,
                attempt.tool,
                truncate_for_halt(err),
            );
            *state = LadderState::default();
            return NoProgress::Halt(summary);
        }

        // ── Nudge: cap retries *before* forcing an alternative ──────────────
        // Same tool + same args + same error just repeated: give the model one
        // corrective chance to change strategy before the halt threshold.
        if same_count == IDENTICAL_NUDGE_THRESHOLD && state.nudged_sig.as_deref() != Some(&sig) {
            state.nudged_sig = Some(sig);
            return NoProgress::Nudge(identical_nudge(step, attempt.tool, same_count, err));
        }
        // Varied failures piling up with no success: step back before the
        // any-failure backstop halts.
        if !attempt.recoverable_miss
            && state.consecutive == NO_PROGRESS_NUDGE_THRESHOLD
            && !state.nudged_streak
        {
            state.nudged_streak = true;
            return NoProgress::Nudge(varied_nudge(step, attempt.tool, state.consecutive, err));
        }

        NoProgress::Continue
    }
}

/// The structured "no progress since step X" corrective for an identical
/// repeated failure — the core case (same tool, same args, same error).
fn identical_nudge(step: usize, tool: &str, count: usize, err: &str) -> String {
    format!(
        "[no progress since step {step}] The `{tool}` call has now failed {count} times with the \
         same arguments and the same error — you are retrying an identical action that cannot \
         succeed as-is. Do NOT repeat it. Change strategy on your next step: use a different tool \
         or different arguments (for a missing path, enumerate the directory first; for a failing \
         query, correct or broaden it), or report back that it can't be done here. Last error:\n{}",
        truncate_for_halt(err),
    )
}

/// The structured "no progress since step X" corrective for a run of varied
/// failures — different commands all failing without progress.
fn varied_nudge(step: usize, tool: &str, count: usize, err: &str) -> String {
    format!(
        "[no progress since step {step}] {count} tool calls in a row have failed without making \
         progress. Stop cycling through variations of the same approach — step back and try a \
         different strategy (enumerate/inspect before acting, pick a different tool, or narrow the \
         goal). Last error (from `{tool}`):\n{}",
        truncate_for_halt(err),
    )
}

/// Trim a tool error for inclusion in a nudge/halt summary (keep it bounded but
/// retain the deterministic leading detail the model/user needs). Char-safe so a
/// multibyte boundary is never split.
fn truncate_for_halt(text: &str) -> String {
    const MAX: usize = 600;
    if text.chars().count() <= MAX {
        text.to_string()
    } else {
        let head: String = text.chars().take(MAX).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fail<'a>(tool: &'a str, fp: &'a str, err: &'a str) -> ToolAttempt<'a> {
        ToolAttempt {
            tool,
            arg_fingerprint: fp,
            error: Some(err),
            hard_reject: false,
            recoverable_miss: false,
        }
    }

    fn ok<'a>(tool: &'a str, fp: &'a str) -> ToolAttempt<'a> {
        ToolAttempt {
            tool,
            arg_fingerprint: fp,
            error: None,
            hard_reject: false,
            recoverable_miss: false,
        }
    }

    #[test]
    fn identical_failure_nudges_then_halts() {
        let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        // First identical failure: not enough repetition yet.
        assert_eq!(
            t.record(1, &fail("read_file", "a", "not found")),
            NoProgress::Continue
        );
        // Second: nudge the model to change strategy before the retry cap.
        match t.record(2, &fail("read_file", "a", "not found")) {
            NoProgress::Nudge(msg) => {
                assert!(msg.contains("no progress since step 2"));
                assert!(msg.contains("read_file"));
                assert!(msg.contains("Change strategy"));
            }
            other => panic!("expected a nudge on the second identical failure, got {other:?}"),
        }
        // Third: same-strategy retries exhausted → halt.
        match t.record(3, &fail("read_file", "a", "not found")) {
            NoProgress::Halt(msg) => assert!(msg.contains("retried 3 times")),
            other => panic!("expected a halt on the third identical failure, got {other:?}"),
        }
    }

    #[test]
    fn a_success_resets_the_ladder() {
        let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        let _ = t.record(1, &fail("t", "a", "boom"));
        let _ = t.record(2, &fail("t", "a", "boom")); // nudge
        assert_eq!(t.record(3, &ok("t", "a")), NoProgress::Continue);
        // After the reset, two more identical failures only re-nudge — no halt.
        assert_eq!(t.record(4, &fail("t", "a", "boom")), NoProgress::Continue);
        assert!(matches!(
            t.record(5, &fail("t", "a", "boom")),
            NoProgress::Nudge(_)
        ));
    }

    #[test]
    fn changed_arguments_clear_the_identical_streak() {
        let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        let _ = t.record(1, &fail("read_file", "a", "not found"));
        // The model heeded the nudge and changed args: a new signature, so the
        // identical-repeat counter starts over and never reaches the halt.
        assert_eq!(
            t.record(2, &fail("read_file", "b", "not found")),
            NoProgress::Continue
        );
        assert_eq!(
            t.record(3, &fail("list_dir", "c", "denied")),
            NoProgress::Continue
        );
    }

    #[test]
    fn a_hard_rejection_halts_on_the_second_identical_repeat() {
        let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        let mut a = fail("send_email", "a", "[policy-blocked] denied");
        a.hard_reject = true;
        assert_eq!(t.record(1, &a), NoProgress::Continue);
        let mut b = fail("send_email", "a", "[policy-blocked] denied");
        b.hard_reject = true;
        assert!(
            matches!(t.record(2, &b), NoProgress::Halt(msg) if msg.contains("security policy"))
        );
    }

    #[test]
    fn varied_failures_nudge_then_hit_the_backstop() {
        let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        // Distinct signatures each time: the identical counter never climbs, but
        // the any-failure streak does.
        for i in 1..=3 {
            let (fp, err) = (format!("fp{i}"), format!("err{i}"));
            assert_eq!(t.record(i, &fail("t", &fp, &err)), NoProgress::Continue);
        }
        // Fourth consecutive varied failure → nudge to step back.
        assert!(matches!(
            t.record(4, &fail("t", "fp4", "err4")),
            NoProgress::Nudge(_)
        ));
        assert_eq!(t.record(5, &fail("t", "fp5", "err5")), NoProgress::Continue);
        // Sixth → the no-progress backstop halts.
        assert!(matches!(
            t.record(6, &fail("t", "fp6", "err6")),
            NoProgress::Halt(msg) if msg.contains("in a row failed")
        ));
    }

    #[test]
    fn unknown_tool_recovery_does_not_feed_the_backstop() {
        let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        // Six correctable misses with *distinct* signatures must not trip the
        // any-failure backstop (they don't count toward `consecutive`).
        for i in 0..6 {
            let (fp, err) = (format!("fp{i}"), format!("unknown tool foo{i}"));
            let mut a = fail("__unknown_tool__", &fp, &err);
            a.recoverable_miss = true;
            assert_eq!(t.record(i, &a), NoProgress::Continue);
        }
    }

    #[test]
    fn identical_halt_threshold_is_clamped_above_the_nudge() {
        // A caller asking for a halt threshold of 1 or 2 still gets a nudge
        // before the halt (the nudge lands at 2, so the halt must be >= 3).
        let t = NoProgressTracker::new(1);
        assert_eq!(t.record(1, &fail("t", "a", "boom")), NoProgress::Continue);
        assert!(matches!(
            t.record(2, &fail("t", "a", "boom")),
            NoProgress::Nudge(_)
        ));
        assert!(matches!(
            t.record(3, &fail("t", "a", "boom")),
            NoProgress::Halt(_)
        ));
    }
}
