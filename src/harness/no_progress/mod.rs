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

mod types;

use types::LadderState;
pub use types::{NoProgress, NoProgressTracker, ToolAttempt};

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
mod test;
