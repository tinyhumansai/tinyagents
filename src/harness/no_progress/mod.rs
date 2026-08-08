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
//!
//! # Driving this from an `after_tool` hook (wave 2)
//!
//! The ladder is complete and tested but **nothing in the crate drives it**, so
//! a model looping on the same failing tool call is bounded only by
//! `RunLimits::max_tool_calls` (50) — roughly 48 wasted round trips. The
//! middleware that closes that gap lives in `middleware/library/`; here is the
//! exact contract it must implement.
//!
//! Hold one [`NoProgressTracker`] per turn behind a shared reference (`record`
//! takes `&self` and is interior-mutable, so an `&self` hook needs no
//! `RefCell`). In `after_tool`:
//!
//! 1. Fingerprint the call arguments with [`fingerprint_arguments`]. **Do not
//!    roll your own** — the identical-repeat rung compares fingerprints, so two
//!    drivers disagreeing on canonicalisation would silently change when the
//!    ladder trips.
//! 2. Build the attempt:
//!    - success → [`ToolAttempt::success`]
//!    - failure → [`ToolAttempt::failure`], then
//!      - `.hard_reject()` when the failure is a security/approval denial (a
//!        blocked call re-issued unchanged can never succeed),
//!      - `.recoverable_miss()` for the unknown-tool recovery sentinel, i.e.
//!        the case that raises [`crate::harness::events::AgentEvent::UnknownToolCall`].
//! 3. Pass `step` = the run's current model-call count
//!    (`LimitTracker::model_calls()`). It is used only for the "no progress
//!    since step X" wording, so an approximation is harmless — but it must be
//!    monotonic or the message misleads.
//! 4. Route the verdict:
//!    - [`NoProgress::Continue`] → do nothing.
//!    - [`NoProgress::Nudge`] → append the message as a **system** message to
//!      the working transcript so the next model call sees it, and continue.
//!      Injecting it as a tool result instead would attribute the harness's
//!      instruction to the tool.
//!    - [`NoProgress::Halt`] → stop the turn and surface the message as the
//!      final response. The tracker has already reset itself, so a resumed run
//!      does not immediately re-trip on latched state.
//!
//! [`NoProgress::message`], [`NoProgress::is_nudge`], [`NoProgress::is_halt`]
//! and [`NoProgress::as_str`] exist so step 4 needs no enum match, and
//! `as_str()` gives a stable telemetry label.

mod successful_repeat;
mod types;

pub use successful_repeat::{DEFAULT_REPEAT_CALL_THRESHOLD, DEFAULT_REPEAT_OUTPUT_THRESHOLD};
use types::LadderState;
pub use types::{
    NoProgress, NoProgressTracker, SuccessfulRepeat, SuccessfulRepeatTracker, ToolAttempt,
};

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

/// Computes the stable argument fingerprint the identical-repeat rung compares
/// on.
///
/// The ladder's central question is "did the model re-issue the *same* call?",
/// which is only answerable if every driver canonicalises arguments the same
/// way. JSON object key order is not significant but `serde_json::Value`'s
/// default `to_string` preserves insertion order, so two logically identical
/// argument objects can render differently — enough to make a genuine repeat
/// look novel and let the loop run to the tool-call cap instead.
///
/// This sorts object keys recursively, then hashes, so the result is:
///
/// - **order-independent** for objects,
/// - **order-sensitive** for arrays (list order *is* semantic),
/// - short and allocation-cheap to carry in a [`ToolAttempt`].
///
/// # Example
///
/// ```
/// use tinyagents::harness::no_progress::fingerprint_arguments;
/// use serde_json::json;
///
/// let a = fingerprint_arguments(&json!({"path": "/tmp", "depth": 2}));
/// let b = fingerprint_arguments(&json!({"depth": 2, "path": "/tmp"}));
/// assert_eq!(a, b, "key order must not change the fingerprint");
///
/// let c = fingerprint_arguments(&json!({"path": "/var", "depth": 2}));
/// assert_ne!(a, c);
/// ```
pub fn fingerprint_arguments(arguments: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hash_canonical(arguments, &mut hasher);
    // 16 hex chars (64 bits) is far more than enough to separate the handful of
    // distinct calls within one turn, and keeps the signature string short.
    hasher
        .finalize()
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Feeds `value` into `hasher` with object keys visited in sorted order.
fn hash_canonical(value: &serde_json::Value, hasher: &mut impl sha2::Digest) {
    match value {
        serde_json::Value::Object(map) => {
            hasher.update(b"{");
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            for key in keys {
                hasher.update(key.as_bytes());
                hasher.update(b":");
                hash_canonical(&map[key], hasher);
                hasher.update(b",");
            }
            hasher.update(b"}");
        }
        serde_json::Value::Array(items) => {
            hasher.update(b"[");
            for item in items {
                hash_canonical(item, hasher);
                hasher.update(b",");
            }
            hasher.update(b"]");
        }
        other => hasher.update(other.to_string().as_bytes()),
    }
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
mod test;
