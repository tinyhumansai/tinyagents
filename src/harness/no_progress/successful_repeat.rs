//! Successful-repeat progress detection.
//!
//! [`NoProgressTracker`] handles failing tool calls, but deliberately resets on
//! success. That leaves a second loop shape undetected: a model can repeatedly
//! emit the same response and successfully invoke the same no-op tool call.
//! This tracker owns the provider- and product-neutral streak accounting for
//! those loops; a harness middleware remains responsible for building canonical
//! signatures and deciding which polling tools are exempt.

use std::hash::{Hash, Hasher};
use std::sync::Mutex;

/// Consecutive identical assistant-output batches required to halt.
pub const DEFAULT_REPEAT_OUTPUT_THRESHOLD: u32 = 4;
/// Consecutive identical successful tool-call batches required to halt.
pub const DEFAULT_REPEAT_CALL_THRESHOLD: u32 = 3;

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
struct Streak {
    last_hash: Option<u64>,
    consecutive: u32,
}

impl Streak {
    fn record(&mut self, signature: &str) -> u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        signature.hash(&mut hasher);
        let hash = hasher.finish();
        if self.last_hash == Some(hash) {
            self.consecutive += 1;
        } else {
            self.last_hash = Some(hash);
            self.consecutive = 1;
        }
        self.consecutive
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Tracks identical assistant-output and successful tool-call batches.
///
/// The two streaks are independent: output is observed before tools execute,
/// while a call batch is recorded only after every result is known. Exempt
/// polling batches reset their streak; a failed call batch also resets the
/// successful-call streak so the failure ladder remains authoritative.
pub struct SuccessfulRepeatTracker {
    output_threshold: u32,
    call_threshold: u32,
    output: Mutex<Streak>,
    calls: Mutex<Streak>,
}

impl Default for SuccessfulRepeatTracker {
    fn default() -> Self {
        Self::new(
            DEFAULT_REPEAT_OUTPUT_THRESHOLD,
            DEFAULT_REPEAT_CALL_THRESHOLD,
        )
    }
}

impl SuccessfulRepeatTracker {
    /// Builds a tracker. Thresholds are clamped to one so `0` cannot disable a
    /// safety guard accidentally; callers that do not want this guard should
    /// omit the tracker.
    pub fn new(output_threshold: u32, call_threshold: u32) -> Self {
        Self {
            output_threshold: output_threshold.max(1),
            call_threshold: call_threshold.max(1),
            output: Mutex::new(Streak::default()),
            calls: Mutex::new(Streak::default()),
        }
    }

    /// Records the canonical visible-output plus tool-call signature produced
    /// by one assistant iteration.
    pub fn record_output(&self, signature: &str, exempt: bool) -> SuccessfulRepeat {
        let mut output = self.output.lock().unwrap();
        if exempt {
            output.reset();
            return SuccessfulRepeat::Continue;
        }
        let consecutive = output.record(signature);
        if consecutive < self.output_threshold {
            return SuccessfulRepeat::Continue;
        }
        SuccessfulRepeat::Halt(format!(
            "Stopping: the last {consecutive} iterations produced the identical response and tool call with no change; the run is stuck repeating the same step without making progress."
        ))
    }

    /// Records the canonical tool-name/arguments signature after the whole
    /// batch completes. Failed or exempt batches reset the successful streak.
    pub fn record_call_batch(
        &self,
        signature: &str,
        all_successful: bool,
        exempt: bool,
    ) -> SuccessfulRepeat {
        let mut calls = self.calls.lock().unwrap();
        if exempt || !all_successful {
            calls.reset();
            drop(calls);
            if !all_successful {
                self.output.lock().unwrap().reset();
            }
            return SuccessfulRepeat::Continue;
        }
        let consecutive = calls.record(signature);
        if consecutive < self.call_threshold {
            return SuccessfulRepeat::Continue;
        }
        SuccessfulRepeat::Halt(format!(
            "Stopping: the same successful tool-call batch was issued {consecutive} times in a row with identical arguments and no new information; the run is stuck repeating one action without making progress."
        ))
    }

    /// Clears both streaks, for example when a paused run is resumed.
    pub fn reset(&self) {
        self.output.lock().unwrap().reset();
        self.calls.lock().unwrap().reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_output_halts_at_threshold_and_changes_reset() {
        let tracker = SuccessfulRepeatTracker::new(3, 3);
        assert_eq!(
            tracker.record_output("same", false),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_output("same", false),
            SuccessfulRepeat::Continue
        );
        assert!(matches!(
            tracker.record_output("same", false),
            SuccessfulRepeat::Halt(message) if message.contains("3 iterations")
        ));
        assert_eq!(
            tracker.record_output("different", false),
            SuccessfulRepeat::Continue
        );
    }

    #[test]
    fn successful_call_batches_halt_but_failures_reset() {
        let tracker = SuccessfulRepeatTracker::new(4, 2);
        assert_eq!(
            tracker.record_call_batch("tool:args", true, false),
            SuccessfulRepeat::Continue
        );
        assert!(matches!(
            tracker.record_call_batch("tool:args", true, false),
            SuccessfulRepeat::Halt(message) if message.contains("2 times")
        ));
        assert_eq!(
            tracker.record_call_batch("tool:args", false, false),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_call_batch("tool:args", true, false),
            SuccessfulRepeat::Continue
        );
    }

    #[test]
    fn failed_call_batches_reset_output_repeats() {
        let tracker = SuccessfulRepeatTracker::new(2, 2);
        assert_eq!(
            tracker.record_output("same", false),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_call_batch("same-call", false, false),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_output("same", false),
            SuccessfulRepeat::Continue,
            "a failed prior batch must not count toward a successful output loop"
        );
    }

    #[test]
    fn exempt_batches_reset_both_streaks() {
        let tracker = SuccessfulRepeatTracker::new(2, 2);
        assert_eq!(
            tracker.record_output("poll", false),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_output("poll", true),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_output("poll", false),
            SuccessfulRepeat::Continue
        );

        assert_eq!(
            tracker.record_call_batch("poll", true, false),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_call_batch("poll", true, true),
            SuccessfulRepeat::Continue
        );
        assert_eq!(
            tracker.record_call_batch("poll", true, false),
            SuccessfulRepeat::Continue
        );
    }

    #[test]
    fn zero_thresholds_are_fail_safe() {
        let tracker = SuccessfulRepeatTracker::new(0, 0);
        assert!(matches!(
            tracker.record_output("same", false),
            SuccessfulRepeat::Halt(_)
        ));
        assert!(matches!(
            tracker.record_call_batch("same", true, false),
            SuccessfulRepeat::Halt(_)
        ));
    }
}
