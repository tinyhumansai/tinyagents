//! Successful-repeat progress detection.
//!
//! [`NoProgressTracker`] handles failing tool calls, but deliberately resets on
//! success. That leaves a second loop shape undetected: a model can repeatedly
//! emit the same response and successfully invoke the same no-op tool call.
//! This tracker owns the provider- and product-neutral streak accounting for
//! those loops; a harness middleware remains responsible for building canonical
//! signatures and deciding which polling tools are exempt.

use std::hash::{Hash, Hasher};

use super::types::{Streak, SuccessfulRepeat, SuccessfulRepeatTracker};

/// Consecutive identical assistant-output batches required to halt.
pub const DEFAULT_REPEAT_OUTPUT_THRESHOLD: u32 = 4;
/// Consecutive identical successful tool-call batches required to halt.
pub const DEFAULT_REPEAT_CALL_THRESHOLD: u32 = 3;

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
            output: std::sync::Mutex::new(Streak::default()),
            calls: std::sync::Mutex::new(Streak::default()),
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
            // Output is observed before the completed batch can be classified.
            // An exempt polling batch or a failure therefore resets both
            // trackers so its preceding output cannot leak into the next
            // progress-eligible iteration.
            self.output.lock().unwrap().reset();
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
