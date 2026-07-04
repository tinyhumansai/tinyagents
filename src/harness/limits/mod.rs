//! Run-scoped limit enforcement.
//!
//! Limits are what keep recursion bounded: because agents can call agents and
//! graphs can run graphs, an unbounded run tree could fan out forever or burn a
//! provider budget. [`RunLimits::max_depth`] caps how deep the sub-agent /
//! sub-graph recursion may go, while the call and wall-clock caps bound the
//! work within each run.
//!
//! [`RunLimits`] holds the policy; [`LimitTracker`] tracks live counters and
//! checks them against the policy.  Every model call and tool call must go
//! through the tracker so limits are fail-closed.
//!
//! # Example
//!
//! ```
//! use tinyagents::harness::limits::{RunLimits, LimitTracker};
//!
//! let limits = RunLimits::default();
//! let mut tracker = LimitTracker::new(limits);
//! tracker.record_model_call().expect("within limit");
//! assert_eq!(tracker.model_calls(), 1);
//! assert_eq!(tracker.remaining_model_calls(), 24);
//! ```

mod types;

pub use types::*;

use std::time::{Duration, Instant};

use crate::error::{Result, TinyAgentsError};

impl RunLimits {
    /// Sets the maximum number of model calls allowed per run.
    pub fn with_max_model_calls(mut self, n: usize) -> Self {
        self.max_model_calls = n;
        self
    }

    /// Sets the maximum number of tool calls allowed per run.
    pub fn with_max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = n;
        self
    }

    /// Sets a wall-clock deadline in milliseconds. `None` removes the limit.
    pub fn with_max_wall_clock_ms(mut self, ms: Option<u64>) -> Self {
        self.max_wall_clock_ms = ms;
        self
    }

    /// Sets the per-call retry cap (a retry *count*, not counting the first
    /// attempt). See [`RunLimits::max_retries_per_call`].
    pub fn with_max_retries_per_call(mut self, n: usize) -> Self {
        self.max_retries_per_call = n;
        self
    }

    /// Sets the maximum sub-agent / recursion depth for the run tree.
    pub fn with_max_depth(mut self, n: usize) -> Self {
        self.max_depth = n;
        self
    }
}

/// Tracks live counters for a single harness run and enforces [`RunLimits`].
///
/// The tracker records the wall-clock start time when it is created and
/// computes elapsed time on demand via [`check_wall_clock`].
///
/// [`check_wall_clock`]: LimitTracker::check_wall_clock
pub struct LimitTracker {
    limits: RunLimits,
    model_calls: usize,
    tool_calls: usize,
    started_at: Instant,
}

impl LimitTracker {
    /// Creates a new tracker with zeroed counters and the current time as the
    /// run start.
    pub fn new(limits: RunLimits) -> Self {
        Self {
            limits,
            model_calls: 0,
            tool_calls: 0,
            started_at: Instant::now(),
        }
    }

    /// Records one model call and returns an error if the cap is exceeded.
    ///
    /// The counter is incremented **before** the check so the limit is
    /// inclusive (a cap of `N` allows exactly `N` calls).
    pub fn record_model_call(&mut self) -> Result<()> {
        self.model_calls += 1;
        if self.model_calls > self.limits.max_model_calls {
            return Err(TinyAgentsError::Validation(format!(
                "max model calls ({}) exceeded",
                self.limits.max_model_calls
            )));
        }
        Ok(())
    }

    /// Records one tool call and returns an error if the cap is exceeded.
    pub fn record_tool_call(&mut self) -> Result<()> {
        self.tool_calls += 1;
        if self.tool_calls > self.limits.max_tool_calls {
            return Err(TinyAgentsError::Validation(format!(
                "max tool calls ({}) exceeded",
                self.limits.max_tool_calls
            )));
        }
        Ok(())
    }

    /// Checks whether the run has exceeded the configured wall-clock deadline.
    ///
    /// Returns `Ok(())` when no deadline is configured or the deadline has not
    /// been reached. Returns a [`Validation`][crate::error::TinyAgentsError::Validation]
    /// error otherwise.
    pub fn check_wall_clock(&self) -> Result<()> {
        if let Some(max_ms) = self.limits.max_wall_clock_ms {
            let elapsed_ms = self.started_at.elapsed().as_millis() as u64;
            if elapsed_ms > max_ms {
                return Err(TinyAgentsError::Validation(format!(
                    "wall-clock limit ({max_ms} ms) exceeded after {elapsed_ms} ms"
                )));
            }
        }
        Ok(())
    }

    /// Returns the wall-clock time elapsed since this tracker was created.
    ///
    /// Exposed so callers (the agent loop) can compute a remaining budget
    /// against a deadline sourced from somewhere other than the run config —
    /// for example the harness-level [`RunLimits::max_wall_clock_ms`].
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns the wall-clock budget still remaining before the configured
    /// deadline, measured from this tracker's start instant.
    ///
    /// Returns `None` when no wall-clock deadline is configured (so callers
    /// should not bound work by time). When a deadline is configured the
    /// returned [`Duration`] is the remaining budget, saturating at
    /// [`Duration::ZERO`] once the deadline has already elapsed.
    ///
    /// This is the budget the agent loop uses to bound an individual model call
    /// (via `tokio::time::timeout`) so a hung or slow provider call is
    /// interrupted rather than only being detected by the between-call
    /// [`check_wall_clock`] check.
    ///
    /// [`check_wall_clock`]: LimitTracker::check_wall_clock
    pub fn remaining_wall_clock(&self) -> Option<Duration> {
        self.limits.max_wall_clock_ms.map(|max_ms| {
            let max = Duration::from_millis(max_ms);
            max.checked_sub(self.started_at.elapsed())
                .unwrap_or(Duration::ZERO)
        })
    }

    /// Returns the number of model calls recorded so far.
    pub fn model_calls(&self) -> usize {
        self.model_calls
    }

    /// Returns the number of tool calls recorded so far.
    pub fn tool_calls(&self) -> usize {
        self.tool_calls
    }

    /// Returns the number of model calls remaining before the cap is hit.
    ///
    /// Returns `0` rather than wrapping if the counter has somehow already
    /// exceeded the limit.
    pub fn remaining_model_calls(&self) -> usize {
        self.limits.max_model_calls.saturating_sub(self.model_calls)
    }

    /// Returns a reference to the active [`RunLimits`] policy.
    pub fn limits(&self) -> &RunLimits {
        &self.limits
    }

    /// Overrides the model-call and tool-call caps in place, preserving
    /// already-recorded counts and the wall-clock start time.
    ///
    /// A `RunContext` derives its tracker's initial limits from its
    /// `RunConfig`, which always carries a concrete default. That can
    /// silently disagree with a harness-wide `RunPolicy` configured with a
    /// different cap, so the *reported* limit (the policy's) and the limit
    /// that actually trips (the tracker's) diverge. The agent loop calls this
    /// once per run to reconcile the two into a single enforced source of
    /// truth before the loop begins.
    pub fn sync_call_limits(&mut self, max_model_calls: usize, max_tool_calls: usize) {
        self.limits.max_model_calls = max_model_calls;
        self.limits.max_tool_calls = max_tool_calls;
    }
}

#[cfg(test)]
mod test;
