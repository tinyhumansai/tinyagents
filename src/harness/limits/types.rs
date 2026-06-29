//! Types for run-scoped limit enforcement.
//!
//! [`RunLimits`] carries the configured policy; [`LimitTracker`] holds the live
//! counters and checks them against the policy.

/// Configures the hard limits applied across a single harness run.
///
/// All limits are checked fail-closed: the first call that exceeds a cap
/// returns an error and the run should be stopped.
///
/// # Examples
///
/// ```
/// use rustagents::harness::limits::RunLimits;
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
    /// Maximum number of retry attempts per individual call.
    pub max_retries_per_call: usize,
    /// Maximum number of concurrent in-flight calls. `None` means no limit.
    pub max_concurrency: Option<usize>,
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_model_calls: 25,
            max_tool_calls: 50,
            max_wall_clock_ms: None,
            max_retries_per_call: 3,
            max_concurrency: None,
        }
    }
}
