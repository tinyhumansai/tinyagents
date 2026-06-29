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
    /// Maximum number of retry attempts per individual call.
    pub max_retries_per_call: usize,
    /// Maximum number of concurrent in-flight calls. `None` means no limit.
    pub max_concurrency: Option<usize>,
    /// Maximum sub-agent / recursion depth allowed for the run tree rooted at
    /// this run. A top-level run is depth `0`; each nested child run increments
    /// the depth. A sub-agent invocation whose child depth would exceed this cap
    /// fails fast (see [`crate::harness::subagent`]). Defaults to
    /// [`RunLimits::DEFAULT_MAX_DEPTH`].
    pub max_depth: usize,
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
            max_concurrency: None,
            max_depth: Self::DEFAULT_MAX_DEPTH,
        }
    }
}
