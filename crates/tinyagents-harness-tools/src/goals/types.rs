//! Domain types for the thread-level goal.
//!
//! A **thread goal** is a single, thread-scoped "completion contract" — a
//! durable objective a graph keeps pursuing across supersteps, interrupts,
//! resumes, and budget boundaries. There is **exactly one** goal per thread,
//! with a small lifecycle and an optional token budget.
//!
//! The shape is ported from OpenHuman's `thread_goals`, re-hosted on the
//! harness [`Store`](tinyagents_harness::store::Store) (see [`super::store`]) and
//! driven by the host's turn lifecycle.

use serde::{Deserialize, Serialize};

/// Lifecycle state of a thread goal.
///
/// Ownership is **asymmetric**: a model may create/replace a goal, mark it
/// [`Complete`](ThreadGoalStatus::Complete) and resume a paused one;
/// [`Paused`](ThreadGoalStatus::Paused) /
/// [`BudgetLimited`](ThreadGoalStatus::BudgetLimited) are entered only by the
/// system (host control and accounting respectively), and clearing deletes the
/// row entirely rather than being a status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ThreadGoalStatus {
    /// The graph may make progress and (when driven) auto-continue.
    Active,
    /// Work is suspended (host/user control); the objective persists and is
    /// reactivated on resume.
    Paused,
    /// The token budget has been reached; substantive work halts until the
    /// budget is raised or the goal is cleared.
    BudgetLimited,
    /// Evidence confirms the objective is satisfied.
    Complete,
}

impl ThreadGoalStatus {
    /// The stable lower-snake-case status label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::BudgetLimited => "budget_limited",
            Self::Complete => "complete",
        }
    }

    /// Whether the goal is in a state where the graph should keep working it
    /// (and a continuation loop may fire).
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Whether the goal is in a terminal state for continuation purposes —
    /// [`Complete`](Self::Complete) or [`BudgetLimited`](Self::BudgetLimited)
    /// never auto-continue.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Complete | Self::BudgetLimited)
    }
}

/// A single thread-scoped goal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoal {
    /// The thread this goal belongs to (one goal per thread).
    pub thread_id: String,
    /// Version identifier, re-minted on **every objective replacement**. Stale
    /// accounting writes that pass a non-matching `expected_goal_id` are
    /// silently ignored — see [`super::store::account_usage`].
    pub goal_id: String,
    /// The durable objective, one or more sentences.
    pub objective: String,
    /// Lifecycle state.
    pub status: ThreadGoalStatus,
    /// Optional token ceiling. When set and `tokens_used >= token_budget`, the
    /// goal transitions to [`ThreadGoalStatus::BudgetLimited`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_budget: Option<u64>,
    /// Cumulative tokens accounted against this goal.
    #[serde(default)]
    pub tokens_used: u64,
    /// Cumulative wall-clock seconds accounted against this goal.
    #[serde(default)]
    pub time_used_seconds: u64,
    /// Nanosecond remainder beyond `time_used_seconds`.
    #[serde(default)]
    pub time_used_subsec_nanos: u32,
    /// Creation time (unix epoch milliseconds).
    pub created_at_ms: u64,
    /// Last-mutation time (unix epoch milliseconds).
    pub updated_at_ms: u64,
}

impl ThreadGoal {
    /// Tokens remaining before the budget cap, if a budget is set.
    pub fn budget_remaining(&self) -> Option<u64> {
        self.token_budget
            .map(|b| b.saturating_sub(self.tokens_used))
    }

    /// Whether accounting has reached or exceeded the configured budget.
    pub fn over_budget(&self) -> bool {
        matches!(self.token_budget, Some(b) if self.tokens_used >= b)
    }
}
