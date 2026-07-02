//! Domain types for the thread-level goal.
//!
//! A **thread goal** is a single, thread-scoped "completion contract" — a
//! durable objective a graph keeps pursuing across supersteps, interrupts,
//! resumes, and budget boundaries. There is **exactly one** goal per thread,
//! with a small lifecycle and an optional token budget.
//!
//! The shape is ported from OpenHuman's `thread_goals`, re-hosted on the
//! harness [`Store`](crate::harness::store::Store) (see [`super::store`]) and
//! driven by the graph runtime rather than an out-of-band heartbeat (see
//! [`super::continuation`]).

use serde::{Deserialize, Serialize};

/// Lifecycle state of a thread goal.
///
/// Ownership is **asymmetric**: a model may create/replace a goal and mark it
/// [`Complete`](ThreadGoalStatus::Complete); [`Paused`](ThreadGoalStatus::Paused)
/// / [`BudgetLimited`](ThreadGoalStatus::BudgetLimited) are system-driven
/// (host control and accounting respectively), and clearing deletes the row
/// entirely rather than being a status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Creation time (unix epoch milliseconds).
    pub created_at_ms: u64,
    /// Last-mutation time (unix epoch milliseconds).
    pub updated_at_ms: u64,
    /// Set when a continuation iteration produced **no progress**, to stop a
    /// continuation loop. Cleared on any user-initiated run or external
    /// mutation (e.g. a fresh `goal_set` or [`super::note_user_turn`]).
    #[serde(default)]
    pub continuation_suppressed: bool,
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

/// Usage accounted against a goal for one work iteration.
///
/// The graph runtime is provider-neutral and does not meter tokens per node, so
/// accounting is **explicit** at the work-node boundary: a work node records
/// what it spent (and whether it advanced the objective) and the continuation
/// gate folds it in. `made_progress` is the graph analogue of OpenHuman's
/// "the turn produced tool calls" — the caller defines it (e.g. the iteration
/// produced tool calls or a non-empty state delta).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalProgress {
    /// Tokens spent by the work iteration that just finished.
    pub tokens_used: u64,
    /// Wall-clock seconds spent by the work iteration that just finished.
    pub elapsed_secs: u64,
    /// Whether the iteration advanced the objective. A `false` here stops the
    /// self-driving loop (one-shot suppression) so it cannot spin uselessly.
    pub made_progress: bool,
}

/// Outcome of running one continuation turn through the external-scheduler
/// driver ([`super::run_continuation_tick`]). Identical in shape to
/// [`GoalProgress`]; named distinctly at the driver boundary for clarity.
pub type TurnOutcome = GoalProgress;

/// Renders the per-iteration context block a caller can prepend to a work
/// node's prompt so the model knows the active objective and how to close it
/// out. Returns `None` for statuses that should not drive further work
/// ([`Paused`](ThreadGoalStatus::Paused) / [`Complete`](ThreadGoalStatus::Complete)).
pub fn active_goal_context_block(goal: &ThreadGoal) -> Option<String> {
    match goal.status {
        ThreadGoalStatus::Active => {
            let budget = match goal.budget_remaining() {
                Some(remaining) => format!(" (~{remaining} tokens of budget remain)"),
                None => String::new(),
            };
            Some(format!(
                "[thread goal] You are working toward this thread's durable goal{budget}.\n\n\
                 Goal: {objective}\n\n\
                 Assess progress against concrete evidence, then take the next useful step. \
                 If the goal is already satisfied, call `goal_complete` now.",
                objective = goal.objective,
            ))
        }
        ThreadGoalStatus::BudgetLimited => Some(format!(
            "[thread goal] The token budget for this goal is exhausted. Summarise progress \
             and stop; do not start new substantive work.\n\nGoal: {}",
            goal.objective,
        )),
        ThreadGoalStatus::Paused | ThreadGoalStatus::Complete => None,
    }
}
