//! One timeout/retry/budget policy for every subagent path.
//!
//! [`SubAgentPolicy`] and [`SubAgentBudget`] are defined beside the graph
//! `SubAgentNode` (`tinyagents-graph`, which this crate depends on, so the
//! types cannot live here without a cycle) and re-exported so a host names
//! them from either place. [`SubagentDriver`](super::SubagentDriver) and
//! [`SubAgentTool`](super::SubAgentTool) apply them like the graph node does:
//!
//! - **timeout** cancels the child and ends the run `Incomplete(Timeout)`;
//!   a timed-out attempt is never retried (it may have run for the whole
//!   window and run tools).
//! - **retry** applies only to a failure the retry policy deems retryable
//!   and never after the attempt ran tools, unless
//!   [`SubAgentPolicy::retry_after_tool_calls`] is set.
//! - **budget** call caps tighten the child's `RunConfig` (enforced during the
//!   run); token caps are checked against the reported usage afterwards;
//!   `max_cost` is carried but not enforced (see [`SubAgentBudget`]).

pub use tinyagents_graph::{SubAgentBudget, SubAgentPolicy};

use tinyagents_harness::CancellationToken;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::message::Message;

use super::{
    AttemptContextFactory, IncompleteKind, PreparedSubagent, ResultPolicy, SubagentError,
    SubagentIncomplete, SubagentOutcome, SubagentOutcomeKind, SubagentRole,
};

/// Whether a failed attempt `attempt` (0-based) may be retried.
pub(crate) fn may_retry(
    policy: &SubAgentPolicy,
    attempt: usize,
    error: &TinyAgentsError,
    tools_ran: bool,
) -> bool {
    !error.is_terminal_limit()
        && policy.retry.should_retry_error(attempt, error)
        && (!tools_ran || policy.retry_after_tool_calls)
}

/// Everything about a plan except its (consumed) run context, kept so a
/// retry can rebuild an identical plan around a freshly minted context.
pub(crate) struct AttemptSource<C> {
    template: Option<Template<C>>,
}

struct Template<C> {
    task_id: String,
    agent_key: String,
    input: Vec<Message>,
    tools: ToolSnapshot,
    role: SubagentRole,
    tool_ceiling: Option<ToolSnapshot>,
    delegation_tools: Vec<String>,
    policy: SubAgentPolicy,
    result_policy: ResultPolicy,
    factory: AttemptContextFactory<C>,
}

impl<C> AttemptSource<C> {
    /// Captures the plan, but only when retrying is possible at all.
    pub(crate) fn from_prepared(prepared: &PreparedSubagent<C>) -> Self {
        let template = (prepared.policy.retry.max_attempts > 1)
            .then(|| prepared.retry_context.clone())
            .flatten()
            .map(|factory| Template {
                task_id: prepared.task_id.clone(),
                agent_key: prepared.agent_key.clone(),
                input: prepared.input.clone(),
                tools: prepared.tools.clone(),
                role: prepared.role,
                tool_ceiling: prepared.tool_ceiling.clone(),
                delegation_tools: prepared.delegation_tools.clone(),
                policy: prepared.policy.clone(),
                result_policy: prepared.result_policy.clone(),
                factory,
            });
        Self { template }
    }

    /// Whether a retry can be built (a context factory was supplied).
    pub(crate) fn can_retry(&self) -> bool {
        self.template.is_some()
    }

    /// The plan for retry attempt `attempt` (1 is the first retry), around a
    /// fresh context bound to `cancellation`.
    pub(crate) fn next(
        &mut self,
        attempt: u32,
        cancellation: CancellationToken,
    ) -> Result<PreparedSubagent<C>, SubagentError> {
        let template = self
            .template
            .as_ref()
            .ok_or(SubagentError::MissingCapability("retry context factory"))?;
        let mut context: RunContext<C> = (template.factory)(attempt)?;
        template.policy.budget.apply_call_caps(&mut context.config);
        let mut prepared = PreparedSubagent::new(
            template.task_id.clone(),
            template.agent_key.clone(),
            template.input.clone(),
            template.tools.clone(),
            context.with_cancellation(cancellation),
        )
        .with_role(template.role)
        .with_delegation_tools(template.delegation_tools.clone())
        .with_policy(template.policy.clone())
        .with_result_policy(template.result_policy.clone())
        .with_retry_context(template.factory.clone());
        prepared.tool_ceiling = template.tool_ceiling.clone();
        Ok(prepared)
    }
}

/// Applies the budget and result policies to a finished outcome.
///
/// A completed run over its token budget becomes `Incomplete(BudgetExceeded)`
/// (its output and usage are kept); an over-budget or completed run then has
/// its output trimmed and schema-checked per `result_policy`.
pub(crate) async fn apply_outcome_policies(
    mut outcome: SubagentOutcome,
    policy: &SubAgentPolicy,
    result_policy: &ResultPolicy,
) -> (SubagentOutcome, usize) {
    let mut omitted_chars = 0;
    if !matches!(outcome.status, SubagentOutcomeKind::Completed) {
        return (outcome, omitted_chars);
    }
    let measured = tinyagents_graph::SubAgentOutput {
        usage: outcome.usage,
        ..Default::default()
    };
    if let Err(error) = policy.budget.check(&measured, &outcome.task_id) {
        outcome.status = SubagentOutcomeKind::Incomplete(
            SubagentIncomplete::new(error.to_string()).with_kind(IncompleteKind::BudgetExceeded),
        );
    }
    if result_policy.is_active() {
        let applied = result_policy
            .apply(&outcome.task_id, &outcome.output, None)
            .await;
        omitted_chars = applied.omitted_chars;
        outcome.output = applied.text;
        outcome.schema_error = applied.schema_error;
        outcome.artifact_error = applied.artifact_error;
        outcome.artifacts.extend(applied.artifact);
    }
    (outcome, omitted_chars)
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
