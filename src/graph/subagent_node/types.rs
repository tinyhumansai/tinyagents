//! Type definitions for [`SubAgentNode`](super::SubAgentNode) — the graph node
//! that delegates to a harness agent resolved by name from a
//! [`CapabilityRegistry`](crate::registry::CapabilityRegistry).
//!
//! See the module [`mod`](super) docs for how these are wired into a node
//! handler. This file holds the plain data: the agent trait the registry
//! resolves to ([`HarnessAgent`]), the structured input/output carriers
//! ([`SubAgentInput`]/[`SubAgentOutput`]), the per-call policy
//! ([`SubAgentPolicy`]/[`SubAgentBudget`]), the parent↔child mapping aliases
//! ([`InputMapper`]/[`OutputMapper`]), and the [`SubAgentNode`] descriptor.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::Result;
use crate::harness::events::EventSink;
use crate::harness::retry::RetryPolicy;
use crate::harness::usage::UsageTotals;
use crate::registry::ComponentId;

/// A harness agent invocable by name from a graph node.
///
/// This is the object-safe surface a [`CapabilityRegistry`](crate::registry::CapabilityRegistry)
/// resolves a registered agent to. It is intentionally decoupled from the
/// parent graph's `State`: a [`SubAgentNode`] projects parent state into a
/// [`SubAgentInput`] (typically a prompt) before calling, and folds the
/// [`SubAgentOutput`] back into a parent update afterwards, so the agent itself
/// never sees the graph state shape.
///
/// The canonical implementor is [`HarnessSubAgent`](super::HarnessSubAgent),
/// which adapts a harness [`SubAgent`](crate::harness::subagent::SubAgent).
#[async_trait]
pub trait HarnessAgent: Send + Sync {
    /// The stable registered name of the agent.
    fn name(&self) -> &str;

    /// Runs the agent as a child run over `input`, fanning the child run's
    /// harness events onto `events` so the parent observer sees the nested run.
    async fn run(&self, input: SubAgentInput, events: EventSink) -> Result<SubAgentOutput>;
}

/// The structured input a [`SubAgentNode`] hands to a delegated agent.
///
/// `prompt` is the user-facing text the child run is seeded with; `data` is an
/// optional structured payload an agent (or an adapter) may consult.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubAgentInput {
    /// The user prompt the child run is seeded with.
    pub prompt: String,
    /// Optional structured side-channel payload.
    pub data: Option<Value>,
}

impl SubAgentInput {
    /// Builds an input carrying just a `prompt`.
    pub fn prompt(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            data: None,
        }
    }

    /// Attaches a structured `data` payload, returning `self` for chaining.
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// The structured output a delegated agent returns to its [`SubAgentNode`].
///
/// Besides the final `text` and any parsed `structured` value, the output
/// carries the child run's [`UsageTotals`] and call counts so the node can roll
/// the child's usage into the parent execution and a budget can be enforced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubAgentOutput {
    /// The child run's final assistant text.
    pub text: String,
    /// Parsed structured output, when the child run produced one.
    pub structured: Option<Value>,
    /// Cumulative token usage across the child run's model calls.
    pub usage: UsageTotals,
    /// Number of model calls the child run dispatched.
    pub model_calls: usize,
    /// Number of tool invocations the child run executed.
    pub tool_calls: usize,
}

/// An optional cap on the work a single sub-agent invocation may perform.
///
/// Enforced *after* the child run returns: a run whose reported model/tool call
/// counts exceed the configured cap fails the node with
/// [`TinyAgentsError::LimitExceeded`](crate::TinyAgentsError::LimitExceeded).
/// The default ([`SubAgentBudget::unlimited`]) imposes no cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubAgentBudget {
    /// Maximum model calls the child run may make (`None` = unbounded).
    pub max_model_calls: Option<usize>,
    /// Maximum tool calls the child run may make (`None` = unbounded).
    pub max_tool_calls: Option<usize>,
}

impl SubAgentBudget {
    /// A budget that imposes no cap.
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Returns `Ok(())` when `output` is within budget, else
    /// [`TinyAgentsError::LimitExceeded`](crate::TinyAgentsError::LimitExceeded).
    pub fn check(&self, output: &SubAgentOutput, agent: &str) -> Result<()> {
        if let Some(max) = self.max_model_calls
            && output.model_calls > max
        {
            return Err(crate::TinyAgentsError::LimitExceeded(format!(
                "sub-agent `{agent}` exceeded model-call budget: {} > {max}",
                output.model_calls
            )));
        }
        if let Some(max) = self.max_tool_calls
            && output.tool_calls > max
        {
            return Err(crate::TinyAgentsError::LimitExceeded(format!(
                "sub-agent `{agent}` exceeded tool-call budget: {} > {max}",
                output.tool_calls
            )));
        }
        Ok(())
    }
}

/// Timeout, retry, and budget policy applied around a sub-agent invocation.
///
/// This is a thin graph-local struct that defers to the harness
/// [`RetryPolicy`] for retry/backoff and reuses the harness usage accounting for
/// budgeting, so a graph node and an in-harness sub-agent share the same
/// resilience semantics. The default is a *single attempt, no timeout, no
/// budget* — deliberately conservative so a node never silently re-runs a
/// non-idempotent agent.
#[derive(Clone, Debug)]
pub struct SubAgentPolicy {
    /// Optional wall-clock timeout for a single attempt. On elapse the node
    /// fails with [`TinyAgentsError::Timeout`](crate::TinyAgentsError::Timeout).
    pub timeout: Option<Duration>,
    /// Retry/backoff policy applied across attempts.
    pub retry: RetryPolicy,
    /// Optional cap on the work the child run may perform.
    pub budget: SubAgentBudget,
}

impl Default for SubAgentPolicy {
    fn default() -> Self {
        Self {
            timeout: None,
            retry: RetryPolicy::default().with_max_attempts(1),
            budget: SubAgentBudget::unlimited(),
        }
    }
}

impl SubAgentPolicy {
    /// Sets the per-attempt timeout, returning `self` for chaining.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the retry policy, returning `self` for chaining.
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Sets the work budget, returning `self` for chaining.
    pub fn with_budget(mut self, budget: SubAgentBudget) -> Self {
        self.budget = budget;
        self
    }
}

/// Maps parent graph `State` into the [`SubAgentInput`] a delegated agent runs
/// over.
pub type InputMapper<State> = Arc<dyn Fn(&State) -> SubAgentInput + Send + Sync>;

/// Maps a delegated agent's [`SubAgentOutput`] into the parent graph `Update`
/// merged through the graph reducer.
pub type OutputMapper<Update> = Arc<dyn Fn(SubAgentOutput) -> Update + Send + Sync>;

/// A graph node that delegates to a harness agent resolved by name.
///
/// A `SubAgentNode` binds an agent [`ComponentId`] (resolved against a
/// [`CapabilityRegistry`](crate::registry::CapabilityRegistry) at run time) to a
/// pair of mappers and a [`SubAgentPolicy`]. Lower it into a graph node handler
/// with [`subagent_node`](super::subagent_node).
pub struct SubAgentNode<State, Update> {
    /// The registered agent name to resolve and delegate to.
    pub agent: ComponentId,
    /// Projects parent state into the child input.
    pub input_mapper: InputMapper<State>,
    /// Folds the child output back into a parent update.
    pub output_mapper: OutputMapper<Update>,
    /// Timeout/retry/budget policy applied around the invocation.
    pub policy: SubAgentPolicy,
    /// Optional sink the child run's harness events are forwarded onto.
    pub(crate) events: Option<EventSink>,
}

impl<State, Update> Clone for SubAgentNode<State, Update> {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            input_mapper: self.input_mapper.clone(),
            output_mapper: self.output_mapper.clone(),
            policy: self.policy.clone(),
            events: self.events.clone(),
        }
    }
}
