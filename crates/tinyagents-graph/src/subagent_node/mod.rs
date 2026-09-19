//! Sub-agent nodes — the graph node that delegates to a harness *agent* (a
//! model-driven agent loop) invoked through an explicit host capability.
//!
//! Where [`crate::subgraph`] embeds an entire [`CompiledGraph`](crate::CompiledGraph) as a
//! node, this module embeds a *harness agent* as a node: a graph step hands its
//! work to a host-selected, independently-observable agent and folds the agent's
//! answer back into the parent graph state.
//!
//! The pieces:
//!
//! - [`SubAgentNode`] binds an agent `ComponentId` to an [`InputMapper`]
//!   (parent `State` → [`SubAgentInput`]), an [`OutputMapper`]
//!   ([`SubAgentOutput`] → parent `Update`), and a [`SubAgentPolicy`].
//! - [`subagent_node`] lowers a [`SubAgentNode`] into an ordinary graph node
//!   `Handler`: it obtains the carried [`AgentInvoker`], creates a distinct
//!   child `run_id` that preserves the run tree's `root_run_id` and is parented
//!   to the enclosing graph run, applies timeout/retry/budget policy, maps the
//!   child output into the parent update, records the child run (with its usage)
//!   onto the parent execution rollup, and forwards the child run's harness
//!   events onto the host-provided event sink.
//!
//! See `types` for the data definitions and `test.rs` for focused tests.

mod types;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

pub use types::*;

use crate::builder::NodeContext;
use crate::command::NodeResult;
use crate::recursion::ChildRun;
use crate::{Result, TinyAgentsError};
use tinyagents_harness::ids::next_seq;
use tinyagents_harness::ids::{GraphId, RunId};

type Handler<S, U> = Box<
    dyn Fn(S, NodeContext) -> Pin<Box<dyn Future<Output = Result<NodeResult<U>>> + Send>>
        + Send
        + Sync,
>;

impl<State, Update> SubAgentNode<State, Update> {
    /// Builds a sub-agent node delegating to the registered agent named `agent`,
    /// with the given parent↔child mappers and a default [`SubAgentPolicy`].
    pub fn new(
        agent: impl Into<String>,
        input_mapper: InputMapper<State>,
        output_mapper: OutputMapper<Update>,
    ) -> Self {
        Self {
            agent: agent.into(),
            input_mapper,
            output_mapper,
            policy: SubAgentPolicy::default(),
        }
    }

    /// Builds a sub-agent node from plain closures (a convenience over
    /// [`SubAgentNode::new`] that wraps them as [`InputMapper`]/[`OutputMapper`]).
    pub fn from_fns<I, O>(agent: impl Into<String>, input: I, output: O) -> Self
    where
        I: Fn(&State) -> SubAgentInput + Send + Sync + 'static,
        O: Fn(SubAgentOutput) -> Update + Send + Sync + 'static,
    {
        Self::new(agent, Arc::new(input), Arc::new(output))
    }

    /// Sets the invocation policy, returning `self` for chaining.
    pub fn with_policy(mut self, policy: SubAgentPolicy) -> Self {
        self.policy = policy;
        self
    }
}

/// Lowers a [`SubAgentNode`] plus a host-bound `invoker` into a graph node
/// `Handler`.
///
/// At each activation the handler:
///
/// 1. resolves [`SubAgentNode::agent`] against `registry` (failing with
///    [`TinyAgentsError::Capability`] when the name is not a registered agent),
/// 2. projects the committed `state` into a [`SubAgentInput`],
/// 3. mints a distinct child `run_id` that preserves the run tree's
///    `root_run_id` and is parented to the enclosing graph run,
/// 4. runs the agent under the node's [`SubAgentPolicy`] (timeout/retry), then
///    enforces the work budget,
/// 5. records the child run — with its rolled-up [`UsageTotals`](tinyinference_llm::usage::UsageTotals) — onto the
///    enclosing run's child-run sink, and
/// 6. folds the [`SubAgentOutput`] into a parent `Update` via the output mapper.
pub fn subagent_node<State, Update>(node: SubAgentNode<State, Update>) -> Handler<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    let node = Arc::new(node);
    Box::new(move |state: State, ctx: NodeContext| {
        let node = node.clone();
        Box::pin(async move {
            let input = (node.input_mapper)(&state);
            let binding = ctx.agent_binding.clone().ok_or_else(|| {
                TinyAgentsError::Capability(format!(
                    "sub-agent `{}` requires an execution-scoped AgentInvocationBinding",
                    node.agent
                ))
            })?;
            let output = run_with_policy(&binding, &node.agent, input, &ctx, &node.policy).await?;

            record_child_run(&ctx, &node.agent, &output);

            let update = (node.output_mapper)(output);
            Ok(NodeResult::Update(update))
        })
    })
}

/// Runs `agent` under `policy`: applies the per-attempt timeout, retries
/// transient failures per the retry policy, then enforces the work budget.
async fn run_with_policy(
    binding: &AgentInvocationBinding,
    agent_id: &str,
    input: SubAgentInput,
    ctx: &NodeContext,
    policy: &SubAgentPolicy,
) -> Result<SubAgentOutput> {
    let mut attempt = 0;
    loop {
        let fut = binding.invoker.invoke(AgentInvocation {
            agent_id: agent_id.to_string(),
            input: input.clone(),
            graph_id: ctx.graph_id.clone(),
            node_id: ctx.node_id.clone(),
            parent_run_id: ctx.run_id.clone(),
            root_run_id: ctx
                .root_run_id
                .clone()
                .unwrap_or_else(|| ctx.run_id.clone()),
            events: binding.events.clone(),
            cancellation: Some(binding.cancellation.clone()),
        });
        let result = match policy.timeout {
            Some(timeout) => match tokio::time::timeout(timeout, fut).await {
                Ok(result) => result,
                Err(_) => Err(TinyAgentsError::Timeout(format!(
                    "sub-agent `{}` timed out after {timeout:?}",
                    agent_id
                ))),
            },
            None => fut.await,
        };

        match result {
            Ok(output) => return policy.budget.check(&output, agent_id).map(|()| output),
            Err(err) => {
                if policy.retry.should_retry(attempt)
                    && tinyagents_harness::retry::is_retryable(&err)
                {
                    attempt += 1;
                    let backoff = policy.retry.backoff_for_attempt(attempt);
                    if backoff > Duration::ZERO {
                        tokio::time::sleep(backoff).await;
                    }
                    continue;
                }
                return Err(err);
            }
        }
    }
}

/// Mints a distinct child run id (preserving the run tree root, parented to the
/// enclosing graph run) and records it — with the child's rolled-up usage —
/// onto the enclosing run's child-run sink, when one is attached.
fn record_child_run(ctx: &NodeContext, agent: &str, output: &SubAgentOutput) {
    let Some(sink) = &ctx.child_runs else {
        return;
    };
    let root_run_id = ctx
        .root_run_id
        .clone()
        .unwrap_or_else(|| ctx.run_id.clone());
    sink.record(ChildRun {
        node: ctx.node_id.clone(),
        graph_id: GraphId::new(format!("agent:{agent}")),
        run_id: RunId::new(format!("subagent-{}", next_seq())),
        root_run_id,
        usage: output.usage,
        checkpoint_id: None,
    });
}

#[cfg(test)]
mod test;
