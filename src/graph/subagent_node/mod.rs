//! Sub-agent nodes — the graph node that delegates to a harness *agent* (a
//! model-driven agent loop) resolved by name from a
//! [`CapabilityRegistry`](crate::registry::CapabilityRegistry).
//!
//! Where [`crate::graph::subgraph`] embeds an entire [`CompiledGraph`] as a
//! node, this module embeds a *harness agent* as a node: a graph step hands its
//! work to a registered, independently-observable agent and folds the agent's
//! answer back into the parent graph state.
//!
//! The pieces:
//!
//! - [`SubAgentNode`] binds an agent [`ComponentId`] to an [`InputMapper`]
//!   (parent `State` → [`SubAgentInput`]), an [`OutputMapper`]
//!   ([`SubAgentOutput`] → parent `Update`), and a [`SubAgentPolicy`].
//! - [`subagent_node`] lowers a [`SubAgentNode`] + a registry into an ordinary
//!   graph node [`Handler`]: it resolves the agent by name, creates a distinct
//!   child `run_id` that preserves the run tree's `root_run_id` and is parented
//!   to the enclosing graph run, applies timeout/retry/budget policy, maps the
//!   child output into the parent update, records the child run (with its usage)
//!   onto the parent execution rollup, and forwards the child run's harness
//!   events onto the node's event sink.
//! - [`HarnessSubAgent`] adapts a harness
//!   [`SubAgent`](crate::harness::subagent::SubAgent) into a registry-storable
//!   [`HarnessAgent`].
//!
//! See [`types`] for the data definitions and `test.rs` for focused tests.

mod types;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

pub use types::*;

use async_trait::async_trait;

use crate::graph::builder::NodeContext;
use crate::graph::command::NodeResult;
use crate::graph::recursion::ChildRun;
use crate::harness::events::EventSink;
use crate::harness::ids::next_seq;
use crate::harness::ids::{GraphId, RunId};
use crate::harness::subagent::SubAgent;
use crate::registry::{CapabilityRegistry, ComponentId};
use crate::{Result, TinyAgentsError};

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
            agent: ComponentId::new(agent),
            input_mapper,
            output_mapper,
            policy: SubAgentPolicy::default(),
            events: None,
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

    /// Forwards the child run's harness events onto `events`, returning `self`
    /// for chaining. This is how a parent observer (or a testkit
    /// [`EventRecorder`](crate::harness::testkit::EventRecorder)) sees the nested
    /// run's lifecycle.
    pub fn with_events(mut self, events: EventSink) -> Self {
        self.events = Some(events);
        self
    }
}

/// Lowers a [`SubAgentNode`] plus a capability `registry` into a graph node
/// [`Handler`].
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
/// 5. records the child run — with its rolled-up [`UsageTotals`] — onto the
///    enclosing run's child-run sink, and
/// 6. folds the [`SubAgentOutput`] into a parent `Update` via the output mapper.
pub fn subagent_node<State, Update, RState>(
    node: SubAgentNode<State, Update>,
    registry: Arc<CapabilityRegistry<RState>>,
) -> Handler<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
    RState: Send + Sync + 'static,
{
    let node = Arc::new(node);
    Box::new(move |state: State, ctx: NodeContext| {
        let node = node.clone();
        let registry = registry.clone();
        Box::pin(async move {
            let agent = registry.agent(node.agent.as_str()).ok_or_else(|| {
                TinyAgentsError::Capability(format!(
                    "sub-agent `{}` is not a registered agent",
                    node.agent.as_str()
                ))
            })?;

            let input = (node.input_mapper)(&state);
            let events = node.events.clone().unwrap_or_default();

            let output = run_with_policy(&agent, input, events, &node.policy).await?;

            record_child_run(&ctx, agent.name(), &output);

            let update = (node.output_mapper)(output);
            Ok(NodeResult::Update(update))
        })
    })
}

/// Runs `agent` under `policy`: applies the per-attempt timeout, retries
/// transient failures per the retry policy, then enforces the work budget.
async fn run_with_policy(
    agent: &Arc<dyn HarnessAgent>,
    input: SubAgentInput,
    events: EventSink,
    policy: &SubAgentPolicy,
) -> Result<SubAgentOutput> {
    let mut attempt = 0;
    loop {
        let fut = agent.run(input.clone(), events.clone());
        let result = match policy.timeout {
            Some(timeout) => match tokio::time::timeout(timeout, fut).await {
                Ok(result) => result,
                Err(_) => Err(TinyAgentsError::Timeout(format!(
                    "sub-agent `{}` timed out after {timeout:?}",
                    agent.name()
                ))),
            },
            None => fut.await,
        };

        match result {
            Ok(output) => return policy.budget.check(&output, agent.name()).map(|()| output),
            Err(err) => {
                if policy.retry.should_retry(attempt) && crate::harness::retry::is_retryable(&err) {
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
    });
}

/// Adapts a harness [`SubAgent`] into a registry-storable [`HarnessAgent`].
///
/// The child run is created with a fresh `State::default()` and `Ctx::default()`
/// — a sub-agent delegated from a graph node receives its task through the
/// mapped [`SubAgentInput`] prompt, not through harness state. Use
/// [`HarnessSubAgent::with_parent_depth`] to express deeper nesting (the child
/// runs at `parent_depth + 1`).
pub struct HarnessSubAgent<S = (), C = ()>
where
    S: Send + Sync,
    C: Send + Sync,
{
    inner: Arc<SubAgent<S, C>>,
    parent_depth: usize,
    state: S,
}

impl<S, C> HarnessSubAgent<S, C>
where
    S: Send + Sync + Default,
    C: Send + Sync + Default,
{
    /// Wraps `inner` as a registry-storable agent invoked at `parent_depth = 0`.
    pub fn new(inner: Arc<SubAgent<S, C>>) -> Self {
        Self {
            inner,
            parent_depth: 0,
            state: S::default(),
        }
    }

    /// Sets the caller depth the child runs at; the child runs at
    /// `parent_depth + 1`. Returns `self` for chaining.
    pub fn with_parent_depth(mut self, parent_depth: usize) -> Self {
        self.parent_depth = parent_depth;
        self
    }

    /// Wraps `inner` in an [`Arc`] as a `dyn HarnessAgent` ready to register.
    pub fn into_dyn(self) -> Arc<dyn HarnessAgent>
    where
        S: 'static,
        C: 'static,
    {
        Arc::new(self)
    }
}

#[async_trait]
impl<S, C> HarnessAgent for HarnessSubAgent<S, C>
where
    S: Send + Sync + Default + 'static,
    C: Send + Sync + Default + 'static,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn run(&self, input: SubAgentInput, events: EventSink) -> Result<SubAgentOutput> {
        let run = self
            .inner
            .invoke_with_events(
                &self.state,
                C::default(),
                self.parent_depth,
                input.prompt,
                &events,
            )
            .await?;
        Ok(SubAgentOutput {
            text: run.text().unwrap_or_default(),
            structured: run.structured.clone(),
            usage: run.usage,
            model_calls: run.model_calls,
            tool_calls: run.tool_calls,
        })
    }
}

#[cfg(test)]
mod test;
