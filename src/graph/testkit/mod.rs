//! Graph-test building blocks — deterministic node doubles, an event recorder, a
//! stream projector, and a fluent run-assertion builder for the durable graph
//! runtime.
//!
//! This is the *graph-level* counterpart to the harness
//! [`testkit`](crate::harness::testkit) (model/tool doubles + trajectories):
//! here the units under test are graph nodes and supersteps, so the doubles are
//! node handlers and the assertions read a run's export/event/checkpoint truth.
//! Because a node can recurse into a [subgraph](crate::graph::subgraph) or a
//! [sub-agent](crate::graph::subagent_node), the same recorder that observes a
//! top-level run also captures the events and child-run rollups of the nested
//! runs it spawns, so recursion stays observable in tests.
//!
//! # Node doubles
//!
//! Each returns a closure ready for
//! [`GraphBuilder::add_node`](crate::graph::GraphBuilder::add_node):
//!
//! | Helper | Behavior |
//! |--------|----------|
//! | [`noop_node`] | Routes onward with no state update |
//! | [`scripted_update_node`] | Emits queued updates (saturating the last) |
//! | [`scripted_route_node`] | Emits queued `goto` route-sets |
//! | [`fanout_node`] | Emits one [`Send`](crate::graph::Send) per arg (fanout) |
//! | [`failing_node`] | Always returns an error |
//! | [`RetryCountingNode`] | Counts activations, fails the first N |
//! | [`interrupting_node`] | Interrupts until resumed, then updates |
//! | [`subgraph_test_node`] | Embeds a child graph (shared state) |
//! | [`subagent_fake_node`] | Records a child run + updates (fake sub-agent) |
//!
//! # Observation & assertions
//!
//! [`GraphEventRecorder`] captures the [`GraphEvent`] stream; [`StreamCollector`]
//! projects it; [`run_recorded`] runs a graph with the recorder wired and
//! bundles the result, recorded events, and checkpoint history into a
//! [`GraphRun`]; [`assert_graph`] opens a fluent [`GraphAssertions`] over it. The
//! single [`GraphRun`] is the test truth — execution, events, and checkpoints
//! all read from it.

mod types;

pub use types::{
    GraphAssertions, GraphEventRecorder, GraphRun, RetryCountingNode, StreamCollector,
};

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use crate::graph::builder::{NodeContext, NodeFuture};
use crate::graph::command::{Command, Interrupt, NodeResult, Send as SendPacket};
use crate::graph::compiled::{CompiledGraph, GraphExecution, StateSnapshot};
use crate::graph::recursion::ChildRun;
use crate::graph::stream::{CollectingSink, GraphEvent, GraphEventSink};
use crate::harness::ids::{GraphId, NodeId, RunId};
use crate::harness::usage::UsageTotals;
use crate::{Result, TinyAgentsError};

// ---------------------------------------------------------------------------
// Node doubles
// ---------------------------------------------------------------------------

/// A node that performs no state update and routes onward via its static or
/// conditional edges (emits an empty [`Command`]).
pub fn noop_node<State, Update>()
-> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Send + 'static,
{
    move |_state, _ctx| -> NodeFuture<Update> {
        Box::pin(async move { Ok(NodeResult::Command(Command::new())) })
    }
}

/// A node that emits a fixed queue of updates, one per activation.
///
/// The first activation returns the first update, the second the second, and so
/// on; once the queue is drained every later activation re-emits the *last*
/// update (so the node stays well-defined inside a loop). An empty queue makes
/// every activation fail with [`TinyAgentsError::Graph`].
pub fn scripted_update_node<State, Update>(
    updates: impl IntoIterator<Item = Update>,
) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Clone + Send + Sync + 'static,
{
    let updates: Arc<Vec<Update>> = Arc::new(updates.into_iter().collect());
    let idx = Arc::new(AtomicUsize::new(0));
    move |_state, _ctx| -> NodeFuture<Update> {
        let updates = updates.clone();
        let idx = idx.clone();
        Box::pin(async move {
            if updates.is_empty() {
                return Err(TinyAgentsError::Graph(
                    "scripted_update_node has no scripted updates".to_string(),
                ));
            }
            let i = idx.fetch_add(1, Ordering::Relaxed).min(updates.len() - 1);
            Ok(NodeResult::Update(updates[i].clone()))
        })
    }
}

/// A node that routes via a fixed queue of `goto` target-sets, one set per
/// activation (saturating the last set once drained).
///
/// Each item is the set of nodes to activate next; route to
/// [`END`](crate::graph::END) to end a branch. An empty queue makes every
/// activation fail with [`TinyAgentsError::Graph`].
pub fn scripted_route_node<State, Update, I, R, N>(
    routes: I,
) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Send + 'static,
    I: IntoIterator<Item = R>,
    R: IntoIterator<Item = N>,
    N: Into<NodeId>,
{
    let routes: Arc<Vec<Vec<NodeId>>> = Arc::new(
        routes
            .into_iter()
            .map(|r| r.into_iter().map(Into::into).collect())
            .collect(),
    );
    let idx = Arc::new(AtomicUsize::new(0));
    move |_state, _ctx| -> NodeFuture<Update> {
        let routes = routes.clone();
        let idx = idx.clone();
        Box::pin(async move {
            if routes.is_empty() {
                return Err(TinyAgentsError::Graph(
                    "scripted_route_node has no scripted routes".to_string(),
                ));
            }
            let i = idx.fetch_add(1, Ordering::Relaxed).min(routes.len() - 1);
            Ok(NodeResult::Command(Command::goto(routes[i].clone())))
        })
    }
}

/// A node that fans out to `target` once per `arg`, delivering each arg through
/// [`NodeContext::send_arg`](crate::graph::NodeContext::send_arg).
///
/// This is the map-reduce / parallel-tool primitive: it emits one
/// [`Send`](crate::graph::Send) per argument, so `target` is scheduled once for
/// each work item with its own per-invocation input.
pub fn fanout_node<State, Update>(
    target: impl Into<NodeId>,
    args: impl IntoIterator<Item = Value>,
) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Send + 'static,
{
    let target = target.into();
    let args: Arc<Vec<Value>> = Arc::new(args.into_iter().collect());
    move |_state, _ctx| -> NodeFuture<Update> {
        let target = target.clone();
        let args = args.clone();
        Box::pin(async move {
            let sends: Vec<SendPacket> = args
                .iter()
                .map(|a| SendPacket::new(target.clone(), a.clone()))
                .collect();
            Ok(NodeResult::Command(Command::send(sends)))
        })
    }
}

/// A node that always fails with [`TinyAgentsError::Graph`] carrying `message`.
pub fn failing_node<State, Update>(
    message: impl Into<String>,
) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Send + 'static,
{
    let message = message.into();
    move |_state, _ctx| -> NodeFuture<Update> {
        let message = message.clone();
        Box::pin(async move { Err(TinyAgentsError::Graph(message)) })
    }
}

/// A node that interrupts (pausing the run for human input) until a resume
/// value arrives, then emits `on_resume`.
///
/// On an activation with no resume value it returns
/// [`NodeResult::Interrupt`](crate::graph::NodeResult::Interrupt) carrying
/// `payload`; on a resumed activation (a non-empty
/// [`NodeContext::resume`](crate::graph::NodeContext)) it returns
/// `NodeResult::Update(on_resume)`. Requires a checkpointer to actually pause
/// and resume.
pub fn interrupting_node<State, Update>(
    payload: Value,
    on_resume: Update,
) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Clone + Send + Sync + 'static,
{
    move |_state, ctx: NodeContext| -> NodeFuture<Update> {
        let payload = payload.clone();
        let on_resume = on_resume.clone();
        Box::pin(async move {
            match ctx.resume {
                Some(_) => Ok(NodeResult::Update(on_resume)),
                None => Ok(NodeResult::Interrupt(Interrupt::new(
                    ctx.node_id.clone(),
                    payload,
                ))),
            }
        })
    }
}

/// Embeds `child` as a shared-state subgraph node (a thin wrapper over
/// [`shared_subgraph_node`](crate::graph::shared_subgraph_node)).
///
/// The child runs over the parent state and its final state becomes the parent
/// update, recording a [`ChildRun`] onto the enclosing run so the subgraph is
/// visible on the parent [`GraphExecution::child_runs`].
pub fn subgraph_test_node<State>(
    child: CompiledGraph<State, State>,
) -> Box<dyn Fn(State, NodeContext) -> NodeFuture<State> + Send + Sync>
where
    State: Clone + Send + Sync + 'static,
{
    crate::graph::subgraph::shared_subgraph_node(child)
}

/// A fake sub-agent node: records a [`ChildRun`] (with `usage`) onto the
/// enclosing run's child-run sink and emits `update`.
///
/// This mimics what [`subagent_node`](crate::graph::subagent_node) records,
/// without needing a registry or a live agent, so tests can assert the
/// parent-run child rollup ([`GraphExecution::child_runs`] /
/// [`run_tree`](crate::graph::GraphExecution::run_tree)) deterministically. The
/// child run preserves the enclosing run's `root_run_id`.
pub fn subagent_fake_node<State, Update>(
    agent: impl Into<String>,
    update: Update,
    usage: UsageTotals,
) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
where
    State: Send + 'static,
    Update: Clone + Send + Sync + 'static,
{
    let agent = agent.into();
    move |_state, ctx: NodeContext| -> NodeFuture<Update> {
        let agent = agent.clone();
        let update = update.clone();
        Box::pin(async move {
            if let Some(sink) = &ctx.child_runs {
                let root_run_id = ctx
                    .root_run_id
                    .clone()
                    .unwrap_or_else(|| ctx.run_id.clone());
                sink.record(ChildRun {
                    node: ctx.node_id.clone(),
                    graph_id: GraphId::new(format!("agent:{agent}")),
                    run_id: RunId::new(format!(
                        "subagent-fake-{}",
                        crate::graph::compiled::next_seq()
                    )),
                    root_run_id,
                    usage,
                });
            }
            Ok(NodeResult::Update(update))
        })
    }
}

impl RetryCountingNode {
    /// Creates a counter whose nodes fail their first `fail_times` activations
    /// and succeed afterwards.
    pub fn new(fail_times: usize) -> Self {
        Self {
            attempts: Arc::new(AtomicUsize::new(0)),
            fail_times,
        }
    }

    /// The number of activations recorded so far across every handler this
    /// counter produced.
    pub fn attempts(&self) -> usize {
        self.attempts.load(Ordering::Relaxed)
    }

    /// Produces a node handler sharing this counter: the first `fail_times`
    /// activations return [`TinyAgentsError::Graph`]; later ones succeed with
    /// `success`.
    pub fn handler<State, Update>(
        &self,
        success: Update,
    ) -> impl Fn(State, NodeContext) -> NodeFuture<Update> + Send + Sync + 'static
    where
        State: Send + 'static,
        Update: Clone + Send + Sync + 'static,
    {
        let attempts = self.attempts.clone();
        let fail_times = self.fail_times;
        move |_state, _ctx| -> NodeFuture<Update> {
            let attempts = attempts.clone();
            let success = success.clone();
            Box::pin(async move {
                let n = attempts.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= fail_times {
                    Err(TinyAgentsError::Graph(format!(
                        "retry_counting_node: attempt {n} of {fail_times} failing"
                    )))
                } else {
                    Ok(NodeResult::Update(success))
                }
            })
        }
    }
}

// ---------------------------------------------------------------------------
// GraphEventRecorder
// ---------------------------------------------------------------------------

impl GraphEventRecorder {
    /// Creates an empty recorder.
    pub fn new() -> Self {
        Self {
            sink: CollectingSink::new(),
        }
    }

    /// An [`Arc`]-wrapped event sink to hand to
    /// [`CompiledGraph::with_event_sink`](crate::graph::CompiledGraph::with_event_sink).
    pub fn sink(&self) -> Arc<dyn GraphEventSink> {
        Arc::new(self.sink.clone())
    }

    /// A snapshot of the recorded events, in emission order.
    pub fn events(&self) -> Vec<GraphEvent> {
        self.sink.events()
    }

    /// The `kind()` string of each recorded event, in emission order.
    pub fn kinds(&self) -> Vec<String> {
        self.sink
            .events()
            .iter()
            .map(|e| e.kind().to_string())
            .collect()
    }

    /// A [`StreamCollector`] over the events recorded so far.
    pub fn collector(&self) -> StreamCollector {
        StreamCollector::new(self.sink.events())
    }
}

// ---------------------------------------------------------------------------
// StreamCollector
// ---------------------------------------------------------------------------

impl StreamCollector {
    /// Wraps an owned event sequence for projection.
    pub fn new(events: Vec<GraphEvent>) -> Self {
        Self { events }
    }

    /// The recorded events, unprojected.
    pub fn events(&self) -> &[GraphEvent] {
        &self.events
    }

    /// The executed-node order (one entry per
    /// [`GraphEvent::NodeCompleted`](crate::graph::GraphEvent)).
    pub fn node_order(&self) -> Vec<NodeId> {
        self.events
            .iter()
            .filter_map(|e| match e {
                GraphEvent::NodeCompleted { node, .. } => Some(node.clone()),
                _ => None,
            })
            .collect()
    }

    /// The nodes that wrote state, in order (one entry per `StateUpdated`).
    pub fn updates(&self) -> Vec<NodeId> {
        self.events
            .iter()
            .filter_map(|e| match e {
                GraphEvent::StateUpdated { node, .. } => Some(node.clone()),
                _ => None,
            })
            .collect()
    }

    /// The `(from, to)` routes the executor selected, in order.
    pub fn routes(&self) -> Vec<(NodeId, NodeId)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                GraphEvent::RouteSelected { node, target } => Some((node.clone(), target.clone())),
                _ => None,
            })
            .collect()
    }

    /// The interrupts emitted during the run, in order.
    pub fn interrupts(&self) -> Vec<Interrupt> {
        self.events
            .iter()
            .filter_map(|e| match e {
                GraphEvent::InterruptEmitted { interrupt } => Some(interrupt.clone()),
                _ => None,
            })
            .collect()
    }

    /// The number of persisted checkpoints (one per `CheckpointSaved`).
    pub fn checkpoint_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, GraphEvent::CheckpointSaved { .. }))
            .count()
    }

    /// The custom node writes, as `(name, data)` pairs in order.
    pub fn custom(&self) -> Vec<(String, Value)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                GraphEvent::Custom { name, data } => Some((name.clone(), data.clone())),
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GraphRun + run_recorded
// ---------------------------------------------------------------------------

impl<State> GraphRun<State> {
    /// Bundles an execution with no recorded events or checkpoint history.
    pub fn new(execution: GraphExecution<State>) -> Self {
        Self {
            execution,
            events: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Attaches a recorded event stream.
    pub fn with_events(mut self, events: Vec<GraphEvent>) -> Self {
        self.events = events;
        self
    }

    /// Attaches a checkpoint history (newest-first).
    pub fn with_history(mut self, history: Vec<StateSnapshot<State>>) -> Self {
        self.history = history;
        self
    }

    /// The recorded events as a [`StreamCollector`].
    pub fn collector(&self) -> StreamCollector {
        StreamCollector::new(self.events.clone())
    }
}

/// Runs `graph` to completion with an event recorder wired in and bundles the
/// result into a [`GraphRun`].
///
/// When `thread` is `Some`, the run executes under that thread id (so a
/// configured checkpointer persists boundary checkpoints) and the thread's
/// checkpoint history is collected into [`GraphRun::history`]; when `None`, the
/// run executes without a thread (no checkpoints, empty history). The graph is
/// cloned so the caller's instance is untouched.
pub async fn run_recorded<State, Update>(
    graph: &CompiledGraph<State, Update>,
    thread: Option<&str>,
    state: State,
) -> Result<GraphRun<State>>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    let recorder = GraphEventRecorder::new();
    let graph = graph.clone().with_event_sink(recorder.sink());
    let execution = match thread {
        Some(thread) => graph.run_with_thread(thread, state).await?,
        None => graph.run(state).await?,
    };
    let history = match thread {
        Some(thread) => graph
            .get_state_history(thread, None)
            .await
            .unwrap_or_default(),
        None => Vec::new(),
    };
    Ok(GraphRun {
        execution,
        events: recorder.events(),
        history,
    })
}

// ---------------------------------------------------------------------------
// assert_graph
// ---------------------------------------------------------------------------

/// Opens a fluent [`GraphAssertions`] over a [`GraphRun`].
///
/// ```ignore
/// assert_graph(&run)
///     .visited(["agent", "tools", "agent"])
///     .routed("agent", "tools")
///     .checkpoint_count(3)
///     .completed();
/// ```
pub fn assert_graph<State>(run: &GraphRun<State>) -> GraphAssertions<'_, State> {
    GraphAssertions { run }
}

impl<State> GraphAssertions<'_, State> {
    /// Asserts the run visited exactly `expected`, in order (repeats included).
    pub fn visited<I, N>(&self, expected: I) -> &Self
    where
        I: IntoIterator<Item = N>,
        N: Into<NodeId>,
    {
        let expected: Vec<NodeId> = expected.into_iter().map(Into::into).collect();
        assert_eq!(
            self.run.execution.visited, expected,
            "assert_graph: expected visited {expected:?} but run visited {:?}",
            self.run.execution.visited
        );
        self
    }

    /// Asserts the executor selected a route from `from` to `to`.
    ///
    /// Reads the recorded `RouteSelected` events when present; when no events
    /// were recorded it falls back to adjacency in the visited sequence (`to`
    /// immediately follows `from`).
    pub fn routed(&self, from: impl Into<NodeId>, to: impl Into<NodeId>) -> &Self {
        let from = from.into();
        let to = to.into();
        let by_event = self.run.events.iter().any(|e| {
            matches!(
                e,
                GraphEvent::RouteSelected { node, target }
                    if *node == from && *target == to
            )
        });
        let by_visited = || {
            self.run
                .execution
                .visited
                .windows(2)
                .any(|w| w[0] == from && w[1] == to)
        };
        assert!(
            by_event || (self.run.events.is_empty() && by_visited()),
            "assert_graph: expected a route from `{from}` to `{to}` but none was found"
        );
        self
    }

    /// Asserts the run persisted exactly `n` checkpoints.
    ///
    /// Counts the recorded `CheckpointSaved` events when present, else falls
    /// back to the collected checkpoint-history length.
    pub fn checkpoint_count(&self, n: usize) -> &Self {
        let count = if self.run.events.is_empty() {
            self.run.history.len()
        } else {
            self.run.collector().checkpoint_count()
        };
        assert_eq!(
            count, n,
            "assert_graph: expected {n} checkpoint(s) but found {count}"
        );
        self
    }

    /// Asserts against the thread's checkpoint history (newest-first) via a
    /// caller-supplied predicate closure.
    pub fn state_history(&self, f: impl FnOnce(&[StateSnapshot<State>])) -> &Self {
        f(&self.run.history);
        self
    }

    /// Asserts against the latest checkpoint snapshot via a caller-supplied
    /// closure. Panics when the run has no checkpoint history.
    pub fn checkpoint(&self, f: impl FnOnce(&StateSnapshot<State>)) -> &Self {
        let latest = self
            .run
            .history
            .first()
            .expect("assert_graph: expected a checkpoint but the run history is empty");
        f(latest);
        self
    }

    /// Asserts the run completed (no pending interrupts and a `Completed`
    /// status).
    pub fn completed(&self) -> &Self {
        assert!(
            !self.run.execution.is_interrupted(),
            "assert_graph: expected the run to complete but it was interrupted: {:?}",
            self.run.execution.interrupts
        );
        assert_eq!(
            self.run.execution.status.status,
            crate::harness::ids::ExecutionStatus::Completed,
            "assert_graph: expected a Completed status but found {:?}",
            self.run.execution.status.status
        );
        self
    }

    /// Asserts the run paused on an interrupt rather than completing.
    pub fn interrupted(&self) -> &Self {
        assert!(
            self.run.execution.is_interrupted(),
            "assert_graph: expected the run to be interrupted but it completed"
        );
        self
    }
}

#[cfg(test)]
mod test;
