//! Superstep executor for the durable graph.
//!
//! This is the engine that makes the recursive runtime durable: it drives a
//! [`CompiledGraph`] in checkpointed supersteps, and because a node handler may
//! recurse into another compiled graph (a subgraph) or a sub-agent, every level
//! of that recursion is observed through the same step/boundary/checkpoint
//! discipline — child runs roll their state, events, and interrupts up through
//! the parent's reducer and checkpointer.
//!
//! The executor runs in supersteps. Each step: take the active node set, run
//! each active node against the committed state snapshot, collect updates /
//! commands / interrupts, apply the reducer at the step boundary, persist a
//! checkpoint at the boundary (when a checkpointer is configured), then select
//! the next active set. The loop stops when the active set empties, every
//! branch reaches [`END`], an interrupt pauses the run, or the recursion limit
//! is hit (a deterministic [`TinyAgentsError::RecursionLimit`]).
//!
//! By default execution is sequential within a step. When the graph is compiled
//! with [`crate::graph::GraphBuilder::with_parallel`], a step with more than one
//! active node runs every branch concurrently via
//! [`futures::future::join_all`], yet the data flow — snapshot reads, boundary
//! reducer application, boundary checkpointing — is identical: each branch reads
//! the same committed snapshot (its own clone), and results are folded into the
//! reducer in deterministic active-set order at the step boundary, so the merged
//! state is reproducible regardless of which branch finishes first.
//!
//! ## Concurrency and interrupt semantics
//!
//! - All active branches in a parallel step start before any is awaited, and all
//!   are driven to completion (`join_all`) before the step boundary runs.
//! - Branch results are then folded in active-set index order. The reducer is
//!   the fan-in / join: lower-index branches' updates are applied first.
//! - The *lowest-index* branch that errors or interrupts is the step's terminal
//!   outcome. Updates produced by lower-index successful branches are still
//!   applied/persisted; an error persists a resumable failure boundary (see
//!   below) and aborts, an interrupt persists a checkpoint whose pending nodes
//!   are that branch and every later active node.
//! - Because branches run on cloned snapshots and never share mutable state,
//!   concurrency is data-race free; the reducer alone resolves conflicting
//!   writes (deterministically, by index).
//!
//! ## Network resilience and resumable failures
//!
//! Two opt-in mechanisms make a run durable under transient failure and
//! restartable after a hard one:
//!
//! - **Node retry.** With
//!   [`CompiledGraph::with_node_retry`], a node whose handler fails with a
//!   [retryable][crate::harness::retry::is_retryable] error (a model or tool
//!   error — the transient class) is re-run from its start up to the policy's
//!   attempt cap, emitting
//!   [`GraphEvent::NodeRetryScheduled`](crate::graph::stream::GraphEvent::NodeRetryScheduled)
//!   and sleeping the opt-in backoff between attempts. A single network blip is
//!   absorbed without touching the run.
//! - **Resumable failure.** When a handler fails beyond the retry budget (or the
//!   error is non-retryable), the executor does not discard the step. On a
//!   checkpointed thread it folds the branches that already completed into
//!   committed state and persists a failure-boundary checkpoint whose
//!   `next_nodes` schedule the failed node (and the not-yet-run tail) for a
//!   later [`CompiledGraph::resume`]/[`CompiledGraph::retry`], with the error
//!   and failed node stamped into the checkpoint metadata. The run then reports
//!   `Failed` (carrying that checkpoint id) and returns the error. A caller can
//!   restart it as-is, or continue on operator feedback by editing state with
//!   [`CompiledGraph::update_state`] before resuming. Without a checkpointer the
//!   run aborts immediately, exactly as before.

mod executor;
mod routing;
mod state_api;
mod types;

pub use types::{CompiledGraph, GraphExecution, GraphInput, ResumeTarget, StateSnapshot};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::graph::builder::{
    Branch, BuilderNode, END, ForkId, NodeContext, NodeFuture, NodeHandler, NodeMeta, START,
};
use crate::graph::checkpoint::{
    BarrierArrivals, Checkpoint, CheckpointConfig, CheckpointTuple, Checkpointer, DurabilityMode,
    PendingActivation,
};
use crate::graph::command::{Command, Interrupt, NodeResult, RouteTarget};
use crate::graph::recursion::{
    ChildRun, ChildRunSink, RecursionFrame, RecursionPolicy, RecursionStack,
};
use crate::graph::reducer::StateReducer;
use crate::graph::status::GraphRunStatus;
use crate::graph::stream::{GraphEvent, GraphEventSink};
use crate::harness::ids::{
    CheckpointId, ExecutionStatus, GraphId, InterruptId, NodeId, RunId, ThreadId,
};
use crate::harness::retry::is_retryable;
use crate::{Result, TinyAgentsError};

/// Allocates a fresh checkpoint id (string form) that is collision-free across
/// process restarts.
///
/// Delegates to [`crate::harness::ids::new_checkpoint_id`]: a resumed thread
/// restarted in a new process must never re-mint a checkpoint id it already
/// used, or the lineage map (`prune`) and time-travel resume corrupt. The bare
/// process-local counter this used to build ids from restarted at `0` every
/// process and did exactly that.
fn next_checkpoint_id() -> String {
    crate::harness::ids::new_checkpoint_id()
        .as_str()
        .to_string()
}

/// Projects a loaded [`CheckpointTuple`] onto a [`StateSnapshot`] for the
/// state-inspection API. The checkpoint's listing metadata is derived through
/// [`Checkpoint::to_metadata`](crate::graph::Checkpoint::to_metadata) so a
/// snapshot's `metadata` always matches what `Checkpointer::list` reports.
fn snapshot_from_tuple<State>(tuple: CheckpointTuple<State>) -> StateSnapshot<State> {
    let CheckpointTuple {
        config,
        checkpoint,
        parent_config,
        ..
    } = tuple;
    let metadata = checkpoint.to_metadata();
    let next_nodes = checkpoint.next_nodes.clone();
    StateSnapshot {
        values: checkpoint.state,
        tasks: next_nodes.clone(),
        next_nodes,
        config,
        metadata,
        parent_config,
        pending_interrupts: checkpoint.interrupts,
    }
}

/// The folded result of running a superstep's active node set, ready to apply
/// at the step boundary.
struct StepRun<Update> {
    /// Branch updates in deterministic active-set index order.
    updates: Vec<Update>,
    /// Explicit routing (plain `goto` nodes and/or [`Send`] packets) keyed by the
    /// producing branch's active-set index.
    ///
    /// Keyed by index rather than node id so repeated [`Send`] activations of
    /// the *same* node within a step (map-reduce fanout) each keep their own
    /// [`Command::goto`] — a node-keyed map would let a later activation's
    /// command clobber an earlier one's routing.
    goto_map: HashMap<usize, Vec<RouteTarget>>,
    /// The lowest-index branch interrupt, if any (its active-set index + value).
    interrupt: Option<(usize, Interrupt)>,
    /// A node-handler failure that survived the node-retry policy, if any. When
    /// set, `updates` still carries the updates of the branches that completed
    /// *before* the failing branch, so the executor can fold that partial
    /// progress into committed state and persist a resumable failure boundary.
    failure: Option<StepFailure>,
}

/// A node-handler failure captured by a runner so the executor can persist a
/// resumable failure-boundary checkpoint instead of discarding partial progress.
struct StepFailure {
    /// Active-set index of the branch whose handler ultimately failed (after any
    /// retries). The executor derives the failed node, the completed lower-index
    /// branches (whose successors it schedules) and the pending tail (which it
    /// re-runs) from this index against the step's active set — preserving each
    /// pending branch's [`Send`] argument.
    failed_index: usize,
    /// The escalated error.
    error: TinyAgentsError,
}

/// One scheduled activation in the active set: the node plus an optional
/// per-invocation [`Send`] argument delivered via [`NodeContext::send_arg`].
///
/// Plain edge/`goto`/conditional activations carry `send_arg == None`; a
/// [`crate::graph::Send`] packet carries `Some(arg)`. Multiple activations may
/// target the same node within a step (map-reduce fanout), so the active set is
/// a `Vec` of these rather than a deduplicated node set.
#[derive(Clone)]
struct Activation {
    node: NodeId,
    send_arg: Option<serde_json::Value>,
}

impl Activation {
    fn node(node: NodeId) -> Self {
        Self {
            node,
            send_arg: None,
        }
    }
}

impl From<&Activation> for PendingActivation {
    fn from(a: &Activation) -> Self {
        PendingActivation {
            node: a.node.clone(),
            send_arg: a.send_arg.clone(),
        }
    }
}

impl From<&PendingActivation> for Activation {
    fn from(p: &PendingActivation) -> Self {
        Activation {
            node: p.node.clone(),
            send_arg: p.send_arg.clone(),
        }
    }
}

/// Projects the live barrier-arrival map onto its serializable checkpoint form.
fn barriers_to_persisted(map: &HashMap<NodeId, HashSet<NodeId>>) -> Vec<BarrierArrivals> {
    map.iter()
        .map(|(node, arrived)| BarrierArrivals {
            node: node.clone(),
            arrived: arrived.iter().cloned().collect(),
        })
        .collect()
}

/// Rebuilds the live barrier-arrival map from a checkpoint's persisted form.
fn barriers_from_persisted(persisted: &[BarrierArrivals]) -> HashMap<NodeId, HashSet<NodeId>> {
    persisted
        .iter()
        .map(|b| (b.node.clone(), b.arrived.iter().cloned().collect()))
        .collect()
}

/// Maps an [`Activation`] slice to its node ids (for events, status, and
/// checkpoint records, which are node-keyed).
fn activation_nodes(active: &[Activation]) -> Vec<NodeId> {
    active.iter().map(|a| a.node.clone()).collect()
}

impl<State, Update> CompiledGraph<State, Update> {
    /// Internal constructor used by the builder.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        graph_id: GraphId,
        name: Option<String>,
        nodes: HashMap<NodeId, BuilderNode<State, Update>>,
        edges: HashMap<NodeId, NodeId>,
        branches: HashMap<NodeId, Branch<State>>,
        command_nodes: HashSet<NodeId>,
        waiting: HashMap<NodeId, HashSet<NodeId>>,
        entry: NodeId,
        reducer: Arc<dyn StateReducer<State, Update>>,
        recursion_limit: usize,
        parallel: bool,
        max_concurrency: Option<usize>,
        node_timeout: Option<Duration>,
        node_meta: HashMap<NodeId, NodeMeta>,
    ) -> Self {
        Self {
            graph_id,
            name,
            nodes: Arc::new(nodes),
            edges: Arc::new(edges),
            branches: Arc::new(branches),
            command_nodes: Arc::new(command_nodes),
            waiting: Arc::new(waiting),
            node_meta: Arc::new(node_meta),
            entry,
            reducer,
            recursion_limit,
            recursion_policy: crate::graph::recursion::RecursionPolicy::default(),
            recursion_frames: Vec::new(),
            recursion_node: None,
            checkpointer: None,
            event_sink: None,
            journal: None,
            status_store: None,
            namespace: Vec::new(),
            parallel,
            max_concurrency,
            node_timeout,
            durability: crate::graph::checkpoint::DurabilityMode::default(),
            node_retry: None,
        }
    }

    /// The graph id.
    pub fn graph_id(&self) -> &GraphId {
        &self.graph_id
    }

    /// The optional human-readable graph name, if one was set via
    /// [`GraphBuilder::with_name`](crate::graph::GraphBuilder::with_name).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The checkpoint namespace (empty for top-level graphs).
    pub fn namespace(&self) -> &[String] {
        &self.namespace
    }

    /// Attaches a checkpointer, enabling durability, interrupts, and resume.
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer<State>>) -> Self
    where
        State: Send + Sync + 'static,
    {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Attaches an event sink for low-level streaming/observability.
    pub fn with_event_sink(mut self, sink: Arc<dyn GraphEventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }

    /// Sets the [`DurabilityMode`] that governs when boundary checkpoints are
    /// persisted.
    ///
    /// The default is [`DurabilityMode::Sync`] (persist before the next step).
    /// [`DurabilityMode::Async`] is currently treated like `Sync` — it persists
    /// the boundary state once committed — and documents the intent to move
    /// persistence off the critical path. [`DurabilityMode::Exit`] persists only
    /// the terminal checkpoint (and any interrupt boundary, which is required
    /// for resume), skipping intermediate boundaries.
    pub fn with_durability(mut self, durability: DurabilityMode) -> Self {
        self.durability = durability;
        self
    }

    /// Sets the per-node [`RetryPolicy`] applied around every node handler.
    ///
    /// Opt-in network resilience for the graph: when a node handler fails with a
    /// [retryable][crate::harness::retry::is_retryable] error (a model or tool
    /// error — the transient class), the executor re-runs the node from its
    /// start up to the policy's attempt cap, emitting a
    /// [`GraphEvent::NodeRetryScheduled`](crate::graph::stream::GraphEvent::NodeRetryScheduled)
    /// before each retry. Backoff between attempts is slept on only when the
    /// policy opts in via
    /// [`RetryPolicy::with_backoff_sleep`](crate::harness::retry::RetryPolicy::with_backoff_sleep).
    ///
    /// Non-retryable errors, and retryable errors once attempts are exhausted,
    /// escalate — and, on a checkpointed thread, leave a resumable
    /// failure-boundary checkpoint (see [`CompiledGraph::resume`]) so the run can
    /// be restarted or continued rather than lost. Without a policy (the
    /// default) the first node error aborts the run immediately.
    pub fn with_node_retry(mut self, policy: crate::harness::retry::RetryPolicy) -> Self {
        self.node_retry = Some(policy);
        self
    }

    /// Sets the checkpoint namespace (used by subgraph wrappers).
    pub fn with_namespace(mut self, namespace: Vec<String>) -> Self {
        self.namespace = namespace;
        self
    }

    /// Sets the [`RecursionPolicy`] enforced while this graph runs.
    ///
    /// The policy bounds three independently-tracked recursion dimensions:
    /// run-tree depth (`max_depth`), per-node activations within a run
    /// (`max_visits_per_node`), and total super-steps per run
    /// (`max_total_steps`). The effective per-run step cap is the smaller of
    /// the policy's `max_total_steps` and the builder's recursion limit, so
    /// configuring a policy never *loosens* an existing limit.
    pub fn with_recursion_policy(mut self, policy: RecursionPolicy) -> Self {
        self.recursion_policy = policy;
        self
    }

    /// Seeds the inherited recursion frames of an enclosing run.
    ///
    /// A subgraph or sub-agent wrapper passes the parent run's frame stack so
    /// this run extends the parent's recursion tree (its root frame's `depth`
    /// and `parent` continue from the caller) rather than starting a fresh tree
    /// at depth zero. Top-level graphs leave this empty.
    pub fn with_recursion_frames(mut self, frames: Vec<RecursionFrame>) -> Self {
        self.recursion_frames = frames;
        self
    }

    /// Sets the hosting node id used as this run's root recursion-frame node.
    ///
    /// A subgraph wrapper sets this to the embedding node id so the child run's
    /// frame (and the parent/child [`RunTree`](crate::graph::RunTree)) names the
    /// node that ran the embedded graph. Top-level graphs leave this unset.
    pub fn with_recursion_node(mut self, node: NodeId) -> Self {
        self.recursion_node = Some(node);
        self
    }

    /// Attaches a durable event journal. Every emitted [`GraphEvent`] is wrapped
    /// into a [`crate::graph::observability::GraphObservation`] (stamped with the
    /// run's lineage, the graph's checkpoint namespace, and the run id) and
    /// appended for offset-addressable replay. Opt-in; default off.
    pub fn with_event_journal(
        mut self,
        journal: Arc<dyn crate::graph::observability::GraphEventJournal>,
    ) -> Self {
        self.journal = Some(journal);
        self
    }

    /// Attaches a run-status store. The executor writes a compact
    /// [`GraphRunStatus`] at every lifecycle boundary (start, terminal,
    /// interrupt, failure) so observers can poll run state. Opt-in; default off.
    pub fn with_status_store(
        mut self,
        status_store: Arc<dyn crate::graph::observability::GraphStatusStore>,
    ) -> Self {
        self.status_store = Some(status_store);
        self
    }

    fn emit(&self, event: GraphEvent) {
        if let Some(sink) = &self.event_sink {
            // Durable sinks persist asynchronously off the executor thread. On a
            // terminal run event, flush so a caller that reads the journal right
            // after the run returns sees a complete log.
            let terminal = matches!(
                event,
                GraphEvent::RunCompleted { .. } | GraphEvent::RunFailed { .. }
            );
            sink.emit(event);
            if terminal {
                sink.flush();
            }
        }
    }
}

#[cfg(test)]
mod test;
