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

mod types;

pub use types::{CompiledGraph, GraphExecution, GraphInput, ResumeTarget, StateSnapshot};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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

static SEQ: AtomicU64 = AtomicU64::new(0);

/// Returns a process-unique monotonic sequence number for id generation.
pub(crate) fn next_seq() -> u64 {
    SEQ.fetch_add(1, Ordering::Relaxed)
}

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
            sink.emit(event);
        }
    }
}

impl<State, Update> CompiledGraph<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    /// Runs the graph to completion (or to an interrupt) without a thread.
    ///
    /// Without a thread id no checkpoints are persisted even if a checkpointer
    /// is configured, since checkpoints are keyed by thread.
    pub async fn run(&self, state: State) -> Result<GraphExecution<State>> {
        self.execute(
            state,
            vec![Activation::node(self.entry.clone())],
            None,
            HashMap::new(),
            HashMap::new(),
            None,
        )
        .await
    }

    /// Runs the graph with one or more external inputs in the first superstep.
    ///
    /// [`GraphInput::start`] targets the graph's compiled entry node, preserving
    /// the usual `START -> entry` contract for user input. Additional inputs may
    /// target any real node directly, so separate LLM/tool loops can be seeded
    /// together. Inputs are not deduplicated: two inputs aimed at the same node
    /// produce two separate activations, each with its own
    /// [`NodeContext::send_arg`](crate::graph::NodeContext::send_arg).
    pub async fn run_with_inputs(
        &self,
        state: State,
        inputs: impl IntoIterator<Item = GraphInput>,
    ) -> Result<GraphExecution<State>> {
        let active = self.initial_inputs(inputs)?;
        self.execute(state, active, None, HashMap::new(), HashMap::new(), None)
            .await
    }

    /// Runs the graph under a thread id, persisting checkpoints at every
    /// superstep boundary when a checkpointer is configured.
    pub async fn run_with_thread(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            state,
            vec![Activation::node(self.entry.clone())],
            Some(thread_id.into()),
            HashMap::new(),
            HashMap::new(),
            None,
        )
        .await
    }

    /// Runs the graph under a thread id with one or more external inputs in the
    /// first superstep, persisting checkpoints at every boundary when a
    /// checkpointer is configured.
    pub async fn run_with_thread_inputs(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
        inputs: impl IntoIterator<Item = GraphInput>,
    ) -> Result<GraphExecution<State>> {
        let active = self.initial_inputs(inputs)?;
        self.execute(
            state,
            active,
            Some(thread_id.into()),
            HashMap::new(),
            HashMap::new(),
            None,
        )
        .await
    }

    /// Resumes an interrupted run from its latest checkpoint, re-running the
    /// interrupted node(s) with the resume value supplied by `command`.
    ///
    /// Requires a checkpointer and an existing checkpoint for the thread;
    /// otherwise returns [`TinyAgentsError::Resume`].
    pub async fn resume(
        &self,
        thread_id: impl Into<ThreadId>,
        command: Command<Update>,
    ) -> Result<GraphExecution<State>> {
        self.resume_from(thread_id, ResumeTarget::Latest, command)
            .await
    }

    /// Retries a failed run from its latest (failure-boundary) checkpoint,
    /// re-running the node that failed and the not-yet-run tail of that step.
    ///
    /// This is the resume counterpart for the *failure* path (as opposed to a
    /// human interrupt): after a node handler aborts a checkpointed run — a
    /// transient outage that outlived the node-retry policy, or a hard crash —
    /// the run leaves a resumable checkpoint (see
    /// [`CompiledGraph::with_node_retry`]). Calling `retry` re-runs exactly what
    /// did not complete, carrying no resume value. It is shorthand for
    /// [`CompiledGraph::resume`] with an empty [`Command`].
    ///
    /// To continue on *user feedback* instead of a bare retry, first inspect the
    /// committed state with
    /// [`get_state`](CompiledGraph::get_state), edit it with
    /// [`update_state`](CompiledGraph::update_state), then call `retry` (or
    /// `resume`) — the edited state is what the re-run sees.
    pub async fn retry(&self, thread_id: impl Into<ThreadId>) -> Result<GraphExecution<State>> {
        self.resume_from(thread_id, ResumeTarget::Latest, Command::new())
            .await
    }

    /// Resumes a run from a specific checkpoint (time-travel resume).
    ///
    /// [`ResumeTarget::Latest`] behaves exactly like [`CompiledGraph::resume`];
    /// [`ResumeTarget::Checkpoint`] replays forward from an older checkpoint's
    /// config — re-running its pending nodes (and applying `command`'s resume
    /// value to any interrupted node) without mutating the original record. The
    /// addressed checkpoint is read-only; the replay appends new boundary
    /// checkpoints to the thread rather than rewriting history.
    ///
    /// Requires a checkpointer and a matching checkpoint with pending nodes;
    /// otherwise returns [`TinyAgentsError::Resume`].
    pub async fn resume_from(
        &self,
        thread_id: impl Into<ThreadId>,
        target: ResumeTarget,
        command: Command<Update>,
    ) -> Result<GraphExecution<State>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or_else(|| TinyAgentsError::Resume("no checkpointer configured".to_string()))?;
        let thread_id = thread_id.into();

        let checkpoint_id = match &target {
            ResumeTarget::Latest => None,
            ResumeTarget::Checkpoint(id) => Some(id.as_str()),
        };
        let checkpoint = checkpointer
            .get_scoped(thread_id.as_str(), checkpoint_id, &self.namespace)
            .await?
            .ok_or_else(|| match &target {
                ResumeTarget::Latest => {
                    TinyAgentsError::Resume(format!("no checkpoint found for thread `{thread_id}`"))
                }
                ResumeTarget::Checkpoint(id) => TinyAgentsError::Resume(format!(
                    "no checkpoint `{id}` found for thread `{thread_id}`"
                )),
            })?;
        // Resume *loads* this checkpoint — it is a read, not a write — so emit a
        // restore event, not `CheckpointSaved` (which would falsely inflate
        // persisted-checkpoint counts and mislead durability observers).
        self.emit(GraphEvent::CheckpointRestored {
            checkpoint_id: CheckpointId::new(checkpoint.checkpoint_id.clone()),
        });

        // Prefer the persisted pending activations (which preserve each pending
        // node's `Send` arg); fall back to the node-id projection for
        // checkpoints written before that field existed.
        let active: Vec<Activation> = match &checkpoint.pending_activations {
            Some(pending) if !pending.is_empty() => pending.iter().map(Activation::from).collect(),
            _ => checkpoint
                .next_nodes
                .iter()
                .cloned()
                .map(Activation::node)
                .collect(),
        };
        if active.is_empty() {
            return Err(TinyAgentsError::Resume(
                "checkpoint has no pending nodes to resume".to_string(),
            ));
        }

        let mut resume_map = HashMap::new();
        if let Some(value) = command.resume {
            for activation in &active {
                resume_map.insert(activation.node.clone(), value.clone());
            }
        }

        // Restore accumulated barrier arrivals so a join's precondition survives
        // the interrupt/failure boundary this checkpoint recorded.
        let initial_barriers = barriers_from_persisted(&checkpoint.barrier_arrivals);
        // Chain the first post-resume boundary onto the checkpoint we loaded so
        // the lineage spine stays connected across the resume.
        let initial_parent = Some(checkpoint.checkpoint_id.clone());

        self.execute(
            checkpoint.state,
            active,
            Some(thread_id),
            resume_map,
            initial_barriers,
            initial_parent,
        )
        .await
    }

    fn initial_inputs(
        &self,
        inputs: impl IntoIterator<Item = GraphInput>,
    ) -> Result<Vec<Activation>> {
        let mut active = Vec::new();
        for input in inputs {
            let node = if input.node.as_str() == START {
                self.entry.clone()
            } else if input.node.as_str() == END {
                return Err(TinyAgentsError::Graph(
                    "graph input cannot target END".to_string(),
                ));
            } else {
                if !self.nodes.contains_key(&input.node) {
                    return Err(TinyAgentsError::MissingNode(input.node.to_string()));
                }
                input.node
            };
            active.push(Activation {
                node,
                send_arg: input.payload,
            });
        }
        if active.is_empty() {
            return Err(TinyAgentsError::Validation(
                "run_with_inputs requires at least one input".to_string(),
            ));
        }
        Ok(active)
    }

    // ---- State inspection & time travel ------------------------------------

    /// Returns the configured checkpointer or a [`TinyAgentsError::Checkpoint`]
    /// when inspection is attempted on a graph without durability.
    fn require_checkpointer(&self) -> Result<&Arc<dyn Checkpointer<State>>> {
        self.checkpointer
            .as_ref()
            .ok_or_else(|| TinyAgentsError::Checkpoint("no checkpointer configured".to_string()))
    }

    /// Builds a [`CheckpointConfig`] addressing `checkpoint_id` (or the latest
    /// when `None`) under this graph's namespace.
    fn config_for(&self, thread_id: &str, checkpoint_id: Option<&str>) -> CheckpointConfig {
        CheckpointConfig {
            thread_id: thread_id.to_string(),
            checkpoint_id: checkpoint_id.map(str::to_string),
            namespace: self.namespace.clone(),
        }
    }

    /// Loads a [`StateSnapshot`] for a thread.
    ///
    /// With `checkpoint_id == None` the thread's latest checkpoint is returned;
    /// otherwise the specific checkpoint is addressed. Returns `Ok(None)` when no
    /// matching checkpoint exists. Requires a configured checkpointer.
    pub async fn get_state(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<StateSnapshot<State>>> {
        let checkpointer = self.require_checkpointer()?;
        let config = self.config_for(thread_id, checkpoint_id);
        Ok(checkpointer
            .get_tuple(config)
            .await?
            .map(snapshot_from_tuple))
    }

    /// Returns a thread's state history newest-first, walking the
    /// `parent_checkpoint_id` lineage from the latest checkpoint backwards.
    ///
    /// `limit` caps the number of snapshots returned (the most recent ones).
    /// Requires a configured checkpointer.
    pub async fn get_state_history(
        &self,
        thread_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<State>>> {
        let checkpointer = self.require_checkpointer()?;
        let mut out: Vec<StateSnapshot<State>> = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            if let Some(limit) = limit
                && out.len() >= limit
            {
                break;
            }
            let config = self.config_for(thread_id, cursor.as_deref());
            let Some(tuple) = checkpointer.get_tuple(config).await? else {
                break;
            };
            let parent = tuple.checkpoint.parent_checkpoint_id.clone();
            out.push(snapshot_from_tuple(tuple));
            match parent {
                Some(parent) => cursor = Some(parent),
                None => break,
            }
        }
        Ok(out)
    }

    /// Applies a manual state write to a thread, producing a new checkpoint with
    /// source `update`.
    ///
    /// The write is a genuine graph write: `update` is folded through the same
    /// [`StateReducer`](crate::graph::StateReducer) the executor uses, on top of
    /// the thread's latest committed state. When `as_node` is supplied it must
    /// name a real node (else [`TinyAgentsError::MissingNode`]); the write is
    /// attributed to that node and the new checkpoint's pending nodes become that
    /// node's routing successors (so a subsequent resume continues from after the
    /// attributed node). With `as_node == None` the latest pending node set is
    /// preserved. Requires a configured checkpointer and an existing checkpoint
    /// for the thread.
    pub async fn update_state(
        &self,
        thread_id: &str,
        update: Update,
        as_node: Option<NodeId>,
    ) -> Result<CheckpointConfig> {
        let checkpointer = self.require_checkpointer()?;
        if let Some(node) = &as_node
            && !self.nodes.contains_key(node)
        {
            return Err(TinyAgentsError::MissingNode(node.to_string()));
        }

        let base = checkpointer
            .get_scoped(thread_id, None, &self.namespace)
            .await?
            .ok_or_else(|| {
                TinyAgentsError::Checkpoint(format!(
                    "cannot update state: no checkpoint exists for thread `{thread_id}`"
                ))
            })?;
        let parent_step = base.to_metadata().step;
        let parent_id = base.checkpoint_id.clone();
        let new_state = self.reducer.apply(base.state, update)?;

        // Pending nodes: the attributed node's successors, or the inherited set.
        let next_nodes: Vec<NodeId> = match &as_node {
            Some(node) => self
                .route(node, None, &new_state)?
                .into_iter()
                .map(|t| t.node().clone())
                .filter(|n| n.as_str() != END)
                .collect(),
            None => base.next_nodes.clone(),
        };
        let completed_tasks: Vec<NodeId> = as_node.iter().cloned().collect();
        // With `as_node`, pending becomes that node's (plain) successors, so no
        // send args carry over; without it, inherit the base checkpoint's
        // pending activations verbatim so any pending `Send` args survive.
        let pending_activations = match &as_node {
            Some(_) => None,
            None => base.pending_activations.clone(),
        };
        // Manual writes preserve any accumulated barrier arrivals.
        let barrier_arrivals = base.barrier_arrivals.clone();

        let checkpoint_id = next_checkpoint_id();
        let config = self.config_for(thread_id, Some(&checkpoint_id));
        let checkpoint = Checkpoint {
            thread_id: thread_id.to_string(),
            checkpoint_id,
            run_id: None,
            parent_checkpoint_id: Some(parent_id),
            namespace: self.namespace.clone(),
            state: new_state,
            next_nodes,
            completed_tasks,
            pending_writes: Vec::new(),
            interrupts: Vec::new(),
            pending_activations,
            barrier_arrivals,
            metadata: serde_json::json!({ "source": "update", "step": parent_step + 1 }),
        };
        let id = checkpointer.put(checkpoint).await?;
        self.emit(GraphEvent::CheckpointSaved { checkpoint_id: id });
        Ok(config)
    }

    /// Applies a sequence of manual writes as successive `update` checkpoints,
    /// returning the config of the last one written.
    ///
    /// Each `(update, as_node)` pair is applied with [`CompiledGraph::update_state`]
    /// in order, so every step layers on the previous one's committed state and
    /// produces its own checkpoint. Returns [`TinyAgentsError::Checkpoint`] when
    /// the iterator is empty (there is no resulting config to return).
    pub async fn bulk_update_state(
        &self,
        thread_id: &str,
        updates: impl IntoIterator<Item = (Update, Option<NodeId>)>,
    ) -> Result<CheckpointConfig> {
        let mut last: Option<CheckpointConfig> = None;
        for (update, as_node) in updates {
            last = Some(self.update_state(thread_id, update, as_node).await?);
        }
        last.ok_or_else(|| {
            TinyAgentsError::Checkpoint("bulk_update_state received no updates".to_string())
        })
    }

    /// Forks a checkpoint into a new thread, producing a fresh root checkpoint
    /// with source `fork`.
    ///
    /// Copies the addressed source checkpoint's committed state, pending nodes,
    /// completed tasks, pending writes, and interrupts into `target_thread` under
    /// a brand-new checkpoint id with no parent (the root of the new thread). The
    /// source record is read with `get` and never mutated, so time-travel forks
    /// are non-destructive. With `source_checkpoint_id == None` the source
    /// thread's latest checkpoint is forked. Requires a configured checkpointer.
    pub async fn fork_state(
        &self,
        source_thread: &str,
        source_checkpoint_id: Option<&str>,
        target_thread: &str,
    ) -> Result<CheckpointConfig> {
        let checkpointer = self.require_checkpointer()?;
        let source = checkpointer
            .get_scoped(source_thread, source_checkpoint_id, &self.namespace)
            .await?
            .ok_or_else(|| {
                TinyAgentsError::Checkpoint(format!(
                    "cannot fork: no checkpoint found for thread `{source_thread}`"
                ))
            })?;
        let step = source.to_metadata().step;
        let checkpoint_id = next_checkpoint_id();
        let config = self.config_for(target_thread, Some(&checkpoint_id));
        let forked = Checkpoint {
            thread_id: target_thread.to_string(),
            checkpoint_id,
            run_id: None,
            parent_checkpoint_id: None,
            namespace: source.namespace.clone(),
            state: source.state.clone(),
            next_nodes: source.next_nodes.clone(),
            completed_tasks: source.completed_tasks.clone(),
            pending_writes: source.pending_writes.clone(),
            interrupts: source.interrupts.clone(),
            pending_activations: source.pending_activations.clone(),
            barrier_arrivals: source.barrier_arrivals.clone(),
            metadata: serde_json::json!({ "source": "fork", "step": step }),
        };
        let id = checkpointer.put(forked).await?;
        self.emit(GraphEvent::CheckpointSaved { checkpoint_id: id });
        Ok(config)
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute(
        &self,
        state: State,
        initial_active: Vec<Activation>,
        thread_id: Option<ThreadId>,
        resume_map: HashMap<NodeId, serde_json::Value>,
        initial_barriers: HashMap<NodeId, HashSet<NodeId>>,
        initial_parent: Option<String>,
    ) -> Result<GraphExecution<State>> {
        let run_id = crate::harness::ids::new_run_id();
        // When a durable journal is configured, run against a clone whose event
        // sink wraps every emitted event into a `GraphObservation` and appends
        // it (while still forwarding to any pre-existing live sink). The journal
        // sink carries this graph's checkpoint namespace so subgraph runs record
        // their nested path. Default (no journal) leaves `self` untouched.
        if self.journal.is_some() {
            let this = self.clone_with_journal_sink(&run_id, &thread_id);
            this.execute_run(
                run_id,
                state,
                initial_active,
                thread_id,
                resume_map,
                initial_barriers,
                initial_parent,
            )
            .await
        } else {
            self.execute_run(
                run_id,
                state,
                initial_active,
                thread_id,
                resume_map,
                initial_barriers,
                initial_parent,
            )
            .await
        }
    }

    /// Builds a clone whose `event_sink` is a [`JournalGraphSink`] for `run_id`,
    /// wrapping any existing sink as the live downstream. Returns a plain clone
    /// when no journal is configured.
    fn clone_with_journal_sink(&self, run_id: &RunId, thread_id: &Option<ThreadId>) -> Self {
        let Some(journal) = &self.journal else {
            return self.clone();
        };
        let mut sink = crate::graph::observability::JournalGraphSink::new(
            journal.clone(),
            run_id.clone(),
            self.graph_id.clone(),
        )
        .with_namespace(self.namespace.clone())
        .with_thread(thread_id.clone());
        if let Some(inner) = &self.event_sink {
            sink = sink.with_inner(inner.clone());
        }
        let mut this = self.clone();
        this.event_sink = Some(Arc::new(sink));
        this
    }

    /// Best-effort status write; never aborts the run on a status-store error.
    async fn save_status(&self, status: GraphRunStatus) {
        if let Some(store) = &self.status_store {
            let _ = store.put_status(status).await;
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_run(
        &self,
        run_id: RunId,
        mut state: State,
        initial_active: Vec<Activation>,
        thread_id: Option<ThreadId>,
        mut resume_map: HashMap<NodeId, serde_json::Value>,
        initial_barriers: HashMap<NodeId, HashSet<NodeId>>,
        initial_parent: Option<String>,
    ) -> Result<GraphExecution<State>> {
        let started_at = SystemTime::now();
        let mut visited: Vec<NodeId> = Vec::new();
        let mut steps = 0usize;
        let mut last_checkpoint: Option<CheckpointId> = None;
        // On resume this is the loaded checkpoint's id, so the first boundary
        // checkpoint after a resume chains onto pre-interrupt history rather
        // than orphaning the lineage (which would stop `get_state_history` at
        // the resume point and let `prune` delete the ancestors).
        let mut parent_checkpoint: Option<String> = initial_parent;

        // Build this run's recursion stack from the inherited parent frames and
        // push the frame for this graph call. A push that would exceed
        // `max_depth` fails the run with a clear recursion error before any
        // node executes. Graph-call depth (the stack) is tracked separately
        // from node-loop visits (`node_visits`, below).
        let mut recursion =
            RecursionStack::with_frames(self.recursion_frames.clone(), self.recursion_policy);
        // Run lineage: the root is the first inherited frame's run (the top of
        // the recursion tree) or this run when top-level; the parent is the
        // enclosing run, if any.
        let root_run_id = self
            .recursion_frames
            .first()
            .map(|f| f.run_id.clone())
            .unwrap_or_else(|| run_id.clone());
        let parent_run_id = self.recursion_frames.last().map(|f| f.run_id.clone());
        let this_frame = RecursionFrame {
            graph_id: self.graph_id.clone(),
            node_id: self.recursion_node.clone(),
            run_id: run_id.clone(),
            task_id: None,
            namespace: self.namespace.clone(),
            depth: recursion.depth(),
            parent: parent_run_id.clone(),
        };
        if let Err(err) = recursion.push(this_frame) {
            self.emit(GraphEvent::RunStarted {
                run_id: run_id.clone(),
            });
            self.fail_run(&run_id, &thread_id, started_at, steps, &err, None)
                .await;
            return Err(err);
        }
        // Serialized once per run for embedding in every checkpoint's metadata.
        let recursion_meta =
            serde_json::to_value(recursion.frames()).unwrap_or(serde_json::Value::Null);
        // The live frame stack handed to node contexts so a subgraph node can
        // seed an embedded child with this run's recursion path, plus the
        // per-run sink the node reports its spawned child run into.
        let live_frames = recursion.frames().to_vec();
        let child_sink = ChildRunSink::new();
        // Accumulates every child run spawned across all supersteps for the
        // final `GraphExecution::child_runs`.
        let mut all_child_runs: Vec<ChildRun> = Vec::new();
        // Per-node activation counts for `max_visits_per_node` enforcement.
        let mut node_visits: HashMap<NodeId, usize> = HashMap::new();
        let mut active = initial_active;
        // Barrier/waiting-edge arrivals accumulate across supersteps: a waiting
        // node only activates once every required predecessor has arrived.
        // Seeded from the resumed checkpoint so a join's precondition survives
        // an interrupt/failure boundary.
        let mut barrier_arrivals: HashMap<NodeId, HashSet<NodeId>> = initial_barriers;

        self.emit(GraphEvent::RunStarted {
            run_id: run_id.clone(),
        });
        // Surface this run's recursion depth so observers can attribute nested
        // runs without reconstructing the tree from logs.
        self.emit(GraphEvent::RecursionDepthChanged {
            depth: recursion.depth(),
        });
        // Record the run as live before the first superstep is scheduled.
        let mut running = self.base_status(&run_id, &thread_id, started_at);
        running.active_nodes = activation_nodes(&active);
        self.save_status(running).await;

        while !active.is_empty() {
            // The effective step cap is the smaller of the builder's recursion
            // limit and the policy's `max_total_steps`, so a policy never
            // loosens an existing limit. Both surface a `RecursionLimit`.
            let step_limit = self
                .recursion_limit
                .min(self.recursion_policy.max_total_steps);
            if steps >= step_limit {
                let err = TinyAgentsError::RecursionLimit(step_limit);
                self.fail_run(&run_id, &thread_id, started_at, steps, &err, None)
                    .await;
                return Err(err);
            }
            // Node-loop recursion: enforce `max_visits_per_node` per activation.
            for activation in &active {
                if let Err(err) = recursion.record_node_visit(&mut node_visits, &activation.node) {
                    self.fail_run(&run_id, &thread_id, started_at, steps, &err, None)
                        .await;
                    return Err(err);
                }
            }
            steps += 1;
            self.emit(GraphEvent::StepStarted {
                step: steps,
                active: activation_nodes(&active),
            });

            let run_result = if self.parallel && active.len() > 1 {
                self.run_active_parallel(
                    &active,
                    &state,
                    &run_id,
                    &thread_id,
                    steps,
                    &mut resume_map,
                    &mut visited,
                    &root_run_id,
                    &live_frames,
                    &child_sink,
                )
                .await
            } else {
                self.run_active_sequential(
                    &active,
                    &state,
                    &run_id,
                    &thread_id,
                    steps,
                    &mut resume_map,
                    &mut visited,
                    &root_run_id,
                    &live_frames,
                    &child_sink,
                )
                .await
            };
            let StepRun {
                updates,
                goto_map,
                interrupt,
                failure,
            } = match run_result {
                Ok(step_run) => step_run,
                Err(err) => {
                    self.fail_run(&run_id, &thread_id, started_at, steps, &err, None)
                        .await;
                    return Err(err);
                }
            };

            // Apply collected updates through the reducer at the boundary. A
            // reducer error here must still fail the run (not just unwind
            // leaving it `Running`).
            for update in updates {
                state = match self.reducer.apply(state, update) {
                    Ok(state) => state,
                    Err(err) => {
                        return self
                            .fail_and_return(&run_id, &thread_id, started_at, steps, err)
                            .await;
                    }
                };
            }

            // Collect any child runs spawned by subgraph nodes this step. They
            // are embedded into this boundary's checkpoint metadata (keyed by
            // node) and accumulated onto the final `GraphExecution`.
            let step_child_runs = child_sink.drain();
            all_child_runs.extend(step_child_runs.iter().cloned());
            let child_runs_meta =
                serde_json::to_value(&step_child_runs).unwrap_or(serde_json::Value::Null);

            // Node-handler failure (survived any node-retry policy): the updates
            // of the branches that completed before it are already folded into
            // `state` above, so persist a resumable failure-boundary checkpoint
            // scheduling the failed node (and the not-yet-run tail) for a later
            // `resume`/`retry`, record a `Failed` status carrying the error and
            // that checkpoint, and abort. Without a checkpointer/thread the
            // checkpoint is a no-op and the run aborts exactly as before.
            if let Some(fail) = failure {
                let StepFailure {
                    failed_index,
                    error,
                } = fail;
                let failed_node = active[failed_index].node.clone();
                // Schedule the successors of the branches that completed before
                // the failure (they succeeded; their routing must not be lost)
                // followed by the failed branch and the not-yet-run tail, which
                // re-run on resume with their `Send` args preserved.
                let successors = match self.route_completed(
                    &active[..failed_index],
                    &goto_map,
                    &state,
                    &mut barrier_arrivals,
                ) {
                    Ok(successors) => successors,
                    Err(route_err) => {
                        return self
                            .fail_and_return(&run_id, &thread_id, started_at, steps, route_err)
                            .await;
                    }
                };
                let mut pending = successors;
                pending.extend(active[failed_index..].iter().cloned());
                let completed_nodes = activation_nodes(&active[..failed_index]);
                // A failure-boundary persist error must not replace the original
                // node error: keep reporting the node error and just drop the
                // resumable checkpoint reference.
                let checkpoint_id = self
                    .persist_failure_checkpoint(
                        &thread_id,
                        &run_id,
                        &state,
                        &pending,
                        &completed_nodes,
                        &barrier_arrivals,
                        parent_checkpoint.clone(),
                        steps,
                        &failed_node,
                        &error,
                        &recursion_meta,
                        &child_runs_meta,
                    )
                    .await
                    .unwrap_or(None);
                self.fail_run(
                    &run_id,
                    &thread_id,
                    started_at,
                    steps,
                    &error,
                    checkpoint_id,
                )
                .await;
                return Err(error);
            }

            // Interrupt: persist a checkpoint whose pending activations are the
            // successors of the branches that completed before the interrupt
            // (their routing must survive) followed by the not-yet-completed
            // members of this step (interrupted node first). Each pending branch
            // keeps its `Send` arg; accumulated barrier arrivals are persisted
            // too. Then return control to the caller.
            if let Some((index, emitted)) = interrupt {
                if let Err(err) = self.require_interrupt_durability(&thread_id) {
                    self.fail_run(&run_id, &thread_id, started_at, steps, &err, None)
                        .await;
                    return Err(err);
                }
                let successors = match self.route_completed(
                    &active[..index],
                    &goto_map,
                    &state,
                    &mut barrier_arrivals,
                ) {
                    Ok(successors) => successors,
                    Err(route_err) => {
                        return self
                            .fail_and_return(&run_id, &thread_id, started_at, steps, route_err)
                            .await;
                    }
                };
                let mut pending = successors;
                pending.extend(active[index..].iter().cloned());
                let pending_nodes = activation_nodes(&pending);
                let interrupt_id = InterruptId::new(emitted.id.clone());
                let checkpoint_id = match self
                    .persist_checkpoint(
                        &thread_id,
                        &run_id,
                        &state,
                        &pending,
                        &activation_nodes(&active[..index]),
                        vec![emitted.clone()],
                        &barrier_arrivals,
                        parent_checkpoint.clone(),
                        steps,
                        "loop",
                        &recursion_meta,
                        &child_runs_meta,
                    )
                    .await
                {
                    Ok(id) => id,
                    Err(persist_err) => {
                        return self
                            .fail_and_return(&run_id, &thread_id, started_at, steps, persist_err)
                            .await;
                    }
                };

                let mut status = self.base_status(&run_id, &thread_id, started_at);
                status.status = ExecutionStatus::Interrupted;
                status.current_step = steps;
                status.active_nodes = pending_nodes;
                status.pending_interrupts = vec![interrupt_id];
                status.checkpoint_id = checkpoint_id.clone();
                self.save_status(status.clone()).await;

                return Ok(GraphExecution {
                    state,
                    run_id: run_id.clone(),
                    graph_id: self.graph_id.clone(),
                    root_run_id: root_run_id.clone(),
                    parent_run_id: parent_run_id.clone(),
                    child_runs: all_child_runs,
                    visited,
                    steps,
                    interrupts: vec![emitted],
                    status,
                    checkpoint_id,
                });
            }

            // Select the next active set from commands or static/conditional
            // edges, evaluated against the freshly-committed state. Barrier
            // arrivals accumulate into `barrier_arrivals` (persisted below).
            let completed_nodes = activation_nodes(&active);
            let next = match self.route_completed(&active, &goto_map, &state, &mut barrier_arrivals)
            {
                Ok(next) => next,
                Err(route_err) => {
                    return self
                        .fail_and_return(&run_id, &thread_id, started_at, steps, route_err)
                        .await;
                }
            };

            // Persist a boundary checkpoint. Under `Exit` durability only the
            // terminal boundary (the step that empties the active set) is
            // written; `Sync`/`Async` persist every boundary.
            let persist_now = match self.durability {
                DurabilityMode::Exit => next.is_empty(),
                DurabilityMode::Sync | DurabilityMode::Async => true,
            };
            let checkpoint_id = if persist_now {
                match self
                    .persist_checkpoint(
                        &thread_id,
                        &run_id,
                        &state,
                        &next,
                        &completed_nodes,
                        Vec::new(),
                        &barrier_arrivals,
                        parent_checkpoint.clone(),
                        steps,
                        "loop",
                        &recursion_meta,
                        &child_runs_meta,
                    )
                    .await
                {
                    Ok(id) => id,
                    Err(persist_err) => {
                        return self
                            .fail_and_return(&run_id, &thread_id, started_at, steps, persist_err)
                            .await;
                    }
                }
            } else {
                None
            };
            if let Some(id) = &checkpoint_id {
                last_checkpoint = Some(id.clone());
                parent_checkpoint = Some(id.to_string());
            }

            self.emit(GraphEvent::StepCompleted { step: steps });
            active = next;
        }

        let mut status = self.base_status(&run_id, &thread_id, started_at);
        status.status = ExecutionStatus::Completed;
        status.current_step = steps;
        status.checkpoint_id = last_checkpoint.clone();
        status.ended_at = Some(SystemTime::now());
        self.save_status(status.clone()).await;
        self.emit(GraphEvent::RunCompleted {
            run_id: run_id.clone(),
            steps,
        });

        Ok(GraphExecution {
            state,
            run_id: run_id.clone(),
            graph_id: self.graph_id.clone(),
            root_run_id,
            parent_run_id,
            child_runs: all_child_runs,
            visited,
            steps,
            interrupts: Vec::new(),
            status,
            checkpoint_id: last_checkpoint,
        })
    }

    /// Emits a [`GraphEvent::RunFailed`] and records a terminal `Failed` status
    /// for a run that aborted with `err`.
    ///
    /// `checkpoint_id` is the resumable failure-boundary checkpoint when the run
    /// left one (a node-handler failure on a checkpointed thread), or `None` for
    /// a structural/non-resumable abort. When present it is recorded on the
    /// status so an observer can locate the checkpoint to `resume`/`retry` from.
    async fn fail_run(
        &self,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        started_at: SystemTime,
        steps: usize,
        err: &TinyAgentsError,
        checkpoint_id: Option<CheckpointId>,
    ) {
        self.emit(GraphEvent::RunFailed {
            run_id: run_id.clone(),
            error: err.to_string(),
        });
        let mut status = self.base_status(run_id, thread_id, started_at);
        status.status = ExecutionStatus::Failed;
        status.current_step = steps;
        status.ended_at = Some(SystemTime::now());
        status.error = Some(err.to_string());
        status.checkpoint_id = checkpoint_id;
        self.save_status(status).await;
    }

    /// Records a terminal `Failed` status for `err` (via [`Self::fail_run`]) and
    /// returns it as `Err`.
    ///
    /// Used at the step boundary so an error raised *after* the node runners —
    /// a reducer merge, a routing resolution, or a checkpoint persist — still
    /// transitions the run to `Failed` (rather than leaving observers to see it
    /// stuck in `Running` forever) before the error unwinds out of the run.
    async fn fail_and_return<T>(
        &self,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        started_at: SystemTime,
        steps: usize,
        err: TinyAgentsError,
    ) -> Result<T> {
        self.fail_run(run_id, thread_id, started_at, steps, &err, None)
            .await;
        Err(err)
    }

    /// Persists a resumable failure-boundary checkpoint for a node-handler
    /// failure that survived the node-retry policy.
    ///
    /// Mirrors the interrupt boundary: `next_nodes` schedules the failed node
    /// (and any not-yet-run members of the step) so `resume`/`retry` re-runs
    /// exactly what did not complete, while `completed_tasks` records the
    /// branches that already succeeded (their updates are folded into `state`
    /// before this is called). The rendered error and failed node id are stamped
    /// into the checkpoint metadata for diagnosis. A no-op returning `None` when
    /// no checkpointer/thread is configured — the run then aborts without a
    /// resumable checkpoint, exactly as before this policy existed.
    #[allow(clippy::too_many_arguments)]
    async fn persist_failure_checkpoint(
        &self,
        thread_id: &Option<ThreadId>,
        run_id: &RunId,
        state: &State,
        pending: &[Activation],
        completed_tasks: &[NodeId],
        barrier_arrivals: &HashMap<NodeId, HashSet<NodeId>>,
        parent: Option<String>,
        step: usize,
        failed_node: &NodeId,
        error: &TinyAgentsError,
        recursion: &serde_json::Value,
        child_runs: &serde_json::Value,
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, thread_id) else {
            return Ok(None);
        };
        let checkpoint = Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id: next_checkpoint_id(),
            run_id: Some(run_id.to_string()),
            parent_checkpoint_id: parent,
            namespace: self.namespace.clone(),
            state: state.clone(),
            next_nodes: activation_nodes(pending),
            completed_tasks: completed_tasks.to_vec(),
            pending_writes: Vec::new(),
            interrupts: Vec::new(),
            pending_activations: Some(pending.iter().map(PendingActivation::from).collect()),
            barrier_arrivals: barriers_to_persisted(barrier_arrivals),
            metadata: serde_json::json!({
                "source": "loop",
                "step": step,
                "recursion": recursion,
                "child_runs": child_runs,
                "failed_node": failed_node.as_str(),
                "error": error.to_string(),
            }),
        };
        let id = checkpointer.put(checkpoint).await?;
        self.emit(GraphEvent::CheckpointSaved {
            checkpoint_id: id.clone(),
        });
        Ok(Some(id))
    }

    /// Builds the per-task [`NodeContext`] for `node_id` at the given branch.
    ///
    /// `fork` carries the branch identity in a concurrent step (`None` in
    /// sequential mode or single-node steps). The resume value for the node is
    /// consumed from `resume_map`.
    #[allow(clippy::too_many_arguments)]
    fn node_context(
        &self,
        node_id: &NodeId,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        step: usize,
        resume_map: &mut HashMap<NodeId, serde_json::Value>,
        fork: Option<ForkId>,
        send_arg: Option<serde_json::Value>,
        root_run_id: &RunId,
        frames: &[RecursionFrame],
        child_runs: &ChildRunSink,
    ) -> NodeContext {
        NodeContext {
            node_id: node_id.clone(),
            run_id: run_id.clone(),
            thread_id: thread_id.clone(),
            step,
            resume: resume_map.remove(node_id),
            fork,
            send_arg,
            root_run_id: Some(root_run_id.clone()),
            recursion_frames: frames.to_vec(),
            child_runs: Some(child_runs.clone()),
        }
    }

    /// Wraps a node future in the configured per-node timeout (if any), mapping
    /// an elapsed deadline onto [`TinyAgentsError::Timeout`].
    async fn run_node_future(
        &self,
        node_id: &NodeId,
        fut: NodeFuture<Update>,
    ) -> Result<NodeResult<Update>> {
        match self.node_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, fut).await {
                Ok(result) => result,
                Err(_) => Err(TinyAgentsError::Timeout(format!(
                    "node `{node_id}` exceeded its {timeout:?} timeout"
                ))),
            },
            None => fut.await,
        }
    }

    /// Runs one node handler under the graph's node-retry policy.
    ///
    /// Builds a fresh handler future (and re-clones the context) for each
    /// attempt, so a retried node re-runs from its start — matching the durable
    /// execution model, where a node is never suspended mid-flight. On a
    /// [retryable][crate::harness::retry::is_retryable] error, when a
    /// [`RetryPolicy`](crate::harness::retry::RetryPolicy) is configured and
    /// permits another attempt, it emits
    /// [`GraphEvent::NodeRetryScheduled`], sleeps the (opt-in) backoff, and
    /// retries. Non-retryable errors, absence of a policy, or an exhausted
    /// attempt budget return the error unchanged. The per-node timeout still
    /// bounds every individual attempt via [`Self::run_node_future`].
    async fn run_node_with_retry(
        &self,
        node_id: &NodeId,
        handler: &Arc<NodeHandler<State, Update>>,
        state: &State,
        ctx: NodeContext,
        step: usize,
    ) -> Result<NodeResult<Update>> {
        let mut attempt = 0usize;
        loop {
            let fut = handler(state.clone(), ctx.clone());
            match self.run_node_future(node_id, fut).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let retry = self
                        .node_retry
                        .as_ref()
                        .filter(|policy| policy.should_retry(attempt) && is_retryable(&error));
                    let Some(policy) = retry else {
                        return Err(error);
                    };
                    attempt += 1;
                    self.emit(GraphEvent::NodeRetryScheduled {
                        node: node_id.clone(),
                        step,
                        attempt,
                    });
                    policy.sleep_backoff(attempt).await;
                }
            }
        }
    }

    /// Folds a single successful branch result into the step accumulators.
    ///
    /// Pushes the node to `visited`, records updates/goto, emits the matching
    /// events, and returns the interrupt (with its branch index) when the branch
    /// paused. Shared by the sequential and parallel run paths so both fold
    /// results identically; only the *running* of handlers differs.
    #[allow(clippy::too_many_arguments)]
    fn fold_result(
        &self,
        index: usize,
        node_id: &NodeId,
        step: usize,
        result: NodeResult<Update>,
        updates: &mut Vec<Update>,
        goto_map: &mut HashMap<usize, Vec<RouteTarget>>,
        visited: &mut Vec<NodeId>,
    ) -> Option<(usize, Interrupt)> {
        visited.push(node_id.clone());
        match result {
            NodeResult::Update(update) => {
                updates.push(update);
                self.emit(GraphEvent::StateUpdated {
                    node: node_id.clone(),
                    step,
                });
            }
            NodeResult::Command(command) => {
                if let Some(update) = command.update {
                    updates.push(update);
                    self.emit(GraphEvent::StateUpdated {
                        node: node_id.clone(),
                        step,
                    });
                }
                if !command.goto.is_empty() {
                    goto_map.insert(index, command.goto);
                }
            }
            NodeResult::Interrupt(emitted) => {
                self.emit(GraphEvent::InterruptEmitted {
                    interrupt: emitted.clone(),
                });
                return Some((index, emitted));
            }
        }
        self.emit(GraphEvent::NodeCompleted {
            node: node_id.clone(),
            step,
        });
        None
    }

    /// Runs the active node set one node at a time (default behavior).
    ///
    /// Short-circuits on the first error (run aborts) or interrupt (later nodes
    /// in the step are not started), exactly preserving milestone-1 semantics.
    #[allow(clippy::too_many_arguments)]
    async fn run_active_sequential(
        &self,
        active: &[Activation],
        state: &State,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        step: usize,
        resume_map: &mut HashMap<NodeId, serde_json::Value>,
        visited: &mut Vec<NodeId>,
        root_run_id: &RunId,
        frames: &[RecursionFrame],
        child_runs: &ChildRunSink,
    ) -> Result<StepRun<Update>> {
        let mut updates: Vec<Update> = Vec::new();
        let mut goto_map: HashMap<usize, Vec<RouteTarget>> = HashMap::new();
        let mut interrupt: Option<(usize, Interrupt)> = None;
        let mut failure: Option<StepFailure> = None;

        for (index, activation) in active.iter().enumerate() {
            let node_id = &activation.node;
            let node = self
                .nodes
                .get(node_id)
                .ok_or_else(|| TinyAgentsError::MissingNode(node_id.to_string()))?;

            self.emit(GraphEvent::TaskScheduled {
                node: node_id.clone(),
                step,
            });
            self.emit(GraphEvent::NodeStarted {
                node: node_id.clone(),
                step,
            });

            let ctx = self.node_context(
                node_id,
                run_id,
                thread_id,
                step,
                resume_map,
                None,
                activation.send_arg.clone(),
                root_run_id,
                frames,
                child_runs,
            );
            let result = match self
                .run_node_with_retry(node_id, &node.handler, state, ctx, step)
                .await
            {
                Ok(result) => result,
                Err(error) => {
                    self.emit(GraphEvent::NodeFailed {
                        node: node_id.clone(),
                        step,
                        error: error.to_string(),
                    });
                    // Preserve the progress of the branches that already ran:
                    // the executor records them as completed and schedules their
                    // successors plus this node and the not-yet-run tail for a
                    // resumable retry.
                    failure = Some(StepFailure {
                        failed_index: index,
                        error,
                    });
                    break;
                }
            };

            if let Some(found) = self.fold_result(
                index,
                node_id,
                step,
                result,
                &mut updates,
                &mut goto_map,
                visited,
            ) {
                interrupt = Some(found);
                break;
            }
        }

        Ok(StepRun {
            updates,
            goto_map,
            interrupt,
            failure,
        })
    }

    /// Runs the active node set concurrently (opt-in via `with_parallel`).
    ///
    /// Each branch executes on its own cloned `State` snapshot and a distinct
    /// [`ForkId`], optionally with the [`Send`] argument that scheduled it. With
    /// no `max_concurrency` bound every branch starts before any is awaited and
    /// all are driven via [`futures::future::join_all`]; with a bound the active
    /// set is run in chunks of at most that many futures, so at most that many
    /// node handlers are in flight at once. Results are folded in active-set
    /// index order — the reducer is the join/fan-in — so the merged state is
    /// reproducible regardless of completion order. The lowest-index branch that
    /// errors or interrupts is the step's terminal outcome; lower-index
    /// successful branches still contribute their updates.
    #[allow(clippy::too_many_arguments)]
    async fn run_active_parallel(
        &self,
        active: &[Activation],
        state: &State,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        step: usize,
        resume_map: &mut HashMap<NodeId, serde_json::Value>,
        visited: &mut Vec<NodeId>,
        root_run_id: &RunId,
        frames: &[RecursionFrame],
        child_runs: &ChildRunSink,
    ) -> Result<StepRun<Update>> {
        // Build one forked context + future per branch. Node lookup and resume
        // consumption happen up front so the futures borrow nothing mutable; each
        // branch drives its handler through the node-retry policy (which also
        // applies the per-node timeout), so a transient failure in one branch is
        // retried without disturbing its siblings.
        let mut futures = Vec::with_capacity(active.len());
        for (index, activation) in active.iter().enumerate() {
            let node_id = &activation.node;
            let node = self
                .nodes
                .get(node_id)
                .ok_or_else(|| TinyAgentsError::MissingNode(node_id.to_string()))?;

            self.emit(GraphEvent::TaskScheduled {
                node: node_id.clone(),
                step,
            });
            self.emit(GraphEvent::NodeStarted {
                node: node_id.clone(),
                step,
            });

            self.emit(GraphEvent::ContextForked {
                node: node_id.clone(),
                fork: index,
                step,
            });
            let fork = Some(ForkId::new(index, node_id.clone()));
            let ctx = self.node_context(
                node_id,
                run_id,
                thread_id,
                step,
                resume_map,
                fork,
                activation.send_arg.clone(),
                root_run_id,
                frames,
                child_runs,
            );
            let handler = node.handler.clone();
            let owned_node = node_id.clone();
            futures.push(async move {
                self.run_node_with_retry(&owned_node, &handler, state, ctx, step)
                    .await
            });
        }

        // Drive branches to completion, bounding in-flight count when configured.
        let results = match self.max_concurrency {
            Some(limit) if limit < futures.len() => {
                let mut out = Vec::with_capacity(futures.len());
                let mut iter = futures.into_iter();
                loop {
                    let chunk: Vec<_> = iter.by_ref().take(limit).collect();
                    if chunk.is_empty() {
                        break;
                    }
                    out.extend(futures::future::join_all(chunk).await);
                }
                out
            }
            _ => futures::future::join_all(futures).await,
        };

        // Fold in deterministic active-set index order.
        let mut updates: Vec<Update> = Vec::new();
        let mut goto_map: HashMap<usize, Vec<RouteTarget>> = HashMap::new();
        let mut interrupt: Option<(usize, Interrupt)> = None;
        let mut failure: Option<StepFailure> = None;

        for (index, (activation, result)) in active.iter().zip(results).enumerate() {
            let node_id = &activation.node;
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    self.emit(GraphEvent::NodeFailed {
                        node: node_id.clone(),
                        step,
                        error: error.to_string(),
                    });
                    // The lowest-index failing branch is terminal: fold the
                    // lower-index successes (already applied above) and schedule
                    // their successors plus this branch and the rest for a
                    // resumable retry.
                    failure = Some(StepFailure {
                        failed_index: index,
                        error,
                    });
                    break;
                }
            };

            if let Some(found) = self.fold_result(
                index,
                node_id,
                step,
                result,
                &mut updates,
                &mut goto_map,
                visited,
            ) {
                interrupt = Some(found);
                break;
            }
        }

        Ok(StepRun {
            updates,
            goto_map,
            interrupt,
            failure,
        })
    }

    /// Routes a set of completed activations into their successor activations.
    ///
    /// Honors per-activation command `goto` (keyed by active-set index), static
    /// and conditional edges, barrier gating (a waiting node is held until every
    /// required predecessor has arrived, accumulating into `barrier_arrivals`
    /// across supersteps), and per-node dedup — while preserving each `Send`
    /// packet's per-invocation argument. Emits a
    /// [`GraphEvent::RouteSelected`] per selected edge.
    ///
    /// Shared by the normal step boundary (routes the whole active set) and the
    /// interrupt/failure boundaries (route just the branches that completed
    /// before the pause, so their successors are still scheduled on resume).
    fn route_completed(
        &self,
        completed: &[Activation],
        goto_map: &HashMap<usize, Vec<RouteTarget>>,
        state: &State,
        barrier_arrivals: &mut HashMap<NodeId, HashSet<NodeId>>,
    ) -> Result<Vec<Activation>> {
        let mut next: Vec<Activation> = Vec::new();
        let mut next_seen: HashSet<NodeId> = HashSet::new();
        for (index, activation) in completed.iter().enumerate() {
            let node_id = &activation.node;
            let targets = self.route(node_id, goto_map.get(&index).map(Vec::as_slice), state)?;
            for target in targets {
                let tnode = target.node().clone();
                if tnode.as_str() == END {
                    continue;
                }
                self.emit(GraphEvent::RouteSelected {
                    node: node_id.clone(),
                    target: tnode.clone(),
                });
                // Barrier gating: hold a waiting node until every required
                // predecessor has arrived (possibly across supersteps).
                if let Some(required) = self.waiting.get(&tnode) {
                    let arrived = barrier_arrivals.entry(tnode.clone()).or_default();
                    arrived.insert(node_id.clone());
                    if !required.is_subset(arrived) {
                        continue;
                    }
                    barrier_arrivals.remove(&tnode);
                }
                // `Send` activations may repeat the same node (each carries its
                // own arg); plain activations are deduplicated by node.
                let send_arg = target.send_arg().cloned();
                if send_arg.is_some() {
                    next.push(Activation {
                        node: tnode,
                        send_arg,
                    });
                } else if next_seen.insert(tnode.clone()) {
                    next.push(Activation {
                        node: tnode,
                        send_arg: None,
                    });
                }
            }
        }
        Ok(next)
    }

    /// Resolves the next routing targets for `node_id`.
    ///
    /// Command `goto` (which may include [`Send`] packets) wins over static and
    /// conditional edges; edge/conditional targets are plain node activations.
    ///
    /// `goto` carries this specific activation's [`Command::goto`] targets (when
    /// it returned a command), passed per-activation rather than looked up by
    /// node id so repeated activations of one node never share routing.
    fn route(
        &self,
        node_id: &NodeId,
        goto: Option<&[RouteTarget]>,
        state: &State,
    ) -> Result<Vec<RouteTarget>> {
        if let Some(targets) = goto {
            self.validate_route_targets(node_id, targets)?;
            return Ok(targets.to_vec());
        }
        if let Some(target) = self.edges.get(node_id) {
            return Ok(vec![RouteTarget::Node(target.clone())]);
        }
        if let Some(branch) = self.branches.get(node_id) {
            let route = (branch.router)(state);
            let target = branch.routes.get(&route).cloned().ok_or_else(|| {
                TinyAgentsError::MissingRoute {
                    node: node_id.to_string(),
                    route,
                }
            })?;
            return Ok(vec![RouteTarget::Node(target)]);
        }
        // Sink: no outgoing routing, the branch ends here.
        Ok(Vec::new())
    }

    fn validate_route_targets(&self, node_id: &NodeId, targets: &[RouteTarget]) -> Result<()> {
        for target in targets {
            let target_node = target.node();
            if target_node.as_str() == END {
                continue;
            }
            if target_node.as_str() == START {
                return Err(TinyAgentsError::Graph(format!(
                    "command goto from node `{node_id}` cannot target START"
                )));
            }
            if !self.nodes.contains_key(target_node) {
                return Err(TinyAgentsError::MissingNode(target_node.to_string()));
            }
        }
        Ok(())
    }

    fn require_interrupt_durability(&self, thread_id: &Option<ThreadId>) -> Result<()> {
        if self.checkpointer.is_none() {
            return Err(TinyAgentsError::Resume(
                "interrupt emitted without a configured checkpointer".to_string(),
            ));
        }
        if thread_id.is_none() {
            return Err(TinyAgentsError::Resume(
                "interrupt emitted without a thread id".to_string(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn persist_checkpoint(
        &self,
        thread_id: &Option<ThreadId>,
        run_id: &RunId,
        state: &State,
        pending: &[Activation],
        completed_tasks: &[NodeId],
        interrupts: Vec<Interrupt>,
        barrier_arrivals: &HashMap<NodeId, HashSet<NodeId>>,
        parent: Option<String>,
        step: usize,
        source: &str,
        recursion: &serde_json::Value,
        child_runs: &serde_json::Value,
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, thread_id) else {
            return Ok(None);
        };
        let checkpoint_id = next_checkpoint_id();
        let checkpoint = Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id,
            run_id: Some(run_id.to_string()),
            parent_checkpoint_id: parent,
            namespace: self.namespace.clone(),
            state: state.clone(),
            next_nodes: activation_nodes(pending),
            completed_tasks: completed_tasks.to_vec(),
            pending_writes: Vec::new(),
            pending_activations: Some(pending.iter().map(PendingActivation::from).collect()),
            barrier_arrivals: barriers_to_persisted(barrier_arrivals),
            interrupts,
            metadata: serde_json::json!({
                "source": source,
                "step": step,
                "recursion": recursion,
                "child_runs": child_runs,
            }),
        };
        let id = checkpointer.put(checkpoint).await?;
        self.emit(GraphEvent::CheckpointSaved {
            checkpoint_id: id.clone(),
        });
        Ok(Some(id))
    }

    fn base_status(
        &self,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        started_at: SystemTime,
    ) -> GraphRunStatus {
        let mut status = GraphRunStatus::new(
            run_id.clone(),
            self.graph_id.clone(),
            ExecutionStatus::Running,
        );
        status.thread_id = thread_id.clone();
        status.checkpoint_namespace = self.namespace.clone();
        status.started_at = started_at;
        status.updated_at = SystemTime::now();
        status
    }
}

#[cfg(test)]
mod test;
