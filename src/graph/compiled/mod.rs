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
//!   applied/persisted; an error aborts the run, an interrupt persists a
//!   checkpoint whose pending nodes are that branch and every later active node.
//! - Because branches run on cloned snapshots and never share mutable state,
//!   concurrency is data-race free; the reducer alone resolves conflicting
//!   writes (deterministically, by index).

mod types;

pub use types::{CompiledGraph, GraphExecution, ResumeTarget, StateSnapshot};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use crate::graph::builder::{Branch, BuilderNode, END, ForkId, NodeContext, NodeFuture};
use crate::graph::checkpoint::{
    Checkpoint, CheckpointConfig, CheckpointTuple, Checkpointer, DurabilityMode,
};
use crate::graph::command::{Command, Interrupt, NodeResult, RouteTarget};
use crate::graph::reducer::StateReducer;
use crate::graph::status::GraphRunStatus;
use crate::graph::stream::{GraphEvent, GraphEventSink};
use crate::harness::ids::{
    CheckpointId, ExecutionStatus, GraphId, InterruptId, NodeId, RunId, ThreadId,
};
use crate::{Result, TinyAgentsError};

static SEQ: AtomicU64 = AtomicU64::new(0);

/// Returns a process-unique monotonic sequence number for id generation.
pub(crate) fn next_seq() -> u64 {
    SEQ.fetch_add(1, Ordering::Relaxed)
}

fn next_id(prefix: &str) -> String {
    format!("{prefix}-{}", next_seq())
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
    /// node that produced it.
    goto_map: HashMap<NodeId, Vec<RouteTarget>>,
    /// The lowest-index branch interrupt, if any (its active-set index + value).
    interrupt: Option<(usize, Interrupt)>,
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

fn dedupe(nodes: Vec<NodeId>) -> Vec<NodeId> {
    let mut seen = HashSet::new();
    nodes
        .into_iter()
        .filter(|n| seen.insert(n.clone()))
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
    pub(crate) fn from_parts(
        graph_id: GraphId,
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
    ) -> Self {
        Self {
            graph_id,
            nodes: Arc::new(nodes),
            edges: Arc::new(edges),
            branches: Arc::new(branches),
            command_nodes: Arc::new(command_nodes),
            waiting: Arc::new(waiting),
            entry,
            reducer,
            recursion_limit,
            checkpointer: None,
            event_sink: None,
            journal: None,
            status_store: None,
            namespace: Vec::new(),
            parallel,
            max_concurrency,
            node_timeout,
            durability: crate::graph::checkpoint::DurabilityMode::default(),
        }
    }

    /// The graph id.
    pub fn graph_id(&self) -> &GraphId {
        &self.graph_id
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

    /// Sets the checkpoint namespace (used by subgraph wrappers).
    pub fn with_namespace(mut self, namespace: Vec<String>) -> Self {
        self.namespace = namespace;
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
        self.execute(state, vec![self.entry.clone()], None, HashMap::new())
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
            vec![self.entry.clone()],
            Some(thread_id.into()),
            HashMap::new(),
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
            .get(thread_id.as_str(), checkpoint_id)
            .await?
            .ok_or_else(|| match &target {
                ResumeTarget::Latest => {
                    TinyAgentsError::Resume(format!("no checkpoint found for thread `{thread_id}`"))
                }
                ResumeTarget::Checkpoint(id) => TinyAgentsError::Resume(format!(
                    "no checkpoint `{id}` found for thread `{thread_id}`"
                )),
            })?;
        self.emit(GraphEvent::CheckpointSaved {
            checkpoint_id: CheckpointId::new(checkpoint.checkpoint_id.clone()),
        });

        let active = checkpoint.next_nodes.clone();
        if active.is_empty() {
            return Err(TinyAgentsError::Resume(
                "checkpoint has no pending nodes to resume".to_string(),
            ));
        }

        let mut resume_map = HashMap::new();
        if let Some(value) = command.resume {
            for node in &active {
                resume_map.insert(node.clone(), value.clone());
            }
        }

        self.execute(checkpoint.state, active, Some(thread_id), resume_map)
            .await
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

        let base = checkpointer.get(thread_id, None).await?.ok_or_else(|| {
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
                .route(node, &HashMap::new(), &new_state)?
                .into_iter()
                .map(|t| t.node().clone())
                .filter(|n| n.as_str() != END)
                .collect(),
            None => base.next_nodes.clone(),
        };
        let completed_tasks: Vec<NodeId> = as_node.iter().cloned().collect();

        let checkpoint_id = next_id("ckpt");
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
            .get(source_thread, source_checkpoint_id)
            .await?
            .ok_or_else(|| {
                TinyAgentsError::Checkpoint(format!(
                    "cannot fork: no checkpoint found for thread `{source_thread}`"
                ))
            })?;
        let step = source.to_metadata().step;
        let checkpoint_id = next_id("ckpt");
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
            metadata: serde_json::json!({ "source": "fork", "step": step }),
        };
        let id = checkpointer.put(forked).await?;
        self.emit(GraphEvent::CheckpointSaved { checkpoint_id: id });
        Ok(config)
    }

    async fn execute(
        &self,
        state: State,
        initial_active: Vec<NodeId>,
        thread_id: Option<ThreadId>,
        resume_map: HashMap<NodeId, serde_json::Value>,
    ) -> Result<GraphExecution<State>> {
        let run_id = RunId::new(next_id("run"));
        // When a durable journal is configured, run against a clone whose event
        // sink wraps every emitted event into a `GraphObservation` and appends
        // it (while still forwarding to any pre-existing live sink). The journal
        // sink carries this graph's checkpoint namespace so subgraph runs record
        // their nested path. Default (no journal) leaves `self` untouched.
        if self.journal.is_some() {
            let this = self.clone_with_journal_sink(&run_id, &thread_id);
            this.execute_run(run_id, state, initial_active, thread_id, resume_map)
                .await
        } else {
            self.execute_run(run_id, state, initial_active, thread_id, resume_map)
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

    async fn execute_run(
        &self,
        run_id: RunId,
        mut state: State,
        initial_active: Vec<NodeId>,
        thread_id: Option<ThreadId>,
        mut resume_map: HashMap<NodeId, serde_json::Value>,
    ) -> Result<GraphExecution<State>> {
        let started_at = SystemTime::now();
        let mut visited: Vec<NodeId> = Vec::new();
        let mut steps = 0usize;
        let mut last_checkpoint: Option<CheckpointId> = None;
        let mut parent_checkpoint: Option<String> = None;
        let mut active: Vec<Activation> = dedupe(initial_active)
            .into_iter()
            .map(|node| Activation {
                node,
                send_arg: None,
            })
            .collect();
        // Barrier/waiting-edge arrivals accumulate across supersteps: a waiting
        // node only activates once every required predecessor has arrived.
        let mut barrier_arrivals: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

        self.emit(GraphEvent::RunStarted {
            run_id: run_id.clone(),
        });
        // Record the run as live before the first superstep is scheduled.
        let mut running = self.base_status(&run_id, &thread_id, started_at);
        running.active_nodes = activation_nodes(&active);
        self.save_status(running).await;

        while !active.is_empty() {
            if steps >= self.recursion_limit {
                let err = TinyAgentsError::RecursionLimit(self.recursion_limit);
                self.fail_run(&run_id, &thread_id, started_at, steps, &err)
                    .await;
                return Err(err);
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
                )
                .await
            };
            let StepRun {
                updates,
                goto_map,
                interrupt,
            } = match run_result {
                Ok(step_run) => step_run,
                Err(err) => {
                    self.fail_run(&run_id, &thread_id, started_at, steps, &err)
                        .await;
                    return Err(err);
                }
            };

            // Apply collected updates through the reducer at the boundary.
            for update in updates {
                state = self.reducer.apply(state, update)?;
            }

            // Interrupt: persist a checkpoint whose next nodes are the
            // not-yet-completed members of this step (interrupted node first),
            // then return control to the caller.
            if let Some((index, emitted)) = interrupt {
                let pending: Vec<NodeId> = activation_nodes(&active[index..]);
                let interrupt_id = InterruptId::new(emitted.id.clone());
                let checkpoint_id = self
                    .persist_checkpoint(
                        &thread_id,
                        &run_id,
                        &state,
                        &pending,
                        &activation_nodes(&active[..index]),
                        vec![emitted.clone()],
                        parent_checkpoint.clone(),
                        steps,
                        "loop",
                    )
                    .await?;

                let mut status = self.base_status(&run_id, &thread_id, started_at);
                status.status = ExecutionStatus::Interrupted;
                status.current_step = steps;
                status.active_nodes = pending;
                status.pending_interrupts = vec![interrupt_id];
                status.checkpoint_id = checkpoint_id.clone();
                self.save_status(status.clone()).await;

                return Ok(GraphExecution {
                    state,
                    visited,
                    steps,
                    interrupts: vec![emitted],
                    status,
                    checkpoint_id,
                });
            }

            // Select the next active set from commands or static/conditional
            // edges, evaluated against the freshly-committed state.
            let completed = active.clone();
            let mut next: Vec<Activation> = Vec::new();
            let mut next_seen: HashSet<NodeId> = HashSet::new();
            for activation in &completed {
                let node_id = &activation.node;
                let targets = self.route(node_id, &goto_map, &state)?;
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
                    // `Send` activations may repeat the same node (each carries
                    // its own arg); plain activations are deduplicated by node.
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

            // Persist a boundary checkpoint (node-keyed records). Under
            // `Exit` durability only the terminal boundary (the step that
            // empties the active set) is written; `Sync`/`Async` persist every
            // boundary.
            let completed_nodes = activation_nodes(&completed);
            let next_nodes = activation_nodes(&next);
            let persist_now = match self.durability {
                DurabilityMode::Exit => next.is_empty(),
                DurabilityMode::Sync | DurabilityMode::Async => true,
            };
            let checkpoint_id = if persist_now {
                self.persist_checkpoint(
                    &thread_id,
                    &run_id,
                    &state,
                    &next_nodes,
                    &completed_nodes,
                    Vec::new(),
                    parent_checkpoint.clone(),
                    steps,
                    "loop",
                )
                .await?
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
            visited,
            steps,
            interrupts: Vec::new(),
            status,
            checkpoint_id: last_checkpoint,
        })
    }

    /// Emits a [`GraphEvent::RunFailed`] and records a terminal `Failed` status
    /// for a run that aborted with `err`.
    async fn fail_run(
        &self,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        started_at: SystemTime,
        steps: usize,
        err: &TinyAgentsError,
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
        self.save_status(status).await;
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
    ) -> NodeContext {
        NodeContext {
            node_id: node_id.clone(),
            run_id: run_id.clone(),
            thread_id: thread_id.clone(),
            step,
            resume: resume_map.remove(node_id),
            fork,
            send_arg,
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
        goto_map: &mut HashMap<NodeId, Vec<RouteTarget>>,
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
                    goto_map.insert(node_id.clone(), command.goto);
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
    ) -> Result<StepRun<Update>> {
        let mut updates: Vec<Update> = Vec::new();
        let mut goto_map: HashMap<NodeId, Vec<RouteTarget>> = HashMap::new();
        let mut interrupt: Option<(usize, Interrupt)> = None;

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
            );
            let result = match self
                .run_node_future(node_id, (node.handler)(state.clone(), ctx))
                .await
            {
                Ok(result) => result,
                Err(error) => {
                    self.emit(GraphEvent::NodeFailed {
                        node: node_id.clone(),
                        step,
                        error: error.to_string(),
                    });
                    return Err(error);
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
    ) -> Result<StepRun<Update>> {
        let timeout = self.node_timeout;
        // Build one forked context + future per branch. Node lookup and resume
        // consumption happen up front so the futures borrow nothing local.
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
            );
            let raw = (node.handler)(state.clone(), ctx);
            let owned_node = node_id.clone();
            futures.push(async move {
                match timeout {
                    Some(d) => match tokio::time::timeout(d, raw).await {
                        Ok(result) => result,
                        Err(_) => Err(TinyAgentsError::Timeout(format!(
                            "node `{owned_node}` exceeded its {d:?} timeout"
                        ))),
                    },
                    None => raw.await,
                }
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
        let mut goto_map: HashMap<NodeId, Vec<RouteTarget>> = HashMap::new();
        let mut interrupt: Option<(usize, Interrupt)> = None;

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
                    return Err(error);
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
        })
    }

    /// Resolves the next routing targets for `node_id`.
    ///
    /// Command `goto` (which may include [`Send`] packets) wins over static and
    /// conditional edges; edge/conditional targets are plain node activations.
    fn route(
        &self,
        node_id: &NodeId,
        goto_map: &HashMap<NodeId, Vec<RouteTarget>>,
        state: &State,
    ) -> Result<Vec<RouteTarget>> {
        if let Some(targets) = goto_map.get(node_id) {
            return Ok(targets.clone());
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

    #[allow(clippy::too_many_arguments)]
    async fn persist_checkpoint(
        &self,
        thread_id: &Option<ThreadId>,
        run_id: &RunId,
        state: &State,
        next_nodes: &[NodeId],
        completed_tasks: &[NodeId],
        interrupts: Vec<Interrupt>,
        parent: Option<String>,
        step: usize,
        source: &str,
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, thread_id) else {
            return Ok(None);
        };
        let checkpoint_id = next_id("ckpt");
        let checkpoint = Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id,
            run_id: Some(run_id.to_string()),
            parent_checkpoint_id: parent,
            namespace: self.namespace.clone(),
            state: state.clone(),
            next_nodes: next_nodes.to_vec(),
            completed_tasks: completed_tasks.to_vec(),
            pending_writes: Vec::new(),
            interrupts,
            metadata: serde_json::json!({ "source": source, "step": step }),
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
