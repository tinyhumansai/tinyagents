//! Per-run execution context threaded through the superstep loop.
//!
//! [`RunCtx`] bundles run identity (ids, namespace), clocks/deadlines,
//! recursion bookkeeping, the accumulators a superstep loop carries forward
//! (`node_visits`, `barrier_arrivals`, `visited`, `all_child_runs`,
//! `steps`/checkpoint lineage), and the handles for background checkpoint
//! writes and status/event I/O.
//!
//! It exists so the step-running, boundary, and resume helpers split out of
//! `executor.rs` stop threading a dozen positional parameters between them:
//! every one of those helpers takes `&RunCtx`/`&mut RunCtx` plus the handful
//! of values that are genuinely local to that call (the active set, the
//! state snapshot, a step's folded outcome). `RunCtx` is created once per
//! `execute_run` call and never outlives it — it borrows the owning
//! [`CompiledGraph`] for that duration.

use super::*;

use crate::observability::GraphStatusStore;

/// Run-scoped state for one `execute_run` call.
///
/// Fields fall into three groups: identity that never changes for the run
/// (`run_id`, `thread_id`, `root_run_id`, `parent_run_id`, `started_at`,
/// `live_frames`, `recursion_meta`, `binding`), accumulators the superstep
/// loop updates every iteration (`recursion`, `node_visits`,
/// `barrier_arrivals`, `resume_map`, `visited`, `all_child_runs`, `steps`,
/// `last_checkpoint`, `parent_checkpoint`), and I/O handles
/// (`child_sink`, `async_writes`). `graph` is the owning [`CompiledGraph`],
/// kept here so the convenience methods below (`emit`, `save_status`,
/// `base_status`, `node_context`) don't need a separate receiver.
pub(super) struct RunCtx<'a, State, Update> {
    pub(super) graph: &'a CompiledGraph<State, Update>,
    pub(super) run_id: RunId,
    pub(super) thread_id: Option<ThreadId>,
    pub(super) root_run_id: RunId,
    pub(super) parent_run_id: Option<RunId>,
    pub(super) started_at: SystemTime,
    /// Monotonic start instant used for wall-clock-jump-proof deadline
    /// arithmetic (M3): `run_deadline` is checked against
    /// [`std::time::Instant::elapsed`] rather than [`SystemTime::elapsed`],
    /// so a system clock step (NTP sync, VM pause/resume, manual clock
    /// change) cannot make a run time out early or never at all.
    /// `started_at` (above) remains the wall-clock stamp surfaced on
    /// [`GraphRunStatus`], which is what observers expect.
    pub(super) started_instant: std::time::Instant,
    pub(super) live_frames: Vec<RecursionFrame>,
    pub(super) recursion_meta: serde_json::Value,
    pub(super) recursion: RecursionStack,
    pub(super) binding: Option<crate::subagent_node::AgentInvocationBinding>,
    pub(super) child_sink: ChildRunSink,
    pub(super) node_visits: HashMap<NodeId, usize>,
    pub(super) barrier_arrivals: HashMap<NodeId, HashSet<NodeId>>,
    pub(super) async_writes: AsyncCheckpointWrites,
    /// Keyed by task id, falling back to node id (I1/R5); see
    /// [`super::executor::RunSeed::resume_map`].
    pub(super) resume_map: HashMap<String, serde_json::Value>,
    pub(super) visited: Vec<NodeId>,
    pub(super) all_child_runs: Vec<ChildRun>,
    pub(super) steps: usize,
    pub(super) last_checkpoint: Option<CheckpointId>,
    pub(super) parent_checkpoint: Option<String>,
    /// Nodes (with their persisted explicit `Command::goto`, R1) carried
    /// forward from a resumed mid-step checkpoint (an interrupt/failure
    /// boundary whose completed siblings were never routed) — see
    /// [`super::boundary::CompiledGraph::advance`]'s doc. `None` for a fresh
    /// run or a resume from a fully-routed (normal) boundary. Consumed
    /// (`take`n) by the first `advance` call of this run;
    /// [`super::boundary`]'s failure/interrupt boundaries read it (without
    /// consuming it) to keep carrying it forward across a step that
    /// interrupts or fails more than once in a row.
    pub(super) carried_completed: Option<Vec<(NodeId, Vec<RouteTarget>)>>,
    /// Optional cooperative-cancellation token for this run (I4 part 2), from
    /// [`super::RunOptions::cancellation`]. Checked at every superstep
    /// boundary and raced against the step's in-flight node handlers by
    /// [`super::executor::CompiledGraph::run_step_with_cancel`].
    pub(super) cancellation: Option<tinyagents_harness::CancellationToken>,
    /// Guards against the run future being dropped before it reaches a
    /// normal terminal state (I4 part 3) — see [`RunDropGuard`].
    pub(super) drop_guard: RunDropGuard,
}

/// Drop guard (I4 part 3) that guarantees a run's terminal status is
/// durably set to `Cancelled` if the run's future is dropped before it
/// reaches a normal terminal state (completed, failed, interrupted, or an
/// explicit cooperative cancellation the executor already handled) —
/// for example when a caller wraps the run in `tokio::time::timeout` and the
/// deadline fires, or aborts the `JoinHandle` of the task the run was
/// spawned on. Without this guard such a drop leaves the run's last written
/// status stuck at `Running` forever, with nothing to signal that it will
/// never make further progress.
///
/// Constructed armed by [`RunCtx::start`]; [`Self::disarm`] is called at the
/// top of every one of `execute_run`'s terminal exit paths — success
/// ([`super::executor::CompiledGraph::finish_run`]), an aborting error
/// ([`super::boundary::CompiledGraph::fail_and_return`]), a resumable
/// failure boundary
/// ([`super::boundary::CompiledGraph::handle_failure_boundary`]), an
/// interrupt boundary
/// ([`super::boundary::CompiledGraph::handle_interrupt_boundary`]), and an
/// explicit cooperative-cancellation boundary
/// ([`super::boundary::CompiledGraph::handle_cancel_boundary`]) — so a run
/// that reaches a real terminal state on its own never gets a spurious
/// `Cancelled` overwrite from `Drop` racing (or following) that path.
///
/// # Best-effort guarantee
///
/// `Drop::drop` cannot `.await`, so this guard cannot synchronously flush
/// in-flight [`AsyncCheckpointWrites`]. It instead spawns a detached
/// background task (via [`tokio::runtime::Handle::try_current`], a no-op
/// outside a tokio runtime) that persists a `Cancelled` [`GraphRunStatus`];
/// this can still race a runtime shutdown that happens immediately after the
/// drop, in which case even this best-effort write may not land. Any
/// checkpoint write still in flight under `DurabilityMode::Async` is *not*
/// separately re-awaited by this guard — but it is not abandoned either: a
/// tokio `JoinHandle` being dropped only detaches it, it does not abort the
/// task, so the underlying `checkpointer.put`/`put_writes` call keeps
/// running to completion on its own regardless of whether `RunCtx` is still
/// alive to track it. What this guard cannot restore is the tracker's
/// ability to *observe* that write's outcome (the concern
/// [`AsyncCheckpointWrites`]'s own contract documents) — a write that fails
/// after the run future was dropped is only visible in the checkpointer
/// backend's own logs, not through `GraphRunStatus.error`. The one concrete,
/// testable contract this guard gives is: the run's stored status is never
/// left at `Running` forever.
pub(super) struct RunDropGuard {
    armed: bool,
    status_store: Option<Arc<dyn GraphStatusStore>>,
    run_id: RunId,
    thread_id: Option<ThreadId>,
    graph_id: GraphId,
    namespace: Vec<String>,
    started_at: SystemTime,
}

impl RunDropGuard {
    #[allow(clippy::too_many_arguments)]
    fn new(
        status_store: Option<Arc<dyn GraphStatusStore>>,
        run_id: RunId,
        thread_id: Option<ThreadId>,
        graph_id: GraphId,
        namespace: Vec<String>,
        started_at: SystemTime,
    ) -> Self {
        Self {
            armed: true,
            status_store,
            run_id,
            thread_id,
            graph_id,
            namespace,
            started_at,
        }
    }

    /// Disarms the guard so a normal terminal exit does not also trigger the
    /// `Drop`-time `Cancelled` write.
    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for RunDropGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Some(store) = self.status_store.take() else {
            return;
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let run_id = self.run_id.clone();
        let thread_id = self.thread_id.clone();
        let graph_id = self.graph_id.clone();
        let namespace = std::mem::take(&mut self.namespace);
        let started_at = self.started_at;
        handle.spawn(async move {
            let mut status =
                GraphRunStatus::new(run_id.clone(), graph_id, ExecutionStatus::Cancelled);
            status.thread_id = thread_id;
            status.checkpoint_namespace = namespace;
            status.started_at = started_at;
            status.updated_at = SystemTime::now();
            status.ended_at = Some(SystemTime::now());
            if let Err(err) = store.put_status(status).await {
                tracing::warn!(
                    "[graph:drop-guard] failed to persist cancelled status for run `{run_id}` \
                     after its future was dropped before completion: {err}"
                );
            }
        });
    }
}

/// Everything a resumed run seeds `RunCtx` with beyond a fresh run's
/// defaults, bundled into one optional parameter so [`RunCtx::start`] does
/// not grow a positional argument per resume-only field.
///
/// A fresh run (`resume_from_inner` was never called) passes `None`, which
/// is equivalent to `ResumeSeed::default()`.
#[derive(Default)]
pub(super) struct ResumeSeed {
    /// The loaded checkpoint's own step number (`to_metadata().step`), so
    /// this run's `ctx.steps` continues counting up from it instead of
    /// restarting at `0` — see the I3 finding in
    /// `docs/runtime-comparison/code-review-graph.md`: without this,
    /// `metadata.step` (and so `get_state_history`) goes non-monotonic
    /// across a resume, and per-node visit caps
    /// (`RecursionPolicy::max_visits_per_node`) reset every resume rather
    /// than bounding the whole thread's lifetime.
    pub(super) initial_steps: usize,
    /// The loaded checkpoint's persisted `node_visits` metadata (see
    /// [`super::boundary`]'s checkpoint builders), so per-node visit counts
    /// accumulate across a resume instead of resetting.
    pub(super) initial_node_visits: HashMap<NodeId, usize>,
    /// Nodes (with their persisted goto, R1) carried forward from a
    /// mid-step (interrupt/failure) checkpoint whose completed siblings
    /// were never routed — see [`RunCtx::carried_completed`].
    pub(super) carried_completed: Option<Vec<(NodeId, Vec<RouteTarget>)>>,
}

impl<'a, State, Update> RunCtx<'a, State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    /// Forwards to the owning graph's event sink (a no-op without one).
    pub(super) fn emit(&self, event: GraphEvent) {
        self.graph.emit(event);
    }

    /// Whether this run's cooperative-cancellation token (if any) has been
    /// cancelled (I4 part 2).
    pub(super) fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(tinyagents_harness::CancellationToken::is_cancelled)
    }

    /// Disarms this run's [`RunDropGuard`] — called at the top of every
    /// terminal exit path of `execute_run` so a normal completion never
    /// races a spurious `Cancelled` write from `Drop`.
    pub(super) fn disarm_drop_guard(&mut self) {
        self.drop_guard.disarm();
    }

    /// Forwards to the owning graph's status store (a no-op without one).
    pub(super) async fn save_status(&self, status: GraphRunStatus) {
        self.graph.save_status(status).await;
    }

    /// Builds a fresh [`GraphRunStatus`] for this run at `Running` status,
    /// stamped with this context's identity and start time.
    pub(super) fn base_status(&self) -> GraphRunStatus {
        self.graph
            .base_status(&self.run_id, &self.thread_id, self.started_at)
    }

    /// Builds this run's `RunCtx`: constructs the recursion stack from the
    /// inherited parent frames and pushes the frame for this graph call (a
    /// push that would exceed `max_depth` fails the run — emitting
    /// `RunStarted` and a terminal `Failed` status — before any node
    /// executes), then emits `RunStarted`/`RecursionDepthChanged` for a
    /// successful push.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn start(
        graph: &'a CompiledGraph<State, Update>,
        run_id: RunId,
        thread_id: Option<ThreadId>,
        resume_map: HashMap<String, serde_json::Value>,
        initial_barriers: HashMap<NodeId, HashSet<NodeId>>,
        initial_parent: Option<String>,
        binding: Option<crate::subagent_node::AgentInvocationBinding>,
        resume_seed: ResumeSeed,
        cancellation: Option<tinyagents_harness::CancellationToken>,
    ) -> Result<Self> {
        let ResumeSeed {
            initial_steps,
            initial_node_visits,
            carried_completed,
        } = resume_seed;
        let started_at = SystemTime::now();
        let started_instant = std::time::Instant::now();
        // Graph-call depth (the stack) is tracked separately from node-loop
        // visits (`node_visits`, below).
        let mut recursion =
            RecursionStack::with_frames(graph.recursion_frames.clone(), graph.recursion_policy);
        let root_run_id = graph
            .recursion_frames
            .first()
            .map(|f| f.run_id.clone())
            .unwrap_or_else(|| run_id.clone());
        let parent_run_id = graph.recursion_frames.last().map(|f| f.run_id.clone());
        let this_frame = RecursionFrame {
            graph_id: graph.graph_id.clone(),
            node_id: graph.recursion_node.clone(),
            run_id: run_id.clone(),
            task_id: None,
            namespace: graph.namespace.clone(),
            depth: recursion.depth(),
            parent: parent_run_id.clone(),
        };
        if let Err(err) = recursion.push(this_frame) {
            graph.emit(GraphEvent::RunStarted {
                run_id: run_id.clone(),
            });
            graph
                .fail_run(&run_id, &thread_id, started_at, 0, &err, None)
                .await;
            return Err(err);
        }
        // Serialized once per run for embedding in every checkpoint's metadata.
        let recursion_meta =
            serde_json::to_value(recursion.frames()).unwrap_or(serde_json::Value::Null);
        let live_frames = recursion.frames().to_vec();
        let drop_guard = RunDropGuard::new(
            graph.status_store.clone(),
            run_id.clone(),
            thread_id.clone(),
            graph.graph_id.clone(),
            graph.namespace.clone(),
            started_at,
        );

        let ctx = Self {
            graph,
            run_id,
            thread_id,
            root_run_id,
            parent_run_id,
            started_at,
            started_instant,
            live_frames,
            recursion_meta,
            recursion,
            binding,
            child_sink: ChildRunSink::new(),
            node_visits: initial_node_visits,
            barrier_arrivals: initial_barriers,
            async_writes: AsyncCheckpointWrites::default(),
            resume_map,
            visited: Vec::new(),
            all_child_runs: Vec::new(),
            steps: initial_steps,
            last_checkpoint: None,
            parent_checkpoint: initial_parent,
            carried_completed,
            cancellation,
            drop_guard,
        };
        ctx.emit(GraphEvent::RunStarted {
            run_id: ctx.run_id.clone(),
        });
        // Surface this run's recursion depth so observers can attribute
        // nested runs without reconstructing the tree from logs.
        ctx.emit(GraphEvent::RecursionDepthChanged {
            depth: ctx.recursion.depth(),
        });
        Ok(ctx)
    }

    /// Drains this step's child-run sink into `all_child_runs` and returns
    /// its serialized form for embedding into this boundary's checkpoint
    /// metadata.
    pub(super) fn take_step_child_runs(&mut self) -> serde_json::Value {
        let step_child_runs = self.child_sink.drain();
        self.all_child_runs.extend(step_child_runs.iter().cloned());
        serde_json::to_value(&step_child_runs).unwrap_or(serde_json::Value::Null)
    }

    /// Builds the per-task [`NodeContext`] for `activation`, consuming its
    /// entry from `resume_map` (a task can only be handed its resume value
    /// once).
    ///
    /// `fork` carries the branch identity in a concurrent step (`None` in
    /// sequential mode or single-node steps). `siblings` is the number of
    /// activations of `activation.node` in this same step's active set
    /// (I1): more than one means a `Send` fan-out of the same node, which is
    /// what a subgraph node consults to namespace its child checkpoint by
    /// task id instead of sharing one namespace across every fan-out branch.
    ///
    /// Resume lookup prefers `resume_map`'s task-id key (I1/R5: distinguishes
    /// concurrent same-node activations) and falls back to the node-id key
    /// (legacy/whole-node resume, or a resume value fanned across every
    /// pending node with no interrupt provenance).
    pub(super) fn node_context(
        &mut self,
        activation: &Activation,
        step: usize,
        fork: Option<ForkId>,
        siblings: usize,
    ) -> NodeContext {
        let node_id = &activation.node;
        let resume = self
            .resume_map
            .remove(activation.task_id.as_str())
            .or_else(|| self.resume_map.remove(node_id.as_str()));
        NodeContext {
            graph_id: self.graph.graph_id.clone(),
            node_id: node_id.clone(),
            run_id: self.run_id.clone(),
            thread_id: self.thread_id.clone(),
            step,
            resume,
            fork,
            send_arg: activation.send_arg.clone(),
            root_run_id: Some(self.root_run_id.clone()),
            recursion_frames: self.live_frames.clone(),
            child_runs: Some(self.child_sink.clone()),
            agent_binding: self.binding.clone(),
            task_id: activation.task_id.clone(),
            siblings,
        }
    }
}
