//! Public run/resume entry points and the superstep execution engine.
//!
//! Split out of `compiled/mod.rs`; see that module's doc comment for the
//! full executor design (superstep loop, concurrency, and resumable-failure
//! semantics). The superstep loop itself is a thin wire-up over three
//! sibling modules: [`step`] runs a superstep's active node set and folds
//! its results ([`step::StepRunner`]), [`boundary`] applies the reducer,
//! routes, and persists a checkpoint at each of the three boundary shapes a
//! step can end at ([`boundary::StepBoundary`]), and [`resume`] loads a
//! checkpoint back into a fresh run. [`run_ctx::RunCtx`] carries the run
//! identity and bookkeeping all three share.

use super::*;

use crate::compiled::boundary::StepBoundary;
use crate::compiled::run_ctx::{ResumeSeed, RunCtx};
use crate::compiled::step::StepRunner;
use crate::thread_locks::ThreadLockMap;
use std::sync::OnceLock;

/// Default TTL for the durable execution lease claimed in [`CompiledGraph::execute`]
/// (C3/R4). Renewal is not wired up (a single run is expected to complete, or at
/// least reach its next boundary, well inside this window); a lease that
/// outlives its owning process by more than this is reclaimable by the next
/// claimant.
const THREAD_LEASE_TTL: Duration = Duration::from_secs(300);

/// Process-wide map of per-`(thread, namespace)` in-process execution locks
/// (C3/R4). Distinct from `delegation::run::thread_lock`'s map: that one
/// serializes delegation's own pre-`execute` checkpoint classification, this
/// one serializes the executor's run/resume/retry entry points themselves —
/// the gap the review's C3 finding describes (`executor.rs` took no lock of
/// its own). Keyed on `thread_id` *and* namespace so a parent run and a
/// subgraph run sharing a thread id never contend on each other's lock.
fn execution_lock_map() -> &'static ThreadLockMap {
    static LOCKS: OnceLock<ThreadLockMap> = OnceLock::new();
    LOCKS.get_or_init(|| ThreadLockMap::new("graph executor per-thread run lock"))
}

/// Builds the in-process lock map key for `thread_id` scoped to `namespace`.
/// `\u{1}` is not a legal thread-id or namespace-segment character in
/// practice and is used only as an internal separator, never persisted.
fn execution_lock_key(thread_id: &str, namespace: &[String]) -> String {
    let mut key = thread_id.to_string();
    for segment in namespace {
        key.push('\u{1}');
        key.push_str(segment);
    }
    key
}

/// Everything a fresh or resumed run is seeded with, bundled so
/// [`CompiledGraph::execute`]/[`CompiledGraph::execute_run`] take one
/// parameter instead of positional state/thread/resume/barrier/binding
/// arguments.
pub(super) struct RunSeed<State, Update> {
    pub(super) state: State,
    pub(super) active: Vec<Activation>,
    pub(super) thread_id: Option<ThreadId>,
    /// Keyed by task id, falling back to node id (I1/R5), so a `Send`
    /// fan-out of the same node can deliver each interrupted activation its
    /// own resume value.
    pub(super) resume_map: HashMap<String, serde_json::Value>,
    pub(super) barriers: HashMap<NodeId, HashSet<NodeId>>,
    pub(super) parent: Option<String>,
    pub(super) binding: Option<crate::subagent_node::AgentInvocationBinding>,
    /// Resume-only seeding (step/node-visit continuation, carried-forward
    /// mid-step completions) — see [`ResumeSeed`]. Left at its `Default`
    /// (empty/zero) for a fresh run.
    pub(super) resume_seed: ResumeSeed,
    /// Optional per-run options (I4 part 2) — currently the cooperative
    /// cancellation token, if the caller opted in via
    /// [`CompiledGraph::run_with_options`]/[`CompiledGraph::resume_with_options`].
    pub(super) options: RunOptions,
    pub(super) _update: std::marker::PhantomData<Update>,
}

impl<State, Update> RunSeed<State, Update> {
    pub(super) fn fresh(
        state: State,
        active: Vec<Activation>,
        thread_id: Option<ThreadId>,
    ) -> Self {
        Self {
            state,
            active,
            thread_id,
            resume_map: HashMap::new(),
            barriers: HashMap::new(),
            parent: None,
            binding: None,
            resume_seed: ResumeSeed::default(),
            options: RunOptions::default(),
            _update: std::marker::PhantomData,
        }
    }

    pub(super) fn with_binding(
        mut self,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Self {
        self.binding = Some(binding);
        self
    }

    pub(super) fn with_options(mut self, options: RunOptions) -> Self {
        self.options = options;
        self
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
        self.execute(RunSeed::fresh(
            state,
            vec![Activation::node(self.entry.clone())],
            None,
        ))
        .await
    }

    /// Runs the graph to completion (or to an interrupt/cancellation) without
    /// a thread, honoring `options` (I4 part 2) — currently a cooperative
    /// [`RunOptions::cancellation`] token checked at every superstep
    /// boundary and raced against that step's in-flight node handlers.
    ///
    /// Without a thread id, cancellation still stops the run and records a
    /// `Cancelled` status, but there is nothing to persist a resumable
    /// checkpoint against (checkpoints are keyed by thread), exactly like
    /// [`Self::run`].
    pub async fn run_with_options(
        &self,
        state: State,
        options: RunOptions,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            RunSeed::fresh(state, vec![Activation::node(self.entry.clone())], None)
                .with_options(options),
        )
        .await
    }

    /// Runs the graph under a thread id, honoring `options` (I4 part 2).
    ///
    /// This is the checkpointed counterpart to [`Self::run_with_options`]: a
    /// cancellation observed mid-run persists a resumable checkpoint naming
    /// the still-pending activations, so the run can be continued later with
    /// [`Self::resume`]/[`Self::retry`].
    pub async fn run_with_thread_options(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
        options: RunOptions,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            RunSeed::fresh(
                state,
                vec![Activation::node(self.entry.clone())],
                Some(thread_id.into()),
            )
            .with_options(options),
        )
        .await
    }

    /// Resumes a run from its latest checkpoint, honoring `options` (I4 part
    /// 2) — see [`Self::run_with_thread_options`].
    pub async fn resume_with_options(
        &self,
        thread_id: impl Into<ThreadId>,
        command: Command<Update>,
        options: RunOptions,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_inner(
            thread_id.into(),
            ResumeTarget::Latest,
            command,
            None,
            options,
        )
        .await
    }

    /// Runs one execution with an explicit host-bound recursive-agent binding.
    ///
    /// The binding is scoped to this run and descendants spawned from it; it is
    /// never retained by this reusable graph value.
    pub async fn run_with_agent_binding(
        &self,
        state: State,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            RunSeed::fresh(state, vec![Activation::node(self.entry.clone())], None)
                .with_binding(binding),
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
    /// [`NodeContext::send_arg`](crate::NodeContext::send_arg).
    pub async fn run_with_inputs(
        &self,
        state: State,
        inputs: impl IntoIterator<Item = GraphInput>,
    ) -> Result<GraphExecution<State>> {
        let active = self.initial_inputs(inputs)?;
        self.execute(RunSeed::fresh(state, active, None)).await
    }

    /// Runs the graph under a thread id, persisting checkpoints at every
    /// superstep boundary when a checkpointer is configured.
    pub async fn run_with_thread(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
    ) -> Result<GraphExecution<State>> {
        self.execute(RunSeed::fresh(
            state,
            vec![Activation::node(self.entry.clone())],
            Some(thread_id.into()),
        ))
        .await
    }

    /// Runs one threaded execution with an explicit recursive-agent binding.
    pub async fn run_with_thread_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            RunSeed::fresh(
                state,
                vec![Activation::node(self.entry.clone())],
                Some(thread_id.into()),
            )
            .with_binding(binding),
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
        self.execute(RunSeed::fresh(state, active, Some(thread_id.into())))
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

    /// Resumes an interrupted run with a host-bound recursive-agent binding.
    ///
    /// Like [`Self::resume`], this reloads the latest checkpoint for `thread_id`.
    /// The binding is scoped solely to the resumed execution and is propagated
    /// to every resumed node and nested subgraph; it is never retained by this
    /// reusable graph value.
    pub async fn resume_with_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        command: Command<Update>,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_with_agent_binding(thread_id, ResumeTarget::Latest, command, binding)
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

    /// Retries a failed run with a host-bound recursive-agent binding.
    ///
    /// This is the binding-aware counterpart to [`Self::retry`]. The supplied
    /// binding is available only to this retry and any descendants it spawns.
    pub async fn retry_with_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_with_agent_binding(
            thread_id,
            ResumeTarget::Latest,
            Command::new(),
            binding,
        )
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
        self.resume_from_inner(
            thread_id.into(),
            target,
            command,
            None,
            RunOptions::default(),
        )
        .await
    }

    /// Resumes a run from `target` with a host-bound recursive-agent binding.
    ///
    /// This is the binding-aware counterpart to [`Self::resume_from`]. It is
    /// useful when a durable continuation reaches a
    /// [`SubAgentNode`](crate::SubAgentNode) after an interrupt or retry.
    /// The binding remains execution-scoped, including for resumed nested
    /// subgraphs, and is not stored on [`CompiledGraph`](crate::CompiledGraph).
    pub async fn resume_from_with_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        target: ResumeTarget,
        command: Command<Update>,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_inner(
            thread_id.into(),
            target,
            command,
            Some(binding),
            RunOptions::default(),
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
                task_id: TaskId::from(String::new()),
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
    pub(super) async fn execute(
        &self,
        seed: RunSeed<State, Update>,
    ) -> Result<GraphExecution<State>> {
        let run_id = tinyagents_harness::ids::new_run_id();

        // C3/R4: hold this thread's execution lock for the run's whole
        // lifetime, in-process first (cheap, always available) then a
        // durable lease when a checkpointer is configured (cross-process).
        // Held across the entire `execute_run` below — including checkpoint
        // reads that would otherwise race a concurrent caller's — not just
        // around the write, which is what closes the interleaving the C3
        // finding describes (two concurrent `run_with_thread`/`resume` on
        // one thread id previously had nothing serializing them at this
        // layer at all).
        let _in_process_guard = if let Some(thread) = &seed.thread_id {
            let key = execution_lock_key(thread.as_str(), &self.namespace);
            Some(execution_lock_map().lock_for(&key).lock_owned().await)
        } else {
            None
        };
        let lease_owner =
            if let (Some(checkpointer), Some(thread)) = (&self.checkpointer, &seed.thread_id) {
                match checkpointer
                    .try_claim(thread.as_str(), run_id.as_str(), THREAD_LEASE_TTL)
                    .await
                {
                    Ok(true) => Some((checkpointer.clone(), thread.clone())),
                    Ok(false) => {
                        return Err(TinyAgentsError::Validation(format!(
                            "thread `{thread}` is leased by another run"
                        )));
                    }
                    // A lease-claim I/O error must not silently degrade to
                    // running unprotected: propagate it rather than proceeding
                    // as if the claim had succeeded.
                    Err(err) => return Err(err),
                }
            } else {
                None
            };

        // When a durable journal is configured, run against a clone whose event
        // sink wraps every emitted event into a `GraphObservation` and appends
        // it (while still forwarding to any pre-existing live sink). The journal
        // sink carries this graph's checkpoint namespace so subgraph runs record
        // their nested path. Default (no journal) leaves `self` untouched.
        let result = if self.journal.is_some() {
            let this = self.clone_with_journal_sink(&run_id, &seed.thread_id);
            this.execute_run(run_id.clone(), seed).await
        } else {
            self.execute_run(run_id.clone(), seed).await
        };

        if let Some((checkpointer, thread)) = lease_owner {
            let _ = checkpointer.release(thread.as_str(), run_id.as_str()).await;
        }
        result
    }

    /// Builds a clone whose `event_sink` is a [`JournalGraphSink`] for `run_id`,
    /// wrapping any existing sink as the live downstream. Returns a plain clone
    /// when no journal is configured.
    fn clone_with_journal_sink(&self, run_id: &RunId, thread_id: &Option<ThreadId>) -> Self {
        let Some(journal) = &self.journal else {
            return self.clone();
        };
        let mut sink = crate::observability::JournalGraphSink::new(
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

    /// Drives one run's superstep loop to completion, an interrupt, or a
    /// failure.
    ///
    /// Builds this run's [`RunCtx`] (identity, recursion stack, and the
    /// accumulators the loop carries forward) and its [`StepRunner`], then
    /// loops: check the recursion/deadline/visit-count guards, run the
    /// active set's node handlers ([`StepRunner::run_sequential`] or
    /// [`StepRunner::run_parallel`]), fold the results
    /// ([`StepRunner::fold_step`]), apply updates through the reducer
    /// ([`CompiledGraph::apply_updates`]), and dispatch to whichever
    /// boundary the step ended at — failure
    /// ([`CompiledGraph::handle_failure_boundary`]), interrupt
    /// ([`CompiledGraph::handle_interrupt_boundary`]), or the normal boundary
    /// ([`CompiledGraph::advance`], which returns the next active set).
    async fn execute_run(
        &self,
        run_id: RunId,
        seed: RunSeed<State, Update>,
    ) -> Result<GraphExecution<State>> {
        let RunSeed {
            mut state,
            active: initial_active,
            thread_id,
            resume_map,
            barriers: initial_barriers,
            parent: initial_parent,
            binding,
            resume_seed,
            options,
            ..
        } = seed;
        let cancellation = options.cancellation;

        let mut ctx = RunCtx::start(
            self,
            run_id,
            thread_id,
            resume_map,
            initial_barriers,
            initial_parent,
            binding,
            resume_seed,
            cancellation,
        )
        .await?;
        let runner = StepRunner { graph: self };

        // Record the run as live before the first superstep is scheduled.
        let mut running = ctx.base_status();
        running.active_nodes = activation_nodes(&initial_active);
        ctx.save_status(running).await;

        let mut active = initial_active;
        while !active.is_empty() {
            // I4 part 2: check cooperative cancellation at every superstep
            // boundary, before starting a new step. `active` at this point is
            // exactly what the next step would run, so a cancellation here
            // schedules the whole set as pending (nothing of this step has
            // executed yet).
            if ctx.is_cancelled() {
                return self.handle_cancel_boundary(&mut ctx, &active, &state).await;
            }

            let step = match self.begin_step(&mut ctx, &mut active).await {
                Ok(step) => step,
                Err(err) => return self.fail_and_return(&mut ctx, err).await,
            };

            let step_run = match self
                .run_step_with_cancel(&runner, &mut ctx, &active, &state, step)
                .await
            {
                Ok(Some(step_run)) => step_run,
                // Cancelled while this step's handlers were in flight: none
                // of them are trusted to have applied (the step's own future
                // was raced and abandoned, not awaited to completion), so the
                // whole active set is still pending.
                Ok(None) => return self.handle_cancel_boundary(&mut ctx, &active, &state).await,
                Err(err) => return self.fail_and_return(&mut ctx, err).await,
            };

            // Apply collected updates through the reducer at the boundary. A
            // reducer error here must still fail the run (not just unwind
            // leaving it `Running`).
            state = match self.apply_updates(state, step_run.updates) {
                Ok(state) => state,
                Err(err) => return self.fail_and_return(&mut ctx, err).await,
            };

            // Child runs spawned by subgraph nodes this step are embedded
            // into this boundary's checkpoint metadata (keyed by node) and
            // accumulated onto the final `GraphExecution`.
            let child_runs_meta = ctx.take_step_child_runs();
            let sb = StepBoundary {
                active: &active,
                completed: &step_run.completed,
                stalled: &step_run.stalled,
                goto_map: &step_run.goto_map,
                child_runs_meta: &child_runs_meta,
                step,
            };

            // Node-handler failure (survived any node-retry policy) or an
            // interrupt: both are terminal for this run, persisting a
            // resumable boundary checkpoint before returning.
            if let Some(fail) = step_run.failure {
                return self
                    .handle_failure_boundary(&mut ctx, sb, &state, fail)
                    .await;
            }
            if !step_run.interrupted.is_empty() {
                return self
                    .handle_interrupt_boundary(&mut ctx, sb, state, step_run.interrupted)
                    .await;
            }

            active = match self.advance(&mut ctx, sb, &state).await {
                Ok(next) => next,
                Err(err) => return self.fail_and_return(&mut ctx, err).await,
            };
        }

        Ok(self.finish_run(&mut ctx, state).await)
    }

    /// Runs one superstep, racing it against this run's cooperative
    /// cancellation token (I4 part 2) so a long-running node's handlers
    /// cannot indefinitely block a cancellation request once requested.
    ///
    /// Returns `Ok(Some(step_run))` when the step completed first,
    /// `Ok(None)` when the token was already cancelled or was cancelled
    /// while the step's handlers were still in flight (the step's own future
    /// is then dropped, abandoning it — see [`super::run_ctx::RunDropGuard`]'s
    /// doc for what that does and does not guarantee for any checkpoint
    /// write the abandoned step's handlers had already triggered), and
    /// `Err` for an ordinary step failure.
    async fn run_step_with_cancel(
        &self,
        runner: &StepRunner<'_, State, Update>,
        ctx: &mut RunCtx<'_, State, Update>,
        active: &[Activation],
        state: &State,
        step: usize,
    ) -> Result<Option<crate::compiled::step::StepRun<Update>>> {
        let Some(token) = ctx.cancellation.clone() else {
            return runner.run_step(ctx, active, state, step).await.map(Some);
        };
        if token.is_cancelled() {
            return Ok(None);
        }
        tokio::select! {
            biased;
            _ = token.cancelled() => Ok(None),
            result = runner.run_step(ctx, active, state, step) => result.map(Some),
        }
    }

    /// Checks the recursion-limit, wall-clock-deadline, and per-node
    /// visit-count guards for the next superstep, then advances `ctx.steps`,
    /// assigns any missing task ids in `active` (a failure checkpoint
    /// carries these with its pending activations, letting a later resume
    /// skip only the completed fan-out task), and emits `StepStarted`.
    /// Returns the step number on success.
    async fn begin_step(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        active: &mut [Activation],
    ) -> Result<usize> {
        // The effective step cap is the smaller of the builder's recursion
        // limit and the policy's `max_total_steps`, so a policy never
        // loosens an existing limit. Both surface a `RecursionLimit`.
        let step_limit = self
            .recursion_limit
            .min(self.recursion_policy.max_total_steps);
        if ctx.steps >= step_limit {
            return Err(TinyAgentsError::RecursionLimit(step_limit));
        }
        // Whole-run wall-clock deadline: stop *between* super-steps once the
        // elapsed run time reaches it, leaving the last committed boundary
        // checkpoint intact (unlike an external `tokio::time::timeout`, which
        // aborts mid-super-step and cannot). The already-completed super-steps
        // and their checkpoints are preserved; the run fails with `Timeout`.
        if let Some(deadline) = self.run_deadline {
            let elapsed = ctx.started_instant.elapsed();
            if elapsed >= deadline {
                return Err(TinyAgentsError::Timeout(format!(
                    "graph run exceeded its {deadline:?} deadline after {} super-step(s) \
                     ({elapsed:?} elapsed)",
                    ctx.steps
                )));
            }
        }
        // Node-loop recursion: enforce `max_visits_per_node` per activation.
        for activation in active.iter() {
            ctx.recursion
                .record_node_visit(&mut ctx.node_visits, &activation.node)?;
        }
        ctx.steps += 1;
        for (index, activation) in active.iter_mut().enumerate() {
            if activation.task_id.as_str().is_empty() {
                activation.task_id =
                    TaskId::from(format!("{}:{}:{}", ctx.steps, index, activation.node));
            }
        }
        ctx.emit(GraphEvent::StepStarted {
            step: ctx.steps,
            active: activation_nodes(active),
        });
        Ok(ctx.steps)
    }

    /// Builds the terminal [`GraphExecution`] for a run that emptied its
    /// active set without interrupting or failing: records a `Completed`
    /// status and emits `RunCompleted`.
    async fn finish_run(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        state: State,
    ) -> GraphExecution<State> {
        ctx.disarm_drop_guard();
        let mut status = ctx.base_status();
        status.status = ExecutionStatus::Completed;
        status.current_step = ctx.steps;
        status.checkpoint_id = ctx.last_checkpoint.clone();
        status.ended_at = Some(SystemTime::now());
        ctx.save_status(status.clone()).await;
        ctx.emit(GraphEvent::RunCompleted {
            run_id: ctx.run_id.clone(),
            steps: ctx.steps,
        });

        GraphExecution {
            state,
            run_id: ctx.run_id.clone(),
            graph_id: self.graph_id.clone(),
            root_run_id: ctx.root_run_id.clone(),
            parent_run_id: ctx.parent_run_id.clone(),
            child_runs: std::mem::take(&mut ctx.all_child_runs),
            visited: std::mem::take(&mut ctx.visited),
            steps: ctx.steps,
            interrupts: Vec::new(),
            status,
            checkpoint_id: ctx.last_checkpoint.clone(),
        }
    }
}
