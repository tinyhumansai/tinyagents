//! Public run/resume entry points and the superstep execution engine.
//!
//! Split out of `compiled/mod.rs`; see that module's doc comment for the
//! full executor design (superstep loop, concurrency, and resumable-failure
//! semantics).

use super::*;

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
        // Under `DurabilityMode::Async`, boundary checkpoint writes run on
        // spawned background tasks tracked here. Failures are surfaced at the
        // next durability boundary; every terminal path drains the tracker so
        // the run result reflects persistence failures (see
        // `AsyncCheckpointWrites`).
        let mut async_writes = AsyncCheckpointWrites::default();

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
            // Whole-run wall-clock deadline: stop *between* super-steps once the
            // elapsed run time reaches it, leaving the last committed boundary
            // checkpoint intact (unlike an external `tokio::time::timeout`, which
            // aborts mid-super-step and cannot). The already-completed super-steps
            // and their checkpoints are preserved; the run fails with `Timeout`.
            if let Some(deadline) = self.run_deadline {
                let elapsed = started_at.elapsed().unwrap_or_default();
                if elapsed >= deadline {
                    let err = TinyAgentsError::Timeout(format!(
                        "graph run exceeded its {deadline:?} deadline after {steps} super-step(s) \
                         ({elapsed:?} elapsed)"
                    ));
                    self.fail_run(&run_id, &thread_id, started_at, steps, &err, None)
                        .await;
                    return Err(err);
                }
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
                // Settle any in-flight Async background writes before the
                // failure-boundary persist so earlier boundaries are durable
                // when the run aborts. Like the persist error below, a
                // background write error must not replace the original node
                // error, so it is intentionally dropped here.
                let _ = async_writes.drain().await;
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
                // An interrupt hands control back to the caller expecting a
                // fully durable pause point: settle any in-flight Async
                // background writes first, failing the run if one was lost
                // (a broken lineage cannot be safely resumed from).
                if let Err(err) = async_writes.drain().await {
                    return self
                        .fail_and_return(&run_id, &thread_id, started_at, steps, err)
                        .await;
                }
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
            // written; `Sync`/`Async` persist every boundary. `Async` hands
            // non-terminal writes to background tasks instead of awaiting them
            // inline.
            let persist_now = match self.durability {
                DurabilityMode::Exit => next.is_empty(),
                DurabilityMode::Sync | DurabilityMode::Async => true,
            };
            // Async durability: surface any background write failure recorded
            // since the previous boundary. The run fails at the first
            // durability boundary that observes the loss rather than silently
            // continuing with a hole in its lineage.
            if let Some(err) = async_writes.take_failure().await {
                return self
                    .fail_and_return(&run_id, &thread_id, started_at, steps, err)
                    .await;
            }
            let terminal = next.is_empty();
            let checkpoint_id = if persist_now {
                let persisted = if matches!(self.durability, DurabilityMode::Async) && !terminal {
                    self.persist_checkpoint_nonblocking(
                        &mut async_writes,
                        &thread_id,
                        &run_id,
                        &state,
                        &next,
                        &completed_nodes,
                        &barrier_arrivals,
                        parent_checkpoint.clone(),
                        steps,
                        &recursion_meta,
                        &child_runs_meta,
                    )
                    .await
                } else {
                    // Terminal boundary: drain every in-flight background
                    // write first (the "final await at run end"), so a lost
                    // Async checkpoint fails the run instead of being
                    // swallowed. The final checkpoint itself is then written
                    // synchronously in every mode.
                    if terminal && let Err(err) = async_writes.drain().await {
                        return self
                            .fail_and_return(&run_id, &thread_id, started_at, steps, err)
                            .await;
                    }
                    self.persist_checkpoint(
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
                };
                match persisted {
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
            // Box each branch future behind a concrete `Send` bound. This keeps
            // the `buffer_unordered` rolling window below (used for a
            // `max_concurrency` bound) from requiring a higher-ranked `Send`
            // proof over the borrowed recursion frames, which the compiler
            // cannot discharge for the bare `async` blocks.
            let fut: std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<NodeResult<Update>>> + Send + '_>,
            > = Box::pin(async move {
                self.run_node_with_retry(&owned_node, &handler, state, ctx, step)
                    .await
            });
            futures.push(fut);
        }

        // Drive branches to completion, bounding in-flight count when configured.
        // With a bound, keep a rolling window of `limit` branches in flight
        // instead of fixed `join_all` chunks. A chunked join runs each chunk to
        // completion before starting the next, so a single slow branch
        // head-of-line blocks the whole chunk; the rolling window starts a new
        // branch as soon as *any* in-flight one finishes. `select_all` reports
        // which pending future completed; a parallel index Vec maps it back to
        // the branch's active-set position, so results are re-ordered into
        // deterministic order for the fold below.
        let results = match self.max_concurrency {
            Some(limit) if limit < futures.len() => {
                let total = futures.len();
                let mut slots: Vec<Option<Result<NodeResult<Update>>>> =
                    (0..total).map(|_| None).collect();
                let mut source = futures.into_iter().enumerate();
                let mut running = Vec::with_capacity(limit);
                let mut running_index = Vec::with_capacity(limit);
                for (index, fut) in source.by_ref().take(limit) {
                    running.push(fut);
                    running_index.push(index);
                }
                while !running.is_empty() {
                    let (result, completed, rest) = futures::future::select_all(running).await;
                    let index = running_index.remove(completed);
                    slots[index] = Some(result);
                    running = rest;
                    if let Some((index, fut)) = source.next() {
                        running.push(fut);
                        running_index.push(index);
                    }
                }
                slots
                    .into_iter()
                    .map(|slot| slot.expect("every branch produced a result"))
                    .collect::<Vec<_>>()
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
        let checkpoint = self.build_loop_checkpoint(
            thread,
            run_id,
            state,
            pending,
            completed_tasks,
            interrupts,
            barrier_arrivals,
            parent,
            step,
            source,
            recursion,
            child_runs,
        );
        let id = checkpointer.put(checkpoint).await?;
        self.emit(GraphEvent::CheckpointSaved {
            checkpoint_id: id.clone(),
        });
        Ok(Some(id))
    }

    /// Persists a boundary checkpoint without blocking the superstep loop
    /// ([`DurabilityMode::Async`]).
    ///
    /// The checkpoint id is minted up front and returned immediately so the
    /// loop keeps chaining lineage onto it, while the actual `put` (and the
    /// [`GraphEvent::CheckpointSaved`] emitted on its success) runs on a
    /// spawned background task tracked in `writes`.
    ///
    /// # Failure semantics
    ///
    /// A background write error is never dropped: it is recorded in `writes`
    /// and surfaced by the executor at the next durability boundary, or at the
    /// latest when the run drains all in-flight writes at its terminal /
    /// interrupt boundary — so the run result reflects persistence failures.
    /// Because the `CheckpointSaved` event is emitted from the background
    /// task, its ordering relative to subsequent step events is not
    /// deterministic under `Async` durability.
    ///
    /// Outside a tokio runtime there is nothing to spawn onto, so the write
    /// happens inline — degrading to [`DurabilityMode::Sync`] behavior.
    #[allow(clippy::too_many_arguments)]
    async fn persist_checkpoint_nonblocking(
        &self,
        writes: &mut AsyncCheckpointWrites,
        thread_id: &Option<ThreadId>,
        run_id: &RunId,
        state: &State,
        pending: &[Activation],
        completed_tasks: &[NodeId],
        barrier_arrivals: &HashMap<NodeId, HashSet<NodeId>>,
        parent: Option<String>,
        step: usize,
        recursion: &serde_json::Value,
        child_runs: &serde_json::Value,
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, thread_id) else {
            return Ok(None);
        };
        let checkpoint = self.build_loop_checkpoint(
            thread,
            run_id,
            state,
            pending,
            completed_tasks,
            Vec::new(),
            barrier_arrivals,
            parent,
            step,
            "loop",
            recursion,
            child_runs,
        );
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let checkpointer = Arc::clone(checkpointer);
                let sink = self.event_sink.clone();
                writes.push(handle.spawn(async move {
                    let id = checkpointer.put(checkpoint).await?;
                    if let Some(sink) = sink {
                        sink.emit(GraphEvent::CheckpointSaved {
                            checkpoint_id: id.clone(),
                        });
                    }
                    Ok(id)
                }));
                Ok(Some(id))
            }
            Err(_) => {
                let id = checkpointer.put(checkpoint).await?;
                self.emit(GraphEvent::CheckpointSaved {
                    checkpoint_id: id.clone(),
                });
                Ok(Some(id))
            }
        }
    }

    /// Builds the loop-boundary [`Checkpoint`] record shared by the sync and
    /// async persist paths, minting a fresh checkpoint id.
    #[allow(clippy::too_many_arguments)]
    fn build_loop_checkpoint(
        &self,
        thread: &ThreadId,
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
    ) -> Checkpoint<State> {
        Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id: next_checkpoint_id(),
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
        }
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
