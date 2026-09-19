//! The step boundary: applying reducer updates, routing a completed step's
//! active set into the next superstep, persisting checkpoints, and the
//! failure/interrupt boundaries that pause or abort a run.
//!
//! Routing itself (`route_completed`, static/conditional/`Send` resolution,
//! barrier gating) stays in `routing.rs`; this module is what calls it at
//! each of the three boundary shapes a superstep can end at — the normal
//! boundary ([`CompiledGraph::advance`]), a node-handler failure
//! ([`CompiledGraph::handle_failure_boundary`]), and an interrupt
//! ([`CompiledGraph::handle_interrupt_boundary`]) — plus the run-abort
//! bookkeeping ([`CompiledGraph::fail_run`], [`CompiledGraph::fail_and_return`])
//! shared by every early-exit path in `execute_run`.

use super::*;

use crate::compiled::run_ctx::RunCtx;

/// The step data a boundary persist needs beyond the (possibly narrowed)
/// pending/completed activation slices: the committed state snapshot and
/// this step's child-run metadata. Bundled so the persist helpers below stay
/// under the arity that would otherwise need `#[allow(too_many_arguments)]`.
pub(super) struct BoundaryCheckpoint<'a, State> {
    pub(super) state: &'a State,
    pub(super) pending: &'a [Activation],
    pub(super) completed_tasks: &'a [Activation],
    /// Explicit `Command::goto` routing for each entry of
    /// `completed_tasks`, positionally aligned (R1: see
    /// [`crate::checkpoint::Checkpoint::completed_routes`]).
    pub(super) completed_routes: &'a [Vec<RouteTarget>],
    pub(super) child_runs: &'a serde_json::Value,
}

/// The transient data one superstep's boundary handling needs: the step's
/// active set, its fold outcome (`completed`/`stalled`, per [`StepRun`]),
/// its folded routing (`goto_map`), the step's child-run metadata, and the
/// step number. Shared by the normal, failure, and interrupt boundaries so
/// none of them re-take these as separate parameters.
pub(super) struct StepBoundary<'a> {
    pub(super) active: &'a [Activation],
    /// Every branch of this step that completed, original-index-paired (see
    /// [`crate::compiled::step::StepRun::completed`]) — used both to build
    /// the persisted `completed_tasks` and, at the normal boundary, as the
    /// routing input.
    pub(super) completed: &'a [(usize, Activation)],
    /// Every branch of this step that errored or interrupted — the
    /// `pending` set at a failure/interrupt boundary.
    pub(super) stalled: &'a [(usize, Activation)],
    pub(super) goto_map: &'a HashMap<usize, Vec<RouteTarget>>,
    pub(super) child_runs_meta: &'a serde_json::Value,
    pub(super) step: usize,
}

impl<State, Update> CompiledGraph<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    /// Applies each collected update through the reducer, in order, at the
    /// step boundary. A reducer error here must still fail the run (not just
    /// unwind leaving it `Running`) — surfaced to the caller as `Err`.
    pub(super) fn apply_updates(&self, mut state: State, updates: Vec<Update>) -> Result<State> {
        for update in updates {
            state = self.reducer.apply(state, update)?;
        }
        Ok(state)
    }

    /// The normal (non-interrupt/non-failure) step boundary: routes the
    /// completed active set into the next superstep's activations and
    /// persists a boundary checkpoint per the configured
    /// [`DurabilityMode`], updating `ctx.last_checkpoint`/`parent_checkpoint`
    /// when one is written. Returns the next active set.
    ///
    /// When this run was resumed from a mid-step checkpoint (an
    /// interrupt/failure boundary whose completed siblings were never
    /// routed — see [`Self::handle_interrupt_boundary`] /
    /// [`Self::handle_failure_boundary`]), `ctx.carried_completed` carries
    /// those siblings' node ids forward. The *first* `advance` call of the
    /// resumed run consumes it (`take`) and routes it together with this
    /// step's own `sb.completed`, so every branch of the original step is
    /// routed in one pass against one committed state — matching what an
    /// uninterrupted run would have done (the C2 fix). Those carried
    /// branches have no persisted `goto_map` entry (a `Command`'s explicit
    /// `goto` is not durable across the boundary), so they route via
    /// static/conditional edges only; see the module and `RunCtx` docs.
    pub(super) async fn advance(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        sb: StepBoundary<'_>,
        state: &State,
    ) -> Result<Vec<Activation>> {
        // Select the next active set from commands or static/conditional
        // edges, evaluated against the freshly-committed state. Barrier
        // arrivals accumulate into `ctx.barrier_arrivals` (persisted below).
        let carried = ctx.carried_completed.take();
        let completed_tasks: Vec<Activation>;
        let next = match &carried {
            Some(carried_completions) => {
                // Reserve an index range that cannot collide with `sb`'s own
                // (0-based) active-set indices, so `goto_map.get(&index)`
                // correctly misses for every carried entry instead of
                // aliasing onto this step's own routing.
                let offset = sb.active.len().max(sb.completed.len()) + 1;
                let mut pairs: Vec<(usize, Activation)> = carried_completions
                    .iter()
                    .enumerate()
                    .map(|(i, (node, _))| (offset + i, Activation::node(node.clone())))
                    .collect();
                pairs.extend(sb.completed.iter().cloned());
                // Merge in each carried completion's persisted `goto` (R1):
                // without this, a completed sibling's explicit
                // `Command::goto` is lost across the boundary and it
                // re-resolves via static/conditional edges only.
                let mut merged_goto_map = sb.goto_map.clone();
                for (i, (_, goto)) in carried_completions.iter().enumerate() {
                    if !goto.is_empty() {
                        merged_goto_map.insert(offset + i, goto.clone());
                    }
                }
                let next = self.route_completed(
                    &pairs,
                    &merged_goto_map,
                    state,
                    &mut ctx.barrier_arrivals,
                )?;
                completed_tasks = pairs.into_iter().map(|(_, a)| a).collect();
                next
            }
            None => {
                completed_tasks = sb.completed.iter().map(|(_, a)| a.clone()).collect();
                self.route_completed(sb.completed, sb.goto_map, state, &mut ctx.barrier_arrivals)?
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
        if let Some(err) = ctx.async_writes.take_failure().await {
            return Err(err);
        }
        let terminal = next.is_empty();
        let checkpoint_id = if persist_now {
            let boundary = BoundaryCheckpoint {
                state,
                pending: &next,
                completed_tasks: &completed_tasks,
                // Fully routed at this normal boundary, so nothing is left
                // to carry forward.
                completed_routes: &[],
                child_runs: sb.child_runs_meta,
            };
            if matches!(self.durability, DurabilityMode::Async) && !terminal {
                self.persist_checkpoint_nonblocking(ctx, boundary, sb.step)
                    .await?
            } else {
                // Terminal boundary: drain every in-flight background write
                // first (the "final await at run end"), so a lost Async
                // checkpoint fails the run instead of being swallowed. The
                // final checkpoint itself is then written synchronously in
                // every mode.
                if terminal {
                    ctx.async_writes.drain().await?;
                }
                self.persist_checkpoint(ctx, boundary, sb.step, Vec::new(), &[])
                    .await?
            }
        } else {
            None
        };
        if let Some(id) = &checkpoint_id {
            ctx.last_checkpoint = Some(id.clone());
            ctx.parent_checkpoint = Some(id.to_string());
        }

        ctx.emit(GraphEvent::StepCompleted { step: sb.step });
        Ok(next)
    }

    /// The failure boundary: a node-handler failure that survived the
    /// node-retry policy.
    ///
    /// The updates of every branch that completed this step — regardless of
    /// its index relative to the failed one — are already folded into
    /// `state` (by [`Self::apply_updates`] before this is called, from
    /// [`crate::compiled::step::StepRun::updates`]). This boundary does
    /// *not* route those completed branches yet (see [`Self::advance`]'s
    /// `carried_completed` doc): routing them now, before the failed/pending
    /// branches are known, would let their successors observe a state that
    /// omits whatever those pending branches eventually write — the exact
    /// same-superstep ordering bug C2 describes for the interrupt boundary.
    /// Instead `pending` is exactly `sb.stalled` (the failed node plus any
    /// other branch that also errored/interrupted this step — the
    /// not-yet-run set, not `active[failed_index..]`), and the completed
    /// branches' node ids are stamped into the checkpoint's
    /// `completed_tasks` (merged with any already-carried-forward ones from
    /// an earlier resume of this same logical step) so a resuming
    /// `retry`/`resume` can route the whole step together once the pending
    /// branches finish. Persists a resumable failure-boundary checkpoint,
    /// records a `Failed` status carrying the error and that checkpoint, and
    /// returns the error. Without a checkpointer/thread the checkpoint is a
    /// no-op and the run aborts exactly as before.
    pub(super) async fn handle_failure_boundary(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        sb: StepBoundary<'_>,
        state: &State,
        fail: StepFailure,
    ) -> Result<GraphExecution<State>> {
        ctx.disarm_drop_guard();
        let StepFailure {
            failed_index,
            error,
        } = fail;
        let failed_node = sb.active[failed_index].node.clone();
        let pending: Vec<Activation> = sb.stalled.iter().map(|(_, a)| a.clone()).collect();
        let (completed_tasks, completed_routes) =
            self.merged_completed(ctx, sb.completed, sb.goto_map);
        // Settle any in-flight Async background writes before the
        // failure-boundary persist so earlier boundaries are durable when
        // the run aborts. Like the persist error below, a background write
        // error must not replace the original node error, so it is
        // intentionally dropped here.
        let _ = ctx.async_writes.drain().await;
        // A failure-boundary persist error must not replace the original
        // node error: keep reporting the node error and just drop the
        // resumable checkpoint reference.
        let checkpoint_id = self
            .persist_failure_checkpoint(
                ctx,
                BoundaryCheckpoint {
                    state,
                    pending: &pending,
                    completed_tasks: &completed_tasks,
                    completed_routes: &completed_routes,
                    child_runs: sb.child_runs_meta,
                },
                sb.step,
                &failed_node,
                &error,
            )
            .await
            .unwrap_or(None);
        self.fail_run(
            &ctx.run_id,
            &ctx.thread_id,
            ctx.started_at,
            sb.step,
            &error,
            checkpoint_id,
        )
        .await;
        Err(error)
    }

    /// The interrupt boundary: persists a checkpoint whose pending
    /// activations are the successors of the branches that completed before
    /// the interrupt (their routing must survive) followed by the
    /// not-yet-completed members of this step (interrupted node first).
    /// Each pending branch keeps its `Send` arg; accumulated barrier
    /// arrivals are persisted too. Returns control to the caller.
    ///
    /// `interrupted` is every branch of this step whose result was an
    /// interrupt (I1), in ascending active-set index order — a `Send`
    /// fan-out of one subgraph node interrupting on every one of its
    /// concurrent activations, for example, surfaces all of them here rather
    /// than only the (arbitrarily chosen) lowest-index one. Each is stamped
    /// with its own branch's task id before being persisted/returned, so the
    /// caller's subsequent `resume` can address each individually (see
    /// `resume_from_inner`'s `resume_map` / `Command::resume_tasks`).
    pub(super) async fn handle_interrupt_boundary(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        sb: StepBoundary<'_>,
        state: State,
        interrupted: Vec<(usize, Interrupt)>,
    ) -> Result<GraphExecution<State>> {
        ctx.disarm_drop_guard();
        if let Err(err) = self.require_interrupt_durability(&ctx.thread_id) {
            return self.fail_and_return(ctx, err).await;
        }
        let stamped: Vec<Interrupt> = interrupted
            .into_iter()
            .map(|(index, interrupt)| interrupt.with_task_id(sb.active[index].task_id.clone()))
            .collect();
        // Deferred routing, same as the failure boundary above: the
        // completed siblings (whichever side of the interrupted branches
        // they fall on) are not routed here. `pending` is exactly
        // `sb.stalled` (every interrupted branch — no error can be mixed in
        // here, since the executor dispatches a step with any failure to
        // `handle_failure_boundary` first), and `completed_tasks` carries
        // every completed node id forward (merged with anything already
        // carried from an earlier resume of this step) for `advance` to
        // route once the pending set finishes.
        let pending: Vec<Activation> = sb.stalled.iter().map(|(_, a)| a.clone()).collect();
        let (completed_tasks, completed_routes) =
            self.merged_completed(ctx, sb.completed, sb.goto_map);
        let pending_nodes = activation_nodes(&pending);
        let interrupt_ids: Vec<InterruptId> = stamped
            .iter()
            .map(|i| InterruptId::new(i.id.clone()))
            .collect();
        // An interrupt hands control back to the caller expecting a fully
        // durable pause point: settle any in-flight Async background writes
        // first, failing the run if one was lost (a broken lineage cannot
        // be safely resumed from).
        if let Err(err) = ctx.async_writes.drain().await {
            return self.fail_and_return(ctx, err).await;
        }
        let checkpoint_id = match self
            .persist_checkpoint(
                ctx,
                BoundaryCheckpoint {
                    state: &state,
                    pending: &pending,
                    completed_tasks: &completed_tasks,
                    completed_routes: &completed_routes,
                    child_runs: sb.child_runs_meta,
                },
                sb.step,
                stamped.clone(),
                &pending_nodes,
            )
            .await
        {
            Ok(id) => id,
            Err(persist_err) => return self.fail_and_return(ctx, persist_err).await,
        };

        let mut status = ctx.base_status();
        status.status = ExecutionStatus::Interrupted;
        status.current_step = sb.step;
        status.active_nodes = pending_nodes;
        status.pending_interrupts = interrupt_ids;
        status.checkpoint_id = checkpoint_id.clone();
        ctx.save_status(status.clone()).await;

        Ok(GraphExecution {
            state,
            run_id: ctx.run_id.clone(),
            graph_id: self.graph_id.clone(),
            root_run_id: ctx.root_run_id.clone(),
            parent_run_id: ctx.parent_run_id.clone(),
            child_runs: std::mem::take(&mut ctx.all_child_runs),
            visited: std::mem::take(&mut ctx.visited),
            steps: sb.step,
            interrupts: stamped,
            status,
            checkpoint_id,
        })
    }

    /// The cancellation boundary (I4 part 2): the run's cooperative
    /// cancellation token was observed cancelled, either between supersteps
    /// or while this step's node handlers were still in flight and had to be
    /// abandoned (raced against the token via
    /// [`super::executor::CompiledGraph::run_step_with_cancel`]).
    ///
    /// Unlike the failure/interrupt boundaries, nothing from this step is
    /// trusted to have completed — a mid-step cancellation abandons the
    /// step's future rather than awaiting it to a folded result — so `active`
    /// (exactly what the next superstep would have run) is persisted whole
    /// as the resumable checkpoint's pending set, mirroring how the failure
    /// boundary reuses the checkpoint machinery. Persists a resumable
    /// checkpoint (on a checkpointed thread), records a `Cancelled` status,
    /// and returns `Ok` (cancellation is a normal, requested outcome, not an
    /// error) carrying no interrupts.
    pub(super) async fn handle_cancel_boundary(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        active: &[Activation],
        state: &State,
    ) -> Result<GraphExecution<State>> {
        ctx.disarm_drop_guard();
        // Settle in-flight Async background writes before persisting the
        // cancellation checkpoint, same as the failure boundary — best
        // effort, since a lost background write here must not turn a
        // successfully-requested cancellation into a hard error.
        let _ = ctx.async_writes.drain().await;
        let checkpoint_id = self
            .persist_cancel_checkpoint(ctx, state, active)
            .await
            .unwrap_or(None);

        let mut status = ctx.base_status();
        status.status = ExecutionStatus::Cancelled;
        status.current_step = ctx.steps;
        status.active_nodes = activation_nodes(active);
        status.checkpoint_id = checkpoint_id.clone();
        status.ended_at = Some(SystemTime::now());
        ctx.save_status(status.clone()).await;
        ctx.emit(GraphEvent::RunCancelled {
            run_id: ctx.run_id.clone(),
        });

        Ok(GraphExecution {
            state: state.clone(),
            run_id: ctx.run_id.clone(),
            graph_id: self.graph_id.clone(),
            root_run_id: ctx.root_run_id.clone(),
            parent_run_id: ctx.parent_run_id.clone(),
            child_runs: std::mem::take(&mut ctx.all_child_runs),
            visited: std::mem::take(&mut ctx.visited),
            steps: ctx.steps,
            interrupts: Vec::new(),
            status,
            checkpoint_id,
        })
    }

    /// Persists a resumable cancellation-boundary checkpoint, mirroring
    /// [`Self::persist_failure_checkpoint`]: `next_nodes` schedules exactly
    /// the activations that were still pending when the cancellation was
    /// observed, so `resume`/`retry` re-runs exactly what did not complete.
    /// A no-op returning `None` without a checkpointer/thread, exactly like
    /// the failure boundary.
    async fn persist_cancel_checkpoint(
        &self,
        ctx: &RunCtx<'_, State, Update>,
        state: &State,
        pending: &[Activation],
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, &ctx.thread_id) else {
            return Ok(None);
        };
        let checkpoint = Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id: next_checkpoint_id(),
            run_id: Some(ctx.run_id.to_string()),
            parent_checkpoint_id: ctx.parent_checkpoint.clone(),
            namespace: self.namespace.clone(),
            state: state.clone(),
            next_nodes: activation_nodes(pending),
            completed_tasks: Vec::new(),
            completed_routes: Vec::new(),
            pending_writes: Vec::new(),
            interrupts: Vec::new(),
            pending_activations: Some(pending.iter().map(PendingActivation::from).collect()),
            barrier_arrivals: barriers_to_persisted(&ctx.barrier_arrivals),
            metadata: serde_json::json!({
                "source": "loop",
                "step": ctx.steps,
                "recursion": ctx.recursion_meta,
                "cancelled": true,
                "node_visits": node_visits_to_json(&ctx.node_visits),
            }),
        };
        let id = checkpointer.put(checkpoint).await?;
        self.emit(GraphEvent::CheckpointSaved {
            checkpoint_id: id.clone(),
        });
        Ok(Some(id))
    }

    /// Emits a [`GraphEvent::RunFailed`] and records a terminal `Failed`
    /// status for a run that aborted with `err`.
    ///
    /// `checkpoint_id` is the resumable failure-boundary checkpoint when the
    /// run left one (a node-handler failure on a checkpointed thread), or
    /// `None` for a structural/non-resumable abort. When present it is
    /// recorded on the status so an observer can locate the checkpoint to
    /// `resume`/`retry` from.
    pub(super) async fn fail_run(
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

    /// Records a terminal `Failed` status for `err` (via [`Self::fail_run`],
    /// reading identity/timing off `ctx`) and returns it as `Err`.
    ///
    /// Used at every early-exit path in `execute_run` — a guard trip, a
    /// node-runner error, a reducer merge, a routing resolution, or a
    /// checkpoint persist — so the run transitions to `Failed` (rather than
    /// leaving observers to see it stuck in `Running` forever) before the
    /// error unwinds out of the run.
    ///
    /// Any in-flight `Async` background write is drained first: dropping the
    /// tracker would detach those tasks, discarding their outcome (contrary
    /// to [`AsyncCheckpointWrites`]' contract) and racing a caller that
    /// immediately `retry`s the thread. A background write error must not
    /// replace the error that aborted the run, so it is dropped here.
    pub(super) async fn fail_and_return<T>(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        err: TinyAgentsError,
    ) -> Result<T> {
        ctx.disarm_drop_guard();
        let _ = ctx.async_writes.drain().await;
        self.fail_run(
            &ctx.run_id,
            &ctx.thread_id,
            ctx.started_at,
            ctx.steps,
            &err,
            None,
        )
        .await;
        Err(err)
    }

    /// Persists a resumable failure-boundary checkpoint for a node-handler
    /// failure that survived the node-retry policy.
    ///
    /// Mirrors the interrupt boundary: `next_nodes` schedules the failed
    /// node (and any not-yet-run members of the step) so `resume`/`retry`
    /// re-runs exactly what did not complete, while `completed_tasks`
    /// records the branches that already succeeded (their updates are
    /// folded into `state` before this is called). The rendered error and
    /// failed node id are stamped into the checkpoint metadata for
    /// diagnosis. A no-op returning `None` when no checkpointer/thread is
    /// configured — the run then aborts without a resumable checkpoint,
    /// exactly as before this policy existed.
    async fn persist_failure_checkpoint(
        &self,
        ctx: &RunCtx<'_, State, Update>,
        boundary: BoundaryCheckpoint<'_, State>,
        step: usize,
        failed_node: &NodeId,
        error: &TinyAgentsError,
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, &ctx.thread_id) else {
            return Ok(None);
        };
        let checkpoint = Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id: next_checkpoint_id(),
            run_id: Some(ctx.run_id.to_string()),
            parent_checkpoint_id: ctx.parent_checkpoint.clone(),
            namespace: self.namespace.clone(),
            state: boundary.state.clone(),
            next_nodes: activation_nodes(boundary.pending),
            completed_tasks: activation_nodes(boundary.completed_tasks),
            completed_routes: boundary.completed_routes.to_vec(),
            pending_writes: Self::completion_writes(boundary.completed_tasks),
            interrupts: Vec::new(),
            pending_activations: Some(
                boundary
                    .pending
                    .iter()
                    .map(PendingActivation::from)
                    .collect(),
            ),
            barrier_arrivals: barriers_to_persisted(&ctx.barrier_arrivals),
            metadata: serde_json::json!({
                "source": "loop",
                "step": step,
                "recursion": ctx.recursion_meta,
                "child_runs": boundary.child_runs,
                "failed_node": failed_node.as_str(),
                "error": error.to_string(),
                "node_visits": node_visits_to_json(&ctx.node_visits),
            }),
        };
        let writes = checkpoint.pending_writes.clone();
        let config = CheckpointConfig {
            thread_id: checkpoint.thread_id.clone(),
            checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
            namespace: checkpoint.namespace.clone(),
        };
        let id = checkpointer.put(checkpoint).await?;
        // Also record the ledger through the write protocol, so backends
        // that implement it can answer "did this task run?" without loading
        // the whole state payload.
        checkpointer.put_writes(&config, &writes).await?;
        self.emit(GraphEvent::CheckpointSaved {
            checkpoint_id: id.clone(),
        });
        Ok(Some(id))
    }

    /// Persists a loop-boundary checkpoint (the normal step boundary, or an
    /// interrupt boundary when `interrupts`/`interrupted` are non-empty).
    async fn persist_checkpoint(
        &self,
        ctx: &RunCtx<'_, State, Update>,
        boundary: BoundaryCheckpoint<'_, State>,
        step: usize,
        interrupts: Vec<Interrupt>,
        interrupted: &[NodeId],
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, &ctx.thread_id) else {
            return Ok(None);
        };
        let checkpoint =
            self.build_loop_checkpoint(ctx, thread, boundary, step, interrupts, interrupted);
        let writes = checkpoint.pending_writes.clone();
        let config = CheckpointConfig {
            thread_id: checkpoint.thread_id.clone(),
            checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
            namespace: checkpoint.namespace.clone(),
        };
        let id = checkpointer.put(checkpoint).await?;
        checkpointer.put_writes(&config, &writes).await?;
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
    /// spawned background task tracked in `ctx.async_writes`.
    ///
    /// # Failure semantics
    ///
    /// A background write error is never dropped: it is recorded in
    /// `ctx.async_writes` and surfaced by the executor at the next
    /// durability boundary, or at the latest when the run drains all
    /// in-flight writes at its terminal/interrupt boundary — so the run
    /// result reflects persistence failures. Because the `CheckpointSaved`
    /// event is emitted from the background task, its ordering relative to
    /// subsequent step events is not deterministic under `Async` durability.
    ///
    /// Outside a tokio runtime there is nothing to spawn onto, so the write
    /// happens inline — degrading to [`DurabilityMode::Sync`] behavior.
    async fn persist_checkpoint_nonblocking(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        boundary: BoundaryCheckpoint<'_, State>,
        step: usize,
    ) -> Result<Option<CheckpointId>> {
        let (Some(checkpointer), Some(thread)) = (&self.checkpointer, &ctx.thread_id) else {
            return Ok(None);
        };
        let thread = thread.clone();
        let checkpoint = self.build_loop_checkpoint(ctx, &thread, boundary, step, Vec::new(), &[]);
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        // M5: mirror the synchronous path, which persists both the state
        // record (`put`) and the write ledger (`put_writes`). Without this
        // the ledger tooling sees no completion markers for any checkpoint
        // written under `DurabilityMode::Async`.
        let writes = checkpoint.pending_writes.clone();
        let write_config = CheckpointConfig {
            thread_id: checkpoint.thread_id.clone(),
            checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
            namespace: checkpoint.namespace.clone(),
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let checkpointer = Arc::clone(checkpointer);
                let sink = self.event_sink.clone();
                ctx.async_writes.spawn_ordered(&handle, async move {
                    let id = checkpointer.put(checkpoint).await?;
                    checkpointer.put_writes(&write_config, &writes).await?;
                    if let Some(sink) = sink {
                        sink.emit(GraphEvent::CheckpointSaved {
                            checkpoint_id: id.clone(),
                        });
                    }
                    Ok(id)
                });
                Ok(Some(id))
            }
            Err(_) => {
                let id = checkpointer.put(checkpoint).await?;
                checkpointer.put_writes(&write_config, &writes).await?;
                self.emit(GraphEvent::CheckpointSaved {
                    checkpoint_id: id.clone(),
                });
                Ok(Some(id))
            }
        }
    }

    /// Builds the `completed_tasks`/`completed_routes` pair for a
    /// failure/interrupt boundary checkpoint: this step's own completed
    /// branches (`sb.completed`, with their `goto_map` routing captured
    /// positionally), prefixed by any node ids (and their persisted
    /// `goto`) already carried forward from an earlier interrupt/failure of
    /// this *same* logical step (`ctx.carried_completed` — set once, at
    /// resume, from the loaded checkpoint's `completed_tasks`/
    /// `completed_routes`, and left untouched here; only [`Self::advance`]
    /// consumes it, once the step finally finishes routing). This is what
    /// lets a step interrupt or fail more than once across repeated resumes
    /// without losing track of which of its branches have already
    /// completed, or what they explicitly routed to (R1).
    fn merged_completed(
        &self,
        ctx: &RunCtx<'_, State, Update>,
        completed: &[(usize, Activation)],
        goto_map: &HashMap<usize, Vec<RouteTarget>>,
    ) -> (Vec<Activation>, Vec<Vec<RouteTarget>>) {
        let mut tasks: Vec<Activation> = Vec::new();
        let mut routes: Vec<Vec<RouteTarget>> = Vec::new();
        if let Some(carried) = &ctx.carried_completed {
            for (node, goto) in carried {
                tasks.push(Activation::node(node.clone()));
                routes.push(goto.clone());
            }
        }
        for (index, activation) in completed {
            tasks.push(activation.clone());
            routes.push(goto_map.get(index).cloned().unwrap_or_default());
        }
        (tasks, routes)
    }

    /// Records completion markers for the tasks that finished in the step a
    /// boundary checkpoint closes.
    ///
    /// A graph's `Update` carries no `Serialize` bound, so the executor
    /// cannot persist *what* a task wrote — but it does not need to: the
    /// applied value is already durable in the checkpoint's `state`. What
    /// was missing was the other half, the per-task record of *that* it
    /// ran, which is what lets a resume distinguish "already done" from
    /// "not yet started". See [`PendingWrite`](crate::checkpoint::PendingWrite)'s
    /// docs for why that distinction is the whole point of the ledger.
    ///
    /// The task id is persisted on the activation itself, so a resume can
    /// match a marker to one fan-out task rather than every task with its
    /// node.
    fn completion_writes(completed_tasks: &[Activation]) -> Vec<crate::checkpoint::PendingWrite> {
        completed_tasks
            .iter()
            .map(|activation| {
                crate::checkpoint::PendingWrite::completion_marker(
                    activation.node.clone(),
                    activation.task_id.clone(),
                )
            })
            .collect()
    }

    /// Builds the loop-boundary [`Checkpoint`] record shared by the sync and
    /// async persist paths, minting a fresh checkpoint id.
    fn build_loop_checkpoint(
        &self,
        ctx: &RunCtx<'_, State, Update>,
        thread: &ThreadId,
        boundary: BoundaryCheckpoint<'_, State>,
        step: usize,
        interrupts: Vec<Interrupt>,
        interrupted: &[NodeId],
    ) -> Checkpoint<State> {
        let mut metadata = serde_json::json!({
            "source": "loop",
            "step": step,
            "recursion": ctx.recursion_meta,
            "child_runs": boundary.child_runs,
            "node_visits": node_visits_to_json(&ctx.node_visits),
        });
        // Which node of *this* graph paused, as opposed to the (possibly
        // re-emitted, child-owned) `Interrupt::node`. Resume keys the resume
        // value on it; omitted entirely when nothing interrupted.
        if !interrupted.is_empty() {
            metadata["interrupted_nodes"] = serde_json::json!(
                interrupted
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
            );
        }
        Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id: next_checkpoint_id(),
            run_id: Some(ctx.run_id.to_string()),
            parent_checkpoint_id: ctx.parent_checkpoint.clone(),
            namespace: self.namespace.clone(),
            state: boundary.state.clone(),
            next_nodes: activation_nodes(boundary.pending),
            completed_tasks: activation_nodes(boundary.completed_tasks),
            completed_routes: boundary.completed_routes.to_vec(),
            pending_writes: Self::completion_writes(boundary.completed_tasks),
            pending_activations: Some(
                boundary
                    .pending
                    .iter()
                    .map(PendingActivation::from)
                    .collect(),
            ),
            barrier_arrivals: barriers_to_persisted(&ctx.barrier_arrivals),
            interrupts,
            metadata,
        }
    }

    pub(super) fn base_status(
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

    /// Best-effort status write; never aborts the run on a status-store
    /// error, but logs it (M4) so a dead status backend is at least visible
    /// rather than silently discarded.
    pub(super) async fn save_status(&self, status: GraphRunStatus) {
        if let Some(store) = &self.status_store {
            let run_id = status.run_id.clone();
            if let Err(err) = store.put_status(status).await {
                tracing::warn!(
                    "[graph:status] failed to persist run status for run `{run_id}`: {err}"
                );
            }
        }
    }
}
