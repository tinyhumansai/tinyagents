//! Superstep executor for the durable graph.
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

pub use types::{CompiledGraph, GraphExecution};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use crate::graph::builder::{Branch, BuilderNode, END, ForkId, NodeContext};
use crate::graph::checkpoint::{Checkpoint, Checkpointer};
use crate::graph::command::{Command, Interrupt, NodeResult};
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

/// The folded result of running a superstep's active node set, ready to apply
/// at the step boundary.
struct StepRun<Update> {
    /// Branch updates in deterministic active-set index order.
    updates: Vec<Update>,
    /// Explicit `goto` routing keyed by the node that produced it.
    goto_map: HashMap<NodeId, Vec<NodeId>>,
    /// The lowest-index branch interrupt, if any (its active-set index + value).
    interrupt: Option<(usize, Interrupt)>,
}

fn dedupe(nodes: Vec<NodeId>) -> Vec<NodeId> {
    let mut seen = HashSet::new();
    nodes
        .into_iter()
        .filter(|n| seen.insert(n.clone()))
        .collect()
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
        entry: NodeId,
        reducer: Arc<dyn StateReducer<State, Update>>,
        recursion_limit: usize,
        parallel: bool,
    ) -> Self {
        Self {
            graph_id,
            nodes: Arc::new(nodes),
            edges: Arc::new(edges),
            branches: Arc::new(branches),
            command_nodes: Arc::new(command_nodes),
            entry,
            reducer,
            recursion_limit,
            checkpointer: None,
            event_sink: None,
            namespace: Vec::new(),
            parallel,
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

    /// Sets the checkpoint namespace (used by subgraph wrappers).
    pub fn with_namespace(mut self, namespace: Vec<String>) -> Self {
        self.namespace = namespace;
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
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or_else(|| TinyAgentsError::Resume("no checkpointer configured".to_string()))?;
        let thread_id = thread_id.into();

        let checkpoint = checkpointer
            .get(thread_id.as_str(), None)
            .await?
            .ok_or_else(|| {
                TinyAgentsError::Resume(format!("no checkpoint found for thread `{thread_id}`"))
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

    async fn execute(
        &self,
        mut state: State,
        initial_active: Vec<NodeId>,
        thread_id: Option<ThreadId>,
        mut resume_map: HashMap<NodeId, serde_json::Value>,
    ) -> Result<GraphExecution<State>> {
        let run_id = RunId::new(next_id("run"));
        let started_at = SystemTime::now();
        let mut visited: Vec<NodeId> = Vec::new();
        let mut steps = 0usize;
        let mut last_checkpoint: Option<CheckpointId> = None;
        let mut parent_checkpoint: Option<String> = None;
        let mut active = dedupe(initial_active);

        while !active.is_empty() {
            if steps >= self.recursion_limit {
                return Err(TinyAgentsError::RecursionLimit(self.recursion_limit));
            }
            steps += 1;
            self.emit(GraphEvent::StepStarted {
                step: steps,
                active: active.clone(),
            });

            let StepRun {
                updates,
                goto_map,
                interrupt,
            } = if self.parallel && active.len() > 1 {
                self.run_active_parallel(
                    &active,
                    &state,
                    &run_id,
                    &thread_id,
                    steps,
                    &mut resume_map,
                    &mut visited,
                )
                .await?
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
                .await?
            };

            // Apply collected updates through the reducer at the boundary.
            for update in updates {
                state = self.reducer.apply(state, update)?;
            }

            // Interrupt: persist a checkpoint whose next nodes are the
            // not-yet-completed members of this step (interrupted node first),
            // then return control to the caller.
            if let Some((index, emitted)) = interrupt {
                let pending: Vec<NodeId> = active[index..].to_vec();
                let interrupt_id = InterruptId::new(emitted.id.clone());
                let checkpoint_id = self
                    .persist_checkpoint(
                        &thread_id,
                        &state,
                        &pending,
                        &active[..index],
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
            let mut next: Vec<NodeId> = Vec::new();
            for node_id in &completed {
                let targets = self.route(node_id, &goto_map, &state)?;
                for target in targets {
                    if target.as_str() == END {
                        continue;
                    }
                    self.emit(GraphEvent::RouteSelected {
                        node: node_id.clone(),
                        target: target.clone(),
                    });
                    next.push(target);
                }
            }
            let next = dedupe(next);

            // Persist a boundary checkpoint.
            let checkpoint_id = self
                .persist_checkpoint(
                    &thread_id,
                    &state,
                    &next,
                    &completed,
                    Vec::new(),
                    parent_checkpoint.clone(),
                    steps,
                    "loop",
                )
                .await?;
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

        Ok(GraphExecution {
            state,
            visited,
            steps,
            interrupts: Vec::new(),
            status,
            checkpoint_id: last_checkpoint,
        })
    }

    /// Builds the per-task [`NodeContext`] for `node_id` at the given branch.
    ///
    /// `fork` carries the branch identity in a concurrent step (`None` in
    /// sequential mode or single-node steps). The resume value for the node is
    /// consumed from `resume_map`.
    fn node_context(
        &self,
        node_id: &NodeId,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        step: usize,
        resume_map: &mut HashMap<NodeId, serde_json::Value>,
        fork: Option<ForkId>,
    ) -> NodeContext {
        NodeContext {
            node_id: node_id.clone(),
            run_id: run_id.clone(),
            thread_id: thread_id.clone(),
            step,
            resume: resume_map.remove(node_id),
            fork,
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
        goto_map: &mut HashMap<NodeId, Vec<NodeId>>,
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
        active: &[NodeId],
        state: &State,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        step: usize,
        resume_map: &mut HashMap<NodeId, serde_json::Value>,
        visited: &mut Vec<NodeId>,
    ) -> Result<StepRun<Update>> {
        let mut updates: Vec<Update> = Vec::new();
        let mut goto_map: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut interrupt: Option<(usize, Interrupt)> = None;

        for (index, node_id) in active.iter().enumerate() {
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

            let ctx = self.node_context(node_id, run_id, thread_id, step, resume_map, None);
            let result = match (node.handler)(state.clone(), ctx).await {
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
    /// Every branch starts before any is awaited and all are driven to
    /// completion via [`futures::future::join_all`]; each branch executes on its
    /// own cloned `State` snapshot and a distinct [`ForkId`]. Results are then
    /// folded in active-set index order — the reducer is the join/fan-in — so the
    /// merged state is reproducible regardless of completion order. The
    /// lowest-index branch that errors or interrupts is the step's terminal
    /// outcome; lower-index successful branches still contribute their updates.
    #[allow(clippy::too_many_arguments)]
    async fn run_active_parallel(
        &self,
        active: &[NodeId],
        state: &State,
        run_id: &RunId,
        thread_id: &Option<ThreadId>,
        step: usize,
        resume_map: &mut HashMap<NodeId, serde_json::Value>,
        visited: &mut Vec<NodeId>,
    ) -> Result<StepRun<Update>> {
        // Build one forked context + future per branch. Node lookup and resume
        // consumption happen up front so the futures borrow nothing local.
        let mut futures = Vec::with_capacity(active.len());
        for (index, node_id) in active.iter().enumerate() {
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

            let fork = Some(ForkId::new(index, node_id.clone()));
            let ctx = self.node_context(node_id, run_id, thread_id, step, resume_map, fork);
            futures.push((node.handler)(state.clone(), ctx));
        }

        // Drive all branches concurrently to completion.
        let results = futures::future::join_all(futures).await;

        // Fold in deterministic active-set index order.
        let mut updates: Vec<Update> = Vec::new();
        let mut goto_map: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut interrupt: Option<(usize, Interrupt)> = None;

        for (index, (node_id, result)) in active.iter().zip(results).enumerate() {
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

    /// Resolves the next-node targets for `node_id`.
    fn route(
        &self,
        node_id: &NodeId,
        goto_map: &HashMap<NodeId, Vec<NodeId>>,
        state: &State,
    ) -> Result<Vec<NodeId>> {
        if let Some(targets) = goto_map.get(node_id) {
            return Ok(targets.clone());
        }
        if let Some(target) = self.edges.get(node_id) {
            return Ok(vec![target.clone()]);
        }
        if let Some(branch) = self.branches.get(node_id) {
            let route = (branch.router)(state);
            let target = branch.routes.get(&route).cloned().ok_or_else(|| {
                TinyAgentsError::MissingRoute {
                    node: node_id.to_string(),
                    route,
                }
            })?;
            return Ok(vec![target]);
        }
        // Sink: no outgoing routing, the branch ends here.
        Ok(Vec::new())
    }

    #[allow(clippy::too_many_arguments)]
    async fn persist_checkpoint(
        &self,
        thread_id: &Option<ThreadId>,
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
