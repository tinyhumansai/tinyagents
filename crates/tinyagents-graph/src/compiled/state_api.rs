//! State inspection and manual state-write API (`get_state`,
//! `get_state_history`, `update_state`, `bulk_update_state`, `fork_state`).
//!
//! Split out of `compiled/mod.rs`; see that module's doc comment for the
//! executor's overall durability design.

use super::*;

impl<State, Update> CompiledGraph<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
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
        // Delegate the parent-lineage walk to the checkpointer so backends that
        // would otherwise re-read the whole thread per hop (the file backend) can
        // read it once and walk in memory (O(H) instead of O(H²)).
        let tuples = checkpointer
            .state_history(thread_id, &self.namespace, limit)
            .await?;
        Ok(tuples.into_iter().map(snapshot_from_tuple).collect())
    }

    /// Applies a manual state write to a thread, producing a new checkpoint with
    /// source `update`.
    ///
    /// The write is a genuine graph write: `update` is folded through the same
    /// [`StateReducer`](crate::StateReducer) the executor uses, on top of
    /// the thread's latest committed state. When `as_node` is supplied it must
    /// name a real node (else [`TinyAgentsError::MissingNode`]); the write is
    /// attributed to that node, which is treated as having just completed: it
    /// leaves the pending set and its routing successors are *merged into* the
    /// base checkpoint's remaining pending work (so a subsequent resume
    /// continues from after the attributed node without dropping the branches
    /// it never touched). Sibling branches keep their `Send` args, and a
    /// successor that is already pending is not scheduled twice. When the
    /// attributed node has several pending `Send` activations, the write
    /// completes all of them at once — a manual write cannot name which packet
    /// it stands for — and the successor is scheduled once. A command node
    /// cannot be used as `as_node` (it routes dynamically and has no static
    /// successors); doing so returns [`TinyAgentsError::Graph`] rather than
    /// silently producing a non-resumable checkpoint. A successor reached by a
    /// waiting edge is barrier-gated exactly as it would be during a run: it is
    /// scheduled only once every required predecessor has arrived (the write
    /// records the attributed node's arrival), so a manual write can never fire
    /// a join ahead of a still-pending branch — and because the other pending
    /// predecessors are retained, they still run and clear the join. With
    /// `as_node == None` the latest pending node set is preserved. Requires a
    /// configured checkpointer and an existing checkpoint for the thread.
    pub async fn update_state(
        &self,
        thread_id: &str,
        update: Update,
        as_node: Option<NodeId>,
    ) -> Result<CheckpointConfig> {
        let checkpointer = self.require_checkpointer()?;
        if let Some(node) = &as_node {
            if !self.nodes.contains_key(node) {
                return Err(TinyAgentsError::MissingNode(node.to_string()));
            }
            // A command node routes dynamically (via the [`Command`] it returns
            // at runtime), so it has no static successors to schedule here.
            // Attributing a manual write to one would persist an empty
            // `next_nodes` and silently render the thread non-resumable, so
            // reject it at write time instead.
            if self.command_nodes.contains(node) {
                return Err(TinyAgentsError::Graph(format!(
                    "cannot update state as node `{node}`: it routes dynamically \
                     via Command and has no static successors, so the resulting \
                     checkpoint would be non-resumable"
                )));
            }
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

        // The base checkpoint may itself be mid-step: an interrupt/failure
        // boundary whose completed siblings were deferred rather than
        // routed (see `boundary::advance`'s `carried_completed` doc, the C2
        // fix). Those node ids live in `base.completed_tasks` with no
        // persisted `goto_map` entry of their own, so — exactly like
        // `advance`'s carried-completed handling — they route via
        // static/conditional edges only, not any `Command::goto` they might
        // have returned. Routing them here (rather than silently dropping
        // them) is what keeps a manual write from permanently losing a
        // step's other branches the moment it touches a mid-step thread.
        // Same `source == "loop"` gate as `resume::resume_from_inner`'s
        // `mid_step` check, and for the same reason: an `update`-sourced
        // checkpoint can carry `interrupted_nodes` forward for provenance
        // (I2) without its `completed_tasks` representing owed routing —
        // `update_state` always resolves every carried completion itself
        // before writing.
        let base_source_is_loop = base
            .metadata
            .get("source")
            .and_then(serde_json::Value::as_str)
            == Some("loop");
        let carried_completed: Vec<NodeId> = if base_source_is_loop
            && (base.metadata.get("interrupted_nodes").is_some()
                || base.metadata.get("failed_node").is_some())
        {
            base.completed_tasks.clone()
        } else {
            Vec::new()
        };
        let new_state = self.reducer.apply(base.state, update)?;

        // Manual writes preserve any accumulated barrier arrivals, and an
        // attributed write (or a carried-forward completion routed here)
        // records its own arrival into them.
        let mut arrivals = barriers_from_persisted(&base.barrier_arrivals);
        // Pending schedule: the attributed node's successors and any
        // carried-forward completions' successors, merged into the base
        // checkpoint's still-pending work.
        //
        // `next_nodes` and `pending_activations` are derived from one merged
        // activation list so they can never disagree — resume prefers the
        // activations, so a node named by only one of them would be silently
        // dropped (or re-scheduled without its `Send` arg).
        let mut merged: Vec<Activation> = match &base.pending_activations {
            Some(pending) if !pending.is_empty() => pending
                .iter()
                .map(Activation::from)
                .filter(|activation| Some(&activation.node) != as_node.as_ref())
                .collect(),
            // Checkpoints written before `pending_activations` existed only
            // carry the node-id projection.
            _ => base
                .next_nodes
                .iter()
                .filter(|pending| Some(*pending) != as_node.as_ref())
                .cloned()
                .map(Activation::node)
                .collect(),
        };
        let mut seen: HashSet<NodeId> = merged
            .iter()
            .filter(|activation| activation.send_arg.is_none())
            .map(|activation| activation.node.clone())
            .collect();
        // Routes one completed node's (static/conditional-only) successors
        // into `merged`, applying the same barrier gate `route_completed`
        // applies at the normal boundary. The merge is unconditional rather
        // than a fallback for the nothing-was-scheduled case: `route(node,
        // None, ..)` resolves a static or conditional edge, so today it
        // yields at most one target and a withheld barrier is the only way
        // to end up with none — but keying the merge on that would silently
        // drop the untouched branches the moment a single call ever
        // resolves a withheld target *and* a schedulable one.
        let mut route_into_merged = |node: &NodeId| -> Result<()> {
            for target in self.route(node, None, &new_state)? {
                let tnode = target.node().clone();
                if tnode.as_str() == END {
                    continue;
                }
                if let Some(required) = self.waiting.get(&tnode) {
                    let arrived = arrivals.entry(tnode.clone()).or_default();
                    arrived.insert(node.clone());
                    if !required.is_subset(arrived) {
                        continue;
                    }
                    arrivals.remove(&tnode);
                }
                // `Send` activations may legitimately repeat a node (each
                // carries its own arg); plain ones are deduplicated so a
                // successor already pending is not scheduled twice.
                let send_arg = target.send_arg().cloned();
                if send_arg.is_some() || seen.insert(tnode.clone()) {
                    merged.push(Activation {
                        node: tnode,
                        send_arg,
                        task_id: TaskId::from(String::new()),
                    });
                }
            }
            Ok(())
        };
        for node in &carried_completed {
            route_into_merged(node)?;
        }
        if let Some(node) = &as_node {
            route_into_merged(node)?;
        }
        let next_nodes = activation_nodes(&merged);
        let pending_activations = if merged.is_empty() {
            None
        } else {
            Some(merged.iter().map(PendingActivation::from).collect())
        };
        // This write resolves every carried-forward completion's routing
        // (above), so none of them are still "owed" afterward; only the
        // attributed node (if any) is freshly completed by this write.
        let completed_tasks: Vec<NodeId> = as_node.iter().cloned().collect();
        let barrier_arrivals = barriers_to_persisted(&arrivals);

        // I2: carry the base checkpoint's interrupt provenance through this
        // manual write unless `as_node` names the node that actually
        // interrupted — clearing it in exactly that case is how the
        // documented "inspect -> update_state -> resume(value)" flow keeps
        // working (`resume` fans the value across the pending set when it
        // finds no interrupt provenance at all, so blindly erasing
        // `interrupts`/`interrupted_nodes` on every manual write — as before
        // this fix — handed the resume value to nodes that never paused).
        let base_interrupted_stamped: Vec<NodeId> = base
            .metadata
            .get("interrupted_nodes")
            .and_then(serde_json::Value::as_array)
            .map(|nodes| {
                nodes
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(NodeId::from)
                    .collect()
            })
            .unwrap_or_default();
        let names_interrupted_node = |node: &NodeId| -> bool {
            if base_interrupted_stamped.is_empty() {
                base.interrupts.iter().any(|i| &i.node == node)
            } else {
                base_interrupted_stamped.contains(node)
            }
        };
        let clears_interrupt = as_node.as_ref().is_some_and(names_interrupted_node);
        let (interrupts, interrupted_nodes_meta) = if clears_interrupt {
            (Vec::new(), Vec::new())
        } else {
            (base.interrupts.clone(), base_interrupted_stamped)
        };

        let checkpoint_id = next_checkpoint_id();
        let config = self.config_for(thread_id, Some(&checkpoint_id));
        let mut metadata = serde_json::json!({ "source": "update", "step": parent_step + 1 });
        if !interrupted_nodes_meta.is_empty() {
            metadata["interrupted_nodes"] = serde_json::json!(
                interrupted_nodes_meta
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
            );
        }
        let checkpoint = Checkpoint {
            thread_id: thread_id.to_string(),
            checkpoint_id,
            run_id: None,
            parent_checkpoint_id: Some(parent_id),
            namespace: self.namespace.clone(),
            state: new_state,
            next_nodes,
            completed_tasks,
            completed_routes: Vec::new(),
            pending_writes: Vec::new(),
            interrupts,
            pending_activations,
            barrier_arrivals,
            metadata,
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
            completed_routes: source.completed_routes.clone(),
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
}
