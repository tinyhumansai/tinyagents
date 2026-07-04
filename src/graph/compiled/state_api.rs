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
    /// [`StateReducer`](crate::graph::StateReducer) the executor uses, on top of
    /// the thread's latest committed state. When `as_node` is supplied it must
    /// name a real node (else [`TinyAgentsError::MissingNode`]); the write is
    /// attributed to that node and the new checkpoint's pending nodes become that
    /// node's routing successors (so a subsequent resume continues from after the
    /// attributed node). A command node cannot be used as `as_node` (it routes
    /// dynamically and has no static successors); doing so returns
    /// [`TinyAgentsError::Graph`] rather than silently producing a non-resumable
    /// checkpoint. With `as_node == None` the latest pending node set is
    /// preserved. Requires a configured checkpointer and an existing checkpoint
    /// for the thread.
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
}
