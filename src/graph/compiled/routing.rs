//! Routing: resolving a completed step's active set into the next
//! superstep's activations (`goto`/conditional branches, [`Send`]
//! fanout, and interrupt-durability preconditions).
//!
//! Split out of `compiled/mod.rs`; see that module's doc comment for the
//! executor's overall design.

use super::*;

impl<State, Update> CompiledGraph<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    pub(super) fn route_completed(
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
    pub(super) fn route(
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

    pub(super) fn require_interrupt_durability(&self, thread_id: &Option<ThreadId>) -> Result<()> {
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
}
