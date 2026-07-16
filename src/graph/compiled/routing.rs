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

        // Mixed fan-in barrier relief: a waiting/barrier node normally
        // activates only once every registered predecessor has arrived. When
        // one of those predecessors is reachable only via a conditional
        // branch, and the *taken* branch does not lead toward it, that
        // predecessor will never arrive on its own — register a phantom
        // arrival on its behalf so the barrier can still clear on the
        // predecessors that actually ran, instead of deadlocking forever.
        //
        // The check is keyed on `source`'s *resolved routing target* this
        // step (via `reaches_deterministically`), not on whether
        // `relief_node` is freshly scheduled into `next` this same
        // superstep. A same-superstep check is wrong for a multi-hop
        // conditional predecessor (`source --branch--> x --main-->
        // relief_node`, `x` a plain pass-through): `relief_node` would not
        // yet be in `next` on the step `source` itself completes — even on
        // the *taken* branch, where `relief_node` WILL run once `x` does —
        // so a same-superstep check would fire a premature phantom arrival
        // and let the barrier clear before the real predecessor's data
        // commits, reintroducing the exact data-loss bug this primitive
        // exists to prevent. Resolving `source`'s actual target and walking
        // it forward through deterministic (non-branching) edges is correct
        // for both direct and multi-hop cases because it answers "was the
        // branch leading to `relief_node` taken", not "did `relief_node`
        // happen to run in lockstep with `source`".
        for relief in self.barrier_reliefs.iter() {
            let source_indices: Vec<usize> = completed
                .iter()
                .enumerate()
                .filter(|(_, activation)| activation.node == relief.source)
                .map(|(index, _)| index)
                .collect();
            if source_indices.is_empty() {
                continue;
            }
            let mut branch_taken = false;
            for index in &source_indices {
                let targets = self.route(
                    &relief.source,
                    goto_map.get(index).map(Vec::as_slice),
                    state,
                )?;
                if targets.iter().any(|target| {
                    self.reaches_deterministically(
                        target.node(),
                        &relief.relief_node,
                        &relief.barrier_node,
                    )
                }) {
                    branch_taken = true;
                    break;
                }
            }
            // A relief_node freshly scheduled into `next` this step (a
            // direct, single-hop predecessor) is also proof the branch was
            // taken; kept as a defensive fallback alongside the resolved-
            // target check above.
            if branch_taken || next_seen.contains(&relief.relief_node) {
                continue;
            }
            let Some(required) = self.waiting.get(&relief.barrier_node) else {
                continue;
            };
            let arrived = barrier_arrivals
                .entry(relief.barrier_node.clone())
                .or_default();
            arrived.insert(relief.relief_node.clone());
            if !required.is_subset(arrived) {
                continue;
            }
            barrier_arrivals.remove(&relief.barrier_node);
            if next_seen.insert(relief.barrier_node.clone()) {
                next.push(Activation {
                    node: relief.barrier_node.clone(),
                    send_arg: None,
                });
            }
        }

        Ok(next)
    }

    /// Whether `to` is reachable from `from` by following only deterministic
    /// static routing — plain/waiting edges (`self.edges`), which resolve to
    /// exactly one successor with no runtime decision — without ever
    /// expanding through `stop`.
    ///
    /// This is what makes barrier-relief evaluation correct for a
    /// multi-hop conditional predecessor: once a brancher's routing decision
    /// for this step is resolved to a concrete target, whether that target
    /// eventually leads to a barrier's conditional predecessor is a static
    /// property of the compiled topology for any chain of plain pass-through
    /// nodes — it does not depend on when each hop happens to run. A further
    /// conditional/command node along the way (no `self.edges` entry) is a
    /// second runtime decision this walk cannot resolve ahead of time, so it
    /// conservatively reports unreachable there (falling back to the
    /// same-superstep check).
    fn reaches_deterministically(&self, from: &NodeId, to: &NodeId, stop: &NodeId) -> bool {
        if from == to {
            return true;
        }
        let mut current = from;
        let mut seen: HashSet<&NodeId> = HashSet::new();
        while let Some(next) = self.edges.get(current) {
            if next == to {
                return true;
            }
            if next == stop || !seen.insert(next) {
                return false;
            }
            current = next;
        }
        false
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
