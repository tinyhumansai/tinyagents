//! Recursion policy and depth-tracking types.
//!
//! See the module [`mod`](super) docs for how these are wired into the
//! executor. This file holds the plain data definitions: a [`RecursionFrame`]
//! (one level of the run tree), a [`RecursionPolicy`] (the configured caps), and
//! a [`RecursionStack`] (the live frame stack plus enforcement helpers).

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::harness::ids::{GraphId, NodeId, RunId, TaskId};
use crate::{Result, TinyAgentsError};

/// One level of the graph/subgraph/sub-agent recursion tree.
///
/// A frame is pushed for every recursive call (a graph invoking a subgraph, a
/// subgraph invoking a sub-agent, a router looping a graph back into itself) and
/// popped on return, so the live [`RecursionStack`] always describes the path
/// from the root run down to the currently-executing run. Frames are serialized
/// into checkpoint metadata so a UI can render nested runs without
/// reconstructing the tree from event logs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionFrame {
    /// The graph this frame is executing.
    pub graph_id: GraphId,
    /// The hosting node, when this frame was entered from a parent node (the
    /// root frame of a top-level run has `None`).
    pub node_id: Option<NodeId>,
    /// The run id of this frame's execution.
    pub run_id: RunId,
    /// The scheduled task this frame descends from, when graph-backed (for
    /// example a `Send` fanout task).
    pub task_id: Option<TaskId>,
    /// The checkpoint namespace of this frame's graph (empty for top-level).
    pub namespace: Vec<String>,
    /// Zero-based depth of this frame in the recursion tree (the root is `0`).
    pub depth: usize,
    /// The run id of the enclosing frame, when this frame is a child run.
    pub parent: Option<RunId>,
}

/// Explicit limits that bound recursive graph execution.
///
/// The three caps are tracked independently so graph-call recursion (nested
/// runs), node-loop recursion (a router cycling the same node), and total work
/// (super-steps) can each be reasoned about and surfaced separately:
///
/// - `max_depth` bounds the run-tree depth ([`RecursionStack::push`] enforces
///   it, failing with [`TinyAgentsError::SubAgentDepth`]).
/// - `max_visits_per_node` optionally bounds how many times any single node may
///   be activated within one run (failing with
///   [`TinyAgentsError::NodeVisitLimit`]).
/// - `max_total_steps` bounds the number of super-steps a single run may execute
///   (failing with [`TinyAgentsError::RecursionLimit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionPolicy {
    /// Maximum run-tree depth (root run included). Reaching it on a push fails.
    pub max_depth: usize,
    /// Optional cap on activations of a single node within one run.
    pub max_visits_per_node: Option<usize>,
    /// Maximum number of super-steps a single run may execute.
    pub max_total_steps: usize,
}

impl Default for RecursionPolicy {
    /// A conservative default: 25 levels of run-tree depth, no per-node visit
    /// cap, and 1000 super-steps per run.
    fn default() -> Self {
        Self {
            max_depth: 25,
            max_visits_per_node: None,
            max_total_steps: 1000,
        }
    }
}

/// The live stack of [`RecursionFrame`]s for an executing run tree, paired with
/// the [`RecursionPolicy`] that bounds it.
///
/// Every graph/subgraph/sub-agent call [`push`](RecursionStack::push)es a frame
/// (enforcing `max_depth`) and every return [`pop`](RecursionStack::pop)s it, so
/// the stack stays symmetric. The executor builds one stack per run from the
/// inherited parent frames, pushes this run's frame, and consults it when
/// persisting checkpoints and enforcing the step cap.
#[derive(Clone, Debug, Default)]
pub struct RecursionStack {
    frames: Vec<RecursionFrame>,
    policy: RecursionPolicy,
}

impl RecursionStack {
    /// Creates an empty stack governed by `policy`.
    pub fn new(policy: RecursionPolicy) -> Self {
        Self {
            frames: Vec::new(),
            policy,
        }
    }

    /// Creates a stack pre-seeded with the inherited `frames` of an enclosing
    /// run, governed by `policy`. Used when a subgraph/sub-agent run continues
    /// the parent's recursion tree.
    pub fn with_frames(frames: Vec<RecursionFrame>, policy: RecursionPolicy) -> Self {
        Self { frames, policy }
    }

    /// The policy bounding this stack.
    pub fn policy(&self) -> &RecursionPolicy {
        &self.policy
    }

    /// The current recursion depth (number of frames on the stack).
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// The frames currently on the stack, root-first.
    pub fn frames(&self) -> &[RecursionFrame] {
        &self.frames
    }

    /// Pushes a recursion frame, enforcing [`RecursionPolicy::max_depth`].
    ///
    /// Returns [`TinyAgentsError::SubAgentDepth`] when the push would make the
    /// stack deeper than `max_depth` allows; the frame is **not** pushed in that
    /// case, so a caller that recovers keeps a consistent stack.
    pub fn push(&mut self, frame: RecursionFrame) -> Result<()> {
        if self.frames.len() + 1 > self.policy.max_depth {
            return Err(TinyAgentsError::SubAgentDepth(self.policy.max_depth));
        }
        self.frames.push(frame);
        Ok(())
    }

    /// Pops the deepest frame, returning it (or `None` when empty).
    pub fn pop(&mut self) -> Option<RecursionFrame> {
        self.frames.pop()
    }

    /// Enforces [`RecursionPolicy::max_total_steps`] against an executed-step
    /// count, returning [`TinyAgentsError::RecursionLimit`] when the run has
    /// reached the cap.
    pub fn check_total_steps(&self, steps: usize) -> Result<()> {
        if steps >= self.policy.max_total_steps {
            return Err(TinyAgentsError::RecursionLimit(self.policy.max_total_steps));
        }
        Ok(())
    }

    /// Records one activation of `node` against `counts` and enforces
    /// [`RecursionPolicy::max_visits_per_node`] when configured.
    ///
    /// Increments the per-node counter, returning
    /// [`TinyAgentsError::NodeVisitLimit`] when the node has now been activated
    /// more times than the policy allows.
    pub fn record_node_visit(
        &self,
        counts: &mut std::collections::HashMap<NodeId, usize>,
        node: &NodeId,
    ) -> Result<()> {
        let Some(max) = self.policy.max_visits_per_node else {
            return Ok(());
        };
        let count = counts.entry(node.clone()).or_insert(0);
        *count += 1;
        if *count > max {
            return Err(TinyAgentsError::NodeVisitLimit {
                node: node.to_string(),
                limit: max,
            });
        }
        Ok(())
    }
}

/// A reference to a child run spawned from a node within an enclosing run.
///
/// Recorded when a subgraph node embeds and runs a [`CompiledGraph`](crate::graph::CompiledGraph):
/// the child gets its own [`run_id`](ChildRun::run_id) while preserving the
/// enclosing run's [`root_run_id`](ChildRun::root_run_id), so a caller can
/// reconstruct the parent/child run lineage after a run completes. Child runs
/// are surfaced on [`GraphExecution`](crate::graph::GraphExecution) and embedded
/// into the parent checkpoint metadata under a `child_runs` array.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildRun {
    /// The embedding node that ran the child graph.
    pub node: NodeId,
    /// The graph id of the embedded child.
    pub graph_id: GraphId,
    /// The child run's own (distinct) run id.
    pub run_id: RunId,
    /// The shared root run id of the whole recursion tree (preserved from the
    /// enclosing run).
    pub root_run_id: RunId,
    /// Token usage rolled up from the child run, when the child reported any.
    ///
    /// Subgraph children leave this at the default (their usage is tracked by
    /// their own model calls); a [`crate::graph::subagent_node`] sub-agent child
    /// folds the delegated harness agent's [`UsageTotals`] here so it is visible
    /// on the parent [`GraphExecution`](crate::graph::GraphExecution) rollup.
    #[serde(default)]
    pub usage: crate::harness::usage::UsageTotals,
}

/// A thread-safe collector the executor hands to node contexts so that a
/// subgraph node can report the [`ChildRun`] it spawned back to the enclosing
/// run.
///
/// One sink is created per run; the executor drains it at each superstep
/// boundary to embed that step's child runs into the boundary checkpoint and to
/// accumulate them onto the final [`GraphExecution`](crate::graph::GraphExecution).
#[derive(Clone, Debug, Default)]
pub struct ChildRunSink {
    inner: Arc<Mutex<Vec<ChildRun>>>,
}

impl ChildRunSink {
    /// Creates an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a spawned child run.
    pub fn record(&self, child: ChildRun) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.push(child);
        }
    }

    /// Removes and returns every child run recorded since the last drain.
    pub fn drain(&self) -> Vec<ChildRun> {
        match self.inner.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(_) => Vec::new(),
        }
    }
}

/// A flat parent/child run-lineage view derived from a completed
/// [`GraphExecution`](crate::graph::GraphExecution).
///
/// `RunTree` is the run-id counterpart to a [`RecursionStack`]: where the stack
/// describes the live path while a run executes, a `RunTree` is the after-the-fact
/// summary a caller reads to see this run's id, the shared root, the enclosing
/// parent (when this was itself a child run), and every child run it spawned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunTree {
    /// This run's id.
    pub run_id: RunId,
    /// The root run id of the whole recursion tree (equals `run_id` for a
    /// top-level run).
    pub root_run_id: RunId,
    /// The enclosing run's id, when this run was spawned as a child.
    pub parent_run_id: Option<RunId>,
    /// The child runs spawned from nodes within this run.
    pub children: Vec<ChildRun>,
}

impl RunTree {
    /// True when this run is the root of its recursion tree.
    pub fn is_root(&self) -> bool {
        self.parent_run_id.is_none()
    }
}
