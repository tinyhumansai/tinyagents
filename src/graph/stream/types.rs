//! Low-level graph events and high-level stream modes — the wire vocabulary the
//! recursive executor uses to narrate its own execution.
//!
//! [`GraphEvent`] is the fine-grained, per-node/per-step lifecycle signal the
//! durable executor emits at every boundary; [`StreamMode`] is the LangGraph-
//! style selection of *which projection* of that stream a caller wants (full
//! values, per-node updates, model messages, debug detail, interrupts, or
//! custom node writes). Together they let observers — including a parent run
//! consuming a subgraph — follow a run without inspecting its internal state.

use serde::{Deserialize, Serialize};

use crate::graph::command::Interrupt;
use crate::harness::ids::{CheckpointId, NodeId, RunId};

/// A low-level graph lifecycle event emitted through a [`super::GraphEventSink`].
///
/// These are the durable-executor analogues of the observability event model in
/// the graph spec, reduced to the set the milestone executor actually emits.
///
/// The variants are serde-serializable so a single event can be wrapped into a
/// durable [`crate::graph::observability::GraphObservation`] envelope, journaled,
/// and replayed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphEvent {
    /// The run began (emitted once before the first superstep).
    RunStarted {
        /// The run that started.
        run_id: RunId,
    },
    /// The run finished successfully.
    RunCompleted {
        /// The run that completed.
        run_id: RunId,
        /// Total supersteps executed.
        steps: usize,
    },
    /// The run aborted with an error.
    RunFailed {
        /// The run that failed.
        run_id: RunId,
        /// Rendered error.
        error: String,
    },
    /// A superstep started with the given active node set.
    StepStarted {
        /// 1-based step number.
        step: usize,
        /// Nodes scheduled to run this step.
        active: Vec<NodeId>,
    },
    /// A superstep finished and its boundary work (reducer, checkpoint) ran.
    StepCompleted {
        /// 1-based step number.
        step: usize,
    },
    /// A task was scheduled for a node in the active set.
    TaskScheduled {
        /// Target node.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A node handler began executing.
    NodeStarted {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A node handler completed successfully.
    NodeCompleted {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A node handler returned an error.
    NodeFailed {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
        /// Rendered error.
        error: String,
    },
    /// A node produced a state update applied at the boundary.
    StateUpdated {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A route was selected for a node.
    RouteSelected {
        /// Source node.
        node: NodeId,
        /// Selected next node.
        target: NodeId,
    },
    /// A checkpoint was persisted at a superstep boundary.
    CheckpointSaved {
        /// Persisted checkpoint id.
        checkpoint_id: CheckpointId,
    },
    /// A node emitted an interrupt and the run paused.
    InterruptEmitted {
        /// The emitted interrupt.
        interrupt: Interrupt,
    },
    /// An embedded subgraph began executing under a child namespace.
    SubgraphStarted {
        /// The parent node hosting the subgraph.
        node: NodeId,
        /// The child checkpoint namespace.
        namespace: Vec<String>,
    },
    /// An embedded subgraph finished executing.
    SubgraphCompleted {
        /// The parent node hosting the subgraph.
        node: NodeId,
        /// The child checkpoint namespace.
        namespace: Vec<String>,
    },
    /// A parallel superstep forked an execution branch for a node.
    ContextForked {
        /// The node whose branch was forked.
        node: NodeId,
        /// The branch (fork) index within the active set.
        fork: usize,
        /// Step number.
        step: usize,
    },
    /// The effective recursion/namespace depth changed.
    RecursionDepthChanged {
        /// The new depth (number of enclosing namespaces).
        depth: usize,
    },
    /// An arbitrary user-defined event written from inside a node.
    Custom {
        /// A stable name for the custom event.
        name: String,
        /// Free-form structured payload.
        data: serde_json::Value,
    },
}

impl GraphEvent {
    /// Returns a stable, dot-separated string that names the kind of event.
    ///
    /// The returned string is a static literal, suitable for logging,
    /// filtering, and serde-independent routing. Examples: `"run.started"`,
    /// `"step.started"`, `"node.completed"`.
    pub fn kind(&self) -> &'static str {
        match self {
            GraphEvent::RunStarted { .. } => "run.started",
            GraphEvent::RunCompleted { .. } => "run.completed",
            GraphEvent::RunFailed { .. } => "run.failed",
            GraphEvent::StepStarted { .. } => "step.started",
            GraphEvent::StepCompleted { .. } => "step.completed",
            GraphEvent::TaskScheduled { .. } => "task.scheduled",
            GraphEvent::NodeStarted { .. } => "node.started",
            GraphEvent::NodeCompleted { .. } => "node.completed",
            GraphEvent::NodeFailed { .. } => "node.failed",
            GraphEvent::StateUpdated { .. } => "state.updated",
            GraphEvent::RouteSelected { .. } => "route.selected",
            GraphEvent::CheckpointSaved { .. } => "checkpoint.saved",
            GraphEvent::InterruptEmitted { .. } => "interrupt.emitted",
            GraphEvent::SubgraphStarted { .. } => "subgraph.started",
            GraphEvent::SubgraphCompleted { .. } => "subgraph.completed",
            GraphEvent::ContextForked { .. } => "context.forked",
            GraphEvent::RecursionDepthChanged { .. } => "recursion.depth_changed",
            GraphEvent::Custom { .. } => "custom",
        }
    }

    /// Returns the superstep number this event is associated with, when the
    /// variant carries one. Used to stamp the `step` field of a durable
    /// [`crate::graph::observability::GraphObservation`].
    pub fn step(&self) -> Option<usize> {
        match self {
            GraphEvent::StepStarted { step, .. }
            | GraphEvent::StepCompleted { step }
            | GraphEvent::TaskScheduled { step, .. }
            | GraphEvent::NodeStarted { step, .. }
            | GraphEvent::NodeCompleted { step, .. }
            | GraphEvent::NodeFailed { step, .. }
            | GraphEvent::StateUpdated { step, .. }
            | GraphEvent::ContextForked { step, .. } => Some(*step),
            GraphEvent::RunCompleted { steps, .. } => Some(*steps),
            _ => None,
        }
    }
}

/// High-level projection modes for a graph run stream.
///
/// These mirror the LangGraph stream modes. The milestone executor exposes them
/// as a selection enum; richer typed `StreamPart` projection is future work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamMode {
    /// Full state values after each step.
    Values,
    /// Per-node/per-task state updates.
    Updates,
    /// Harness message or token deltas from model nodes.
    Messages,
    /// Checkpoints plus task internals.
    Debug,
    /// Pending interrupts only.
    Interrupts,
    /// Arbitrary user stream writes from inside nodes.
    Custom,
}
