//! Low-level graph events and high-level stream modes — the wire vocabulary the
//! recursive executor uses to narrate its own execution.
//!
//! [`GraphEvent`] is the fine-grained, per-node/per-step lifecycle signal the
//! durable executor emits at every boundary; [`StreamMode`] is the LangGraph-
//! style selection of *which projection* of that stream a caller wants (full
//! values, per-node updates, model messages, debug detail, interrupts, or
//! custom node writes). Together they let observers — including a parent run
//! consuming a subgraph — follow a run without inspecting its internal state.

use crate::graph::command::Interrupt;
use crate::harness::ids::{CheckpointId, NodeId};

/// A low-level graph lifecycle event emitted through a [`super::GraphEventSink`].
///
/// These are the durable-executor analogues of the observability event model in
/// the graph spec, reduced to the set the milestone executor actually emits.
#[derive(Clone, Debug)]
pub enum GraphEvent {
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
