//! Partial-update node results, commands, and interrupts.
//!
//! In the durable execution model a node no longer returns whole state. It
//! returns a [`NodeResult`], which is one of:
//!
//! - [`NodeResult::Update`]: a partial update merged through the graph reducer
//!   at the superstep boundary.
//! - [`NodeResult::Command`]: a [`Command`] combining an optional update with
//!   explicit routing (`goto`) and/or an interrupt resume value.
//! - [`NodeResult::Interrupt`]: an [`Interrupt`] that pauses the run for
//!   human-in-the-loop input.

use crate::harness::ids::NodeId;

/// The outcome of running a durable graph node.
#[derive(Clone, Debug)]
pub enum NodeResult<Update> {
    /// A partial state update to merge through the reducer.
    Update(Update),
    /// A command that may update state and/or route explicitly.
    Command(Command<Update>),
    /// An interrupt that pauses execution until a resume command arrives.
    Interrupt(Interrupt),
}

/// A routing/update/resume directive returned from a node.
///
/// Commands combine three orthogonal effects:
///
/// - `update`: a partial state update applied through the reducer.
/// - `goto`: explicit next-node targets, overriding static/conditional edges.
/// - `resume`: a value paired with an interrupt resume (set on the caller side).
#[derive(Clone, Debug)]
pub struct Command<Update> {
    /// Optional partial update applied before routing.
    pub update: Option<Update>,
    /// Explicit routing targets for the next superstep.
    pub goto: Vec<NodeId>,
    /// Resume value for an interrupted node (used by `CompiledGraph::resume`).
    pub resume: Option<serde_json::Value>,
}

/// A human-in-the-loop pause point.
///
/// Interrupts require a checkpointer. When a node returns an interrupt the
/// executor persists a checkpoint at the boundary and returns control to the
/// caller; `CompiledGraph::resume` re-runs the interrupted node.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interrupt {
    /// Stable id for matching a resume value to this interrupt.
    pub id: String,
    /// The node that emitted the interrupt.
    pub node: NodeId,
    /// Arbitrary payload presented to the human/approver.
    pub payload: serde_json::Value,
}
