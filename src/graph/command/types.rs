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

/// A dynamic-fanout packet: schedule `node` for the next superstep with a
/// custom `arg` delivered to it through [`crate::graph::NodeContext::send_arg`],
/// independent of the graph's main committed state.
///
/// `Send` is the primitive for map-reduce, search fanout, parallel tool calls,
/// and per-item scoring: a node can emit one `Send` per work item — even many
/// pointing at the *same* target node — and each scheduled invocation receives
/// its own `arg`. Distinct from a plain `goto`, which simply activates a node
/// against the shared state with no per-activation input.
#[derive(Clone, Debug)]
pub struct Send {
    /// The node to schedule.
    pub node: NodeId,
    /// The per-invocation input delivered via `NodeContext::send_arg`.
    pub arg: serde_json::Value,
}

impl Send {
    /// Creates a `Send` scheduling `node` with `arg`.
    pub fn new(node: impl Into<NodeId>, arg: serde_json::Value) -> Self {
        Self {
            node: node.into(),
            arg,
        }
    }
}

/// A single routing target produced by a [`Command`]: either a plain node
/// activation ([`RouteTarget::Node`]) or a [`Send`] packet carrying
/// per-invocation input ([`RouteTarget::Send`]).
#[derive(Clone, Debug)]
pub enum RouteTarget {
    /// Activate the node against the shared committed state.
    Node(NodeId),
    /// Schedule the node with a custom per-invocation argument.
    Send(Send),
}

impl RouteTarget {
    /// The destination node id regardless of target kind.
    pub fn node(&self) -> &NodeId {
        match self {
            RouteTarget::Node(node) => node,
            RouteTarget::Send(send) => &send.node,
        }
    }

    /// The per-invocation argument when this is a [`Send`], else `None`.
    pub fn send_arg(&self) -> Option<&serde_json::Value> {
        match self {
            RouteTarget::Node(_) => None,
            RouteTarget::Send(send) => Some(&send.arg),
        }
    }
}

/// A routing/update/resume directive returned from a node.
///
/// Commands combine three orthogonal effects:
///
/// - `update`: a partial state update applied through the reducer.
/// - `goto`: explicit next-step targets — plain node activations and/or [`Send`]
///   fanout packets — overriding static/conditional edges.
/// - `resume`: a value paired with an interrupt resume (set on the caller side).
#[derive(Clone, Debug)]
pub struct Command<Update> {
    /// Optional partial update applied before routing.
    pub update: Option<Update>,
    /// Explicit routing targets for the next superstep. Each entry is either a
    /// plain node activation or a [`Send`] packet (see [`RouteTarget`]).
    pub goto: Vec<RouteTarget>,
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
