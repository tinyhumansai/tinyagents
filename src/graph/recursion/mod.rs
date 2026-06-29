//! Recursion policy and depth tracking for the durable graph runtime.
//!
//! The graph allows recursion — a graph can invoke itself as a subgraph, a
//! subgraph can call back into its parent, an agent node can call another agent
//! that re-enters the same graph, a router can loop nodes until state
//! converges, and a `Send` fanout can schedule many recursive child tasks — but
//! only with explicit limits and tracking.
//!
//! This module provides that contract:
//!
//! - [`RecursionFrame`] names one level of the run tree (graph, node, run, task,
//!   namespace, depth, parent run).
//! - [`RecursionPolicy`] holds the caps: `max_depth` (run-tree depth),
//!   `max_visits_per_node` (node-loop recursion), and `max_total_steps`
//!   (super-steps per run).
//! - [`RecursionStack`] is the live frame stack: every graph/subgraph/sub-agent
//!   call pushes a frame (enforcing `max_depth`) and every return pops it, so
//!   the stack stays symmetric and always describes the path from the root run
//!   to the current one.
//!
//! The executor ([`crate::graph::CompiledGraph::execute`]) builds one stack per
//! run, tracks graph-call depth separately from node-loop visits, enforces the
//! caps with clear recursion errors ([`crate::TinyAgentsError::SubAgentDepth`],
//! [`crate::TinyAgentsError::NodeVisitLimit`],
//! [`crate::TinyAgentsError::RecursionLimit`]), records the current stack into
//! checkpoint metadata under a `recursion` array, and emits the live depth on
//! [`crate::graph::GraphEvent::RecursionDepthChanged`].

mod types;

pub use types::{ChildRun, ChildRunSink, RecursionFrame, RecursionPolicy, RecursionStack, RunTree};

#[cfg(test)]
mod test;
