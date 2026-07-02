//! Per-thread durable **goal**: one objective per thread, carried across
//! supersteps, interrupts, and resumes.
//!
//! A thread goal is a single "completion contract" — a durable objective a
//! graph keeps pursuing until the model marks it [`Complete`], a token budget
//! is exhausted, or a host pauses it. This module owns the data model
//! ([`types`]), harness-[`Store`](crate::harness::store::Store)-backed
//! persistence ([`store`]), the model-facing controls exposed as harness tools
//! ([`tool`]), and the graph-native continuation surface ([`continuation`]).
//!
//! It is the graph analogue of OpenHuman's `thread_goals`, minus the
//! app-specific coupling (event bus, RPC envelopes, heartbeat scheduler): the
//! primitive is provider-neutral and drives off the graph runtime.

pub mod store;
mod tool;
mod types;

pub use tool::{GoalTool, GoalToolKind, goal_tools, register_goal_tools};
pub use types::{
    GoalProgress, ThreadGoal, ThreadGoalStatus, TurnOutcome, active_goal_context_block,
};

#[cfg(test)]
mod test;
