//! Identifier newtypes and lifecycle enums for the harness.
//!
//! Every long-lived harness concept (runs, threads, calls, events, graph
//! nodes, checkpoints, interrupts) is keyed by a small, cheap-to-clone string
//! newtype. Wrapping the raw `String` in a distinct type prevents accidentally
//! passing, say, a [`ThreadId`] where a [`RunId`] is expected.

use serde::{Deserialize, Serialize};

/// Identifies a single harness run (one model call, one agent loop, or one
/// graph-node invocation of the harness).
///
/// A `RunId` is the unit of recursion in TinyAgents: when a run spawns a
/// sub-agent or sub-graph, the child gets its own `RunId` while
/// [`crate::harness::events::HarnessRunStatus`] records that child's
/// `parent_run_id` (the spawning run) and `root_run_id` (the top-level
/// ancestor), so the full recursion tree is reconstructable from these ids.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(pub(crate) String);

/// Identifies a conversation thread that may span many runs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ThreadId(pub(crate) String);

/// Identifies an individual model or tool call inside a run.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallId(pub(crate) String);

/// Identifies a single emitted harness event.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub(crate) String);

/// Identifies a harness component such as a model, tool, or middleware.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId(pub(crate) String);

/// Identifies a state graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphId(pub(crate) String);

/// Identifies a node within a state graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub(crate) String);

/// Identifies a unit of scheduled graph work (a node activation or a `Send`
/// fanout task) within a run.
///
/// A `TaskId` distinguishes the individual recursive tasks a graph schedules —
/// for example each child task produced by a `Send` fanout — so a
/// [`crate::graph::recursion::RecursionFrame`] can name the exact task a nested
/// call descends from, independent of the [`NodeId`] it ran.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub(crate) String);

/// Identifies a single interactive `.ragsh` REPL session.
///
/// A `SessionId` names one [`crate::repl`] session: a persistent namespace and
/// capability boundary that a (possibly model-driven) orchestrator drives one
/// cell at a time. Pairing it with a [`RunId`] lets REPL events and the child
/// model/tool/graph runs a session spawns be correlated back to the session
/// that issued them.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub(crate) String);

/// Identifies a single executed cell within a [`SessionId`] REPL session.
///
/// Each `.ragsh` cell evaluated against a session is named by a `CellId` so
/// stdout chunks, variable changes, capability calls, and diagnostics can be
/// attributed to the exact cell that produced them.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellId(pub(crate) String);

/// Identifies a persisted graph checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointId(pub(crate) String);

/// Identifies a human-in-the-loop interrupt.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InterruptId(pub(crate) String);

/// Coarse lifecycle status shared by direct model calls, agent loops, and
/// graph-node harness invocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    /// Created but not yet started.
    Pending,
    /// Actively executing.
    Running,
    /// Paused awaiting external input (for example a human interrupt).
    Interrupted,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Cancelled before completion.
    Cancelled,
}

/// The active operation within a harness run, used for compact status reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessPhase {
    /// No work in progress.
    Idle,
    /// Assembling the model request from messages, tools, and config.
    BuildingRequest,
    /// Awaiting a model response.
    Model,
    /// Executing tool calls.
    Tools,
    /// Running middleware hooks.
    Middleware,
    /// Persisting memory, events, or status.
    Persisting,
    /// Run finished.
    Done,
}
