//! Type definitions for orchestrator → sub-agent steering.
//!
//! Steering is runtime control sent to an already-running agent loop. An
//! orchestrator (a parent agent, a human UI, a graph supervisor, or a test
//! harness) holds a [`SteeringHandle`] and enqueues [`SteeringCommand`]s on it;
//! the agent loop drains the handle at a safe checkpoint (before each model
//! call) and applies the commands the run's [`SteeringPolicy`] permits.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::harness::steering` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs` files.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::harness::message::Message;

/// A typed runtime control instruction delivered to a running agent loop.
///
/// Commands are enqueued on a [`SteeringHandle`] by an orchestrator and drained
/// by the agent loop at the next safe checkpoint. Each command is gated by the
/// run's [`SteeringPolicy`]; a command whose [`SteeringCommandKind`] is not in
/// the allowlist is rejected with
/// [`crate::error::TinyAgentsError::Steering`].
///
/// `SteeringCommand` is `Serialize`/`Deserialize` so steering can be described,
/// logged, transported across a control channel, and replayed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "command")]
pub enum SteeringCommand {
    /// Cooperatively pause the run: the loop stops issuing further model and
    /// tool work at the next checkpoint until a [`SteeringCommand::Resume`] is
    /// delivered in the same drained batch.
    Pause,

    /// Clear a pending pause so the loop continues. A `Resume` with no
    /// preceding `Pause` in the same batch is a no-op.
    Resume,

    /// Terminate the run cooperatively at the next checkpoint. Cancel takes
    /// precedence over every other command in the same batch and surfaces as
    /// [`crate::error::TinyAgentsError::Cancelled`].
    Cancel,

    /// Inject a structured instruction into the running agent's working
    /// transcript so the next model call sees it. The message carries explicit
    /// provenance through its role rather than being anonymous user text.
    InjectMessage(Message),

    /// Redirect the agent toward a new instruction. Lowered into a system
    /// message (`[steering:redirect] {instruction}`) appended to the working
    /// transcript before the next model call.
    Redirect {
        /// Human- or orchestrator-authored redirection instruction.
        instruction: String,
    },

    /// Replace the run's free-form metadata blob (for example to record an
    /// orchestrator decision or a human review tag). Applied to the live
    /// [`crate::harness::context::RunConfig::metadata`].
    SetMetadata {
        /// The new metadata value.
        metadata: serde_json::Value,
    },
}

impl SteeringCommand {
    /// Returns the policy-relevant [`SteeringCommandKind`] of this command.
    pub fn kind(&self) -> SteeringCommandKind {
        match self {
            SteeringCommand::Pause => SteeringCommandKind::Pause,
            SteeringCommand::Resume => SteeringCommandKind::Resume,
            SteeringCommand::Cancel => SteeringCommandKind::Cancel,
            SteeringCommand::InjectMessage(_) => SteeringCommandKind::InjectMessage,
            SteeringCommand::Redirect { .. } => SteeringCommandKind::Redirect,
            SteeringCommand::SetMetadata { .. } => SteeringCommandKind::SetMetadata,
        }
    }
}

/// A payload-free discriminant for a [`SteeringCommand`], used to build a
/// [`SteeringPolicy`] allowlist and to label observability events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteeringCommandKind {
    /// See [`SteeringCommand::Pause`].
    Pause,
    /// See [`SteeringCommand::Resume`].
    Resume,
    /// See [`SteeringCommand::Cancel`].
    Cancel,
    /// See [`SteeringCommand::InjectMessage`].
    InjectMessage,
    /// See [`SteeringCommand::Redirect`].
    Redirect,
    /// See [`SteeringCommand::SetMetadata`].
    SetMetadata,
}

impl SteeringCommandKind {
    /// Every steering command kind, in declaration order.
    pub const ALL: [SteeringCommandKind; 6] = [
        SteeringCommandKind::Pause,
        SteeringCommandKind::Resume,
        SteeringCommandKind::Cancel,
        SteeringCommandKind::InjectMessage,
        SteeringCommandKind::Redirect,
        SteeringCommandKind::SetMetadata,
    ];

    /// Returns a stable, lower-snake-case name for this kind, suitable for
    /// logging and event labels (e.g. `"inject_message"`).
    pub fn as_str(self) -> &'static str {
        match self {
            SteeringCommandKind::Pause => "pause",
            SteeringCommandKind::Resume => "resume",
            SteeringCommandKind::Cancel => "cancel",
            SteeringCommandKind::InjectMessage => "inject_message",
            SteeringCommandKind::Redirect => "redirect",
            SteeringCommandKind::SetMetadata => "set_metadata",
        }
    }
}

/// An allowlist of the [`SteeringCommandKind`]s a run will accept.
///
/// The policy is conservative by default: [`SteeringPolicy::new`] permits
/// nothing, so a run that opts into steering must explicitly grant the kinds it
/// trusts. The agent loop consults the policy for every drained command and
/// rejects disallowed ones with [`crate::error::TinyAgentsError::Steering`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SteeringPolicy {
    /// The set of permitted command kinds.
    pub(crate) allowed: HashSet<SteeringCommandKind>,
}

/// The control-flow decision produced by applying a batch of steering commands
/// at a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteeringOutcome {
    /// No steering, or only transcript/metadata mutations: continue the loop.
    Continue,
    /// A net pause is in effect: the loop should cooperatively stop.
    Pause,
    /// A cancel was requested: the loop should terminate the run.
    Cancel,
}

/// A cloneable, thread-safe handle to a running agent's steering queue.
///
/// An orchestrator holds a `SteeringHandle` and calls [`SteeringHandle::send`]
/// to enqueue commands; the same handle (attached to the run's
/// [`crate::harness::context::RunContext`]) is drained by the agent loop via
/// [`SteeringHandle::drain`]. All clones share one underlying queue and policy
/// through an `Arc<Mutex<…>>`, so the sender and the receiver are the same type
/// and there is no separate receiver to wire up.
///
/// The handle is std-only — it carries no async runtime dependency. Delivery is
/// pull-based: enqueued commands become visible to the loop on its next
/// checkpoint, never mid-stream.
#[derive(Clone)]
pub struct SteeringHandle {
    pub(crate) inner: Arc<SteeringInner>,
}

/// Shared interior of a [`SteeringHandle`].
pub(crate) struct SteeringInner {
    /// FIFO queue of pending commands.
    pub(crate) queue: Mutex<VecDeque<SteeringCommand>>,
    /// The allowlist gating which drained commands may be applied.
    pub(crate) policy: SteeringPolicy,
}
