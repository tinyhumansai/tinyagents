//! Policy-checked, observable orchestrator → sub-agent steering.
//!
//! Steering lets an orchestrator (a parent agent, a human UI, a graph
//! supervisor, or a test harness) guide an already-running agent loop without
//! breaking its run identity or observability. The flow is:
//!
//! 1. The orchestrator builds a [`SteeringPolicy`] allowlist and a
//!    [`SteeringHandle`], and attaches the handle to the run's
//!    [`RunContext`][crate::harness::context::RunContext] via
//!    [`RunContext::with_steering`][crate::harness::context::RunContext::with_steering].
//! 2. While the run executes, the orchestrator calls
//!    [`SteeringHandle::send`] to enqueue [`SteeringCommand`]s.
//! 3. The agent loop, at a safe checkpoint (before each model call), calls
//!    [`apply_pending_steering`] which drains the handle, checks each command
//!    against the policy, applies the permitted ones, and emits an
//!    [`AgentEvent::Steered`] for every command.
//!
//! Delivery is conservative and pull-based: commands become visible only at the
//! checkpoint, never in the middle of a provider stream or a side-effecting
//! tool call.
//!
//! # Example
//!
//! ```
//! use tinyagents::harness::context::{RunConfig, RunContext};
//! use tinyagents::harness::message::Message;
//! use tinyagents::harness::steering::{
//!     apply_pending_steering, SteeringCommand, SteeringCommandKind, SteeringHandle,
//!     SteeringOutcome, SteeringPolicy,
//! };
//!
//! let policy = SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage);
//! let handle = SteeringHandle::new(policy);
//! handle.send(SteeringCommand::InjectMessage(Message::user("focus on billing")));
//!
//! let mut ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ())
//!     .with_steering(handle.clone());
//!
//! let mut messages = vec![Message::user("start")];
//! let outcome = apply_pending_steering(&mut ctx, &mut messages).unwrap();
//! assert_eq!(outcome, SteeringOutcome::Continue);
//! // The injected instruction is now visible to the next model call.
//! assert_eq!(messages.len(), 2);
//! ```

mod types;

pub use types::*;

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::error::{Result, TinyAgentsError};
use crate::harness::context::RunContext;
use crate::harness::events::AgentEvent;
use crate::harness::message::Message;

// ── SteeringPolicy ────────────────────────────────────────────────────────────

impl SteeringPolicy {
    /// Creates an empty policy that permits **no** steering commands.
    ///
    /// Grant kinds explicitly with [`SteeringPolicy::allow`] or start from
    /// [`SteeringPolicy::allow_all`].
    pub fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Creates a policy that permits every [`SteeringCommandKind`].
    pub fn allow_all() -> Self {
        Self {
            allowed: SteeringCommandKind::ALL.into_iter().collect(),
        }
    }

    /// Adds `kind` to the allowlist (builder style) and returns the policy.
    pub fn allow(mut self, kind: SteeringCommandKind) -> Self {
        self.allowed.insert(kind);
        self
    }

    /// Returns `true` when `kind` is permitted by this policy.
    pub fn is_allowed(&self, kind: SteeringCommandKind) -> bool {
        self.allowed.contains(&kind)
    }
}

// ── SteeringHandle ──────────────────────────────────────────────────────────

impl SteeringHandle {
    /// Builds a handle backed by a fresh, empty queue gated by `policy`.
    pub fn new(policy: SteeringPolicy) -> Self {
        Self {
            inner: Arc::new(SteeringInner {
                queue: Mutex::new(VecDeque::new()),
                policy,
            }),
        }
    }

    /// Convenience constructor for a handle whose policy permits every command
    /// kind. Equivalent to `SteeringHandle::new(SteeringPolicy::allow_all())`.
    pub fn allow_all() -> Self {
        Self::new(SteeringPolicy::allow_all())
    }

    /// Enqueues `command` for delivery to the running agent loop.
    ///
    /// The command becomes visible to the loop at its next steering checkpoint;
    /// this method never blocks and does not itself check the policy.
    pub fn send(&self, command: SteeringCommand) {
        self.inner
            .queue
            .lock()
            .expect("steering queue mutex poisoned")
            .push_back(command);
    }

    /// Removes and returns all currently queued commands in FIFO order, leaving
    /// the queue empty. Called by the agent loop at each checkpoint.
    pub fn drain(&self) -> Vec<SteeringCommand> {
        let mut queue = self
            .inner
            .queue
            .lock()
            .expect("steering queue mutex poisoned");
        queue.drain(..).collect()
    }

    /// Returns `true` when no commands are currently queued.
    pub fn is_empty(&self) -> bool {
        self.inner
            .queue
            .lock()
            .expect("steering queue mutex poisoned")
            .is_empty()
    }

    /// Returns the number of commands currently queued.
    pub fn pending(&self) -> usize {
        self.inner
            .queue
            .lock()
            .expect("steering queue mutex poisoned")
            .len()
    }

    /// Returns the policy gating this handle.
    pub fn policy(&self) -> &SteeringPolicy {
        &self.inner.policy
    }
}

// ── Checkpoint application ────────────────────────────────────────────────────

/// Drains any steering commands attached to `ctx`, applies the policy-permitted
/// ones to the working `messages` (and run metadata), and returns the resulting
/// control-flow [`SteeringOutcome`].
///
/// This is the single steering checkpoint used by the agent loop. It is also a
/// standalone, synchronous function so it can be unit-tested without a full
/// run. Behaviour:
///
/// - When `ctx` has no [`SteeringHandle`] (or its queue is empty), returns
///   [`SteeringOutcome::Continue`] without emitting anything.
/// - Every drained command is checked against the handle's
///   [`SteeringPolicy`]. A disallowed command emits an
///   [`AgentEvent::Steered`] with `accepted = false` and returns
///   [`TinyAgentsError::Steering`], aborting the run; no later command in the
///   batch is applied.
/// - [`SteeringCommand::Cancel`] takes precedence: it is applied (emitting an
///   accepted event) and the function returns [`SteeringOutcome::Cancel`]
///   immediately, ignoring the rest of the batch.
/// - [`SteeringCommand::Pause`] sets a net-pause outcome; a later
///   [`SteeringCommand::Resume`] in the same batch clears it.
/// - [`SteeringCommand::InjectMessage`] and [`SteeringCommand::Redirect`]
///   append to `messages`; [`SteeringCommand::SetMetadata`] replaces
///   `ctx.config.metadata`.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Steering`] when a drained command is not
/// permitted by the run's [`SteeringPolicy`].
pub fn apply_pending_steering<Ctx>(
    ctx: &mut RunContext<Ctx>,
    messages: &mut Vec<Message>,
) -> Result<SteeringOutcome> {
    // Clone the Arc-backed handle out so we do not hold a borrow of `ctx`
    // while we mutate its config/metadata below.
    let Some(handle) = ctx.steering.clone() else {
        return Ok(SteeringOutcome::Continue);
    };
    let commands = handle.drain();
    if commands.is_empty() {
        return Ok(SteeringOutcome::Continue);
    }

    let mut outcome = SteeringOutcome::Continue;
    for command in commands {
        let kind = command.kind();

        if !handle.policy().is_allowed(kind) {
            ctx.emit(AgentEvent::Steered {
                command_kind: kind.as_str().to_string(),
                accepted: false,
            });
            return Err(TinyAgentsError::Steering(format!(
                "steering command `{}` is not permitted by the run policy",
                kind.as_str()
            )));
        }

        // Apply the permitted command.
        match command {
            SteeringCommand::Pause => outcome = SteeringOutcome::Pause,
            SteeringCommand::Resume => {
                if outcome == SteeringOutcome::Pause {
                    outcome = SteeringOutcome::Continue;
                }
            }
            SteeringCommand::Cancel => {
                ctx.emit(AgentEvent::Steered {
                    command_kind: kind.as_str().to_string(),
                    accepted: true,
                });
                // Cancel wins over everything else in the batch.
                return Ok(SteeringOutcome::Cancel);
            }
            SteeringCommand::InjectMessage(message) => messages.push(message),
            SteeringCommand::Redirect { instruction } => {
                messages.push(Message::system(format!(
                    "[steering:redirect] {instruction}"
                )));
            }
            SteeringCommand::SetMetadata { metadata } => {
                ctx.config.metadata = metadata;
            }
        }

        ctx.emit(AgentEvent::Steered {
            command_kind: kind.as_str().to_string(),
            accepted: true,
        });
    }

    Ok(outcome)
}

#[cfg(test)]
mod test;
