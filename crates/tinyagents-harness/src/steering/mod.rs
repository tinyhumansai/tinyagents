//! Policy-checked, observable orchestrator → sub-agent steering.
//!
//! In the recursive architecture this is how a *parent* in the run tree exerts
//! typed control over a *child* it is currently running: an orchestrating agent
//! (or human, or graph supervisor) injects [`SteeringCommand`]s into a live
//! sub-agent loop without killing or restarting it. It is the mid-run
//! counterpart to [`crate::subagent::SubAgentSession`] reuse (which
//! resumes a *completed* child) — together they cover both ways an orchestrator
//! keeps a sub-agent "in play".
//!
//! Steering lets an orchestrator (a parent agent, a human UI, a graph
//! supervisor, or a test harness) guide an already-running agent loop without
//! breaking its run identity or observability. The flow is:
//!
//! 1. The orchestrator builds a [`SteeringPolicy`] allowlist and a
//!    [`SteeringHandle`], and attaches the handle to the run's
//!    [`RunContext`][crate::context::RunContext] via
//!    [`RunContext::with_steering`][crate::context::RunContext::with_steering].
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
//! use tinyagents_harness::context::{RunConfig, RunContext};
//! use tinyagents_harness::message::Message;
//! use tinyagents_harness::steering::{
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

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::message::Message;

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
                paused: Mutex::new(None),
                checkpoints: Mutex::new(0),
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
    ///
    /// Queue accessors recover from a poisoned mutex (a panic in another
    /// holder) instead of panicking: the queue is a plain `VecDeque` with no
    /// invariants that a panicking holder could break mid-update.
    pub fn send(&self, command: SteeringCommand) {
        self.inner
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(command);
    }

    /// Removes and returns all currently queued commands in FIFO order, leaving
    /// the queue empty. Called by the agent loop at each checkpoint.
    pub fn drain(&self) -> Vec<SteeringCommand> {
        let mut queue = self
            .inner
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queue.drain(..).collect()
    }

    /// Returns `true` when no commands are currently queued.
    pub fn is_empty(&self) -> bool {
        self.inner
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }

    /// Returns the number of commands currently queued.
    pub fn pending(&self) -> usize {
        self.inner
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Returns the policy gating this handle.
    pub fn policy(&self) -> &SteeringPolicy {
        &self.inner.policy
    }

    /// Returns the latched [`PauseState`] when the run is paused, `None`
    /// otherwise.
    ///
    /// This is how a caller distinguishes a run that stopped for a human from a
    /// run that finished with nothing to say: on a
    /// [`SteeringOutcome::Pause`] this is always `Some`.
    pub fn pause_state(&self) -> Option<PauseState> {
        self.lock_paused().clone()
    }

    /// Returns `true` when a pause is latched.
    pub fn is_paused(&self) -> bool {
        self.lock_paused().is_some()
    }

    /// Latches a pause with an optional reason, recorded as having taken effect
    /// at `checkpoint`. Idempotent: an existing pause keeps its original reason
    /// and checkpoint, so a repeated `Pause` does not rewrite why the run
    /// stopped.
    fn latch_pause(&self, checkpoint: usize, reason: Option<String>) -> PauseState {
        let mut paused = self.lock_paused();
        let state = paused.get_or_insert(PauseState {
            reason,
            paused_at_checkpoint: checkpoint,
        });
        tinyagents_tracing::debug!(
            target: "tinyagents::steering",
            checkpoint = state.paused_at_checkpoint,
            reason = state.reason.as_deref(),
            "[steering] pause latched"
        );
        state.clone()
    }

    /// Clears any latched pause, returning the state that was cleared.
    ///
    /// Equivalent to delivering a [`SteeringCommand::Resume`]; exposed directly
    /// so a host UI can resume without going through the queue. A no-op when no
    /// pause is in effect.
    pub fn resume(&self) -> Option<PauseState> {
        let cleared = self.lock_paused().take();
        if cleared.is_some() {
            tinyagents_tracing::debug!(target: "tinyagents::steering", "[steering] pause cleared by resume");
        }
        cleared
    }

    /// Returns this checkpoint's zero-based index and advances the counter, so
    /// the *current* checkpoint is what a pause records (not the next one).
    fn advance_checkpoint(&self) -> usize {
        let mut checkpoints = self
            .inner
            .checkpoints
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = *checkpoints;
        *checkpoints += 1;
        current
    }

    /// Locks the pause latch, recovering from poisoning (see
    /// [`SteeringHandle::send`]).
    fn lock_paused(&self) -> std::sync::MutexGuard<'_, Option<PauseState>> {
        self.inner
            .paused
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
/// - When `ctx` has no [`SteeringHandle`], returns
///   [`SteeringOutcome::Continue`] without emitting anything.
/// - The batch is **validated in full before anything is applied**. If any
///   command is disallowed, an [`AgentEvent::Steered`] with `accepted = false`
///   is emitted for it and [`TinyAgentsError::Steering`] is returned — with the
///   working transcript and run metadata completely untouched. (It used to
///   validate lazily while applying, so a rejected command at position *n* left
///   commands `0..n` already applied, commands after it dropped, and the run
///   erroring: a partially-steered run and no way to reason about its state.)
/// - [`SteeringCommand::Cancel`] takes precedence: it is applied (emitting an
///   accepted event) and the function returns [`SteeringOutcome::Cancel`]
///   immediately, ignoring the rest of the batch.
/// - [`SteeringCommand::Pause`] / [`SteeringCommand::PauseWith`] latch a pause
///   **on the handle**, so it survives past this batch. Any later
///   [`SteeringCommand::Resume`] — in this batch or a subsequent one — clears
///   it. While latched, every checkpoint returns [`SteeringOutcome::Pause`]
///   even with an empty queue.
/// - [`SteeringCommand::InjectMessage`] and [`SteeringCommand::Redirect`]
///   append to `messages`; [`SteeringCommand::SetMetadata`] replaces
///   `ctx.config.metadata`.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Steering`] when any drained command is not
/// permitted by the run's [`SteeringPolicy`]. No command in the batch is
/// applied in that case.
pub fn apply_pending_steering<Ctx>(
    ctx: &mut RunContext<Ctx>,
    messages: &mut Vec<Message>,
) -> Result<SteeringOutcome> {
    // Clone the Arc-backed handle out so we do not hold a borrow of `ctx`
    // while we mutate its config/metadata below.
    let Some(handle) = ctx.steering.clone() else {
        return Ok(SteeringOutcome::Continue);
    };
    let checkpoint = handle.advance_checkpoint();
    let commands = handle.drain();

    // ── Phase 1: validate the whole batch, mutating nothing ─────────────────
    //
    // A policy violation must abort the checkpoint *atomically*. Checking as we
    // apply means the run dies with some of the batch already in the
    // transcript.
    if let Some(rejected) = commands
        .iter()
        .map(SteeringCommand::kind)
        .find(|kind| !handle.policy().is_allowed(*kind))
    {
        tinyagents_tracing::debug!(
            target: "tinyagents::steering",
            checkpoint,
            command_kind = rejected.as_str(),
            batch_size = commands.len(),
            "[steering] batch rejected by policy; nothing applied"
        );
        ctx.emit(AgentEvent::Steered {
            command_kind: rejected.as_str().to_string(),
            accepted: false,
        });
        return Err(TinyAgentsError::Steering(format!(
            "steering command `{}` is not permitted by the run policy",
            rejected.as_str()
        )));
    }

    // ── Phase 2: apply ──────────────────────────────────────────────────────
    tinyagents_tracing::debug!(
        target: "tinyagents::steering",
        checkpoint,
        batch_size = commands.len(),
        already_paused = handle.is_paused(),
        "[steering] applying checkpoint batch"
    );
    for command in commands {
        let kind = command.kind();

        match command {
            SteeringCommand::Pause => {
                handle.latch_pause(checkpoint, None);
            }
            SteeringCommand::PauseWith { reason } => {
                handle.latch_pause(checkpoint, Some(reason));
            }
            SteeringCommand::Resume => {
                handle.resume();
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

    // The latch — not this batch — decides the outcome, so a pause applied at an
    // earlier checkpoint keeps holding the run.
    if handle.is_paused() {
        Ok(SteeringOutcome::Pause)
    } else {
        Ok(SteeringOutcome::Continue)
    }
}

#[cfg(test)]
mod test;
