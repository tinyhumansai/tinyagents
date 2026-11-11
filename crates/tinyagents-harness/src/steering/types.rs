//! Type definitions for orchestrator → sub-agent steering.
//!
//! Steering is runtime control sent to an already-running agent loop. An
//! orchestrator (a parent agent, a human UI, a graph supervisor, or a test
//! harness) holds a [`SteeringHandle`] and enqueues [`SteeringCommand`]s on it;
//! the agent loop drains the handle at a safe checkpoint (before each model
//! call) and applies the commands the run's [`SteeringPolicy`] permits.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::steering` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs` files.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::ids::RunId;
use tinyinference_llm::message::Message;

/// Which run in the recursion tree a queued [`SteeringCommand`] is addressed
/// to.
///
/// Every [`SteeringHandle`] clone shares one underlying queue (so an
/// orchestrator can hand a single handle to a deeply nested tree and still
/// reach any run in it), but each level of the tree only *drains* the entries
/// addressed to itself — see `SteeringHandle::for_child`. Without this, a
/// command meant for the orchestrating run could be consumed by whichever
/// sub-agent happened to reach a checkpoint first.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SteeringTarget {
    /// The root run of the tree this handle belongs to. This is the default
    /// target for [`SteeringHandle::send`].
    Root,
    /// A specific run, named by [`RunId`].
    Run(RunId),
    /// Matched by any run sharing this handle (root or any descendant).
    ///
    /// Delivery is still pull-and-consume-once, exactly like every other
    /// target: whichever run's checkpoint drains the queue first removes the
    /// entry, so `All` is not a broadcast to every run in the tree — it only
    /// widens *which* run may claim the command, not how many do.
    All,
}

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
#[non_exhaustive]
// Keep `InjectMessage(Message)` source-compatible: boxing it would break the
// public command API, and commands are short-lived control messages.
#[allow(clippy::large_enum_variant)]
pub enum SteeringCommand {
    /// Cooperatively pause the run: the loop stops issuing further model and
    /// tool work at the next checkpoint, and **stays** paused until a
    /// [`SteeringCommand::Resume`] arrives — in this batch or any later one.
    ///
    /// The pause is latched on the [`SteeringHandle`], not on the batch. It used
    /// to be batch-scoped, which made a pause unresumable in practice: a
    /// `Resume` sent after the pause had already been applied found nothing to
    /// clear.
    Pause,

    /// Pause with a human-readable reason recorded in the resulting
    /// [`PauseState`], for example `"waiting for human approval of the refund"`.
    ///
    /// Identical to [`SteeringCommand::Pause`] in every other respect,
    /// including its [`SteeringCommandKind::Pause`] policy gate — a policy that
    /// allows one allows the other.
    PauseWith {
        /// Why the run was paused. Surfaced to the caller through
        /// [`PauseState::reason`].
        reason: String,
    },

    /// Clear a latched pause so the loop continues. A `Resume` with no pause in
    /// effect is a no-op.
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
    /// [`crate::context::RunConfig::metadata`].
    SetMetadata {
        /// The new metadata value.
        metadata: serde_json::Value,
    },

    /// Switch the model used for every subsequent model call in this run.
    ///
    /// `model` is a registry name, the same namespace as
    /// [`crate::retry::FallbackPolicy::models`] and
    /// [`tinyinference_llm::model::ModelRequest::model`]. The command is
    /// recorded at the steering checkpoint and applied at the **next model-call
    /// boundary, before the model binding is resolved**, so the call's
    /// `ModelStarted`/`ModelFailed` events, `ctx.model_profile`, the
    /// cross-provider handoff transform, the host budget estimate and the
    /// tool-dialect decision all see the new model.
    ///
    /// The switch is **sticky** for the rest of the run (the latest accepted
    /// one wins) and does not touch the transcript. A name that is unknown,
    /// ineligible for the call's capability/lifecycle gate, or addressed to a
    /// host-routed run is rejected: the run keeps its current model and emits
    /// [`crate::events::AgentEvent::ModelOverrideSkipped`] plus an
    /// [`crate::events::AgentEvent::Steered`] with `accepted = false`. A blank
    /// name is rejected immediately.
    ///
    /// # Opt-in
    ///
    /// A switch changes cost, rate limits and which provider sees the
    /// transcript, so [`SteeringPolicy::allow_all`] deliberately does **not**
    /// grant it. Use [`SteeringPolicy::allow_all_with_model_switch`] or
    /// `allow(SteeringCommandKind::SwitchModel)`.
    SwitchModel {
        /// Registry name of the model to switch to.
        model: String,
    },
}

impl SteeringCommand {
    /// Returns the policy-relevant [`SteeringCommandKind`] of this command.
    pub fn kind(&self) -> SteeringCommandKind {
        match self {
            SteeringCommand::Pause | SteeringCommand::PauseWith { .. } => {
                SteeringCommandKind::Pause
            }
            SteeringCommand::Resume => SteeringCommandKind::Resume,
            SteeringCommand::Cancel => SteeringCommandKind::Cancel,
            SteeringCommand::InjectMessage(_) => SteeringCommandKind::InjectMessage,
            SteeringCommand::Redirect { .. } => SteeringCommandKind::Redirect,
            SteeringCommand::SetMetadata { .. } => SteeringCommandKind::SetMetadata,
            SteeringCommand::SwitchModel { .. } => SteeringCommandKind::SwitchModel,
        }
    }
}

/// A payload-free discriminant for a [`SteeringCommand`], used to build a
/// [`SteeringPolicy`] allowlist and to label observability events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
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
    /// See [`SteeringCommand::SwitchModel`]. Not part of
    /// [`SteeringPolicy::allow_all`]; it needs an explicit opt-in.
    SwitchModel,
}

impl SteeringCommandKind {
    /// Every steering command kind, in declaration order.
    pub const ALL: [SteeringCommandKind; 7] = [
        SteeringCommandKind::Pause,
        SteeringCommandKind::Resume,
        SteeringCommandKind::Cancel,
        SteeringCommandKind::InjectMessage,
        SteeringCommandKind::Redirect,
        SteeringCommandKind::SetMetadata,
        SteeringCommandKind::SwitchModel,
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
            SteeringCommandKind::SwitchModel => "switch_model",
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
///
/// Deliberately still `Copy` and payload-free: the agent loop matches on it
/// directly, and widening a variant would break every one of those call sites
/// for no gain. The *state* behind a [`SteeringOutcome::Pause`] lives on the
/// [`SteeringHandle`] — read it with [`SteeringHandle::pause_state`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteeringOutcome {
    /// No steering, or only transcript/metadata mutations: continue the loop.
    Continue,
    /// A pause is latched: the loop should cooperatively stop and report the
    /// run as **paused**, not completed.
    ///
    /// # Contract for the agent loop (wave 2)
    ///
    /// The loop currently treats this as a bare `break`, which falls through to
    /// the success epilogue and reports the run completed with
    /// `final_response: None`. A caller cannot then tell "paused waiting for a
    /// human" from "the model produced an empty answer". On this outcome the
    /// loop must instead:
    ///
    /// 1. Read [`SteeringHandle::pause_state`] from `ctx.steering` — it is
    ///    always `Some` when this outcome is returned — and surface the
    ///    [`PauseState`] (reason, checkpoint index) to the caller.
    /// 2. Report the run as paused/interrupted rather than completed
    ///    (`HarnessRunStatus::mark_interrupted`, or the crate's
    ///    `Interrupted` shape) so it is distinguishable from success.
    /// 3. Leave the pause latched. It is *not* cleared by breaking out of the
    ///    loop: sending [`SteeringCommand::Resume`] on the same handle clears
    ///    it, and re-invoking the run continues from the checkpoint.
    Pause,
    /// A cancel was requested: the loop should terminate the run.
    Cancel,
}

impl SteeringOutcome {
    /// `true` when the loop should cooperatively stop for a pause.
    pub fn is_pause(self) -> bool {
        matches!(self, SteeringOutcome::Pause)
    }
}

/// The latched state behind a [`SteeringOutcome::Pause`].
///
/// A pause used to be scoped to the drained batch: [`SteeringCommand::Resume`]
/// only cleared a [`SteeringCommand::Pause`] that arrived in the *same* batch,
/// so a pause applied at one checkpoint could never be lifted — the run was
/// stuck. The state now lives on the [`SteeringHandle`], so a `Resume` sent at
/// any later moment resumes the run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PauseState {
    /// Why the run was paused, when the orchestrator supplied one via
    /// [`SteeringCommand::PauseWith`]. `None` for a bare
    /// [`SteeringCommand::Pause`].
    pub reason: Option<String>,
    /// Zero-based index of the steering checkpoint at which the pause took
    /// effect, i.e. how many checkpoints this handle had already processed.
    /// Lets a caller (and a resumed run) report *where* the run stopped.
    pub paused_at_checkpoint: usize,
}

/// A cloneable, thread-safe handle to a running agent's steering queue.
///
/// An orchestrator holds a `SteeringHandle` and calls [`SteeringHandle::send`]
/// to enqueue commands; the same handle (attached to the run's
/// [`crate::context::RunContext`]) is drained by the agent loop via
/// [`SteeringHandle::drain`]. All clones share one underlying queue and policy
/// through an `Arc<Mutex<…>>`, so the sender and the receiver are the same type
/// and there is no separate receiver to wire up.
///
/// The handle is std-only — it carries no async runtime dependency. Delivery is
/// pull-based: enqueued commands become visible to the loop on its next
/// checkpoint, never mid-stream.
///
/// # Routing
///
/// A plain `clone()` is a bare alias: it shares this handle's identity
/// (`run_id`/`is_root`) as well as its queue, so it drains exactly the same
/// commands this handle would. When a run spawns a child,
/// [`crate::context::RunContext::child`] calls `SteeringHandle::for_child`
/// (not `clone`) so the child only drains commands addressed to it or to
/// [`SteeringTarget::All`] — see that method's docs.
#[derive(Clone)]
pub struct SteeringHandle {
    pub(crate) inner: Arc<SteeringInner>,
    /// The identity of the run *this handle instance* drains for.
    pub(crate) run_id: RunId,
    /// Whether `run_id` is the root of the steering tree, for matching
    /// [`SteeringTarget::Root`].
    pub(crate) is_root: bool,
    /// This handle's own pause/checkpoint state. Deliberately **not** shared
    /// with a parent/child handle derived via [`SteeringHandle::for_child`]:
    /// a pause addressed to one run must not latch every run sharing the
    /// underlying queue.
    pub(crate) local: Arc<SteeringLocal>,
}

/// Shared interior of a [`SteeringHandle`]: the queue and policy every level
/// of a steering tree drains from.
pub(crate) struct SteeringInner {
    /// FIFO queue of pending, addressed commands.
    pub(crate) queue: Mutex<VecDeque<(SteeringTarget, SteeringCommand)>>,
    /// The allowlist gating which drained commands may be applied.
    pub(crate) policy: SteeringPolicy,
}

/// Per-run steering state: **not** shared across a [`SteeringHandle::for_child`]
/// boundary, so a pause or checkpoint count is scoped to the run it belongs to.
#[derive(Default)]
pub(crate) struct SteeringLocal {
    /// The latched pause, if one is in effect. Survives across checkpoints so a
    /// [`SteeringCommand::Resume`] delivered in a *later* batch can lift it.
    pub(crate) paused: Mutex<Option<PauseState>>,
    /// How many steering checkpoints this handle has processed. Recorded into
    /// [`PauseState::paused_at_checkpoint`].
    pub(crate) checkpoints: Mutex<usize>,
    /// Pause commands applied since the last resume, so the loop can tell a
    /// guard's lone pause from one a host sent at the same checkpoint.
    pub(crate) pause_requests: Mutex<usize>,
    /// The model a [`SteeringCommand::SwitchModel`] selected for *this* run,
    /// sticky until replaced or rejected by the model call. Per run like the
    /// pause latch, so it never crosses a [`SteeringHandle::for_child`]
    /// boundary in either direction.
    pub(crate) model_override: Mutex<Option<String>>,
    /// Whether the current [`Self::model_override`] has already been reported
    /// as applied (`Steered { accepted: true }`), so a sticky switch is
    /// announced once, not on every model call. Reset by every new switch.
    pub(crate) model_override_announced: std::sync::atomic::AtomicBool,
}

/// Bounded memory of recently applied steering `request_id`s, used to make a
/// retried steering request idempotent.
///
/// A steering caller that retries (a flaky transport, a model repeating a tool
/// call) attaches the same `request_id` each time; the owner of the steered
/// task keeps one `RecentRequestIds` per task and delivers a request only when
/// [`Self::claim`] reports it as new. Only the most recent
/// [`Self::DEFAULT_CAPACITY`] ids are remembered by default, so memory per task
/// stays constant however long it runs.
#[derive(Clone, Debug)]
pub struct RecentRequestIds {
    pub(crate) capacity: usize,
    pub(crate) order: VecDeque<String>,
}
