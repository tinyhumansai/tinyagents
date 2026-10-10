//! Type definitions for the harness run-context module.
//!
//! These types carry the recursion bookkeeping (depth / max-depth) and the
//! shared signals (cancellation, steering, events) that let a parent run and
//! its nested sub-runs behave as one coordinated tree.
//!
//! [`RunConfig`] is the serializable, declarative description of a run (its
//! identity, limits, and metadata). [`RunContext`] is the live, in-process
//! handle threaded through model calls, tool calls, middleware, and graph
//! nodes; it bundles the config with the dependencies a run needs (stores,
//! events, and limit tracking) plus arbitrary user data.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::context` directly. Implementations and tests live in the
//! sibling `mod.rs` and `test.rs`.

use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::events::EventSink;
use crate::ids::{CallId, RunId, ThreadId};
use crate::limits::LimitTracker;
use crate::steering::SteeringHandle;
use crate::store::StoreRegistry;

/// One-shot observer invoked with a cheap summary of the accumulated run when
/// a driver completes or is dropped. Kept crate-private: it is runtime
/// lifecycle glue, not a host policy extension point.
///
/// Takes [`TerminalRunSummary`], not the full [`crate::middleware::AgentRun`]
/// (M-6): every installed observer only ever reads the final text, usage, and
/// executed-tool names, never the full transcript, and the observer needs an
/// *owned* value (the hosted path moves it into a spawned task that can
/// outlive the caller's stack frame) — so `&AgentRun` will not do either. The
/// summary is `Clone` and carries none of `AgentRun::messages`, which can be
/// the largest field by far on a long-running conversation.
pub(crate) type TerminalObserver =
    Box<dyn FnOnce(TerminalRunSummary, bool, Option<String>) + Send + Sync + 'static>;

/// Cheap, owned summary of an [`crate::middleware::AgentRun`] for
/// [`TerminalObserver`] — see that type's docs for why this exists instead of
/// the full run.
#[derive(Clone, Debug, Default)]
pub(crate) struct TerminalRunSummary {
    /// The final response text, if the run produced one. Mirrors
    /// [`crate::middleware::AgentRun::text`].
    pub(crate) text: Option<String>,
    /// Cumulative token usage across the run. `Copy`, so cloning this summary
    /// is not where any cost lives.
    pub(crate) usage: tinyinference_llm::usage::UsageTotals,
    /// Names of calls that reached a tool executor, in execution order.
    /// Mirrors [`crate::middleware::AgentRun::executed_tools`].
    pub(crate) executed_tools: Vec<String>,
}

impl TerminalRunSummary {
    /// Builds a summary from a live run without cloning its transcript.
    pub(crate) fn from_run(run: &crate::middleware::AgentRun) -> Self {
        Self {
            text: run.text(),
            usage: run.usage,
            executed_tools: run.executed_tools.clone(),
        }
    }
}

/// The immutable ancestry of a run in a recursive harness invocation tree.
///
/// A lineage names the root run, the immediate parent (when this is a child),
/// and the depth cap shared by the complete tree.  It is deliberately data-only
/// so hosts can persist, display, or replay ancestry without retaining a live
/// [`RunContext`].  The live context remains the authority for cancellation,
/// stores, events, and other process-local capabilities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunLineage {
    /// The top-level run that began this recursive tree.
    pub root_run_id: RunId,
    /// The run that directly created this run, or `None` for the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<RunId>,
    /// This run's zero-based depth in the tree.
    pub depth: usize,
    /// Inclusive maximum permitted child depth for the tree.
    pub max_depth: usize,
}

/// Declarative, serializable configuration for a single harness run.
///
/// `RunConfig` captures everything that defines a run independent of live
/// runtime state: its identity, the thread it belongs to, classification tags,
/// free-form metadata, and the hard limits applied to it. Because it is
/// `Serialize`/`Deserialize`, a run can be described, stored, and replayed.
///
/// Construct one with [`RunConfig::new`] and refine it with the `with_*`
/// builder methods.
///
/// # Example
///
/// ```
/// use tinyagents_harness::context::RunConfig;
///
/// let config = RunConfig::new("run-1")
///     .with_thread("thread-7")
///     .with_tag("nightly")
///     .with_max_model_calls(10);
/// assert_eq!(config.run_id.as_str(), "run-1");
/// assert_eq!(config.max_model_calls, Some(10));
/// assert_eq!(config.effective_max_model_calls(), 10);
///
/// // An unset cap reads as `None` but still resolves to the crate default.
/// let defaulted = RunConfig::new("run-2");
/// assert_eq!(defaulted.max_model_calls, None);
/// assert_eq!(defaulted.effective_max_model_calls(), 25);
/// ```
#[derive(Clone, Debug, Serialize)]
pub struct RunConfig {
    /// Unique identifier for this run.
    pub run_id: RunId,
    /// Conversation thread this run belongs to, when threaded.
    pub thread_id: Option<ThreadId>,
    /// Free-form classification tags (for example `"nightly"`, `"eval"`).
    pub tags: Vec<String>,
    /// Arbitrary caller-supplied metadata. Defaults to JSON `null`.
    pub metadata: serde_json::Value,
    /// Wall-clock timeout in milliseconds. `None` means no deadline.
    pub timeout_ms: Option<u64>,
    /// Maximum number of model calls permitted for this run, when the caller
    /// set one explicitly.
    ///
    /// `None` means "unset": the run falls back to
    /// [`RunConfig::effective_max_model_calls`] (the crate-default cap of 25)
    /// and a harness-wide
    /// [`RunPolicy::limits`][crate::runtime::RunPolicy] is free to
    /// raise *or* lower it.
    ///
    /// # Why this is an `Option`
    ///
    /// The agent loop reconciles this cap with the harness policy's, and the
    /// two directions are only safe to distinguish when "the caller asked for
    /// 2" is distinguishable from "nobody asked, so it defaulted to 25":
    ///
    /// - **Explicitly set** (`Some`) → the loop takes the **stricter** of the
    ///   two caps, so a permissive policy default can never silently widen a
    ///   budget the caller deliberately narrowed.
    /// - **Unset** (`None`) → the policy is the only real source of truth and
    ///   wins outright, including when it raises the cap above the default.
    ///
    /// While this was a bare `usize` the loop could not tell those apart and
    /// resolved every case by plain assignment, so
    /// `RunConfig::new("r").with_max_model_calls(2)` against a default policy
    /// ran **25** model calls.
    #[serde(default)]
    pub max_model_calls: Option<usize>,
    /// Maximum number of tool invocations permitted for this run, when the
    /// caller set one explicitly. See [`RunConfig::max_model_calls`] for the
    /// set-versus-unset semantics; the tool cap follows the identical rule.
    #[serde(default)]
    pub max_tool_calls: Option<usize>,
    /// Maximum output tokens requested for each model turn in this run.
    ///
    /// When set, the agent loop applies this as an upper bound to
    /// [`tinyinference_llm::model::ModelRequest::max_tokens`] before dispatching a
    /// model call. Child sub-agent runs inherit the same cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turn_output_tokens: Option<u32>,
    /// Recursive ancestry and depth cap for this run.
    pub lineage: RunLineage,
}

/// Where [`MiddlewareControl::JumpTo`] sends the agent loop next.
///
/// Modelled on LangChain's `jump_to: "model" | "tools" | "end"`. See
/// `docs/modules/harness/middleware.md` for exactly how each target is
/// realized against the loop's checkpoint structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopTarget {
    /// Skip any remaining tool execution for this turn and go straight to the
    /// next model call.
    Model,
    /// Proceed to (or continue) tool execution for this turn. A no-op when the
    /// turn has no tool calls to run — there is nothing to jump to.
    Tools,
    /// Stop the loop now, finishing the run with the transcript as it stands.
    End,
}

/// A typed hook that mutates application state, carried by
/// [`MiddlewareControl::UpdateState`].
///
/// `State` is type-erased on construction (`RunContext` is not generic over
/// it) and recovered by [`Self::apply`] via a runtime check. Built with
/// [`StateUpdate::new`], which captures an `Fn(&mut State)` closure in an
/// `Arc` so [`MiddlewareControl`] (and therefore `StateUpdate`) stays
/// [`Clone`] — required because [`RunContext::request_control`] may compare
/// and replace a pending request.
///
/// The agent loop only ever sees `state: &State` (a shared reference), so it
/// cannot apply this itself. [`RunContext::take_state_updates`] queues every
/// requested update instead; a host that owns `&mut State` between runs (or
/// between turns, via its own checkpoint) drains and applies them. See
/// `docs/modules/harness/middleware.md` for the full contract.
/// The type-erased closure a [`StateUpdate`] wraps.
type ErasedStateUpdateFn = std::sync::Arc<dyn Fn(&mut dyn std::any::Any) + Send + Sync>;

#[derive(Clone)]
pub struct StateUpdate {
    apply: ErasedStateUpdateFn,
}

impl StateUpdate {
    /// Captures `f` as a state update for the concrete application state type
    /// `S`. Applying the update against any other type is a documented no-op
    /// (see [`Self::apply`]).
    pub fn new<S: 'static>(f: impl Fn(&mut S) + Send + Sync + 'static) -> Self {
        Self {
            apply: std::sync::Arc::new(move |state: &mut dyn std::any::Any| {
                if let Some(state) = state.downcast_mut::<S>() {
                    f(state);
                }
            }),
        }
    }

    /// Applies this update to `state` when `state`'s concrete type matches the
    /// type this update was constructed for. A mismatched type is a silent
    /// no-op: the update was requested by middleware generic over a different
    /// `State`, which a host wiring several harnesses together can otherwise
    /// hit legitimately.
    pub fn apply<S: 'static>(&self, state: &mut S) {
        (self.apply)(state as &mut dyn std::any::Any);
    }
}

impl std::fmt::Debug for StateUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StateUpdate(..)")
    }
}

/// A structured control outcome a middleware (or any step) can request on the
/// [`RunContext`] to steer the agent loop from outside its `Result<()>` return
/// channel.
///
/// This is the harness-native complement to the graph
/// `Command`/`Interrupt` vocabulary (in `tinyagents-graph`): the agent loop drains any requested control at its safe
/// checkpoints (after each model response) and acts on it, so behaviors like
/// "stop after an early-exit tool" or "pause on budget" no longer need a
/// bespoke side channel. Requests are visible via
/// [`RunContext::take_control`].
///
/// A [`Middleware`][crate::middleware::Middleware] hook may also *return* one
/// of these directly from its `_control`-suffixed variant (for example
/// [`before_model_control`][crate::middleware::Middleware::before_model_control]);
/// the [`MiddlewareStack`][crate::middleware::MiddlewareStack] resolves a
/// non-[`Continue`](Self::Continue) return into exactly the same
/// [`RunContext::request_control`] call a hook could have made explicitly —
/// returning control is sugar over the side channel, not a second mechanism.
#[derive(Clone, Debug)]
pub enum MiddlewareControl {
    /// No control requested. The default a `_control` hook returns when it has
    /// nothing to say; never itself installed as a pending request (see
    /// [`RunContext::request_control`]).
    Continue,
    /// Route the loop to `target` at the next safe checkpoint. See
    /// [`LoopTarget`] for what each target does.
    JumpTo(LoopTarget),
    /// Queue a typed state mutation for the host to apply. The loop itself
    /// only ever holds `&State`, so this is queued on
    /// [`RunContext::take_state_updates`] rather than applied in place; see
    /// [`StateUpdate`].
    UpdateState(StateUpdate),
    /// Stop the loop now and use this text as the final assistant response.
    StopWithFinal(String),
    /// Pause the run at the next safe checkpoint, surfacing
    /// [`crate::error::TinyAgentsError::Interrupted`] with this node/message so a
    /// caller can persist a checkpoint and resume later.
    Interrupt {
        /// Logical node/label the interrupt is attributed to.
        node: String,
        /// Human-readable reason surfaced with the interrupt.
        message: String,
    },
}

impl MiddlewareControl {
    /// A stable label for this control outcome, used in audit events.
    pub fn kind(&self) -> &'static str {
        match self {
            MiddlewareControl::Continue => "continue",
            MiddlewareControl::JumpTo(LoopTarget::Model) => "jump_to:model",
            MiddlewareControl::JumpTo(LoopTarget::Tools) => "jump_to:tools",
            MiddlewareControl::JumpTo(LoopTarget::End) => "jump_to:end",
            MiddlewareControl::UpdateState(_) => "update_state",
            MiddlewareControl::StopWithFinal(_) => "stop_with_final",
            MiddlewareControl::Interrupt { .. } => "interrupt",
        }
    }

    /// Precedence rank for resolving competing control requests within one
    /// turn. Higher wins. [`Interrupt`](Self::Interrupt) outranks
    /// [`StopWithFinal`](Self::StopWithFinal) because pausing to preserve state
    /// for a later resume is stronger than terminating with a final answer, so
    /// a pause request is never silently downgraded to a stop.
    /// [`Continue`](Self::Continue) is the lowest rank: it carries no
    /// instruction and is never itself installed as a pending request (see
    /// [`RunContext::request_control`]). [`UpdateState`](Self::UpdateState)
    /// and [`JumpTo`](Self::JumpTo) sit below the two run-ending outcomes so a
    /// state patch or a soft reroute never displaces a stop or an interrupt
    /// that a later hook in the same phase also requested.
    pub fn precedence(&self) -> u8 {
        match self {
            MiddlewareControl::Continue => 0,
            MiddlewareControl::UpdateState(_) => 1,
            MiddlewareControl::JumpTo(_) => 2,
            MiddlewareControl::StopWithFinal(_) => 3,
            MiddlewareControl::Interrupt { .. } => 4,
        }
    }
}

/// Live, in-process handle threaded through every step of a harness run.
///
/// A `RunContext` bundles the declarative [`RunConfig`] with the runtime
/// dependencies a run needs:
/// - `stores`: the [`StoreRegistry`] for long-term persistence.
/// - `events`: the [`EventSink`] for observability fan-out.
/// - `limits`: the [`LimitTracker`] enforcing the run's caps.
///
/// The generic `Ctx` parameter carries arbitrary user data (dependencies,
/// shared services, accumulated state). It defaults to `()` for runs that need
/// no extra data.
///
/// Unlike [`RunConfig`], `RunContext` is **not** serializable: it owns live
/// counters, listener lists, and user handles.
pub struct RunContext<Ctx = ()> {
    /// Process-unique identity of *this context instance*, minted on
    /// construction. Unlike [`RunConfig::run_id`] — a caller-supplied label two
    /// concurrent runs may well share — it distinguishes concurrent runs, so
    /// shared per-run bookkeeping (a middleware's in-flight reservation, say)
    /// can be keyed on it. Read it with [`RunContext::instance_id`].
    pub(crate) instance_id: u64,
    /// Live marker used by middleware to prune state after a cancelled run
    /// drops its context without reaching a terminal lifecycle hook.
    pub(crate) lifecycle: std::sync::Arc<()>,
    /// The declarative configuration this context was built from.
    pub config: RunConfig,
    /// Arbitrary user-supplied run data.
    pub data: Ctx,
    /// Number of frozen leading System messages supplied by a durable
    /// session. `None` lets a standalone harness infer its initial prefix.
    /// A child context starts at `None` for its own conversation.
    pub frozen_system_prefix_len: Option<usize>,
    /// Capability profile of the model the next call is expected to reach.
    ///
    /// The agent loop previews model resolution and sets this just before
    /// `before_model` middleware runs, so middleware can shape what it adds
    /// for the target (for example
    /// [`push_ephemeral_instruction`](crate::middleware::push_ephemeral_instruction)
    /// avoids new system messages on a model that hoists them). `None` when no
    /// model has been previewed — outside the agent loop, or when only a host
    /// routing decision (made after `before_model`) can name the model.
    pub model_profile: Option<tinyinference_llm::model::ModelProfile>,
    /// Registry of named long-term stores.
    pub stores: StoreRegistry,
    /// Optional hierarchical long-term store handed to every tool this run
    /// invokes as
    /// [`ToolExecutionContext::store`][crate::tool::ToolExecutionContext::store]
    /// (B1). Distinct from [`Self::stores`], the flat named registry: this
    /// is the one [`NamespacedStore`][crate::store::namespaced::NamespacedStore]
    /// a tool may read and write directly — memories, scratch state, a
    /// per-user cache — without the harness minting a name for it. `None`
    /// means tools get no store. Attach one with
    /// [`RunContext::with_namespaced_store`]; shared with child contexts
    /// exactly like `stores`.
    pub namespaced_store: Option<std::sync::Arc<dyn crate::store::namespaced::NamespacedStore>>,
    /// Optional type-erased, read-only view of the application state handed
    /// to every tool this run invokes, recovered by
    /// [`ToolExecutionContext::state`][crate::tool::ToolExecutionContext::state]
    /// (B1). Erased because `RunContext` is not generic over `State` and the
    /// agent loop only ever holds a borrowed `&State` it cannot lend to a
    /// concurrent tool future; the host attaches an owned `Arc<S>` snapshot
    /// with [`RunContext::with_state_view`] instead. Shared with child
    /// contexts, which run against the same application state.
    pub state_view: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// Event fan-out bus for observability.
    pub events: EventSink,
    /// Live limit tracker derived from `config`.
    pub limits: LimitTracker,
    /// Optional steering channel an orchestrator (parent agent, human UI,
    /// graph supervisor, or test) uses to guide this run at safe checkpoints.
    ///
    /// `None` means the run accepts no steering. Attach one with
    /// [`RunContext::with_steering`]; the agent loop drains it before each
    /// model call via
    /// [`crate::steering::apply_pending_steering`].
    pub steering: Option<SteeringHandle>,
    /// Optional multi-lane message queue the agent loop drains at its turn
    /// boundaries (A4): `Steer` after each tool batch and at a natural
    /// finish, `Followup` at a natural finish only, `Collect` once at run
    /// end onto [`crate::middleware::AgentRun::collected`]. `None` means the
    /// loop consumes no queued messages. Attach one with
    /// [`RunContext::with_run_queue`]; never inherited by a child context,
    /// because a queue has no per-run addressing and a child draining its
    /// parent's queue would steal the parent's messages.
    pub run_queue: Option<crate::run_queue::RunQueueHandle>,
    /// Cooperative cancellation token for this run.
    ///
    /// Defaults to a fresh, never-cancelled [`CancellationToken`], so a run is
    /// only cancellable if a caller installs a shared token via
    /// [`RunContext::with_cancellation`]. The agent loop polls
    /// [`CancellationToken::is_cancelled`] at the same safe checkpoints used for
    /// steering — before each model call and before each tool call — and the
    /// streaming pipeline races [`CancellationToken::cancelled`] against the
    /// provider stream. On observing cancellation the run ends with
    /// [`crate::error::TinyAgentsError::Cancelled`].
    ///
    /// Child contexts ([`RunContext::child`]) receive a
    /// [`CancellationToken::child_token`]: cancelling the parent cancels every
    /// descendant, but cancelling one child does not affect the parent or its
    /// siblings.
    pub cancellation: CancellationToken,
    /// A one-shot control request a middleware or step can set to steer the
    /// loop (stop with a final response, or interrupt). Drained by the agent
    /// loop at its safe checkpoints via [`RunContext::take_control`].
    pub control: std::sync::Arc<std::sync::Mutex<Option<MiddlewareControl>>>,
    /// Set by a middleware that just noted a repeat on a tool result (an
    /// identical call re-issued, an identical reply) and read once by the
    /// agent loop before its next model call. The loop's reasoning fallback
    /// treats it as the signal that a model running without reasoning is
    /// looping, and hands reasoning back for the next call. See
    /// [`RunContext::note_repeat`].
    pub repeat_noted: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Queued [`StateUpdate`]s a middleware or tool requested via
    /// [`MiddlewareControl::UpdateState`], drained by a host through
    /// [`RunContext::take_state_updates`]. See that method's docs for why the
    /// loop cannot apply these itself.
    pub(crate) state_updates: std::sync::Arc<std::sync::Mutex<Vec<StateUpdate>>>,
    /// Queued raw JSON state updates a tool requested via
    /// [`tinytools::ToolControl::state_update`]. See
    /// [`RunContext::push_tool_state_update`].
    pub(crate) tool_state_updates: std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    /// One entry per tool call finished in the current batch, in call order:
    /// `Some(output)` when the call asked the loop to end the run
    /// ([`tinytools::ToolControl::terminate`] / `return_direct`), `None`
    /// otherwise. The batch driver settles it once every call has answered —
    /// the run ends only if **every** call voted `Some`.
    pub(crate) terminate_votes: Vec<Option<String>>,
    /// Positions (within the batch about to run) of the tool calls a length
    /// stop may have cut off; admission answers each with a synthetic error
    /// instead of running it (see `RunPolicy::reject_truncated_tool_calls`).
    /// Positional, not by provider id, so duplicate or empty ids fail closed.
    /// Set per turn and cleared when the batch ends.
    pub(crate) truncated_call_positions: std::collections::HashSet<usize>,
    /// The subset of `truncated_call_positions` whose tool was also cut off
    /// on the previous truncated turn: answered with the stronger "stop
    /// sending calls this large" corrective. Same lifetime as the set above.
    pub(crate) truncated_repeat_positions: std::collections::HashSet<usize>,
    /// How many calls the current batch has admitted so far; the position
    /// `truncated_call_positions` is matched against. Reset per batch.
    pub(crate) batch_admissions: usize,
    /// An optional host-supplied workspace/sandbox descriptor threaded into every
    /// [`ToolExecutionContext`][crate::tool::ToolExecutionContext] this
    /// run creates, so tools discover their allowed root from context rather
    /// than an application global. A host can populate it directly with
    /// [`RunContext::with_workspace`] or from around-agent middleware; `None`
    /// means no workspace policy is in effect.
    pub workspace: Option<tinytools::WorkspaceDescriptor>,
    /// Whether the middleware stack already fanned `on_error` out to every
    /// middleware for the error currently unwinding this run. The stack sets it
    /// when a lifecycle hook fails (it dispatches `on_error` itself before
    /// propagating), and the agent-loop driver reads it so the same failure is
    /// not delivered to every middleware a second time.
    pub(crate) on_error_dispatched: bool,
    /// Whether this run is being driven through the streaming loop path
    /// (`ChatModel::stream`), set by the agent-loop driver. Threaded into each
    /// [`ToolExecutionContext`][crate::tool::ToolExecutionContext] so a
    /// sub-agent tool can run its child in the matching mode — the child's
    /// model/reasoning deltas then propagate to the parent's stream via the
    /// shared [`EventSink`]. A non-streaming parent leaves this `false`, so its
    /// event stream is unchanged.
    pub streaming: bool,
    /// Host definition id currently driving this context, propagated into a
    /// child so recursive delegation can be authorized by the host registry.
    pub(crate) host_agent_id: Option<String>,
    /// Type-erased, runtime-owned host authority inherited by children.  This
    /// is deliberately not serializable or public: it keeps a hosted parent
    /// from accidentally delegating through a child's unrelated (or absent)
    /// capability bundle.
    ///
    /// Erased through [`crate::runtime::ErasedHostAuthority`] rather than
    /// `dyn Any`: the generic explicit-model loop must stay callable with a
    /// borrowed (non-`'static`) `State`/`Ctx`, and `Any::downcast_ref`
    /// requires `'static` at the *read* site, which such a caller can never
    /// prove. The custom trait instead exposes a type-name check that needs
    /// no `'static` bound on either side; see
    /// [`crate::runtime::host_invocation_binding`] for how the read side
    /// uses it to fail closed on a mismatch.
    pub(crate) host_authority: Option<std::sync::Arc<dyn crate::runtime::ErasedHostAuthority>>,
    /// Runtime-owned terminal lifecycle callback, consumed exactly once by the
    /// agent-loop guard even when the driving future is cancelled or dropped.
    pub(crate) terminal_observer: Option<TerminalObserver>,
    /// Lifecycle cursor over the working transcript: which messages were
    /// announced and which turn is open (see `agent_loop::lifecycle`).
    pub(crate) turns: crate::agent_loop::TurnTracker,
    /// Set by a no-progress / repeat guard that paused the run, holding its
    /// root-cause summary, so the loop can report `TerminalReason::Halted`.
    pub(crate) halted_by_guard: Option<String>,
    /// The kind of the most recent `LimitReached` event this run emitted.
    pub(crate) last_limit: std::sync::Mutex<Option<crate::events::LimitKind>>,
    /// The [`CallId`] the agent loop minted for the model call currently in
    /// flight through the model-wrap middleware onion, mirroring
    /// [`crate::events::HarnessRunStatus::active_model_call`].
    ///
    /// Set by the loop immediately before invoking
    /// [`crate::middleware::MiddlewareStack::run_wrapped_model`] and cleared
    /// right after, so a `ModelMiddleware` such as
    /// [`crate::middleware::library::RetryMiddleware`] can correlate its own
    /// `RetryScheduled` events with the same call id the loop uses, instead of
    /// deriving an uncorrelated one from `ctx.run_id()` alone (see I-7).
    /// `None` outside that window, and always `None` for a caller that never
    /// goes through the agent loop.
    pub active_model_call: Option<CallId>,
    pub(crate) provider_started: bool,
    /// Whether the *current* model call reached the provider; reset when a
    /// call begins (see `begin_model_call`).
    pub(crate) call_provider_started: bool,
    /// Set when a model call returned an error, so the terminal classifier
    /// still knows the failure surfaced inside the provider call after
    /// `active_model_call` was cleared.
    pub(crate) model_call_failed: bool,
    /// Whether the model call most recently dispatched by the agent loop used
    /// the streaming path. Set by the loop's innermost model call before the
    /// wrap middleware sees the result, so a middleware can tell that
    /// discarding a response would also discard output a consumer already
    /// received as deltas.
    pub call_streamed: bool,
    /// Identifies the current shape of the prompt prefix: `0` until a
    /// middleware that rewrites it (a compaction, a truncation) calls
    /// [`RunContext::mark_prompt_prefix_changed`], then a process-unique
    /// value. Provider cache accounting keys on it so the uncached tokens of a
    /// deliberately rewritten prefix are not reported as a miss.
    pub(crate) prefix_epoch: u64,
    /// Usage of responses a wrap middleware paid for and then discarded (a
    /// retry after an overflow reported by a successful response). The agent
    /// loop folds it into the run's totals and the host budget at its next
    /// accounting point; see [`RunContext::record_discarded_usage`].
    pub(crate) discarded_usage: Vec<tinyinference_llm::usage::Usage>,
    /// Resolutions for the deferred tool calls left pending on the transcript
    /// this run is resuming (A2). Taken by the agent loop before its first
    /// model call and applied to the unanswered tool calls on the last
    /// assistant row; see
    /// [`crate::runtime::AgentHarness::resume_deferred`]. Never inherited by
    /// a child context.
    pub(crate) deferred_results: Option<crate::tool::DeferredToolResults>,
    /// Tool-call ids a human approved on resume (A2). Admission skips the
    /// deferral checks for these, and a `before_tool` hook's
    /// `ApprovalRequired` is ignored for them, so an approved call cannot be
    /// deferred a second time by the same gate. Read with
    /// [`RunContext::is_call_approved`].
    pub(crate) approved_calls: std::collections::HashSet<String>,
    /// Host-only metadata a `before_tool` hook asked to stamp on the result
    /// that answers a call it refused, keyed by call id. See
    /// [`RunContext::set_refusal_metadata`].
    pub(crate) refusal_metadata: std::collections::HashMap<String, serde_json::Value>,
    /// Monotonic, per-context (not process-global) counter handed out by
    /// [`RunContext::next_child_ordinal`], used to derive deterministic child
    /// run ids (e.g. [`crate::subagent::SubAgent`]'s `{name}-d{depth}-{parent
    /// run id}-{ordinal}`) instead of a process-global sequence (M-2). Starts
    /// at `0` for every freshly constructed context — including a child
    /// context, which gets its own fresh counter rather than inheriting the
    /// parent's — so two processes that call the same parent context's child
    /// spawner in the same order derive identical ordinals, and therefore
    /// identical child run ids.
    pub(crate) child_ordinal: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Durable tool-effect ledger for this run, when a host wants crash-safe
    /// bookkeeping of tool-call side effects (B5). `None` (the default) means
    /// no ledger writes happen and [`tinytools::ToolPolicy`]'s
    /// `runtime.replay` declaration has nothing to guard resume against — the
    /// agent loop behaves exactly as it did before this existed. Attach one
    /// with [`RunContext::with_tool_effect_ledger`]; a child context inherits
    /// its parent's ledger, matching how `stores`/`events` propagate.
    pub tool_effect_ledger: Option<std::sync::Arc<dyn crate::tool::ToolEffectLedger>>,
    /// How the agent loop reacts when a `started` write to
    /// [`Self::tool_effect_ledger`] itself fails, before the tool call it was
    /// about to journal has executed. See
    /// [`crate::tool::LedgerFailure`] for the two modes; defaults to
    /// [`crate::tool::LedgerFailure::Abort`].
    pub tool_effect_ledger_failure: crate::tool::LedgerFailure,
    /// Run-wide gate for nested tool calls across every concurrent parent:
    /// concurrency-safe calls hold it shared, calls to tools (or wrap
    /// middleware) that are not concurrency-safe hold it exclusively. A call already running under the gate does not retake it.
    pub(crate) nested_serial: std::sync::Arc<tokio::sync::RwLock<()>>,
    /// Durable sink for [`crate::summarization::CompactionRecord`]s this run
    /// produces, when a host wants every compaction persisted somewhere
    /// durable rather than only kept in
    /// [`crate::middleware::ContextCompressionMiddleware::records`]'s
    /// in-process buffer.
    ///
    /// `None` (the default) means compaction runs exactly as it did before
    /// this existed — no persistence side effect. Attach one with
    /// [`RunContext::with_compaction_sink`]; a child context inherits its
    /// parent's sink, matching how `stores`/`events`/`tool_effect_ledger`
    /// propagate. `tinyagents-harness` cannot depend on
    /// `tinyagents-session` (the dependency runs the other way), so this is
    /// a trait object rather than a concrete `Arc<EntryTree>` — see
    /// [`crate::summarization::CompactionSink`]'s docs.
    pub compaction_sink: Option<std::sync::Arc<dyn crate::summarization::CompactionSink>>,
}
