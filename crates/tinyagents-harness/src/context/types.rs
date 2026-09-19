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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MiddlewareControl {
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
            MiddlewareControl::StopWithFinal(_) => "stop_with_final",
            MiddlewareControl::Interrupt { .. } => "interrupt",
        }
    }

    /// Precedence rank for resolving competing control requests within one
    /// turn. Higher wins. [`Interrupt`](Self::Interrupt) outranks
    /// [`StopWithFinal`](Self::StopWithFinal) because pausing to preserve state
    /// for a later resume is stronger than terminating with a final answer, so
    /// a pause request is never silently downgraded to a stop.
    pub fn precedence(&self) -> u8 {
        match self {
            MiddlewareControl::StopWithFinal(_) => 1,
            MiddlewareControl::Interrupt { .. } => 2,
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
    /// The declarative configuration this context was built from.
    pub config: RunConfig,
    /// Arbitrary user-supplied run data.
    pub data: Ctx,
    /// Registry of named long-term stores.
    pub stores: StoreRegistry,
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
    pub cancellation: CancellationToken,
    /// A one-shot control request a middleware or step can set to steer the
    /// loop (stop with a final response, or interrupt). Drained by the agent
    /// loop at its safe checkpoints via [`RunContext::take_control`].
    pub control: std::sync::Arc<std::sync::Mutex<Option<MiddlewareControl>>>,
    /// The isolated workspace/sandbox descriptor threaded into every
    /// [`ToolExecutionContext`][crate::tool::ToolExecutionContext] this
    /// run creates, so tools discover their allowed root from context rather
    /// than an application global. Populated by
    /// [`RunContext::with_workspace`] or by preparing a
    /// [`WorkspaceIsolation`][crate::workspace::WorkspaceIsolation]
    /// provider; `None` means no workspace policy is in effect.
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
}
