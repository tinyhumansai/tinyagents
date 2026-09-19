//! Host-driven agent invocation.
//!
//! The lower-level `invoke*` APIs intentionally accept an already-selected
//! model. This module is the separate product-host boundary: it resolves an
//! agent definition and model, composes/screen contexts, and installs a
//! per-`RunContext` binding that the normal loop consumes. Keeping the two
//! entry points separate prevents an embedding SDK call from accidentally
//! acquiring product policy merely because a harness was also configured for a
//! hosted turn.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;

use crate::agent_loop::AgentStreamItem;
use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::events::{AgentEvent, EventRecord};
use crate::host::{
    ContentOrigin, Experience, ProgressEvent, RecallRequest, ScreenOutcome, TurnContextRequest,
    TurnSummary,
};
use crate::ids::ThreadId;
use crate::middleware::AgentRun;

use super::{AgentHarness, HostInvocationBinding, InvocationRuntime};

/// The exact host bundle that authorized the parent invocation.
///
/// A child context carries this as type-erased runtime state because
/// [`RunContext`] is intentionally independent of the application's `State`.
/// `SubAgent` downcasts it at the recursive boundary and therefore cannot
/// substitute an unhosted or differently-hosted child harness for the
/// parent's policy.
pub(crate) struct HostInvocationAuthority<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) binding: std::sync::Arc<HostInvocationBinding<State, Ctx>>,
}

/// Type-erasure boundary for [`RunContext::host_authority`][crate::context::RunContext].
///
/// This is a hand-written alternative to `dyn Any`. `Any::downcast_ref`
/// requires the caller's own generic parameters to be provably `'static`,
/// which the generic agent loop cannot promise: it deliberately keeps
/// working with a borrowed `State`/`Ctx` on the explicit-model path (see
/// `explicit_model_paths_accept_borrowed_state`). [`type_name`][Self::type_name]
/// is callable with no `'static` bound at all (`std::any::type_name` never
/// requires one), so [`host_invocation_binding`] can use it as a fail-closed
/// guard in front of the unavoidable unsafe cast, without forcing `'static`
/// onto the whole generic loop.
///
/// `type_name` is documented as not a guaranteed-unique identifier, so this
/// is a defensive, best-effort check rather than the same soundness
/// guarantee `TypeId` gives genuinely `'static` types. It still closes the
/// realistic C-1 repro (a hosted context read by a *different* harness):
/// distinct concrete `HostInvocationAuthority<State, Ctx>` monomorphizations
/// in this crate reliably produce distinct strings.
pub(crate) trait ErasedHostAuthority: Send + Sync {
    fn type_name(&self) -> &'static str;
}

impl<State: Send + Sync, Ctx: Send + Sync> ErasedHostAuthority
    for HostInvocationAuthority<State, Ctx>
{
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Closed, non-leaking classification of a hosted invocation failure.
///
/// A host reading [`HostedError::kind`] can distinguish "the caller cancelled
/// this" from "a configured limit was exhausted" from "the provider failed"
/// without inspecting [`HostedError::message`] (which stays a fixed,
/// sanitized string per kind — see that field's doc) or attaching a private
/// event listener to reconstruct the same information from the event stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HostedErrorKind {
    /// The run was cancelled before completion.
    Cancelled,
    /// The run exceeded a wall-clock deadline (the run's own, or a per-call
    /// ceiling — see [`TinyAgentsError::Timeout`] and
    /// [`TinyAgentsError::CallTimeout`]).
    Timeout,
    /// A configured run limit (model calls, tool calls, recursion depth, a
    /// host budget) was exhausted.
    LimitExceeded,
    /// The host's own policy rejected the invocation (an unresolvable
    /// definition, a failed security screen, an unauthorized delegate).
    Policy,
    /// The model provider failed the call.
    Provider,
    /// Any other internal failure not covered by a more specific kind.
    Internal,
}

/// The typed failure returned by the hosted entry points
/// ([`AgentHarness::invoke_agent`] and its streaming counterpart) in place of
/// a generic `TinyAgentsError::Model("hosted agent invocation failed")`.
///
/// This intentionally does not implement `TinyAgentsError`'s "one error type"
/// convention: it is the harness's product-host boundary type, not another
/// case folded into the crate-wide error, and it is deliberately smaller —
/// `message` is a fixed, sanitized string selected by `kind` (never the
/// underlying provider/middleware/budget error text; that stays available to
/// the host only through its own capability bundle's own logging and through
/// the internal (non-hosted) event stream if it chose to attach a listener).
#[derive(Debug)]
pub struct HostedError {
    /// Closed classification of the failure. See [`HostedErrorKind`].
    pub kind: HostedErrorKind,
    /// Fixed, sanitized message selected by `kind` — never raw provider,
    /// middleware, or budget error text.
    pub message: String,
    /// The accumulated transcript, usage, and executed-tool summary as far as
    /// the run got before failing, when available.
    pub run: Option<Box<AgentRun>>,
}

impl std::fmt::Display for HostedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for HostedError {}

/// Classifies a raw loop error into the closed [`HostedErrorKind`] vocabulary
/// a host is allowed to see.
fn classify_hosted_error(error: &TinyAgentsError) -> HostedErrorKind {
    match error {
        TinyAgentsError::Cancelled => HostedErrorKind::Cancelled,
        TinyAgentsError::Timeout(_) | TinyAgentsError::CallTimeout(_) => HostedErrorKind::Timeout,
        TinyAgentsError::LimitExceeded(_) | TinyAgentsError::SubAgentDepth(_) => {
            HostedErrorKind::LimitExceeded
        }
        TinyAgentsError::Validation(_) | TinyAgentsError::Steering(_) => HostedErrorKind::Policy,
        TinyAgentsError::Provider(_) | TinyAgentsError::Model(_) => HostedErrorKind::Provider,
        _ => HostedErrorKind::Internal,
    }
}

/// The fixed, sanitized message for each [`HostedErrorKind`]. Never derived
/// from the underlying error's own text.
fn hosted_error_message(kind: HostedErrorKind) -> &'static str {
    match kind {
        HostedErrorKind::Cancelled => "hosted agent invocation was cancelled",
        HostedErrorKind::Timeout => "hosted agent invocation timed out",
        HostedErrorKind::LimitExceeded => "hosted agent invocation exceeded a configured limit",
        HostedErrorKind::Policy => "hosted agent invocation was rejected by policy",
        HostedErrorKind::Provider => "hosted agent invocation failed at the model provider",
        HostedErrorKind::Internal => "hosted agent invocation failed",
    }
}

/// Builds a [`HostedError`] from the raw loop error and whatever partial
/// [`AgentRun`] the loop accumulated before failing.
fn hosted_error(error: &TinyAgentsError, run: AgentRun) -> HostedError {
    let kind = classify_hosted_error(error);
    HostedError {
        kind,
        message: hosted_error_message(kind).to_string(),
        run: Some(Box::new(run)),
    }
}

/// Reconstructs a crate-wide [`TinyAgentsError`] from a [`HostedError`] for
/// internal callers (recursive hosted delegation) that must keep propagating
/// through the ordinary `Result<T>` = `Result<T, TinyAgentsError>` surface.
/// This is a lossless-enough round trip for control flow: `Cancelled` and
/// `Timeout` map back to their own variants (so cancellation/deadline
/// semantics upstream keep working, e.g. the fallback gate in
/// `invoke_model_resolving`), and the rest become typed but message-generic
/// variants — never worse than what this boundary already returned before
/// `HostedError` existed.
impl From<HostedError> for TinyAgentsError {
    fn from(error: HostedError) -> Self {
        match error.kind {
            HostedErrorKind::Cancelled => TinyAgentsError::Cancelled,
            HostedErrorKind::Timeout => TinyAgentsError::Timeout(error.message),
            HostedErrorKind::LimitExceeded => TinyAgentsError::LimitExceeded(error.message),
            HostedErrorKind::Policy => TinyAgentsError::Validation(error.message),
            HostedErrorKind::Provider | HostedErrorKind::Internal => {
                TinyAgentsError::Model(error.message)
            }
        }
    }
}

/// A host-owned turn request.
///
/// `agent_id` is opaque to the harness. It is resolved only through the host
/// definition registry and is threaded back to every host capability as an
/// attribution value.
#[derive(Clone, Debug)]
pub struct AgentTurnRequest {
    /// Host definition to invoke.
    pub agent_id: String,
    /// Initial transcript supplied by the host.
    pub messages: Vec<tinyinference_llm::message::Message>,
}

impl AgentTurnRequest {
    /// Creates a request for `agent_id` with the supplied conversation input.
    pub fn new(
        agent_id: impl Into<String>,
        messages: Vec<tinyinference_llm::message::Message>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            messages,
        }
    }
}

/// One host-authorized execution of an agent.
///
/// `AgentHarness` is intentionally reusable: it retains durable, process-wide
/// dependencies such as model/tool registries, middleware, policy, and caches.
/// A host bundle is not one of those dependencies. Progress observers,
/// approval decisions, request security and other host capabilities can differ
/// for two roots running at the same time, so they belong to this invocation
/// object rather than the harness.
///
/// The contained [`RunContext`] is live-only and this type has no serialization
/// implementation. Consequently neither the host bundle nor its authority can
/// enter graph/checkpoint state. Recursive children receive a clone of this
/// invocation's bundle through the parent context, never through a child
/// harness configuration.
pub struct AgentInvocation<State: Send + Sync, Ctx: Send + Sync = ()> {
    /// The host capability bundle authorizing this one recursive run tree.
    pub host: std::sync::Arc<crate::host::HostCapabilities<State>>,
    /// Definition identifier and input messages for the root turn.
    pub request: AgentTurnRequest,
    /// Live execution context for this root run.
    pub context: RunContext<Ctx>,
    /// Invocation-owned registries and middleware. Absence selects the durable
    /// harness that receives `invoke_agent`.
    runtime: Option<std::sync::Arc<InvocationRuntime<State, Ctx>>>,
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentInvocation<State, Ctx> {
    /// Creates a host-authorized invocation for `request` in `context`.
    pub fn new(
        host: crate::host::HostCapabilities<State>,
        request: AgentTurnRequest,
        context: RunContext<Ctx>,
    ) -> Self {
        Self {
            host: std::sync::Arc::new(host),
            request,
            context,
            runtime: None,
        }
    }

    /// Attaches isolated model, tool, and middleware mechanics to this one
    /// invocation tree without mutating the durable harness.
    pub fn with_runtime(mut self, runtime: InvocationRuntime<State, Ctx>) -> Self {
        self.runtime = Some(std::sync::Arc::new(runtime));
        self
    }

    /// Reuses the parent's exact live host bundle for a recursive child.
    ///
    /// This is crate-private because only the runtime can prove that a child
    /// is authorized to inherit the parent invocation's authority.
    pub(crate) fn from_shared_host(
        host: std::sync::Arc<crate::host::HostCapabilities<State>>,
        request: AgentTurnRequest,
        context: RunContext<Ctx>,
        runtime: Option<std::sync::Arc<InvocationRuntime<State, Ctx>>>,
    ) -> Self {
        Self {
            host,
            request,
            context,
            runtime,
        }
    }
}

/// A caller-consumable hosted stream.
///
/// Dropping it cancels the live invocation context and drives its terminal
/// observer even when a caller stops listening before a terminal item. The
/// invocation's host authority is owned by that context, never by the harness.
pub struct AgentStream<'a, State: Send + Sync + 'static, Ctx: Send + Sync> {
    // Owns its inputs (the harness borrow or the invocation-local runtime is
    // moved into the driving future itself, inside `invoke_stream_with_runner`)
    // so nothing outside this field needs to outlive it and no lifetime
    // extension is required to store it here.
    inner: Option<Pin<Box<dyn Stream<Item = AgentStreamItem> + Send + 'a>>>,
    cancellation: crate::CancellationToken,
    terminal_observer: std::sync::Arc<std::sync::Mutex<Option<crate::context::TerminalObserver>>>,
    terminal_observed: bool,
    // `fn() -> Ctx` (rather than bare `Ctx`) keeps this marker `Unpin`
    // regardless of `Ctx`, which is what lets `poll_next` use the safe
    // `Pin::get_mut` below instead of `get_unchecked_mut`.
    #[allow(clippy::type_complexity)]
    marker: std::marker::PhantomData<(&'a State, fn() -> Ctx)>,
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync> Stream for AgentStream<'_, State, Ctx> {
    type Item = AgentStreamItem;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Every field is `Unpin` (`Option<Pin<Box<..>>>`, `CancellationToken`,
        // an `Arc<Mutex<..>>`, `bool`, and a `fn()`-based `PhantomData`), so
        // `AgentStream` itself is `Unpin` and this projection is safe.
        let stream = self.get_mut();
        match stream.inner.as_mut() {
            Some(inner) => match inner.as_mut().poll_next(context) {
                Poll::Ready(Some(item)) => {
                    // The ordinary SDK stream intentionally carries rich
                    // provider diagnostics.  A hosted stream crosses into a
                    // product boundary, however, so it must not expose raw
                    // provider, middleware, or budget error text to its
                    // caller.  This projects the item after the internal
                    // event sink has recorded the typed diagnostic; host
                    // observability therefore remains useful without making
                    // the public stream a disclosure channel.
                    let item = sanitize_hosted_stream_item(item);
                    stream.terminal_observed = matches!(
                        item,
                        AgentStreamItem::Completed(_) | AgentStreamItem::Failed { .. }
                    );
                    Poll::Ready(Some(item))
                }
                Poll::Ready(None) => {
                    stream.terminal_observed = true;
                    Poll::Ready(None)
                }
                Poll::Pending => Poll::Pending,
            },
            None => Poll::Ready(None),
        }
    }
}

/// Removes internal failure details from the caller-consumable hosted stream.
///
/// This deliberately transforms only the clone forwarded by `AgentStream`.
/// The run loop's event sink, status, and error values retain their precise
/// typed details for host-owned diagnostics and policy decisions.
fn sanitize_hosted_stream_item(mut item: AgentStreamItem) -> AgentStreamItem {
    match &mut item {
        AgentStreamItem::Failed { error, .. } => {
            *error = "hosted agent invocation failed".to_string();
        }
        AgentStreamItem::Event(record) => sanitize_hosted_event(record),
        AgentStreamItem::Completed(_) => {}
    }
    item
}

/// Replaces every error message on an [`AgentEvent`] with a fixed, family-
/// specific string before it reaches a hosted caller, mirroring
/// [`sanitize_hosted_stream_item`] for the individual event variants that
/// carry raw provider/tool/middleware diagnostics.
fn sanitize_hosted_event(record: &mut EventRecord) {
    match &mut record.event {
        AgentEvent::ToolCompleted {
            error: Some(error), ..
        }
        | AgentEvent::ToolFailed { error, .. } => {
            *error = "hosted tool invocation failed".to_string();
        }
        AgentEvent::ModelFailed { error, .. } => {
            *error = "hosted model invocation failed".to_string();
        }
        AgentEvent::SubAgentFailed { error, .. } => {
            *error = "hosted sub-agent invocation failed".to_string();
        }
        AgentEvent::InvalidToolArgs { error, .. } => {
            *error = "hosted tool arguments are invalid".to_string();
        }
        AgentEvent::WorkspaceCleanup {
            error: Some(error), ..
        } => {
            *error = "hosted workspace cleanup failed".to_string();
        }
        AgentEvent::MiddlewareFailed { error, .. } => {
            *error = "hosted middleware failed".to_string();
        }
        AgentEvent::RunFailed { error, .. } => {
            *error = "hosted agent invocation failed".to_string();
        }
        _ => {}
    }
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync> Drop for AgentStream<'_, State, Ctx> {
    fn drop(&mut self) {
        // Dropping the driving stream drops the loop's `TerminalRunGuard`,
        // which forwards its actual partial run to the installed observer.
        if !self.terminal_observed {
            self.cancellation.cancel();
        }
        self.inner.take();
        if let Ok(mut observer) = self.terminal_observer.lock()
            && let Some(observer) = observer.take()
        {
            observer(
                crate::context::TerminalRunSummary::default(),
                false,
                Some("hosted stream cancelled before execution began".to_string()),
            );
        }
    }
}

/// Everything a hosted turn needs to run, assembled by
/// [`AgentHarness::prepare_agent_turn`] before the loop starts.
///
/// Bundles the per-invocation host binding with the composed transcript so the
/// unary and streaming entry points can share one preparation path and one
/// terminal-observer closure.
struct PreparedAgentTurn<State: Send + Sync, Ctx: Send + Sync> {
    /// Host capability bundle and definition-derived routing facts for this run.
    binding: HostInvocationBinding<State, Ctx>,
    /// Thread the prepared turn belongs to (derived from the run id when the
    /// caller supplied none).
    thread_id: ThreadId,
    /// Run id of the invocation this turn was prepared for.
    run_id: crate::ids::RunId,
    /// Screened user-visible text, used for memory recall and later stamped
    /// onto the turn summary and experience record.
    input_text: String,
    /// The fully composed transcript (system prompt, preamble, user messages)
    /// handed to the agent loop.
    messages: Vec<tinyinference_llm::message::Message>,
}

/// Bounded, best-effort fan-out from the agent loop to a host's
/// [`crate::host::ProgressSink`].
///
/// Backed by a bounded channel plus a semaphore so a slow or absent consumer
/// cannot retain unbounded memory for a long-running turn; see
/// [`start_progress_dispatcher`] for the slot accounting this coordinates with.
#[derive(Clone)]
pub(crate) struct ProgressSender {
    tx: tokio::sync::mpsc::Sender<ProgressEvent>,
    nonterminal_slots: std::sync::Arc<tokio::sync::Semaphore>,
}

impl ProgressSender {
    /// Sends a non-terminal progress event, dropping it if no nonterminal slot
    /// is free.
    ///
    /// Never blocks: `try_acquire_owned` and `try_send` both fail fast rather
    /// than stalling the turn on a slow sink. The permit is intentionally
    /// forgotten on success — it is returned later when the dispatcher observes
    /// the event was consumed, not when this call returns.
    fn send_nonterminal(&self, event: ProgressEvent) {
        let Ok(permit) = self.nonterminal_slots.clone().try_acquire_owned() else {
            return;
        };
        if self.tx.try_send(event).is_ok() {
            permit.forget();
        }
    }

    /// Sends the one terminal event for this turn (`Finished` or `Error`).
    ///
    /// Bypasses the nonterminal-slot semaphore: the channel reserves a slot
    /// specifically for this so a saturated stream of progress updates can
    /// never suppress the outcome. Still best-effort — a full channel drops it
    /// rather than blocking the finalizer.
    fn send_terminal(&self, event: ProgressEvent) {
        let _ = self.tx.try_send(event);
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> Clone for PreparedAgentTurn<State, Ctx> {
    fn clone(&self) -> Self {
        Self {
            binding: self.binding.clone(),
            thread_id: self.thread_id.clone(),
            run_id: self.run_id.clone(),
            input_text: self.input_text.clone(),
            messages: self.messages.clone(),
        }
    }
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> AgentHarness<State, Ctx> {
    /// Runs an agent through this invocation's host-capability bundle.
    ///
    /// The invocation type requires the four mandatory capabilities at
    /// construction; a caller cannot accidentally use this hosted path without
    /// supplying them. Optional capabilities remain genuinely optional: when
    /// absent they are neither constructed nor called.
    pub async fn invoke_agent(
        &self,
        invocation: AgentInvocation<State, Ctx>,
        state: &State,
    ) -> std::result::Result<AgentRun, HostedError>
    where
        State: 'static,
    {
        self.invoke_agent_with_capabilities(invocation, state).await
    }

    /// Re-enters the canonical hosted entry point with the parent's exact
    /// capabilities. Used only by recursive delegation after the parent
    /// authority has authorized the child.
    pub(crate) async fn invoke_agent_with_capabilities(
        &self,
        invocation: AgentInvocation<State, Ctx>,
        state: &State,
    ) -> std::result::Result<AgentRun, HostedError>
    where
        State: 'static,
    {
        let (runtime, context, prepared) = self
            .prepare_hosted_turn(invocation)
            .await
            .map_err(|error| hosted_error(&error, AgentRun::new()))?;
        let runner = runtime
            .as_deref()
            .map(InvocationRuntime::harness)
            .unwrap_or(self);

        let outcome = runner
            .invoke_in_context_collecting_partial(state, context, prepared.messages.clone())
            .await;
        match outcome.error {
            None => Ok(outcome.run),
            Some(error) => Err(hosted_error(&error, outcome.run)),
        }
    }

    /// Collects a hosted turn through the streaming driver while preserving the
    /// parent's exact capability bundle. Recursive streaming delegation uses
    /// this rather than the unary entry point so model deltas and delta
    /// middleware remain part of the shared parent event stream.
    ///
    /// Drives [`AgentHarness::invoke_streaming_in_context_collecting_partial`]
    /// directly rather than going through [`AgentHarness::invoke_agent_stream`]:
    /// the public stream sanitizes every item (see
    /// [`sanitize_hosted_stream_item`]), which would throw away the real
    /// [`TinyAgentsError`] this method needs to classify into a
    /// [`HostedErrorKind`] before its own, separate sanitization.
    pub(crate) async fn invoke_agent_streaming_with_capabilities(
        &self,
        invocation: AgentInvocation<State, Ctx>,
        state: &State,
    ) -> std::result::Result<AgentRun, HostedError>
    where
        Ctx: 'static,
        State: 'static,
    {
        let (runtime, context, prepared) = self
            .prepare_hosted_turn(invocation)
            .await
            .map_err(|error| hosted_error(&error, AgentRun::new()))?;
        let runner = runtime
            .as_deref()
            .map(InvocationRuntime::harness)
            .unwrap_or(self);

        let outcome = runner
            .invoke_streaming_in_context_collecting_partial(
                state,
                context,
                prepared.messages.clone(),
            )
            .await;
        match outcome.error {
            None => Ok(outcome.run),
            Some(error) => Err(hosted_error(&error, outcome.run)),
        }
    }

    /// Shared setup for both hosted drivers: resolves and authorizes the
    /// turn, installs the host authority and terminal observer on `context`,
    /// and emits [`ProgressEvent::Started`]. Returns the invocation-local
    /// runtime overlay (if any — the caller derives the harness that should
    /// actually run the turn from it, since a reference borrowed from it here
    /// cannot outlive this function), the prepared `context`, and the
    /// prepared turn.
    async fn prepare_hosted_turn(
        &self,
        invocation: AgentInvocation<State, Ctx>,
    ) -> Result<(
        Option<std::sync::Arc<InvocationRuntime<State, Ctx>>>,
        RunContext<Ctx>,
        PreparedAgentTurn<State, Ctx>,
    )> {
        let AgentInvocation {
            host,
            request,
            mut context,
            runtime,
        } = invocation;
        let runner = runtime
            .as_deref()
            .map(InvocationRuntime::harness)
            .unwrap_or(self);
        let mut prepared = runner
            .prepare_agent_turn_bounded(host, request, &context)
            .await?;
        prepared.binding.runtime = runtime.clone();
        let agent_id = prepared.binding.agent_id.clone();
        context.host_agent_id = Some(agent_id.clone());
        context.host_authority = Some(std::sync::Arc::new(HostInvocationAuthority {
            binding: std::sync::Arc::new(prepared.binding.clone()),
        }));
        runner.install_host_terminal_observer(&mut context, prepared.clone());
        emit_host_progress::<State, Ctx>(
            &context,
            ProgressEvent::Started {
                run: context.run_id().clone(),
                thread: context.thread_id().cloned(),
                agent: agent_id,
            },
        );
        Ok((runtime, context, prepared))
    }

    /// Starts a hosted streaming turn.
    ///
    /// The returned stream is the existing event projection, so host-driven
    /// and explicit-model streams expose the same canonical event vocabulary.
    /// The run-scoped model binding is removed when its terminal item is
    /// observed; callers must drain (or drop) the stream to end the turn.
    pub async fn invoke_agent_stream<'a>(
        &'a self,
        invocation: AgentInvocation<State, Ctx>,
        state: &'a State,
    ) -> Result<AgentStream<'a, State, Ctx>>
    where
        Ctx: 'static,
        State: 'static,
    {
        self.invoke_agent_stream_with_capabilities(invocation, state)
            .await
    }

    async fn invoke_agent_stream_with_capabilities<'a>(
        &'a self,
        invocation: AgentInvocation<State, Ctx>,
        state: &'a State,
    ) -> Result<AgentStream<'a, State, Ctx>>
    where
        Ctx: 'static,
        State: 'static,
    {
        let AgentInvocation {
            host,
            request,
            mut context,
            runtime,
        } = invocation;
        let runner = runtime
            .as_deref()
            .map(InvocationRuntime::harness)
            .unwrap_or(self);
        // Unlike `prepare_hosted_turn` (used by the unary/streaming-collecting
        // drivers, which classify the raw error into a `HostedError` and do
        // their own, kind-scoped sanitization), this public-stream setup
        // returns a plain `TinyAgentsError` directly to the caller — so it
        // still needs `sanitize_hosted_preparation_error` applied here.
        let mut prepared = runner
            .prepare_agent_turn_bounded(host, request, &context)
            .await
            .map_err(sanitize_hosted_preparation_error)?;
        prepared.binding.runtime = runtime.clone();
        let agent_id = prepared.binding.agent_id.clone();
        context.host_agent_id = Some(agent_id.clone());
        context.host_authority = Some(std::sync::Arc::new(HostInvocationAuthority {
            binding: std::sync::Arc::new(prepared.binding.clone()),
        }));
        let cancellation = context.cancellation.clone();
        let terminal_observer =
            runner.install_host_terminal_observer(&mut context, prepared.clone());
        emit_host_progress::<State, Ctx>(
            &context,
            ProgressEvent::Started {
                run: context.run_id().clone(),
                thread: context.thread_id().cloned(),
                agent: agent_id,
            },
        );
        // Build the stream from a `StreamRunner` that either borrows `self`
        // (unhosted-runtime case) or owns `runtime` outright (hosted-runtime
        // case). `invoke_stream_with_runner` moves whichever one it gets into
        // the driving future's own state, so the returned stream needs no
        // extra field or lifetime extension to keep an owned runtime alive:
        // the future already owns it for exactly as long as it is needed.
        let stream_runner = match runtime {
            Some(runtime) => crate::agent_loop::StreamRunner::Owned(runtime),
            None => crate::agent_loop::StreamRunner::Borrowed(self),
        };
        let stream = crate::agent_loop::invoke_stream_with_runner(
            stream_runner,
            state,
            context,
            prepared.messages.clone(),
        );
        Ok(AgentStream {
            inner: Some(Box::pin(stream)),
            cancellation,
            terminal_observer,
            terminal_observed: false,
            marker: std::marker::PhantomData,
        })
    }

    /// Runs host-owned preparation under the same cancellation and wall-clock
    /// controls as model resolution.  Definitions, security screening, and
    /// context composition are all host I/O boundaries, not setup work that
    /// may outlive a cancelled turn.
    ///
    /// Returns the raw, unsanitized [`TinyAgentsError`] — callers decide how
    /// to sanitize: `prepare_hosted_turn` classifies it into a
    /// [`HostedError`] (which does its own kind-scoped sanitization), while
    /// `invoke_agent_stream_with_capabilities` (the one caller that still
    /// returns a plain `TinyAgentsError` to the public API) applies
    /// [`sanitize_hosted_preparation_error`] itself.
    async fn prepare_agent_turn_bounded(
        &self,
        host: std::sync::Arc<crate::host::HostCapabilities<State>>,
        request: AgentTurnRequest,
        context: &RunContext<Ctx>,
    ) -> Result<PreparedAgentTurn<State, Ctx>> {
        let cancellation = context.cancellation.clone();
        let preparation = self.prepare_agent_turn(host, request, context);
        match self.host_io_budget(context) {
            Some(remaining) => tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(TinyAgentsError::Cancelled),
                result = tokio::time::timeout(remaining, preparation) => result.map_err(|_| TinyAgentsError::Timeout(format!(
                    "host turn preparation for run `{}` exceeded its remaining wall-clock budget",
                    context.run_id()
                )))?,
            },
            None => tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(TinyAgentsError::Cancelled),
                result = preparation => result,
            },
        }
    }

    /// The wall-clock time left for host I/O (definition lookup, security
    /// screening, context composition) before this turn must give up.
    ///
    /// Takes the smaller of the context's own remaining budget and what
    /// `RunPolicy::limits.max_wall_clock_ms` still allows, so host preparation
    /// never outlives either the caller's deadline or the harness's own cap.
    /// `None` means neither source imposes a limit.
    fn host_io_budget(&self, context: &RunContext<Ctx>) -> Option<Duration> {
        let config = context.remaining_wall_clock();
        let policy = self.policy.limits.max_wall_clock_ms.map(|milliseconds| {
            Duration::from_millis(milliseconds)
                .checked_sub(context.limits.elapsed())
                .unwrap_or(Duration::ZERO)
        });
        match (config, policy) {
            (Some(config), Some(policy)) => Some(config.min(policy)),
            (Some(config), None) => Some(config),
            (None, Some(policy)) => Some(policy),
            (None, None) => None,
        }
    }

    /// Resolves the host agent definition, screens the caller's messages,
    /// composes the system prompt/preamble, and folds in memory and experience
    /// recall to produce a [`PreparedAgentTurn`].
    ///
    /// Order matters: the definition must resolve and validate before any host
    /// I/O runs, user messages are screened before their text is used as a
    /// memory/experience recall query, and recalled/stored text is screened a
    /// second time (as [`ContentOrigin::Stored`]) before it is appended to the
    /// transcript — a host's own stored content is not exempt from screening.
    async fn prepare_agent_turn(
        &self,
        host: std::sync::Arc<crate::host::HostCapabilities<State>>,
        mut request: AgentTurnRequest,
        context: &RunContext<Ctx>,
    ) -> Result<PreparedAgentTurn<State, Ctx>> {
        if request.agent_id.trim().is_empty() {
            return Err(TinyAgentsError::Validation(
                "host-driven invocation requires a non-empty agent id".into(),
            ));
        }
        let definition = host
            .definitions
            .resolve(&request.agent_id)
            .await
            .map_err(|error| {
                tracing::warn!(
                    agent_id = %request.agent_id,
                    error = %error,
                    "[host] definition lookup failed"
                );
                TinyAgentsError::Validation(format!(
                    "agent definition lookup failed for `{}`: {error}",
                    request.agent_id
                ))
            })?
            .ok_or_else(|| {
                TinyAgentsError::Validation(format!(
                    "agent definition `{}` was not found",
                    request.agent_id
                ))
            })?;
        if !definition.is_valid() {
            return Err(TinyAgentsError::Validation(format!(
                "agent definition `{}` is invalid: {}",
                request.agent_id,
                definition
                    .diagnostics()
                    .into_iter()
                    .map(|diagnostic| diagnostic.message)
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }

        let thread_id = context
            .thread_id()
            .cloned()
            .unwrap_or_else(|| ThreadId::from(context.run_id().as_str()));
        let input_text = screen_user_messages(&host, &mut request.messages).await?;
        let context_request =
            TurnContextRequest::new(&request.agent_id, thread_id.clone(), &input_text);
        let system = host.context.compose_system_prompt(&context_request).await?;
        let mut preamble = host.context.preamble(&context_request).await?;

        if let Some(memory) = &host.memory {
            if let Some(summary) = memory.thread_summary(&thread_id).await? {
                preamble.push(tinyinference_llm::message::Message::system(
                    screen_stored(&host, &summary).await?,
                ));
            }
            for memory in memory
                .recall(
                    RecallRequest::new(&input_text)
                        .with_agent(&request.agent_id)
                        .with_thread(thread_id.clone()),
                )
                .await?
            {
                preamble.push(tinyinference_llm::message::Message::system(
                    screen_stored(&host, &memory.text).await?,
                ));
            }
        }
        if let Some(experience) = &host.experience {
            for experience in experience
                .recall_for(&request.agent_id, &input_text)
                .await?
            {
                preamble.push(tinyinference_llm::message::Message::system(
                    screen_stored(&host, &experience.outcome).await?,
                ));
            }
        }
        let mut messages = Vec::with_capacity(request.messages.len() + preamble.len() + 1);
        if !system.is_empty() {
            messages.push(tinyinference_llm::message::Message::system(system));
        }
        messages.append(&mut preamble);
        messages.append(&mut request.messages);
        let progress = start_progress_dispatcher(host.progress.clone());
        // An empty declared list and "declared nothing" are indistinguishable
        // on `AgentDefinition::tools` (`Vec<String>`), so both collapse to
        // `None` here — `resolve_tool_allowlist` in `agent_loop::tools`
        // treats that as fail-closed by default (I-9), not as "unrestricted".
        let declared_tools: std::collections::HashSet<String> =
            definition.tools.into_iter().collect();
        let allowed_tools = (!declared_tools.is_empty()).then_some(declared_tools);
        Ok(PreparedAgentTurn {
            binding: HostInvocationBinding {
                host: host.clone(),
                agent_id: request.agent_id.clone(),
                model_pin: definition.model,
                role: definition.role,
                allowed_tools,
                progress: progress.clone(),
                runtime: None,
            },
            thread_id,
            run_id: context.run_id().clone(),
            input_text,
            messages,
        })
    }

    /// Wires a terminal observer that hands the finished (or cancelled)
    /// `AgentRun` off to [`spawn_host_finalizer`] exactly once.
    ///
    /// Returns the shared `Option` slot so a caller (`AgentStream::drop`) can
    /// also fire the observer if the stream is abandoned before a terminal
    /// item is produced — the `Option::take` inside both closures is what
    /// keeps "run finished" and "stream dropped early" from double-firing.
    fn install_host_terminal_observer(
        &self,
        context: &mut RunContext<Ctx>,
        prepared: PreparedAgentTurn<State, Ctx>,
    ) -> std::sync::Arc<std::sync::Mutex<Option<crate::context::TerminalObserver>>>
    where
        State: 'static,
    {
        let observer = std::sync::Arc::new(std::sync::Mutex::new(Some(Box::new(
            move |run, succeeded, error: Option<String>| {
                spawn_host_finalizer(
                    prepared,
                    run,
                    succeeded,
                    error.map(|error| {
                        let _ = error;
                        tracing::warn!("[host] agent run failed");
                        "agent run failed".to_string()
                    }),
                );
            },
        )
            as crate::context::TerminalObserver)));
        let for_context = std::sync::Arc::clone(&observer);
        context.set_terminal_observer(Box::new(move |run, succeeded, error| {
            if let Ok(mut observer) = for_context.lock()
                && let Some(observer) = observer.take()
            {
                observer(run, succeeded, error);
            }
        }));
        observer
    }
}

/// Returns this live context's host authorization, if it is a hosted run.
///
/// The binding is carried by the non-serializable context rather than the
/// reusable harness, so concurrent roots have no shared mutable authority.
///
/// `host_authority` is `Option<Arc<dyn Any + Send + Sync>>` and is installed
/// only by the hosted entry points in this module, which require
/// `State: 'static, Ctx: 'static` and store exactly
/// `HostInvocationAuthority<State, Ctx>`. Nothing about [`RunContext`]
/// prevents a caller from handing a hosted context to a *different* harness
/// (a different `State`, or — via [`RunContext::child_with_data`] changing
/// `Ctx`), so the erased type is checked with [`Any::downcast_ref`] rather
/// than assumed. A mismatch fails closed with
/// [`TinyAgentsError::Validation`] instead of reinterpreting memory through
/// the wrong type. Absence of any authority is the ordinary, cheap case (an
/// explicit-model run, or the generic loop when no hosted invocation
/// installed one) and returns `Ok(None)` without touching `Any` at all, so
/// this function itself still only needs `State: 'static, Ctx: 'static` on
/// the (rare) hosted path — its callers already carry that bound.
pub(crate) fn host_invocation_binding<State: Send + Sync, Ctx: Send + Sync>(
    context: &RunContext<Ctx>,
) -> Result<Option<std::sync::Arc<HostInvocationBinding<State, Ctx>>>> {
    let Some(authority) = context.host_authority.as_ref() else {
        return Ok(None);
    };
    // `RunContext::child` (the only authority-propagating path) requires the
    // same `Ctx` as its parent, and `RunContext::child_with_data` (the only
    // path that changes `Ctx`) always clears `host_authority` first — so a
    // present authority's `Ctx` already matches this call's `Ctx` by
    // construction. `State` has no such structural guarantee (nothing
    // prevents handing a hosted context to a *different* harness), so it is
    // checked here at read time via `ErasedHostAuthority::type_name` (see
    // that trait's doc comment for why this, and not `Any`, is used).
    let expected = std::any::type_name::<HostInvocationAuthority<State, Ctx>>();
    if authority.type_name() != expected {
        return Err(TinyAgentsError::Validation(
            "host authority type mismatch: this run context was hosted by a different \
             State/Ctx harness than the one reading it"
                .to_string(),
        ));
    }
    #[allow(unsafe_code)]
    // SAFETY: `context.host_authority` is crate-private and is installed
    // only by the hosted entry points in this module, which always store
    // exactly `HostInvocationAuthority<State, Ctx>` for the harness they are
    // called on. The `type_name` check above additionally rejects any value
    // whose concrete type does not match this call's own `State`/`Ctx`
    // before this cast runs, so a mismatched authority never reaches it.
    let authority = unsafe {
        &*(std::sync::Arc::as_ptr(authority) as *const HostInvocationAuthority<State, Ctx>)
    };
    Ok(Some(authority.binding.clone()))
}

/// Best-effort progress projection. A host UI must never make the turn wait or
/// fail, so delivery is detached and dropped when no Tokio runtime is available.
pub(crate) fn emit_host_progress<State: Send + Sync, Ctx: Send + Sync>(
    context: &RunContext<Ctx>,
    event: ProgressEvent,
) {
    let Ok(Some(binding)) = host_invocation_binding::<State, Ctx>(context) else {
        return;
    };
    let Some(progress) = binding.progress.as_ref() else {
        return;
    };
    progress.send_nonterminal(event);
}

/// Collapses any preparation failure into a caller-safe message, preserving
/// only [`TinyAgentsError::Cancelled`] and [`TinyAgentsError::Timeout`] — the
/// two variants a caller needs to distinguish to react correctly (do not
/// retry vs. may retry with a longer budget). Everything else (a definition
/// lookup failure, a security block, a malformed definition) collapses to the
/// same generic message so internal detail never reaches a hosted caller.
fn sanitize_hosted_preparation_error(error: TinyAgentsError) -> TinyAgentsError {
    match error {
        TinyAgentsError::Cancelled
        | TinyAgentsError::Timeout(_)
        | TinyAgentsError::CallTimeout(_) => error,
        _ => TinyAgentsError::Model("hosted agent invocation failed".to_string()),
    }
}

/// Runs [`finish_host_turn`] on a Tokio task, tolerating the case where the
/// terminal observer fires from a context with no ambient runtime (for
/// example, a `Drop` impl running during unwind).
///
/// Falls back to a dedicated current-thread runtime rather than dropping the
/// finalization work, since memory/learning/experience recording must still
/// happen even when the turn ends off the normal async call path.
fn spawn_host_finalizer<State: Send + Sync + 'static, Ctx: Send + Sync + 'static>(
    prepared: PreparedAgentTurn<State, Ctx>,
    run: crate::context::TerminalRunSummary,
    succeeded: bool,
    error: Option<String>,
) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(async move { finish_host_turn(prepared, run, succeeded, error).await });
    } else {
        tracing::warn!(
            run_id = %prepared.run_id,
            "[host] no Tokio runtime during terminal cleanup; starting fallback finalizer"
        );
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("fallback host finalizer runtime must initialize");
            runtime.block_on(finish_host_turn(prepared, run, succeeded, error));
        });
    }
}

/// The hosted turn's epilogue: emits the terminal progress event, then feeds
/// the finished run into whichever optional host capabilities are configured
/// (memory, learning, experience). Runs once per turn, after the caller-facing
/// result has already been produced.
///
/// Each optional sink's failure is logged and does not affect the others or
/// propagate anywhere — by the time this runs the turn is already over, so
/// there is nothing left to fail.
async fn finish_host_turn<State: Send + Sync, Ctx: Send + Sync + 'static>(
    prepared: PreparedAgentTurn<State, Ctx>,
    run: crate::context::TerminalRunSummary,
    succeeded: bool,
    error: Option<String>,
) {
    // The per-turn queue preserves event order while keeping a slow progress
    // consumer completely outside the agent's critical path.
    if let Some(progress) = &prepared.binding.progress {
        if let Some(message) = error {
            progress.send_terminal(ProgressEvent::Error {
                run: prepared.run_id.clone(),
                message,
            });
        } else {
            progress.send_terminal(ProgressEvent::Finished {
                run: prepared.run_id.clone(),
                usage: Some(run.usage.usage),
            });
        }
    }
    let output = run.text.clone().unwrap_or_default();
    let mut summary = TurnSummary::new(prepared.thread_id.clone(), &prepared.binding.agent_id)
        .with_text(&prepared.input_text, &output)
        .with_usage(run.usage.usage);
    for tool in &run.executed_tools {
        summary.record_tool(tool);
    }
    if let Some(memory) = &prepared.binding.host.memory {
        let item = crate::host::NewMemory::new(&output)
            .with_thread(prepared.thread_id.clone())
            .with_agent(&prepared.binding.agent_id)
            .with_tag(if succeeded {
                "turn_success"
            } else {
                "turn_failure"
            });
        if let Err(error) = memory.remember(item).await {
            tracing::warn!(%error, "[host] memory sink failed after terminal turn");
        }
    }
    if let Some(learning) = &prepared.binding.host.learning
        && let Err(error) = learning.on_turn_complete(&summary).await
    {
        tracing::warn!(%error, "[host] learning sink failed after terminal turn");
    }
    if let Some(store) = &prepared.binding.host.experience {
        let mut experience =
            Experience::new(&prepared.binding.agent_id, &prepared.input_text, &output);
        if succeeded {
            experience = experience.succeeded();
        }
        if let Err(error) = store.record(&experience).await {
            tracing::warn!(%error, "[host] experience store failed after terminal turn");
        }
    }
}

/// Spawns the background task that drains a [`ProgressSender`]'s channel into
/// `sink`, and returns the sender half — or `None` when there is no sink or no
/// ambient Tokio runtime to spawn onto.
///
/// See the field comment below for the 128 nonterminal + 1 terminal slot
/// accounting this pairs with in [`ProgressSender::send_nonterminal`] /
/// [`ProgressSender::send_terminal`].
fn start_progress_dispatcher(
    sink: Option<std::sync::Arc<dyn crate::host::ProgressSink>>,
) -> Option<ProgressSender> {
    let sink = sink?;
    let handle = tokio::runtime::Handle::try_current().ok()?;
    // Progress is observational. Bound it so a slow sink cannot retain every
    // streamed token; producers use `try_send` and drop overflowed updates.
    // One slot is reserved for the single terminal outcome. Producers use
    // `try_send` for ordinary progress, so at most 128 nonterminal events can
    // fill before finalization claims the remaining slot.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(129);
    let nonterminal_slots = std::sync::Arc::new(tokio::sync::Semaphore::new(128));
    let released_slots = nonterminal_slots.clone();
    handle.spawn(async move {
        while let Some(event) = rx.recv().await {
            if !event.is_terminal() {
                released_slots.add_permits(1);
            }
            sink.emit(event).await;
        }
    });
    Some(ProgressSender {
        tx,
        nonterminal_slots,
    })
}

/// Screens every text/JSON block of every user message through the host's
/// [`crate::host::SecurityGate`], rewriting redacted blocks in place, and
/// returns the visible text joined by newlines for use as the turn's
/// memory/experience recall query.
///
/// A [`ScreenOutcome::Block`] on any block aborts the whole turn — there is no
/// partial admission of a user message once one piece of it is deemed unsafe.
async fn screen_user_messages<State: Send + Sync>(
    host: &crate::host::HostCapabilities<State>,
    messages: &mut [tinyinference_llm::message::Message],
) -> Result<String> {
    let mut visible = Vec::new();
    for message in messages {
        if let tinyinference_llm::message::Message::User(user) = message {
            for block in &mut user.content {
                if let tinyinference_llm::message::ContentBlock::Text(text)
                | tinyinference_llm::message::ContentBlock::Thinking { text, .. } = block
                {
                    let screened = host
                        .security
                        .screen_input(text, ContentOrigin::User)
                        .await?;
                    match screened {
                        ScreenOutcome::Pass => visible.push(text.clone()),
                        ScreenOutcome::Redacted(redacted) => {
                            *text = redacted;
                            visible.push(text.clone());
                        }
                        ScreenOutcome::Block { reason } => {
                            return Err(TinyAgentsError::Validation(reason));
                        }
                    }
                } else if let tinyinference_llm::message::ContentBlock::Json(value)
                | tinyinference_llm::message::ContentBlock::ProviderExtension(value) =
                    block
                {
                    let rendered = value.to_string();
                    match host
                        .security
                        .screen_input(&rendered, ContentOrigin::User)
                        .await?
                    {
                        ScreenOutcome::Pass => visible.push(rendered),
                        ScreenOutcome::Redacted(redacted) => {
                            *value = serde_json::from_str(&redacted)
                                .unwrap_or(serde_json::Value::String(redacted));
                            visible.push(value.to_string());
                        }
                        ScreenOutcome::Block { reason } => {
                            return Err(TinyAgentsError::Validation(reason));
                        }
                    }
                }
            }
        }
    }
    Ok(visible.join("\n"))
}

/// Screens one piece of host-stored text (a thread summary, a recalled
/// memory, a recalled experience) as [`ContentOrigin::Stored`] before it is
/// injected into the transcript.
///
/// Applies to content the host itself produced or persisted earlier — it is
/// screened anyway because a memory or experience record can still carry text
/// that originated from an untrusted source further upstream.
async fn screen_stored<State: Send + Sync>(
    host: &crate::host::HostCapabilities<State>,
    text: &str,
) -> Result<String> {
    match host
        .security
        .screen_input(text, ContentOrigin::Stored)
        .await?
    {
        ScreenOutcome::Pass => Ok(text.to_string()),
        ScreenOutcome::Redacted(text) => Ok(text),
        ScreenOutcome::Block { reason } => Err(TinyAgentsError::Validation(reason)),
    }
}

/// Covers the [`ProgressSender`] / [`start_progress_dispatcher`] backpressure
/// contract: a saturated stream of nonterminal progress events must never
/// crowd out the one reserved terminal slot.
#[cfg(test)]
mod progress_dispatcher_tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use super::start_progress_dispatcher;
    use crate::host::{ProgressEvent, ProgressSink};
    use crate::ids::RunId;

    struct BlockingProgressSink {
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Semaphore>,
        events: Mutex<Vec<ProgressEvent>>,
    }

    #[async_trait]
    impl ProgressSink for BlockingProgressSink {
        async fn emit(&self, event: ProgressEvent) {
            self.entered.notify_one();
            self.release
                .acquire()
                .await
                .expect("test progress sink remains open")
                .forget();
            self.events.lock().expect("progress lock").push(event);
        }
    }

    #[tokio::test]
    async fn terminal_progress_is_delivered_after_nonterminal_slots_are_saturated() {
        let sink = Arc::new(BlockingProgressSink {
            entered: Arc::new(tokio::sync::Notify::new()),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
            events: Mutex::new(Vec::new()),
        });
        let sender = start_progress_dispatcher(Some(sink.clone()))
            .expect("Tokio test runtime provides a dispatcher");
        let run = RunId::new("saturated-progress");
        sender.send_nonterminal(ProgressEvent::Token {
            run: run.clone(),
            text: "first".to_string(),
        });
        sink.entered.notified().await;

        for slot in 0..128 {
            sender.send_nonterminal(ProgressEvent::Token {
                run: run.clone(),
                text: format!("queued-{slot}"),
            });
        }
        sender.send_terminal(ProgressEvent::Finished { run, usage: None });

        // The sink holds the receiver on the first item, so all 128 ordinary
        // slots are occupied. Finished therefore proves the reserved terminal
        // slot was still available after nonterminal backpressure saturated.
        sink.release.add_permits(130);
        for _ in 0..256 {
            if sink
                .events
                .lock()
                .expect("progress lock")
                .iter()
                .any(ProgressEvent::is_terminal)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            sink.events
                .lock()
                .expect("progress lock")
                .iter()
                .filter(|event| event.is_terminal())
                .count(),
            1,
            "the terminal event cannot be dropped behind 128 progress updates"
        );
    }
}
