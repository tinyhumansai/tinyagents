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

use futures::{Stream, StreamExt};

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
    pub(crate) binding: HostInvocationBinding<State, Ctx>,
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
    inner: Option<Pin<Box<dyn Stream<Item = AgentStreamItem> + Send + 'a>>>,
    // Kept after `inner` so Rust drops the borrowed stream before the overlay
    // that owns its harness. See `extend_overlay_stream_lifetime`.
    _runtime: Option<std::sync::Arc<InvocationRuntime<State, Ctx>>>,
    cancellation: crate::CancellationToken,
    terminal_observer: std::sync::Arc<std::sync::Mutex<Option<crate::context::TerminalObserver>>>,
    terminal_observed: bool,
    marker: std::marker::PhantomData<(&'a State, Ctx)>,
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync> Stream for AgentStream<'_, State, Ctx> {
    type Item = AgentStreamItem;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `inner` is pinned independently by `Box`; this projection never moves
        // the boxed stream or any other field of `AgentStream`.
        let stream = unsafe { self.get_unchecked_mut() };
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
                AgentRun::new(),
                false,
                Some("hosted stream cancelled before execution began".to_string()),
            );
        }
    }
}

struct PreparedAgentTurn<State: Send + Sync, Ctx: Send + Sync> {
    binding: HostInvocationBinding<State, Ctx>,
    thread_id: ThreadId,
    run_id: crate::ids::RunId,
    input_text: String,
    messages: Vec<tinyinference_llm::message::Message>,
}

#[derive(Clone)]
pub(crate) struct ProgressSender {
    tx: tokio::sync::mpsc::Sender<ProgressEvent>,
    nonterminal_slots: std::sync::Arc<tokio::sync::Semaphore>,
}

impl ProgressSender {
    fn send_nonterminal(&self, event: ProgressEvent) {
        let Ok(permit) = self.nonterminal_slots.clone().try_acquire_owned() else {
            return;
        };
        if self.tx.try_send(event).is_ok() {
            permit.forget();
        }
    }

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
    ) -> Result<AgentRun>
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
    ) -> Result<AgentRun>
    where
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
        let mut prepared = runner
            .prepare_agent_turn_bounded(host, request, &context)
            .await?;
        prepared.binding.runtime = runtime.clone();
        let agent_id = prepared.binding.agent_id.clone();
        context.host_agent_id = Some(agent_id.clone());
        context.host_authority = Some(std::sync::Arc::new(HostInvocationAuthority {
            binding: prepared.binding.clone(),
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

        let outcome = runner
            .invoke_in_context_collecting_partial(state, context, prepared.messages.clone())
            .await;
        match outcome.error {
            None => Ok(outcome.run),
            Some(TinyAgentsError::Cancelled) => Err(TinyAgentsError::Cancelled),
            Some(TinyAgentsError::Timeout(message)) => Err(TinyAgentsError::Timeout(message)),
            Some(_) => Err(TinyAgentsError::Model(
                "hosted agent invocation failed".to_string(),
            )),
        }
    }

    /// Collects a hosted turn through the streaming driver while preserving the
    /// parent's exact capability bundle. Recursive streaming delegation uses
    /// this rather than the unary entry point so model deltas and delta
    /// middleware remain part of the shared parent event stream.
    pub(crate) async fn invoke_agent_streaming_with_capabilities(
        &self,
        invocation: AgentInvocation<State, Ctx>,
        state: &State,
    ) -> Result<AgentRun>
    where
        Ctx: 'static,
        State: 'static,
    {
        let stream = self
            .invoke_agent_stream_with_capabilities(invocation, state)
            .await?;
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            match item {
                AgentStreamItem::Completed(run) => return Ok(*run),
                AgentStreamItem::Failed { .. } => {
                    return Err(TinyAgentsError::Model(
                        "hosted agent invocation failed".to_string(),
                    ));
                }
                AgentStreamItem::Event(_) => {}
            }
        }
        Err(TinyAgentsError::Model(
            "hosted stream ended without a terminal result".to_string(),
        ))
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
        let mut prepared = runner
            .prepare_agent_turn_bounded(host, request, &context)
            .await?;
        prepared.binding.runtime = runtime.clone();
        let agent_id = prepared.binding.agent_id.clone();
        context.host_agent_id = Some(agent_id.clone());
        context.host_authority = Some(std::sync::Arc::new(HostInvocationAuthority {
            binding: prepared.binding.clone(),
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
        let stream = runner
            .invoke_stream_in_context(state, context, prepared.messages.clone())
            .map(|item| item);
        Ok(AgentStream {
            // `runtime` is retained by this stream and is declared after
            // `inner`, so it outlives the stream's borrow of its harness. The
            // explicit helper records that otherwise non-obvious lifetime
            // relationship at the one boundary where the owned hosted
            // invocation meets the borrowed stream API.
            inner: Some(unsafe { extend_overlay_stream_lifetime(Box::pin(stream)) }),
            _runtime: runtime,
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
    async fn prepare_agent_turn_bounded(
        &self,
        host: std::sync::Arc<crate::host::HostCapabilities<State>>,
        request: AgentTurnRequest,
        context: &RunContext<Ctx>,
    ) -> Result<PreparedAgentTurn<State, Ctx>> {
        let cancellation = context.cancellation.clone();
        let preparation = self.prepare_agent_turn(host, request, context);
        let outcome = match self.host_io_budget(context) {
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
        };
        outcome.map_err(sanitize_hosted_preparation_error)
    }

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
                let _ = error;
                tinyagents_tracing::warn!(agent_id = %request.agent_id, "[host] definition lookup failed");
                TinyAgentsError::Validation("agent definition lookup failed".to_string())
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
        Ok(PreparedAgentTurn {
            binding: HostInvocationBinding {
                host: host.clone(),
                agent_id: request.agent_id.clone(),
                model_pin: definition.model,
                role: definition.role,
                allowed_tools: definition.tools.into_iter().collect(),
                progress: progress.clone(),
                runtime: None,
            },
            thread_id,
            run_id: context.run_id().clone(),
            input_text,
            messages,
        })
    }

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
                        tinyagents_tracing::warn!("[host] agent run failed");
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

/// Extends a stream borrow from an invocation overlay to the caller's stream
/// lifetime.
///
/// Safety is established by `AgentStream`: it stores the exact `Arc` that owns
/// the overlay alongside `inner`, and field order drops `inner` first. The
/// other borrowed inputs (`&self` and `&State`) already have the public `'a`
/// lifetime. No reference can escape the private `AgentStream` wrapper.
unsafe fn extend_overlay_stream_lifetime<'a>(
    stream: Pin<Box<dyn Stream<Item = AgentStreamItem> + Send + '_>>,
) -> Pin<Box<dyn Stream<Item = AgentStreamItem> + Send + 'a>> {
    // SAFETY: documented above; the owning Arc is retained by AgentStream.
    unsafe { std::mem::transmute(stream) }
}

/// Returns this live context's host authorization, if it is a hosted run.
///
/// The binding is carried by the non-serializable context rather than the
/// reusable harness, so concurrent roots have no shared mutable authority.
pub(crate) fn host_invocation_binding<State: Send + Sync, Ctx: Send + Sync>(
    context: &RunContext<Ctx>,
) -> Result<Option<HostInvocationBinding<State, Ctx>>> {
    let Some(authority) = context.host_authority.as_ref() else {
        return Ok(None);
    };
    // `host_authority` is crate-private and is installed only by the hosted
    // entry points, which require `State: 'static` and store exactly
    // `HostInvocationAuthority<State>`. Explicit-model entry points never
    // install it, so they return at the `None` branch without requiring
    // `State: 'static` or consulting `Any` at all. Keeping this cast at the
    // private hosted-context boundary restores borrowed-state support to the
    // generic loop without creating a harness registry or any cross-invocation
    // authority channel.
    //
    // SAFETY: no public API can construct or mutate `host_authority`; its only
    // assignment is the hosted `AgentInvocation` path in this module.
    // `RunContext::child` clones that same `Arc` only for recursive calls with
    // the same `State`. Thus a present authority always points at the concrete
    // type requested here for the active harness invocation.
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
    let Some(progress) = binding.progress else {
        return;
    };
    progress.send_nonterminal(event);
}

fn sanitize_hosted_preparation_error(error: TinyAgentsError) -> TinyAgentsError {
    match error {
        TinyAgentsError::Cancelled | TinyAgentsError::Timeout(_) => error,
        _ => TinyAgentsError::Model("hosted agent invocation failed".to_string()),
    }
}

fn spawn_host_finalizer<State: Send + Sync + 'static, Ctx: Send + Sync + 'static>(
    prepared: PreparedAgentTurn<State, Ctx>,
    run: AgentRun,
    succeeded: bool,
    error: Option<String>,
) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(async move { finish_host_turn(prepared, run, succeeded, error).await });
    } else {
        tinyagents_tracing::warn!(
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

async fn finish_host_turn<State: Send + Sync, Ctx: Send + Sync + 'static>(
    prepared: PreparedAgentTurn<State, Ctx>,
    run: AgentRun,
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
    let output = run.text().unwrap_or_default();
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
            tinyagents_tracing::warn!(%error, "[host] memory sink failed after terminal turn");
        }
    }
    if let Some(learning) = &prepared.binding.host.learning
        && let Err(error) = learning.on_turn_complete(&summary).await
    {
        tinyagents_tracing::warn!(%error, "[host] learning sink failed after terminal turn");
    }
    if let Some(store) = &prepared.binding.host.experience {
        let mut experience =
            Experience::new(&prepared.binding.agent_id, &prepared.input_text, &output);
        if succeeded {
            experience = experience.succeeded();
        }
        if let Err(error) = store.record(&experience).await {
            tinyagents_tracing::warn!(%error, "[host] experience store failed after terminal turn");
        }
    }
}

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
