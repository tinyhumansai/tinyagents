//! Harness-owned execution context for canonical `tinytools::Tool` calls.
//!
//! Tool vocabulary belongs to `tinytools`. This module deliberately contains
//! only the narrow bridge from a live harness run to `ToolRunContext`; it does
//! not redeclare a tool, result, policy, timeout, or call type.

use std::any::Any;
use std::sync::Arc;

use crate::cancel::CancellationToken;
use crate::context::RunContext;
use crate::events::{AgentEvent, EventSink};
use crate::ids::{CallId, RunId, ThreadId};
use crate::store::namespaced::NamespacedStore;

/// The run facts a canonical tool may inspect through
/// [`tinytools::ToolRunContext`].
///
/// Cancellation, event emission, and run identity deliberately remain owned by
/// the harness. Recursive tools that need an actual child `RunContext` use the
/// explicit dispatch seam at registration rather than widening TinyTools'
/// portable context trait.
///
/// # Reaching this from a `tinytools::Tool`
///
/// A canonical tool receives `Option<&dyn ToolRunContext>`, whose typed
/// methods cover only the portable facts (workspace, thread id, output cap).
/// The rest of this struct — the call id, the store, the typed state view,
/// [`Self::custom`] — is reachable by downcasting the erased host extension
/// (B1):
///
/// ```ignore
/// let harness = context
///     .and_then(tinytools::ToolRunContext::host_extension)
///     .and_then(|any| any.downcast_ref::<ToolExecutionContext>());
/// ```
///
/// [`Self::of`] performs exactly that downcast, and [`Self::call_id_of`]
/// reads the call id through it.
#[derive(Clone)]
pub struct ToolExecutionContext {
    /// Run that invoked the tool.
    pub run_id: RunId,
    /// The id of the tool call being executed — the same id the transcript's
    /// tool-result row and the `ToolStarted`/`ToolCompleted` events carry, so
    /// a tool can correlate anything it records or emits with the call.
    pub call_id: CallId,
    /// Caller thread id, when the parent run is threaded.
    pub thread_id: Option<ThreadId>,
    /// Caller recursion depth.
    pub depth: usize,
    /// Maximum output tokens requested for each model turn in the caller run.
    pub max_turn_output_tokens: Option<u32>,
    /// Shared event sink for nested-run observability.
    pub events: EventSink,
    /// The caller run's cancellation token.
    pub cancellation: CancellationToken,
    /// Whether the caller is driven through the streaming loop path.
    pub streaming: bool,
    /// The isolated workspace/sandbox available to the tool.
    pub workspace: Option<tinytools::WorkspaceDescriptor>,
    /// The run's hierarchical long-term store, when the host attached one
    /// with [`RunContext::with_namespaced_store`]; `None` otherwise.
    pub store: Option<Arc<dyn NamespacedStore>>,
    /// Type-erased application state snapshot, when the host attached one
    /// with [`RunContext::with_state_view`]. Read it through [`Self::state`].
    pub state_view: Option<Arc<dyn Any + Send + Sync>>,
    /// Typed host extensions of the run (see [`RunContext::with_host`]).
    /// Read through [`Self::host`].
    pub host_ext: crate::context::HostExtensions,
    /// Sink behind [`tinytools::ToolRunContext::report_progress`], captured
    /// from the loop's scoped progress gate when the context is built. Being
    /// held here (not looked up per report), it survives a move into a spawned
    /// task; the gate's closed flag silences it once the call settles.
    pub progress: Option<tinytools::ProgressSink>,
    /// Runner behind [`Self::call_tool`], installed by the agent loop for the
    /// call it is executing. `None` when the tool runs outside the loop.
    pub nested: Option<Arc<dyn super::NestedToolRunner>>,
}

impl ToolExecutionContext {
    /// Captures the non-generic, tool-visible parts of a live run for the
    /// tool call `call_id`.
    pub fn from_run_context<Ctx>(ctx: &RunContext<Ctx>, call_id: CallId) -> Self {
        Self {
            run_id: ctx.config.run_id.clone(),
            call_id: call_id.clone(),
            thread_id: ctx.config.thread_id.clone(),
            depth: ctx.depth(),
            max_turn_output_tokens: ctx.config.max_turn_output_tokens,
            events: ctx.events.clone(),
            cancellation: ctx.cancellation.clone(),
            streaming: ctx.streaming,
            workspace: ctx.workspace.clone(),
            store: ctx.namespaced_store.clone(),
            state_view: ctx.state_view.clone(),
            host_ext: ctx.host_ext.clone(),
            progress: super::progress::ToolProgressGate::current_sink_for(&call_id),
            nested: super::nested::current_for(&call_id),
        }
    }

    /// The harness context behind a canonical tool's erased run context.
    ///
    /// This is the one downcast documented above: it reads
    /// [`tinytools::ToolRunContext::host_extension`] and returns the
    /// `ToolExecutionContext` the agent loop installed there. `None` when there
    /// is no context, or when the context is not harness-driven (a test double
    /// or another host's `ToolRunContext`).
    #[must_use]
    pub fn of(context: Option<&dyn tinytools::ToolRunContext>) -> Option<&Self> {
        context
            .and_then(tinytools::ToolRunContext::host_extension)
            .and_then(|extension| extension.downcast_ref::<Self>())
    }

    /// The provider-assigned id of the tool call a canonical tool is running
    /// as, read from its erased run context.
    ///
    /// The portable [`tinytools::ToolRunContext`] surface deliberately does not
    /// carry the call id; it is harness-owned. This reads [`Self::call_id`] off
    /// the context [`Self::of`] recovers, so it is `None` exactly when that is.
    #[must_use]
    pub fn call_id_of(context: Option<&dyn tinytools::ToolRunContext>) -> Option<String> {
        Self::of(context).map(|harness| harness.call_id.as_str().to_string())
    }

    /// Attaches an isolated workspace descriptor the tool may operate in.
    #[must_use]
    pub fn with_workspace(mut self, workspace: tinytools::WorkspaceDescriptor) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// The application state the host attached, as `S`.
    ///
    /// `None` when no view was attached **or** when it was attached as a
    /// different type — a mismatch is never a panic, matching how
    /// [`crate::context::StateUpdate::apply`] treats a type it was not built
    /// for.
    #[must_use]
    pub fn state<S: 'static>(&self) -> Option<&S> {
        self.state_view
            .as_deref()
            .and_then(|view| view.downcast_ref::<S>())
    }

    /// The host extension of type `X` the run installed with
    /// [`RunContext::with_host`], or `None`.
    #[must_use]
    pub fn host<X: Any + Send + Sync>(&self) -> Option<&X> {
        self.host_ext.get::<X>()
    }

    /// The host extension of type `X` as a shared handle, for a tool that
    /// moves it into a spawned task.
    #[must_use]
    pub fn host_arc<X: Any + Send + Sync>(&self) -> Option<Arc<X>> {
        self.host_ext.get_arc::<X>()
    }

    /// The run's whole host extension map.
    #[must_use]
    pub fn host_extensions(&self) -> &crate::context::HostExtensions {
        &self.host_ext
    }

    /// Calls another registered tool as a **nested call** of this one and
    /// returns its result.
    ///
    /// The call is admitted like one the model issued, with exceptions: tool
    /// lookup and the host allow-list, argument validation, the approval
    /// refusal, [`Middleware::check_nested_tool`][crate::middleware::Middleware::check_nested_tool]
    /// on every middleware, host authorization (the request carries
    /// `parent_call_id`), the tool-wrap onion, timeouts, and the run's
    /// `max_tool_calls` budget (shared with model-issued calls) all apply, and
    /// [`Middleware::observe_nested_result`][crate::middleware::Middleware::observe_nested_result]
    /// sees the result. `before_tool` / `after_tool` proper, the progress gate
    /// and a result's `ToolControl` do **not** apply, so enforcement that is
    /// only a `before_tool` binds nested calls only if the middleware also
    /// implements `check_nested_tool`. See
    /// `docs/modules/harness/nested-tool-calls.md`.
    ///
    /// Nested calls are **disabled by default**: until the host sets
    /// `RunLimits::max_nested_depth` above `0` this returns "nested tool calls
    /// are disabled". Enable only once every `before_tool` enforcement also
    /// implements `check_nested_tool`.
    ///
    /// # Errors
    ///
    /// - No harness runner is installed (the tool runs outside the agent
    ///   loop).
    /// - The tool is unknown, not callable, or the arguments are invalid.
    /// - The nested depth (`RunLimits::max_nested_depth`) or the run's
    ///   tool-call budget is spent, or the run is cancelled.
    /// - The call would need approval or be deferred. A nested call **never**
    ///   defers its parent; it fails with
    ///   `nested call '<name>' requires approval; nested calls cannot be deferred`.
    ///
    /// A tool that reports its own failure (`ToolResult::is_error`) is an `Ok`
    /// result, as for a model-issued call.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> crate::error::Result<tinytools::ToolResult> {
        match &self.nested {
            Some(runner) => runner.call_tool(name, arguments).await,
            None => Err(crate::error::TinyAgentsError::ToolFailed(format!(
                "cannot call tool '{name}' from tool call '{}': nested tool calls are only \
                 available while the agent loop executes the calling tool",
                self.call_id
            ))),
        }
    }

    /// Emits an [`AgentEvent::Custom`] carrying `payload` on the run's event
    /// stream, correlated to this call.
    ///
    /// This is the tool's structured progress channel: it lands on the same
    /// ordered stream as the loop's own events (between this call's
    /// `ToolStarted` and `ToolCompleted`), so a UI or journal sees it in
    /// context. The harness attaches no meaning to the payload.
    pub fn custom(&self, payload: serde_json::Value) {
        self.events.emit(AgentEvent::Custom {
            call_id: Some(self.call_id.clone()),
            payload,
        });
    }
}

impl tinytools::ToolRunContext for ToolExecutionContext {
    fn workspace(&self) -> Option<&tinytools::WorkspaceDescriptor> {
        self.workspace.as_ref()
    }

    fn thread_id(&self) -> Option<&str> {
        self.thread_id.as_ref().map(ThreadId::as_str)
    }

    fn max_turn_output_tokens(&self) -> Option<u32> {
        self.max_turn_output_tokens
    }

    fn report_progress(&self, update: tinytools::ToolProgress) {
        if let Some(sink) = &self.progress {
            sink.report(update);
        }
    }

    fn host_extension(&self) -> Option<&(dyn Any + Send + Sync)> {
        Some(self)
    }
}
