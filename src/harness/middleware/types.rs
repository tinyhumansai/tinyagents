//! Type definitions for the harness middleware module.
//!
//! These are the typed extension points that wrap each level of the recursive
//! harness: the [`Middleware`] trait's hooks fire identically around the parent
//! agent loop and around every nested model/tool/agent call beneath it, so
//! observation and policy compose the same way at any recursion depth.
//!
//! This file holds every public type in `crate::harness::middleware`: the
//! [`AgentRun`] result record, the core [`Middleware`] trait, the
//! [`MiddlewareStack`] composer, and the built-in middleware implementations.
//! Behavioral code (trait default bodies, the stack runner, and built-in
//! `Middleware` impls) lives in the sibling `mod.rs`; focused tests live in
//! `test.rs`.
//!
//! All public items are re-exported through [`super`] so callers import from
//! `crate::harness::middleware` directly.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::{Result, TinyAgentsError};
use crate::harness::cache::CacheLayoutEvent;
use crate::harness::context::RunContext;
use crate::harness::model::{ModelDelta, ModelRequest, ModelResponse};
use crate::harness::summarization::{SummarizationPolicy, Summarizer, SummaryRecord, TrimStrategy};
use crate::harness::tool::{ToolCall, ToolDelta, ToolResult};
use crate::harness::usage::UsageTotals;

// ── AgentRun ────────────────────────────────────────────────────────────────

/// The accumulated result of a single agent run.
///
/// `AgentRun` decouples middleware (and other observers) from the internals of
/// the agent loop. The loop builds and threads an `AgentRun` through the run,
/// updating its counters and message log as model and tool calls complete, and
/// hands a `&mut AgentRun` to [`Middleware::after_agent`] so middleware can
/// inspect or post-process the final result without owning loop state.
///
/// # Example
///
/// ```
/// use tinyagents::harness::middleware::AgentRun;
///
/// let mut run = AgentRun::new();
/// run.model_calls += 1;
/// run.steps += 1;
/// assert_eq!(run.text(), None);
/// ```
#[derive(Clone, Debug, Default)]
pub struct AgentRun {
    /// The full conversation transcript produced by the run, in order.
    pub messages: Vec<crate::harness::message::Message>,
    /// The final model response, when the run produced one.
    pub final_response: Option<ModelResponse>,
    /// Parsed structured output, when the run requested a structured format.
    pub structured: Option<serde_json::Value>,
    /// Cumulative token usage across every model call in the run.
    pub usage: UsageTotals,
    /// Number of model calls dispatched during the run.
    pub model_calls: usize,
    /// Number of tool invocations executed during the run.
    pub tool_calls: usize,
    /// Number of loop iterations (model/tool super-steps) executed.
    pub steps: usize,
}

// ── Middleware trait ──────────────────────────────────────────────────────────

/// A cross-cutting extension point invoked around agent, model, and tool
/// execution.
///
/// Middleware is the primary way to add behavior — tracing, guardrails,
/// trimming, caching protection, usage accounting, retries — without touching
/// the agent loop or graph internals. Every hook has a no-op default so an
/// implementor overrides only the ones it cares about.
///
/// # Ordering (onion model)
///
/// When composed in a [`MiddlewareStack`], `before_*` hooks run in registration
/// order while `after_*` hooks run in **reverse** registration order. The first
/// registered middleware is therefore the outermost layer: it sets up first and
/// tears down last, mirroring common web-middleware stacks and keeping cleanup
/// symmetrical.
///
/// # Mutation
///
/// Hooks receive mutable references to the value flowing through the run
/// (`request`, `delta`, `response`, `call`, `result`) so they can transform it
/// in place. They also receive `&mut RunContext<Ctx>` for emitting events and
/// recording limits, plus a shared `&State` for read-only application state.
///
/// All hooks are async and return [`Result`]; returning `Err` short-circuits
/// the stack (see [`MiddlewareStack`]).
#[async_trait]
pub trait Middleware<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    /// A short, stable label used in `MiddlewareStarted`/`MiddlewareCompleted`
    /// events. This is intentionally synchronous and should return a `'static`
    /// string literal.
    fn name(&self) -> &str;

    /// Runs once before the agent loop begins, before any model call.
    async fn before_agent(&self, _ctx: &mut RunContext<Ctx>, _state: &State) -> Result<()> {
        Ok(())
    }

    /// Runs once after the agent loop finishes, with the completed [`AgentRun`]
    /// available for inspection or post-processing.
    async fn after_agent(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _run: &mut AgentRun,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs before each model request is dispatched, allowing the middleware to
    /// mutate the outgoing [`ModelRequest`].
    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _request: &mut ModelRequest,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs for each streamed [`ModelDelta`] before it is forwarded or
    /// accumulated, allowing inspection or transformation of the chunk.
    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ModelDelta,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs after each model call completes, allowing the middleware to mutate
    /// the [`ModelResponse`].
    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _response: &mut ModelResponse,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs before each tool invocation, allowing the middleware to mutate the
    /// outgoing [`ToolCall`].
    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _call: &mut ToolCall,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs for each streamed [`ToolDelta`] of progress emitted while a tool
    /// runs.
    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ToolDelta,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs after each tool invocation completes, allowing the middleware to
    /// mutate the [`ToolResult`].
    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _result: &mut ToolResult,
    ) -> Result<()> {
        Ok(())
    }

    /// Runs when any hook in the stack errors, giving every middleware a chance
    /// to log, redact, or react to the failure. The original error is still
    /// returned to the caller after this runs; errors from `on_error` itself
    /// are ignored so they cannot mask the root cause.
    async fn on_error(&self, _ctx: &mut RunContext<Ctx>, _error: &TinyAgentsError) -> Result<()> {
        Ok(())
    }
}

// ── Wrap (around-call) middleware ─────────────────────────────────────────────

/// A pinned, boxed future producing a [`ModelResponse`].
///
/// This is the return type of [`ModelBaseCall::call`] and of the futures wrap
/// middleware drive. The lifetime `'a` ties the future to the borrows it
/// captures (the run context, application state, and the base handler).
pub type BoxModelFuture<'a> = Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>>;

/// A pinned, boxed future producing a [`ToolResult`].
///
/// The tool-wrap counterpart of [`BoxModelFuture`].
pub type BoxToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;

/// The innermost model call wrapped by the [`ModelMiddleware`] onion.
///
/// This is the *real* model invocation — the cache + retry + fallback core of
/// the agent loop. The loop supplies it as the `base` of
/// [`MiddlewareStack::run_wrapped_model`]. A wrap middleware reaches it (the
/// innermost layer) by calling [`ModelHandler::run`], possibly more than once
/// (for retry) or not at all (to short-circuit).
pub trait ModelBaseCall<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Invokes the wrapped model call with the (possibly middleware-mutated)
    /// `request`.
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        request: ModelRequest,
    ) -> BoxModelFuture<'a>;
}

/// The innermost tool call wrapped by the [`ToolMiddleware`] onion.
///
/// The tool-wrap counterpart of [`ModelBaseCall`]; the loop supplies the real
/// tool invocation as the `base` of [`MiddlewareStack::run_wrapped_tool`].
pub trait ToolBaseCall<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Invokes the wrapped tool with the (possibly middleware-mutated) `call`.
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        call: ToolCall,
    ) -> BoxToolFuture<'a>;
}

/// The outcome of a wrapped model call.
///
/// Carries the [`ModelResponse`] the wrapped call resolves to. A
/// [`ModelMiddleware`] produces one by either:
///
/// - **proceeding** — forwarding the response returned by [`ModelHandler::run`];
/// - **short-circuiting / replacing** — constructing a [`Self::Response`]
///   without ever calling `next`;
/// - **retrying** — calling `next` in a loop until it succeeds or a budget is
///   exhausted; or
/// - **falling back** — calling `next`, then substituting a response on error.
///
/// Retry and replacement are therefore expressed by *how* a middleware uses
/// `next` rather than by distinct enum variants; the enum only needs to carry
/// the resolved response. It is [`non_exhaustive`] so future control variants
/// can be added without breaking callers.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MiddlewareModelOutcome {
    /// The response to use as the result of the wrapped model call.
    Response(ModelResponse),
}

impl MiddlewareModelOutcome {
    /// Unwraps the contained [`ModelResponse`].
    pub fn into_response(self) -> ModelResponse {
        match self {
            Self::Response(response) => response,
        }
    }
}

impl From<ModelResponse> for MiddlewareModelOutcome {
    fn from(response: ModelResponse) -> Self {
        Self::Response(response)
    }
}

/// The outcome of a wrapped tool call.
///
/// The tool-wrap counterpart of [`MiddlewareModelOutcome`]; see its docs for the
/// proceed / replace / retry / fallback patterns. [`non_exhaustive`] for the
/// same forward-compatibility reason.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MiddlewareToolOutcome {
    /// The result to use as the result of the wrapped tool call.
    Result(ToolResult),
}

impl MiddlewareToolOutcome {
    /// Unwraps the contained [`ToolResult`].
    pub fn into_result(self) -> ToolResult {
        match self {
            Self::Result(result) => result,
        }
    }
}

impl From<ToolResult> for MiddlewareToolOutcome {
    fn from(result: ToolResult) -> Self {
        Self::Result(result)
    }
}

/// A handle to the remainder of the model-wrap onion: the inner wrap middleware
/// plus the innermost [`ModelBaseCall`].
///
/// A [`ModelMiddleware`] receives this as its `next` argument. Calling
/// [`ModelHandler::run`] **proceeds** to the next layer (eventually the base
/// model call); not calling it **short-circuits**; calling it repeatedly
/// **retries**. `run` borrows `&self`, so a middleware may invoke `next` as many
/// times as it likes.
pub struct ModelHandler<'a, State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) remaining: &'a [Arc<dyn ModelMiddleware<State, Ctx>>],
    pub(crate) base: &'a dyn ModelBaseCall<State, Ctx>,
}

/// A handle to the remainder of the tool-wrap onion: the inner wrap middleware
/// plus the innermost [`ToolBaseCall`].
///
/// The tool-wrap counterpart of [`ModelHandler`]; see its docs.
pub struct ToolHandler<'a, State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) remaining: &'a [Arc<dyn ToolMiddleware<State, Ctx>>],
    pub(crate) base: &'a dyn ToolBaseCall<State, Ctx>,
}

/// Around-call ("wrap") middleware for model invocations.
///
/// Unlike the lifecycle [`Middleware`] hooks (which only observe/mutate values
/// flowing past), a `ModelMiddleware` *surrounds* the inner model pipeline: it
/// receives a [`ModelHandler`] (`next`) that runs the rest of the onion plus the
/// real model call. This is the most powerful extension point — it can proceed,
/// short-circuit with a replacement response, retry `next` in a loop, or fall
/// back — all while keeping setup and teardown symmetrical around the call.
///
/// # Ordering
///
/// Wrap middleware compose as a nested onion: the first-registered middleware is
/// the **outermost** layer (it runs first and finishes last), and the innermost
/// layer is the real model call. See [`MiddlewareStack::run_wrapped_model`].
#[async_trait]
pub trait ModelMiddleware<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    /// A short, stable label used in
    /// `MiddlewareStarted`/`MiddlewareCompleted` events.
    fn name(&self) -> &str;

    /// Wraps the inner model pipeline. Call `next.run(ctx, state, request)` to
    /// proceed (zero or more times), or return a [`MiddlewareModelOutcome`]
    /// without calling it to short-circuit.
    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome>;
}

/// Around-call ("wrap") middleware for tool invocations.
///
/// The tool-wrap counterpart of [`ModelMiddleware`]; see its docs for the full
/// proceed / replace / retry / fallback model and onion ordering.
#[async_trait]
pub trait ToolMiddleware<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    /// A short, stable label used in
    /// `MiddlewareStarted`/`MiddlewareCompleted` events.
    fn name(&self) -> &str;

    /// Wraps the inner tool pipeline. Call `next.run(ctx, state, call)` to
    /// proceed (zero or more times), or return a [`MiddlewareToolOutcome`]
    /// without calling it to short-circuit.
    async fn wrap_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome>;
}

// ── MiddlewareStack ───────────────────────────────────────────────────────────

/// An ordered collection of [`Middleware`] composed with onion semantics.
///
/// `before_*` runner methods invoke each middleware in registration order;
/// `after_*` runner methods invoke them in reverse. Every per-middleware hook
/// invocation is bracketed by `AgentEvent::MiddlewareStarted` and
/// `MiddlewareCompleted` events emitted through the [`RunContext`]. The first
/// hook that returns `Err` short-circuits the stack: every middleware's
/// [`Middleware::on_error`] is invoked, then the original error is returned.
///
/// In addition to those lifecycle hooks, the stack holds two ordered lists of
/// **wrap** (around-call) middleware — [`ModelMiddleware`] and
/// [`ToolMiddleware`] — composed by [`MiddlewareStack::run_wrapped_model`] and
/// [`MiddlewareStack::run_wrapped_tool`] as a nested onion whose innermost layer
/// is the real model/tool call.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use tinyagents::harness::middleware::{LoggingMiddleware, MiddlewareStack};
///
/// let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
/// stack.push(Arc::new(LoggingMiddleware::new()));
/// assert_eq!(stack.len(), 1);
/// ```
pub struct MiddlewareStack<State: Send + Sync, Ctx: Send + Sync = ()> {
    pub(crate) middlewares: Vec<Arc<dyn Middleware<State, Ctx>>>,
    pub(crate) model_middlewares: Vec<Arc<dyn ModelMiddleware<State, Ctx>>>,
    pub(crate) tool_middlewares: Vec<Arc<dyn ToolMiddleware<State, Ctx>>>,
}

// ── LoggingMiddleware ─────────────────────────────────────────────────────────

/// Per-hook invocation counts captured by [`LoggingMiddleware`].
///
/// A snapshot is returned from [`LoggingMiddleware::counts`] so tests and
/// dashboards can assert which hooks fired and how often.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HookCounts {
    /// Number of `before_agent` invocations.
    pub before_agent: usize,
    /// Number of `after_agent` invocations.
    pub after_agent: usize,
    /// Number of `before_model` invocations.
    pub before_model: usize,
    /// Number of `on_model_delta` invocations.
    pub on_model_delta: usize,
    /// Number of `after_model` invocations.
    pub after_model: usize,
    /// Number of `before_tool` invocations.
    pub before_tool: usize,
    /// Number of `on_tool_delta` invocations.
    pub on_tool_delta: usize,
    /// Number of `after_tool` invocations.
    pub after_tool: usize,
    /// Number of `on_error` invocations.
    pub on_error: usize,
}

/// Observation-only middleware that records how often each hook fired.
///
/// `LoggingMiddleware` mutates nothing in the run; it only increments interior
/// counters so callers can inspect hook activity via [`LoggingMiddleware::counts`].
/// The surrounding [`MiddlewareStack`] already emits start/completed events, so
/// this type adds no events of its own.
pub struct LoggingMiddleware {
    pub(crate) label: &'static str,
    pub(crate) counts: Mutex<HookCounts>,
}

// ── MessageTrimMiddleware ─────────────────────────────────────────────────────

/// Middleware that trims the request transcript before each model call.
///
/// In `before_model` it replaces `request.messages` with the result of
/// [`crate::harness::summarization::trim_messages`] under the configured
/// [`TrimStrategy`], bounding prompt growth across long agent loops.
pub struct MessageTrimMiddleware {
    /// The trimming strategy applied to `request.messages`.
    pub strategy: TrimStrategy,
}

// ── ContextCompressionMiddleware ──────────────────────────────────────────────

/// Middleware that summarizes/compresses the request transcript, but **only**
/// when it nears the model's context window.
///
/// In `before_model` it consults the configured [`SummarizationPolicy`]. The
/// policy is normally built with a context window (for example via
/// [`SummarizationPolicy::from_profile`] or
/// [`SummarizationPolicy::with_context_window`]) and a `threshold_fraction`
/// (default `0.9`). When the estimated transcript tokens are **below** the
/// window threshold this middleware is a complete no-op: `request.messages` is
/// left untouched and no event is emitted. When the threshold is reached, the
/// older messages are condensed by the [`Summarizer`] into a single summary
/// message, the recent window and system messages are kept verbatim, the
/// resulting [`SummaryRecord`] (with its compression provenance) is recorded,
/// and an [`AgentEvent::Compressed`][crate::harness::events::AgentEvent::Compressed]
/// event is emitted.
///
/// [`ConcatSummarizer`][crate::harness::summarization::ConcatSummarizer] is used
/// by default; supply any [`Summarizer`] via
/// [`ContextCompressionMiddleware::with_summarizer`].
/// Default cap on the number of [`SummaryRecord`]s a
/// [`ContextCompressionMiddleware`] retains before evicting the oldest.
pub const DEFAULT_COMPRESSION_RECORD_CAP: usize = 1024;

pub struct ContextCompressionMiddleware {
    pub(crate) label: &'static str,
    pub(crate) policy: SummarizationPolicy,
    pub(crate) summarizer: Box<dyn Summarizer>,
    pub(crate) records: Mutex<VecDeque<SummaryRecord>>,
    pub(crate) max_records: usize,
}

// ── MicrocompactMiddleware ────────────────────────────────────────────────────

/// Middleware that clears the bodies of older **tool-result** messages while
/// keeping the `keep_recent` most recent ones verbatim.
///
/// In `before_model` it walks `request.messages`, finds the tool-result
/// messages, and — once there are more than `keep_recent` of them — replaces the
/// content of every tool result *except* the newest `keep_recent` with a fixed
/// [`placeholder`](Self::placeholder) string, preserving each message's
/// `tool_call_id`. Non-tool messages (system/user/assistant) are never touched
/// and no chat turn is dropped, so this bounds the cost of a long, tool-heavy
/// thread without the semantic loss of summarization.
///
/// This is the "micro-compaction" companion to
/// [`ContextCompressionMiddleware`]: the latter summarizes *older chat history*
/// when the transcript nears the context window, whereas this one only ever
/// blanks *stale tool payloads* that the model no longer needs verbatim. The two
/// compose cleanly.
///
/// The operation is **idempotent**: a body already equal to the placeholder is
/// left as-is, so repeated `before_model` passes converge. When there are at
/// most `keep_recent` tool results the middleware is a complete no-op.
///
/// The placeholder text is caller-supplied (via
/// [`MicrocompactMiddleware::new`]) so host applications can keep their own
/// model-facing wording stable. Event emission is **opt-in** (default off, see
/// [`MicrocompactMiddleware::with_events`]): when enabled and at least one body
/// is cleared, an
/// [`AgentEvent::Compressed`][crate::harness::events::AgentEvent::Compressed]
/// event carrying the before/after token estimate is emitted; when disabled the
/// middleware mutates the request silently.
pub struct MicrocompactMiddleware {
    pub(crate) label: &'static str,
    pub(crate) keep_recent: usize,
    pub(crate) placeholder: String,
    pub(crate) emit_events: bool,
}

// ── PromptCacheGuardMiddleware ────────────────────────────────────────────────

/// Middleware that watches the prompt cache layout for accidental prefix
/// invalidations.
///
/// In `before_model` it computes the request's
/// [`crate::harness::cache::PromptCacheLayout`]. If a layout from a previous
/// call was stored and the cacheable prefix changed, it records a
/// [`CacheLayoutEvent`] (retrievable via
/// [`PromptCacheGuardMiddleware::layout_events`]) so KV-cache regressions are
/// observable. This demonstrates provider prompt/KV-cache prefix protection.
/// Default cap on the number of [`CacheLayoutEvent`]s a
/// [`PromptCacheGuardMiddleware`] retains before evicting the oldest.
pub const DEFAULT_CACHE_GUARD_EVENT_CAP: usize = 1024;

pub struct PromptCacheGuardMiddleware {
    pub(crate) label: &'static str,
    pub(crate) previous: Mutex<Option<crate::harness::cache::PromptCacheLayout>>,
    pub(crate) events: Mutex<VecDeque<CacheLayoutEvent>>,
    pub(crate) max_events: usize,
}

// ── UsageAccountingMiddleware ─────────────────────────────────────────────────

/// Middleware that folds each model response's usage into a running total.
///
/// In `after_model` it records `response.usage` into an internal
/// [`UsageTotals`]. The accumulated totals are available via
/// [`UsageAccountingMiddleware::totals`] for cost reporting and tests.
pub struct UsageAccountingMiddleware {
    pub(crate) label: &'static str,
    pub(crate) totals: Mutex<UsageTotals>,
}
