//! Type definitions for the harness middleware module.
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

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::{Result, TinyAgentsError};
use crate::harness::cache::CacheLayoutEvent;
use crate::harness::context::RunContext;
use crate::harness::model::{ModelDelta, ModelRequest, ModelResponse};
use crate::harness::summarization::TrimStrategy;
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
pub struct PromptCacheGuardMiddleware {
    pub(crate) label: &'static str,
    pub(crate) previous: Mutex<Option<crate::harness::cache::PromptCacheLayout>>,
    pub(crate) events: Mutex<Vec<CacheLayoutEvent>>,
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
