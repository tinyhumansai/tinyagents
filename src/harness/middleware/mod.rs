//! Middleware stack.
//!
//! In the recursive (RLM-style) harness, middleware is the layer that wraps
//! *every* level of the recursion identically: because a sub-agent or sub-graph
//! is just another agent loop, the same before/after hooks bracket the parent
//! run and each nested model/tool/agent call beneath it. That uniform wrapping
//! is what lets cross-cutting concerns — tracing, usage/cost roll-up, guardrails
//! — compose consistently as models call models and graphs run graphs.
//!
//! Owns the before/after hooks that wrap agent, model, and tool execution.
//! Cross-cutting behavior such as tracing, guardrails, message trimming,
//! prompt-cache protection, and usage accounting lives here as [`Middleware`]
//! implementations composed through a [`MiddlewareStack`].
//!
//! # Layout
//!
//! - [`types`] holds every public type (the [`Middleware`] trait, [`AgentRun`],
//!   [`MiddlewareStack`], and the built-in middleware).
//! - This file holds the impls: trait default bodies live with the trait in
//!   `types.rs`; here are the [`AgentRun`] helpers, the stack runner, and the
//!   built-in `Middleware` implementations.
//!
//! # Onion ordering
//!
//! `before_*` hooks run in registration order and `after_*` hooks run in
//! reverse, so the first-registered middleware is the outermost layer. The
//! first hook that errors short-circuits the stack: every middleware's
//! [`Middleware::on_error`] runs, then the original error is returned.

mod types;

pub use types::*;

pub mod library;
pub use library::*;

use std::sync::Arc;

use crate::error::{Result, TinyAgentsError};
use crate::harness::context::RunContext;
use crate::harness::events::AgentEvent;
use crate::harness::model::{ModelDelta, ModelRequest, ModelResponse};
use crate::harness::tool::{ToolCall, ToolDelta, ToolResult};

/// Runs one per-middleware lifecycle hook across the whole stack, bracketing
/// each call with `MiddlewareStarted`/`MiddlewareCompleted` events and fanning
/// `on_error` out to every middleware on the first failure (so the originating
/// error is never masked).
///
/// This is factored as a macro rather than an async helper because each hook
/// takes different arguments and borrows `ctx` mutably across its `await`, which
/// a closure-based helper cannot express without heap-boxing every call.
///
/// Crucially, `MiddlewareCompleted` is emitted on *both* the success and error
/// paths: a hook that returns `Err` can no longer leave a dangling
/// `MiddlewareStarted` with no matching `Completed` in the event stream. `$iter`
/// selects registration order (`.iter()`) or reverse order (`.iter().rev()`);
/// `$call` is the (un-awaited) hook invocation on `$mw`.
macro_rules! run_stack_hook {
    ($self:ident, $ctx:ident, $iter:expr, |$mw:ident| $call:expr) => {{
        for $mw in $iter {
            let name = $mw.name().to_string();
            $ctx.emit(AgentEvent::MiddlewareStarted { name: name.clone() });
            let result = $call.await;
            $ctx.emit(AgentEvent::MiddlewareCompleted { name });
            if let Err(e) = result {
                $self.fan_out_on_error($ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }};
}

// ── AgentRun ────────────────────────────────────────────────────────────────

impl AgentRun {
    /// Creates an empty agent-run record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the final response text, if the run produced a final response.
    pub fn text(&self) -> Option<String> {
        self.final_response.as_ref().map(|r| r.text())
    }
}

// ── MiddlewareStack ───────────────────────────────────────────────────────────

impl<State: Send + Sync, Ctx: Send + Sync> Default for MiddlewareStack<State, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> MiddlewareStack<State, Ctx> {
    /// Creates an empty middleware stack.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
            model_middlewares: Vec::new(),
            tool_middlewares: Vec::new(),
        }
    }

    /// Appends a lifecycle [`Middleware`] to the stack. Registration order is the
    /// onion order: the first pushed middleware is the outermost layer.
    pub fn push(&mut self, middleware: Arc<dyn Middleware<State, Ctx>>) {
        self.middlewares.push(middleware);
    }

    /// Appends a [`ModelMiddleware`] (around-model wrap hook). Registration order
    /// is the onion order: the first pushed wrap middleware is the **outermost**
    /// layer and the real model call is the innermost.
    pub fn push_model_middleware(&mut self, middleware: Arc<dyn ModelMiddleware<State, Ctx>>) {
        self.model_middlewares.push(middleware);
    }

    /// Appends a [`ToolMiddleware`] (around-tool wrap hook). Registration order
    /// is the onion order: the first pushed wrap middleware is the **outermost**
    /// layer and the real tool call is the innermost.
    pub fn push_tool_middleware(&mut self, middleware: Arc<dyn ToolMiddleware<State, Ctx>>) {
        self.tool_middlewares.push(middleware);
    }

    /// Returns the number of registered [`ModelMiddleware`] wrap hooks.
    pub fn model_middleware_len(&self) -> usize {
        self.model_middlewares.len()
    }

    /// Returns the number of registered [`ToolMiddleware`] wrap hooks.
    pub fn tool_middleware_len(&self) -> usize {
        self.tool_middlewares.len()
    }

    /// Returns the number of registered middleware.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Returns `true` if no middleware are registered.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Fans `on_error` out to every middleware, ignoring their results so the
    /// original error is never masked. No start/completed events are emitted on
    /// this internal recovery path.
    async fn fan_out_on_error(&self, ctx: &mut RunContext<Ctx>, error: &TinyAgentsError) {
        for mw in self.middlewares.iter() {
            let _ = mw.on_error(ctx, error).await;
        }
    }

    /// Runs every middleware's [`Middleware::before_agent`] in registration
    /// order.
    pub async fn run_before_agent(&self, ctx: &mut RunContext<Ctx>, state: &State) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter(), |mw| mw
            .before_agent(ctx, state))
    }

    /// Runs every middleware's [`Middleware::after_agent`] in reverse
    /// registration order.
    pub async fn run_after_agent(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        run: &mut AgentRun,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter().rev(), |mw| mw
            .after_agent(ctx, state, run))
    }

    /// Runs every middleware's [`Middleware::before_model`] in registration
    /// order, threading the mutable request through each.
    pub async fn run_before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter(), |mw| mw
            .before_model(ctx, state, request))
    }

    /// Runs every middleware's [`Middleware::on_model_delta`] in registration
    /// order for one streamed delta.
    ///
    /// Unlike the other stack runners, the per-delta hook is deliberately *not*
    /// bracketed by `MiddlewareStarted`/`MiddlewareCompleted` events. This runs
    /// on the streaming hot path — potentially hundreds of times per second per
    /// middleware — and emitting two events (each cloning `mw.name()` and
    /// acquiring the recorder mutex) per middleware per token dominated the
    /// stream loop's cost for zero observability value. Callers that need to
    /// observe delta-level middleware activity should instrument the hook
    /// itself.
    pub async fn run_on_model_delta(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        delta: &mut ModelDelta,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            if let Err(e) = mw.on_model_delta(ctx, state, delta).await {
                self.fan_out_on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::after_model`] in reverse
    /// registration order, threading the mutable response through each.
    pub async fn run_after_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter().rev(), |mw| mw
            .after_model(ctx, state, response))
    }

    /// Runs every middleware's [`Middleware::before_tool`] in registration
    /// order, threading the mutable tool call through each.
    pub async fn run_before_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter(), |mw| mw
            .before_tool(ctx, state, call))
    }

    /// Runs every middleware's [`Middleware::on_tool_delta`] in registration
    /// order for one streamed tool-progress delta.
    pub async fn run_on_tool_delta(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        delta: &mut ToolDelta,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter(), |mw| mw
            .on_tool_delta(ctx, state, delta))
    }

    /// Runs every middleware's [`Middleware::after_tool`] in reverse
    /// registration order, threading the mutable tool result through each.
    pub async fn run_after_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        result: &mut ToolResult,
    ) -> Result<()> {
        run_stack_hook!(self, ctx, self.middlewares.iter().rev(), |mw| mw
            .after_tool(ctx, state, result))
    }

    /// Runs every middleware's [`Middleware::on_error`] in registration order,
    /// bracketing each with start/completed events. Inner errors are ignored so
    /// the originating error is never masked; this method always returns `Ok`.
    pub async fn run_on_error(
        &self,
        ctx: &mut RunContext<Ctx>,
        error: &TinyAgentsError,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            let _ = mw.on_error(ctx, error).await;
            ctx.emit(AgentEvent::MiddlewareCompleted {
                name: mw.name().to_string(),
            });
        }
        Ok(())
    }

    /// Runs the registered [`ModelMiddleware`] wrap hooks as a nested onion
    /// around `base` (the real model call) and returns the resolved
    /// [`MiddlewareModelOutcome`].
    ///
    /// The first-registered wrap middleware is the outermost layer; `base` is
    /// the innermost. With no wrap middleware registered this simply runs `base`
    /// and wraps its response. Each layer is bracketed by
    /// `MiddlewareStarted`/`MiddlewareCompleted` events.
    pub async fn run_wrapped_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        base: &dyn ModelBaseCall<State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        let handler = ModelHandler {
            remaining: &self.model_middlewares,
            base,
        };
        handler.run(ctx, state, request).await
    }

    /// Runs the registered [`ToolMiddleware`] wrap hooks as a nested onion around
    /// `base` (the real tool call) and returns the resolved
    /// [`MiddlewareToolOutcome`].
    ///
    /// The tool-wrap counterpart of [`Self::run_wrapped_model`].
    pub async fn run_wrapped_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        base: &dyn ToolBaseCall<State, Ctx>,
    ) -> Result<MiddlewareToolOutcome> {
        let handler = ToolHandler {
            remaining: &self.tool_middlewares,
            base,
        };
        handler.run(ctx, state, call).await
    }
}

// ── Wrap onion handlers ───────────────────────────────────────────────────────

impl<State: Send + Sync, Ctx: Send + Sync> ModelHandler<'_, State, Ctx> {
    /// Advances the model-wrap onion one layer: invokes the next
    /// [`ModelMiddleware`] (bracketed by start/completed events), or the base
    /// model call when no wrap middleware remain.
    ///
    /// Borrows `&self`, so a wrap middleware may call `run` zero times
    /// (short-circuit), once (proceed), or many times (retry).
    pub async fn run(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
    ) -> Result<MiddlewareModelOutcome> {
        match self.remaining.split_first() {
            Some((head, tail)) => {
                let next = ModelHandler {
                    remaining: tail,
                    base: self.base,
                };
                let name = head.name().to_string();
                ctx.emit(AgentEvent::MiddlewareStarted { name: name.clone() });
                // Emit `Completed` whether the wrap layer succeeds or errors, so
                // a failing layer never leaves a dangling `Started` in the event
                // stream (the onion's balance invariant).
                let outcome = head.wrap_model(ctx, state, request, next).await;
                ctx.emit(AgentEvent::MiddlewareCompleted { name });
                outcome
            }
            None => Ok(MiddlewareModelOutcome::Response(
                self.base.call(ctx, state, request).await?,
            )),
        }
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolHandler<'_, State, Ctx> {
    /// Advances the tool-wrap onion one layer. The tool-wrap counterpart of
    /// [`ModelHandler::run`].
    pub async fn run(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: ToolCall,
    ) -> Result<MiddlewareToolOutcome> {
        match self.remaining.split_first() {
            Some((head, tail)) => {
                let next = ToolHandler {
                    remaining: tail,
                    base: self.base,
                };
                let name = head.name().to_string();
                ctx.emit(AgentEvent::MiddlewareStarted { name: name.clone() });
                // Balance `Started` with `Completed` even when the wrap layer
                // errors (see `ModelHandler::run`).
                let outcome = head.wrap_tool(ctx, state, call, next).await;
                ctx.emit(AgentEvent::MiddlewareCompleted { name });
                outcome
            }
            None => Ok(MiddlewareToolOutcome::Result(
                self.base.call(ctx, state, call).await?,
            )),
        }
    }
}

#[cfg(test)]
mod test;
