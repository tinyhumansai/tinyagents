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
use crate::harness::cache::{CacheLayoutEvent, PromptCacheLayout};
use crate::harness::context::RunContext;
use crate::harness::events::AgentEvent;
use crate::harness::model::{ModelDelta, ModelRequest, ModelResponse};
use crate::harness::summarization::{
    ConcatSummarizer, SummarizationPolicy, Summarizer, SummaryRecord, TrimStrategy,
    estimate_tokens, trim_messages,
};
use crate::harness::tool::{ToolCall, ToolDelta, ToolResult};
use crate::harness::usage::UsageTotals;

use async_trait::async_trait;

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
        for mw in self.middlewares.iter() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.before_agent(ctx, state).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::after_agent`] in reverse
    /// registration order.
    pub async fn run_after_agent(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        run: &mut AgentRun,
    ) -> Result<()> {
        for mw in self.middlewares.iter().rev() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.after_agent(ctx, state, run).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::before_model`] in registration
    /// order, threading the mutable request through each.
    pub async fn run_before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.before_model(ctx, state, request).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::on_model_delta`] in registration
    /// order for one streamed delta.
    pub async fn run_on_model_delta(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        delta: &mut ModelDelta,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.on_model_delta(ctx, state, delta).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
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
        for mw in self.middlewares.iter().rev() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.after_model(ctx, state, response).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::before_tool`] in registration
    /// order, threading the mutable tool call through each.
    pub async fn run_before_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.before_tool(ctx, state, call).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::on_tool_delta`] in registration
    /// order for one streamed tool-progress delta.
    pub async fn run_on_tool_delta(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        delta: &mut ToolDelta,
    ) -> Result<()> {
        for mw in self.middlewares.iter() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.on_tool_delta(ctx, state, delta).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Runs every middleware's [`Middleware::after_tool`] in reverse
    /// registration order, threading the mutable tool result through each.
    pub async fn run_after_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        result: &mut ToolResult,
    ) -> Result<()> {
        for mw in self.middlewares.iter().rev() {
            ctx.emit(AgentEvent::MiddlewareStarted {
                name: mw.name().to_string(),
            });
            match mw.after_tool(ctx, state, result).await {
                Ok(()) => {
                    ctx.emit(AgentEvent::MiddlewareCompleted {
                        name: mw.name().to_string(),
                    });
                }
                Err(e) => {
                    self.fan_out_on_error(ctx, &e).await;
                    return Err(e);
                }
            }
        }
        Ok(())
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
                ctx.emit(AgentEvent::MiddlewareStarted {
                    name: head.name().to_string(),
                });
                let outcome = head.wrap_model(ctx, state, request, next).await?;
                ctx.emit(AgentEvent::MiddlewareCompleted {
                    name: head.name().to_string(),
                });
                Ok(outcome)
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
                ctx.emit(AgentEvent::MiddlewareStarted {
                    name: head.name().to_string(),
                });
                let outcome = head.wrap_tool(ctx, state, call, next).await?;
                ctx.emit(AgentEvent::MiddlewareCompleted {
                    name: head.name().to_string(),
                });
                Ok(outcome)
            }
            None => Ok(MiddlewareToolOutcome::Result(
                self.base.call(ctx, state, call).await?,
            )),
        }
    }
}

// ── LoggingMiddleware ─────────────────────────────────────────────────────────

impl LoggingMiddleware {
    /// Creates a logging middleware with the default label `"logging"`.
    pub fn new() -> Self {
        Self::with_label("logging")
    }

    /// Creates a logging middleware with a custom static label.
    pub fn with_label(label: &'static str) -> Self {
        Self {
            label,
            counts: std::sync::Mutex::new(HookCounts::default()),
        }
    }

    /// Returns a snapshot of the per-hook invocation counts recorded so far.
    pub fn counts(&self) -> HookCounts {
        self.counts.lock().expect("counts mutex poisoned").clone()
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for LoggingMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_agent(&self, _ctx: &mut RunContext<Ctx>, _state: &State) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .before_agent += 1;
        Ok(())
    }

    async fn after_agent(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _run: &mut AgentRun,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .after_agent += 1;
        Ok(())
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .before_model += 1;
        Ok(())
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ModelDelta,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .on_model_delta += 1;
        Ok(())
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _response: &mut ModelResponse,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .after_model += 1;
        Ok(())
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _call: &mut ToolCall,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .before_tool += 1;
        Ok(())
    }

    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ToolDelta,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .on_tool_delta += 1;
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _result: &mut ToolResult,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .after_tool += 1;
        Ok(())
    }

    async fn on_error(&self, _ctx: &mut RunContext<Ctx>, _error: &TinyAgentsError) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").on_error += 1;
        Ok(())
    }
}

// ── MessageTrimMiddleware ─────────────────────────────────────────────────────

impl MessageTrimMiddleware {
    /// Creates a trim middleware using the given [`TrimStrategy`].
    pub fn new(strategy: TrimStrategy) -> Self {
        Self { strategy }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for MessageTrimMiddleware {
    fn name(&self) -> &str {
        "message_trim"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.messages = trim_messages(&request.messages, &self.strategy);
        Ok(())
    }
}

// ── ContextCompressionMiddleware ──────────────────────────────────────────────

/// Estimate the total tokens of a message slice using the same per-message
/// heuristic the [`SummarizationPolicy`] uses internally.
fn total_message_tokens(messages: &[crate::harness::message::Message]) -> u64 {
    messages.iter().map(|m| estimate_tokens(&m.text())).sum()
}

impl ContextCompressionMiddleware {
    /// Creates a compression middleware backed by the default
    /// [`ConcatSummarizer`].
    pub fn new(policy: SummarizationPolicy) -> Self {
        Self::with_summarizer(policy, Box::new(ConcatSummarizer))
    }

    /// Creates a compression middleware with a custom [`Summarizer`].
    pub fn with_summarizer(policy: SummarizationPolicy, summarizer: Box<dyn Summarizer>) -> Self {
        Self {
            label: "context_compression",
            policy,
            summarizer,
            records: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Returns the configured [`SummarizationPolicy`].
    pub fn policy(&self) -> &SummarizationPolicy {
        &self.policy
    }

    /// Returns the [`SummaryRecord`]s produced so far, in order. Each record
    /// carries the compression provenance for one compaction.
    pub fn records(&self) -> Vec<SummaryRecord> {
        self.records.lock().expect("records mutex poisoned").clone()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ContextCompressionMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        // Below the window threshold: pass through untouched (no-op, no event).
        if !self.policy.should_summarize(&request.messages) {
            return Ok(());
        }

        let (to_summarize, to_keep) = self.policy.plan(&request.messages);
        // Nothing old enough to compress (e.g. keep_last covers everything):
        // leave the transcript untouched rather than summarizing an empty set.
        if to_summarize.is_empty() {
            return Ok(());
        }

        let from_tokens = total_message_tokens(&request.messages);
        let record = self.summarizer.summarize(&to_summarize).await?;

        let mut new_messages = Vec::with_capacity(to_keep.len() + 1);
        new_messages.push(record.summary.clone());
        new_messages.extend(to_keep);
        let to_tokens = total_message_tokens(&new_messages);

        self.records
            .lock()
            .expect("records mutex poisoned")
            .push(record);
        request.messages = new_messages;

        ctx.emit(AgentEvent::Compressed {
            from_tokens,
            to_tokens,
        });
        Ok(())
    }
}

// ── PromptCacheGuardMiddleware ────────────────────────────────────────────────

impl PromptCacheGuardMiddleware {
    /// Creates a cache-guard middleware with the default label
    /// `"prompt_cache_guard"`.
    pub fn new() -> Self {
        Self {
            label: "prompt_cache_guard",
            previous: std::sync::Mutex::new(None),
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Returns the cache-layout change events recorded so far, in order.
    pub fn layout_events(&self) -> Vec<CacheLayoutEvent> {
        self.events.lock().expect("events mutex poisoned").clone()
    }
}

impl Default for PromptCacheGuardMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for PromptCacheGuardMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        let layout = PromptCacheLayout::from_request(request);
        let mut previous = self.previous.lock().expect("previous mutex poisoned");
        if let Some(prev) = previous.as_ref()
            && !prev.is_prefix_stable_against(&layout)
        {
            let event = CacheLayoutEvent::new(prev, &layout);
            self.events
                .lock()
                .expect("events mutex poisoned")
                .push(event);
        }
        *previous = Some(layout);
        Ok(())
    }
}

// ── UsageAccountingMiddleware ─────────────────────────────────────────────────

impl UsageAccountingMiddleware {
    /// Creates a usage-accounting middleware with the default label
    /// `"usage_accounting"`.
    pub fn new() -> Self {
        Self {
            label: "usage_accounting",
            totals: std::sync::Mutex::new(UsageTotals::new()),
        }
    }

    /// Returns a snapshot of the accumulated usage totals.
    pub fn totals(&self) -> UsageTotals {
        *self.totals.lock().expect("totals mutex poisoned")
    }
}

impl Default for UsageAccountingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for UsageAccountingMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        if let Some(usage) = response.usage {
            self.totals
                .lock()
                .expect("totals mutex poisoned")
                .record(usage);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test;
