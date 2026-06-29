//! Built-in middleware library.
//!
//! This module collects the ready-to-use middleware that ship with the harness.
//! They are split across two extension surfaces from
//! [`crate::harness::middleware`]:
//!
//! - **Resilience (wrap)** — [`RetryMiddleware`], [`TimeoutMiddleware`],
//!   [`ModelFallbackMiddleware`], and [`RateLimitMiddleware`] implement the
//!   around-call [`ModelMiddleware`] trait and surround the real model call.
//! - **Policy / guard / observation (lifecycle)** —
//!   [`ToolAllowlistMiddleware`], [`DynamicToolSelectionMiddleware`],
//!   [`HumanApprovalMiddleware`], [`StructuredOutputValidatorMiddleware`],
//!   [`DynamicPromptMiddleware`], [`RedactionMiddleware`], and
//!   [`TracingMiddleware`] implement the lifecycle [`Middleware`] trait.
//!
//! Type definitions live in [`types`]; this file holds the constructors and
//! trait impls. Tests live in `test.rs`.
//!
//! # Testability
//!
//! None of these middleware sleep on the wall clock in a way that tests cannot
//! control: [`RetryMiddleware`] computes but never sleeps on backoff,
//! [`TimeoutMiddleware`] is exercised under `tokio::time` paused-time tests, and
//! [`RateLimitMiddleware`] takes an injectable clock and a configurable poll
//! interval so its wait loop can be driven deterministically.

mod types;

pub use types::*;

use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::error::{Result, TinyAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::AgentEvent;
use crate::harness::ids::CallId;
use crate::harness::message::{ContentBlock, Message};
use crate::harness::middleware::{
    Middleware, MiddlewareModelOutcome, ModelHandler, ModelMiddleware,
};
use crate::harness::model::{ModelDelta, ModelRequest, ModelResponse, ResponseFormat};
use crate::harness::retry::{RateLimiter, RetryPolicy, is_retryable};
use crate::harness::structured::{StructuredExtractor, StructuredStrategy};
use crate::harness::tool::{ToolCall, ToolDelta, ToolResult, ToolSchema};

// ── RetryMiddleware ───────────────────────────────────────────────────────────

impl RetryMiddleware {
    /// Creates a retry middleware using the given [`RetryPolicy`].
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            label: "retry",
            policy,
        }
    }

    /// Creates a retry middleware with the default [`RetryPolicy`].
    pub fn with_default_policy() -> Self {
        Self::new(RetryPolicy::default())
    }

    /// Returns the policy-derived backoff for the given retry `attempt`.
    ///
    /// Exposed for callers that want to sleep before a retry; the middleware
    /// itself never sleeps.
    pub fn backoff_for_attempt(&self, attempt: usize) -> Duration {
        self.policy.backoff_for_attempt(attempt)
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for RetryMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut attempt = 0usize;
        loop {
            match next.run(ctx, state, request.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(error) => {
                    if is_retryable(&error) && self.policy.should_retry(attempt) {
                        attempt += 1;
                        let call_id = CallId::new(format!("{}-model", ctx.run_id()));
                        ctx.emit(AgentEvent::RetryScheduled { call_id, attempt });
                        // Compute (but do not sleep on) the backoff.
                        let _ = self.policy.backoff_for_attempt(attempt);
                        continue;
                    }
                    return Err(error);
                }
            }
        }
    }
}

// ── TimeoutMiddleware ─────────────────────────────────────────────────────────

impl TimeoutMiddleware {
    /// Creates a timeout middleware bounding each model call to `timeout`.
    pub fn new(timeout: Duration) -> Self {
        Self {
            label: "timeout",
            timeout,
        }
    }

    /// Creates a timeout middleware from a millisecond duration.
    pub fn from_millis(ms: u64) -> Self {
        Self::new(Duration::from_millis(ms))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for TimeoutMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        let run_id = ctx.run_id().as_str().to_string();
        let fut = next.run(ctx, state, request);
        match tokio::time::timeout(self.timeout, fut).await {
            Ok(result) => result,
            Err(_) => Err(TinyAgentsError::Timeout(format!(
                "model call for run `{run_id}` exceeded the {} ms middleware timeout",
                self.timeout.as_millis()
            ))),
        }
    }
}

// ── ModelFallbackMiddleware ───────────────────────────────────────────────────

impl ModelFallbackMiddleware {
    /// Creates a fallback middleware that tries each model name in order after
    /// the primary call fails.
    pub fn new(fallbacks: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "model_fallback",
            fallbacks: fallbacks.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for ModelFallbackMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        match next.run(ctx, state, request.clone()).await {
            Ok(outcome) => Ok(outcome),
            Err(mut last_error) => {
                let mut current = request.model.clone().unwrap_or_default();
                for fallback in &self.fallbacks {
                    ctx.emit(AgentEvent::FallbackSelected {
                        from: current.clone(),
                        to: fallback.clone(),
                    });
                    let mut req = request.clone();
                    req.model = Some(fallback.clone());
                    match next.run(ctx, state, req).await {
                        Ok(outcome) => return Ok(outcome),
                        Err(error) => {
                            last_error = error;
                            current = fallback.clone();
                        }
                    }
                }
                Err(last_error)
            }
        }
    }
}

// ── RateLimitMiddleware ───────────────────────────────────────────────────────

impl RateLimitMiddleware {
    /// Creates a rate-limit middleware gating one token per call through
    /// `limiter`, failing immediately when the bucket is empty
    /// ([`RateLimitBehavior::Error`]).
    pub fn new(limiter: Arc<RateLimiter>) -> Self {
        Self {
            label: "rate_limit",
            limiter,
            tokens: 1,
            behavior: RateLimitBehavior::Error,
            poll_interval: Duration::from_millis(50),
            now: Arc::new(Instant::now),
        }
    }

    /// Sets the number of tokens each call consumes.
    pub fn with_tokens(mut self, tokens: u64) -> Self {
        self.tokens = tokens;
        self
    }

    /// Sets the behavior when the bucket lacks capacity.
    pub fn with_behavior(mut self, behavior: RateLimitBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Switches to [`RateLimitBehavior::Wait`] with the given poll interval.
    pub fn waiting(mut self, poll_interval: Duration) -> Self {
        self.behavior = RateLimitBehavior::Wait;
        self.poll_interval = poll_interval;
        self
    }

    /// Replaces the clock used to read the current instant (for deterministic
    /// tests).
    pub fn with_clock(mut self, now: NowFn) -> Self {
        self.now = now;
        self
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ModelMiddleware<State, Ctx> for RateLimitMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome> {
        loop {
            let now = (self.now)();
            if self.limiter.try_acquire(self.tokens, now) {
                break;
            }
            match self.behavior {
                RateLimitBehavior::Error => {
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "rate limit: could not acquire {} token(s)",
                        self.tokens
                    )));
                }
                RateLimitBehavior::Wait => {
                    ctx.emit(AgentEvent::RateLimitWaited {
                        waited_ms: self.poll_interval.as_millis() as u64,
                    });
                    tokio::time::sleep(self.poll_interval).await;
                }
            }
        }
        next.run(ctx, state, request).await
    }
}

// ── ToolAllowlistMiddleware ───────────────────────────────────────────────────

impl ToolAllowlistMiddleware {
    /// Creates an allowlist middleware permitting only the named tools.
    pub fn new(allowed: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "tool_allowlist",
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns `true` if `name` is on the allowlist.
    pub fn allows(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ToolAllowlistMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        if !self.allowed.contains(&call.name) {
            return Err(TinyAgentsError::Validation(format!(
                "tool `{}` is not on the allowlist",
                call.name
            )));
        }
        Ok(())
    }
}

// ── DynamicToolSelectionMiddleware ────────────────────────────────────────────

impl DynamicToolSelectionMiddleware {
    /// Creates a selection middleware exposing only tools for which `predicate`
    /// returns `true`.
    pub fn new(predicate: ToolPredicate) -> Self {
        Self {
            label: "dynamic_tool_selection",
            predicate,
        }
    }

    /// Creates a selection middleware exposing only the named tools.
    pub fn allowing(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let allowed: HashSet<String> = names.into_iter().map(Into::into).collect();
        Self::new(Arc::new(move |schema: &ToolSchema| {
            allowed.contains(&schema.name)
        }))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for DynamicToolSelectionMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.tools.retain(|schema| (self.predicate)(schema));
        Ok(())
    }
}

// ── HumanApprovalMiddleware ───────────────────────────────────────────────────

impl HumanApprovalMiddleware {
    /// Creates an approval middleware that interrupts when any flagged tool is
    /// called and no approval callback is configured.
    pub fn new(flagged: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "human_approval",
            flagged: flagged.into_iter().map(Into::into).collect(),
            approve: None,
        }
    }

    /// Attaches an approval callback consulted for flagged tools. Returning
    /// `true` admits the call; `false` (or no callback) raises an interrupt.
    pub fn with_approval(mut self, approve: ApprovalFn) -> Self {
        self.approve = Some(approve);
        self
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for HumanApprovalMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        if self.flagged.contains(&call.name) {
            let approved = self
                .approve
                .as_ref()
                .map(|approve| approve(call))
                .unwrap_or(false);
            if !approved {
                return Err(TinyAgentsError::Interrupted {
                    node: "tool".to_string(),
                    message: format!("tool `{}` requires human approval", call.name),
                });
            }
        }
        Ok(())
    }
}

// ── StructuredOutputValidatorMiddleware ───────────────────────────────────────

impl StructuredOutputValidatorMiddleware {
    /// Creates a validator middleware checking responses against `format`.
    pub fn new(format: ResponseFormat) -> Self {
        Self {
            label: "structured_output_validator",
            format,
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for StructuredOutputValidatorMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        match &self.format {
            ResponseFormat::Text => Ok(()),
            ResponseFormat::JsonObject => {
                let text = response.text();
                serde_json::from_str::<serde_json::Value>(&text).map_err(|e| {
                    TinyAgentsError::StructuredOutput(format!(
                        "response text is not valid JSON: {e}"
                    ))
                })?;
                Ok(())
            }
            ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
                let extractor = StructuredExtractor::new(
                    StructuredStrategy::ProviderSchema,
                    name.clone(),
                    schema.clone(),
                );
                extractor.extract(response)?;
                Ok(())
            }
        }
    }
}

// ── DynamicPromptMiddleware ───────────────────────────────────────────────────

impl<State, Ctx> DynamicPromptMiddleware<State, Ctx> {
    /// Creates a dynamic-prompt middleware deriving a system message from
    /// `prompt`.
    pub fn new(prompt: PromptFn<State>) -> Self {
        Self {
            label: "dynamic_prompt",
            prompt,
            _marker: PhantomData,
        }
    }

    /// Creates a dynamic-prompt middleware from a closure over the shared state
    /// and the run's [`RunConfig`].
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(&State, &RunConfig) -> Option<String> + Send + Sync + 'static,
    {
        Self::new(Arc::new(f))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for DynamicPromptMiddleware<State, Ctx>
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        if let Some(text) = (self.prompt)(state, &ctx.config) {
            request.messages.insert(0, Message::system(text));
        }
        Ok(())
    }
}

// ── RedactionMiddleware ───────────────────────────────────────────────────────

impl RedactionMiddleware {
    /// Creates a redaction middleware replacing each pattern with `"[REDACTED]"`.
    pub fn new(patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::with_mask(patterns, "[REDACTED]")
    }

    /// Creates a redaction middleware replacing each pattern with `mask`.
    pub fn with_mask(
        patterns: impl IntoIterator<Item = impl Into<String>>,
        mask: impl Into<String>,
    ) -> Self {
        Self {
            label: "redaction",
            patterns: patterns
                .into_iter()
                .map(Into::into)
                .filter(|p| !p.is_empty())
                .collect(),
            mask: mask.into(),
            redactions: Mutex::new(0),
        }
    }

    /// Returns the total number of pattern occurrences redacted so far.
    pub fn redactions(&self) -> usize {
        *self.redactions.lock().expect("redactions mutex poisoned")
    }

    /// Replaces every configured pattern in `text`, returning the redacted
    /// string and the number of occurrences replaced.
    fn redact(&self, text: &str) -> (String, usize) {
        let mut out = text.to_string();
        let mut hits = 0usize;
        for pattern in &self.patterns {
            let occurrences = out.matches(pattern.as_str()).count();
            if occurrences > 0 {
                hits += occurrences;
                out = out.replace(pattern.as_str(), &self.mask);
            }
        }
        (out, hits)
    }

    /// Records `hits` redactions against the running total.
    fn record(&self, hits: usize) {
        if hits > 0 {
            *self.redactions.lock().expect("redactions mutex poisoned") += hits;
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for RedactionMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        let mut hits = 0usize;
        for block in &mut response.message.content {
            if let ContentBlock::Text(text) = block {
                let (redacted, n) = self.redact(text);
                if n > 0 {
                    *text = redacted;
                    hits += n;
                }
            }
        }
        self.record(hits);
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        result: &mut ToolResult,
    ) -> Result<()> {
        let (redacted, hits) = self.redact(&result.content);
        if hits > 0 {
            result.content = redacted;
        }
        self.record(hits);
        Ok(())
    }
}

// ── TracingMiddleware ─────────────────────────────────────────────────────────

impl TracingMiddleware {
    /// Creates a tracing middleware with the default label `"tracing"`.
    pub fn new() -> Self {
        Self::with_label("tracing")
    }

    /// Creates a tracing middleware with a custom static label.
    pub fn with_label(label: &'static str) -> Self {
        Self {
            label,
            records: Mutex::new(Vec::new()),
            counts: Mutex::new(TraceCounts::default()),
        }
    }

    /// Returns the structured begin/end traces recorded so far, in order.
    pub fn records(&self) -> Vec<PhaseTrace> {
        self.records.lock().expect("records mutex poisoned").clone()
    }

    /// Returns a snapshot of the per-phase begin counts.
    pub fn counts(&self) -> TraceCounts {
        self.counts.lock().expect("counts mutex poisoned").clone()
    }

    fn push(&self, phase: &'static str, boundary: TraceBoundary) {
        self.records
            .lock()
            .expect("records mutex poisoned")
            .push(PhaseTrace { phase, boundary });
    }
}

impl Default for TracingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for TracingMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_agent(&self, _ctx: &mut RunContext<Ctx>, _state: &State) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").agent += 1;
        self.push("agent", TraceBoundary::Begin);
        Ok(())
    }

    async fn after_agent(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _run: &mut crate::harness::middleware::AgentRun,
    ) -> Result<()> {
        self.push("agent", TraceBoundary::End);
        Ok(())
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").model += 1;
        self.push("model", TraceBoundary::Begin);
        Ok(())
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ModelDelta,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").delta += 1;
        Ok(())
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _response: &mut ModelResponse,
    ) -> Result<()> {
        self.push("model", TraceBoundary::End);
        Ok(())
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _call: &mut ToolCall,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").tool += 1;
        self.push("tool", TraceBoundary::Begin);
        Ok(())
    }

    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ToolDelta,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").delta += 1;
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _result: &mut ToolResult,
    ) -> Result<()> {
        self.push("tool", TraceBoundary::End);
        Ok(())
    }

    async fn on_error(&self, _ctx: &mut RunContext<Ctx>, _error: &TinyAgentsError) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").error += 1;
        Ok(())
    }
}

#[cfg(test)]
mod test;
