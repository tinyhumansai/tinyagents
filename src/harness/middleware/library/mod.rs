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

// ── BudgetMiddleware ──────────────────────────────────────────────────────────

impl BudgetTracker {
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a snapshot of the accumulated spend.
    pub fn snapshot(&self) -> BudgetSpend {
        self.inner.lock().map(|g| *g).unwrap_or_default()
    }

    /// Folds a model call's usage and estimated cost into the tracker.
    pub fn record(
        &self,
        usage: crate::harness::usage::Usage,
        cost: crate::harness::cost::CostTotals,
    ) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.usage += usage;
            guard.cost += cost;
        }
    }
}

impl BudgetLimits {
    /// Returns a human-readable reason when `spend` meets or exceeds any limit.
    fn exceeded_reason(&self, spend: &BudgetSpend) -> Option<String> {
        let u = &spend.usage.usage;
        if let Some(max) = self.max_input_tokens
            && u.input_tokens >= max
        {
            return Some(format!("input tokens {} >= budget {max}", u.input_tokens));
        }
        if let Some(max) = self.max_output_tokens
            && u.output_tokens >= max
        {
            return Some(format!("output tokens {} >= budget {max}", u.output_tokens));
        }
        if let Some(max) = self.max_total_tokens
            && u.effective_total() >= max
        {
            return Some(format!(
                "total tokens {} >= budget {max}",
                u.effective_total()
            ));
        }
        if let Some(max) = self.max_reasoning_tokens
            && u.reasoning_tokens >= max
        {
            return Some(format!(
                "reasoning tokens {} >= budget {max}",
                u.reasoning_tokens
            ));
        }
        if let Some(max) = self.max_cost
            && spend.cost.total_cost >= max
        {
            return Some(format!(
                "cost {:.6} >= budget {max:.6}",
                spend.cost.total_cost
            ));
        }
        None
    }

    /// Returns a reason when `spend` crosses `warn_fraction` of any set limit.
    fn warn_reason(&self, spend: &BudgetSpend) -> Option<String> {
        let frac = self.warn_fraction?;
        let u = &spend.usage.usage;
        let over = |value: f64, max: Option<u64>| -> bool {
            max.is_some_and(|m| m > 0 && value >= frac * m as f64)
        };
        if over(u.input_tokens as f64, self.max_input_tokens) {
            return Some("input token budget".to_string());
        }
        if over(u.output_tokens as f64, self.max_output_tokens) {
            return Some("output token budget".to_string());
        }
        if over(u.effective_total() as f64, self.max_total_tokens) {
            return Some("total token budget".to_string());
        }
        if over(u.reasoning_tokens as f64, self.max_reasoning_tokens) {
            return Some("reasoning token budget".to_string());
        }
        if let Some(max) = self.max_cost
            && max > 0.0
            && spend.cost.total_cost >= frac * max
        {
            return Some("cost budget".to_string());
        }
        None
    }
}

impl BudgetMiddleware {
    /// Creates a budget middleware with its own fresh tracker and no pricing
    /// table (token budgets only; cost stays zero until pricing is supplied).
    pub fn new(limits: BudgetLimits) -> Self {
        Self {
            label: "budget",
            limits,
            tracker: BudgetTracker::new(),
            pricing: std::collections::HashMap::new(),
        }
    }

    /// Shares an existing [`BudgetTracker`] so this middleware's spend rolls up
    /// into a run-tree-wide budget (hand the same tracker to sub-agents).
    pub fn with_tracker(mut self, tracker: BudgetTracker) -> Self {
        self.tracker = tracker;
        self
    }

    /// Supplies a per-model-name [`ModelPricing`] table so `after_model` can
    /// price usage and enforce the money budget.
    pub fn with_pricing(
        mut self,
        pricing: std::collections::HashMap<String, crate::registry::catalog::ModelPricing>,
    ) -> Self {
        self.pricing = pricing;
        self
    }

    /// Returns the shared tracker (for reading accumulated spend).
    pub fn tracker(&self) -> BudgetTracker {
        self.tracker.clone()
    }

    fn price(&self, response: &ModelResponse) -> crate::harness::cost::CostTotals {
        let Some(usage) = response.usage else {
            return crate::harness::cost::CostTotals::default();
        };
        let Some(name) = response.resolved_model.as_ref().map(|r| r.name.as_str()) else {
            return crate::harness::cost::CostTotals::default();
        };
        match self.pricing.get(name) {
            Some(pricing) => crate::harness::cost::estimate_cost(pricing, &usage),
            None => crate::harness::cost::CostTotals::default(),
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for BudgetMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        _request: &mut ModelRequest,
    ) -> Result<()> {
        let spend = self.tracker.snapshot();
        if let Some(reason) = self.limits.exceeded_reason(&spend) {
            ctx.emit(AgentEvent::BudgetExceeded {
                reason: reason.clone(),
                blocked: true,
            });
            return Err(TinyAgentsError::LimitExceeded(format!(
                "budget exhausted: {reason}"
            )));
        }
        Ok(())
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        let Some(usage) = response.usage else {
            return Ok(());
        };
        let cost = self.price(response);
        self.tracker.record(usage, cost);

        ctx.emit(AgentEvent::UsageRecorded { usage });
        if cost.total_cost > 0.0 {
            ctx.emit(AgentEvent::CostRecorded { cost });
        }

        let mut spend = self.tracker.snapshot();
        // Warn-once on threshold crossing.
        if !spend.warned
            && let Some(reason) = self.limits.warn_reason(&spend)
        {
            ctx.emit(AgentEvent::BudgetWarning {
                reason: format!("approaching {reason}"),
            });
            if let Ok(mut guard) = self.tracker.inner.lock() {
                guard.warned = true;
            }
            spend.warned = true;
        }
        if let Some(reason) = self.limits.exceeded_reason(&spend) {
            ctx.emit(AgentEvent::BudgetExceeded {
                reason,
                blocked: false,
            });
        }
        Ok(())
    }
}

// ── ToolPolicyMiddleware ──────────────────────────────────────────────────────

impl ToolPolicyMiddleware {
    /// Creates a policy middleware from a name→policy snapshot (typically
    /// [`ToolRegistry::policies`][crate::harness::tool::ToolRegistry::policies]).
    ///
    /// Defaults are permissive: nothing is required or denied until configured.
    /// Use [`strict`](Self::strict) for a fail-closed baseline.
    pub fn new(
        policies: std::collections::HashMap<String, crate::harness::tool::ToolPolicy>,
    ) -> Self {
        Self {
            label: "tool_policy",
            policies,
            require_classification: false,
            require_background_safe: false,
            deny: crate::harness::tool::ToolSideEffects::default(),
            require_sandbox: false,
            require_approval: false,
            approved: std::collections::HashSet::new(),
            enforce_result_bytes: false,
        }
    }

    /// Creates a fail-closed policy middleware: unclassified tools are rejected,
    /// and tools declaring `destructive` or `payment` side effects are denied.
    pub fn strict(
        policies: std::collections::HashMap<String, crate::harness::tool::ToolPolicy>,
    ) -> Self {
        Self {
            label: "tool_policy",
            policies,
            require_classification: true,
            require_background_safe: false,
            deny: crate::harness::tool::ToolSideEffects {
                destructive: true,
                payment: true,
                ..crate::harness::tool::ToolSideEffects::default()
            },
            require_sandbox: false,
            require_approval: false,
            approved: std::collections::HashSet::new(),
            enforce_result_bytes: false,
        }
    }

    /// Requires every tool to carry a classified policy (fail closed on
    /// unclassified or unknown tools).
    pub fn require_classification(mut self, require: bool) -> Self {
        self.require_classification = require;
        self
    }

    /// Requires every exposed/executed tool to be `background_safe`.
    pub fn require_background_safe(mut self, require: bool) -> Self {
        self.require_background_safe = require;
        self
    }

    /// Denies tools declaring any side effect present in `mask`.
    pub fn deny_side_effects(mut self, mask: crate::harness::tool::ToolSideEffects) -> Self {
        self.deny = mask;
        self
    }

    /// Enforces that a tool declaring
    /// [`SandboxMode::Required`][crate::harness::tool::SandboxMode::Required]
    /// only runs when the run carries a sandboxed workspace (fail closed
    /// otherwise). See [`RunContext::with_workspace`][crate::harness::context::RunContext::with_workspace].
    pub fn require_sandbox(mut self, require: bool) -> Self {
        self.require_sandbox = require;
        self
    }

    /// Blocks any tool declaring `approval_required` unless its name is in
    /// `approved`, turning the declarative approval flag into a fail-closed gate.
    pub fn require_approval(
        mut self,
        approved: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.require_approval = true;
        self.approved = approved.into_iter().map(Into::into).collect();
        self
    }

    /// Enforces each tool's declared `max_result_bytes` cap by truncating and
    /// flagging oversized results in `after_tool`.
    pub fn enforce_result_bytes(mut self, enforce: bool) -> Self {
        self.enforce_result_bytes = enforce;
        self
    }

    /// Returns `Ok(())` if the named tool is permitted, otherwise an explanation
    /// of why it is blocked. Used by both the exposure and execution hooks so a
    /// hidden tool cannot be executed by a divergent decision.
    fn evaluate(&self, name: &str) -> std::result::Result<(), String> {
        let Some(policy) = self.policies.get(name) else {
            if self.require_classification {
                return Err(format!("tool `{name}` has no declared policy"));
            }
            return Ok(());
        };
        if self.require_classification && !policy.classified {
            return Err(format!("tool `{name}` is unclassified"));
        }
        let s = &policy.side_effects;
        let d = &self.deny;
        let denied = (d.writes_files && s.writes_files)
            || (d.network && s.network)
            || (d.installs_dependencies && s.installs_dependencies)
            || (d.destructive && s.destructive)
            || (d.external_service && s.external_service)
            || (d.payment && s.payment);
        if denied {
            return Err(format!("tool `{name}` declares a denied side effect"));
        }
        if self.require_background_safe && !policy.access.background_safe {
            return Err(format!("tool `{name}` is not background-safe"));
        }
        if self.require_approval && policy.access.approval_required && !self.approved.contains(name)
        {
            return Err(format!(
                "tool `{name}` requires approval that was not granted"
            ));
        }
        Ok(())
    }

    /// The context-aware slice of policy enforcement: the sandbox requirement
    /// depends on the run's workspace, which `evaluate` (name-only) cannot see.
    fn evaluate_sandbox<Ctx>(
        &self,
        name: &str,
        ctx: &RunContext<Ctx>,
    ) -> std::result::Result<(), String> {
        if !self.require_sandbox {
            return Ok(());
        }
        let Some(policy) = self.policies.get(name) else {
            return Ok(());
        };
        if policy.runtime.sandbox != crate::harness::tool::SandboxMode::Required {
            return Ok(());
        }
        let sandboxed = ctx
            .workspace
            .as_ref()
            .is_some_and(|ws| ws.sandbox == crate::harness::tool::SandboxMode::Required);
        if sandboxed {
            Ok(())
        } else {
            Err(format!(
                "tool `{name}` requires a sandbox but the run has none"
            ))
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ToolPolicyMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.tools.retain(|schema| {
            self.evaluate(&schema.name).is_ok() && self.evaluate_sandbox(&schema.name, ctx).is_ok()
        });
        Ok(())
    }

    async fn before_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        self.evaluate(&call.name)
            .and_then(|_| self.evaluate_sandbox(&call.name, ctx))
            .map_err(TinyAgentsError::Validation)
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        result: &mut ToolResult,
    ) -> Result<()> {
        if !self.enforce_result_bytes {
            return Ok(());
        }
        if let Some(policy) = self.policies.get(&result.name)
            && let Some(limit) = policy.runtime.max_result_bytes
            && result.content.len() > limit
        {
            // Truncate on a char boundary at or below the byte limit so the
            // enforced payload is still valid UTF-8.
            let mut end = limit;
            while end > 0 && !result.content.is_char_boundary(end) {
                end -= 1;
            }
            result.content.truncate(end);
            let note = format!("tool result exceeded max_result_bytes ({limit}); truncated");
            result.error = Some(match result.error.take() {
                Some(existing) => format!("{existing}; {note}"),
                None => note,
            });
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

// ── ContextualToolSelectionMiddleware ─────────────────────────────────────────

impl ContextualToolSelectionMiddleware {
    /// Creates a selection middleware from a context-aware predicate.
    pub fn new(predicate: ContextualToolPredicate) -> Self {
        Self {
            label: "contextual_tool_selection",
            predicate,
        }
    }

    /// Builds a selection middleware from explicit allow/deny lists.
    ///
    /// Composition rules (fail-closed):
    /// - a tool named in `deny` is always hidden;
    /// - when `allow` is `Some`, a tool must be named in it to be exposed
    ///   (unknown tools are hidden);
    /// - when `allow` is `None`, everything not denied is exposed.
    pub fn from_lists(
        allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        deny: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let allow: Option<HashSet<String>> =
            allow.map(|names| names.into_iter().map(Into::into).collect());
        let deny: HashSet<String> = deny.into_iter().map(Into::into).collect();
        Self::from_resolved_lists(allow, deny)
    }

    /// Builds a selection middleware whose effective policy is a child allow/deny
    /// pair *composed with* an inherited parent policy, so a delegated sub-agent
    /// can only ever narrow — never widen — the tools its parent allowed.
    ///
    /// Inheritance rules:
    /// - **deny is additive**: the effective denylist is `parent_deny ∪ child_deny`
    ///   (a child cannot un-deny what the parent denied);
    /// - **allow is intersective**: if both parent and child restrict to an
    ///   allowlist, the effective allowlist is their intersection; if only one
    ///   restricts, that allowlist applies; if neither does, all-not-denied is
    ///   exposed.
    ///
    /// The result is fail-closed for the same reasons as [`Self::from_lists`].
    pub fn inheriting(
        parent_allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        parent_deny: impl IntoIterator<Item = impl Into<String>>,
        child_allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        child_deny: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let parent_allow: Option<HashSet<String>> =
            parent_allow.map(|n| n.into_iter().map(Into::into).collect());
        let child_allow: Option<HashSet<String>> =
            child_allow.map(|n| n.into_iter().map(Into::into).collect());
        let allow = match (parent_allow, child_allow) {
            (Some(p), Some(c)) => Some(p.intersection(&c).cloned().collect()),
            (Some(p), None) => Some(p),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };
        let mut deny: HashSet<String> = parent_deny.into_iter().map(Into::into).collect();
        deny.extend(child_deny.into_iter().map(Into::into));
        Self::from_resolved_lists(allow, deny)
    }

    fn from_resolved_lists(allow: Option<HashSet<String>>, deny: HashSet<String>) -> Self {
        Self::new(Arc::new(move |schema: &ToolSchema, _ctx| {
            if deny.contains(&schema.name) {
                return false;
            }
            match &allow {
                Some(set) => set.contains(&schema.name),
                None => true,
            }
        }))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for ContextualToolSelectionMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        let selection = ToolSelectionContext {
            run_id: ctx.config.run_id.as_str().to_string(),
            depth: ctx.config.depth,
            tags: ctx.config.tags.clone(),
            requested_model: request.model.clone(),
        };
        let mut excluded = Vec::new();
        request.tools.retain(|schema| {
            let keep = (self.predicate)(schema, &selection);
            if !keep {
                excluded.push(schema.name.clone());
            }
            keep
        });
        // Make the exposure decision auditable when it actually withheld tools.
        if !excluded.is_empty() {
            ctx.emit(AgentEvent::ToolsFiltered {
                by: self.label.to_string(),
                excluded,
                remaining: request.tools.len(),
            });
        }
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
