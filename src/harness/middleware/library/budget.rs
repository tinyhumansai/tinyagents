//! Budget middleware: token/cost preflight reservation and enforcement
//! (`BudgetTracker`, `BudgetLimits`, `BudgetMiddleware`).
//!
//! Split out of `library/mod.rs`; see that module's doc comment for the
//! full built-in middleware library overview.

use super::*;

// ── BudgetMiddleware ──────────────────────────────────────────────────────────

impl BudgetTracker {
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Locks the inner spend, recovering the last-known state if the mutex
    /// was poisoned by a panicking holder.
    ///
    /// A poisoned mutex still holds a valid (if possibly stale) last-written
    /// value; treating poisoning as "spend unknown, default to zero" would
    /// make the budget enforcer fail *open* (every subsequent call sees an
    /// empty budget and is admitted). Recovering the poisoned guard keeps
    /// enforcement fail-closed: accumulated spend is never lost.
    fn lock_recovering(&self) -> std::sync::MutexGuard<'_, BudgetSpend> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Returns a snapshot of the accumulated spend.
    pub fn snapshot(&self) -> BudgetSpend {
        *self.lock_recovering()
    }

    /// Folds a model call's usage and estimated cost into the tracker.
    pub fn record(
        &self,
        usage: crate::harness::usage::Usage,
        cost: crate::harness::cost::CostTotals,
    ) {
        let mut guard = self.lock_recovering();
        guard.usage += usage;
        guard.cost += cost;
    }
}

impl BudgetLimits {
    /// Returns a human-readable reason when `spend` meets or exceeds any limit.
    pub(super) fn exceeded_reason(&self, spend: &BudgetSpend) -> Option<String> {
        let u = &spend.usage.usage;
        if let Some(max) = self.max_input_tokens
            && u.input_tokens >= max
        {
            return Some(format!("input tokens {} >= budget {max}", u.input_tokens));
        }
        if let Some(max) = self.max_cached_input_tokens
            && u.cache_read_tokens >= max
        {
            return Some(format!(
                "cached input tokens {} >= budget {max}",
                u.cache_read_tokens
            ));
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
        if over(u.cache_read_tokens as f64, self.max_cached_input_tokens) {
            return Some("cached input token budget".to_string());
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

/// Estimates the input tokens a request will consume by summing a
/// heuristic token estimate over every message's text. Used for budget
/// preflight reservation, which only needs an order-of-magnitude bound.
fn estimated_input_tokens(request: &ModelRequest) -> u64 {
    request
        .messages
        .iter()
        .map(|m| crate::harness::summarization::estimate_tokens(&m.text()))
        .sum()
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
            pending_reservation: std::sync::Mutex::new(0),
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
        request: &mut ModelRequest,
    ) -> Result<()> {
        // Preflight is check-and-reserve under a single lock acquisition so
        // concurrent runs sharing this tracker cannot all observe capacity
        // and reserve past it (a separate check-then-lock-then-write window
        // lets N concurrent callers each pass the check before any of them
        // records a reservation, collectively overshooting the budget).
        let estimated = estimated_input_tokens(request);
        {
            let mut guard = self.tracker.lock_recovering();
            // (1) Already exhausted before this call.
            if let Some(reason) = self.limits.exceeded_reason(&guard) {
                drop(guard);
                ctx.emit(AgentEvent::BudgetExceeded {
                    reason: reason.clone(),
                    blocked: true,
                });
                return Err(TinyAgentsError::LimitExceeded(format!(
                    "budget exhausted: {reason}"
                )));
            }

            // (2) Preflight reservation: estimate this call's input tokens
            // (plus every other in-flight reservation on this shared
            // tracker) and block *before* dispatching if it would breach the
            // input budget, so a single large call — or several concurrent
            // ones — cannot collectively overshoot it.
            if let Some(max) = self.limits.max_input_tokens
                && guard.usage.usage.input_tokens + guard.reserved_input_total + estimated > max
            {
                let reason = format!(
                    "reserved input tokens {} + {estimated} > budget {max}",
                    guard.usage.usage.input_tokens + guard.reserved_input_total
                );
                drop(guard);
                ctx.emit(AgentEvent::BudgetExceeded {
                    reason: reason.clone(),
                    blocked: true,
                });
                return Err(TinyAgentsError::LimitExceeded(format!(
                    "budget reservation exceeded: {reason}"
                )));
            }
            guard.reserved_input_total += estimated;
        }

        // Remember this run's own outstanding reservation for reconciliation
        // in `after_model` (local to this middleware instance, so concurrent
        // runs sharing the tracker never clobber each other's amount).
        *self
            .pending_reservation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = estimated;

        ctx.emit(AgentEvent::BudgetReserved {
            estimated_input_tokens: estimated,
        });
        Ok(())
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        // Release this run's outstanding reservation regardless of whether
        // usage came back, so a call that fails to report usage (or errors
        // out before this hook) never leaks a permanent reservation that
        // starves later calls on a shared tracker.
        let reserved = std::mem::take(
            &mut *self
                .pending_reservation
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        {
            let mut guard = self.tracker.lock_recovering();
            guard.reserved_input_total = guard.reserved_input_total.saturating_sub(reserved);
        }

        let Some(usage) = response.usage else {
            return Ok(());
        };
        let cost = self.price(response);
        ctx.emit(AgentEvent::BudgetReconciled {
            estimated_input_tokens: reserved,
            actual_input_tokens: usage.input_tokens,
        });
        self.tracker.record(usage, cost);

        ctx.emit(AgentEvent::UsageRecorded { usage });
        if cost.total_cost > 0.0 {
            ctx.emit(AgentEvent::CostRecorded { cost });
        }

        // Warn-once on threshold crossing: check-and-set the `warned` flag
        // under a single lock so two concurrent calls crossing the
        // threshold at once can't both observe `warned == false` and both
        // emit the warning.
        let (spend, warning) = {
            let mut guard = self.tracker.lock_recovering();
            let warning = if !guard.warned {
                let reason = self.limits.warn_reason(&guard);
                if reason.is_some() {
                    guard.warned = true;
                }
                reason
            } else {
                None
            };
            (*guard, warning)
        };
        if let Some(reason) = warning {
            ctx.emit(AgentEvent::BudgetWarning {
                reason: format!("approaching {reason}"),
            });
        }
        if let Some(reason) = self.limits.exceeded_reason(&spend) {
            ctx.emit(AgentEvent::BudgetExceeded {
                reason,
                blocked: false,
            });
        }
        Ok(())
    }

    async fn on_error(&self, _ctx: &mut RunContext<Ctx>, _error: &TinyAgentsError) -> Result<()> {
        // A model call that fails (retries/fallback exhausted, hard provider
        // error, middleware timeout, ...) short-circuits with `?` before
        // `after_model` ever runs, so the reservation `before_model` added to
        // the shared tracker would otherwise never be released. Release it
        // here so a run of failures cannot permanently inflate
        // `reserved_input_total` and starve every future call on a
        // process-lifetime-shared `BudgetTracker`.
        let reserved = std::mem::take(
            &mut *self
                .pending_reservation
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        if reserved > 0 {
            let mut guard = self.tracker.lock_recovering();
            guard.reserved_input_total = guard.reserved_input_total.saturating_sub(reserved);
        }
        Ok(())
    }
}
