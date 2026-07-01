# Harness Cost Feature

Cost accounting prices model and cache usage so callers can enforce budgets and
render spend in web UIs.

## Responsibilities

- Store model pricing.
- Price prompt, completion, cached, and reasoning tokens.
- Track per-request fixed fees when needed.
- Enforce per-run and per-thread budgets.
- Roll costs up across sub-agents and graph nodes.
- Emit cost events.
- Keep price data updateable outside provider adapter code.
- Mark estimated versus provider-confirmed cost.
- Carry currency explicitly.

## Source Inspiration

LangChain standardizes usage on messages and traces, while LangSmith tracks
additional cost information. LangChain model profiles intentionally focus on
capability data, not prices:

- usage metadata:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/ai.py>
- usage callbacks:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/usage.py>
- model profiles:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/model_profile.py>

## Core Types

```rust
pub struct ModelPrice {
    pub model: ModelName,
    pub input_per_million: Decimal,
    pub output_per_million: Decimal,
    pub cached_input_per_million: Option<Decimal>,
    pub reasoning_per_million: Option<Decimal>,
    pub currency: Currency,
}

pub struct CostRecord {
    pub usage: UsageRecord,
    pub estimated_cost: Decimal,
    pub currency: Currency,
}
```

Pricing tables are time-sensitive. They should be updateable through config or a
store-backed table, not hardcoded permanently in provider adapters.

## Budget Enforcement

Budget policy should support:

- per-run maximum cost
- per-thread maximum cost
- per-model-call maximum cost estimate
- soft warning thresholds
- hard stop thresholds
- currency validation

The harness should estimate cost before calls when enough information exists,
then reconcile after provider usage is known. Unknown prices should either fail
closed or emit an unpriced usage record depending on policy.

## Budget Middleware

`BudgetMiddleware` (`src/harness/middleware/library/`) enforces a
`BudgetLimits` across a run — or across a whole recursive run tree, when the same
`BudgetTracker` is shared with every sub-agent harness (cloning a tracker shares
its accumulator). Every `BudgetLimits` field is optional; an unset limit is not
enforced:

```rust
pub struct BudgetLimits {
    pub max_input_tokens: Option<u64>,
    pub max_cached_input_tokens: Option<u64>, // vs Usage.cache_read_tokens
    pub max_output_tokens: Option<u64>,
    pub max_total_tokens: Option<u64>,
    pub max_reasoning_tokens: Option<u64>,
    pub max_cost: Option<f64>,
    pub warn_fraction: Option<f64>,           // e.g. 0.9
}
```

The middleware acts at two hooks:

- **`before_model` (preflight reservation).** It estimates the upcoming call's
  input tokens from the request and, if that reservation would breach the input
  budget, emits `AgentEvent::BudgetExceeded { blocked: true }` and fails the call
  with `TinyAgentsError::LimitExceeded` **before** any provider dispatch. On a
  successful reservation it emits
  `AgentEvent::BudgetReserved { estimated_input_tokens }`. If the accumulated
  spend already meets any limit, the preflight fails closed the same way.
- **`after_model` (spend + reconcile).** It folds the response `Usage` into the
  tracker, prices it via the configured per-model `ModelPricing` table (emitting
  `AgentEvent::UsageRecorded` / `AgentEvent::CostRecorded`), reconciles the prior
  reservation against the provider-reported input tokens
  (`AgentEvent::BudgetReconciled { estimated_input_tokens, actual_input_tokens }`),
  and emits `BudgetWarning` (past `warn_fraction`) or `BudgetExceeded { blocked:
  false }` as thresholds are crossed.

`max_cached_input_tokens` is enforced against `Usage.cache_read_tokens`, so a run
can bound how many cached (cache-read) input tokens it consumes independently of
fresh input tokens.

```rust
use std::sync::Arc;
use tinyagents::harness::middleware::{BudgetLimits, BudgetMiddleware, MiddlewareStack};

let mw = BudgetMiddleware::new(BudgetLimits {
    max_input_tokens: Some(5),
    ..BudgetLimits::default()
});
let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
stack.push(Arc::new(mw));

// A long prompt estimates well over 5 input tokens -> the reservation preflight
// blocks it before any call is dispatched (LimitExceeded + BudgetExceeded{blocked:true}).
let big = "word ".repeat(200);
let mut req = ModelRequest::new(vec![Message::user(big)]);
assert!(stack.run_before_model(&mut ctx, &(), &mut req).await.is_err());
```

Attach pricing with `BudgetMiddleware::new(limits).with_pricing(map)` and read
the shared accumulator with `.tracker()` (a `BudgetTracker` whose `.snapshot()`
yields a `BudgetSpend { usage, cost, warned, last_reserved_input }`).

### Emitted events

| Event | When |
| --- | --- |
| `BudgetReserved { estimated_input_tokens }` | preflight reserved an estimate for the upcoming call |
| `BudgetReconciled { estimated_input_tokens, actual_input_tokens }` | after the call, estimate reconciled against actual |
| `BudgetWarning { reason }` | cumulative spend crossed `warn_fraction` of a limit (warn-once) |
| `BudgetExceeded { reason, blocked }` | a limit was hit (`blocked: true` = preflight blocked a call; `false` = detected post-spend) |
| `UsageRecorded` / `CostRecorded` | after each priced spend |
