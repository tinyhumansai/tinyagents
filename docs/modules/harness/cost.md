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
