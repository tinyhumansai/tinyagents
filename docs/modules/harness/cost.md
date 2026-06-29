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
