//! Regression coverage for CACHE-3: a cache-served response must not be billed.
//!
//! A replay makes no provider call and consumes no tokens, but the loop folded
//! its (replayed) `usage` into the run totals and the budget tracker anyway. On
//! a cache-heavy run that is phantom spend, and enough of it can abort a run
//! through `BudgetMiddleware` on money that was never spent.

use std::sync::Arc;

use async_trait::async_trait;

use tinyagents::harness::message::Message;
use tinyagents::harness::middleware::{BudgetLimits, BudgetMiddleware};
use tinyagents::harness::model::{ChatModel, ModelRequest, ModelResponse};
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::usage::Usage;

/// A model that answers with a fixed response, optionally flagged as a replay.
struct FlaggedModel {
    served_from_cache: bool,
}

#[async_trait]
impl ChatModel<()> for FlaggedModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyagents::Result<ModelResponse> {
        let mut response = ModelResponse::assistant("done").with_usage(Usage::new(100, 50));
        response.served_from_cache = self.served_from_cache;
        Ok(response)
    }
}

async fn run_usage(served_from_cache: bool) -> tinyagents::harness::usage::UsageTotals {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("m", Arc::new(FlaggedModel { served_from_cache }));
    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds")
        .usage
}

/// The control: a real provider call is billed exactly as before.
#[tokio::test]
async fn a_real_model_call_is_still_billed() {
    let usage = run_usage(false).await;
    assert_eq!(usage.usage.input_tokens, 100);
    assert_eq!(usage.usage.output_tokens, 50);
}

/// The fix: a replay is not.
#[tokio::test]
async fn a_cache_served_response_is_not_billed_to_the_run() {
    let usage = run_usage(true).await;
    assert_eq!(
        usage.usage.input_tokens, 0,
        "a cache replay consumed no provider tokens"
    );
    assert_eq!(usage.usage.output_tokens, 0);
}

/// The same rule in `BudgetMiddleware`: a cache hit must not consume budget, so
/// a budget that only fits one real call still admits any number of replays.
#[tokio::test]
async fn a_cache_served_response_does_not_consume_the_budget() {
    let budget = BudgetMiddleware::new(BudgetLimits {
        max_total_tokens: Some(120),
        ..BudgetLimits::default()
    });
    let tracker = budget.tracker();

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "m",
        Arc::new(FlaggedModel {
            served_from_cache: true,
        }),
    );
    harness.push_middleware(Arc::new(budget));

    for _ in 0..5 {
        harness
            .invoke_default(&(), vec![Message::user("go")])
            .await
            .expect("cache replays must never exhaust a budget they did not spend");
    }

    let snapshot = tracker.snapshot();
    assert_eq!(
        snapshot.usage.usage.input_tokens, 0,
        "phantom spend was recorded for cache replays: {snapshot:?}"
    );
}
