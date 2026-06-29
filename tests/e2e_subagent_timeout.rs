//! TRUE end-to-end (offline, deterministic): sub-agent TIMEOUT enforcement.
//!
//! A [`SubAgent`] is built over a child harness driven by a [`SlowModel`] that
//! sleeps 200ms before replying. The child harness policy sets a 20ms
//! wall-clock cap, so the per-model-call budget interrupts the slow call and
//! [`SubAgent::invoke`] returns `Err(TinyAgentsError::Timeout(..))` — even
//! though the child [`RunConfig`] built internally by `invoke` carries no
//! per-run timeout (the loop also honors the harness policy's wall-clock cap).
//!
//! A control case shows a *fast* model under the same small cap completes
//! successfully — the timeout fires only on genuinely slow calls.

use std::sync::Arc;
use std::time::Duration;

use tinyagents::SubAgent;
use tinyagents::error::TinyAgentsError;
use tinyagents::harness::limits::RunLimits;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
use tinyagents::harness::testkit::SlowModel;

/// Builds a child harness whose policy caps wall-clock time at `timeout_ms` and
/// whose registered model is `model`.
fn capped_child_harness<M>(model: M, timeout_ms: u64) -> AgentHarness<()>
where
    M: tinyagents::harness::model::ChatModel<()> + 'static,
{
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("model", Arc::new(model));
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_wall_clock_ms(Some(timeout_ms)),
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn subagent_invoke_times_out_on_slow_model() {
    // 20ms cap against a 200ms model call: the per-call budget must fire.
    let child = capped_child_harness(SlowModel::new(Duration::from_millis(200), "eventually"), 20);
    let subagent = SubAgent::new(
        "slow_worker",
        "a worker backed by a slow model",
        Arc::new(child),
    );

    let err = subagent
        .invoke(&(), (), 0, "go")
        .await
        .expect_err("a 200ms model call under a 20ms cap must time out");

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
}

#[tokio::test]
async fn fast_subagent_invoke_succeeds_under_same_cap() {
    // Control: a fast model under the same 20ms cap completes cleanly.
    let child = capped_child_harness(MockModel::constant("quick"), 20);
    let subagent = SubAgent::new(
        "fast_worker",
        "a worker backed by a fast model",
        Arc::new(child),
    );

    let run = subagent
        .invoke(&(), (), 0, "go")
        .await
        .expect("a fast model call completes within the cap");

    assert_eq!(run.text(), Some("quick".to_string()));
}
