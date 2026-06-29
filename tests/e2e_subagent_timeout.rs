//! TRUE end-to-end (offline, deterministic): sub-agent TIMEOUT enforcement.
//!
//! A [`SubAgent`] is built over a child harness driven by a [`SlowModel`] that
//! sleeps 200ms before replying. The child run is given a 20ms wall-clock
//! timeout, so the per-model-call budget interrupts the slow call and
//! [`SubAgent::invoke`] returns `Err(TinyAgentsError::Timeout(..))`.
//!
//! A control case shows a *fast* model under the same small timeout completes
//! successfully — the timeout fires only on genuinely slow calls.

use std::sync::Arc;
use std::time::Duration;

use tinyagents::error::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::message::Message;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::SlowModel;
use tinyagents::SubAgent;

/// A child harness whose `SlowModel` sleeps `delay` before replying.
fn slow_child_harness(delay: Duration) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("slow", Arc::new(SlowModel::new(delay, "eventually")));
    harness
}

/// Runs `subagent` at depth 0 inside a child context carrying `timeout_ms`.
async fn invoke_with_timeout(
    subagent: &SubAgent<()>,
    timeout_ms: u64,
) -> tinyagents::error::Result<tinyagents::harness::middleware::AgentRun> {
    // SubAgent::invoke builds the child RunConfig internally, so drive the
    // child harness through a context we control to attach the timeout.
    let config = RunConfig::new("timeout-child").with_timeout_ms(timeout_ms);
    let ctx = RunContext::new(config, ());
    subagent
        .harness()
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
}

#[tokio::test]
async fn subagent_times_out_on_slow_model() {
    let subagent = SubAgent::new(
        "slow_worker",
        "a worker backed by a slow model",
        Arc::new(slow_child_harness(Duration::from_millis(200))),
    );

    // 20ms budget against a 200ms model call: the per-call timeout must fire.
    let err = invoke_with_timeout(&subagent, 20)
        .await
        .expect_err("a 200ms model call under a 20ms budget must time out");

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
}

#[tokio::test]
async fn fast_subagent_succeeds_under_same_timeout() {
    // Control: a fast model under the same 20ms budget completes cleanly.
    let mut child: AgentHarness<()> = AgentHarness::new();
    child.register_model("fast", Arc::new(MockModel::constant("quick")));
    let subagent = SubAgent::new("fast_worker", "a worker backed by a fast model", Arc::new(child));

    let run = invoke_with_timeout(&subagent, 20)
        .await
        .expect("a fast model call completes within the budget");

    assert_eq!(run.text(), Some("quick".to_string()));
}
