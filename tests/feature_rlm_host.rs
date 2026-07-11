//! Feature tests for the **RLM host capability boundary**: the fatal-vs-
//! script-visible error split (`is_fatal`), the live capability listing shown
//! to the driver model, and the fail-closed enforcement of the model/agent
//! recursion contracts.
//!
//! These exercise host behaviours the unit tests do not: `is_fatal`
//! classification directly, `capabilities()` over populated and empty
//! registries, an `llm` call with no default model, an unknown model surfaced
//! inside a script, and the shared sub-agent depth guard tripping fatally.

#![cfg(feature = "rlm")]

use std::sync::Arc;

use tinyagents::TinyAgentsError;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{FakeTool, ScriptedModel};
use tinyagents::registry::CapabilityRegistry;
use tinyagents::rlm::{InterpreterSpec, RlmHost, RlmPolicy, RlmSession, is_fatal};
use tinyagents::{HarnessSubAgent, SubAgent};

#[test]
fn is_fatal_flags_only_policy_bounds() {
    assert!(is_fatal(&TinyAgentsError::LimitExceeded("x".into())));
    assert!(is_fatal(&TinyAgentsError::Timeout("x".into())));
    assert!(is_fatal(&TinyAgentsError::Cancelled));
    assert!(is_fatal(&TinyAgentsError::SubAgentDepth(4)));

    // Recoverable, script-visible failures are NOT fatal — the model adapts.
    assert!(!is_fatal(&TinyAgentsError::Tool("boom".into())));
    assert!(!is_fatal(&TinyAgentsError::ToolNotFound("missing".into())));
    assert!(!is_fatal(&TinyAgentsError::ModelNotFound("missing".into())));
    assert!(!is_fatal(&TinyAgentsError::Validation("bad".into())));
    assert!(!is_fatal(&TinyAgentsError::Model(
        "provider blew up".into()
    )));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capabilities_listing_reflects_the_registry() {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(vec!["unused"])))
        .expect("register model");
    registry
        .register_tool(Arc::new(FakeTool::returning("echo", "ok")))
        .expect("register tool");

    let mut child: AgentHarness<()> = AgentHarness::new();
    child
        .register_model("cm", Arc::new(ScriptedModel::replies(vec!["done"])))
        .set_default_model("cm");
    let subagent = Arc::new(SubAgent::new(
        "helper",
        "A helpful sub-agent.",
        Arc::new(child),
    ));
    registry
        .register_agent(Arc::new(HarnessSubAgent::new(subagent)))
        .expect("register agent");

    let host = RlmHost::new(Arc::new(registry), Arc::new(()));
    let listing = tinyagents::rlm::RlmHostApi::capabilities(&host);
    assert_eq!(listing.models, vec!["mock".to_string()]);
    assert_eq!(listing.agents, vec!["helper".to_string()]);
    assert_eq!(listing.tools.len(), 1);
    assert_eq!(listing.tools[0].0, "echo");
    assert!(listing.tools[0].1.contains("Fake tool"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capabilities_listing_is_empty_for_a_bare_registry() {
    let host: RlmHost<()> = RlmHost::new(Arc::new(CapabilityRegistry::new()), Arc::new(()));
    let listing = tinyagents::rlm::RlmHostApi::capabilities(&host);
    assert!(listing.models.is_empty());
    assert!(listing.tools.is_empty());
    assert!(listing.agents.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn llm_without_a_default_model_surfaces_a_catchable_error() {
    // No `with_default_model`, and the script names no model: the host raises a
    // (recoverable) validation error that the script can catch.
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(vec!["unused"])))
        .expect("register model");
    let host = Arc::new(RlmHost::new(Arc::new(registry), Arc::new(())));
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host).expect("session");

    let outcome = session
        .eval(r#"try { llm("hi") } catch (e) { print("caught: " + e); } "handled""#)
        .await
        .expect("cell should complete — the error is recoverable");
    assert!(outcome.stdout.contains("caught"));
    assert!(outcome.stdout.contains("no model"));
    assert_eq!(outcome.error, None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_named_model_is_catchable_in_a_script() {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(vec!["unused"])))
        .expect("register model");
    let host = Arc::new(RlmHost::new(Arc::new(registry), Arc::new(())).with_default_model("mock"));
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host).expect("session");

    let outcome = session
        .eval(
            r#"try { llm(#{ model: "nope", prompt: "hi" }) } catch (e) { print("err: " + e); } "ok""#,
        )
        .await
        .expect("cell");
    assert!(outcome.stdout.contains("err:"));
    assert!(outcome.stdout.contains("nope"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sub_agent_depth_guard_trips_fatally() {
    // Seed the session at the depth cap so any `agent(...)` call would recurse
    // one level too deep — the shared harness guard aborts the cell.
    let policy = RlmPolicy {
        max_depth: 2,
        ..RlmPolicy::default()
    };
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(vec!["unused"])))
        .expect("register model");
    let host = Arc::new(
        RlmHost::new(Arc::new(registry), Arc::new(()))
            .with_policy(policy)
            .with_default_model("mock")
            .with_run_depth(2),
    );
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host).expect("session");

    let err = session
        .eval(r#"agent("anything", "go")"#)
        .await
        .expect_err("depth guard must abort the cell");
    assert!(
        matches!(err, TinyAgentsError::SubAgentDepth(2)),
        "got {err:?}"
    );
}
