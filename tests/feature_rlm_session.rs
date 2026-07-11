//! Feature tests for the **RLM session lifecycle** and embedded-Rhai execution
//! semantics that the module's own unit tests leave under-covered: the
//! programmatic accessors (`language`, `usage_guide`, `cells_run`), building a
//! session over a pre-constructed interpreter backend, injecting structured
//! context, the fail-closed operation limit, cumulative call counts, idempotent
//! shutdown, and the sticky cancellation flag.
//!
//! All tests are offline: model/tool doubles come from `harness::testkit` and
//! the embedded Rhai engine has no OS access.

#![cfg(feature = "rlm")]

use std::sync::Arc;

use serde_json::json;
use tinyagents::harness::testkit::{FakeTool, ScriptedModel};
use tinyagents::registry::CapabilityRegistry;
use tinyagents::rlm::{
    InterpreterSpec, RlmCancelFlag, RlmHost, RlmPolicy, RlmSession, build_interpreter,
};

fn registry(replies: Vec<&str>) -> Arc<CapabilityRegistry<()>> {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(replies)))
        .expect("register model");
    registry
        .register_tool(Arc::new(FakeTool::returning("echo", "echoed")))
        .expect("register tool");
    Arc::new(registry)
}

fn host(replies: Vec<&str>, policy: RlmPolicy) -> Arc<RlmHost<()>> {
    Arc::new(
        RlmHost::new(registry(replies), Arc::new(()))
            .with_policy(policy)
            .with_default_model("mock"),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_exposes_language_and_usage_guide() {
    let session = RlmSession::new(&InterpreterSpec::Rhai, host(vec![], RlmPolicy::default()))
        .expect("build session");
    assert_eq!(session.language(), "rhai");
    let guide = session.usage_guide();
    assert!(guide.contains("llm("));
    assert!(guide.contains("final_answer("));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_tracks_the_number_of_cells_run() {
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host(vec![], RlmPolicy::default()))
        .expect("build session");
    assert_eq!(session.cells_run(), 0);
    session.eval("1").await.expect("cell 1");
    session.eval("2").await.expect("cell 2");
    assert_eq!(session.cells_run(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_can_be_built_over_a_prebuilt_interpreter_backend() {
    let interpreter =
        build_interpreter(&InterpreterSpec::Rhai, RlmPolicy::default().max_operations)
            .expect("build interpreter");
    let mut session = RlmSession::from_interpreter(interpreter, host(vec![], RlmPolicy::default()));
    let outcome = session.eval("40 + 2").await.expect("cell");
    assert_eq!(outcome.value, Some(json!(42)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn injected_context_supports_nested_structures() {
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host(vec![], RlmPolicy::default()))
        .expect("build session");
    session
        .set_variable(
            "context",
            json!({ "user": { "roles": ["admin", "editor"] } }),
        )
        .await
        .expect("set context");
    let outcome = session.eval("context.user.roles[0]").await.expect("cell");
    assert_eq!(outcome.value, Some(json!("admin")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runaway_operation_count_fails_closed() {
    let policy = RlmPolicy {
        max_operations: 10_000,
        ..RlmPolicy::default()
    };
    let mut session =
        RlmSession::new(&InterpreterSpec::Rhai, host(vec![], policy)).expect("build session");
    let err = session
        .eval("let n = 0; while n < 100000000 { n += 1; } n")
        .await
        .expect_err("operation limit must abort the cell");
    assert!(
        matches!(err, tinyagents::TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_host_accumulates_capability_call_counts() {
    let mut session = RlmSession::new(
        &InterpreterSpec::Rhai,
        host(vec!["a", "b"], RlmPolicy::default()),
    )
    .expect("build session");
    assert_eq!(session.host().call_counts(), (0, 0, 0));

    session
        .eval(r#"llm("one"); tool("echo")"#)
        .await
        .expect("cell 1");
    session.eval(r#"llm("two")"#).await.expect("cell 2");

    let (llm, tool, agent) = session.host().call_counts();
    assert_eq!((llm, tool, agent), (2, 1, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_is_idempotent_for_the_embedded_backend() {
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host(vec![], RlmPolicy::default()))
        .expect("build session");
    session.shutdown().await.expect("first shutdown");
    session.shutdown().await.expect("second shutdown");
}

#[test]
fn cancel_flag_is_sticky_and_shared_across_clones() {
    let flag = RlmCancelFlag::new();
    let clone = flag.clone();
    assert!(!flag.is_cancelled());
    assert!(!clone.is_cancelled());

    clone.cancel();
    assert!(
        flag.is_cancelled(),
        "cancellation is observed by every clone"
    );
    // Idempotent: cancelling again keeps it cancelled.
    clone.cancel();
    assert!(flag.is_cancelled());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_mid_session_refuses_further_cells() {
    let flag = RlmCancelFlag::new();
    let host = Arc::new(
        RlmHost::new(registry(vec![]), Arc::new(()))
            .with_default_model("mock")
            .with_cancel_flag(flag.clone()),
    );
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host).expect("build session");

    session
        .eval("1 + 1")
        .await
        .expect("first cell before cancel");
    flag.cancel();
    let err = session
        .eval("2 + 2")
        .await
        .expect_err("cancelled session must refuse work");
    assert!(matches!(err, tinyagents::TinyAgentsError::Cancelled));
}
