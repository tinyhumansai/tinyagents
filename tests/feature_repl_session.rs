//! Feature tests for the Rhai-backed `.ragsh` session runtime
//! (`src/repl/session/`, gated behind the `repl` cargo feature).
//!
//! Run with `cargo test --features repl --test feature_repl_session`. The whole
//! file compiles to nothing without the feature (mirroring `e2e_rlm.rs`).
//!
//! These drive the crate-root [`ReplSession`] (the scripting session, exported
//! as `crate::ReplSession` when `repl` is enabled) from *outside* the crate to
//! cover user-facing capability features that the in-crate unit tests do not
//! exercise at the integration boundary: the `model_query` /
//! `model_query_batched` capability functions wired to a registered model, the
//! `graph_run` blueprint-resolution reference, the `show_vars()` built-in, the
//! persistent namespace across cells, per-session call budgets, and
//! reserved-name protection.
//!
//! Everything is deterministic and offline: models are testkit doubles
//! (`ScriptedModel`, `MockModel`) that never touch the network.

#![cfg(feature = "repl")]

use std::sync::Arc;

use tinyagents::harness::providers::MockModel;
use tinyagents::harness::testkit::ScriptedModel;
use tinyagents::language::Blueprint;
use tinyagents::registry::CapabilityRegistry;
use tinyagents::{
    ReplCallKind, ReplCapabilities, ReplPolicy, ReplSession, ReplValue, TinyAgentsError,
};

/// Builds a session over a registry that carries a single scripted model named
/// `assistant` returning the given replies in order.
fn session_with_model(replies: Vec<&str>) -> ReplSession {
    let mut registry = CapabilityRegistry::<()>::new();
    registry
        .register_model("assistant", Arc::new(ScriptedModel::replies(replies)))
        .expect("register model");
    ReplSession::<()>::new().with_capabilities(ReplCapabilities::new(Arc::new(registry)))
}

#[test]
fn model_query_calls_a_registered_model_and_records_the_call() {
    let mut s = session_with_model(vec!["the-answer"]);

    let result = s
        .eval_cell(r#"model_query(#{ model: "assistant", prompt: "hi" })"#)
        .expect("model_query should succeed against the registered model");

    // The cell value is the model's reply text.
    assert_eq!(
        result.value,
        Some(ReplValue::String("the-answer".to_string()))
    );

    // Exactly one Model capability call was recorded, naming the model.
    assert_eq!(result.calls.len(), 1);
    assert_eq!(result.calls[0].kind, ReplCallKind::Model);
    assert_eq!(result.calls[0].name, "assistant");
}

#[test]
fn model_query_structured_returns_content_and_finish_reason() {
    let mut s = session_with_model(vec!["structured-reply"]);

    let result = s
        .eval_cell(
            r#"let r = model_query(#{ model: "assistant", prompt: "hi", structured: true }); r.content"#,
        )
        .expect("structured model_query");

    assert_eq!(
        result.value,
        Some(ReplValue::String("structured-reply".to_string()))
    );
}

#[test]
fn model_query_on_an_unregistered_model_reports_model_not_found() {
    let mut s = ReplSession::<()>::new();

    let err = s
        .eval_cell(r#"model_query(#{ model: "ghost", prompt: "hi" })"#)
        .expect_err("querying an unregistered model must fail");
    assert!(
        matches!(err, TinyAgentsError::ModelNotFound(ref m) if m == "ghost"),
        "expected ModelNotFound(ghost), got {err:?}"
    );
}

#[test]
fn model_query_batched_fans_out_and_preserves_order() {
    // `MockModel::constant` makes each leg deterministic regardless of the
    // concurrency scheduling in the batch.
    let mut registry = CapabilityRegistry::<()>::new();
    registry
        .register_model("m", Arc::new(MockModel::constant("reply")))
        .expect("register model");
    let mut s =
        ReplSession::<()>::new().with_capabilities(ReplCapabilities::new(Arc::new(registry)));

    let result = s
        .eval_cell(
            r#"model_query_batched([
                #{ model: "m", prompt: "a" },
                #{ model: "m", prompt: "b" },
                #{ model: "m", prompt: "c" },
            ])"#,
        )
        .expect("batched model query");

    let value = result.value.expect("array value").to_json();
    let items = value.as_array().expect("array");
    assert_eq!(items.len(), 3);
    for item in items {
        assert_eq!(item, &serde_json::json!("reply"));
    }
    // One recorded Model call per leg.
    assert_eq!(result.calls.len(), 3);
    assert!(result.calls.iter().all(|c| c.kind == ReplCallKind::Model));
}

#[test]
fn model_call_budget_fails_closed() {
    let policy = ReplPolicy {
        max_model_calls: 2,
        ..ReplPolicy::default()
    };
    let mut registry = CapabilityRegistry::<()>::new();
    registry
        .register_model("m", Arc::new(MockModel::constant("ok")))
        .unwrap();
    let mut s = ReplSession::<()>::new()
        .with_policy(policy)
        .with_capabilities(ReplCapabilities::new(Arc::new(registry)));

    let call = r#"model_query(#{ model: "m", prompt: "x" })"#;
    s.eval_cell(call).expect("call 1 within budget");
    s.eval_cell(call).expect("call 2 within budget");

    let err = s
        .eval_cell(call)
        .expect_err("call 3 exceeds max_model_calls");
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

#[test]
fn graph_run_resolves_a_registered_blueprint_to_a_reference() {
    // Register a compiled blueprint by name; graph_run resolves it and hands
    // back a script-visible reference (graph id, start node, node count).
    let mut registry = CapabilityRegistry::<()>::new();
    let blueprint = Blueprint {
        graph_id: "triage".to_string(),
        start: "classify".to_string(),
        ..Blueprint::default()
    };
    registry
        .register_graph_blueprint("triage", blueprint)
        .expect("register blueprint");
    let mut s =
        ReplSession::<()>::new().with_capabilities(ReplCapabilities::new(Arc::new(registry)));

    let result = s
        .eval_cell(r#"graph_run(#{ graph: "triage" })"#)
        .expect("graph_run resolves the registered blueprint");

    let value = result.value.expect("reference map").to_json();
    assert_eq!(value["graph"], serde_json::json!("triage"));
    assert_eq!(value["start"], serde_json::json!("classify"));
    assert_eq!(value["resolved"], serde_json::json!(true));

    assert_eq!(result.calls.len(), 1);
    assert_eq!(result.calls[0].kind, ReplCallKind::Graph);
}

#[test]
fn graph_run_on_an_unregistered_graph_is_rejected() {
    let mut s = ReplSession::<()>::new();

    let err = s
        .eval_cell(r#"graph_run(#{ graph: "missing" })"#)
        .expect_err("running an unregistered graph must fail");
    assert!(
        matches!(err, TinyAgentsError::Capability(ref m) if m.contains("missing")),
        "got {err:?}"
    );
}

#[test]
fn persistent_namespace_survives_across_cells_and_reads_via_show_vars() {
    let mut s = ReplSession::<()>::new();

    // A binding from cell 1 is visible in later cells and to `show_vars()`.
    s.eval_cell(r#"let ticket = "T-42";"#)
        .expect("seed binding");
    let follow = s.eval_cell("ticket").expect("read binding");
    assert_eq!(follow.value, Some(ReplValue::String("T-42".to_string())));

    // `show_vars()` prints the persistent namespace captured at the start of
    // the cell (so it reflects `ticket` seeded earlier).
    let shown = s.eval_cell("show_vars(); ()").expect("show_vars runs");
    assert!(
        shown.stdout.contains("ticket"),
        "show_vars stdout should mention the persistent binding, got: {:?}",
        shown.stdout
    );
}

#[test]
fn reserved_capability_name_cannot_be_replaced_via_variables_api() {
    let mut s = ReplSession::<()>::new();

    // The public ReplVariables surface refuses to bind a reserved capability
    // name, so a caller cannot smuggle a replacement for `model_query`.
    let err = s
        .variables
        .set("model_query", ReplValue::Int(1))
        .expect_err("reserved capability names are protected");
    assert!(matches!(err, TinyAgentsError::Capability(_)), "got {err:?}");
}

#[test]
fn capabilities_expose_registered_names_by_kind() {
    let mut registry = CapabilityRegistry::<()>::new();
    registry
        .register_model("assistant", Arc::new(MockModel::constant("hi")))
        .unwrap();
    let caps = ReplCapabilities::new(Arc::new(registry));

    assert_eq!(caps.models(), vec!["assistant"]);
    assert!(caps.tools().is_empty());
    assert!(caps.graphs().is_empty());
    assert!(caps.agents().is_empty());
}
