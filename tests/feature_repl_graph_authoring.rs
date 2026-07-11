//! Feature tests for the `.ragsh` session graph-authoring lifecycle
//! (`src/repl/session/builtins/authoring.rs`, gated behind the `repl` feature).
//!
//! Run with `cargo test --features repl --test feature_repl_graph_authoring`.
//! The whole file compiles to nothing without the feature.
//!
//! A session that drafts a `.rag` graph must route it through the compiler, the
//! capability resolver, and the policy review gate before it can be registered
//! — generated topology is never installed directly. These tests cover that
//! `graph_define` → `graph_validate` → `graph_compile` → `graph_register` flow
//! and the review-token gate, which the in-crate unit tests only touch for the
//! `graph_define` limit-accounting edge case.
//!
//! Everything is deterministic and offline.

#![cfg(feature = "repl")]

use std::sync::Arc;

use tinyagents::harness::providers::MockModel;
use tinyagents::registry::CapabilityRegistry;
use tinyagents::{ReplCapabilities, ReplPolicy, ReplSession, TinyAgentsError};

/// A `.rag` graph whose single model node binds to a model named `assistant`.
const GRAPH_SOURCE: &str = r#"graph triage {
  start classify
  node classify {
    kind model
    model "assistant"
    next END
  }
}"#;

/// A session over a registry that carries the `assistant` model the graph binds
/// to, so `graph_compile`'s resolver gate passes.
fn session_with_assistant(policy: ReplPolicy) -> ReplSession {
    let mut registry = CapabilityRegistry::<()>::new();
    registry
        .register_model("assistant", Arc::new(MockModel::constant("ok")))
        .expect("register model");
    ReplSession::<()>::new()
        .with_policy(policy)
        .with_capabilities(ReplCapabilities::new(Arc::new(registry)))
}

/// A `graph_define(...)` cell for `GRAPH_SOURCE` under graph name `triage`.
fn define_cell() -> String {
    format!(r#"graph_define(#{{ name: "triage", source: `{GRAPH_SOURCE}` }})"#)
}

#[test]
fn graph_define_drafts_an_uncompiled_review_gated_blueprint() {
    let mut s = session_with_assistant(ReplPolicy::default());

    let result = s.eval_cell(&define_cell()).expect("graph_define drafts");
    let descriptor = result.value.expect("descriptor map").to_json();

    assert_eq!(descriptor["name"], serde_json::json!("triage"));
    // One model node.
    assert_eq!(descriptor["nodes"], serde_json::json!(1));
    // A fresh draft is not yet compiled and, under the default policy, requires
    // review before registration.
    assert_eq!(descriptor["compiled"], serde_json::json!(false));
    assert_eq!(descriptor["requires_review"], serde_json::json!(true));
}

#[test]
fn graph_validate_reports_unresolved_capability_references() {
    // The graph binds to `assistant`, which is NOT registered here, so the
    // resolver-backed `graph_validate` surfaces a diagnostic message.
    let mut s = ReplSession::<()>::new();

    let script = format!(
        r#"let g = graph_define(#{{ name: "triage", source: `{GRAPH_SOURCE}` }});
           graph_validate(g)"#
    );
    let result = s.eval_cell(&script).expect("define + validate");
    let messages = result.value.expect("array of messages").to_json();
    let messages = messages.as_array().expect("array");

    assert!(
        messages
            .iter()
            .any(|m| m.as_str().is_some_and(|s| s.contains("assistant"))),
        "validation should flag the unresolved `assistant` reference, got {messages:?}"
    );
}

#[test]
fn full_lifecycle_compiles_then_registers_with_a_review_token() {
    let mut s = session_with_assistant(ReplPolicy::default());

    // Define, then compile — compilation binds through the resolver and marks
    // the draft compiled.
    let compiled = s
        .eval_cell(&format!(
            r#"let g = graph_define(#{{ name: "triage", source: `{GRAPH_SOURCE}` }});
               graph_compile(g)"#
        ))
        .expect("define + compile");
    let descriptor = compiled.value.expect("compiled descriptor").to_json();
    assert_eq!(descriptor["compiled"], serde_json::json!(true));
    assert_eq!(descriptor["requires_review"], serde_json::json!(true));

    // Registering a review-gated graph without a review_id is rejected.
    let err = s
        .eval_cell(
            r#"let g = graph_compile(#{ name: "triage" });
               graph_register(#{ graph: g })"#,
        )
        .expect_err("registration without review must fail");
    assert!(
        matches!(err, TinyAgentsError::Validation(ref m) if m.contains("review")),
        "got {err:?}"
    );

    // Supplying a review token lets registration succeed; it returns the name.
    let registered = s
        .eval_cell(
            r#"let g = graph_compile(#{ name: "triage" });
               graph_register(#{ graph: g, review_id: "reviewed-by-human" })"#,
        )
        .expect("registration with review token succeeds");
    assert_eq!(
        registered.value.expect("name").to_json(),
        serde_json::json!("triage")
    );
}

#[test]
fn register_before_compile_is_rejected() {
    let mut s = session_with_assistant(ReplPolicy::default());

    // Draft exists but was never compiled: registration must fail closed.
    let err = s
        .eval_cell(&format!(
            r#"let g = graph_define(#{{ name: "triage", source: `{GRAPH_SOURCE}` }});
               graph_register(#{{ graph: g }})"#
        ))
        .expect_err("registering an uncompiled draft must fail");
    assert!(
        matches!(err, TinyAgentsError::Validation(ref m) if m.contains("compiled")),
        "got {err:?}"
    );
}

#[test]
fn review_gate_can_be_disabled_by_policy() {
    // With the review gate off, a compiled graph registers without a token.
    let policy = ReplPolicy {
        generated_graphs_require_review: false,
        ..ReplPolicy::default()
    };
    let mut s = session_with_assistant(policy);

    let registered = s
        .eval_cell(&format!(
            r#"let g = graph_define(#{{ name: "triage", source: `{GRAPH_SOURCE}` }});
               let c = graph_compile(g);
               graph_register(#{{ graph: c }})"#
        ))
        .expect("no-review policy registers without a token");
    assert_eq!(
        registered.value.expect("name").to_json(),
        serde_json::json!("triage")
    );
}
