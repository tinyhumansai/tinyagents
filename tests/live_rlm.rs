//! Live, network-gated RLM tests (feature = "rlm").
//!
//! Skips gracefully (early return, not a panic) when `OPENAI_API_KEY` is not
//! set, following the `live_*.rs` convention. Assertions are structural only
//! (an answer was produced, cells actually executed) — never on exact model
//! prose.

#![cfg(feature = "rlm")]

use std::sync::Arc;

use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::registry::CapabilityRegistry;
use tinyagents::rlm::{RlmConfig, RlmRunner, RlmStopReason};

fn live_registry() -> Option<Arc<CapabilityRegistry<()>>> {
    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping: OPENAI_API_KEY is not set");
        return None;
    }
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("openai", Arc::new(OpenAiModel::from_env().expect("model")))
        .expect("register model");
    Some(Arc::new(registry))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_rhai_rlm_computes_with_code() {
    let Some(registry) = live_registry() else {
        return;
    };
    let config = RlmConfig::from_json(
        r#"{
            "interpreter": {"kind": "rhai"},
            "driver_model": "openai",
            "template": "general",
            "policy": {"max_cells": 6, "cell_timeout": 60000}
        }"#,
    )
    .expect("config");
    let mut runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("runner");
    let outcome = runner
        .run(
            "Compute the exact sum of the cubes of the integers from 1 to 25 with code, then \
             return just that number as the final answer.",
        )
        .await
        .expect("run");
    // 1³+…+25³ = (25·26/2)² = 105625.
    let answer = outcome.answer.expect("an answer");
    assert!(answer.contains("105625"), "unexpected answer: {answer}");
    assert!(
        !outcome.steps.is_empty(),
        "the model must have executed at least one cell"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_python_rlm_probes_an_injected_context() {
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: python3 is not on PATH");
        return;
    }
    let Some(registry) = live_registry() else {
        return;
    };
    let config = RlmConfig::from_json(
        r#"{
            "interpreter": {"kind": "python"},
            "driver_model": "openai",
            "template": "context-explorer",
            "policy": {"max_cells": 6, "cell_timeout": 60000}
        }"#,
    )
    .expect("config");
    let mut runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("runner");

    // A context with a needle the model must find programmatically.
    let mut lines: Vec<String> = (0..500)
        .map(|i| format!("log line {i}: heartbeat ok"))
        .collect();
    lines[317] = "log line 317: FATAL disk failure on node srv-42".to_string();
    runner
        .set_context(serde_json::json!(lines.join("\n")))
        .await
        .expect("set context");

    let outcome = runner
        .run("Exactly one log line in `context` is not a heartbeat. Which node failed?")
        .await
        .expect("run");
    let answer = outcome.answer.expect("an answer");
    assert!(answer.contains("srv-42"), "unexpected answer: {answer}");
    assert_ne!(outcome.stop_reason, RlmStopReason::CellBudgetExhausted);
    runner.shutdown().await.expect("shutdown");
}
