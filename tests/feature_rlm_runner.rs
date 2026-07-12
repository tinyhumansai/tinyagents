//! Feature tests for **RLM runner orchestration** beyond the happy-path loop
//! the unit tests already cover: construction validation, the `sub_model`
//! fallback to the driver model, context injection into the sandbox, the
//! capability-call tallies rolled into the outcome, and the rendered system
//! prompt reflecting the configured template.
//!
//! Offline throughout — the "driver model" is a `ScriptedModel` whose replies
//! double as both driver turns and sub-LLM answers (they share one registry
//! model), which is exactly how the `sub_model` fallback is observed.

#![cfg(feature = "rlm")]

use std::sync::Arc;

use serde_json::json;
use tinyagents::harness::testkit::{FakeTool, ScriptedModel};
use tinyagents::registry::CapabilityRegistry;
use tinyagents::rlm::{RlmConfig, RlmRunner, RlmStopReason, TemplateSpec};

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

fn config_with_driver() -> RlmConfig {
    RlmConfig {
        driver_model: Some("mock".to_string()),
        ..RlmConfig::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn building_a_runner_without_any_model_fails_closed() {
    let empty: Arc<CapabilityRegistry<()>> = Arc::new(CapabilityRegistry::new());
    let err = RlmRunner::from_config(RlmConfig::default(), empty, Arc::new(()))
        .err()
        .expect("no model means no runner");
    assert!(
        matches!(err, tinyagents::TinyAgentsError::Validation(_)),
        "got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_driver_only_config_uses_the_driver_as_the_sub_llm() {
    // reply 0: the driver turn (a cell that calls the *unnamed* sub-LLM).
    // reply 1: served to that `llm(...)` call — proving the sub-LLM defaulted
    //          to the driver model since `sub_model` was left unset.
    let registry = registry(vec![
        "```rhai\nlet r = llm(\"inner question\");\nfinal_answer(r)\n```",
        "sub-llm answered",
    ]);
    let mut runner =
        RlmRunner::from_config(config_with_driver(), registry, Arc::new(())).expect("runner");
    let outcome = runner.run("solve it").await.expect("run");
    assert_eq!(outcome.answer.as_deref(), Some("sub-llm answered"));
    assert_eq!(outcome.stop_reason, RlmStopReason::Answered);
    assert_eq!(outcome.sub_llm_calls, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn injected_context_is_visible_to_driver_written_cells() {
    let registry = registry(vec!["```rhai\nfinal_answer(context.label)\n```"]);
    let mut runner =
        RlmRunner::from_config(config_with_driver(), registry, Arc::new(())).expect("runner");
    runner
        .set_context(json!({ "label": "from-context" }))
        .await
        .expect("set context");
    let outcome = runner.run("use the context").await.expect("run");
    assert_eq!(outcome.answer.as_deref(), Some("from-context"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_outcome_tallies_script_capability_calls() {
    let registry = registry(vec![
        "```rhai\nlet t = tool(\"echo\", #{ q: 1 });\nlet l = llm(\"hi\");\nfinal_answer(\"done\")\n```",
        "sub reply",
    ]);
    let mut runner =
        RlmRunner::from_config(config_with_driver(), registry, Arc::new(())).expect("runner");
    let outcome = runner.run("do work").await.expect("run");
    assert_eq!(outcome.answer.as_deref(), Some("done"));
    assert_eq!(outcome.tool_calls, 1);
    assert_eq!(outcome.sub_llm_calls, 1);
    assert_eq!(outcome.agent_calls, 0);
    assert_eq!(outcome.driver_calls, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_system_prompt_reflects_the_configured_template_and_capabilities() {
    let registry = registry(vec![]);
    let config = RlmConfig {
        driver_model: Some("mock".to_string()),
        template: TemplateSpec::Named("orchestrator".to_string()),
        ..RlmConfig::default()
    };
    let runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("runner");
    let prompt = runner.system_prompt();
    // Orchestrator scaffold language plus the live capability listing.
    assert!(prompt.to_lowercase().contains("orchestrator"));
    assert!(prompt.contains("```rhai"));
    assert!(prompt.contains("echo"));
    assert!(!prompt.contains("{{"));
}
