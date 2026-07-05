//! End-to-end tests for the `rlm` surface (feature = "rlm").
//!
//! Deterministic tests drive the embedded Rhai backend and the model-driven
//! runner against testkit doubles. The external-interpreter tests run against
//! a real `python3` / `node` from `PATH` and **skip gracefully** (early
//! return) when the binary is missing, mirroring the `live_*.rs` gating
//! convention — no network or API key is needed either way.

#![cfg(feature = "rlm")]

use std::sync::Arc;

use serde_json::json;
use tinyagents::harness::message::Message;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{FakeTool, ScriptedModel};
use tinyagents::registry::CapabilityRegistry;
use tinyagents::rlm::{
    InterpreterSpec, RlmConfig, RlmHost, RlmPolicy, RlmRunner, RlmSession, RlmStopReason,
};
use tinyagents::{HarnessSubAgent, SubAgent};

fn registry_with_doubles(replies: Vec<&str>) -> Arc<CapabilityRegistry<()>> {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(replies)))
        .expect("register model");
    registry
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-result-42")))
        .expect("register tool");
    Arc::new(registry)
}

fn session_for(spec: &InterpreterSpec, registry: Arc<CapabilityRegistry<()>>) -> RlmSession<()> {
    let host = Arc::new(
        RlmHost::new(registry, Arc::new(()))
            .with_policy(RlmPolicy::default())
            .with_default_model("mock"),
    );
    RlmSession::new(spec, host).expect("build session")
}

fn binary_available(binary: &str) -> bool {
    std::process::Command::new(binary)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

// ── Sub-agent delegation from inside a script ───────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rhai_script_delegates_to_a_registered_subagent() {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(vec!["unused"])))
        .expect("register model");

    // A real harness-backed sub-agent with its own scripted model.
    let mut child_harness: AgentHarness<()> = AgentHarness::new();
    child_harness
        .register_model(
            "child-model",
            Arc::new(ScriptedModel::replies(vec!["report from the child agent"])),
        )
        .set_default_model("child-model");
    let subagent = Arc::new(SubAgent::new(
        "researcher",
        "Investigates a question and reports back.",
        Arc::new(child_harness),
    ));
    registry
        .register_agent(Arc::new(HarnessSubAgent::new(subagent)))
        .expect("register agent");

    let mut session = session_for(&InterpreterSpec::Rhai, Arc::new(registry));
    let outcome = session
        .eval(r#"let report = agent("researcher", "investigate X"); final_answer(report)"#)
        .await
        .expect("cell");
    assert_eq!(
        outcome.final_answer.as_deref(),
        Some("report from the child agent")
    );
}

// ── External Python interpreter ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn python_interpreter_runs_cells_and_calls_capabilities() {
    if !binary_available("python3") {
        eprintln!("skipping: python3 is not on PATH");
        return;
    }
    let spec = InterpreterSpec::Python {
        binary: None,
        args: vec![],
    };
    let mut session = session_for(&spec, registry_with_doubles(vec!["sub-llm reply"]));

    // Globals persist across cells; last expression is the value.
    let outcome = session.eval("x = 40\nx + 2").await.expect("cell 1");
    assert_eq!(outcome.value, Some(json!(42)));

    // Context injection without source splicing.
    session
        .set_variable("context", json!({"items": [1, 2, 3]}))
        .await
        .expect("set context");
    let outcome = session
        .eval("len(context['items']) + x")
        .await
        .expect("cell 2");
    assert_eq!(outcome.value, Some(json!(43)));

    // Capability calls: llm + tool + final_answer through the wire protocol.
    let outcome = session
        .eval(
            "reply = llm('hello?')\nprint(reply)\nr = tool('lookup', {'q': 1})\nfinal_answer(reply)",
        )
        .await
        .expect("cell 3");
    assert!(outcome.stdout.contains("sub-llm reply"));
    assert_eq!(outcome.final_answer.as_deref(), Some("sub-llm reply"));

    // Script exceptions are recoverable and RlmError is catchable.
    let outcome = session
        .eval("try:\n    tool('missing')\nexcept RlmError as e:\n    print('caught', e)\n'ok'")
        .await
        .expect("cell 4");
    assert!(outcome.stdout.contains("caught"));
    assert_eq!(outcome.value, Some(json!("ok")));

    session.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn python_cell_timeout_kills_the_child_fail_closed() {
    if !binary_available("python3") {
        eprintln!("skipping: python3 is not on PATH");
        return;
    }
    let registry = registry_with_doubles(vec![]);
    let host = Arc::new(
        RlmHost::new(registry, Arc::new(()))
            .with_policy(RlmPolicy {
                cell_timeout: Some(std::time::Duration::from_millis(400)),
                ..RlmPolicy::default()
            })
            .with_default_model("mock"),
    );
    let mut session = RlmSession::new(
        &InterpreterSpec::Python {
            binary: None,
            args: vec![],
        },
        host,
    )
    .expect("session");
    let started = std::time::Instant::now();
    let err = session
        .eval("import time\ntime.sleep(60)")
        .await
        .expect_err("must time out");
    assert!(
        matches!(err, tinyagents::TinyAgentsError::Timeout(_)),
        "got {err:?}"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

// ── External JavaScript interpreter ─────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn javascript_interpreter_runs_cells_and_calls_capabilities() {
    if !binary_available("node") {
        eprintln!("skipping: node is not on PATH");
        return;
    }
    let spec = InterpreterSpec::Javascript {
        binary: None,
        args: vec![],
    };
    let mut session = session_for(&spec, registry_with_doubles(vec!["js sub-llm reply"]));

    let outcome = session.eval("let x = 40; x + 2").await.expect("cell 1");
    assert_eq!(outcome.value, Some(json!(42)));

    session
        .set_variable("context", json!("needle in a haystack"))
        .await
        .expect("set context");
    let outcome = session
        .eval("context.includes('needle')")
        .await
        .expect("cell 2");
    assert_eq!(outcome.value, Some(json!(true)));

    let outcome = session
        .eval(
            "const reply = llm('hello?'); console.log(reply); tool('lookup', {q: 1}); final_answer(reply)",
        )
        .await
        .expect("cell 3");
    assert!(outcome.stdout.contains("js sub-llm reply"));
    assert_eq!(outcome.final_answer.as_deref(), Some("js sub-llm reply"));

    session.shutdown().await.expect("shutdown");
}

// ── Config-driven runner over an external interpreter ───────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runner_drives_a_python_session_from_a_json_config() {
    if !binary_available("python3") {
        eprintln!("skipping: python3 is not on PATH");
        return;
    }
    let registry = registry_with_doubles(vec![
        // The scripted "driver model" writes a python cell, then answers.
        "```python\ntotal = sum(range(10))\nprint(total)\ntotal\n```",
        "```python\nfinal_answer(f'the sum is {total}')\n```",
    ]);
    let config = RlmConfig::from_json(
        r#"{
            "interpreter": {"kind": "python"},
            "driver_model": "mock",
            "template": "general",
            "policy": {"max_cells": 4, "cell_timeout": 30000}
        }"#,
    )
    .expect("parse config");
    let mut runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("runner");
    let outcome = runner.run("sum the numbers below 10").await.expect("run");
    assert_eq!(outcome.answer.as_deref(), Some("the sum is 45"));
    assert_eq!(outcome.stop_reason, RlmStopReason::Answered);
    assert_eq!(outcome.steps.len(), 2);
    assert!(outcome.steps[0].outcome.stdout.contains("45"));
    runner.shutdown().await.expect("shutdown");
}

// ── Prompt surface sanity ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn system_prompt_lists_live_capabilities_and_language() {
    let registry = registry_with_doubles(vec![]);
    let config = RlmConfig {
        driver_model: Some("mock".to_string()),
        ..RlmConfig::default()
    };
    let runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("runner");
    let prompt = runner.system_prompt();
    assert!(prompt.contains("```rhai"));
    assert!(prompt.contains("lookup"));
    assert!(prompt.contains("final_answer"));
}

// ── Messages are ordinary harness messages ──────────────────────────────────

#[test]
fn observation_shapes_are_plain_messages() {
    // Guard against accidental coupling: the runner speaks in ordinary
    // Message values, so any ChatModel implementation can drive it.
    let m = Message::user("observation");
    assert_eq!(m.text(), "observation");
}
