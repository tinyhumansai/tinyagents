//! Module-local unit tests for the RLM runtime: config serialization,
//! template rendering, code-fence extraction, and the embedded-Rhai session
//! against deterministic capability doubles.

use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::harness::testkit::{FakeTool, ScriptedModel};
use crate::registry::CapabilityRegistry;

fn registry_with_mock(replies: Vec<&str>) -> Arc<CapabilityRegistry<()>> {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(ScriptedModel::replies(replies)))
        .expect("register model");
    registry
        .register_tool(Arc::new(FakeTool::returning("echo", "echoed")))
        .expect("register tool");
    Arc::new(registry)
}

fn rhai_session(registry: Arc<CapabilityRegistry<()>>, policy: RlmPolicy) -> RlmSession<()> {
    let host = Arc::new(
        RlmHost::new(registry, Arc::new(()))
            .with_policy(policy)
            .with_default_model("mock"),
    );
    RlmSession::new(&InterpreterSpec::Rhai, host).expect("build session")
}

// ── Config round-trips ──────────────────────────────────────────────────────

#[test]
fn config_round_trips_through_json() {
    let config = RlmConfig {
        interpreter: InterpreterSpec::Python {
            binary: Some("python3".to_string()),
            args: vec![],
        },
        driver_model: Some("openai".to_string()),
        sub_model: None,
        policy: RlmPolicy::default(),
        template: TemplateSpec::Named("context-explorer".to_string()),
    };
    let json = config.to_json().expect("serialize");
    let back = RlmConfig::from_json(&json).expect("parse");
    assert_eq!(config, back);
}

#[test]
fn minimal_config_document_parses_with_defaults() {
    let config = RlmConfig::from_json(r#"{ "interpreter": {"kind": "rhai"} }"#).expect("parse");
    assert_eq!(config.interpreter, InterpreterSpec::Rhai);
    assert_eq!(config.template, TemplateSpec::Named("general".to_string()));
    assert_eq!(config.policy, RlmPolicy::default());
}

#[test]
fn host_call_wire_shape_is_stable() {
    let call: HostCall = serde_json::from_value(json!({
        "capability": "llm",
        "prompt": "hi",
        "model": null,
        "system": null,
    }))
    .expect("parse llm call");
    assert_eq!(
        call,
        HostCall::Llm {
            model: None,
            prompt: "hi".to_string(),
            system: None
        }
    );
    let call: HostCall = serde_json::from_value(json!({
        "capability": "tool",
        "tool": "echo",
    }))
    .expect("parse tool call without arguments");
    assert!(matches!(call, HostCall::Tool { arguments, .. } if arguments.is_null()));
}

// ── Code-fence extraction ───────────────────────────────────────────────────

#[test]
fn extracts_fenced_code_and_rejects_prose() {
    assert_eq!(
        extract_code_cell("Let me try:\n```rhai\nlet x = 1;\nx\n```\nDone."),
        Some("let x = 1;\nx".to_string())
    );
    assert_eq!(
        extract_code_cell("```\nprint(1)\n```"),
        Some("print(1)".to_string())
    );
    assert_eq!(extract_code_cell("no code here"), None);
    assert_eq!(extract_code_cell("unterminated ```python\nprint(1)"), None);
    assert_eq!(extract_code_cell("```rhai\n\n```"), None);
}

// ── Template rendering ──────────────────────────────────────────────────────

#[test]
fn renders_placeholders_into_the_system_prompt() {
    let listing = CapabilityListing {
        models: vec!["mock".to_string()],
        tools: vec![("echo".to_string(), "Echoes.".to_string())],
        agents: vec!["helper".to_string()],
    };
    let prompt = templates::render_system_prompt(
        &templates::general(),
        "rhai",
        "USAGE GUIDE",
        &listing,
        &RlmPolicy::default(),
    );
    assert!(prompt.contains("```rhai"));
    assert!(prompt.contains("USAGE GUIDE"));
    assert!(prompt.contains("echo: Echoes."));
    assert!(prompt.contains("helper"));
    assert!(!prompt.contains("{{"));
}

#[test]
fn unknown_named_template_fails_closed() {
    let err = templates::resolve(&TemplateSpec::Named("nope".to_string()));
    assert!(err.is_err());
}

// ── Embedded Rhai session ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rhai_cell_evaluates_and_persists_variables() {
    let mut session = rhai_session(registry_with_mock(vec!["unused"]), RlmPolicy::default());
    let outcome = session.eval("let x = 21; x").await.expect("cell 1");
    assert_eq!(outcome.value, Some(json!(21)));
    let outcome = session.eval("x * 2").await.expect("cell 2");
    assert_eq!(outcome.value, Some(json!(42)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rhai_cell_calls_llm_tool_and_final_answer() {
    let mut session = rhai_session(
        registry_with_mock(vec!["sub-model says hi"]),
        RlmPolicy::default(),
    );
    let outcome = session
        .eval(
            r#"
            let reply = llm("hello?");
            print(reply);
            let echoed = tool("echo", #{ q: 7 });
            final_answer(reply);
            "#,
        )
        .await
        .expect("cell");
    assert!(outcome.stdout.contains("sub-model says hi"));
    assert_eq!(outcome.final_answer.as_deref(), Some("sub-model says hi"));
    assert_eq!(outcome.calls.len(), 3);
    assert_eq!(outcome.calls[0].kind, RlmCallKind::Llm);
    assert_eq!(outcome.calls[1].kind, RlmCallKind::Tool);
    assert_eq!(outcome.calls[2].kind, RlmCallKind::FinalAnswer);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn script_error_is_recoverable_not_fatal() {
    let mut session = rhai_session(registry_with_mock(vec![]), RlmPolicy::default());
    let outcome = session.eval("this is not rhai ][").await.expect("cell");
    assert!(outcome.error.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_tool_error_is_catchable_in_script() {
    let mut session = rhai_session(registry_with_mock(vec![]), RlmPolicy::default());
    let outcome = session
        .eval(r#"try { tool("missing") } catch (e) { print("caught: " + e); } "ok""#)
        .await
        .expect("cell");
    assert!(outcome.stdout.contains("caught"));
    assert_eq!(outcome.value, Some(json!("ok")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn llm_call_limit_is_fatal_and_aborts_the_cell() {
    let policy = RlmPolicy {
        max_llm_calls: 1,
        ..RlmPolicy::default()
    };
    let mut session = rhai_session(registry_with_mock(vec!["one", "two"]), policy);
    let err = session
        .eval(r#"llm("first"); llm("second")"#)
        .await
        .expect_err("limit must abort");
    assert!(
        matches!(err, crate::error::TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cell_budget_and_script_size_fail_closed() {
    let policy = RlmPolicy {
        max_cells: 1,
        max_script_bytes: 16,
        ..RlmPolicy::default()
    };
    let mut session = rhai_session(registry_with_mock(vec![]), policy);
    let err = session
        .eval("1 + 1 + 1 + 1 + 1 + 1 + 1")
        .await
        .expect_err("script too large");
    assert!(matches!(
        err,
        crate::error::TinyAgentsError::LimitExceeded(_)
    ));
    session.eval("1").await.expect("first small cell");
    let err = session.eval("2").await.expect_err("cell budget");
    assert!(matches!(
        err,
        crate::error::TinyAgentsError::LimitExceeded(_)
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_stdout_is_truncated_with_a_marker() {
    let policy = RlmPolicy {
        max_output_bytes: 64,
        ..RlmPolicy::default()
    };
    let mut session = rhai_session(registry_with_mock(vec![]), policy);
    let outcome = session
        .eval(r#"for i in 0..100 { print("aaaaaaaaaaaaaaaaaaaaaaaa"); }"#)
        .await
        .expect("cell");
    assert!(outcome.stdout.len() < 200);
    assert!(outcome.stdout.contains("truncated"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn context_variable_is_visible_to_scripts() {
    let mut session = rhai_session(registry_with_mock(vec![]), RlmPolicy::default());
    session
        .set_variable("context", json!({"words": ["alpha", "beta"]}))
        .await
        .expect("set context");
    let outcome = session.eval("context.words[1]").await.expect("cell");
    assert_eq!(outcome.value, Some(json!("beta")));
}

// ── The model-driven runner ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runner_loops_until_final_answer() {
    // Cell 1 computes and prints; cell 2 answers with the observed value.
    let registry = registry_with_mock(vec![
        "Let me compute.\n```rhai\nlet x = 6 * 7;\nprint(x);\nx\n```",
        "Now I know.\n```rhai\nfinal_answer(\"the answer is 42\")\n```",
    ]);
    let config = RlmConfig {
        driver_model: Some("mock".to_string()),
        ..RlmConfig::default()
    };
    let mut runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("build runner");
    let outcome = runner.run("multiply 6 by 7").await.expect("run");
    assert_eq!(outcome.answer.as_deref(), Some("the answer is 42"));
    assert_eq!(outcome.stop_reason, RlmStopReason::Answered);
    assert_eq!(outcome.steps.len(), 2);
    assert_eq!(outcome.driver_calls, 2);
    assert!(outcome.steps[0].outcome.stdout.contains("42"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runner_nudges_once_then_accepts_prose_as_the_answer() {
    // A fence-less reply first earns a nudge (it may be unfenced code, not an
    // answer); only a second fence-less reply is accepted as prose.
    let registry = registry_with_mock(vec!["The answer is 4.", "The answer is 4."]);
    let config = RlmConfig {
        driver_model: Some("mock".to_string()),
        ..RlmConfig::default()
    };
    let mut runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("build runner");
    let outcome = runner.run("what is 2+2?").await.expect("run");
    assert_eq!(outcome.answer.as_deref(), Some("The answer is 4."));
    assert_eq!(outcome.stop_reason, RlmStopReason::ModelAnswered);
    assert_eq!(outcome.driver_calls, 2);
    assert!(outcome.steps.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runner_recovers_a_cell_after_a_nudge() {
    // Unfenced code first (would previously have been mistaken for an
    // answer), fenced after the nudge, then a final answer.
    let registry = registry_with_mock(vec![
        "let x = 6 * 7; x",
        "```rhai\nlet x = 6 * 7;\nx\n```",
        "```rhai\nfinal_answer(\"42\")\n```",
    ]);
    let config = RlmConfig {
        driver_model: Some("mock".to_string()),
        ..RlmConfig::default()
    };
    let mut runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("build runner");
    let outcome = runner.run("multiply 6 by 7").await.expect("run");
    assert_eq!(outcome.answer.as_deref(), Some("42"));
    assert_eq!(outcome.stop_reason, RlmStopReason::Answered);
    assert_eq!(outcome.steps.len(), 2);
    assert_eq!(outcome.driver_calls, 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runner_stops_at_the_cell_budget() {
    // The driver keeps emitting cells and never answers.
    let cells: Vec<&str> = vec!["```rhai\n1\n```"; 4];
    let registry = registry_with_mock(cells);
    let config = RlmConfig {
        driver_model: Some("mock".to_string()),
        policy: RlmPolicy {
            max_cells: 2,
            ..RlmPolicy::default()
        },
        ..RlmConfig::default()
    };
    let mut runner = RlmRunner::from_config(config, registry, Arc::new(())).expect("build runner");
    let outcome = runner.run("loop forever").await.expect("run");
    assert_eq!(outcome.answer, None);
    assert_eq!(outcome.stop_reason, RlmStopReason::CellBudgetExhausted);
    assert_eq!(outcome.steps.len(), 2);
}

// ── Cancellation ────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_cancelled_session_refuses_cells() {
    let cancel = RlmCancelFlag::new();
    cancel.cancel();
    let host = Arc::new(
        RlmHost::new(registry_with_mock(vec![]), Arc::new(()))
            .with_default_model("mock")
            .with_cancel_flag(cancel),
    );
    let mut session = RlmSession::new(&InterpreterSpec::Rhai, host).expect("session");
    let err = session.eval("1").await.expect_err("must refuse");
    assert!(matches!(err, crate::error::TinyAgentsError::Cancelled));
}
