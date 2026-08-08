//! Module-local unit tests for the RLM runtime: config serialization,
//! template rendering, code-fence extraction, and the embedded-Rhai session
//! against deterministic capability doubles.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use super::*;
use crate::harness::testkit::{FakeTool, ScriptedModel, SlowModel};
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
async fn oversized_multibyte_stdout_is_truncated_without_panicking() {
    // Regression test: `String::truncate` panics unless the cut index is a
    // UTF-8 char boundary. `max_output_bytes: 64` is not a multiple of the
    // 3-byte-wide "日" character printed below, so the naive raw-byte cut
    // used to panic partway through evaluating the cell.
    let policy = RlmPolicy {
        max_output_bytes: 64,
        ..RlmPolicy::default()
    };
    let mut session = rhai_session(registry_with_mock(vec![]), policy);
    let outcome = session
        .eval(r#"for i in 0..100 { print("日日日日日日日日日日"); }"#)
        .await
        .expect("cell must not panic on a multi-byte truncation boundary");
    assert!(outcome.stdout.contains("truncated"));
    // The truncated prefix must itself still be valid UTF-8 (no half-cut
    // multi-byte character), which `String::truncate` guarantees once the
    // cut lands on a char boundary.
    assert!(std::str::from_utf8(outcome.stdout.as_bytes()).is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_multibyte_value_is_truncated_without_panicking() {
    // Same boundary hazard as stdout, but for the rendered cell value.
    let policy = RlmPolicy {
        max_output_bytes: 64,
        ..RlmPolicy::default()
    };
    let mut session = rhai_session(registry_with_mock(vec![]), policy);
    let outcome = session
        .eval(r#"let s = ""; for i in 0..100 { s += "日"; } s"#)
        .await
        .expect("cell must not panic on a multi-byte truncation boundary");
    let value = outcome.value.expect("truncated value");
    let text = value.as_str().expect("string value");
    assert!(text.contains("truncated"));
    assert!(std::str::from_utf8(text.as_bytes()).is_ok());
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_a_cell_future_does_not_lose_the_persistent_scope() {
    // Regression test: `eval_cell` used to `mem::take` the scope out of
    // `self` and only restore it after its `spawn_blocking` task joined. A
    // caller that drops the `eval_cell` future mid-flight — the documented
    // `tokio::time::timeout`/`select!`/task-abort cancellation shape — left
    // `self.scope` permanently empty, since the detached blocking task's
    // `(scope, result)` was discarded along with the dropped future. Keeping
    // the scope behind `Arc<Mutex<_>>` means the orphaned task still writes
    // its updates into the *shared* scope, so a later cell still sees them.
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model(
            "mock",
            Arc::new(SlowModel::new(Duration::from_millis(150), "done")),
        )
        .expect("register model");
    let mut session = rhai_session(Arc::new(registry), RlmPolicy::default());

    // `x` is assigned before the script blocks on the slow `llm(...)` call,
    // so the assignment has already happened by the time this future is
    // dropped.
    let cancelled = tokio::time::timeout(
        Duration::from_millis(20),
        session.eval(r#"let x = 42; llm("wait"); x"#),
    )
    .await;
    assert!(
        cancelled.is_err(),
        "the timeout must fire before the slow llm() call resolves"
    );

    // Give the orphaned blocking task time to actually finish the call and
    // write the scope back.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let outcome = session.eval("x").await.expect("second cell");
    assert_eq!(outcome.value, Some(json!(42)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicked_cell_poisons_the_scope_instead_of_silently_emptying_it() {
    // The other half of the scope-loss defect: a blocking closure that
    // panics mid-eval (holding the scope lock) must not leave the
    // interpreter usable-but-amnesiac. The mutex poisons, and every
    // subsequent cell must fail loudly with a clear diagnostic rather than
    // silently running against an empty namespace.
    struct PanicModel;

    #[async_trait::async_trait]
    impl crate::harness::model::ChatModel<()> for PanicModel {
        async fn invoke(
            &self,
            _state: &(),
            _request: crate::harness::model::ModelRequest,
        ) -> crate::error::Result<crate::harness::model::ModelResponse> {
            panic!("simulated provider panic while holding the rlm scope");
        }
    }

    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("mock", Arc::new(PanicModel))
        .expect("register model");
    let mut session = rhai_session(Arc::new(registry), RlmPolicy::default());

    let first = session.eval(r#"let x = 1; llm("boom")"#).await;
    assert!(
        first.is_err(),
        "the panicking cell must fail, not silently succeed"
    );

    let second = session
        .eval("x")
        .await
        .expect_err("the scope is poisoned, so the next cell must fail loudly");
    assert!(
        matches!(&second, crate::error::TinyAgentsError::Model(msg) if msg.contains("interpreter state lost")),
        "got {second:?}"
    );
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_run_after_the_cell_budget_stops_gracefully_instead_of_erroring() {
    // Regression test: `RlmRunner::run` used to gate its loop on
    // `steps.len()`, a per-call counter that resets to zero on every `run()`
    // call, while `RlmSession::eval` enforces `max_cells` against its own
    // session-cumulative `cells_run` counter that nothing ever reset. The two
    // checks agreed only on the first call: a second `run()` on the same
    // (long-lived, `&mut self`) runner — legal, and a natural thing to do —
    // saw an empty `steps`, paid for a driver-model call it didn't need, and
    // then hit `self.session.eval`'s hard `LimitExceeded` error instead of
    // the graceful `CellBudgetExhausted` outcome the identical condition
    // produces on the first run.
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

    let first = runner.run("loop forever").await.expect("first run");
    assert_eq!(first.stop_reason, RlmStopReason::CellBudgetExhausted);
    assert_eq!(first.steps.len(), 2);

    let second = runner
        .run("try again")
        .await
        .expect("a second run must stop gracefully, not return a hard error");
    assert_eq!(second.stop_reason, RlmStopReason::CellBudgetExhausted);
    // No cells executed (the budget was already spent) and no driver call
    // wasted producing one that would only be rejected.
    assert_eq!(second.steps.len(), 0);
    assert_eq!(second.driver_calls, 0);
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
