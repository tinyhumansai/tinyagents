//! Unit tests for the Rhai-backed `.ragsh` session runtime.

use std::time::{Duration, Instant};

use super::*;
use crate::error::TinyAgentsError;

/// A fresh stateless session for tests.
fn session() -> ReplSession {
    ReplSession::new()
}

#[test]
fn evaluates_an_expression_and_returns_the_value() {
    let mut s = session();
    let result = s.eval_cell("1 + 2").expect("eval");
    assert_eq!(result.value, Some(ReplValue::Int(3)));
    assert!(result.calls.is_empty());
    assert_eq!(result.final_answer, None);
}

#[test]
fn variables_persist_across_cells() {
    let mut s = session();

    let first = s.eval_cell("let counter = 5; counter").expect("cell 1");
    assert_eq!(first.value, Some(ReplValue::Int(5)));
    assert!(first.variables_changed.contains(&"counter".to_string()));

    // The binding from cell 1 is visible in cell 2.
    let second = s.eval_cell("counter + 1").expect("cell 2");
    assert_eq!(second.value, Some(ReplValue::Int(6)));

    // And can be reassigned, persisting again. `counter` is still 5 (cell 2 did
    // not mutate it), so doubling yields 10.
    let third = s
        .eval_cell("counter = counter * 2; counter")
        .expect("cell 3");
    assert_eq!(third.value, Some(ReplValue::Int(10)));

    // The reassignment persists into a fourth cell.
    let fourth = s.eval_cell("counter").expect("cell 4");
    assert_eq!(fourth.value, Some(ReplValue::Int(10)));
}

#[test]
fn variables_changed_diffs_against_shared_baseline() {
    // The pre-cell baseline is snapshotted once into the shared `vars_snapshot`
    // and the change diff reads it back from there (no second retained copy).
    // A cell that introduces several new bindings and mutates an existing one
    // must still report every changed name, and leave an unchanged binding out.
    let mut s = session();
    s.eval_cell("let kept = 1; let touched = 2;").expect("seed");

    let result = s
        .eval_cell("let a = 10; let b = 20; touched = 99; kept")
        .expect("cell");

    assert!(result.variables_changed.contains(&"a".to_string()));
    assert!(result.variables_changed.contains(&"b".to_string()));
    assert!(result.variables_changed.contains(&"touched".to_string()));
    assert!(
        !result.variables_changed.contains(&"kept".to_string()),
        "an unmodified binding is not reported as changed"
    );
}

#[test]
fn over_limit_script_fails_closed() {
    // A tiny operation budget makes an otherwise-bounded loop trip the limit.
    let policy = ReplPolicy {
        max_operations: 100,
        ..ReplPolicy::default()
    };
    let mut s = ReplSession::<()>::new().with_policy(policy);

    let err = s
        .eval_cell("let total = 0; for i in 0..1000000 { total += i; } total")
        .expect_err("should exceed the operation limit");

    match err {
        TinyAgentsError::LimitExceeded(msg) => {
            assert!(msg.contains("operation limit"), "unexpected message: {msg}");
        }
        other => panic!("expected LimitExceeded, got {other:?}"),
    }
}

#[test]
fn timeout_fails_closed_on_a_runaway_script() {
    // Regression test: `ReplPolicy::timeout` used to be parsed but never
    // enforced — a runaway or hanging cell could block the session forever.
    // `max_operations` is left effectively unbounded here so only the
    // wall-clock deadline (enforced via the engine's `on_progress` hook) can
    // stop the loop.
    let policy = ReplPolicy {
        timeout: Some(Duration::from_millis(30)),
        max_operations: 0,
        ..ReplPolicy::default()
    };
    let mut s = ReplSession::<()>::new().with_policy(policy);

    let start = Instant::now();
    let err = s
        .eval_cell("let total = 0; loop { total += 1; }")
        .expect_err("should exceed the wall-clock deadline");
    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    // The property under test: `eval_cell` returns promptly once the
    // deadline elapses rather than running the loop forever.
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "eval_cell took {:?}, should have returned near the 30ms deadline",
        start.elapsed()
    );
}

#[test]
fn max_iterations_limit_fails_closed() {
    // Regression test: `ReplPolicy::max_iterations` was parsed and defaulted
    // but no code path ever checked it.
    let policy = ReplPolicy {
        max_iterations: 2,
        ..ReplPolicy::default()
    };
    let mut s = ReplSession::<()>::new().with_policy(policy);

    s.eval_cell("1").expect("cell 1 within the limit");
    s.eval_cell("2").expect("cell 2 within the limit");

    let err = s.eval_cell("3").expect_err("cell 3 exceeds max_iterations");
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

#[test]
fn script_byte_limit_fails_closed() {
    let policy = ReplPolicy {
        max_script_bytes: 8,
        ..ReplPolicy::default()
    };
    let mut s = ReplSession::<()>::new().with_policy(policy);

    let err = s
        .eval_cell("let a = 1234567890;")
        .expect_err("should exceed the script byte limit");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)));
}

#[test]
fn reserved_names_are_restored_after_each_cell() {
    let mut s = session();
    s.set_context(ReplValue::String("original".to_string()));

    // A cell may read and temporarily overwrite a reserved name.
    let result = s
        .eval_cell(r#"context = "tampered"; context"#)
        .expect("cell");
    assert_eq!(
        result.value,
        Some(ReplValue::String("tampered".to_string()))
    );

    // But the next cell sees the restored baseline, not the tampered value.
    let after = s.eval_cell("context").expect("read context");
    assert_eq!(after.value, Some(ReplValue::String("original".to_string())));

    // Reserved names never show up as persistent changed variables.
    assert!(!result.variables_changed.contains(&"context".to_string()));
}

#[test]
fn reserved_names_contains_no_duplicates() {
    // `answer` is a capability function (see RESERVED_FUNCTIONS), not a
    // readable session variable; it must not also appear in
    // RESERVED_VARIABLES, or `ReplVariables::seeded` double-pushes the same
    // scope entry.
    let names: Vec<&str> = reserved_names().collect();
    let mut seen = std::collections::HashSet::new();
    for name in &names {
        assert!(seen.insert(*name), "duplicate reserved name: {name}");
    }
    assert!(names.contains(&"answer"));
}

#[test]
fn answer_variable_is_seeded_exactly_once_in_scope() {
    let s = session();
    let count = s
        .variables
        .scope
        .iter()
        .filter(|(name, _, _)| *name == "answer")
        .count();
    assert_eq!(count, 1, "`answer` must be seeded into scope exactly once");
}

#[test]
fn reserved_capability_name_cannot_be_set_as_a_variable() {
    let mut s = session();
    let err = s
        .variables
        .set("model_query", ReplValue::Int(1))
        .expect_err("reserved name");
    assert!(matches!(err, TinyAgentsError::Capability(_)));
}

#[test]
fn print_is_captured_as_stdout() {
    let mut s = session();
    let result = s
        .eval_cell(r#"print("hello"); print("world");"#)
        .expect("cell");
    assert_eq!(result.stdout, "hello\nworld\n");
}

#[test]
fn emit_records_a_call() {
    let mut s = session();
    let result = s
        .eval_cell(r#"emit("found", #{ count: 3 }); 1"#)
        .expect("cell");
    assert_eq!(result.calls.len(), 1);
    let call = &result.calls[0];
    assert_eq!(call.kind, ReplCallKind::Emit);
    assert_eq!(call.name, "found");
    assert_eq!(call.detail, serde_json::json!({ "count": 3 }));
}

#[test]
fn answer_records_the_final_answer() {
    let mut s = session();
    let result = s
        .eval_cell(r#"answer("escalate to a human"); ()"#)
        .expect("cell");
    assert_eq!(result.final_answer, Some("escalate to a human".to_string()));
}

#[test]
fn output_byte_limit_fails_closed() {
    let policy = ReplPolicy {
        max_output_bytes: 4,
        ..ReplPolicy::default()
    };
    let mut s = ReplSession::<()>::new().with_policy(policy);

    let err = s
        .eval_cell(r#"print("this is definitely longer than four bytes");"#)
        .expect_err("should exceed the output byte limit");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)));
}

#[test]
fn output_byte_limit_bounds_intra_cell_buffering_in_a_print_loop() {
    // A script that prints in a tight loop must not be allowed to buffer
    // unbounded output before the limit is noticed: push_stdout_line itself
    // must stop growing the buffer (and eval_cell must fail closed) well
    // before the loop's total output would otherwise reach many times the
    // configured budget.
    let policy = ReplPolicy {
        max_output_bytes: 100,
        max_operations: 1_000_000,
        ..ReplPolicy::default()
    };
    let mut s = ReplSession::<()>::new().with_policy(policy);

    let err = s
        .eval_cell(
            r#"for i in 0..100000 { print("0123456789012345678901234567890123456789012345"); }"#,
        )
        .expect_err("should exceed the output byte limit");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)), "{err:?}");
}

#[test]
fn graph_define_does_not_consume_the_limit_on_a_failed_draft() {
    // A `graph_define` call whose source parses but names a graph that isn't
    // in the source must not consume a definition slot: only a successfully
    // recorded draft should count against `max_graph_definitions`.
    let policy = ReplPolicy {
        max_graph_definitions: 1,
        ..ReplPolicy::default()
    };
    let mut s = ReplSession::<()>::new().with_policy(policy);

    let source = r#"graph g { start a node a { kind model next END } }"#;

    // First call: wrong graph name, so the draft is never recorded — this
    // must fail without spending the one available slot.
    let bad = s.eval_cell(&format!(
        r#"graph_define(#{{ name: "missing", source: `{source}` }})"#
    ));
    assert!(
        bad.is_err(),
        "expected a failure for the unknown graph name"
    );

    // Second call: the correct graph name must still succeed, proving the
    // failed attempt above did not consume the definition budget.
    let good = s
        .eval_cell(&format!(
            r#"graph_define(#{{ name: "g", source: `{source}` }})"#
        ))
        .expect("a valid graph_define should still have a slot available");
    assert!(good.value.is_some());

    // A third attempt now must fail: the one slot has genuinely been spent.
    let over_limit = s.eval_cell(&format!(
        r#"graph_define(#{{ name: "g", source: `{source}` }})"#
    ));
    assert!(
        over_limit.is_err(),
        "the definition limit must be enforced once a slot is actually consumed"
    );
}

/// A tool that succeeds for every call except one whose `arguments.id`
/// matches `fail_id`, for which it returns a *tool-reported* error (a
/// `ToolResult` with `error: Some(..)`), not a `Result::Err` — exercising the
/// per-item error path distinct from a harness/transport-level failure.
struct SometimesFailingTool {
    fail_id: String,
}

#[async_trait::async_trait]
impl crate::harness::tool::Tool<()> for SometimesFailingTool {
    fn name(&self) -> &str {
        "sometimes_fails"
    }

    fn description(&self) -> &str {
        "Succeeds unless called with the configured failing id."
    }

    fn schema(&self) -> crate::harness::tool::ToolSchema {
        crate::harness::tool::ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: serde_json::json!({ "type": "object" }),
            format: Default::default(),
        }
    }

    async fn call(
        &self,
        _state: &(),
        call: crate::harness::tool::ToolCall,
    ) -> crate::Result<crate::harness::tool::ToolResult> {
        let id = call
            .arguments
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if id == self.fail_id {
            Ok(crate::harness::tool::ToolResult::error(
                call.id,
                call.name,
                format!("tool reported an error for id {id}"),
            ))
        } else {
            Ok(crate::harness::tool::ToolResult::text(
                call.id,
                call.name,
                format!("ok:{id}"),
            ))
        }
    }
}

fn session_with_sometimes_failing_tool(fail_id: &str) -> ReplSession {
    let mut registry = crate::registry::CapabilityRegistry::<()>::new();
    registry
        .register_tool(std::sync::Arc::new(SometimesFailingTool {
            fail_id: fail_id.to_string(),
        }))
        .expect("register tool");
    let capabilities = ReplCapabilities::new(std::sync::Arc::new(registry));
    ReplSession::<()>::new().with_capabilities(capabilities)
}

#[test]
fn tool_call_batched_keeps_successes_when_one_item_tool_errors() {
    // Regression test: a per-item *tool-reported* error (ToolResult::error,
    // as opposed to a harness/transport-level Err) used to abort the whole
    // batch, discarding every other item's already-computed successful
    // result. Each item's outcome must be reported independently.
    let mut s = session_with_sometimes_failing_tool("2");

    let script = r#"
        tool_call_batched([
            #{ tool: "sometimes_fails", arguments: #{ id: "1" } },
            #{ tool: "sometimes_fails", arguments: #{ id: "2" } },
            #{ tool: "sometimes_fails", arguments: #{ id: "3" } },
        ])
    "#;
    let result = s.eval_cell(script).expect("batch call should not abort");
    let value = result.value.expect("value").to_json();
    let items = value.as_array().expect("array result");
    assert_eq!(items.len(), 3, "{items:?}");

    assert_eq!(items[0]["ok"], serde_json::json!(true));
    assert_eq!(items[0]["content"], serde_json::json!("ok:1"));

    assert_eq!(items[1]["ok"], serde_json::json!(false));
    assert!(
        items[1]["error"]
            .as_str()
            .unwrap()
            .contains("tool reported an error"),
        "{items:?}"
    );

    assert_eq!(items[2]["ok"], serde_json::json!(true));
    assert_eq!(items[2]["content"], serde_json::json!("ok:3"));
}

/// A trivial [`HarnessAgent`] that returns a fixed response, for exercising
/// `agent_query` without a real model/harness run.
struct StubAgent;

#[async_trait::async_trait]
impl crate::graph::subagent_node::HarnessAgent for StubAgent {
    fn name(&self) -> &str {
        "stub"
    }

    async fn run(
        &self,
        input: crate::graph::subagent_node::SubAgentInput,
        _events: crate::harness::events::EventSink,
    ) -> crate::Result<crate::graph::subagent_node::SubAgentOutput> {
        Ok(crate::graph::subagent_node::SubAgentOutput {
            text: format!("stub replied to: {}", input.prompt),
            ..Default::default()
        })
    }
}

fn session_with_stub_agent(policy: ReplPolicy) -> ReplSession {
    let mut registry = crate::registry::CapabilityRegistry::<()>::new();
    registry
        .register_agent(std::sync::Arc::new(StubAgent))
        .expect("register stub agent");
    let capabilities = ReplCapabilities::new(std::sync::Arc::new(registry));
    ReplSession::<()>::new()
        .with_policy(policy)
        .with_capabilities(capabilities)
}

#[test]
fn agent_call_limit_is_independent_of_the_model_call_limit() {
    // Regression test: `bump_agent` used to compare the agent-call counter
    // against `max_model_calls` (with an "agent call limit" message quoting
    // that same number), so a session's *combined* model spend — direct
    // `model_query` calls plus every model call a delegated `agent_query`
    // itself drives — could reach roughly twice the configured
    // `max_model_calls` before anything failed closed. `max_agent_calls` is
    // now tracked and enforced independently.
    let policy = ReplPolicy {
        max_model_calls: 64,
        max_agent_calls: 2,
        ..ReplPolicy::default()
    };
    let mut s = session_with_stub_agent(policy);

    let script = r#"agent_query(#{ agent: "stub", prompt: "hi" })"#;
    s.eval_cell(script).expect("call 1 within the limit");
    s.eval_cell(script).expect("call 2 within the limit");

    let err = s
        .eval_cell(script)
        .expect_err("call 3 exceeds max_agent_calls");
    match err {
        TinyAgentsError::LimitExceeded(msg) => {
            assert!(
                msg.contains("agent call limit (2)"),
                "expected the message to cite max_agent_calls (2), got: {msg}"
            );
        }
        other => panic!("expected LimitExceeded, got {other:?}"),
    }
}

// ── External cancellation (Phase 2) ──────────────────────────────────────────

/// A tool whose call never resolves, modelling a hung provider/tool so a test
/// can prove external cancellation drops the in-flight future via the blocking
/// bridge rather than blocking the session forever.
struct HangingTool;

#[async_trait::async_trait]
impl crate::harness::tool::Tool<()> for HangingTool {
    fn name(&self) -> &str {
        "hangs"
    }

    fn description(&self) -> &str {
        "Never returns; used to test the cancellation bridge."
    }

    fn schema(&self) -> crate::harness::tool::ToolSchema {
        crate::harness::tool::ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: serde_json::json!({ "type": "object" }),
            format: Default::default(),
        }
    }

    async fn call(
        &self,
        _state: &(),
        _call: crate::harness::tool::ToolCall,
    ) -> crate::Result<crate::harness::tool::ToolResult> {
        futures::future::pending().await
    }
}

fn session_with_hanging_tool(policy: ReplPolicy) -> ReplSession {
    let mut registry = crate::registry::CapabilityRegistry::<()>::new();
    registry
        .register_tool(std::sync::Arc::new(HangingTool))
        .expect("register hanging tool");
    let capabilities = ReplCapabilities::new(std::sync::Arc::new(registry));
    ReplSession::<()>::new()
        .with_policy(policy)
        .with_capabilities(capabilities)
}

#[test]
fn cancel_set_before_eval_fails_closed_without_running_the_cell() {
    // A flag tripped before the cell starts short-circuits to `Cancelled`, and
    // must not consume the session's iteration budget: a later cell with a
    // fresh flag still runs.
    let flag = ReplCancelFlag::new();
    let mut s = session().with_cancel_flag(flag.clone());
    flag.cancel();

    let err = s.eval_cell("let x = 1; x").expect_err("pre-cancelled");
    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");

    // A fresh flag re-enables the session — the cancelled cell did not burn an
    // iteration or otherwise poison the namespace.
    let mut s = s.with_cancel_flag(ReplCancelFlag::new());
    let ok = s.eval_cell("1 + 1").expect("session usable after cancel");
    assert_eq!(ok.value, Some(ReplValue::Int(2)));
}

#[test]
fn cancel_mid_script_loop_terminates_promptly() {
    // A pure `loop {}` with no timeout and an unbounded operation budget can
    // only be stopped by the cancel flag, enforced via the `on_progress` hook.
    let policy = ReplPolicy {
        timeout: None,
        max_operations: 0,
        ..ReplPolicy::default()
    };
    let flag = ReplCancelFlag::new();
    let mut s = session().with_policy(policy).with_cancel_flag(flag.clone());

    let trigger = flag.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(40));
        trigger.cancel();
    });

    let start = Instant::now();
    let err = s
        .eval_cell("let total = 0; loop { total += 1; }")
        .expect_err("cancel must terminate the loop");
    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "eval_cell took {:?}, should return near the ~40ms cancel",
        start.elapsed()
    );
}

#[test]
fn cancel_during_a_hanging_capability_call_terminates_promptly() {
    // A tool call that never resolves can only be released by the cancel flag,
    // enforced via the blocking bridge (`on_progress` never fires inside a
    // blocked native call). No timeout is configured so only cancel can stop it.
    let policy = ReplPolicy {
        timeout: None,
        ..ReplPolicy::default()
    };
    let flag = ReplCancelFlag::new();
    let mut s = session_with_hanging_tool(policy).with_cancel_flag(flag.clone());

    let trigger = flag.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(40));
        trigger.cancel();
    });

    let start = Instant::now();
    let err = s
        .eval_cell(r#"tool_call(#{ tool: "hangs" })"#)
        .expect_err("cancel must drop the hung tool future");
    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "eval_cell took {:?}, should return near the ~40ms cancel",
        start.elapsed()
    );
}

// ── Live capability-call events (Phase 2) ────────────────────────────────────

#[test]
fn capability_calls_stream_started_and_completed_events() {
    use crate::harness::events::{AgentEvent, ReplCallPhase};

    // `fail_id` "none" never matches, so the "1" call succeeds.
    let mut s = session_with_sometimes_failing_tool("none");
    let recorder = std::sync::Arc::new(crate::harness::events::RecordingListener::new());
    s.events.subscribe(recorder.clone());

    let session_label = s.session_id.as_str().to_string();
    s.eval_cell(r#"tool_call(#{ tool: "sometimes_fails", arguments: #{ id: "1" } })"#)
        .expect("tool call");

    // Exactly one Started and one Completed ReplCall event, paired by call_id,
    // both naming the tool and correlated back to this session.
    let repl_calls: Vec<(ReplCallPhase, String, String)> = recorder
        .events()
        .into_iter()
        .filter_map(|rec| match rec.event {
            AgentEvent::ReplCall {
                session_id,
                record,
                phase,
            } => Some((phase, session_id, record.call_id.as_str().to_string())),
            _ => None,
        })
        .collect();

    assert_eq!(
        repl_calls.len(),
        2,
        "expected start + completion: {repl_calls:?}"
    );
    assert_eq!(repl_calls[0].0, ReplCallPhase::Started);
    assert_eq!(repl_calls[1].0, ReplCallPhase::Completed);
    assert_eq!(repl_calls[0].1, session_label, "session_id correlates");
    assert_eq!(repl_calls[1].1, session_label);
    assert_eq!(
        repl_calls[0].2, repl_calls[1].2,
        "start and completion share one call_id"
    );
}

#[test]
fn map_and_array_values_round_trip_to_json() {
    let mut s = session();
    let result = s.eval_cell(r#"#{ a: 1, b: [true, "x"] }"#).expect("cell");
    let value = result.value.expect("value");
    assert_eq!(
        value.to_json(),
        serde_json::json!({ "a": 1, "b": [true, "x"] })
    );
}

#[test]
fn syntax_error_maps_to_validation() {
    let mut s = session();
    let err = s.eval_cell("let = ;").expect_err("syntax error");
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[test]
fn variables_helper_reads_persistent_value() {
    let mut s = session();
    s.eval_cell("let note = \"hi\";").expect("cell");
    assert_eq!(
        s.variables.get("note"),
        Some(ReplValue::String("hi".to_string()))
    );
}

#[test]
fn default_policy_has_review_gate_enabled() {
    let policy = ReplPolicy::default();
    assert!(policy.generated_graphs_require_review);
    assert_eq!(policy.max_depth, 8);
}

#[test]
fn capabilities_expose_registered_names() {
    let caps = ReplCapabilities::<()>::default();
    assert!(caps.models().is_empty());
    assert!(caps.tools().is_empty());
    assert!(caps.language.is_none());
}
