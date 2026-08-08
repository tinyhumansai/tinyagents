//! Regression coverage for the wave-2 tool-execution defects in
//! `harness::agent_loop::tools`.
//!
//! Each test pins one defect:
//!
//! | Test | Defect |
//! |------|--------|
//! | `tool_result_call_id_is_overwritten_with_the_admitted_id` | TOOL-1: the transcript took the tool's `call_id`, not the admitted one |
//! | `concurrent_admission_failure_emits_no_tool_started` | TOOL-3: `ToolStarted` was emitted for calls that never ran |
//! | `serial_tool_error_becomes_a_model_visible_result` | TOOL-5 (serial): an `Err` killed the run |
//! | `concurrent_tool_error_becomes_a_model_visible_result` | TOOL-5 (concurrent) |
//! | `fatal_tool_error_emits_tool_failed_and_clears_active_calls` | TOOL-6 |
//! | `duplicate_call_ids_do_not_clear_each_others_active_entry` | TOOL-10 |
//! | `unknown_tool_recovery_emits_started_and_completed` | TOOL-11 |
//! | `before_tool_rejection_does_not_consume_a_tool_call_slot` | TOOL-12 |
//! | `injected_arguments_are_stripped_before_validation` | injected-argument enforcement |

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};

use tinyagents::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::AgentEvent;
use tinyagents::harness::limits::RunLimits;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::middleware::Middleware;
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy, UnknownToolPolicy};
use tinyagents::harness::testkit::EventRecorder;
use tinyagents::harness::tool::{Tool, ToolCall, ToolErrorPolicy, ToolResult, ToolSchema};
use tinyagents::harness::usage::Usage;

// ── Scripted model helpers ────────────────────────────────────────────────────

/// Builds an assistant turn that requests `calls`.
///
/// Deliberately built from [`ModelResponse::assistant`] rather than a struct
/// literal so the fixture survives new fields being added to `ModelResponse`.
fn tool_calls_response(calls: Vec<ToolCall>) -> ModelResponse {
    let mut response = ModelResponse::assistant("");
    response.message.content = Vec::new();
    response.message.tool_calls = calls;
    response.finish_reason = Some("tool_calls".into());
    response
}

fn text_response(text: &str) -> ModelResponse {
    let mut response = ModelResponse::assistant(text);
    response.finish_reason = Some("stop".into());
    response
}

fn empty_object_schema(name: &str) -> ToolSchema {
    ToolSchema::new(name, "test tool", json!({ "type": "object" }))
}

// ── Test tools ────────────────────────────────────────────────────────────────

/// A third-party tool that stamps its own (wrong) `call_id` on the result.
struct WrongCallIdTool;

#[async_trait]
impl Tool<()> for WrongCallIdTool {
    fn name(&self) -> &str {
        "wrong_id"
    }
    fn description(&self) -> &str {
        "returns a result carrying a hard-coded call id"
    }
    fn schema(&self) -> ToolSchema {
        empty_object_schema("wrong_id")
    }
    async fn call(&self, _state: &(), _call: ToolCall) -> tinyagents::Result<ToolResult> {
        // The classic third-party mistake: `call_id` is the first positional
        // argument, so a hard-coded or empty string slips in unnoticed.
        Ok(ToolResult::text("", "wrong_id", "ok"))
    }
}

/// A tool that always returns `Err`, with a configurable error policy.
struct ErringTool {
    name: String,
    policy: ToolErrorPolicy,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool<()> for ErringTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "always fails"
    }
    fn schema(&self) -> ToolSchema {
        empty_object_schema(&self.name)
    }
    fn error_policy(&self) -> ToolErrorPolicy {
        self.policy.clone()
    }
    async fn call(&self, _state: &(), _call: ToolCall) -> tinyagents::Result<ToolResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(TinyAgentsError::Tool("transient 503".into()))
    }
}

/// Records the arguments it was invoked with, and declares one injected key.
struct InjectedArgTool {
    seen: Arc<std::sync::Mutex<Vec<Value>>>,
}

#[async_trait]
impl Tool<()> for InjectedArgTool {
    fn name(&self) -> &str {
        "injected"
    }
    fn description(&self) -> &str {
        "declares a host-injected argument"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "injected",
            "declares a host-injected argument",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "thread_id": { "type": "string" }
                },
                "required": ["query", "thread_id"],
                "additionalProperties": false
            }),
        )
    }
    fn injected_arguments(&self) -> &[&str] {
        &["thread_id"]
    }
    async fn call(&self, _state: &(), call: ToolCall) -> tinyagents::Result<ToolResult> {
        self.seen.lock().unwrap().push(call.arguments.clone());
        Ok(ToolResult::text(call.id, "injected", "ok"))
    }
}

/// A plain echo tool used to fill multi-call turns.
struct EchoTool {
    name: String,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool<()> for EchoTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "echoes"
    }
    fn schema(&self) -> ToolSchema {
        empty_object_schema(&self.name)
    }
    async fn call(&self, _state: &(), call: ToolCall) -> tinyagents::Result<ToolResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::text(call.id, self.name.clone(), "echo"))
    }
}

/// Middleware that rejects every tool call from `before_tool`.
struct RejectingMiddleware;

#[async_trait]
impl Middleware<(), ()> for RejectingMiddleware {
    fn name(&self) -> &str {
        "rejecting"
    }
    async fn before_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _call: &mut ToolCall,
    ) -> tinyagents::Result<()> {
        Err(TinyAgentsError::Middleware("approval denied".into()))
    }
}

// ── Event helpers ─────────────────────────────────────────────────────────────

fn started_call_ids(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolStarted { call_id, .. } => Some(call_id.as_str().to_string()),
            _ => None,
        })
        .collect()
}

fn completed_call_ids(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCompleted { call_id, .. } => Some(call_id.as_str().to_string()),
            _ => None,
        })
        .collect()
}

fn tool_message_ids(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool(tool) => Some(tool.tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

// ── TOOL-1 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tool_result_call_id_is_overwritten_with_the_admitted_id() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_calls_response(vec![ToolCall::new("call_abc", "wrong_id", json!({}))]),
            text_response("done"),
        ])),
    );
    harness.register_tool(Arc::new(WrongCallIdTool));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run should complete");

    assert_eq!(
        tool_message_ids(&run.messages),
        vec!["call_abc".to_string()],
        "the transcript must answer the admitted tool_call_id, not the id the tool stamped"
    );
}

// ── TOOL-3 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn concurrent_admission_failure_emits_no_tool_started() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![tool_calls_response(vec![
            ToolCall::new("c1", "echo_a", json!({})),
            ToolCall::new("c2", "echo_b", json!({})),
            ToolCall::new("c3", "echo_c", json!({})),
        ])])),
    );
    for name in ["echo_a", "echo_b", "echo_c"] {
        harness.register_tool(Arc::new(EchoTool {
            name: name.to_string(),
            calls: calls.clone(),
        }));
    }

    let recorder = EventRecorder::new();
    let config = RunConfig::new()
        .with_events(recorder.sink())
        .with_limits(RunLimits::new().with_max_tool_calls(2));
    let mut ctx = RunContext::new(config);

    let err = harness
        .invoke(&(), vec![Message::user("go")], &mut ctx)
        .await
        .expect_err("the third call must trip the tool-call cap");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)), "{err:?}");

    let events = recorder.events();
    assert!(
        started_call_ids(&events).is_empty(),
        "no ToolStarted may be emitted for calls that never run: {:?}",
        started_call_ids(&events)
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no tool may execute when admission fails"
    );
}

// ── TOOL-5 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn serial_tool_error_becomes_a_model_visible_result() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_calls_response(vec![ToolCall::new("c1", "flaky", json!({}))]),
            text_response("recovered"),
        ])),
    );
    harness.register_tool(Arc::new(ErringTool {
        name: "flaky".into(),
        policy: ToolErrorPolicy::ReturnToError,
        calls: calls.clone(),
    }));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("a ReturnToError tool failure must not kill the run");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let tool_text = run
        .messages
        .iter()
        .find_map(|m| match m {
            Message::Tool(t) => Some(t.content.clone()),
            _ => None,
        })
        .expect("a tool message should have been appended");
    let rendered = format!("{tool_text:?}");
    assert!(
        rendered.contains("transient 503"),
        "the model should see the tool error: {rendered}"
    );
}

#[tokio::test]
async fn concurrent_tool_error_becomes_a_model_visible_result() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_calls_response(vec![
                ToolCall::new("c1", "flaky", json!({})),
                ToolCall::new("c2", "echo_a", json!({})),
            ]),
            text_response("recovered"),
        ])),
    );
    harness.register_tool(Arc::new(ErringTool {
        name: "flaky".into(),
        policy: ToolErrorPolicy::ReturnToError,
        calls: calls.clone(),
    }));
    harness.register_tool(Arc::new(EchoTool {
        name: "echo_a".into(),
        calls: calls.clone(),
    }));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("a ReturnToError tool failure must not kill a concurrent turn");

    assert_eq!(
        tool_message_ids(&run.messages),
        vec!["c1".to_string(), "c2".to_string()],
        "both calls must be answered, in original order"
    );
}

// ── TOOL-6 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fatal_tool_error_emits_tool_failed_and_clears_active_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![tool_calls_response(vec![
            ToolCall::new("c1", "fatal", json!({})),
        ])])),
    );
    harness.register_tool(Arc::new(ErringTool {
        name: "fatal".into(),
        policy: ToolErrorPolicy::Fail,
        calls: calls.clone(),
    }));

    let recorder = EventRecorder::new();
    let config = RunConfig::new().with_events(recorder.sink());
    let mut ctx = RunContext::new(config);

    harness
        .invoke(&(), vec![Message::user("go")], &mut ctx)
        .await
        .expect_err("a Fail-policy tool error must abort the run");

    let events = recorder.events();
    let failed: Vec<_> = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::ToolFailed { .. }))
        .collect();
    assert_eq!(
        failed.len(),
        1,
        "every ToolStarted needs a terminal partner; got kinds {:?}",
        events.iter().map(AgentEvent::kind).collect::<Vec<_>>()
    );
    match failed[0] {
        AgentEvent::ToolFailed {
            call_id,
            tool_name,
            error,
            ..
        } => {
            assert_eq!(call_id.as_str(), "c1");
            assert_eq!(tool_name, "fatal");
            assert!(error.contains("transient 503"), "{error}");
        }
        other => panic!("expected ToolFailed, got {other:?}"),
    }
}

// ── TOOL-10 ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn duplicate_call_ids_do_not_clear_each_others_active_entry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            // A provider that reuses one id across two calls in the same turn.
            tool_calls_response(vec![
                ToolCall::new("dup", "echo_a", json!({})),
                ToolCall::new("dup", "flaky", json!({})),
            ]),
            text_response("done"),
        ])),
    );
    harness.register_tool(Arc::new(EchoTool {
        name: "echo_a".into(),
        calls: calls.clone(),
    }));
    harness.register_tool(Arc::new(ErringTool {
        name: "flaky".into(),
        policy: ToolErrorPolicy::Fail,
        calls: calls.clone(),
    }));

    let recorder = EventRecorder::new();
    let config = RunConfig::new().with_events(recorder.sink());
    let mut ctx = RunContext::new(config);

    harness
        .invoke(&(), vec![Message::user("go")], &mut ctx)
        .await
        .expect_err("the second (failing) call aborts the run");

    // The first call completed, so exactly one of the two duplicate entries may
    // have been removed; the second is removed by its ToolFailed. A `retain`
    // that drops every match would have cleared both on the first completion,
    // leaving the failure path with nothing to clear.
    let status = ctx.status();
    assert!(
        status.active_tool_calls.is_empty(),
        "both duplicate entries must be accounted for: {:?}",
        status.active_tool_calls
    );
    let events = recorder.events();
    assert_eq!(completed_call_ids(&events).len(), 1);
}

// ── TOOL-11 ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_tool_recovery_emits_started_and_completed() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_calls_response(vec![ToolCall::new("c1", "missing", json!({}))]),
            text_response("recovered"),
        ])),
    );
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::ReturnToolError,
        ..RunPolicy::default()
    });

    let recorder = EventRecorder::new();
    let config = RunConfig::new().with_events(recorder.sink());
    let mut ctx = RunContext::new(config);

    harness
        .invoke(&(), vec![Message::user("go")], &mut ctx)
        .await
        .expect("ReturnToolError recovers");

    let events = recorder.events();
    assert_eq!(
        started_call_ids(&events),
        vec!["c1".to_string()],
        "a recovery must still open a tool span"
    );
    assert_eq!(
        completed_call_ids(&events),
        vec!["c1".to_string()],
        "a recovery must close its tool span"
    );
}

// ── TOOL-12 ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn before_tool_rejection_does_not_consume_a_tool_call_slot() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![tool_calls_response(vec![
            ToolCall::new("c1", "echo_a", json!({})),
        ])])),
    );
    harness.register_tool(Arc::new(EchoTool {
        name: "echo_a".into(),
        calls: calls.clone(),
    }));
    harness.register_middleware(Arc::new(RejectingMiddleware));

    let config = RunConfig::new().with_limits(RunLimits::new().with_max_tool_calls(4));
    let mut ctx = RunContext::new(config);

    let err = harness
        .invoke(&(), vec![Message::user("go")], &mut ctx)
        .await
        .expect_err("the rejecting middleware aborts the run");
    assert!(matches!(err, TinyAgentsError::Middleware(_)), "{err:?}");

    assert_eq!(
        ctx.limits().tool_calls(),
        0,
        "a call rejected before it ran must not burn a tool-call slot"
    );
}

// ── Injected-argument enforcement ─────────────────────────────────────────────

#[tokio::test]
async fn injected_arguments_are_stripped_before_validation() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_calls_response(vec![ToolCall::new(
                "c1",
                "injected",
                // The model forges the hidden key it was never shown.
                json!({ "query": "hi", "thread_id": "forged" }),
            )]),
            text_response("done"),
        ])),
    );
    harness.register_tool(Arc::new(InjectedArgTool { seen: seen.clone() }));

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("the call validates against the model-facing schema");

    let observed = seen.lock().unwrap().clone();
    assert_eq!(observed.len(), 1);
    assert!(
        observed[0].get("thread_id").is_none(),
        "a forged injected argument must be stripped: {:?}",
        observed[0]
    );
    assert_eq!(observed[0]["query"], "hi");
}
