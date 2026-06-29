//! Tests for the default agent loop.
//!
//! These exercise the loop end to end with [`MockModel`] and a local
//! `FakeTool`: a single text response, a multi-step tool loop, limit
//! enforcement, middleware request mutation, usage accumulation, the
//! tool-not-found path, structured output extraction, and retry/fallback.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::json;

use crate::error::{Result, TinyAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::limits::RunLimits;
use crate::harness::message::{AssistantMessage, ContentBlock, Message};
use crate::harness::middleware::{AgentRun, Middleware};
use crate::harness::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ResponseFormat, ToolChoice,
};
use crate::harness::providers::MockModel;
use crate::harness::retry::{FallbackPolicy, RetryPolicy};
use crate::harness::runtime::{AgentHarness, RunPolicy};
use crate::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};
use crate::harness::usage::Usage;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// A tool that records its invocations and returns a fixed reply.
struct FakeTool {
    name: &'static str,
    reply: &'static str,
    calls: Mutex<usize>,
}

impl FakeTool {
    fn new(name: &'static str, reply: &'static str) -> Self {
        Self {
            name,
            reply,
            calls: Mutex::new(0),
        }
    }
}

#[async_trait]
impl Tool<()> for FakeTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "fake tool"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name, "fake tool", json!({"type": "object"}))
    }
    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::text(call.id, self.name, self.reply))
    }
}

/// Builds a tool-call assistant response (no text, one tool call).
fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// Builds a plain-text assistant response with explicit usage.
fn text_response(text: &str, input: u64, output: u64) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(input, output)),
        },
        usage: Some(Usage::new(input, output)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// Middleware that appends a user message to every model request.
struct InjectMiddleware {
    text: &'static str,
}

#[async_trait]
impl Middleware<(), ()> for InjectMiddleware {
    fn name(&self) -> &str {
        "inject"
    }
    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.messages.push(Message::user(self.text));
        // Also flip tool choice so we can assert mutation visibly.
        request.tool_choice = ToolChoice::None;
        Ok(())
    }
}

/// A model whose profile lacks native structured output, so `Auto` should
/// resolve to the tool-call strategy. It answers with a tool call named after
/// the artificial structured tool the loop appends, carrying the structured
/// arguments.
struct ToolStructuredModel {
    profile: ModelProfile,
}

impl ToolStructuredModel {
    fn new() -> Self {
        Self {
            profile: ModelProfile {
                tool_calling: true,
                native_structured_output: false,
                json_schema: false,
                ..ModelProfile::default()
            },
        }
    }
}

#[async_trait]
impl ChatModel<()> for ToolStructuredModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    async fn invoke(&self, _state: &(), request: ModelRequest) -> Result<ModelResponse> {
        // The loop appends an artificial structured tool and forces the choice
        // to it; the tool name is the schema name.
        assert_eq!(request.tool_choice, ToolChoice::Tool("answer".to_string()));
        let name = request
            .tools
            .last()
            .map(|t| t.name.clone())
            .unwrap_or_default();
        Ok(tool_call_response(
            "s1",
            &name,
            json!({"value":"viatool","score":7}),
        ))
    }
}

/// A model that always fails with a retryable error and counts attempts.
struct FailingModel {
    attempts: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for FailingModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Err(TinyAgentsError::Model("transient boom".to_string()))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn single_model_call_no_tools() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hello there")));

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 1);
    assert_eq!(run.tool_calls, 0);
    assert_eq!(run.steps, 1);
    assert_eq!(run.text(), Some("hello there".to_string()));
    // input user + assistant reply.
    assert_eq!(run.messages.len(), 2);
}

#[tokio::test]
async fn model_requests_tool_then_finishes() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "lookup", json!({"q": "x"})),
            text_response("done", 4, 2),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::new("lookup", "tool-output")));

    let run = harness
        .invoke_default(&(), vec![Message::user("please look up")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 2);
    assert_eq!(run.tool_calls, 1);
    assert_eq!(run.steps, 2);
    assert_eq!(run.text(), Some("done".to_string()));
    // user, assistant(tool call), tool result, assistant(final).
    assert_eq!(run.messages.len(), 4);
    assert!(matches!(run.messages[2], Message::Tool(_)));
    assert_eq!(run.messages[2].text(), "tool-output");
}

#[tokio::test]
async fn max_model_calls_limit_triggers_limit_exceeded() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    // A model that always asks for the tool -> the loop would never stop.
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("spin", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::new("spin", "again")));
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_calls(1),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("limit should be exceeded");
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn max_tool_calls_limit_triggers_limit_exceeded() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("spin", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::new("spin", "again")));
    harness.with_policy(RunPolicy {
        limits: RunLimits::default()
            .with_max_model_calls(10)
            .with_max_tool_calls(0),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("tool limit should be exceeded");
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn before_model_middleware_mutates_request() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    // Echo returns the last user message; the middleware injects one.
    harness.register_model("mock", Arc::new(MockModel::echo()));
    harness.push_middleware(Arc::new(InjectMiddleware { text: "injected" }));

    let run = harness
        .invoke_default(&(), vec![Message::user("original")])
        .await
        .expect("run succeeds");

    assert_eq!(run.text(), Some("injected".to_string()));
}

#[tokio::test]
async fn usage_accumulates_across_calls() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "lookup", json!({})),
            text_response("done", 4, 2),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::new("lookup", "out")));

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(run.usage.calls, 2);
    // tool-call response: 7 in / 3 out; text response: 4 in / 2 out.
    assert_eq!(run.usage.usage.input_tokens, 11);
    assert_eq!(run.usage.usage.output_tokens, 5);
}

#[tokio::test]
async fn tool_not_found_errors() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("missing", json!({}))),
    );
    // No tool registered.

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("tool should be missing");
    match err {
        TinyAgentsError::ToolNotFound(name) => assert_eq!(name, "missing"),
        other => panic!("expected ToolNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn structured_output_is_extracted() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::constant(r#"{"value":"hi","score":42}"#)),
    );
    harness.with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::json_schema(
            "answer",
            json!({"type": "object"}),
        )),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("answer")])
        .await
        .expect("run succeeds");

    let structured = run.structured.expect("structured output present");
    assert_eq!(structured["value"], "hi");
    assert_eq!(structured["score"], 42);
}

#[tokio::test]
async fn auto_format_uses_provider_schema_for_native_model() {
    // MockModel advertises a permissive profile (native structured output), so
    // `Auto` resolves to provider-native schema mode and parses the JSON text.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::constant(r#"{"value":"native","score":1}"#)),
    );
    harness.with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::auto("answer", json!({"type": "object"}))),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("answer")])
        .await
        .expect("run succeeds");

    let structured = run.structured.expect("structured output present");
    assert_eq!(structured["value"], "native");
}

#[tokio::test]
async fn auto_format_uses_tool_call_for_non_native_model() {
    // A model without native structured output drives `Auto` down the tool-call
    // fallback; the structured value is read from the tool-call arguments and
    // the artificial tool call is treated as the final response.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("tool", Arc::new(ToolStructuredModel::new()));
    harness.with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::auto("answer", json!({"type": "object"}))),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("answer")])
        .await
        .expect("run succeeds");

    let structured = run.structured.expect("structured output present");
    assert_eq!(structured["value"], "viatool");
    assert_eq!(structured["score"], 7);
    // Exactly one model call: the structured tool call ends the loop.
    assert_eq!(run.model_calls, 1);
}

#[tokio::test]
async fn no_model_registered_errors() {
    let harness: AgentHarness<()> = AgentHarness::new();
    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("no model");
    assert!(
        matches!(err, TinyAgentsError::ModelNotFound(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn retry_then_fallback_succeeds() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let failing = Arc::new(FailingModel {
        attempts: Mutex::new(0),
    });
    harness.register_model("primary", failing.clone());
    harness.register_model("backup", Arc::new(MockModel::constant("recovered")));
    harness.with_policy(RunPolicy {
        // 2 attempts on primary, then fall back to backup.
        retry: RetryPolicy::default().with_max_attempts(2),
        fallback: Some(FallbackPolicy::new(["primary", "backup"])),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("fallback recovers");

    assert_eq!(run.text(), Some("recovered".to_string()));
    // Primary tried max_attempts (2) times before falling back.
    assert_eq!(*failing.attempts.lock().unwrap(), 2);
}

#[tokio::test]
async fn non_retryable_or_exhausted_without_fallback_errors() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "primary",
        Arc::new(FailingModel {
            attempts: Mutex::new(0),
        }),
    );
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(1),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("no fallback, error propagates");
    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
}

#[tokio::test]
async fn invoke_with_status_reports_completed() {
    use crate::harness::ids::{ExecutionStatus, HarnessPhase};

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("ok")));

    let result = harness
        .invoke_with_status(&(), (), RunConfig::new("run-x"), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(result.status.status, ExecutionStatus::Completed);
    assert_eq!(result.status.current_phase, HarnessPhase::Done);
    assert_eq!(result.status.model_calls, 1);
    assert_eq!(result.run.text(), Some("ok".to_string()));
}

// Touch `AgentRun` constructor so the import is meaningful even if all
// assertions above use returned runs.
#[test]
fn agent_run_default_is_empty() {
    let run = AgentRun::new();
    assert_eq!(run.model_calls, 0);
}
