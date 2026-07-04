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
use crate::harness::middleware::{
    AgentRun, Middleware, MiddlewareModelOutcome, MiddlewareToolOutcome, ModelHandler,
    ModelMiddleware, ToolHandler, ToolMiddleware,
};
use crate::harness::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ResponseFormat, ToolChoice,
};
use crate::harness::providers::MockModel;
use crate::harness::retry::{FallbackPolicy, RetryPolicy};
use crate::harness::runtime::{AgentHarness, RunPolicy, UnknownToolPolicy};
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

/// A strict tool used to prove harness-level schema validation runs before the
/// tool implementation is invoked.
struct StrictLookupTool {
    calls: Arc<Mutex<usize>>,
}

#[async_trait]
impl Tool<()> for StrictLookupTool {
    fn name(&self) -> &str {
        "strict_lookup"
    }
    fn description(&self) -> &str {
        "strict lookup"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "strict_lookup",
            "strict lookup",
            json!({
                "type": "object",
                "required": ["query"],
                "additionalProperties": false,
                "properties": {
                    "query": { "type": "string" },
                    "filters": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "limit": { "type": "integer" }
                        }
                    }
                }
            }),
        )
    }
    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::text(call.id, self.name(), "strict-output"))
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

/// Around-model wrap middleware that calls the inner pipeline then stamps the
/// finish reason on the resulting response.
struct StampModelWrap;

#[async_trait]
impl ModelMiddleware<()> for StampModelWrap {
    fn name(&self) -> &str {
        "stamp_model"
    }
    async fn wrap_model(
        &self,
        ctx: &mut RunContext<()>,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut response = next.run(ctx, state, request).await?.into_response();
        response.finish_reason = Some("wrapped".to_string());
        Ok(response.into())
    }
}

/// Around-tool wrap middleware that calls the inner pipeline then prefixes the
/// result content.
struct StampToolWrap;

#[async_trait]
impl ToolMiddleware<()> for StampToolWrap {
    fn name(&self) -> &str {
        "stamp_tool"
    }
    async fn wrap_tool(
        &self,
        ctx: &mut RunContext<()>,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        let mut result = next.run(ctx, state, call).await?.into_result();
        result.content = format!("[wrapped] {}", result.content);
        Ok(result.into())
    }
}

/// Around-model wrap middleware that short-circuits with a canned response and
/// never calls the inner pipeline (so the provider is never contacted).
struct ShortCircuitModelWrap;

#[async_trait]
impl ModelMiddleware<()> for ShortCircuitModelWrap {
    fn name(&self) -> &str {
        "short_circuit_model"
    }
    async fn wrap_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _request: ModelRequest,
        _next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        Ok(text_response("canned", 0, 0).into())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn wrap_middleware_fires_around_model_and_tool_calls() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "lookup", json!({"q": "x"})),
            text_response("done", 4, 2),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::new("lookup", "tool-output")));
    harness.push_model_middleware(Arc::new(StampModelWrap));
    harness.push_tool_middleware(Arc::new(StampToolWrap));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    // The tool-wrap mutated the tool result that was appended to the transcript.
    assert_eq!(run.messages[2].text(), "[wrapped] tool-output");
    // The model-wrap stamped the final response's finish reason.
    assert_eq!(
        run.final_response.unwrap().finish_reason.as_deref(),
        Some("wrapped")
    );
}

#[tokio::test]
async fn wrap_model_short_circuit_skips_provider() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    // FailingModel errors on every invoke and counts attempts; if the wrap
    // middleware short-circuits, the provider is never contacted.
    let model = Arc::new(FailingModel {
        attempts: Mutex::new(0),
    });
    harness.register_model("mock", model.clone());
    harness.push_model_middleware(Arc::new(ShortCircuitModelWrap));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("short-circuited run succeeds without the provider");

    assert_eq!(run.text(), Some("canned".to_string()));
    assert_eq!(*model.attempts.lock().unwrap(), 0);
}

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
async fn policy_model_call_limit_above_run_config_default_is_honored() {
    // Regression test: `RunConfig::new` defaults `max_model_calls` to 25, but
    // a harness-wide `RunPolicy` can configure a higher cap. Before the two
    // limit sources were unified, the context's tracker (seeded from the
    // `RunConfig` default) tripped at call 26 while the error message
    // incorrectly reported the policy's higher limit. This asserts the run
    // survives past 25 calls and, once it does trip, reports the limit that
    // actually applies.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("spin", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::new("spin", "again")));
    harness.with_policy(RunPolicy {
        limits: RunLimits::default()
            .with_max_model_calls(30)
            .with_max_tool_calls(1000),
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
    assert!(
        err.to_string().contains("30"),
        "expected error to report the policy's limit (30), got: {err}"
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
async fn unknown_tool_return_tool_error_recovers() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "missing", json!({})),
            text_response("recovered", 1, 1),
        ])),
    );
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::ReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("unknown tool is recoverable");

    assert_eq!(run.final_response.unwrap().text(), "recovered");
    // The injected tool-error message names the requested tool for repair.
    let injected = run
        .messages
        .iter()
        .any(|m| format!("{m:?}").contains("unknown tool `missing`"));
    assert!(
        injected,
        "recovery message should be injected into transcript"
    );
}

#[tokio::test]
async fn unknown_tool_rewrite_retargets_to_real_tool() {
    let lookup = Arc::new(FakeTool::new("lookup", "out"));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "missing", json!({})),
            text_response("done", 1, 1),
        ])),
    );
    harness.register_tool(lookup.clone());
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::Rewrite {
            tool_name: "lookup".to_string(),
        },
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("rewrite recovers");

    assert_eq!(run.final_response.unwrap().text(), "done");
    // The rewritten call actually executed the real tool.
    assert_eq!(*lookup.calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn invalid_tool_arguments_fail_before_tool_execution() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call(
            "strict_lookup",
            json!({ "query": 42, "extra": true }),
        )),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(StrictLookupTool {
        calls: Arc::clone(&calls),
    }));

    let err = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect_err("invalid arguments should fail closed");

    assert!(matches!(err, TinyAgentsError::Validation(_)), "got {err:?}");
    assert_eq!(
        *calls.lock().unwrap(),
        0,
        "tool implementation must not run"
    );
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
async fn fallback_chain_with_repeated_model_name_terminates() {
    // Regression test: a fallback chain that repeats a model name
    // (`[primary, backup, primary]`) used to alternate primary <-> backup
    // forever because `FallbackPolicy::next_after` always resolves from the
    // *first* occurrence of the current name. Both models fail every call, so
    // without a visited-set/hop-cap this run would never terminate.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let primary = Arc::new(FailingModel {
        attempts: Mutex::new(0),
    });
    let backup = Arc::new(FailingModel {
        attempts: Mutex::new(0),
    });
    harness.register_model("primary", primary.clone());
    harness.register_model("backup", backup.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(1),
        fallback: Some(FallbackPolicy::new(["primary", "backup", "primary"])),
        ..RunPolicy::default()
    });

    let err = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        harness.invoke_default(&(), vec![Message::user("hi")]),
    )
    .await
    .expect("fallback chain must terminate, not hang")
    .expect_err("both models fail, so the run must error out");

    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
    // Each model is visited at most once: primary, then backup, then the
    // chain's repeated `primary` entry is skipped as already-visited.
    assert_eq!(*primary.attempts.lock().unwrap(), 1);
    assert_eq!(*backup.attempts.lock().unwrap(), 1);
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

// ── Streaming path ────────────────────────────────────────────────────────────

/// Middleware that records every `on_model_delta` invocation and the text of
/// each delta it observes.
struct DeltaRecorder {
    count: Arc<Mutex<usize>>,
    texts: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<(), ()> for DeltaRecorder {
    fn name(&self) -> &str {
        "delta-recorder"
    }
    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut crate::harness::model::ModelDelta,
    ) -> Result<()> {
        *self.count.lock().unwrap() += 1;
        self.texts.lock().unwrap().push(delta.content.clone());
        Ok(())
    }
}

#[tokio::test]
async fn invoke_streaming_fires_on_model_delta_per_delta_and_accumulates() {
    use crate::harness::testkit::StreamingMock;

    let count = Arc::new(Mutex::new(0usize));
    let texts = Arc::new(Mutex::new(Vec::new()));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::from_text_chunks(["Hel", "lo, ", "world"])),
    );
    harness.push_middleware(Arc::new(DeltaRecorder {
        count: count.clone(),
        texts: texts.clone(),
    }));

    let run = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("stream-run"),
            vec![Message::user("hi")],
        )
        .await
        .expect("streaming run succeeds");

    // The merged response equals the concatenated chunks.
    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text(), Some("Hello, world".to_string()));

    // on_model_delta fired exactly once per streamed message delta.
    assert_eq!(*count.lock().unwrap(), 3);
    assert_eq!(
        *texts.lock().unwrap(),
        vec!["Hel".to_string(), "lo, ".to_string(), "world".to_string()]
    );
}

#[tokio::test]
async fn invoke_streaming_emits_model_delta_events() {
    use crate::harness::testkit::{EventRecorder, StreamingMock};

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::from_text_chunks(["a", "b"])),
    );

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("stream-run"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("streaming run succeeds");

    assert_eq!(run.text(), Some("ab".to_string()));
    let delta_run_ids: Vec<_> = recorder
        .events()
        .into_iter()
        .filter_map(|e| match e {
            crate::harness::events::AgentEvent::ModelDelta { run_id, .. } => Some(run_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        delta_run_ids.len(),
        2,
        "one model.delta event per streamed delta"
    );
    // Every delta is attributed to its run, so a UI can route it by lineage
    // without depending on which (shared) sink it arrived on.
    assert!(
        delta_run_ids.iter().all(|id| id.as_str() == "stream-run"),
        "deltas must carry their run id"
    );
}

// ── Cooperative cancellation ──────────────────────────────────────────────────

use crate::harness::cancel::CancellationToken;

/// A model that records how many times it was invoked and always asks for a
/// tool, so the loop only stops via a limit or cancellation.
struct CountingToolModel {
    name: &'static str,
    invocations: Arc<Mutex<usize>>,
}

#[async_trait]
impl ChatModel<()> for CountingToolModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> Result<ModelResponse> {
        *self.invocations.lock().unwrap() += 1;
        Ok(tool_call_response("call-1", self.name, json!({})))
    }
}

/// A tool that cancels the run's token the first time it is called, then
/// returns a fixed reply.
struct CancelOnCallTool {
    token: CancellationToken,
}

#[async_trait]
impl Tool<()> for CancelOnCallTool {
    fn name(&self) -> &str {
        "cancel_me"
    }
    fn description(&self) -> &str {
        "cancels the run"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("cancel_me", "cancels the run", json!({"type": "object"}))
    }
    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        self.token.cancel();
        Ok(ToolResult::text(call.id, "cancel_me", "cancelled"))
    }
}

#[tokio::test]
async fn token_cancelled_before_run_yields_cancelled() {
    let invocations = Arc::new(Mutex::new(0usize));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(CountingToolModel {
            name: "cancel_me",
            invocations: invocations.clone(),
        }),
    );

    // Pre-cancel the token before the run starts.
    let token = CancellationToken::new();
    token.cancel();
    let ctx = RunContext::new(RunConfig::new("cancel-run"), ()).with_cancellation(token);

    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect_err("a pre-cancelled run must not complete");

    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
    // The model was never invoked: cancellation is observed at the first
    // checkpoint, before any model call.
    assert_eq!(*invocations.lock().unwrap(), 0);
}

#[tokio::test]
async fn cancelled_mid_run_stops_before_next_model_call() {
    let invocations = Arc::new(Mutex::new(0usize));
    let token = CancellationToken::new();

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(CountingToolModel {
            name: "cancel_me",
            invocations: invocations.clone(),
        }),
    );
    harness.register_tool(Arc::new(CancelOnCallTool {
        token: token.clone(),
    }));

    let ctx = RunContext::new(RunConfig::new("cancel-mid"), ()).with_cancellation(token);

    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect_err("cancellation during a tool call must stop the run");

    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
    // Exactly one model call happened (the turn that requested the tool); the
    // tool cancelled the run, so the loop unwound before the second model call.
    assert_eq!(*invocations.lock().unwrap(), 1);
}

// ── Per-model-call timeout ─────────────────────────────────────────────────────

#[tokio::test]
async fn slow_model_call_is_timed_out_by_remaining_budget() {
    use std::time::Duration;

    use crate::harness::testkit::SlowModel;

    // The model sleeps far longer (200ms) than the run's wall-clock budget
    // (20ms), so the per-call timeout must interrupt it mid-flight and surface a
    // `Timeout` error rather than waiting for the model to return.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "slow",
        Arc::new(SlowModel::new(Duration::from_millis(200), "too late")),
    );

    let config = RunConfig::new("timeout-run").with_timeout_ms(20);
    let err = harness
        .invoke(&(), (), config, vec![Message::user("hi")])
        .await
        .expect_err("a model call slower than the budget must time out");

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
}

#[tokio::test]
async fn fast_model_call_succeeds_under_same_budget() {
    // Control: under the same small timeout a fast model completes well within
    // the budget, proving the timeout only fires on genuinely slow calls.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("fast", Arc::new(MockModel::constant("done")));

    let config = RunConfig::new("fast-run").with_timeout_ms(20);
    let run = harness
        .invoke(&(), (), config, vec![Message::user("hi")])
        .await
        .expect("a fast model call completes within the budget");

    assert_eq!(run.text(), Some("done".to_string()));
}

#[tokio::test]
async fn slow_streaming_model_call_is_timed_out() {
    use std::time::Duration;

    use crate::harness::testkit::SlowModel;

    // The streaming path must enforce the same per-call budget: the default
    // `stream` impl delegates to `invoke`, whose sleep exceeds the 20ms budget.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "slow",
        Arc::new(SlowModel::new(Duration::from_millis(200), "too late")),
    );

    let config = RunConfig::new("timeout-stream-run").with_timeout_ms(20);
    let err = harness
        .invoke_streaming(&(), (), config, vec![Message::user("hi")])
        .await
        .expect_err("a slow streaming model call must time out");

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
}

// ── Response caching ──────────────────────────────────────────────────────────

#[tokio::test]
async fn response_cache_serves_repeated_request_without_calling_model() {
    use crate::harness::cache::InMemoryResponseCache;
    use crate::harness::testkit::EventRecorder;

    // A scripted model that would yield *different* text on a second call; if
    // the cache works the second run must reuse the first response, proving the
    // model was not invoked again.
    let model = Arc::new(MockModel::with_responses(vec![
        text_response("first-answer", 4, 2),
        text_response("second-answer", 4, 2),
    ]));

    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_response_cache(cache.clone());

    // First run: cache miss, model invoked once.
    let recorder1 = EventRecorder::new();
    let ctx1 = RunContext::new(RunConfig::new("cache-run"), ()).with_events(recorder1.sink());
    let run1 = harness
        .invoke_in_context(&(), ctx1, vec![Message::user("same question")])
        .await
        .expect("first run succeeds");

    assert_eq!(model.call_count(), 1, "model invoked once on first run");
    assert_eq!(run1.text(), Some("first-answer".to_string()));
    assert!(
        recorder1.kinds().iter().any(|k| k == "cache.miss"),
        "first run should emit a cache miss"
    );
    assert!(
        !recorder1.kinds().iter().any(|k| k == "cache.hit"),
        "first run should not emit a cache hit"
    );

    // Second run with the SAME input: served from cache, model NOT invoked.
    let recorder2 = EventRecorder::new();
    let ctx2 = RunContext::new(RunConfig::new("cache-run-2"), ()).with_events(recorder2.sink());
    let run2 = harness
        .invoke_in_context(&(), ctx2, vec![Message::user("same question")])
        .await
        .expect("second run succeeds");

    assert_eq!(
        model.call_count(),
        1,
        "model must NOT be invoked again on a cache hit"
    );
    assert_eq!(
        run2.text(),
        Some("first-answer".to_string()),
        "cached response text is reused"
    );
    assert!(
        recorder2.kinds().iter().any(|k| k == "cache.hit"),
        "second run should emit a cache hit"
    );
    // Accounting stays consistent: the hit is still counted as a model call.
    assert_eq!(run2.model_calls, 1);
}

#[tokio::test]
async fn no_cache_attached_invokes_model_each_run() {
    // Control: without a cache the model is invoked on every run.
    let model = Arc::new(MockModel::echo());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());

    harness
        .invoke_default(&(), vec![Message::user("hello")])
        .await
        .expect("first run succeeds");
    harness
        .invoke_default(&(), vec![Message::user("hello")])
        .await
        .expect("second run succeeds");

    assert_eq!(
        model.call_count(),
        2,
        "without a cache the model is invoked on every run"
    );
}

#[tokio::test]
async fn request_cache_policy_overrides_run_policy_to_disable_caching() {
    use crate::harness::cache::{CachePolicy, InMemoryResponseCache};

    // A middleware that disables caching for the call via the request-level
    // cache policy, overriding the harness default (which is enabled).
    struct DisableCaching;
    #[async_trait]
    impl Middleware<(), ()> for DisableCaching {
        fn name(&self) -> &str {
            "disable-caching"
        }
        async fn before_model(
            &self,
            _ctx: &mut RunContext<()>,
            _state: &(),
            request: &mut ModelRequest,
        ) -> Result<()> {
            request.cache_policy = Some(CachePolicy {
                response_cache_enabled: false,
                protect_prompt_prefix: false,
            });
            Ok(())
        }
    }

    let model = Arc::new(MockModel::echo());
    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_response_cache(cache.clone());
    harness.push_middleware(Arc::new(DisableCaching));

    harness
        .invoke_default(&(), vec![Message::user("hello")])
        .await
        .expect("first run succeeds");
    harness
        .invoke_default(&(), vec![Message::user("hello")])
        .await
        .expect("second run succeeds");

    assert_eq!(
        model.call_count(),
        2,
        "request-level cache_policy disabling caching must bypass the cache"
    );
}

/// Middleware that requests an early stop-with-final control outcome after the
/// first model response, exercising the harness control channel (gap #13).
struct EarlyStopMiddleware;

#[async_trait]
impl Middleware<(), ()> for EarlyStopMiddleware {
    fn name(&self) -> &str {
        "early_stop"
    }
    async fn after_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> Result<()> {
        ctx.request_control(crate::harness::context::MiddlewareControl::StopWithFinal(
            "stopped early".into(),
        ));
        Ok(())
    }
}

#[tokio::test]
async fn middleware_control_stops_loop_with_final_response() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    // The model asks for a tool; without control the loop would execute it and
    // continue, but the control outcome stops the run first.
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("lookup", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::new("lookup", "out")));
    harness.push_middleware(Arc::new(EarlyStopMiddleware));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("control stop yields a run");
    assert_eq!(run.final_response.unwrap().text(), "stopped early");
    // The tool was never executed because the loop stopped first.
    assert_eq!(run.tool_calls, 0);
}

/// Middleware that requests an interrupt after the first model response.
struct InterruptMiddleware;

#[async_trait]
impl Middleware<(), ()> for InterruptMiddleware {
    fn name(&self) -> &str {
        "interrupt_ctl"
    }
    async fn after_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> Result<()> {
        ctx.request_control(crate::harness::context::MiddlewareControl::Interrupt {
            node: "review".into(),
            message: "needs approval".into(),
        });
        Ok(())
    }
}

#[tokio::test]
async fn middleware_control_can_interrupt_run() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hi")));
    harness.push_middleware(Arc::new(InterruptMiddleware));

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("control interrupt surfaces as an error");
    match err {
        TinyAgentsError::Interrupted { node, message } => {
            assert_eq!(node, "review");
            assert_eq!(message, "needs approval");
        }
        other => panic!("expected Interrupted, got {other:?}"),
    }
}
