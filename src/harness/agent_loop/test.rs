//! Tests for the default agent loop.
//!
//! These exercise the loop end to end with [`MockModel`] and a local
//! `FakeTool`: a single text response, a multi-step tool loop, limit
//! enforcement, middleware request mutation, usage accumulation, the
//! tool-not-found path, structured output extraction, and retry/fallback.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

use super::AgentStreamItem;
use crate::error::{Result, TinyAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, EventSink};
use crate::harness::limits::RunLimits;
use crate::harness::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use crate::harness::middleware::{
    AgentRun, Middleware, MiddlewareModelOutcome, MiddlewareToolOutcome, ModelHandler,
    ModelMiddleware, ToolHandler, ToolMiddleware,
};
use crate::harness::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStreamItem, ResponseFormat,
    ToolChoice,
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

/// A tool that sleeps `delay` before returning a fixed reply, used to prove a
/// hanging tool call is bounded by the run's remaining wall-clock budget the
/// same way a hanging model call is.
struct SlowTool {
    delay: std::time::Duration,
}

#[async_trait]
impl Tool<()> for SlowTool {
    fn name(&self) -> &str {
        "slow"
    }
    fn description(&self) -> &str {
        "slow tool"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("slow", "slow tool", json!({"type": "object"}))
    }
    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        tokio::time::sleep(self.delay).await;
        Ok(ToolResult::text(call.id, "slow", "too late"))
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

/// A model that always fails with a retryable error and records the
/// (virtual) `tokio::time::Instant` of each `invoke` call, so a test can
/// assert on the actual elapsed time between retries rather than just the
/// attempt count.
struct TimestampingFailingModel {
    timestamps: Mutex<Vec<tokio::time::Instant>>,
}

#[async_trait]
impl ChatModel<()> for TimestampingFailingModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> Result<ModelResponse> {
        self.timestamps
            .lock()
            .unwrap()
            .push(tokio::time::Instant::now());
        Err(TinyAgentsError::Model("transient boom".to_string()))
    }
}

/// A model that always fails with a structured `TinyAgentsError::Provider`
/// error whose `retryable` flag is fixed at construction, and counts
/// attempts. Used to prove the agent loop's retry decision consults the
/// structured flag rather than retrying every provider failure.
struct ProviderFailingModel {
    retryable: bool,
    status: u16,
    attempts: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for ProviderFailingModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Err(TinyAgentsError::Provider(Box::new(
            crate::harness::model::ProviderError {
                provider: "test-provider".to_string(),
                status: Some(self.status),
                retryable: self.retryable,
                message: "boom".to_string(),
                ..crate::harness::model::ProviderError::default()
            },
        )))
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

#[tokio::test(start_paused = true)]
async fn retry_backoff_sleeps_the_documented_schedule() {
    // Regression test: the loop used to compute the backoff from the
    // *post-increment* attempt number, so the first retry's sleep skipped
    // `initial_backoff_ms` entirely and the whole exponential schedule was
    // shifted one step higher than `RetryPolicy::backoff_for_attempt`
    // documents. With `initial_backoff_ms = 100`, `multiplier = 2.0`, no
    // jitter: attempt 0 -> 100ms, attempt 1 -> 200ms, attempt 2 -> 400ms.
    use std::time::Duration;

    let policy = RetryPolicy::default()
        .with_max_attempts(4)
        .with_initial_backoff_ms(100)
        .with_multiplier(2.0)
        .with_jitter(false)
        .with_backoff_sleep(true);

    let model = Arc::new(TimestampingFailingModel {
        timestamps: Mutex::new(Vec::new()),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("flaky", model.clone());
    harness.with_policy(RunPolicy {
        retry: policy,
        ..RunPolicy::default()
    });

    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("all 4 attempts fail");

    let timestamps = model.timestamps.lock().unwrap().clone();
    assert_eq!(timestamps.len(), 4, "expected exactly max_attempts calls");

    let gaps: Vec<Duration> = timestamps
        .windows(2)
        .map(|w| w[1].duration_since(w[0]))
        .collect();
    assert_eq!(
        gaps,
        vec![
            Duration::from_millis(100), // before retry 1 (attempt 0's backoff)
            Duration::from_millis(200), // before retry 2 (attempt 1's backoff)
            Duration::from_millis(400), // before retry 3 (attempt 2's backoff)
        ],
        "backoff schedule does not match RetryPolicy::backoff_for_attempt"
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
async fn run_limits_max_retries_per_call_caps_a_looser_retry_policy() {
    // Regression test: `RunLimits::max_retries_per_call` was parsed but never
    // enforced, so a `RetryPolicy` with a higher `max_attempts` silently
    // ignored the harness's "hard" limit. `max_retries_per_call: 1` (one
    // retry, so 2 attempts total) must win over `max_attempts: 5`.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let failing = Arc::new(FailingModel {
        attempts: Mutex::new(0),
    });
    harness.register_model("primary", failing.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(5),
        limits: RunLimits::default().with_max_retries_per_call(1),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("no fallback, retries capped by RunLimits");
    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
    assert_eq!(*failing.attempts.lock().unwrap(), 2);
}

#[tokio::test]
async fn provider_error_401_is_not_retried() {
    // Regression test: before `ProviderError` was preserved structurally, a
    // 401 flattened into `Model(String)` was retried like any other model
    // error. A non-retryable `Provider` error must fail on the first attempt.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let model = Arc::new(ProviderFailingModel {
        retryable: false,
        status: 401,
        attempts: Mutex::new(0),
    });
    harness.register_model("primary", model.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(5),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("401 is not retryable");
    assert!(matches!(err, TinyAgentsError::Provider(_)), "got {err:?}");
    assert_eq!(*model.attempts.lock().unwrap(), 1);
}

#[tokio::test]
async fn provider_error_429_is_retried_up_to_max_attempts() {
    // Contrast with the 401 case: a retryable `Provider` error (e.g. a 429)
    // must still be retried up to `max_attempts`.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let model = Arc::new(ProviderFailingModel {
        retryable: true,
        status: 429,
        attempts: Mutex::new(0),
    });
    harness.register_model("primary", model.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(3),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("retries exhausted");
    assert!(matches!(err, TinyAgentsError::Provider(_)), "got {err:?}");
    assert_eq!(*model.attempts.lock().unwrap(), 3);
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
    reasonings: Arc<Mutex<Vec<String>>>,
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
        self.reasonings
            .lock()
            .unwrap()
            .push(delta.reasoning.clone());
        Ok(())
    }
}

#[tokio::test]
async fn invoke_streaming_fires_on_model_delta_per_delta_and_accumulates() {
    use crate::harness::testkit::StreamingMock;

    let count = Arc::new(Mutex::new(0usize));
    let texts = Arc::new(Mutex::new(Vec::new()));
    let reasonings = Arc::new(Mutex::new(Vec::new()));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::from_text_chunks(["Hel", "lo, ", "world"])),
    );
    harness.push_middleware(Arc::new(DeltaRecorder {
        count: count.clone(),
        texts: texts.clone(),
        reasonings: reasonings.clone(),
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
    assert_eq!(
        *reasonings.lock().unwrap(),
        vec![String::new(), String::new(), String::new()]
    );
}

#[tokio::test]
async fn invoke_streaming_forwards_reasoning_deltas_to_middleware_and_events() {
    use crate::harness::testkit::{EventRecorder, StreamingMock};

    let count = Arc::new(Mutex::new(0usize));
    let texts = Arc::new(Mutex::new(Vec::new()));
    let reasonings = Arc::new(Mutex::new(Vec::new()));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::new(vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(MessageDelta::reasoning("think ")),
            ModelStreamItem::MessageDelta(MessageDelta::text("answer")),
            ModelStreamItem::Completed(ModelResponse::assistant("answer")),
        ])),
    );
    harness.push_middleware(Arc::new(DeltaRecorder {
        count: count.clone(),
        texts: texts.clone(),
        reasonings: reasonings.clone(),
    }));

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("stream-run"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("streaming run succeeds");

    assert_eq!(run.text(), Some("answer".to_string()));
    assert_eq!(*count.lock().unwrap(), 2);
    assert_eq!(
        *texts.lock().unwrap(),
        vec![String::new(), "answer".to_string()]
    );
    assert_eq!(
        *reasonings.lock().unwrap(),
        vec!["think ".to_string(), String::new()]
    );

    let event_reasoning: String = recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            crate::harness::events::AgentEvent::ModelDelta { delta, .. } => Some(delta.reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(event_reasoning, "think ");
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

/// A model whose unary `invoke` never returns on its own, signalling once it has
/// started so a test can cancel the run while the call is genuinely in flight.
struct BlockForeverModel {
    started: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl ChatModel<()> for BlockForeverModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> Result<ModelResponse> {
        self.started.notify_one();
        // Simulate a long buffered (non-streamed) provider call that only ends
        // when the caller drops this future. Without the loop racing
        // cancellation against the in-flight call, the run would hang here.
        std::future::pending::<()>().await;
        unreachable!("pending future never resolves")
    }
}

#[tokio::test]
async fn cancelled_during_unary_model_call_drops_the_in_flight_call() {
    let token = CancellationToken::new();
    let started = Arc::new(tokio::sync::Notify::new());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(BlockForeverModel {
            started: started.clone(),
        }),
    );

    let ctx = RunContext::new(RunConfig::new("cancel-unary"), ()).with_cancellation(token.clone());

    // Cancel only once the model call has actually begun, so the pre-call
    // checkpoint cannot short-circuit and we exercise the in-flight race.
    let canceller = tokio::spawn(async move {
        started.notified().await;
        token.cancel();
    });

    let err = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        harness.invoke_in_context(&(), ctx, vec![Message::user("hi")]),
    )
    .await
    .expect("run must not hang: a cancel mid unary call must drop the in-flight future")
    .expect_err("a run cancelled mid model call must not complete");

    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
    canceller.await.unwrap();
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

#[tokio::test]
async fn slow_tool_call_is_timed_out_by_remaining_budget() {
    use std::time::Duration;

    // Regression test: the remaining wall-clock budget was previously only
    // enforced around model calls, so a hanging tool call could block the run
    // past its deadline. The tool sleeps far longer (200ms) than the run's
    // budget (20ms), so the same per-call timeout used for model calls must
    // interrupt it too.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("slow", json!({}))),
    );
    harness.register_tool(Arc::new(SlowTool {
        delay: Duration::from_millis(200),
    }));

    let config = RunConfig::new("tool-timeout-run").with_timeout_ms(20);
    let err = harness
        .invoke(&(), (), config, vec![Message::user("go")])
        .await
        .expect_err("a tool call slower than the budget must time out");

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
async fn multi_turn_request_with_prior_assistant_turn_is_not_cached() {
    use crate::harness::cache::InMemoryResponseCache;

    // A request whose transcript already contains an assistant turn can never be
    // re-served identically, so it must bypass the cache entirely: the model is
    // invoked on every run even for identical multi-turn input.
    let model = Arc::new(MockModel::with_responses(vec![
        text_response("a1", 4, 2),
        text_response("a2", 4, 2),
    ]));
    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_response_cache(cache.clone());

    let convo = vec![
        Message::user("q1"),
        Message::assistant("prior answer"),
        Message::user("q2"),
    ];

    harness
        .invoke_default(&(), convo.clone())
        .await
        .expect("first run succeeds");
    harness
        .invoke_default(&(), convo)
        .await
        .expect("second run succeeds");

    assert_eq!(
        model.call_count(),
        2,
        "multi-turn requests bypass the cache, so the model runs each time"
    );
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

/// A model that records the tool schemas presented on each `invoke` and replays
/// scripted responses in order. Used to prove the loop exposes the registered
/// tool set on *every* turn, not just the first.
struct ToolCapturingModel {
    responses: Mutex<std::collections::VecDeque<ModelResponse>>,
    seen_tools: Arc<Mutex<Vec<Vec<ToolSchema>>>>,
}

#[async_trait]
impl ChatModel<()> for ToolCapturingModel {
    async fn invoke(&self, _state: &(), request: ModelRequest) -> Result<ModelResponse> {
        self.seen_tools.lock().unwrap().push(request.tools.clone());
        let next = self.responses.lock().unwrap().pop_front();
        next.ok_or_else(|| TinyAgentsError::Validation("no scripted response left".into()))
    }
}

/// Regression test for the per-run tool-schema cache: hoisting
/// `self.tools.schemas()` out of the loop must not change the tools the model
/// sees on any turn. A two-turn run (tool call, then final text) must present
/// the identical, non-empty schema set on both turns.
#[tokio::test]
async fn tool_schemas_are_stable_across_turns() {
    let seen_tools = Arc::new(Mutex::new(Vec::new()));
    let model = ToolCapturingModel {
        responses: Mutex::new(
            vec![
                tool_call_response("c1", "spin", json!({})),
                text_response("done", 5, 2),
            ]
            .into(),
        ),
        seen_tools: Arc::clone(&seen_tools),
    };

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(model));
    harness.register_tool(Arc::new(FakeTool::new("spin", "again")));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.text(), Some("done".to_string()));

    let seen = seen_tools.lock().unwrap();
    assert_eq!(seen.len(), 2, "model should be invoked twice");
    assert!(
        !seen[0].is_empty(),
        "first turn must expose the registered tool schema"
    );
    assert_eq!(
        seen[0], seen[1],
        "cached schemas must reach the model identically on every turn"
    );
    assert_eq!(seen[0][0].name, "spin");
}

// ── Explicit model override fall-through diagnostics ────────────────────────

/// Middleware that stamps an explicit model override onto every request.
struct OverrideModelMiddleware(&'static str);

#[async_trait]
impl Middleware<(), ()> for OverrideModelMiddleware {
    fn name(&self) -> &str {
        "override_model"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.model = Some(self.0.to_string());
        Ok(())
    }
}

/// When an explicit request override cannot be honored (here: an unregistered
/// model name) resolution falls through to the registry default by documented
/// fail-closed semantics — but the fall-through must be observable via
/// `ModelOverrideSkipped` rather than silent.
#[tokio::test]
async fn skipped_model_override_emits_diagnostic_event() {
    use crate::harness::events::AgentEvent;
    use crate::harness::testkit::EventRecorder;

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("ok")));
    harness.push_middleware(Arc::new(OverrideModelMiddleware("missing-model")));

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("override-run"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("run falls through to the default model");
    assert_eq!(run.text(), Some("ok".to_string()));

    assert!(
        recorder.events().iter().any(|e| matches!(
            e,
            AgentEvent::ModelOverrideSkipped { requested, resolved }
                if requested == "missing-model" && resolved == "mock"
        )),
        "the skipped override must be surfaced as a diagnostic event; got kinds {:?}",
        recorder.kinds()
    );
}

/// An override that resolution honors must not emit the diagnostic.
#[tokio::test]
async fn honored_model_override_emits_no_diagnostic_event() {
    use crate::harness::testkit::EventRecorder;

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("default")));
    harness.register_model("special", Arc::new(MockModel::constant("special answer")));
    harness.push_middleware(Arc::new(OverrideModelMiddleware("special")));

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("override-ok-run"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("run uses the override");
    assert_eq!(run.text(), Some("special answer".to_string()));
    assert!(
        !recorder
            .kinds()
            .iter()
            .any(|k| k == "model.override_skipped"),
        "an honored override must not emit the diagnostic"
    );
}

// ── Parallel tool execution ─────────────────────────────────────────────────

/// A tool that tracks how many probe tools are in flight at once (and the
/// maximum observed), sleeping briefly so overlapping calls are observable.
struct ConcurrencyProbeTool {
    name: &'static str,
    reply: &'static str,
    delay: std::time::Duration,
    active: Arc<std::sync::atomic::AtomicUsize>,
    max_seen: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait]
impl Tool<()> for ConcurrencyProbeTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "concurrency probe"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name, "concurrency probe", json!({"type": "object"}))
    }
    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        use std::sync::atomic::Ordering;
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_seen.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::text(call.id, self.name, self.reply))
    }
}

/// Builds an assistant response carrying several tool calls in one turn.
fn multi_tool_call_response(calls: Vec<(&str, &str)>) -> ModelResponse {
    let tool_calls = calls
        .into_iter()
        .map(|(id, name)| ToolCall::new(id, name, json!({})))
        .collect::<Vec<_>>();
    ModelResponse {
        message: AssistantMessage {
            id: Some("msg-multi".to_string()),
            content: Vec::new(),
            tool_calls,
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// Registers two probe tools sharing one active/max counter pair.
fn probe_pair(
    harness: &mut AgentHarness<()>,
    delay_ms: (u64, u64),
) -> Arc<std::sync::atomic::AtomicUsize> {
    let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    harness.register_tool(Arc::new(ConcurrencyProbeTool {
        name: "alpha",
        reply: "alpha-out",
        delay: std::time::Duration::from_millis(delay_ms.0),
        active: active.clone(),
        max_seen: max_seen.clone(),
    }));
    harness.register_tool(Arc::new(ConcurrencyProbeTool {
        name: "beta",
        reply: "beta-out",
        delay: std::time::Duration::from_millis(delay_ms.1),
        active,
        max_seen: max_seen.clone(),
    }));
    max_seen
}

#[tokio::test]
async fn independent_tool_calls_in_one_turn_run_concurrently() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            multi_tool_call_response(vec![("call-a", "alpha"), ("call-b", "beta")]),
            text_response("done", 4, 2),
        ])),
    );
    let max_seen = probe_pair(&mut harness, (80, 80));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.tool_calls, 2);
    assert_eq!(
        max_seen.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "both tools must be in flight at once (latency ~max, not ~sum)"
    );
}

#[tokio::test]
async fn parallel_tool_results_keep_original_call_order_and_ids() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            multi_tool_call_response(vec![("call-a", "alpha"), ("call-b", "beta")]),
            text_response("done", 4, 2),
        ])),
    );
    // alpha finishes *after* beta; the transcript must still list alpha first.
    let _ = probe_pair(&mut harness, (120, 0));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    // user, assistant(2 tool calls), tool(alpha), tool(beta), assistant(final).
    assert_eq!(run.messages.len(), 5);
    let Message::Tool(first) = &run.messages[2] else {
        panic!("expected tool message at index 2");
    };
    let Message::Tool(second) = &run.messages[3] else {
        panic!("expected tool message at index 3");
    };
    assert_eq!(first.tool_call_id, "call-a");
    assert_eq!(run.messages[2].text(), "alpha-out");
    assert_eq!(second.tool_call_id, "call-b");
    assert_eq!(run.messages[3].text(), "beta-out");
}

#[tokio::test]
async fn tool_wrap_middleware_forces_serial_execution() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            multi_tool_call_response(vec![("call-a", "alpha"), ("call-b", "beta")]),
            text_response("done", 4, 2),
        ])),
    );
    let max_seen = probe_pair(&mut harness, (40, 40));
    // A tool-wrap middleware holds `&mut RunContext` across each wrapped call,
    // so the loop must fall back to serial execution.
    harness.push_tool_middleware(Arc::new(StampToolWrap));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.tool_calls, 2);
    assert_eq!(
        max_seen.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "wrapped tool calls must never overlap"
    );
    // The wrap still fired around each call.
    assert_eq!(run.messages[2].text(), "[wrapped] alpha-out");
    assert_eq!(run.messages[3].text(), "[wrapped] beta-out");
}

#[tokio::test]
async fn unknown_tool_recovery_keeps_its_slot_in_a_parallel_turn() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            multi_tool_call_response(vec![("call-x", "missing"), ("call-b", "beta")]),
            text_response("done", 4, 2),
        ])),
    );
    let _ = probe_pair(&mut harness, (0, 0));
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::ReturnToolError,
        ..Default::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run recovers from the unknown tool");

    assert_eq!(run.tool_calls, 2, "recovery consumes a tool-call slot");
    let Message::Tool(first) = &run.messages[2] else {
        panic!("expected tool message at index 2");
    };
    let Message::Tool(second) = &run.messages[3] else {
        panic!("expected tool message at index 3");
    };
    // The recovery message occupies the unknown call's original slot.
    assert_eq!(first.tool_call_id, "call-x");
    assert!(run.messages[2].text().contains("unknown tool `missing`"));
    assert_eq!(second.tool_call_id, "call-b");
    assert_eq!(run.messages[3].text(), "beta-out");
}

// ── invoke_stream (caller-consumable streaming) ──────────────────────────────

/// Collects an `invoke_stream` run into a vec and returns (events, terminal).
async fn collect_stream(items: Vec<AgentStreamItem>) -> (Vec<AgentEvent>, AgentStreamItem) {
    let terminal = items
        .last()
        .cloned()
        .expect("stream must yield at least a terminal item");
    // Exactly one terminal, and it is the final item.
    let terminals = items
        .iter()
        .filter(|i| !matches!(i, AgentStreamItem::Event(_)))
        .count();
    assert_eq!(terminals, 1, "exactly one terminal item");
    assert!(
        !matches!(
            items[..items.len() - 1].last(),
            Some(AgentStreamItem::Completed(_))
        ) && !matches!(
            items[..items.len() - 1].last(),
            Some(AgentStreamItem::Failed(_))
        ),
        "terminal must be last"
    );
    let events = items
        .into_iter()
        .filter_map(|i| match i {
            AgentStreamItem::Event(r) => Some(r.event),
            _ => None,
        })
        .collect();
    (events, terminal)
}

#[tokio::test]
async fn invoke_stream_yields_events_then_completed() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hello there")));

    let items: Vec<AgentStreamItem> = harness
        .invoke_stream(
            &(),
            (),
            RunConfig::new("run-stream"),
            vec![Message::user("hi")],
        )
        .collect()
        .await;
    let (events, terminal) = collect_stream(items).await;

    match terminal {
        AgentStreamItem::Completed(run) => {
            assert_eq!(run.text().as_deref(), Some("hello there"));
        }
        other => panic!("expected Completed terminal, got {other:?}"),
    }
    // Live events flowed before the terminal: run lifecycle + a model delta.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ModelDelta { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunCompleted { .. }))
    );
}

#[tokio::test]
async fn invoke_stream_in_context_preserves_caller_context() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hello there")));
    let ctx = RunContext::new(
        RunConfig::new("caller-run").with_thread("caller-thread"),
        (),
    );

    let items: Vec<AgentStreamItem> = harness
        .invoke_stream_in_context(&(), ctx, vec![Message::user("hi")])
        .collect()
        .await;
    let (events, terminal) = collect_stream(items).await;

    let started = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::RunStarted { run_id, thread_id } => Some((run_id, thread_id)),
            _ => None,
        })
        .expect("RunStarted event");
    assert_eq!(started.0.as_str(), "caller-run");
    assert_eq!(
        started.1.as_ref().map(|thread| thread.as_str()),
        Some("caller-thread")
    );
    match terminal {
        AgentStreamItem::Completed(run) => {
            assert_eq!(run.text().as_deref(), Some("hello there"));
        }
        other => panic!("expected Completed terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn invoke_stream_in_context_unsubscribes_channel_listener() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hello there")));
    let events = EventSink::new();
    let ctx = RunContext::new(RunConfig::new("shared-events-run"), ()).with_events(events.clone());

    assert_eq!(events.listener_count(), 0);
    let items: Vec<AgentStreamItem> = harness
        .invoke_stream_in_context(&(), ctx, vec![Message::user("hi")])
        .collect()
        .await;
    let (_events, terminal) = collect_stream(items).await;

    assert!(matches!(terminal, AgentStreamItem::Completed(_)));
    assert_eq!(events.listener_count(), 0);
}

#[test]
fn invoke_stream_in_context_stream_is_send() {
    fn assert_send<T: Send>(_value: T) {}

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hello there")));
    let ctx = RunContext::new(RunConfig::new("send-stream-run"), ());

    assert_send(harness.invoke_stream_in_context(&(), ctx, vec![Message::user("hi")]));
}

#[tokio::test]
async fn invoke_stream_surfaces_tool_lifecycle() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "spin", json!({})),
            text_response("done", 5, 2),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::new("spin", "again")));

    let items: Vec<AgentStreamItem> = harness
        .invoke_stream(
            &(),
            (),
            RunConfig::new("run-tools"),
            vec![Message::user("go")],
        )
        .collect()
        .await;
    let (events, terminal) = collect_stream(items).await;

    let started = events.iter().position(
        |e| matches!(e, AgentEvent::ToolStarted { tool_name, .. } if tool_name == "spin"),
    );
    let completed = events.iter().position(
        |e| matches!(e, AgentEvent::ToolCompleted { tool_name, .. } if tool_name == "spin"),
    );
    assert!(started.is_some(), "ToolStarted for spin must be streamed");
    assert!(
        completed.is_some(),
        "ToolCompleted for spin must be streamed"
    );
    assert!(
        started < completed,
        "ToolStarted must precede ToolCompleted in the stream"
    );
    match terminal {
        AgentStreamItem::Completed(run) => assert_eq!(run.text().as_deref(), Some("done")),
        other => panic!("expected Completed terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn invoke_stream_yields_failed_terminal_on_error() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    // A model that always asks for the tool, capped so the loop trips the limit.
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("spin", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::new("spin", "again")));
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_calls(1),
        ..RunPolicy::default()
    });

    let items: Vec<AgentStreamItem> = harness
        .invoke_stream(
            &(),
            (),
            RunConfig::new("run-fail"),
            vec![Message::user("go")],
        )
        .collect()
        .await;
    let (_events, terminal) = collect_stream(items).await;

    match terminal {
        AgentStreamItem::Failed(message) => {
            assert!(message.contains("max model calls"), "got: {message}");
        }
        other => panic!("expected Failed terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn invoke_stream_scripted_incremental_deltas_surface_reasoning() {
    // A scripted stream drives truly incremental deltas — a reasoning fragment
    // and two text fragments — which must all surface on the event stream.
    let script = vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::reasoning("thinking hard")),
        ModelStreamItem::MessageDelta(MessageDelta::text("hel")),
        ModelStreamItem::MessageDelta(MessageDelta::text("lo")),
        ModelStreamItem::Completed(ModelResponse::assistant("hello")),
    ];
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::streaming_script(script)));

    let items: Vec<AgentStreamItem> = harness
        .invoke_stream(
            &(),
            (),
            RunConfig::new("run-script"),
            vec![Message::user("hi")],
        )
        .collect()
        .await;
    let (events, terminal) = collect_stream(items).await;

    // The scripted reasoning fragment surfaces on a ModelDelta event.
    assert!(events.iter().any(
        |e| matches!(e, AgentEvent::ModelDelta { delta, .. } if delta.reasoning == "thinking hard")
    ));
    // Text arrives as multiple incremental deltas, not one merged blob.
    let text_deltas = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ModelDelta { delta, .. } if !delta.text.is_empty()))
        .count();
    assert!(
        text_deltas >= 2,
        "expected incremental text deltas, got {text_deltas}"
    );

    match terminal {
        AgentStreamItem::Completed(run) => assert_eq!(run.text().as_deref(), Some("hello")),
        other => panic!("expected Completed terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn mock_streaming_script_invoke_folds_items_to_response() {
    // `invoke` on a scripted-stream mock folds the items into the equivalent
    // unary response (delta-only script → reconstructed from deltas).
    let model = MockModel::streaming_script(vec![
        ModelStreamItem::MessageDelta(MessageDelta::text("ab")),
        ModelStreamItem::MessageDelta(MessageDelta::text("cd")),
    ]);
    let response = ChatModel::invoke(&model, &(), ModelRequest::new(vec![]))
        .await
        .expect("fold succeeds");
    assert_eq!(response.text(), "abcd");
}

#[tokio::test]
async fn tool_completed_event_carries_outcome() {
    use crate::harness::events::RecordingListener;

    // A tool that fails: its ToolCompleted event must carry the failure message,
    // a real duration, and the output size — from the event itself, not a
    // side-channel — so journal-backed exporters can render the outcome.
    struct FailTool;
    #[async_trait]
    impl Tool<()> for FailTool {
        fn name(&self) -> &str {
            "boom"
        }
        fn description(&self) -> &str {
            "always fails"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("boom", "always fails", json!({ "type": "object" }))
        }
        async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult {
                call_id: call.id,
                name: "boom".to_string(),
                content: "nope".to_string(),
                raw: None,
                error: Some("kaboom".to_string()),
                elapsed_ms: 0,
            })
        }
    }

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "boom", json!({})),
            text_response("done", 1, 1),
        ])),
    );
    harness.register_tool(Arc::new(FailTool));

    let recorder = Arc::new(RecordingListener::new());
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-tc"), ());
    ctx.events.subscribe(recorder.clone());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let (error, duration_ms, output_bytes) = recorder
        .events()
        .into_iter()
        .find_map(|record| match record.event {
            AgentEvent::ToolCompleted {
                tool_name,
                error,
                duration_ms,
                output_bytes,
                ..
            } if tool_name == "boom" => Some((error, duration_ms, output_bytes)),
            _ => None,
        })
        .expect("a ToolCompleted event for `boom`");

    assert_eq!(
        error.as_deref(),
        Some("kaboom"),
        "failure message on the event"
    );
    assert!(duration_ms.is_some(), "wall-clock duration present");
    assert_eq!(output_bytes, Some(4), "\"nope\".len() == 4");
}
