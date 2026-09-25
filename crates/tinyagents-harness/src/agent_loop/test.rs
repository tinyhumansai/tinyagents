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
use crate::context::{RunConfig, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::{AgentEvent, EventSink};
use crate::limits::RunLimits;
use crate::middleware::{
    AgentHandler, AgentMiddleware, AgentRequest, AgentRun, Middleware, MiddlewareModelOutcome,
    MiddlewareToolOutcome, ModelHandler, ModelMiddleware, ToolHandler, ToolInvocationIdentity,
    ToolMiddleware,
};
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, InvalidArgsPolicy, RunPolicy, UnknownToolPolicy};
use crate::tool::ToolTimeoutSettings;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use tinyinference_llm::model::{
    CapabilitySet, ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStreamItem,
    ReasoningConfig, ReasoningEffort, ResponseFormat, SchemaTransform, StructuredMode, ToolChoice,
};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::{ToolCall, ToolSchema};
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolContent, ToolResult, ToolTimeout};

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
impl Tool for FakeTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "fake tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success(self.reply))
    }
}

/// A tool that sleeps `delay` before returning a fixed reply, used to prove a
/// hanging tool call is bounded by the run's remaining wall-clock budget the
/// same way a hanging model call is.
struct SlowTool {
    delay: std::time::Duration,
}

#[async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &str {
        "slow"
    }
    fn description(&self) -> &str {
        "slow tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        tokio::time::sleep(self.delay).await;
        Ok(ToolResult::success("too late"))
    }
}

/// A slow tool with an explicit timeout policy, used to prove the runtime
/// consumes the crate timeout vocabulary rather than only the run deadline.
struct PolicySlowTool {
    name: &'static str,
    delay: std::time::Duration,
    timeout: ToolTimeout,
}

#[async_trait]
impl Tool for PolicySlowTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "policy-bounded slow tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn timeout_policy(&self, arguments: &serde_json::Value) -> ToolTimeout {
        if arguments
            .get("unbounded")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            ToolTimeout::Unbounded
        } else {
            self.timeout
        }
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        tokio::time::sleep(self.delay).await;
        Ok(ToolResult::success("too late"))
    }
}

/// A strict tool used to prove harness-level schema validation runs before the
/// tool implementation is invoked.
struct StrictLookupTool {
    calls: Arc<Mutex<usize>>,
}

struct StringEchoTool {
    calls: Arc<Mutex<usize>>,
}

struct RequiredOnlyTool {
    calls: Arc<Mutex<usize>>,
}

struct ObjectEnumTool {
    calls: Arc<Mutex<usize>>,
}

struct ObjectArrayTool {
    calls: Arc<Mutex<usize>>,
}

struct OptionalStrictObjectTool {
    calls: Arc<Mutex<usize>>,
}

#[async_trait]
impl Tool for StringEchoTool {
    fn name(&self) -> &str {
        "string_echo"
    }

    fn description(&self) -> &str {
        "echo a string"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": ["object", "string"],
            "properties": { "value": { "type": "string" } }
        })
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success(arguments.to_string()))
    }
}

#[async_trait]
impl Tool for RequiredOnlyTool {
    fn name(&self) -> &str {
        "required_only"
    }

    fn description(&self) -> &str {
        "accepts an implicitly object-shaped schema"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({ "required": ["query"] })
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success("required-output"))
    }
}

#[async_trait]
impl Tool for ObjectEnumTool {
    fn name(&self) -> &str {
        "object_enum"
    }

    fn description(&self) -> &str {
        "accepts one enumerated object"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({ "enum": [{ "query": "rust" }] })
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success("enum-output"))
    }
}

#[async_trait]
impl Tool for ObjectArrayTool {
    fn name(&self) -> &str {
        "object_array"
    }

    fn description(&self) -> &str {
        "accepts an object or array"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": ["object", "array"],
            "properties": { "value": { "type": "string" } }
        })
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success(arguments.to_string()))
    }
}

#[async_trait]
impl Tool for OptionalStrictObjectTool {
    fn name(&self) -> &str {
        "optional_strict_object"
    }

    fn description(&self) -> &str {
        "accepts only an optional query field"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": { "query": { "type": "string" } }
        })
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success("unexpected"))
    }
}

#[async_trait]
impl Tool for StrictLookupTool {
    fn name(&self) -> &str {
        "strict_lookup"
    }
    fn description(&self) -> &str {
        "strict lookup"
    }
    fn parameters_schema(&self) -> serde_json::Value {
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
        })
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success("strict-output"))
    }
}

/// A [`crate::tool::toolset::ToolSet`] whose live tool set changes on its
/// second call — used to prove `agent_loop::tool_changes`'s wiring actually
/// fires on a genuine mid-run toolset change (B6).
struct DynamicToolSet {
    calls: std::sync::atomic::AtomicUsize,
    search: Arc<dyn Tool>,
    browse: Arc<dyn Tool>,
}

#[async_trait]
impl crate::tool::toolset::ToolSet<(), ()> for DynamicToolSet {
    async fn tools(&self, _ctx: &RunContext<()>) -> Result<Vec<Arc<dyn Tool>>> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            Ok(vec![self.search.clone()])
        } else {
            Ok(vec![self.search.clone(), self.browse.clone()])
        }
    }

    async fn call(
        &self,
        name: &str,
        args: serde_json::Value,
        _ctx: &RunContext<()>,
    ) -> Result<ToolResult> {
        let tool = if name == self.search.name() {
            &self.search
        } else if name == self.browse.name() {
            &self.browse
        } else {
            return Err(TinyAgentsError::ToolNotFound(name.to_string()));
        };
        tool.execute(args)
            .await
            .map_err(|err| TinyAgentsError::Tool(err.to_string()))
    }
}

/// Wraps [`MockModel`] to advertise a caller-supplied [`ModelProfile`]
/// instead of the fixed permissive one `MockModel::profile` returns — used to
/// exercise the `mid_conversation_system_messages = true` insert path (B6)
/// end to end, which no built-in test provider otherwise advertises.
struct ProfiledModel {
    inner: MockModel,
    profile: ModelProfile,
}

#[async_trait]
impl ChatModel<()> for ProfiledModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    async fn invoke(
        &self,
        state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        <MockModel as ChatModel<()>>::invoke(&self.inner, state, request).await
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
            origin: None,
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// Builds a tool-call response whose arguments the provider could not parse,
/// mirroring what the OpenAI provider produces for a small local model that
/// emitted malformed argument JSON: the raw string is preserved and the call is
/// marked [`ToolCall::invalid`].
fn invalid_tool_call_response(id: &str, name: &str, raw: &str) -> ModelResponse {
    let reason = format!(
        "openai response contained invalid JSON arguments for tool call `{id}` (`{name}`): \
         EOF while parsing a value; raw arguments: {raw:?}"
    );
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::invalid(id, name, raw, reason)],
            usage: Some(Usage::new(7, 3)),
            origin: None,
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
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
            origin: None,
        },
        usage: Some(Usage::new(input, output)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// Builds a *truncated empty* completion: `finish_reason == "length"` with no
/// text, no tool calls, and no structured output — the failure mode of a local
/// reasoning model that burned its whole token budget on the hidden reasoning
/// channel. `reasoning_tokens` is folded into `output_tokens` so the usage
/// mirrors a real length-truncated response.
fn truncated_empty_response(reasoning_tokens: u64) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, reasoning_tokens)),
            origin: None,
        },
        usage: Some(Usage::new(4, reasoning_tokens)),
        finish_reason: Some("length".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// A provider may finish normally after emitting only the hidden reasoning
/// channel. Unlike a length-truncated response, another call should keep the
/// same output-token limit rather than doubling it.
fn reasoning_only_stop_response() -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Thinking {
                text: "A greeting that never reached visible content".into(),
                signature: None,
            }],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, 20)),
            origin: None,
        },
        usage: Some(Usage::new(4, 20)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// Middleware that appends a user message to every model request.
struct InjectMiddleware {
    text: &'static str,
}

struct ToolInvocationRecorder {
    seen: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl Middleware<(), ()> for ToolInvocationRecorder {
    fn name(&self) -> &str {
        "tool-invocation-recorder"
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        invocation: &ToolInvocationIdentity,
        _result: &mut ToolResult,
    ) -> Result<()> {
        self.seen.lock().expect("recorder lock").push((
            invocation.call_id().as_str().to_string(),
            invocation.tool_name().to_string(),
        ));
        Ok(())
    }
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
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
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

/// A model whose profile lacks native structured output, paired with a
/// forced P-Format dialect: unlike [`ToolStructuredModel`], the loop strips
/// *every* schema off the wire for a text dialect (including the synthetic
/// structured-output fallback tool), so this narrates the call back in
/// P-Format syntax — `name[0|value|1|value]` — instead of returning a
/// structured `tool_calls` entry.
struct PFormatStructuredModel {
    profile: ModelProfile,
    received: Mutex<Vec<ModelRequest>>,
}

impl PFormatStructuredModel {
    fn new() -> Self {
        Self {
            profile: ModelProfile {
                tool_calling: true,
                native_structured_output: false,
                json_schema: false,
                ..ModelProfile::default()
            },
            received: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ChatModel<()> for PFormatStructuredModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.received
            .lock()
            .expect("PFormatStructuredModel received lock poisoned")
            .push(request);
        Ok(ModelResponse::assistant(
            "<tool_call>answer[0|viatool|1|7]</tool_call>",
        ))
    }
}

/// A model that always fails with a retryable error and counts attempts.
struct FailingModel {
    attempts: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for FailingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Err(tinyinference_llm::Error::Model(
            "transient boom".to_string(),
        ))
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
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.timestamps
            .lock()
            .unwrap()
            .push(tokio::time::Instant::now());
        Err(tinyinference_llm::Error::Model(
            "transient boom".to_string(),
        ))
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
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Err(tinyinference_llm::Error::Provider(Box::new(
            tinyinference_llm::model::ProviderError {
                provider: "test-provider".to_string(),
                status: Some(self.status),
                retryable: self.retryable,
                retry_after_ms: None,
                message: "boom".to_string(),
                ..tinyinference_llm::model::ProviderError::default()
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
        result.content = vec![ToolContent::Text {
            text: format!("[wrapped] {}", result.output()),
        }];
        Ok(result.into())
    }
}

/// Rewrites the call to opt out of the inherited timeout, proving timeout
/// resolution observes the call that actually reaches the tool.
struct UnboundToolWrap;

#[async_trait]
impl ToolMiddleware<()> for UnboundToolWrap {
    fn name(&self) -> &str {
        "unbound_tool"
    }
    async fn wrap_tool(
        &self,
        ctx: &mut RunContext<()>,
        state: &(),
        mut call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        call.arguments["unbounded"] = json!(true);
        next.run(ctx, state, call).await
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

/// Host-owned memory policy expressed entirely as around-agent middleware.
struct HostMemoryMiddleware {
    finalized: Arc<Mutex<Vec<(bool, usize)>>>,
}

#[async_trait]
impl AgentMiddleware<()> for HostMemoryMiddleware {
    fn name(&self) -> &str {
        "host_memory"
    }

    async fn wrap_agent(
        &self,
        ctx: &mut RunContext<()>,
        state: &(),
        mut request: AgentRequest,
        run: &mut AgentRun,
        next: AgentHandler<'_, (), ()>,
    ) -> Result<()> {
        request
            .input
            .insert(0, Message::system("remembered by the host"));
        let outcome = next.run(ctx, state, request, run).await;
        self.finalized
            .lock()
            .unwrap()
            .push((outcome.is_ok(), run.messages.len()));
        outcome
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn agent_middleware_owns_context_and_observes_failed_partial_runs() {
    let finalized = Arc::new(Mutex::new(Vec::new()));
    let middleware = Arc::new(HostMemoryMiddleware {
        finalized: finalized.clone(),
    });

    let mut successful: AgentHarness<()> = AgentHarness::new();
    successful.register_model("mock", Arc::new(MockModel::constant("done")));
    successful.push_agent_middleware(middleware.clone());
    let run = successful
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .unwrap();
    assert_eq!(run.messages[0].text(), "remembered by the host");

    let mut failing: AgentHarness<()> = AgentHarness::new();
    failing.register_model(
        "mock",
        Arc::new(FailingModel {
            attempts: Mutex::new(0),
        }),
    );
    failing.push_agent_middleware(middleware);
    assert!(
        failing
            .invoke_default(&(), vec![Message::user("fail")])
            .await
            .is_err()
    );

    assert_eq!(&*finalized.lock().unwrap(), &[(true, 3), (false, 2)]);
}

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
async fn empty_response_fails_the_run_when_guard_enabled() {
    // openhuman#4638: an empty provider completion (no text, no tool calls, no
    // structured output) must not terminate the run with a blank final answer
    // when the guard is enabled — it fails with a typed `EmptyResponse` so the
    // caller can re-prompt instead of silently succeeding on empty content.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("")));
    harness.with_policy(RunPolicy {
        error_on_empty_response: true,
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("an empty response should fail the run");
    assert!(
        matches!(err, TinyAgentsError::EmptyResponse),
        "expected EmptyResponse, got {err:?}"
    );
}

#[tokio::test]
async fn empty_response_terminates_normally_when_guard_disabled() {
    // The guard is opt-in: with the default policy an empty completion still
    // terminates the run with a blank final answer (preserved behavior).
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("")));

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds with a blank final by default");
    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text(), Some(String::new()));
}

#[tokio::test]
async fn reasoning_only_stop_retries_once_when_enabled() {
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        reasoning_only_stop_response(),
        text_response("recovered", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_policy(RunPolicy {
        empty_response_retries: 1,
        ..RunPolicy::default()
    });

    let ctx = RunContext::new(
        RunConfig::new("reasoning-only-retry").with_max_turn_output_tokens(2048),
        (),
    );
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("a second usable response should recover the turn");

    assert_eq!(run.text(), Some("recovered".to_string()));
    assert_eq!(run.model_calls, 2);
    assert_eq!(run.messages.len(), 2, "discard the blank assistant row");
    assert_eq!(
        model
            .requests()
            .iter()
            .map(|request| request.max_tokens)
            .collect::<Vec<_>>(),
        vec![Some(2048), Some(2048)],
        "a non-truncated empty completion must not boost the token cap"
    );
}

#[tokio::test]
async fn reasoning_only_stop_keeps_default_blank_final() {
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        reasoning_only_stop_response(),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("the new retry remains opt-in");
    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text(), Some(String::new()));
}

#[tokio::test]
async fn reasoning_only_stop_retry_exhaustion_respects_empty_guard() {
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        reasoning_only_stop_response(),
        reasoning_only_stop_response(),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_policy(RunPolicy {
        empty_response_retries: 1,
        error_on_empty_response: true,
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("two reasoning-only completions must fail the guarded run");
    assert!(matches!(err, TinyAgentsError::EmptyResponse));
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn empty_response_retry_budget_resets_after_a_tool_turn() {
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        reasoning_only_stop_response(),
        tool_call_response("c1", "fake", json!({})),
        reasoning_only_stop_response(),
        text_response("done", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::clone(&model) as _)
        .register_tool(Arc::new(FakeTool::new("fake", "tool output")));
    harness.with_policy(RunPolicy {
        empty_response_retries: 1,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("each logical turn gets its own retry allowance");
    assert_eq!(run.text(), Some("done".to_string()));
    assert_eq!(run.model_calls, 4);
}

#[tokio::test]
async fn empty_response_retry_does_not_replace_an_explicit_continuation() {
    let mut continuing = reasoning_only_stop_response();
    continuing.continue_turn = Some("Please finish your answer".to_string());
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        continuing,
        text_response("done", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_policy(RunPolicy {
        empty_response_retries: 1,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("the model's explicit continuation should run");
    assert_eq!(run.text(), Some("done".to_string()));
    assert_eq!(run.model_calls, 2);
    assert_eq!(
        run.messages.len(),
        4,
        "retain the continuing assistant row and nudge"
    );
}

#[tokio::test]
async fn empty_response_retry_does_not_replay_a_cached_blank() {
    use crate::cache::InMemoryResponseCache;

    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        reasoning_only_stop_response(),
        text_response("recovered", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_response_cache(Arc::new(InMemoryResponseCache::new()));
    harness.with_policy(RunPolicy {
        empty_response_retries: 1,
        ..RunPolicy::default()
    });

    let input = vec![Message::user("same request")];
    let first = harness
        .invoke_default(&(), input.clone())
        .await
        .expect("the retry must reach the provider, not the cached blank");
    assert_eq!(first.text(), Some("recovered".to_string()));
    assert_eq!(model.requests().len(), 2);

    let second = harness
        .invoke_default(&(), input)
        .await
        .expect("the usable answer should be cached");
    assert_eq!(second.text(), Some("recovered".to_string()));
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn empty_response_retry_ignores_a_blank_cached_before_opt_in() {
    use crate::cache::InMemoryResponseCache;

    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        reasoning_only_stop_response(),
        text_response("recovered", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_response_cache(Arc::new(InMemoryResponseCache::new()));
    let input = vec![Message::user("same request")];

    let first = harness
        .invoke_default(&(), input.clone())
        .await
        .expect("default policy still accepts a blank final");
    assert_eq!(first.text(), Some(String::new()));
    assert_eq!(model.requests().len(), 1);

    harness.with_policy(RunPolicy {
        empty_response_retries: 1,
        ..RunPolicy::default()
    });
    let second = harness
        .invoke_default(&(), input)
        .await
        .expect("an old blank cache entry must not block recovery");
    assert_eq!(second.text(), Some("recovered".to_string()));
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn truncated_empty_retry_does_not_replay_a_cached_blank() {
    use crate::cache::InMemoryResponseCache;

    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(2048),
        text_response("recovered", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_response_cache(Arc::new(InMemoryResponseCache::new()));

    let ctx = RunContext::new(
        RunConfig::new("truncated-empty-cache").with_max_turn_output_tokens(2048),
        (),
    );
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("length-truncated empty completion should reach the provider again");
    assert_eq!(run.text(), Some("recovered".to_string()));
    assert_eq!(model.requests().len(), 2);
    assert_eq!(model.requests()[1].max_tokens, Some(4096));
}

#[tokio::test]
async fn truncated_empty_response_retries_then_succeeds() {
    // A local reasoning model burns its whole token budget on the hidden
    // reasoning channel and returns finish_reason="length" with empty content.
    // The loop must recover automatically: retry the call with a doubled token
    // budget and finish on the second, usable response.
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(2048),
        text_response("recovered", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);

    use crate::testkit::EventRecorder;
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(
        RunConfig::new("truncated-retry").with_max_turn_output_tokens(2048),
        (),
    )
    .with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the truncated-empty response should be retried, not surfaced");

    assert_eq!(run.text(), Some("recovered".to_string()));
    assert_eq!(
        run.model_calls, 2,
        "the retry counts as a second model call"
    );
    assert!(
        recorder
            .events()
            .iter()
            .any(|e| matches!(e, AgentEvent::RetryScheduled { attempt: 1, .. })),
        "the retry should be observable; got kinds {:?}",
        recorder.kinds()
    );
    // The first attempt sent the configured 2048 cap; the retry doubled it.
    let sent: Vec<Option<u32>> = model.requests().iter().map(|r| r.max_tokens).collect();
    assert_eq!(
        sent,
        vec![Some(2048), Some(4096)],
        "the retried request should carry double the original token budget"
    );
}

#[tokio::test]
async fn truncated_empty_boost_does_not_leak_into_later_turns() {
    // Regression test: the boosted token cap and the retry counter are per-turn
    // recovery state, but they used to live for the whole run — so every turn
    // after a recovered one was dispatched at the boosted cap (overriding the
    // caller's `max_turn_output_tokens`) and a later truncation got no retry.
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(2048),
        tool_call_response("c1", "fake", json!({})),
        text_response("done", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::clone(&model) as _)
        .register_tool(Arc::new(FakeTool::new("fake", "tool output")));

    let ctx = RunContext::new(
        RunConfig::new("truncated-leak").with_max_turn_output_tokens(2048),
        (),
    );
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the recovered run finishes");

    assert_eq!(run.text(), Some("done".to_string()));
    let sent: Vec<Option<u32>> = model.requests().iter().map(|r| r.max_tokens).collect();
    assert_eq!(
        sent,
        vec![Some(2048), Some(4096), Some(2048)],
        "only the retry of the truncated turn carries the boost; the next turn \
         is back at the configured per-turn cap"
    );
}

#[tokio::test]
async fn truncated_empty_retry_budget_is_restored_for_a_later_turn() {
    // The retry budget is per turn too: a second truncated-empty completion in a
    // later turn must still be recoverable with the default single retry.
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(2048),
        tool_call_response("c1", "fake", json!({})),
        truncated_empty_response(2048),
        text_response("done", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::clone(&model) as _)
        .register_tool(Arc::new(FakeTool::new("fake", "tool output")));

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("both truncated turns recover");

    assert_eq!(run.text(), Some("done".to_string()));
    assert_eq!(run.model_calls, 4);
}

#[tokio::test]
async fn truncated_empty_retry_budget_stays_unset_when_request_had_none() {
    // With no per-turn token cap the budget cannot be doubled, but the retry is
    // still worthwhile because the failure is stochastic. The retried request
    // simply carries `None` again.
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(64),
        text_response("recovered", 4, 3),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("a plain retry still recovers a stochastic truncation");

    assert_eq!(run.text(), Some("recovered".to_string()));
    assert_eq!(run.model_calls, 2);
    let sent: Vec<Option<u32>> = model.requests().iter().map(|r| r.max_tokens).collect();
    assert_eq!(sent, vec![None, None]);
}

#[tokio::test]
async fn truncated_empty_retries_exhausted_returns_blank_by_default() {
    // When every attempt truncates, the retry budget is exhausted and the
    // historical behavior applies: a blank final success (the empty-response
    // guard is off by default).
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(2048),
        truncated_empty_response(4096),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);

    let ctx = RunContext::new(
        RunConfig::new("truncated-exhausted").with_max_turn_output_tokens(2048),
        (),
    );
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("exhausted retries fall back to the blank-final behavior");

    assert_eq!(run.text(), Some(String::new()));
    assert_eq!(run.model_calls, 2, "one original attempt plus one retry");
}

#[tokio::test]
async fn truncated_empty_retries_exhausted_errors_when_guard_enabled() {
    // With the empty-response guard on, exhausting the retries surfaces the
    // typed EmptyResponse error rather than a blank success.
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(2048),
        truncated_empty_response(4096),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_policy(RunPolicy {
        error_on_empty_response: true,
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("exhausted retries with the guard on should fail");
    assert!(
        matches!(err, TinyAgentsError::EmptyResponse),
        "expected EmptyResponse, got {err:?}"
    );
}

#[tokio::test]
async fn truncated_empty_retry_disabled_by_zero_policy() {
    // truncated_empty_retries=0 restores exact-replay behavior: no retry, a
    // single model call, blank final.
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![
        truncated_empty_response(2048),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.with_policy(RunPolicy {
        truncated_empty_retries: 0,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("with retries disabled the blank final is returned");
    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text(), Some(String::new()));
}

#[tokio::test]
async fn length_finish_with_text_is_not_treated_as_truncated_empty() {
    // A length-truncated response that still carries visible text is a real
    // answer and must not trigger the retry.
    let mut partial = text_response("partial answer", 4, 2048);
    partial.finish_reason = Some("length".to_string());
    let model = Arc::new(crate::testkit::ScriptedModel::new(vec![partial]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("a length response with text is a normal final");
    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text(), Some("partial answer".to_string()));
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

/// B6: a toolset chain whose live set changes mid-run, against a model whose
/// profile does *not* advertise `mid_conversation_system_messages` (the
/// default `MockModel` profile), gets the delta **folded** into the leading
/// system message rather than appended as a new one — no new
/// [`Message::System`] appears anywhere in the transcript, but the delta is
/// still fully recorded on the leading message.
#[tokio::test]
async fn dynamic_toolset_change_folds_into_the_leading_system_message_by_default() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "search", json!({"q": "x"})),
            text_response("done", 4, 2),
        ])),
    );
    let search: Arc<dyn Tool> = Arc::new(FakeTool::new("search", "search-output"));
    let browse: Arc<dyn Tool> = Arc::new(FakeTool::new("browse", "browse-output"));
    let toolset = Arc::new(DynamicToolSet {
        calls: std::sync::atomic::AtomicUsize::new(0),
        search: search.clone(),
        browse: browse.clone(),
    });
    harness.with_toolset(toolset.clone());
    // Only `search` needs to be dispatchable (the script's only call); a
    // bridge for `browse` too would register it in `self.tools` statically
    // from turn one, defeating the point of this test (the toolset chain
    // alone is what makes `browse` come and go).
    harness.register_tool_dispatch(Arc::new(crate::tool::toolset::ToolSetDispatchBridge::new(
        toolset, search,
    )));
    let _ = browse;

    let run = harness
        .invoke_default(
            &(),
            vec![Message::system("baseline persona"), Message::user("go")],
        )
        .await
        .expect("run succeeds");

    let system_messages: Vec<&Message> = run
        .messages
        .iter()
        .filter(|message| matches!(message, Message::System(_)))
        .collect();
    assert_eq!(
        system_messages.len(),
        1,
        "no new system message was appended"
    );
    let Message::System(leading) = system_messages[0] else {
        unreachable!("filtered above");
    };
    // Turn 1's diff runs against an as-yet-undeclared transcript, so it
    // records the whole live set (both `search` and `browse`) in one fold —
    // not just the later delta — which is what lets a replay reconstruct the
    // complete effective tool set from the transcript alone.
    assert_eq!(
        leading
            .tools_added
            .iter()
            .map(|schema| schema.name.as_str())
            .collect::<Vec<_>>(),
        vec!["browse", "search"]
    );
}

/// B6 (insert path): the same toolset-change scenario against a model whose
/// profile *does* advertise `mid_conversation_system_messages` gets the delta
/// appended as exactly one new [`Message::System`] patch instead.
#[tokio::test]
async fn dynamic_toolset_change_appends_exactly_one_patch_when_the_profile_allows_it() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let profile = ModelProfile {
        tool_calling: true,
        mid_conversation_system_messages: true,
        ..ModelProfile::default()
    };
    harness.register_model(
        "mock",
        Arc::new(ProfiledModel {
            inner: MockModel::with_responses(vec![
                tool_call_response("call-1", "search", json!({"q": "x"})),
                text_response("done", 4, 2),
            ]),
            profile,
        }),
    );
    let search: Arc<dyn Tool> = Arc::new(FakeTool::new("search", "search-output"));
    let browse: Arc<dyn Tool> = Arc::new(FakeTool::new("browse", "browse-output"));
    let toolset = Arc::new(DynamicToolSet {
        calls: std::sync::atomic::AtomicUsize::new(0),
        search: search.clone(),
        browse: browse.clone(),
    });
    harness.with_toolset(toolset.clone());
    harness.register_tool_dispatch(Arc::new(crate::tool::toolset::ToolSetDispatchBridge::new(
        toolset, search,
    )));
    let _ = browse;

    let run = harness
        .invoke_default(
            &(),
            vec![Message::system("baseline persona"), Message::user("go")],
        )
        .await
        .expect("run succeeds");

    let system_messages: Vec<&Message> = run
        .messages
        .iter()
        .filter(|message| matches!(message, Message::System(_)))
        .collect();
    // The original leading system message plus exactly one appended patch.
    assert_eq!(system_messages.len(), 2, "exactly one patch was appended");
    let Message::System(patch) = system_messages[1] else {
        unreachable!("filtered above");
    };
    // As in the fold-path test above, turn 1's patch declares the whole live
    // set, not just a later delta.
    assert_eq!(
        patch
            .tools_added
            .iter()
            .map(|schema| schema.name.as_str())
            .collect::<Vec<_>>(),
        vec!["browse", "search"]
    );

    // The reconstructed effective tool set matches what was actually offered.
    let (_, effective_tools) = tinyinference_llm::message::replay_system_state(&run.messages);
    let mut names: Vec<&str> = effective_tools
        .iter()
        .map(|schema| schema.name.as_str())
        .collect();
    names.sort();
    assert_eq!(names, vec!["browse", "search"]);
}

/// Gap G3: a `defer_loading` capability's tool is not advertised until the
/// model calls `load_capability`, and once it does, the very next turn's
/// existing tool-change diff (B6, `agent_loop::tool_changes`) picks up the
/// change automatically and appends a patch system message — no bespoke
/// capability-specific patch wiring is needed.
#[tokio::test]
async fn defer_loading_capability_is_exposed_only_after_load_capability_and_patches_the_transcript()
{
    let profile = ModelProfile {
        tool_calling: true,
        mid_conversation_system_messages: true,
        ..ModelProfile::default()
    };
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(ProfiledModel {
            inner: MockModel::with_responses(vec![
                tool_call_response(
                    "call-1",
                    crate::capability::LOAD_CAPABILITY_TOOL_NAME,
                    json!({"capability": "advanced"}),
                ),
                // Turn 2 makes no tool call — the point of this test is the
                // *advertisement* change the loop's existing tool-change diff
                // (B6) picks up before this turn's request goes out, not
                // `advanced-tool`'s own dispatch (a deferred capability's
                // tools are advertised automatically but, like any
                // `with_toolset` toolset, need an explicit
                // `ToolSetDispatchBridge` to also be *callable* — see that
                // type's doc comment; orthogonal to what this test covers).
                text_response("done", 4, 2),
            ]),
            profile,
        }),
    );

    // A minimal single-tool toolset behind the capability, independent of
    // `defer_loading` gating (that gating is `CapabilityToolSet`'s job, one
    // layer up).
    struct SingleToolSet {
        tool: Arc<dyn Tool>,
    }
    #[async_trait]
    impl crate::tool::toolset::ToolSet<(), ()> for SingleToolSet {
        async fn tools(&self, _ctx: &RunContext<()>) -> Result<Vec<Arc<dyn Tool>>> {
            Ok(vec![self.tool.clone()])
        }
        async fn call(
            &self,
            name: &str,
            args: serde_json::Value,
            _ctx: &RunContext<()>,
        ) -> Result<ToolResult> {
            if name == self.tool.name() {
                self.tool
                    .execute(args)
                    .await
                    .map_err(|err| TinyAgentsError::Tool(err.to_string()))
            } else {
                Err(TinyAgentsError::ToolNotFound(name.to_string()))
            }
        }
    }

    let capability = crate::capability::Capability::new("advanced")
        .with_instructions("Advanced instructions.")
        .with_toolset(Arc::new(SingleToolSet {
            tool: Arc::new(FakeTool::new("advanced-tool", "advanced-output")),
        }))
        .with_defer_loading(true);
    harness.with_capability(capability);

    let run = harness
        .invoke_default(
            &(),
            vec![Message::system("baseline persona"), Message::user("go")],
        )
        .await
        .expect("run succeeds");

    assert_eq!(run.text(), Some("done".to_string()));

    let system_messages: Vec<&Message> = run
        .messages
        .iter()
        .filter(|message| matches!(message, Message::System(_)))
        .collect();
    // The original leading system message, plus one patch for turn 1 (just
    // `load_capability` itself — `advanced-tool` is still gated), plus one
    // patch for turn 2 (once `load_capability` ran, `advanced-tool` joins the
    // live set).
    assert_eq!(
        system_messages.len(),
        3,
        "expected the leading message plus two tool-change patches"
    );

    let Message::System(turn1_patch) = system_messages[1] else {
        unreachable!("filtered above");
    };
    let turn1_added: Vec<&str> = turn1_patch
        .tools_added
        .iter()
        .map(|schema| schema.name.as_str())
        .collect();
    assert_eq!(
        turn1_added,
        vec![crate::capability::LOAD_CAPABILITY_TOOL_NAME]
    );
    assert!(
        !turn1_added.contains(&"advanced-tool"),
        "the deferred capability's tool must not be advertised before load_capability runs"
    );

    let Message::System(turn2_patch) = system_messages[2] else {
        unreachable!("filtered above");
    };
    assert_eq!(
        turn2_patch
            .tools_added
            .iter()
            .map(|schema| schema.name.as_str())
            .collect::<Vec<_>>(),
        vec!["advanced-tool"],
        "advanced-tool becomes advertised only on the turn after load_capability ran"
    );

    // The final effective tool set (replayed from the transcript alone)
    // includes both the always-registered `load_capability` and the now
    // loaded `advanced-tool`.
    let (_, effective_tools) = tinyinference_llm::message::replay_system_state(&run.messages);
    let mut names: Vec<&str> = effective_tools
        .iter()
        .map(|schema| schema.name.as_str())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "advanced-tool",
            crate::capability::LOAD_CAPABILITY_TOOL_NAME
        ]
    );
}

#[tokio::test]
async fn after_tool_receives_distinct_identity_for_same_named_calls() {
    let mut parallel_calls = ModelResponse::assistant("");
    parallel_calls.message.tool_calls = vec![
        ToolCall::new("lookup-one", "lookup", json!({"q": "one"})),
        ToolCall::new("lookup-two", "lookup", json!({"q": "two"})),
    ];
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            parallel_calls,
            text_response("done", 1, 1),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::new("lookup", "tool-output")));
    harness.push_middleware(Arc::new(ToolInvocationRecorder {
        seen: Arc::clone(&seen),
    }));

    harness
        .invoke_default(&(), vec![Message::user("look up both")])
        .await
        .expect("parallel calls complete");

    assert_eq!(
        *seen.lock().expect("recorder lock"),
        vec![
            ("lookup-one".to_string(), "lookup".to_string()),
            ("lookup-two".to_string(), "lookup".to_string()),
        ]
    );
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
async fn prefix_mutating_middleware_refreshes_provider_cache_key() {
    use crate::cache::{PROMPT_CACHE_KEY_OPTION, prompt_cache_key};
    use tinyinference_llm::cache::CachePolicy;

    struct InsertSystemPrefix;

    #[async_trait]
    impl Middleware<(), ()> for InsertSystemPrefix {
        fn name(&self) -> &str {
            "insert-system-prefix"
        }

        async fn before_model(
            &self,
            _ctx: &mut RunContext<()>,
            _state: &(),
            request: &mut ModelRequest,
        ) -> Result<()> {
            request
                .messages
                .insert(0, Message::system("tenant-specific policy"));
            Ok(())
        }
    }

    let model = Arc::new(crate::testkit::ScriptedModel::replies(vec!["done"]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_policy(RunPolicy {
        cache: CachePolicy {
            protect_prompt_prefix: true,
            ..CachePolicy::default()
        },
        ..RunPolicy::default()
    });
    harness.push_middleware(Arc::new(InsertSystemPrefix));

    harness
        .invoke_default(&(), vec![Message::user("hello")])
        .await
        .expect("run succeeds");

    let request = model
        .requests()
        .into_iter()
        .next()
        .expect("model received one request");
    let mut expected = crate::prompt::PromptBuilder::new();
    expected.push_system("system", vec![Message::system("tenant-specific policy")]);
    assert_eq!(
        request.prompt_fingerprint,
        expected.build(Vec::new()).prompt_fingerprint
    );
    assert_eq!(
        request.provider_options[PROMPT_CACHE_KEY_OPTION],
        serde_json::Value::String(prompt_cache_key(&request).expect("cache key is derived"))
    );
}

#[tokio::test]
async fn prefix_mutating_wrap_middleware_refreshes_provider_cache_key() {
    use crate::cache::{PROMPT_CACHE_KEY_OPTION, prompt_cache_key};
    use tinyinference_llm::cache::CachePolicy;

    struct InsertSystemPrefix;

    #[async_trait]
    impl ModelMiddleware<(), ()> for InsertSystemPrefix {
        fn name(&self) -> &str {
            "insert-system-prefix"
        }

        async fn wrap_model(
            &self,
            ctx: &mut RunContext<()>,
            state: &(),
            mut request: ModelRequest,
            next: ModelHandler<'_, (), ()>,
        ) -> Result<MiddlewareModelOutcome> {
            request
                .messages
                .insert(0, Message::system("tenant-specific policy"));
            next.run(ctx, state, request).await
        }
    }

    let model = Arc::new(crate::testkit::ScriptedModel::replies(vec!["done"]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_policy(RunPolicy {
        cache: CachePolicy {
            protect_prompt_prefix: true,
            ..CachePolicy::default()
        },
        ..RunPolicy::default()
    });
    harness.push_model_middleware(Arc::new(InsertSystemPrefix));

    harness
        .invoke_default(&(), vec![Message::user("hello")])
        .await
        .expect("run succeeds");

    let request = model
        .requests()
        .into_iter()
        .next()
        .expect("model received one request");
    let mut expected = crate::prompt::PromptBuilder::new();
    expected.push_system("system", vec![Message::system("tenant-specific policy")]);
    assert_eq!(
        request.prompt_fingerprint,
        expected.build(Vec::new()).prompt_fingerprint
    );
    assert_eq!(
        request.provider_options[PROMPT_CACHE_KEY_OPTION],
        serde_json::Value::String(prompt_cache_key(&request).expect("cache key is derived"))
    );
}

#[tokio::test]
async fn middleware_owned_cache_segments_are_preserved_and_fingerprinted() {
    use crate::cache::prompt_cache_key;
    use tinyinference_llm::cache::CachePolicy;
    use tinyinference_llm::model::{PromptSegment, SegmentRole};

    struct CustomPrefix;

    #[async_trait]
    impl Middleware<(), ()> for CustomPrefix {
        fn name(&self) -> &str {
            "custom-prefix"
        }

        async fn before_model(
            &self,
            _ctx: &mut RunContext<()>,
            _state: &(),
            request: &mut ModelRequest,
        ) -> Result<()> {
            request
                .messages
                .insert(0, Message::system("tenant context"));
            request.messages.insert(0, Message::system("tenant policy"));
            // These resemble the harness IDs but their order is deliberately
            // middleware-owned. Dispatch must not normalize them.
            request.cache_segments = vec![
                PromptSegment {
                    id: "system.1".to_string(),
                    role: SegmentRole::System,
                    cacheable: true,
                },
                PromptSegment {
                    id: "system".to_string(),
                    role: SegmentRole::System,
                    cacheable: true,
                },
            ];
            Ok(())
        }
    }

    let model = Arc::new(crate::testkit::ScriptedModel::replies(vec!["done"]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_policy(RunPolicy {
        cache: CachePolicy {
            protect_prompt_prefix: true,
            ..CachePolicy::default()
        },
        ..RunPolicy::default()
    });
    harness.push_middleware(Arc::new(CustomPrefix));

    harness
        .invoke_default(&(), vec![Message::user("hello")])
        .await
        .expect("run succeeds");

    let request = model
        .requests()
        .into_iter()
        .next()
        .expect("model received one request");
    assert_eq!(request.cache_segments[0].id, "system.1");
    assert_eq!(request.cache_segments[1].id, "system");
    assert!(request.prompt_fingerprint.is_some());
    assert!(prompt_cache_key(&request).is_some());
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

/// `UnknownToolPolicy::Fail` is opt-in now (the default recovers), so this
/// pins the opted-in fail-closed behavior rather than the default.
#[tokio::test]
async fn tool_not_found_errors_under_the_fail_policy() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_tool_call("missing", json!({}))),
    );
    // No tool registered.
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::Fail,
        ..RunPolicy::default()
    });

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

/// `InvalidArgsPolicy::Fail` is opt-in now (the default recovers), so this pins
/// the opted-in fail-closed behavior rather than the default.
#[tokio::test]
async fn invalid_tool_arguments_fail_before_tool_execution_under_the_fail_policy() {
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
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::Fail,
        ..RunPolicy::default()
    });

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
async fn invalid_tool_arguments_return_tool_error_recovers() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            // `query` is required to be a string; 42 violates the schema, so
            // admission must recover instead of running the tool or aborting.
            tool_call_response("call-1", "strict_lookup", json!({ "query": 42 })),
            text_response("recovered", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(StrictLookupTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::ReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("invalid arguments are recoverable under ReturnToolError");

    assert_eq!(run.final_response.unwrap().text(), "recovered");
    assert_eq!(
        *calls.lock().unwrap(),
        0,
        "the tool implementation must not run on invalid arguments"
    );
    // The injected tool-error message carries the validation detail so the
    // model can self-correct on the next turn.
    let injected = run
        .messages
        .iter()
        .any(|m| format!("{m:?}").contains("invalid arguments for tool `strict_lookup`"));
    assert!(
        injected,
        "recovery message should be injected into the transcript"
    );
}

#[tokio::test]
async fn normalized_json_string_arguments_execute_registered_tool() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response(
                "call-1",
                "strict_lookup",
                json!("```json\n{\"query\":\"rust\"}\n```"),
            ),
            text_response("done", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(StrictLookupTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("a fenced JSON object should be normalized before validation");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn normalized_non_object_executes_tool_without_required_fields() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "permissive", json!(null)),
            text_response("done", 1, 1),
        ])),
    );
    let tool = Arc::new(FakeTool::new("permissive", "ok"));
    harness.register_tool(tool.clone());
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("run")])
        .await
        .expect("a non-object should become an empty object for a permissive schema");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(*tool.calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn normalization_preserves_a_decoded_but_schema_invalid_scalar() {
    // Regression: a stringified JSON scalar (the string `"true"`) decodes
    // successfully to `Value::Bool(true)`, which is schema-invalid for an
    // object schema. That decoded value used to fall through past decode
    // preservation into the has-no-required-fields fallback below — which
    // exists for values that never decoded at all — and get silently
    // replaced with `{}`, letting the tool execute with fabricated empty
    // arguments instead of surfacing the model's real type mismatch.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "permissive", json!("true")),
            text_response("recovered", 1, 1),
        ])),
    );
    let tool = Arc::new(FakeTool::new("permissive", "ok"));
    harness.register_tool(tool.clone());
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("run")])
        .await
        .expect("a decoded-but-invalid scalar is recoverable under ReturnToolError");

    assert_eq!(run.final_response.unwrap().text(), "recovered");
    assert_eq!(
        *tool.calls.lock().unwrap(),
        0,
        "the tool must not run on a schema-invalid decoded scalar"
    );
    let injected = run
        .messages
        .iter()
        .any(|m| format!("{m:?}").contains("invalid arguments for tool `permissive`"));
    assert!(
        injected,
        "the injected message should report the real validation failure, not a fabricated success: {:?}",
        run.messages
    );
}

#[tokio::test]
async fn normalization_preserves_valid_primitive_arguments() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "string_echo", json!("hello")),
            text_response("done", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(StringEchoTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("echo")])
        .await
        .expect("a schema-valid primitive must not be rewritten to an object");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(*calls.lock().unwrap(), 1);
    assert!(
        run.messages
            .iter()
            .any(|message| matches!(message, Message::Tool(_)) && message.text() == "\"hello\""),
        "the tool result must contain the original primitive instead of a rewritten object"
    );
}

#[tokio::test]
async fn normalization_decodes_required_only_object_schema() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "required_only", json!("{\"query\":\"rust\"}")),
            text_response("done", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(RequiredOnlyTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("a required-only object schema should normalize a JSON string");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn normalization_decodes_object_valued_enum_schema() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "object_enum", json!("{\"query\":\"rust\"}")),
            text_response("done", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(ObjectEnumTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("an object-valued enum should normalize a matching JSON string");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn normalization_preserves_decoded_array_union_arguments() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "object_array", json!("[1,2]")),
            text_response("done", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(ObjectArrayTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("an object/array union should preserve a decoded array");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(*calls.lock().unwrap(), 1);
    assert!(
        run.messages
            .iter()
            .any(|message| matches!(message, Message::Tool(_)) && message.text() == "[1,2]"),
        "the tool result must contain the decoded array instead of an empty object"
    );
}

#[tokio::test]
async fn normalization_preserves_decoded_invalid_object_for_validation() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response(
                "call-1",
                "optional_strict_object",
                json!("{\"extra\":true}"),
            ),
            text_response("recovered", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(OptionalStrictObjectTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    use crate::testkit::EventRecorder;
    let recorder = EventRecorder::new();
    let ctx =
        RunContext::new(RunConfig::new("decoded-invalid-args"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("lookup")])
        .await
        .expect("the decoded schema error should be recoverable");

    assert_eq!(run.final_response.unwrap().text(), "recovered");
    assert_eq!(*calls.lock().unwrap(), 0);
    assert!(
        run.messages.iter().any(|message| {
            matches!(message, Message::Tool(_)) && message.text().contains("extra is not allowed")
        }),
        "validation must report the decoded invalid field instead of executing with an empty object"
    );
    assert!(
        recorder.events().iter().any(|event| matches!(
            event,
            AgentEvent::InvalidToolArgs { arguments, .. }
                if arguments == &json!("{\"extra\":true}")
        )),
        "the invalid-args event must retain the raw model-supplied JSON string"
    );
}

#[tokio::test]
async fn normalization_preserves_direct_invalid_object_for_validation() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "optional_strict_object", json!({"extra": true})),
            text_response("recovered", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(OptionalStrictObjectTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("the direct schema error should be recoverable");

    assert_eq!(run.final_response.unwrap().text(), "recovered");
    assert_eq!(*calls.lock().unwrap(), 0);
    assert!(
        run.messages.iter().any(|message| {
            matches!(message, Message::Tool(_)) && message.text().contains("extra is not allowed")
        }),
        "validation must report the direct invalid field instead of executing with an empty object"
    );
}

#[tokio::test]
async fn normalization_preserves_required_field_validation_errors() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "strict_lookup", json!(null)),
            text_response("recovered", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(StrictLookupTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("required-field validation should remain recoverable");

    assert_eq!(run.final_response.unwrap().text(), "recovered");
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn malformed_tool_arguments_recover_as_error_tool_result() {
    // A small local model emitted arguments the provider could not parse, so the
    // response carries a `ToolCall::invalid` call. The loop must feed the parse
    // error back to the model as a tool result and continue — *without* running
    // the tool and *without* failing the run — so it can retry. This holds under
    // the default `InvalidArgsPolicy::Fail` (which governs schema validation of
    // well-formed args, not unparseable ones), proving the leniency is
    // unconditional.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            invalid_tool_call_response("call-x", "strict_lookup", "{\"query\":"),
            text_response("recovered", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(StrictLookupTool {
        calls: Arc::clone(&calls),
    }));

    use crate::testkit::EventRecorder;
    let recorder = EventRecorder::new();
    let ctx =
        RunContext::new(RunConfig::new("malformed-args-run"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("lookup")])
        .await
        .expect("malformed args recover without failing the run");

    assert_eq!(run.final_response.unwrap().text(), "recovered");
    assert_eq!(
        *calls.lock().unwrap(),
        0,
        "the tool implementation must not run on unparseable arguments"
    );
    // The injected tool-error message carries the parse detail so the model can
    // self-correct on the next turn.
    let injected = run
        .messages
        .iter()
        .any(|m| format!("{m:?}").contains("invalid JSON arguments"));
    assert!(
        injected,
        "an error tool result should be injected into the transcript"
    );
}

/// I-13 regression: provider-invalid arguments that `relaxed_json` can
/// actually repair (unquoted object keys, here) must be recovered and the
/// call executed — not turned into a "fix your JSON" round trip the model
/// often cannot act on. Before the fix, admission short-circuited straight
/// to the tool-error path without ever trying `recover_relaxed_object`,
/// even though that module exists specifically for this input shape.
#[tokio::test]
async fn provider_invalid_arguments_recoverable_by_relaxed_json_are_repaired_and_executed() {
    use crate::testkit::EventRecorder;

    let tool = Arc::new(crate::testkit::FakeTool::returning("lookup", "found it"));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            // Unquoted object key: `relaxed_json::recover_relaxed_object`
            // repairs this to `{"query":"weather"}`.
            invalid_tool_call_response("call-x", "lookup", "{query:\"weather\"}"),
            text_response("found it", 1, 1),
        ])),
    );
    harness.register_tool(tool.clone());

    let recorder = EventRecorder::new();
    let ctx =
        RunContext::new(RunConfig::new("relaxed-json-repair"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("lookup the weather")])
        .await
        .expect("repaired arguments let the call execute");

    assert_eq!(run.text().as_deref(), Some("found it"));
    assert_eq!(
        tool.calls(),
        vec![json!({"query": "weather"})],
        "the tool must receive the repaired, strict-JSON arguments"
    );
    assert!(
        recorder.events().iter().any(|event| matches!(
            event,
            AgentEvent::InvalidToolArgs { recovery, .. } if recovery == "repaired"
        )),
        "the repair must be observable as InvalidToolArgs{{ recovery: \"repaired\" }}"
    );
    // The recovery is surfaced as an `InvalidToolArgs` event.
    assert!(
        recorder
            .events()
            .iter()
            .any(|e| matches!(e, AgentEvent::InvalidToolArgs { .. })),
        "an InvalidToolArgs event should be emitted; got kinds {:?}",
        recorder.kinds()
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
async fn pformat_dialect_recovers_the_structured_output_fallback_tool() {
    // The run-level P-Format registry is built once from the schemas offered
    // at the start of the run, before the structured-output fallback tool
    // (`answer`) is pushed onto the request for a non-native model. The
    // catalogue advertising it is rendered fresh from the final tool list on
    // every call, so a model dutifully narrating the call back in P-Format —
    // `answer[0|<value>|1|<score>]` — has to be decodable too, which needs
    // the fallback tool's positional layout in the registry used to parse
    // the answer, not just the one used to render the prompt.
    let model = Arc::new(PFormatStructuredModel::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .with_policy(RunPolicy {
            tool_dialect: crate::config::ToolDispatcher::Pformat,
            default_response_format: Some(ResponseFormat::auto(
                "answer",
                json!({
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                    "required": ["value", "score"],
                }),
            )),
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("answer")])
        .await
        .expect("run succeeds");

    let structured = run.structured.expect("structured output present");
    assert_eq!(structured["value"], "viatool");
    assert_eq!(structured["score"], 7);

    // The catalogue sent to the model already advertised the fallback
    // tool's p-format signature; confirm that, so a failure here could only
    // be the parsing registry, never a missing catalogue entry.
    let request = model
        .received
        .lock()
        .expect("PFormatStructuredModel received lock poisoned")[0]
        .clone();
    let system = request
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(system.contains("answer[0|<value>|1|<score>]"), "{system}");
}

#[tokio::test]
async fn native_tool_dispatcher_requires_tool_calling_capability() {
    // `ToolDispatcher::Native` is documented as *forcing* provider-native
    // tool calls, unlike `Auto`'s "native when available, else Xml". Without
    // a capability requirement that promise was unenforceable at
    // resolution: a model whose profile cannot do native tool calling could
    // still be selected as the (only, default) model and silently receive
    // whatever fallback its own adapter chooses, rather than the run
    // failing closed the way the `Native` name implies.
    let incapable = Arc::new(ProfiledTextModel {
        profile: ModelProfile {
            tool_calling: false,
            ..ModelProfile::default()
        },
        text: "should never be reached",
        attempts: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", incapable.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::new("lookup", "tool-output")))
        .with_policy(RunPolicy {
            tool_dialect: crate::config::ToolDispatcher::Native,
            ..RunPolicy::default()
        });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("no model satisfies the forced-native capability requirement");
    assert!(
        matches!(err, TinyAgentsError::ModelNotFound(_)),
        "got {err:?}"
    );
    assert_eq!(
        *incapable.attempts.lock().unwrap(),
        0,
        "the capability-ineligible model must never be invoked"
    );
}

#[tokio::test]
async fn native_tool_dispatcher_gates_on_the_post_middleware_tool_set() {
    // The capability requirement must be derived from the *effective*
    // request tools, checked after `before_model` middleware has run — not
    // from the earlier `tool_schemas` snapshot taken before it. A run that
    // registers no tools directly but whose `before_model` middleware adds
    // one must still be gated, or that middleware-added tool would silently
    // reach a model that cannot make native tool calls, defeating the
    // `Native` dispatcher's fail-closed promise exactly as if the gate did
    // not exist at all.
    struct InjectToolMiddleware;

    #[async_trait]
    impl Middleware<(), ()> for InjectToolMiddleware {
        fn name(&self) -> &str {
            "inject-tool"
        }
        async fn before_model(
            &self,
            _ctx: &mut RunContext<()>,
            _state: &(),
            request: &mut ModelRequest,
        ) -> Result<()> {
            request.tools.push(ToolSchema::new(
                "lookup",
                "looks something up",
                json!({"type": "object"}),
            ));
            Ok(())
        }
    }

    let incapable = Arc::new(ProfiledTextModel {
        profile: ModelProfile {
            tool_calling: false,
            ..ModelProfile::default()
        },
        text: "should never be reached",
        attempts: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", incapable.clone())
        .set_default_model("mock")
        .push_middleware(Arc::new(InjectToolMiddleware))
        .with_policy(RunPolicy {
            tool_dialect: crate::config::ToolDispatcher::Native,
            ..RunPolicy::default()
        });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("no model satisfies the forced-native capability requirement");
    assert!(
        matches!(err, TinyAgentsError::ModelNotFound(_)),
        "got {err:?}"
    );
    assert_eq!(
        *incapable.attempts.lock().unwrap(),
        0,
        "the capability-ineligible model must never be invoked"
    );
}

#[tokio::test]
async fn native_tool_dispatcher_gates_on_auto_structured_output_with_no_ordinary_tools() {
    // `StructuredStrategy` resolution only ever appends a synthetic
    // tool-call schema for a model whose profile already has `tool_calling`
    // (`StructuredStrategy::for_profile`'s `ToolCall` arm) — but that
    // resolution happens *after* the model is already chosen, so gating on
    // `request.tools` alone (empty here, since no ordinary tool is
    // registered and the synthetic schema hasn't been appended yet at gate
    // time) let an incapable model be selected for a run that would go on to
    // need native tool calling for its `Auto` structured-output fallback.
    let incapable = Arc::new(ProfiledTextModel {
        profile: ModelProfile {
            tool_calling: false,
            ..ModelProfile::default()
        },
        text: "should never be reached",
        attempts: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", incapable.clone())
        .set_default_model("mock")
        .with_policy(RunPolicy {
            tool_dialect: crate::config::ToolDispatcher::Native,
            default_response_format: Some(ResponseFormat::auto(
                "answer",
                json!({"type": "object"}),
            )),
            ..RunPolicy::default()
        });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("no model satisfies the forced-native capability requirement");
    assert!(
        matches!(err, TinyAgentsError::ModelNotFound(_)),
        "got {err:?}"
    );
    assert_eq!(
        *incapable.attempts.lock().unwrap(),
        0,
        "the capability-ineligible model must never be invoked"
    );
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
async fn retry_middleware_and_run_policy_retry_do_not_multiply_attempts() {
    // Regression test (I-7): `RetryMiddleware::wrap_model` retries the whole
    // wrap onion, and `invoke_model_resolving` (the loop's own base call) had
    // its own independent retry loop; with both configured the worst case was
    // `mw.max_attempts x policy.retry.max_attempts` provider calls for one
    // logical failure. A registered `RetryMiddleware` must make the base call
    // skip its own retry loop, so the total attempt count is bounded by the
    // middleware's `max_attempts` alone.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let failing = Arc::new(FailingModel {
        attempts: Mutex::new(0),
    });
    harness.register_model("primary", failing.clone());
    // The middleware allows 3 attempts; the loop's own retry (if it fired
    // too) would allow another 5 — 15 total if the two layers multiplied.
    harness.push_model_middleware(Arc::new(crate::middleware::library::RetryMiddleware::new(
        RetryPolicy::default().with_max_attempts(3),
    )));
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(5),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("FailingModel never succeeds");
    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");

    // Bounded by the middleware's max_attempts (3), not 3 x 5.
    assert_eq!(*failing.attempts.lock().unwrap(), 3);
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

/// A model with an explicit profile that always fails with a retryable error
/// and counts attempts. Unlike [`FailingModel`] it advertises a profile, so it
/// can be selected under a `required_capabilities` gate.
struct ProfiledFailingModel {
    profile: ModelProfile,
    attempts: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for ProfiledFailingModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Err(tinyinference_llm::Error::Model(
            "transient boom".to_string(),
        ))
    }
}

/// A model with an explicit profile that returns fixed text and counts
/// invocations, so a test can prove an ineligible fallback candidate is never
/// invoked while a later eligible one is.
struct ProfiledTextModel {
    profile: ModelProfile,
    text: &'static str,
    attempts: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for ProfiledTextModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Ok(ModelResponse::assistant(self.text))
    }
}

/// Middleware that stamps an explicit model override and a required-capability
/// set onto every request, so a test can drive resolution + fallback under a
/// capability gate without a public request-building entry point.
struct RequireCapsMiddleware {
    model: &'static str,
    caps: CapabilitySet,
}

#[async_trait]
impl Middleware<(), ()> for RequireCapsMiddleware {
    fn name(&self) -> &str {
        "require_caps"
    }
    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.model = Some(self.model.to_string());
        request.required_capabilities = Some(self.caps.clone());
        Ok(())
    }
}

/// Runtime fallback must apply the same capability/lifecycle gate that initial
/// resolution does: on a primary failure, a fallback candidate that cannot
/// satisfy the request's `required_capabilities` is skipped (never invoked) and
/// the chain advances to the next eligible model, emitting `FallbackSkipped`
/// for the skipped candidate (issue #4641).
#[tokio::test]
async fn runtime_fallback_skips_capability_ineligible_candidate() {
    use crate::testkit::EventRecorder;

    let tool_capable = ModelProfile {
        tool_calling: true,
        ..ModelProfile::default()
    };
    let tool_incapable = ModelProfile {
        tool_calling: false,
        ..ModelProfile::default()
    };

    let primary = Arc::new(ProfiledFailingModel {
        profile: tool_capable.clone(),
        attempts: Mutex::new(0),
    });
    // Ineligible under the required `tool_calling` gate: must be skipped and
    // never invoked.
    let backup = Arc::new(ProfiledTextModel {
        profile: tool_incapable,
        text: "from backup",
        attempts: Mutex::new(0),
    });
    // Eligible: the chain must reach and use this one.
    let tertiary = Arc::new(ProfiledTextModel {
        profile: tool_capable,
        text: "recovered",
        attempts: Mutex::new(0),
    });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary.clone());
    harness.register_model("backup", backup.clone());
    harness.register_model("tertiary", tertiary.clone());
    harness.push_middleware(Arc::new(RequireCapsMiddleware {
        model: "primary",
        caps: CapabilitySet {
            tool_calling: true,
            ..CapabilitySet::default()
        },
    }));
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(1),
        fallback: Some(FallbackPolicy::new(["primary", "backup", "tertiary"])),
        ..RunPolicy::default()
    });

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("fallback-caps-run"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("fallback reaches the eligible tertiary model");

    // The ineligible `backup` was skipped; the eligible `tertiary` answered.
    assert_eq!(run.text(), Some("recovered".to_string()));
    assert_eq!(*primary.attempts.lock().unwrap(), 1, "primary tried once");
    assert_eq!(
        *backup.attempts.lock().unwrap(),
        0,
        "capability-ineligible fallback must never be invoked"
    );
    assert_eq!(*tertiary.attempts.lock().unwrap(), 1, "tertiary answered");

    // The skip is observable as a diagnostic event naming the skipped model.
    assert!(
        recorder.events().iter().any(|e| matches!(
            e,
            AgentEvent::FallbackSkipped { model } if model == "backup"
        )),
        "skipped fallback candidate must be surfaced as FallbackSkipped; got kinds {:?}",
        recorder.kinds()
    );
}

/// I-2 end-to-end regression: under the default `Auto` text-dialect recovery
/// policy, a model whose resolved profile reports native tool calling must
/// never have `<tool_call>` markup it merely quotes — here, inside a fenced
/// code block explaining the format — executed as a real tool call. Before
/// the fix, `recover_text_dialect_calls` ran unconditionally whenever the
/// request offered tools and the provider returned no native calls,
/// regardless of the model's own advertised capabilities.
#[tokio::test]
async fn native_tool_calling_model_does_not_execute_quoted_text_dialect_markup() {
    let tool = Arc::new(FakeTool::new("shell", "must not run"));
    let model = Arc::new(ProfiledTextModel {
        profile: ModelProfile {
            tool_calling: true,
            ..ModelProfile::default()
        },
        text: "Here is the tool-call format for reference:\n\
               ```\n\
               <tool_call>{\"name\": \"shell\", \"arguments\": {\"command\": \"id\"}}</tool_call>\n\
               ```\n",
        attempts: Mutex::new(0),
    });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("native", model.clone());
    harness.register_tool(tool.clone());

    let run = harness
        .invoke_default(&(), vec![Message::user("how do tool calls work?")])
        .await
        .expect("run succeeds with a plain text final answer");

    assert_eq!(
        *tool.calls.lock().unwrap(),
        0,
        "the quoted call must not run"
    );
    assert!(run.text().unwrap_or_default().contains("<tool_call>"));
    assert_eq!(
        *model.attempts.lock().unwrap(),
        1,
        "no retry/fallback needed"
    );
}

#[tokio::test]
async fn invoke_with_status_reports_completed() {
    use crate::ids::{ExecutionStatus, HarnessPhase};

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
        delta: &mut tinyinference_llm::model::ModelDelta,
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

/// Replaces a provider secret at the streaming boundary.  The terminal
/// `Completed` response deliberately retains the unmodified value in the
/// regression below, so the test proves the accumulated run and cache use the
/// transformed delta rather than the raw terminal payload.
struct SecretRedactingDelta;

#[async_trait]
impl Middleware<(), ()> for SecretRedactingDelta {
    fn name(&self) -> &str {
        "secret-redacting-delta"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut tinyinference_llm::model::ModelDelta,
    ) -> Result<()> {
        delta.content = delta.content.replace("raw-secret", "[REDACTED]");
        Ok(())
    }
}

/// Suppresses a streamed tool fragment before the accumulator can turn it into
/// an executable call.
struct SuppressToolDelta;

#[async_trait]
impl Middleware<(), ()> for SuppressToolDelta {
    fn name(&self) -> &str {
        "suppress-tool-delta"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut tinyinference_llm::model::ModelDelta,
    ) -> Result<()> {
        delta.tool_call = None;
        Ok(())
    }
}

/// Rewrites a streamed tool call and stops after dispatch so the regression can
/// inspect exactly which canonical call reached the executor.
struct RewriteToolDelta;

#[async_trait]
impl Middleware<(), ()> for RewriteToolDelta {
    fn name(&self) -> &str {
        "rewrite-tool-delta"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut tinyinference_llm::model::ModelDelta,
    ) -> Result<()> {
        if let Some(tool_call) = &mut delta.tool_call {
            tool_call.call_id = "rewritten-call".to_string();
            tool_call.tool_name = Some("safe".to_string());
            tool_call.content = r#"{"source":"middleware"}"#.to_string();
        }
        Ok(())
    }

    async fn after_tool(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _invocation: &ToolInvocationIdentity,
        _result: &mut ToolResult,
    ) -> Result<()> {
        ctx.request_control(crate::context::MiddlewareControl::StopWithFinal(
            "rewritten tool executed".to_string(),
        ));
        Ok(())
    }
}

#[tokio::test]
async fn invoke_streaming_fires_on_model_delta_per_delta_and_accumulates() {
    use crate::testkit::StreamingMock;

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
async fn streamed_repetitive_narration_stops_the_live_model_call() {
    use crate::testkit::StreamingMock;

    let repeated = (0..12)
        .map(|i| {
            format!("Let me inspect source {i} carefully before I answer the user's question. ")
        })
        .collect::<String>();
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::from_text_chunks([repeated])),
    );

    let error = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("stalled-stream"),
            vec![Message::user("Find information about Jev")],
        )
        .await
        .expect_err("repetitive streamed narration must stop the run");
    assert!(matches!(&error, TinyAgentsError::GenerationStalled));
    assert!(!crate::retry::is_retryable(&error));
}

#[tokio::test]
async fn streaming_delta_transform_controls_final_run_and_cached_response() {
    use crate::cache::InMemoryResponseCache;
    use crate::testkit::StreamingMock;

    // The provider exposes the raw secret both incrementally and in the
    // terminal response.  A transform applied to the delta must be the value
    // returned to the caller and retained for a subsequent cache hit.
    let model = Arc::new(StreamingMock::new(vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::text("raw-secret")),
        ModelStreamItem::Completed(ModelResponse::assistant("raw-secret")),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("stream", model.clone());
    harness.with_response_cache(Arc::new(InMemoryResponseCache::new()));
    harness.push_middleware(Arc::new(SecretRedactingDelta));

    let first = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("stream-secret-first"),
            vec![Message::user("same request")],
        )
        .await
        .expect("streaming run succeeds");
    assert_eq!(first.text().as_deref(), Some("[REDACTED]"));
    assert!(
        !first.text().unwrap_or_default().contains("raw-secret"),
        "the returned AgentRun must not restore terminal raw content"
    );

    let second = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("stream-secret-cached"),
            vec![Message::user("same request")],
        )
        .await
        .expect("cached streaming run succeeds");
    assert_eq!(
        model.call_count(),
        1,
        "second response was served from cache"
    );
    assert_eq!(second.text().as_deref(), Some("[REDACTED]"));
    assert!(
        !second.text().unwrap_or_default().contains("raw-secret"),
        "the cached response must not retain the raw terminal secret"
    );
}

/// C-2 regression: a streaming turn whose terminal response carries a signed
/// `Thinking` block ahead of a tool call must keep that exact signature in
/// `run.messages`. Anthropic requires the signed thinking block to precede a
/// `tool_use` block verbatim on replay; synthesizing a fresh, unsigned block
/// from the streamed reasoning text (the old behavior) breaks that replay on
/// the very next model call. No delta middleware is registered here, so the
/// streamed reasoning text is identical to the terminal block's text and the
/// fix's "keep it verbatim" branch is exercised.
#[tokio::test]
async fn streaming_turn_keeps_a_signed_thinking_signature_ahead_of_a_tool_call() {
    use crate::testkit::StreamingMock;

    let tool = Arc::new(FakeTool::new("lookup", "ok"));
    let mut terminal = ModelResponse::assistant("");
    terminal.message.content = vec![tinyinference_llm::message::ContentBlock::Thinking {
        text: "let me think".to_string(),
        signature: Some("sig-123".to_string()),
    }];
    terminal
        .message
        .tool_calls
        .push(ToolCall::new("call-1", "lookup", json!({})));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::new(vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(MessageDelta::reasoning("let me think")),
            ModelStreamItem::ToolCallDelta(tinyinference_llm::tool::ToolDelta {
                call_id: "call-1".to_string(),
                content: "{}".to_string(),
                tool_name: Some("lookup".to_string()),
                ..Default::default()
            }),
            ModelStreamItem::Completed(terminal),
        ])),
    );
    harness.register_tool(tool.clone());

    // Cap the run at one model call: the mock always replays the same
    // scripted tool call, so a second turn would just repeat it forever.
    // Only the first turn's assistant message (the one under test) is
    // needed.
    let ctx = RunContext::new(
        RunConfig::new("thinking-signature").with_max_model_calls(1),
        (),
    );
    let outcome = harness
        .invoke_streaming_in_context_collecting_partial(&(), ctx, vec![Message::user("go")])
        .await;

    let thinking_blocks: Vec<_> = outcome
        .run
        .messages
        .iter()
        .filter_map(|message| match message {
            tinyinference_llm::message::Message::Assistant(assistant) => {
                Some(assistant.content.iter())
            }
            _ => None,
        })
        .flatten()
        .filter(|block| {
            matches!(
                block,
                tinyinference_llm::message::ContentBlock::Thinking { .. }
            )
        })
        .collect();
    assert_eq!(
        thinking_blocks,
        vec![&tinyinference_llm::message::ContentBlock::Thinking {
            text: "let me think".to_string(),
            signature: Some("sig-123".to_string()),
        }],
        "the terminal Thinking block's signature must survive into run.messages verbatim"
    );
}

#[tokio::test]
async fn streaming_middleware_can_suppress_a_standalone_tool_delta() {
    use crate::testkit::StreamingMock;

    let tool = Arc::new(FakeTool::new("blocked", "must not run"));
    let mut terminal = ModelResponse::assistant("");
    terminal
        .message
        .tool_calls
        .push(ToolCall::new("blocked-call", "blocked", json!({})));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::new(vec![
            ModelStreamItem::Started,
            ModelStreamItem::ToolCallDelta(tinyinference_llm::tool::ToolDelta {
                call_id: "blocked-call".to_string(),
                content: "{}".to_string(),
                tool_name: Some("blocked".to_string()),
                ..Default::default()
            }),
            ModelStreamItem::Completed(terminal),
        ])),
    );
    harness.register_tool(tool.clone());
    harness.push_middleware(Arc::new(SuppressToolDelta));

    let run = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("suppressed-tool-delta"),
            vec![Message::user("go")],
        )
        .await
        .expect("suppressed tool call leaves a valid empty completion");

    assert!(run.text().unwrap_or_default().is_empty());
    assert_eq!(
        *tool.calls.lock().unwrap(),
        0,
        "suppressed call must not run"
    );
}

#[tokio::test]
async fn streaming_tool_delta_transform_controls_terminal_dispatch() {
    use crate::testkit::StreamingMock;

    let blocked = Arc::new(FakeTool::new("blocked", "blocked"));
    let safe = Arc::new(FakeTool::new("safe", "safe"));
    let mut terminal = ModelResponse::assistant("");
    terminal
        .message
        .tool_calls
        .push(ToolCall::new("raw-call", "blocked", json!({"raw": true})));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::new(vec![
            ModelStreamItem::Started,
            ModelStreamItem::ToolCallDelta(tinyinference_llm::tool::ToolDelta {
                call_id: "raw-call".to_string(),
                content: r#"{"raw":true}"#.to_string(),
                tool_name: Some("blocked".to_string()),
                ..Default::default()
            }),
            ModelStreamItem::Completed(terminal),
        ])),
    );
    harness.register_tool(blocked.clone());
    harness.register_tool(safe.clone());
    harness.push_middleware(Arc::new(RewriteToolDelta));

    let run = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("rewritten-tool-delta"),
            vec![Message::user("go")],
        )
        .await
        .expect("rewritten call remains executable");

    assert_eq!(run.text().as_deref(), Some("rewritten tool executed"));
    assert_eq!(*safe.calls.lock().unwrap(), 1);
    assert_eq!(*blocked.calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn invoke_streaming_forwards_reasoning_deltas_to_middleware_and_events() {
    use crate::testkit::{EventRecorder, StreamingMock};

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
            crate::events::AgentEvent::ModelDelta { delta, .. } => Some(delta.reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(event_reasoning, "think ");
}

#[tokio::test]
async fn invoke_streaming_emits_model_delta_events() {
    use crate::testkit::{EventRecorder, StreamingMock};

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
            crate::events::AgentEvent::ModelDelta { run_id, .. } => Some(run_id),
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

use crate::cancel::CancellationToken;

/// A model that records how many times it was invoked and always asks for a
/// tool, so the loop only stops via a limit or cancellation.
struct CountingToolModel {
    name: &'static str,
    invocations: Arc<Mutex<usize>>,
}

#[async_trait]
impl ChatModel<()> for CountingToolModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
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
impl Tool for CancelOnCallTool {
    fn name(&self) -> &str {
        "cancel_me"
    }
    fn description(&self) -> &str {
        "cancels the run"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.token.cancel();
        Ok(ToolResult::success("cancelled"))
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
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
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

    use crate::testkit::SlowModel;

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

    use crate::testkit::SlowModel;

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

#[tokio::test]
async fn per_model_call_ceiling_times_out_a_slow_call_with_run_time_left() {
    use std::time::Duration;

    use crate::testkit::SlowModel;

    // The run has plenty of wall clock left (60s), but the per-model-call
    // ceiling (20ms) is tighter than the model's 200ms sleep, so the ceiling
    // interrupts the call — and the error must name the ceiling, not the run's
    // remaining budget, so triage can tell a wedged call from an exhausted run.
    // A per-call ceiling is a `CallTimeout`, not a `Timeout`: it is retryable
    // and must not skip the fallback chain the way a run-deadline timeout
    // does (I-1). One retry attempt is enough to prove that here.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "slow",
        Arc::new(SlowModel::new(Duration::from_millis(200), "too late")),
    );
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_call_ms(Some(20)),
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });

    let config = RunConfig::new("per-call-cap-run").with_timeout_ms(60_000);
    let err = harness
        .invoke(&(), (), config, vec![Message::user("hi")])
        .await
        .expect_err("a call slower than the per-call ceiling must time out");

    match &err {
        TinyAgentsError::CallTimeout(msg) => {
            assert!(msg.contains("per-model-call ceiling"), "{msg}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
}

#[tokio::test]
async fn per_model_call_ceiling_consults_the_fallback_chain_instead_of_aborting() {
    use std::time::Duration;

    use crate::testkit::{ScriptedModel, SlowModel};

    // Same setup as `per_model_call_ceiling_times_out_a_slow_call_with_run_time_left`,
    // but with a fallback model registered. Before the fix, the per-call
    // ceiling produced a plain `Timeout`, which the fallback gate in
    // `invoke_model_resolving` treats as terminal ("the run itself is out of
    // wall-clock budget") and returns immediately — the fallback model is
    // never even consulted, let alone called. With the fix, a `CallTimeout`
    // falls through to the fallback walk, so the run succeeds on the
    // fallback model instead of failing.
    let slow = Arc::new(SlowModel::new(Duration::from_millis(200), "too late"));
    let fallback = Arc::new(ScriptedModel::replies(vec!["fallback answer"]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("slow", slow.clone());
    harness.register_model("fallback", fallback.clone());
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_call_ms(Some(20)),
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy {
            models: vec!["slow".to_string(), "fallback".to_string()],
        }),
        ..RunPolicy::default()
    });

    let config = RunConfig::new("per-call-cap-fallback").with_timeout_ms(60_000);
    let run = harness
        .invoke(&(), (), config, vec![Message::user("hi")])
        .await
        .expect("a retryable CallTimeout must fall back instead of aborting the run");

    assert_eq!(run.text().as_deref(), Some("fallback answer"));
    assert_eq!(
        fallback.requests().len(),
        1,
        "the fallback chain must actually have been consulted and called"
    );
}

#[tokio::test]
async fn per_model_call_ceiling_bounds_calls_without_any_run_deadline() {
    use std::time::Duration;

    use crate::testkit::SlowModel;

    // With no run timeout and no policy wall clock, a model call used to be
    // awaited unbounded. The per-call ceiling alone must bound it.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "slow",
        Arc::new(SlowModel::new(Duration::from_millis(200), "too late")),
    );
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_call_ms(Some(20)),
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("the ceiling alone must bound an otherwise-unbounded call");

    match &err {
        TinyAgentsError::CallTimeout(msg) => {
            assert!(msg.contains("per-model-call ceiling"), "{msg}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
}

#[tokio::test]
async fn run_remainder_bounds_a_model_call_when_tighter_than_the_ceiling() {
    use std::time::Duration;

    use crate::testkit::SlowModel;

    // A generous ceiling (10s) never extends a call past the run's own
    // deadline (20ms): the tighter source wins, and the error blames the
    // remaining wall-clock budget.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "slow",
        Arc::new(SlowModel::new(Duration::from_millis(200), "too late")),
    );
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_call_ms(Some(10_000)),
        ..RunPolicy::default()
    });

    let config = RunConfig::new("remainder-tighter-run").with_timeout_ms(20);
    let err = harness
        .invoke(&(), (), config, vec![Message::user("hi")])
        .await
        .expect_err("the run deadline must still bound the call");

    match &err {
        TinyAgentsError::Timeout(msg) => {
            assert!(msg.contains("remaining wall-clock budget"), "{msg}");
        }
        other => panic!("expected Timeout, got {other:?}"),
    }
}

#[tokio::test]
async fn per_model_call_ceiling_does_not_bound_tool_calls() {
    use std::time::Duration;

    // A 20ms per-model-call ceiling with a 100ms tool: tool calls keep the
    // remaining-only budget (a sub-agent delegation is a tool call wrapping an
    // entire child run), so the run completes despite the tool outliving the
    // model-call ceiling many times over.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "slow", json!({})),
            text_response("done", 0, 0),
        ])),
    );
    harness.register_tool(Arc::new(SlowTool {
        delay: Duration::from_millis(100),
    }));
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_call_ms(Some(20)),
        ..RunPolicy::default()
    });

    let config = RunConfig::new("tool-uncapped-run").with_timeout_ms(60_000);
    let run = harness
        .invoke(&(), (), config, vec![Message::user("go")])
        .await
        .expect("the tool call must not inherit the per-model-call ceiling");

    assert_eq!(run.text(), Some("done".to_string()));
}

#[tokio::test]
async fn inherited_tool_timeout_is_enforced_without_a_run_deadline() {
    use std::time::Duration;

    let settings = ToolTimeoutSettings::new(20, 1, 1_000, 0);
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_tool_timeout_settings(settings.clone());
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "policy_slow", json!({})),
            text_response("recovered", 2, 1),
        ])),
    );
    harness.register_tool(Arc::new(PolicySlowTool {
        name: "policy_slow",
        delay: Duration::from_millis(200),
        timeout: ToolTimeout::Inherit,
    }));

    let run = harness
        .invoke(
            &(),
            (),
            RunConfig::new("inherited-tool-timeout-run"),
            vec![Message::user("go")],
        )
        .await
        .expect("a per-tool timeout should be recoverable by the model");

    assert_eq!(run.text(), Some("recovered".to_string()));
    assert!(run.messages[2].text().contains("timed out after 20 ms"));

    settings.set_inherited_timeout_ms(5);
    assert_eq!(settings.inherited_timeout(), Some(Duration::from_millis(5)));
}

#[tokio::test]
async fn tool_timeout_unwinds_wrap_middleware() {
    use std::time::Duration;

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_tool_timeout_settings(ToolTimeoutSettings::new(10, 1, 1_000, 0));
    harness.push_tool_middleware(Arc::new(StampToolWrap));
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "policy_slow", json!({})),
            text_response("recovered", 2, 1),
        ])),
    );
    harness.register_tool(Arc::new(PolicySlowTool {
        name: "policy_slow",
        delay: Duration::from_millis(200),
        timeout: ToolTimeout::Inherit,
    }));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("middleware should unwind after the inner tool times out");

    assert_eq!(run.text(), Some("recovered".to_string()));
    assert!(run.messages[2].text().starts_with("[wrapped]"));
    assert!(run.messages[2].text().contains("timed out after 10 ms"));
}

#[tokio::test]
async fn tool_timeout_resolves_after_wrap_middleware_mutation() {
    use std::time::Duration;

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_tool_timeout_settings(ToolTimeoutSettings::new(10, 1, 1_000, 0));
    harness.push_tool_middleware(Arc::new(UnboundToolWrap));
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "policy_slow", json!({})),
            text_response("done", 2, 1),
        ])),
    );
    harness.register_tool(Arc::new(PolicySlowTool {
        name: "policy_slow",
        delay: Duration::from_millis(30),
        timeout: ToolTimeout::Inherit,
    }));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("the middleware-mutated unbounded policy should reach the tool");

    assert_eq!(run.text(), Some("done".to_string()));
    assert_eq!(run.messages[2].text(), "too late");
}

#[tokio::test]
async fn parallel_tool_timeouts_return_ordered_recoverable_errors() {
    use std::time::Duration;

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_tool_timeout_settings(ToolTimeoutSettings::new(10, 1, 1_000, 0));
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            multi_tool_call_response(vec![("call-a", "slow_a"), ("call-b", "slow_b")]),
            text_response("recovered", 2, 1),
        ])),
    );
    for name in ["slow_a", "slow_b"] {
        harness.register_tool(Arc::new(PolicySlowTool {
            name,
            delay: Duration::from_millis(200),
            timeout: ToolTimeout::Inherit,
        }));
    }

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("parallel tool timeouts should be recoverable");

    assert_eq!(run.text(), Some("recovered".to_string()));
    assert!(run.messages[2].text().contains("slow_a"));
    assert!(run.messages[3].text().contains("slow_b"));
}

// ── Response caching ──────────────────────────────────────────────────────────

#[tokio::test]
async fn response_cache_serves_repeated_request_without_calling_model() {
    use crate::cache::InMemoryResponseCache;
    use crate::testkit::EventRecorder;

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
    use crate::cache::InMemoryResponseCache;

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
    use crate::cache::InMemoryResponseCache;
    use tinyinference_llm::cache::CachePolicy;

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
                ..CachePolicy::default()
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

/// Middleware whose `before_model` hook always fails, counting how many times
/// the failure is delivered back to it through `on_error`.
struct FailingHookMiddleware {
    on_error_calls: Arc<Mutex<usize>>,
}

#[async_trait]
impl Middleware<(), ()> for FailingHookMiddleware {
    fn name(&self) -> &str {
        "failing_hook"
    }
    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        Err(TinyAgentsError::Model("hook failed".to_string()))
    }
    async fn on_error(&self, _ctx: &mut RunContext<()>, _error: &TinyAgentsError) -> Result<()> {
        *self.on_error_calls.lock().unwrap() += 1;
        Ok(())
    }
}

#[tokio::test]
async fn hook_failure_delivers_on_error_exactly_once() {
    // Regression test: the stack fans `on_error` out itself before propagating a
    // failed lifecycle hook, and the driver used to run the same hook again for
    // the propagated error — so a guardrail counting failures, alerting, or
    // compensating did it twice for one failure.
    let calls = Arc::new(Mutex::new(0usize));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("hi")));
    harness.push_middleware(Arc::new(FailingHookMiddleware {
        on_error_calls: calls.clone(),
    }));

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("the failing hook aborts the run");
    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
    assert_eq!(*calls.lock().unwrap(), 1, "one failure, one on_error");
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
        ctx.request_control(crate::context::MiddlewareControl::StopWithFinal(
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

    // M-1 regression: the assistant row still carries the `tool_calls` the
    // model requested, but the loop must synthesize a tool result for each
    // one so `run.messages` stays replayable (a provider rejects a transcript
    // whose assistant `tool_calls` have no matching tool message).
    // user, assistant(1 tool call), tool(synthetic).
    assert_eq!(run.messages.len(), 3);
    let Message::Assistant(assistant) = &run.messages[1] else {
        panic!(
            "expected assistant message at index 1, got {:?}",
            run.messages[1]
        );
    };
    assert_eq!(assistant.tool_calls.len(), 1);
    let Message::Tool(tool_message) = &run.messages[2] else {
        panic!(
            "expected synthetic tool message at index 2, got {:?}",
            run.messages[2]
        );
    };
    assert_eq!(tool_message.tool_call_id, assistant.tool_calls[0].id);
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
        ctx.request_control(crate::context::MiddlewareControl::Interrupt {
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
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.seen_tools.lock().unwrap().push(request.tools.clone());
        let next = self.responses.lock().unwrap().pop_front();
        next.ok_or_else(|| tinyinference_llm::Error::Validation("no scripted response left".into()))
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
    use crate::events::AgentEvent;
    use crate::testkit::EventRecorder;

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
    use crate::testkit::EventRecorder;

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
impl Tool for ConcurrencyProbeTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "concurrency probe"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn is_concurrency_safe(&self, _arguments: &serde_json::Value) -> bool {
        true
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        use std::sync::atomic::Ordering;
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_seen.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::success(self.reply))
    }
}

/// A concurrency-safe tool that fails fast (a real dispatch error, not a
/// recoverable `ToolResult::error`), used to exercise the concurrent path's
/// first-fatal-error handling.
struct FailingConcurrentTool {
    name: &'static str,
}

#[async_trait]
impl Tool for FailingConcurrentTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "fails fast"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn is_concurrency_safe(&self, _arguments: &serde_json::Value) -> bool {
        true
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Err(anyhow::anyhow!("boom"))
    }
}

/// C-3 regression: on the first fatal error in the concurrent tool path,
/// every already-started sibling call must still get exactly one terminal
/// event (`ToolFailed`), and `active_tool_calls` must end up empty — not just
/// the call that actually failed. Before the fix, siblings whose futures had
/// already resolved (via `join_all`) but were never reached by the fold after
/// the first `Err` kept their `ToolStarted` unanswered and stayed listed in
/// `active_tool_calls` even though the run had already failed.
#[tokio::test]
async fn concurrent_tool_failure_fails_every_started_sibling_before_returning() {
    use crate::testkit::EventRecorder;

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![multi_tool_call_response(
            // "boom" fails fast and comes first in call order, so the fold
            // reaches its fatal error while "alpha" (slower, but already
            // resolved by the time `join_all` returns) is still an
            // unprocessed sibling — exactly the scenario the fix covers.
            vec![("call-a", "boom"), ("call-b", "alpha")],
        )])),
    );
    harness.register_tool(Arc::new(ConcurrencyProbeTool {
        name: "alpha",
        reply: "alpha-out",
        delay: std::time::Duration::from_millis(80),
        active: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_seen: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    }));
    harness.register_tool(Arc::new(FailingConcurrentTool { name: "boom" }));

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("concurrent-fatal"), ()).with_events(recorder.sink());
    let outcome = harness
        .invoke_in_context_collecting_partial(&(), ctx, vec![Message::user("go")])
        .await;

    assert!(
        outcome.error.is_some(),
        "a fatal sibling error must fail the turn"
    );
    assert!(
        outcome.status.active_tool_calls.is_empty(),
        "every started call must have a terminal event before the run reports failure, \
         got active_tool_calls = {:?}",
        outcome.status.active_tool_calls
    );

    let started: Vec<_> = recorder
        .events()
        .iter()
        .filter_map(|record| match record {
            AgentEvent::ToolStarted { call_id, .. } => Some(call_id.as_str().to_string()),
            _ => None,
        })
        .collect();
    let terminal: Vec<_> = recorder
        .events()
        .iter()
        .filter_map(|record| match record {
            AgentEvent::ToolFailed { call_id, .. } => Some(call_id.as_str().to_string()),
            AgentEvent::ToolCompleted { call_id, .. } => Some(call_id.as_str().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(started.len(), 2, "both siblings must have started");
    assert_eq!(
        terminal.len(),
        2,
        "every started call must be answered by exactly one terminal event, got {terminal:?}"
    );
    for call_id in &started {
        assert!(
            terminal.contains(call_id),
            "call `{call_id}` started but has no terminal event"
        );
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
            origin: None,
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
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
async fn max_tool_concurrency_bounds_how_many_tools_run_at_once() {
    // I-8 regression test: with 4 concurrency-safe tools requested in one
    // turn and `RunLimits::max_tool_concurrency` set to 2, at most 2 may be
    // in flight at once, even though all 4 are eligible for the concurrent
    // path. `max_seen` is an atomic high-water mark, so any window where 3+
    // ran together would be caught regardless of scheduling order.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            multi_tool_call_response(vec![
                ("call-a", "alpha"),
                ("call-b", "beta"),
                ("call-c", "gamma"),
                ("call-d", "delta"),
            ]),
            text_response("done", 4, 2),
        ])),
    );
    let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for name in ["alpha", "beta", "gamma", "delta"] {
        harness.register_tool(Arc::new(ConcurrencyProbeTool {
            name,
            reply: "out",
            delay: std::time::Duration::from_millis(60),
            active: active.clone(),
            max_seen: max_seen.clone(),
        }));
    }
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_tool_concurrency(Some(2)),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.tool_calls, 4);
    assert_eq!(
        max_seen.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "no more than max_tool_concurrency (2) tools should ever be in flight at once"
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
            Some(AgentStreamItem::Failed { .. })
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
        AgentStreamItem::Failed { error: message, .. } => {
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
    use crate::events::RecordingListener;

    // A tool that fails: its ToolCompleted event must carry the failure message,
    // a real duration, and the output size — from the event itself, not a
    // side-channel — so journal-backed exporters can render the outcome.
    struct FailTool;
    #[async_trait]
    impl Tool for FailTool {
        fn name(&self) -> &str {
            "boom"
        }
        fn description(&self) -> &str {
            "always fails"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }
        async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::error("kaboom"))
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
    assert_eq!(output_bytes, Some(6), "\"kaboom\".len() == 6");
}

#[tokio::test]
async fn tool_started_event_carries_input_when_capture_enabled() {
    use crate::events::RecordingListener;
    use crate::runtime::PayloadCapture;

    // Hosts render a tool call's arguments as soon as it starts, not only on
    // completion, so `ToolStarted` must carry the same captured input the
    // policy already puts on `ToolCompleted`.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "echo", json!({ "q": "weather" })),
            text_response("done", 1, 1),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::new("echo", "ok")));
    harness.with_policy(RunPolicy {
        capture: PayloadCapture::all(),
        ..RunPolicy::default()
    });

    let recorder = Arc::new(RecordingListener::new());
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-ts"), ());
    ctx.events.subscribe(recorder.clone());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let input = recorder
        .events()
        .into_iter()
        .find_map(|record| match record.event {
            AgentEvent::ToolStarted {
                tool_name, input, ..
            } if tool_name == "echo" => Some(input),
            _ => None,
        })
        .expect("a ToolStarted event for `echo`");

    assert_eq!(input, Some(json!({ "q": "weather" })));
}

#[tokio::test]
async fn tool_started_event_has_no_input_when_capture_disabled() {
    use crate::events::RecordingListener;

    // Default policy is payload-free: `ToolStarted.input` stays `None` so no
    // tool argument is captured unless the host opts in.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "echo", json!({ "q": "weather" })),
            text_response("done", 1, 1),
        ])),
    );
    harness.register_tool(Arc::new(FakeTool::new("echo", "ok")));

    let recorder = Arc::new(RecordingListener::new());
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-ts-off"), ());
    ctx.events.subscribe(recorder.clone());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let input = recorder
        .events()
        .into_iter()
        .find_map(|record| match record.event {
            AgentEvent::ToolStarted {
                tool_name, input, ..
            } if tool_name == "echo" => Some(input),
            _ => None,
        })
        .expect("a ToolStarted event for `echo`");

    assert_eq!(input, None);
}

// ── `ModelResponse::continue_turn` ───────────────────────────────────────────

/// A tool-less response that keeps the floor by carrying `nudge`.
fn continuing(text: &str, nudge: &str) -> ModelResponse {
    let mut response = ModelResponse::assistant(text.to_string());
    response.continue_turn = Some(nudge.to_string());
    response
}

/// A tool-less response normally ends the turn. `continue_turn` overrides that:
/// the loop appends the carried nudge as the next user turn and asks for another
/// reply — which is what lets a model speak before it acts.
#[tokio::test]
async fn continue_turn_keeps_the_floor_and_appends_the_nudge() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            continuing("let me look that up", "(next)"),
            ModelResponse::assistant("found it".to_string()),
        ])),
    );

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(
        run.model_calls, 2,
        "the continuing reply must not end the turn"
    );
    assert_eq!(run.text(), Some("found it".to_string()));
    // user, assistant(continuing), user(nudge), assistant(final)
    assert_eq!(run.messages.len(), 4);
    assert_eq!(run.messages[2].text(), "(next)");
}

/// The default is `None`, which must leave the historical behaviour byte-for-byte
/// intact: the first tool-less reply ends the turn.
#[tokio::test]
async fn no_continue_turn_ends_on_the_first_tool_less_reply() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            ModelResponse::assistant("done".to_string()),
            ModelResponse::assistant("never reached".to_string()),
        ])),
    );

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text(), Some("done".to_string()));
}

/// Continuing needs no cap of its own: each continue costs one model call, so a
/// model that never stops is already bounded by `max_model_calls`.
#[tokio::test]
async fn an_endless_continue_is_bounded_by_max_model_calls() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            continuing("still going", "(next)"),
            continuing("still going", "(next)"),
            continuing("still going", "(next)"),
            continuing("still going", "(next)"),
            continuing("still going", "(next)"),
        ])),
    );

    let config = RunConfig::new("continue-cap").with_max_model_calls(3);
    let err = harness
        .invoke_in_context(&(), RunContext::new(config, ()), vec![Message::user("hi")])
        .await
        .expect_err("an endless continue must hit the model-call cap");

    assert!(
        err.to_string().contains("max model calls"),
        "expected the model-call cap to stop it, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Schema-echo argument recovery
//
// Small local models sometimes fill the tool's own JSON Schema in place and
// send the whole envelope as the arguments. These pin the conservative
// unwrap in `normalize_tool_arguments`, in both directions.
// ---------------------------------------------------------------------------

/// A tool whose arguments genuinely include a field named `properties`, so the
/// echo-unwrap must leave its calls alone.
struct SchemaShapedTool {
    calls: Arc<Mutex<usize>>,
}

#[async_trait]
impl Tool for SchemaShapedTool {
    fn name(&self) -> &str {
        "schema_shaped"
    }
    fn description(&self) -> &str {
        "takes a literal `properties` argument"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["properties"],
            "properties": {
                "properties": { "type": "object" }
            }
        })
    }
    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.calls.lock().unwrap() += 1;
        Ok(ToolResult::success(
            serde_json::to_string(&arguments).unwrap_or_default(),
        ))
    }
}

/// A tool that records the arguments it was actually invoked with, so a test
/// can assert what normalization produced rather than only that it ran.
struct ArgumentRecordingTool {
    seen: Arc<Mutex<Vec<serde_json::Value>>>,
}

#[async_trait]
impl Tool for ArgumentRecordingTool {
    fn name(&self) -> &str {
        "strict_lookup"
    }
    fn description(&self) -> &str {
        "strict lookup"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        StrictLookupTool {
            calls: Arc::new(Mutex::new(0)),
        }
        .parameters_schema()
    }
    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.seen.lock().unwrap().push(arguments);
        Ok(ToolResult::success("strict-output"))
    }
}

#[tokio::test]
async fn echoed_schema_arguments_are_unwrapped_before_validation() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response(
                "call-1",
                "strict_lookup",
                // The exact shape `llama3.2:3b` emits: the declaration filled
                // in place, rather than the arguments object.
                json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": { "query": "rust" }
                }),
            ),
            text_response("done", 1, 1),
        ])),
    );
    let seen = Arc::new(Mutex::new(Vec::new()));
    harness.register_tool(Arc::new(ArgumentRecordingTool {
        seen: Arc::clone(&seen),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("an echoed schema should be unwrapped, not rejected");

    assert_eq!(run.final_response.unwrap().text(), "done");
    // The tool ran once, with the unwrapped arguments — not the envelope.
    assert_eq!(
        *seen.lock().unwrap(),
        vec![json!({ "query": "rust" })],
        "the envelope should have been unwrapped to the inner arguments"
    );
}

/// The other envelope shapes captured from `llama3.2:3b`: the arguments nested
/// under `arguments` alongside a schema echo, and under a bare `param` key.
#[tokio::test]
async fn wrapped_arguments_are_unwrapped_from_every_known_envelope_key() {
    for envelope in [
        json!({
            "properties": { "query": { "type": "string" } },
            "required": ["query"],
            "arguments": { "query": "rust" }
        }),
        json!({ "param": { "query": "rust" } }),
        json!({ "params": { "query": "rust" } }),
        json!({ "args": { "query": "rust" } }),
        json!({ "parameters": { "query": "rust" } }),
        json!({ "input": { "query": "rust" } }),
    ] {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "strict_lookup", envelope.clone()),
                text_response("done", 1, 1),
            ])),
        );
        let seen = Arc::new(Mutex::new(Vec::new()));
        harness.register_tool(Arc::new(ArgumentRecordingTool {
            seen: Arc::clone(&seen),
        }));
        harness.with_policy(RunPolicy {
            invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
            ..RunPolicy::default()
        });

        harness
            .invoke_default(&(), vec![Message::user("lookup")])
            .await
            .unwrap_or_else(|e| panic!("{envelope} should be unwrapped: {e}"));

        assert_eq!(
            *seen.lock().unwrap(),
            vec![json!({ "query": "rust" })],
            "{envelope} should have been unwrapped to the inner arguments"
        );
    }
}

#[tokio::test]
async fn echo_unwrap_leaves_a_tool_that_really_takes_properties_alone() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response(
                "call-1",
                "schema_shaped",
                json!({ "properties": { "query": "rust" } }),
            ),
            text_response("done", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(SchemaShapedTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("run")])
        .await
        .expect("a valid call must not be rewritten");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn echo_unwrap_is_skipped_when_the_inner_value_is_still_invalid() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response(
                "call-1",
                "strict_lookup",
                // `query` is an integer, so unwrapping would not rescue it.
                // The original envelope must survive so the model sees a
                // precise validation error rather than a rewritten one.
                json!({
                    "type": "object",
                    "properties": { "query": 7 }
                }),
            ),
            text_response("done", 1, 1),
        ])),
    );
    let calls = Arc::new(Mutex::new(0));
    harness.register_tool(Arc::new(StrictLookupTool {
        calls: Arc::clone(&calls),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("lookup")])
        .await
        .expect("the loop recovers by handing the validation error back");

    assert_eq!(run.final_response.unwrap().text(), "done");
    assert_eq!(
        *calls.lock().unwrap(),
        0,
        "the tool must not run with arguments that never validated"
    );
}

// ---------------------------------------------------------------------------
// Follow-up F1: harness-side `ModelProfile` wiring
// ---------------------------------------------------------------------------

/// A tool whose declared schema carries `$defs`, so a `SchemaTransform` that
/// strips them (or resolves refs) is observable on the wire.
struct DefsTool;

#[async_trait]
impl Tool for DefsTool {
    fn name(&self) -> &str {
        "lookup"
    }
    fn description(&self) -> &str {
        "look something up"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "$defs": {"Id": {"type": "string"}},
            "properties": {"id": {"$ref": "#/$defs/Id"}},
        })
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

#[tokio::test]
async fn resolved_profile_schema_transform_is_applied_to_tool_schemas() {
    use crate::testkit::ScriptedModel;

    let model = Arc::new(
        ScriptedModel::replies(vec!["done"]).with_profile(ModelProfile {
            tool_calling: true,
            schema_transform: Some(SchemaTransform::StripDefs),
            ..ModelProfile::default()
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.register_tool(Arc::new(DefsTool));

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = model.requests();
    let request = requests.first().expect("one model call");
    let tool = request
        .tools
        .iter()
        .find(|t| t.name == "lookup")
        .expect("lookup tool advertised");
    assert!(
        tool.parameters.get("$defs").is_none(),
        "the resolved profile's StripDefs transform must strip `$defs` before \
         the request is sent: {:?}",
        tool.parameters
    );
}

#[tokio::test]
async fn resolved_profile_schema_transform_is_applied_to_the_structured_output_schema() {
    use crate::testkit::ScriptedModel;

    let model = Arc::new(
        ScriptedModel::replies(vec![r#"{"value":"hi"}"#]).with_profile(ModelProfile {
            native_structured_output: true,
            json_schema: true,
            schema_transform: Some(SchemaTransform::StripDefs),
            ..ModelProfile::default()
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::auto(
            "answer",
            json!({
                "type": "object",
                "$defs": {"Id": {"type": "string"}},
                "properties": {"value": {"$ref": "#/$defs/Id"}},
            }),
        )),
        ..RunPolicy::default()
    });

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = model.requests();
    let request = requests.first().expect("one model call");
    let ResponseFormat::JsonSchema { schema, .. } = request
        .response_format
        .as_ref()
        .expect("provider-native structured output was requested")
    else {
        panic!("expected JsonSchema, got {:?}", request.response_format);
    };
    assert!(
        schema.get("$defs").is_none(),
        "the structured-output schema must be transformed too: {schema:?}"
    );
}

#[tokio::test]
async fn default_structured_mode_prompted_injects_schema_into_the_system_segment() {
    use crate::testkit::ScriptedModel;

    let model = Arc::new(
        ScriptedModel::replies(vec![r#"{"value":"hi"}"#]).with_profile(ModelProfile {
            default_structured_mode: Some(StructuredMode::Prompted),
            prompted_output_template: Some("Reply with JSON matching:".to_string()),
            ..ModelProfile::default()
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::auto(
            "answer",
            json!({"type": "object", "properties": {"value": {"type": "string"}}}),
        )),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = model.requests();
    let request = requests.first().expect("one model call");
    assert_eq!(request.response_format, Some(ResponseFormat::Text));
    let system_text: String = request
        .messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        system_text.contains("Reply with JSON matching:"),
        "the profile's prompted template must be injected: {system_text}"
    );
    assert!(
        system_text.contains("JSON Schema for `answer`"),
        "the schema must be described in the system segment: {system_text}"
    );

    let structured = run.structured.expect("structured output present");
    assert_eq!(structured["value"], "hi");
}

/// A middleware that pins a named reasoning effort onto every model request,
/// standing in for a caller that only knows the generic level name (not this
/// specific model's tuned budget).
struct RequestReasoningEffort(ReasoningEffort);

#[async_trait]
impl Middleware<()> for RequestReasoningEffort {
    fn name(&self) -> &str {
        "request-reasoning-effort"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.reasoning = Some(ReasoningConfig::effort(self.0));
        Ok(())
    }
}

#[tokio::test]
async fn thinking_level_map_resolves_a_named_reasoning_effort() {
    use crate::testkit::ScriptedModel;

    let mut thinking_level_map = std::collections::BTreeMap::new();
    thinking_level_map.insert(
        "high".to_string(),
        ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            budget_tokens: Some(32_000),
            summary: None,
        },
    );
    let model = Arc::new(
        ScriptedModel::replies(vec!["done"]).with_profile(ModelProfile {
            thinking_level_map,
            ..ModelProfile::default()
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.push_middleware(Arc::new(RequestReasoningEffort(ReasoningEffort::High)));

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = model.requests();
    let request = requests.first().expect("one model call");
    assert_eq!(
        request.reasoning.as_ref().and_then(|r| r.budget_tokens),
        Some(32_000),
        "the profile's tuned budget for the `high` level must replace the bare \
         effort the caller asked for: {:?}",
        request.reasoning
    );
}

#[tokio::test]
async fn thinking_level_map_does_not_override_an_explicit_budget() {
    use crate::testkit::ScriptedModel;

    let mut thinking_level_map = std::collections::BTreeMap::new();
    thinking_level_map.insert(
        "high".to_string(),
        ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            budget_tokens: Some(32_000),
            summary: None,
        },
    );
    let model = Arc::new(
        ScriptedModel::replies(vec!["done"]).with_profile(ModelProfile {
            thinking_level_map,
            ..ModelProfile::default()
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());

    struct ExplicitBudget;
    #[async_trait]
    impl Middleware<()> for ExplicitBudget {
        fn name(&self) -> &str {
            "explicit-budget"
        }
        async fn before_model(
            &self,
            _ctx: &mut RunContext<()>,
            _state: &(),
            request: &mut ModelRequest,
        ) -> Result<()> {
            request.reasoning = Some(ReasoningConfig {
                effort: Some(ReasoningEffort::High),
                budget_tokens: Some(1_234),
                summary: None,
            });
            Ok(())
        }
    }
    harness.push_middleware(Arc::new(ExplicitBudget));

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = model.requests();
    let request = requests.first().expect("one model call");
    assert_eq!(
        request.reasoning.as_ref().and_then(|r| r.budget_tokens),
        Some(1_234),
        "an explicit caller budget must win over the profile's mapped default"
    );
}

#[tokio::test]
async fn thinking_tags_are_split_out_of_a_unary_response_into_a_thinking_block() {
    use crate::testkit::ScriptedModel;

    let model = Arc::new(
        ScriptedModel::new(vec![ModelResponse::assistant(
            "<think>reasoning about it</think>the final answer",
        )])
        .with_profile(ModelProfile {
            thinking_tags: Some(("<think>".to_string(), "</think>".to_string())),
            ..ModelProfile::default()
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let response = run.final_response.expect("final response");
    let thinking: Vec<&str> = response
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(thinking, vec!["reasoning about it"]);
    assert_eq!(response.text(), "the final answer");
}

#[tokio::test]
async fn a_model_without_thinking_tags_configured_leaves_text_untouched() {
    use crate::testkit::ScriptedModel;

    let model = Arc::new(ScriptedModel::new(vec![ModelResponse::assistant(
        "<think>not a reasoning tag here</think>plain text",
    )]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let response = run.final_response.expect("final response");
    assert!(
        response
            .message
            .content
            .iter()
            .all(|block| !matches!(block, ContentBlock::Thinking { .. })),
        "no profile means no tag pair to split on"
    );
}

#[tokio::test]
async fn streaming_ignores_leading_whitespace_on_the_first_text_delta_only() {
    use crate::testkit::StreamingMock;

    let model = Arc::new(
        StreamingMock::from_text_chunks(["   Hello", ", world"]).with_profile(ModelProfile {
            ignore_streamed_leading_whitespace: true,
            ..ModelProfile::default()
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("stream", model);

    let run = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("strip-leading-ws"),
            vec![Message::user("hi")],
        )
        .await
        .expect("streaming run succeeds");

    assert_eq!(run.text(), Some("Hello, world".to_string()));
}

#[tokio::test]
async fn streaming_without_the_profile_flag_keeps_leading_whitespace() {
    use crate::testkit::StreamingMock;

    let model = Arc::new(StreamingMock::from_text_chunks(["   Hello", ", world"]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("stream", model);

    let run = harness
        .invoke_streaming(
            &(),
            (),
            RunConfig::new("keep-leading-ws"),
            vec![Message::user("hi")],
        )
        .await
        .expect("streaming run succeeds");

    assert_eq!(run.text(), Some("   Hello, world".to_string()));
}

// ---------------------------------------------------------------------------
// Tool-effect ledger (B5)
// ---------------------------------------------------------------------------

mod tool_effects_test {
    use super::*;
    use crate::ids::{CallId, RunId};
    use crate::tool::{
        LedgerFailure, ToolEffect, ToolEffectLedger, ToolEffectSettle, ToolEffectStart,
        ToolEffectStatus,
    };
    use std::collections::HashMap;
    use tokio::sync::Notify;

    /// In-memory [`ToolEffectLedger`] test double. Optionally fails every
    /// `started` write (`fail_started`) and/or signals a [`Notify`] the
    /// instant a `started` write lands (`on_started`), so a test can await
    /// "the ledger has recorded this call as in flight" before acting.
    #[derive(Clone, Default)]
    struct InMemoryToolEffectLedger {
        inner: Arc<Mutex<HashMap<(String, String), ToolEffect>>>,
        fail_started: bool,
        on_started: Option<Arc<Notify>>,
    }

    impl InMemoryToolEffectLedger {
        fn get(&self, run_id: &str, call_id: &str) -> Option<ToolEffect> {
            self.inner
                .lock()
                .unwrap()
                .get(&(run_id.to_string(), call_id.to_string()))
                .cloned()
        }
    }

    #[async_trait]
    impl ToolEffectLedger for InMemoryToolEffectLedger {
        async fn started(&self, start: ToolEffectStart) -> Result<()> {
            if self.fail_started {
                return Err(TinyAgentsError::Tool("ledger unavailable".to_string()));
            }
            let effect = ToolEffect {
                run_id: start.run_id.as_str().to_string(),
                call_id: start.call_id.as_str().to_string(),
                tool: start.tool,
                status: ToolEffectStatus::Started,
                idempotency_key: Some(start.idempotency_key),
                effect_summary: start.effect_summary,
                started_at: chrono::Utc::now(),
                settled_at: None,
            };
            self.inner
                .lock()
                .unwrap()
                .insert((effect.run_id.clone(), effect.call_id.clone()), effect);
            if let Some(notify) = &self.on_started {
                notify.notify_one();
            }
            Ok(())
        }

        async fn settled(&self, settle: ToolEffectSettle) -> Result<()> {
            let run_id = settle.run_id.as_str().to_string();
            let call_id = settle.call_id.as_str().to_string();
            let mut guard = self.inner.lock().unwrap();
            let entry = guard
                .entry((run_id.clone(), call_id.clone()))
                .or_insert_with(|| ToolEffect {
                    run_id,
                    call_id,
                    tool: String::new(),
                    status: ToolEffectStatus::Started,
                    idempotency_key: None,
                    effect_summary: None,
                    started_at: chrono::Utc::now(),
                    settled_at: None,
                });
            entry.status = settle.status;
            entry.settled_at = Some(chrono::Utc::now());
            if let Some(summary) = settle.effect_summary {
                entry.effect_summary = Some(summary);
            }
            Ok(())
        }

        async fn unresolved(&self, run_id: &str) -> Result<Vec<ToolEffect>> {
            Ok(self
                .inner
                .lock()
                .unwrap()
                .values()
                .filter(|effect| {
                    effect.run_id == run_id && effect.status == ToolEffectStatus::Started
                })
                .cloned()
                .collect())
        }
    }

    /// A tool that never returns, used to prove an interrupted call (dropped
    /// mid-flight, after its `started` ledger row landed) leaves that row
    /// unresolved.
    struct NeverFinishesTool {
        notify: Arc<Notify>,
    }

    #[async_trait]
    impl Tool for NeverFinishesTool {
        fn name(&self) -> &str {
            "hang"
        }
        fn description(&self) -> &str {
            "never returns"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({"type": "object"})
        }
        async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
            // Never notified by the test, so this hangs until the caller
            // drops/aborts the run future.
            self.notify.notified().await;
            Ok(ToolResult::success("unreachable"))
        }
    }

    /// A tool that declares an explicit [`tinytools::ToolReplay`] policy, used
    /// to drive [`AgentHarness::reconcile_tool_effects`] down each branch.
    struct ReplayTool {
        name: &'static str,
        replay: tinytools::ToolReplay,
    }

    #[async_trait]
    impl Tool for ReplayTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "replay-classified tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({"type": "object"})
        }
        fn policy(&self) -> tinytools::ToolPolicy {
            tinytools::ToolPolicy::default().with_runtime(tinytools::ToolRuntime {
                replay: self.replay,
                ..tinytools::ToolRuntime::default()
            })
        }
        async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::success("re-executed"))
        }
    }

    #[tokio::test]
    async fn agent_loop_writes_started_then_completed_around_a_tool_call() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "lookup", json!({"q": "x"})),
                text_response("done", 4, 2),
            ])),
        );
        harness.register_tool(Arc::new(FakeTool::new("lookup", "tool-output")));

        let ledger = Arc::new(InMemoryToolEffectLedger::default());
        let ctx: RunContext<()> =
            RunContext::new(RunConfig::new("run-1"), ()).with_tool_effect_ledger(ledger.clone());

        harness
            .invoke_in_context(&(), ctx, vec![Message::user("please look up")])
            .await
            .expect("run succeeds");

        let effect = ledger
            .get("run-1", "call-1")
            .expect("ledger recorded the call");
        assert_eq!(effect.tool, "lookup");
        assert_eq!(effect.status, ToolEffectStatus::Completed);
        assert!(effect.settled_at.is_some());
    }

    #[tokio::test]
    async fn crash_between_started_and_settled_leaves_an_unresolved_row() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![tool_call_response(
                "call-1",
                "hang",
                json!({}),
            )])),
        );
        let hang_notify = Arc::new(Notify::new());
        harness.register_tool(Arc::new(NeverFinishesTool {
            notify: hang_notify.clone(),
        }));

        let started_signal = Arc::new(Notify::new());
        let ledger = Arc::new(InMemoryToolEffectLedger {
            on_started: Some(started_signal.clone()),
            ..Default::default()
        });
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-crash"), ())
            .with_tool_effect_ledger(ledger.clone());

        let harness = Arc::new(harness);
        let run_harness = harness.clone();
        let handle = tokio::spawn(async move {
            let _ = run_harness
                .invoke_in_context(&(), ctx, vec![Message::user("go")])
                .await;
        });

        // Wait for the ledger to observe the `started` write, then drop the
        // run future (abort) before the tool — which never resolves on its
        // own — could possibly settle it. This simulates a process crash
        // between admission and settlement (TOOL-effect equivalent of a
        // mid-flight kill).
        started_signal.notified().await;
        handle.abort();
        let _ = handle.await;

        let unresolved = ledger
            .unresolved("run-crash")
            .await
            .expect("ledger read succeeds");
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].call_id, "call-1");
        assert_eq!(unresolved[0].status, ToolEffectStatus::Started);
    }

    #[tokio::test]
    async fn reconcile_leaves_a_safe_replay_call_pending_for_re_execution() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_tool(Arc::new(ReplayTool {
            name: "safe_tool",
            replay: tinytools::ToolReplay::Safe,
        }));

        let ledger = Arc::new(InMemoryToolEffectLedger::default());
        ledger
            .started(ToolEffectStart {
                run_id: RunId::new("run-1"),
                call_id: CallId::new("call-1"),
                tool: "safe_tool".to_string(),
                idempotency_key: "key".to_string(),
                effect_summary: None,
            })
            .await
            .unwrap();

        let recorder = crate::testkit::EventRecorder::new();
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-1"), ())
            .with_events(recorder.sink())
            .with_tool_effect_ledger(ledger.clone());

        let mut messages = vec![
            Message::user("go"),
            Message::Assistant(AssistantMessage {
                id: None,
                content: Vec::new(),
                tool_calls: vec![ToolCall::new("call-1", "safe_tool", json!({}))],
                usage: None,
                origin: None,
            }),
        ];

        let synthesized = harness
            .reconcile_tool_effects(&ctx, "run-1", &mut messages, &Default::default())
            .await
            .unwrap();

        assert!(synthesized.is_empty());
        assert_eq!(
            messages.len(),
            2,
            "no tool answer appended for a Safe-replay call — the loop must \
             re-execute it"
        );
        assert!(
            recorder
                .kinds()
                .contains(&"tool.effect_reconciled".to_string())
        );
        // The ledger row is untouched (still `started`): re-execution will
        // settle it normally through the ordinary started/settled path.
        assert_eq!(
            ledger.get("run-1", "call-1").unwrap().status,
            ToolEffectStatus::Started
        );
    }

    #[tokio::test]
    async fn reconcile_synthesizes_an_interrupted_result_for_a_never_replay_call() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_tool(Arc::new(ReplayTool {
            name: "risky_tool",
            replay: tinytools::ToolReplay::Never,
        }));

        let ledger = Arc::new(InMemoryToolEffectLedger::default());
        ledger
            .started(ToolEffectStart {
                run_id: RunId::new("run-1"),
                call_id: CallId::new("call-1"),
                tool: "risky_tool".to_string(),
                idempotency_key: "key".to_string(),
                effect_summary: None,
            })
            .await
            .unwrap();

        let recorder = crate::testkit::EventRecorder::new();
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-1"), ())
            .with_events(recorder.sink())
            .with_tool_effect_ledger(ledger.clone());

        let mut messages = vec![
            Message::user("go"),
            Message::Assistant(AssistantMessage {
                id: None,
                content: Vec::new(),
                tool_calls: vec![ToolCall::new("call-1", "risky_tool", json!({}))],
                usage: None,
                origin: None,
            }),
        ];

        let synthesized = harness
            .reconcile_tool_effects(&ctx, "run-1", &mut messages, &Default::default())
            .await
            .unwrap();

        assert_eq!(synthesized.len(), 1);
        assert!(matches!(synthesized[0], Message::Tool(_)));
        assert_eq!(synthesized[0].text(), "interrupted before settlement");
        assert_eq!(messages.len(), 3, "an interrupted answer was appended");
        assert_eq!(
            ledger.get("run-1", "call-1").unwrap().status,
            ToolEffectStatus::Interrupted
        );
        assert!(
            recorder
                .kinds()
                .contains(&"tool.effect_reconciled".to_string())
        );
    }

    #[tokio::test]
    async fn ledger_started_failure_aborts_the_run_under_the_default_policy() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![tool_call_response(
                "call-1",
                "lookup",
                json!({}),
            )])),
        );
        harness.register_tool(Arc::new(FakeTool::new("lookup", "tool-output")));

        let ledger = Arc::new(InMemoryToolEffectLedger {
            fail_started: true,
            ..Default::default()
        });
        let ctx: RunContext<()> =
            RunContext::new(RunConfig::new("run-1"), ()).with_tool_effect_ledger(ledger);

        let err = harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .expect_err("LedgerFailure::Abort (the default) must fail the call");
        assert!(matches!(err, TinyAgentsError::Tool(_)));
    }

    #[tokio::test]
    async fn ledger_started_failure_is_ignored_under_the_continue_policy() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "lookup", json!({})),
                text_response("done", 4, 2),
            ])),
        );
        let tool = Arc::new(FakeTool::new("lookup", "tool-output"));
        harness.register_tool(tool.clone());

        let ledger = Arc::new(InMemoryToolEffectLedger {
            fail_started: true,
            ..Default::default()
        });
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-1"), ())
            .with_tool_effect_ledger(ledger)
            .with_tool_effect_ledger_failure(LedgerFailure::Continue);

        let run = harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .expect("LedgerFailure::Continue must not fail the run");
        assert_eq!(run.tool_calls, 1);
    }

    // ── `resume_deferred` + tool-effect ledger (reconcile hazard fix) ───────

    use crate::tool::DeferredToolResults;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Raises `ApprovalRequired` on its first invocation (mid-execution
    /// deferral); on every later invocation, either succeeds or fails per
    /// `fail_on_retry`, so a test can drive both the `Completed` and `Failed`
    /// post-resume ledger transitions from the same tool shape.
    struct ApprovalOnceTool {
        attempts: AtomicUsize,
        fail_on_retry: bool,
    }

    #[async_trait]
    impl Tool for ApprovalOnceTool {
        fn name(&self) -> &str {
            "wire"
        }
        fn description(&self) -> &str {
            "defers once, then runs for real"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({"type": "object"})
        }
        async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
            if self.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(TinyAgentsError::ApprovalRequired {
                    metadata: serde_json::Value::Null,
                }
                .into());
            }
            if self.fail_on_retry {
                anyhow::bail!("boom");
            }
            Ok(ToolResult::success("approved-result"))
        }
    }

    #[tokio::test]
    async fn resume_deferred_settles_the_deferred_row_completed_without_a_synthesized_crash_answer()
    {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("call-1", "wire", json!({})),
                text_response("all done", 4, 2),
            ])),
        );
        harness.register_tool(Arc::new(ApprovalOnceTool {
            attempts: AtomicUsize::new(0),
            fail_on_retry: false,
        }));

        let ledger = Arc::new(InMemoryToolEffectLedger::default());
        let ctx: RunContext<()> =
            RunContext::new(RunConfig::new("run-1"), ()).with_tool_effect_ledger(ledger.clone());
        let first = harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .expect("a mid-execution deferral is not an error");
        let pending = first.deferred.clone().expect("approval pending");
        assert_eq!(pending.approvals[0].id, "call-1");

        // The call left `started` only briefly: `defer_started_tool_call`
        // settles it `Deferred` the instant the deferral is filed, not left
        // `started` for `reconcile_tool_effects` to mistake for a crash.
        let effect = ledger
            .get("run-1", "call-1")
            .expect("ledger recorded the call");
        assert_eq!(effect.status, ToolEffectStatus::Deferred);
        assert!(effect.settled_at.is_some());

        let results = DeferredToolResults::new().approve("call-1");
        let ctx2: RunContext<()> =
            RunContext::new(RunConfig::new("run-1"), ()).with_tool_effect_ledger(ledger.clone());
        let run = harness
            .resume_deferred(&(), ctx2, first.messages.clone(), results)
            .await
            .expect("resume completes the run");

        // The call must receive its real answer, never the synthesized
        // "interrupted before settlement" a crash-reconcile would produce.
        let answer = run.messages.iter().find_map(|message| match message {
            Message::Tool(tool) if tool.tool_call_id == "call-1" => Some(message.text()),
            _ => None,
        });
        assert_eq!(answer.as_deref(), Some("approved-result"));
        assert_eq!(run.text().as_deref(), Some("all done"));

        assert_eq!(
            ledger.get("run-1", "call-1").unwrap().status,
            ToolEffectStatus::Completed
        );
    }

    #[tokio::test]
    async fn resume_deferred_settles_the_deferred_row_failed_when_the_retry_errors() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![tool_call_response(
                "call-1",
                "wire",
                json!({}),
            )])),
        );
        harness.register_tool(Arc::new(ApprovalOnceTool {
            attempts: AtomicUsize::new(0),
            fail_on_retry: true,
        }));

        let ledger = Arc::new(InMemoryToolEffectLedger::default());
        let ctx: RunContext<()> =
            RunContext::new(RunConfig::new("run-1"), ()).with_tool_effect_ledger(ledger.clone());
        let first = harness
            .invoke_in_context(&(), ctx, vec![Message::user("go")])
            .await
            .expect("a mid-execution deferral is not an error");
        assert_eq!(
            ledger.get("run-1", "call-1").unwrap().status,
            ToolEffectStatus::Deferred
        );

        let results = DeferredToolResults::new().approve("call-1");
        let ctx2: RunContext<()> =
            RunContext::new(RunConfig::new("run-1"), ()).with_tool_effect_ledger(ledger.clone());
        harness
            .resume_deferred(&(), ctx2, first.messages.clone(), results)
            .await
            .expect_err("the retried execution error propagates");

        assert_eq!(
            ledger.get("run-1", "call-1").unwrap().status,
            ToolEffectStatus::Failed
        );
    }

    #[tokio::test]
    async fn resume_deferred_reconciles_a_crashed_sibling_but_excludes_the_call_it_resolves() {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![text_response(
                "all done", 4, 2,
            )])),
        );
        harness.register_tool(Arc::new(FakeTool::new("approve_tool", "approved-result")));
        harness.register_tool(Arc::new(ReplayTool {
            name: "risky_tool",
            replay: tinytools::ToolReplay::Never,
        }));

        let ledger = Arc::new(InMemoryToolEffectLedger::default());
        // `call-a` was deliberately deferred mid-execution (already settled
        // `Deferred`, matching what `defer_started_tool_call` does).
        ledger
            .started(ToolEffectStart {
                run_id: RunId::new("run-x"),
                call_id: CallId::new("call-a"),
                tool: "approve_tool".to_string(),
                idempotency_key: "key-a".to_string(),
                effect_summary: None,
            })
            .await
            .unwrap();
        ledger
            .settled(ToolEffectSettle {
                run_id: RunId::new("run-x"),
                call_id: CallId::new("call-a"),
                status: ToolEffectStatus::Deferred,
                effect_summary: None,
            })
            .await
            .unwrap();
        // `call-b`'s process crashed mid-flight: admitted and started, but
        // never settled or deferred — the genuine crash artifact.
        ledger
            .started(ToolEffectStart {
                run_id: RunId::new("run-x"),
                call_id: CallId::new("call-b"),
                tool: "risky_tool".to_string(),
                idempotency_key: "key-b".to_string(),
                effect_summary: None,
            })
            .await
            .unwrap();

        let recorder = crate::testkit::EventRecorder::new();
        let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-x"), ())
            .with_events(recorder.sink())
            .with_tool_effect_ledger(ledger.clone());

        let messages = vec![
            Message::user("go"),
            Message::Assistant(AssistantMessage {
                id: None,
                content: Vec::new(),
                tool_calls: vec![
                    ToolCall::new("call-a", "approve_tool", json!({})),
                    ToolCall::new("call-b", "risky_tool", json!({})),
                ],
                usage: None,
                origin: None,
            }),
        ];
        let results = DeferredToolResults::new().approve("call-a");

        let run = harness
            .resume_deferred(&(), ctx, messages, results)
            .await
            .expect("resume completes despite the crashed sibling");

        // `call-b` had no live `results` entry: `reconcile_tool_effects`
        // (run before `results` is applied) answered it as interrupted,
        // per its default `ToolReplay::Never`.
        let call_b_answer = run.messages.iter().find_map(|message| match message {
            Message::Tool(tool) if tool.tool_call_id == "call-b" => Some(message.text()),
            _ => None,
        });
        assert_eq!(
            call_b_answer.as_deref(),
            Some("interrupted before settlement")
        );
        assert_eq!(
            ledger.get("run-x", "call-b").unwrap().status,
            ToolEffectStatus::Interrupted
        );

        // `call-a` is exactly what `results` resolves: it must not receive a
        // synthesized crash answer, and it ran for real through the approval
        // path instead.
        let call_a_answer = run.messages.iter().find_map(|message| match message {
            Message::Tool(tool) if tool.tool_call_id == "call-a" => Some(message.text()),
            _ => None,
        });
        assert_eq!(call_a_answer.as_deref(), Some("approved-result"));
        assert_eq!(
            ledger.get("run-x", "call-a").unwrap().status,
            ToolEffectStatus::Completed
        );

        assert!(
            recorder
                .kinds()
                .contains(&"tool.effect_reconciled".to_string()),
            "the crashed sibling was reconciled"
        );
    }
}

#[tokio::test]
async fn tiered_system_messages_become_one_cacheable_segment_each() {
    use crate::cache::PROMPT_CACHE_KEY_OPTION;
    use tinyinference_llm::cache::CachePolicy;
    // A host that renders its system prompt in tiers sends them as consecutive
    // leading system messages. The request the model sees must keep one
    // segment per tier (so a rewritten volatile tier is attributable) and the
    // fingerprint must cover both, in order.
    let model = Arc::new(crate::testkit::ScriptedModel::replies(vec!["done"]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.with_policy(RunPolicy {
        cache: CachePolicy {
            protect_prompt_prefix: true,
            ..CachePolicy::default()
        },
        ..RunPolicy::default()
    });

    let stable = Message::system("identity and rules");
    let volatile = Message::system("connected services this session");
    harness
        .invoke_default(
            &(),
            vec![stable.clone(), volatile.clone(), Message::user("hello")],
        )
        .await
        .expect("run succeeds");

    let request = model
        .requests()
        .into_iter()
        .next()
        .expect("model received one request");
    let ids: Vec<&str> = request
        .cache_segments
        .iter()
        .map(|segment| segment.id.as_str())
        .collect();
    assert_eq!(ids, vec!["system", "system.1"]);
    assert_eq!(request.messages[0], stable);
    assert_eq!(request.messages[1], volatile);

    let mut expected = crate::prompt::PromptBuilder::new();
    expected.push_system_messages(&[stable, volatile]);
    assert_eq!(
        request.prompt_fingerprint,
        expected.build(Vec::new()).prompt_fingerprint
    );
    assert!(request.provider_options[PROMPT_CACHE_KEY_OPTION].is_string());
}

/// A middleware that declares the harness layout while the schemas are still
/// on the request (`system`, `tools`) must keep its stable-prefix fingerprint
/// under a text dialect, which folds the catalogue into the prompt and clears
/// `tools` afterwards. Before, the rebuilt layout (`system` only) no longer
/// matched the declaration, the request fell through to the whole-request
/// digest, and the provider routing key changed on every call of a thread.
#[test]
fn stripped_tools_segment_still_counts_as_the_harness_layout() {
    use tinyinference_llm::model::{PromptSegment, SegmentRole};
    let system = Message::system("identity and rules");
    let declared = |request: &mut ModelRequest| {
        request.cache_segments = vec![
            PromptSegment {
                id: "system".to_string(),
                role: SegmentRole::System,
                cacheable: true,
            },
            PromptSegment {
                id: "tools".to_string(),
                role: SegmentRole::Tools,
                cacheable: true,
            },
        ];
    };

    // Same declaration, two turns of one thread, schemas already stripped.
    let mut turn_one = ModelRequest::new(vec![system.clone(), Message::user("hi")]);
    declared(&mut turn_one);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut turn_one);
    let mut turn_two = ModelRequest::new(vec![
        system.clone(),
        Message::user("hi"),
        Message::assistant("hello"),
        Message::user("later"),
    ]);
    declared(&mut turn_two);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut turn_two);

    let ids: Vec<&str> = turn_one
        .cache_segments
        .iter()
        .map(|segment| segment.id.as_str())
        .collect();
    assert_eq!(ids, vec!["system"], "the stripped tools segment is dropped");
    let mut expected = crate::prompt::PromptBuilder::new();
    expected.push_system_messages(std::slice::from_ref(&system));
    assert_eq!(
        turn_one.prompt_fingerprint,
        expected.build(Vec::new()).prompt_fingerprint,
        "the fingerprint is the stable-prefix one, not a whole-request digest"
    );
    assert_eq!(turn_one.prompt_fingerprint, turn_two.prompt_fingerprint);

    // A genuinely custom annotation still takes the conservative path.
    let mut custom = ModelRequest::new(vec![system, Message::user("hi")]);
    custom.cache_segments = vec![PromptSegment {
        id: "system:abc".to_string(),
        role: SegmentRole::System,
        cacheable: true,
    }];
    super::run_loop::refresh_prompt_cache_fingerprint(&mut custom);
    assert_ne!(custom.prompt_fingerprint, turn_one.prompt_fingerprint);
}

#[test]
fn compaction_summary_keeps_declared_system_prefix_cache_key() {
    use tinyinference_llm::model::{PromptSegment, SegmentRole};

    let segments = vec![
        PromptSegment {
            id: "system".into(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "system.1".into(),
            role: SegmentRole::System,
            cacheable: true,
        },
    ];
    let mut before = ModelRequest::new(vec![
        Message::system("stable"),
        Message::system("context"),
        Message::user("first"),
    ]);
    before.cache_segments = segments.clone();
    before.prompt_fingerprint = Some("pre-dispatch annotation".into());
    let mut after = ModelRequest::new(vec![
        Message::system("stable"),
        Message::system("context"),
        Message::system("changing history summary"),
        Message::user("later"),
    ]);
    after.cache_segments = segments;
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    super::run_loop::refresh_prompt_cache_fingerprint(&mut before);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut after);

    assert_eq!(before.cache_segments, after.cache_segments);
    assert_eq!(before.prompt_fingerprint, after.prompt_fingerprint);
    assert_eq!(
        crate::cache::prompt_cache_key(&before),
        crate::cache::prompt_cache_key(&after)
    );
    assert_ne!(
        crate::cache::cache_key(&before),
        crate::cache::cache_key(&after),
        "the local response cache must still distinguish the changed history"
    );
}

#[test]
fn changing_or_prepending_a_declared_system_message_invalidates_the_prefix() {
    let build = |system: &str| {
        let mut prompt = crate::prompt::PromptBuilder::new();
        prompt.push_system_messages(&[Message::system(system)]);
        prompt.build(vec![Message::user("question")])
    };
    let original = build("stable");
    let changed = build("revised");
    assert_ne!(
        crate::cache::prompt_cache_key(&original),
        crate::cache::prompt_cache_key(&changed)
    );
    assert!(
        !crate::cache::PromptCacheLayout::from_request(&original)
            .is_prefix_stable_against(&crate::cache::PromptCacheLayout::from_request(&changed))
    );

    let mut prepended = original.clone();
    crate::cache::prepend_system_message(&mut prepended, "new instruction".into());
    assert_ne!(original.prompt_fingerprint, prepended.prompt_fingerprint);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut prepended);
    assert_ne!(
        crate::cache::prompt_cache_key(&original),
        crate::cache::prompt_cache_key(&prepended)
    );
    assert!(
        !crate::cache::PromptCacheLayout::from_request(&original)
            .is_prefix_stable_against(&crate::cache::PromptCacheLayout::from_request(&prepended))
    );
}

#[test]
fn rebuilt_session_request_keeps_the_summary_after_frozen_system_tiers() {
    let messages = vec![
        Message::system("stable"),
        Message::system("context"),
        Message::system("changing summary"),
        Message::user("later"),
    ];
    let system_end = super::run_loop::cacheable_system_prefix_end(&messages, Some(2));
    assert_eq!(system_end, 2);
    assert_eq!(
        super::run_loop::cacheable_system_prefix_end(&messages, None),
        3,
        "standalone harnesses retain their leading-System fallback"
    );
    let mut prompt = crate::prompt::PromptBuilder::new();
    prompt.push_system_messages(&messages[..system_end]);
    let mut request = prompt.build(messages[system_end..].to_vec());

    assert_eq!(request.cache_segments.len(), 2);
    assert_eq!(request.cache_segments[0].id, "system");
    assert_eq!(request.cache_segments[1].id, "system.1");
    assert_eq!(request.messages[2].text(), "changing summary");

    let mut previous = ModelRequest::new(vec![
        Message::system("stable"),
        Message::system("context"),
        Message::user("first"),
    ]);
    previous.cache_segments = request.cache_segments.clone();
    previous.prompt_fingerprint = request.prompt_fingerprint.clone();
    super::run_loop::refresh_prompt_cache_fingerprint(&mut previous);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
    assert_eq!(
        crate::cache::prompt_cache_key(&previous),
        crate::cache::prompt_cache_key(&request)
    );
    assert_ne!(
        crate::cache::cache_key(&previous),
        crate::cache::cache_key(&request),
        "a stable provider route must not merge distinct response-cache entries"
    );
}

#[test]
fn empty_frozen_prefix_does_not_promote_a_system_summary() {
    let build = |summary: &str| {
        let messages = vec![Message::system(summary), Message::user("later")];
        let end = super::run_loop::cacheable_system_prefix_end(&messages, Some(0));
        assert_eq!(end, 0);
        // The request starts user-only. Context compression inserts the
        // System summary after the marker has been declared.
        let mut request = crate::prompt::PromptBuilder::new().build(vec![Message::user("later")]);
        super::run_loop::mark_empty_frozen_prefix(&mut request, Some(0));
        request.messages.insert(0, Message::system(summary));
        super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
        request
    };

    let first = build("summary A");
    let second = build("summary B");
    assert!(first.cacheable_prefix_ids().is_empty());
    assert_eq!(
        first.cache_segments[0].role,
        tinyinference_llm::model::SegmentRole::Volatile
    );
    assert!(!first.cache_segments[0].cacheable);
    assert_ne!(first.prompt_fingerprint, second.prompt_fingerprint);
}

#[test]
fn stable_prepend_promotes_the_zero_prefix_marker_without_caching_history() {
    let build = |summary: &str| {
        let mut request = crate::prompt::PromptBuilder::new().build(vec![Message::user("later")]);
        super::run_loop::mark_empty_frozen_prefix(&mut request, Some(0));
        request.messages.insert(0, Message::system(summary));
        crate::cache::prepend_system_message(&mut request, "dynamic instruction".into());
        super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
        request
    };

    let first = build("summary A");
    let second = build("summary B");
    assert_eq!(first.cache_segments.len(), 1);
    assert_eq!(first.cache_segments[0].id, "system");
    assert!(first.cache_segments[0].cacheable);
    assert_eq!(first.messages[0].text(), "dynamic instruction");
    assert_eq!(first.messages[1].text(), "summary A");
    assert_eq!(
        crate::cache::prompt_cache_key(&first),
        crate::cache::prompt_cache_key(&second)
    );
    assert_ne!(
        crate::cache::cache_key(&first),
        crate::cache::cache_key(&second)
    );
}

#[test]
fn tools_added_after_zero_prefix_marking_keep_summary_volatile() {
    use tinyinference_llm::model::SegmentRole;
    use tinyinference_llm::tool::ToolSchema;

    let build = |summary: &str| {
        let mut request = crate::prompt::PromptBuilder::new().build(vec![Message::user("later")]);
        super::run_loop::mark_empty_frozen_prefix(&mut request, Some(0));
        request.tools = vec![ToolSchema::new(
            "lookup",
            "look up facts",
            serde_json::json!({"type": "object"}),
        )];
        request.messages.insert(0, Message::system(summary));
        super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
        request
    };

    let first = build("summary A");
    let second = build("summary B");
    assert_eq!(first.cache_segments.len(), 1);
    assert_eq!(first.cache_segments[0].id, "tools");
    assert_eq!(first.cache_segments[0].role, SegmentRole::Tools);
    assert!(first.cache_segments[0].cacheable);
    assert_eq!(
        crate::cache::prompt_cache_key(&first),
        crate::cache::prompt_cache_key(&second)
    );
    assert_ne!(
        crate::cache::cache_key(&first),
        crate::cache::cache_key(&second)
    );
}

#[test]
fn zero_prefix_stable_prepend_and_tools_keep_both_segments_in_either_order() {
    use tinyinference_llm::tool::ToolSchema;

    let build = |summary: &str, tools_first: bool| {
        let mut request = crate::prompt::PromptBuilder::new().build(vec![Message::user("later")]);
        super::run_loop::mark_empty_frozen_prefix(&mut request, Some(0));
        request.messages.insert(0, Message::system(summary));
        let add_tools = |request: &mut ModelRequest| {
            request.tools = vec![ToolSchema::new(
                "lookup",
                "look up facts",
                serde_json::json!({"type": "object"}),
            )];
        };
        if tools_first {
            add_tools(&mut request);
        }
        crate::cache::prepend_system_message(&mut request, "dynamic instruction".into());
        if !tools_first {
            add_tools(&mut request);
        }
        super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
        request
    };

    for tools_first in [false, true] {
        let first = build("summary A", tools_first);
        let second = build("summary B", tools_first);
        assert_eq!(
            first.cacheable_prefix_ids(),
            vec!["system".to_string(), "tools".to_string()]
        );
        assert_eq!(first.cache_segments.len(), 2);
        assert_eq!(
            crate::cache::prompt_cache_key(&first),
            crate::cache::prompt_cache_key(&second)
        );
        assert_ne!(
            crate::cache::cache_key(&first),
            crate::cache::cache_key(&second)
        );
    }
    assert_eq!(
        crate::cache::prompt_cache_key(&build("summary A", false)),
        crate::cache::prompt_cache_key(&build("summary A", true)),
    );
}

#[test]
fn tools_only_prefix_survives_a_leading_compaction_summary() {
    use tinyinference_llm::tool::ToolSchema;

    let tool = ToolSchema::new(
        "lookup",
        "look up facts",
        serde_json::json!({"type": "object"}),
    );
    let mut before = ModelRequest::new(vec![Message::user("first")]).with_tools(vec![tool.clone()]);
    super::run_loop::mark_empty_frozen_prefix(&mut before, Some(0));
    let mut after = ModelRequest::new(vec![Message::user("later")]).with_tools(vec![tool]);
    super::run_loop::mark_empty_frozen_prefix(&mut after, Some(0));
    after
        .messages
        .insert(0, Message::system("changing history summary"));

    super::run_loop::refresh_prompt_cache_fingerprint(&mut before);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut after);

    assert_eq!(before.cache_segments, after.cache_segments);
    assert_eq!(before.prompt_fingerprint, after.prompt_fingerprint);
}

#[test]
fn fingerprint_without_declared_segments_still_hashes_a_new_system_message() {
    let mut before = ModelRequest::new(vec![Message::user("first")]);
    before.prompt_fingerprint = Some("stale".into());
    let mut after = ModelRequest::new(vec![
        Message::system("new instruction"),
        Message::user("later"),
    ]);
    after.prompt_fingerprint = before.prompt_fingerprint.clone();

    super::run_loop::refresh_prompt_cache_fingerprint(&mut before);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut after);

    assert_ne!(before.prompt_fingerprint, after.prompt_fingerprint);
}

#[test]
fn grouped_system_segment_cannot_masquerade_as_one_canonical_message() {
    let mut builder = crate::prompt::PromptBuilder::new();
    builder.push_system(
        "system",
        vec![Message::system("first"), Message::system("second")],
    );
    let mut before = builder.build(vec![Message::user("question")]);
    assert_ne!(before.cache_segments[0].id, "system");

    let mut after = before.clone();
    after.messages[1] = Message::system("changed second");
    super::run_loop::refresh_prompt_cache_fingerprint(&mut before);
    super::run_loop::refresh_prompt_cache_fingerprint(&mut after);

    assert_ne!(before.prompt_fingerprint, after.prompt_fingerprint);
}

#[test]
fn prompted_schema_instruction_preserves_all_original_system_tiers() {
    use tinyinference_llm::model::{PromptSegment, SegmentRole};

    let segments = (0..2)
        .map(|index| PromptSegment {
            id: crate::prompt::system_segment_id(index),
            role: SegmentRole::System,
            cacheable: true,
        })
        .collect::<Vec<_>>();
    let build = |second: &str| {
        let mut request = ModelRequest::new(vec![
            Message::system("first"),
            Message::system(second),
            Message::user("question"),
        ]);
        request.cache_segments = segments.clone();
        request.prompt_fingerprint = Some("pre-structured-annotation".into());
        crate::cache::prepend_system_message(&mut request, "JSON Schema: fixed".into());
        super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
        request
    };

    let before = build("second A");
    let after = build("second B");
    let expected_segments = (0..3)
        .map(|index| PromptSegment {
            id: crate::prompt::system_segment_id(index),
            role: SegmentRole::System,
            cacheable: true,
        })
        .collect::<Vec<_>>();
    assert_eq!(before.cache_segments, expected_segments);
    assert_eq!(after.cache_segments, expected_segments);
    assert_eq!(before.messages[0].text(), "JSON Schema: fixed");
    assert_eq!(before.messages[1].text(), "first");
    assert_eq!(before.messages[2].text(), "second A");
    assert_eq!(after.messages[2].text(), "second B");
    assert_ne!(before.prompt_fingerprint, after.prompt_fingerprint);
}

/// A text-dialect run that starts with *no* leading system message declares
/// only the `tools` segment (`PromptBuilder` has no system prefix to name
/// yet). The dialect then synthesizes exactly one new leading system message
/// for its protocol block (`prompt_tools::append_system_block` inserts one
/// when none exists) and clears `tools`. That single synthesized segment is
/// still the harness's own dialect rewrite, not a custom annotation, and
/// must keep the stable-prefix fingerprint rather than falling through to
/// the whole-request digest — which would re-roll the provider routing key
/// as the transcript grows even though the leading system content itself
/// (the dialect's protocol block) never changes.
///
/// Runs the actual dialect rewrite (`RunDialect::apply_to_request`), not a
/// hand-constructed post-rewrite `cache_segments`: the synthesis is resolved
/// there (`sync_stripped_tools_cache_segment`), using the pre-rewrite
/// message shape this test needs to be real for.
#[test]
fn a_dialect_synthesized_first_system_segment_still_counts_as_the_harness_layout() {
    use tinyinference_llm::model::{PromptSegment, SegmentRole, ToolChoice};
    use tinyinference_llm::tool::ToolSchema;

    let tool = ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    );
    let dialect = super::dialect::RunDialect::resolve(
        crate::config::ToolDispatcher::Xml,
        std::slice::from_ref(&tool),
        Some(false),
    );
    let build_turn = |history: Vec<Message>| {
        let mut request = ModelRequest::new(history).with_tools(vec![tool.clone()]);
        request.tool_choice = ToolChoice::Auto;
        // Declared before the rewrite: no leading system message exists yet
        // (`history` is user-only), so only the tools segment is named.
        request.cache_segments = vec![PromptSegment {
            id: "tools".to_string(),
            role: SegmentRole::Tools,
            cacheable: true,
        }];
        dialect.apply_to_request(&mut request, false, &[]);
        super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
        request
    };

    let turn_one = build_turn(vec![Message::user("hi")]);
    let turn_two = build_turn(vec![
        Message::user("hi"),
        Message::assistant("hello"),
        Message::user("later"),
    ]);

    let ids: Vec<&str> = turn_one
        .cache_segments
        .iter()
        .map(|segment| segment.id.as_str())
        .collect();
    assert_eq!(ids, vec!["system"], "the stripped tools segment is dropped");
    assert!(turn_one.prompt_fingerprint.is_some());
    assert_eq!(
        turn_one.prompt_fingerprint, turn_two.prompt_fingerprint,
        "the routing key must not change as the transcript grows, since the \
         synthesized protocol block is identical every turn"
    );
}

/// The counterexample to the synthesis case above: a leading system message
/// *already exists* at declare time, but middleware deliberately names only
/// the trailing `tools` segment — omitting that system message from the
/// cache key on purpose, e.g. because it carries per-request volatile
/// content. The dialect rewrite folds its protocol block into that existing
/// message in place (`prompt_tools::append_system_block` only ever inserts a
/// *new* leading message when none exists), so nothing was synthesized here.
/// The declaration must be left exactly as the middleware wrote it, not
/// promoted to a fresh cacheable system segment the middleware never named.
#[test]
fn a_custom_layout_omitting_an_existing_system_message_is_not_promoted_by_the_dialect_rewrite() {
    use tinyinference_llm::model::{PromptSegment, SegmentRole, ToolChoice};
    use tinyinference_llm::tool::ToolSchema;

    let tool = ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    );
    let dialect = super::dialect::RunDialect::resolve(
        crate::config::ToolDispatcher::Xml,
        std::slice::from_ref(&tool),
        Some(false),
    );
    let system = Message::system("volatile per-request content middleware keeps out of the key");
    let mut request = ModelRequest::new(vec![system, Message::user("hi")]).with_tools(vec![tool]);
    request.tool_choice = ToolChoice::Auto;
    request.cache_segments = vec![PromptSegment {
        id: "tools".to_string(),
        role: SegmentRole::Tools,
        cacheable: true,
    }];

    dialect.apply_to_request(&mut request, false, &[]);

    assert!(
        !request
            .cache_segments
            .iter()
            .any(|segment| segment.role == SegmentRole::System),
        "the rewrite must not invent a system segment the declaration never named: {:?}",
        request.cache_segments
    );

    super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
    // Falls through to the conservative whole-request digest, not the
    // stable-prefix fingerprint a harness-owned layout would get.
    let mut harness_owned = request.clone();
    harness_owned.cache_segments = vec![
        PromptSegment {
            id: "system".to_string(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "tools".to_string(),
            role: SegmentRole::Tools,
            cacheable: true,
        },
    ];
    super::run_loop::refresh_prompt_cache_fingerprint(&mut harness_owned);
    assert_ne!(request.prompt_fingerprint, harness_owned.prompt_fingerprint);
}

/// A middleware that deliberately opts a custom trailing `tools` segment out
/// of caching (`cacheable: false`) is not the harness's own canonical
/// segment, even though its role and id match. Stripping the schemas off the
/// wire must not silently promote that opt-out to cacheable by matching it
/// against the harness layout on role/id alone.
#[test]
fn a_custom_tools_segment_opted_out_of_caching_is_not_mistaken_for_the_harness_layout() {
    use tinyinference_llm::model::{PromptSegment, SegmentRole};

    let system = Message::system("identity and rules");
    let mut request = ModelRequest::new(vec![system.clone(), Message::user("hi")]);
    request.cache_segments = vec![
        PromptSegment {
            id: "system".to_string(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "tools".to_string(),
            role: SegmentRole::Tools,
            cacheable: false,
        },
    ];
    super::run_loop::refresh_prompt_cache_fingerprint(&mut request);

    // Declared segments are left untouched: this is treated as a genuinely
    // custom annotation, not silently rewritten to the harness's stripped
    // layout.
    assert_eq!(request.cache_segments.len(), 2);
    assert!(!request.cache_segments[1].cacheable);

    // And the fingerprint takes the conservative whole-request digest path,
    // not the stable-prefix one a harness-owned layout would get.
    let mut harness_owned = ModelRequest::new(vec![system, Message::user("hi")]);
    harness_owned.cache_segments = vec![
        PromptSegment {
            id: "system".to_string(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "tools".to_string(),
            role: SegmentRole::Tools,
            cacheable: true,
        },
    ];
    super::run_loop::refresh_prompt_cache_fingerprint(&mut harness_owned);
    assert_ne!(request.prompt_fingerprint, harness_owned.prompt_fingerprint);
}

/// A second custom-layout counterexample: middleware names a *non-empty*
/// but middleware-owned head ahead of the canonical trailing `tools`
/// segment — `[system:tenant, tools]` — while a leading system message
/// really does exist. `head` is non-empty here (unlike the omission case
/// above), so an earlier, less careful version of the sync helper dropped
/// the trailing tools segment unconditionally whenever `head` was
/// non-empty, silently mutating this declaration down to `[system:tenant]`
/// before `refresh_prompt_cache_fingerprint` ever got a chance to recognize
/// it as custom and take the conservative path over the *original* bytes.
/// The declaration must survive completely intact — trailing tools segment
/// included — since it never matched the harness's own canonical shape in
/// the first place.
#[test]
fn a_custom_head_that_does_not_match_the_canonical_shape_is_left_completely_untouched() {
    use tinyinference_llm::model::{PromptSegment, SegmentRole, ToolChoice};
    use tinyinference_llm::tool::ToolSchema;

    let tool = ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    );
    let dialect = super::dialect::RunDialect::resolve(
        crate::config::ToolDispatcher::Xml,
        std::slice::from_ref(&tool),
        Some(false),
    );
    let system = Message::system("tenant-scoped instructions");
    let original_segments = vec![
        PromptSegment {
            id: "system:tenant".to_string(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "tools".to_string(),
            role: SegmentRole::Tools,
            cacheable: true,
        },
    ];
    let mut request = ModelRequest::new(vec![system, Message::user("hi")]).with_tools(vec![tool]);
    request.tool_choice = ToolChoice::Auto;
    request.cache_segments = original_segments.clone();

    dialect.apply_to_request(&mut request, false, &[]);

    assert_eq!(
        request.cache_segments, original_segments,
        "a non-canonical custom head must not be partially rewritten"
    );
}

/// The exact edge case tinysweeper flagged: a host-rendered, no-tools-
/// synthesized, `Auto`-choice turn with no leading system message leaves
/// `request.messages` completely untouched (see the `block.is_empty()`
/// branch of `apply_to_request`) — no system message is ever inserted. The
/// sync helper must not synthesize a leading system cache segment for a
/// message that was never created, or `refresh_prompt_cache_fingerprint`
/// mismatches the (fictitious) declared segment against the real, empty
/// message layout and falls back to the conservative digest for a turn that
/// is actually the trivial empty-declaration case.
#[test]
fn host_rendered_with_nothing_to_say_drops_the_stale_tools_segment_without_inventing_one() {
    use tinyinference_llm::model::{ModelRequest, PromptSegment, SegmentRole, ToolChoice};
    use tinyinference_llm::tool::ToolSchema;

    let tool = ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    );
    let dialect = super::dialect::RunDialect::resolve(
        crate::config::ToolDispatcher::Xml,
        std::slice::from_ref(&tool),
        Some(false),
    );
    let mut request = ModelRequest::new(vec![Message::user("hi")]).with_tools(vec![tool]);
    request.tool_choice = ToolChoice::Auto;
    request.cache_segments = vec![PromptSegment {
        id: "tools".to_string(),
        role: SegmentRole::Tools,
        cacheable: true,
    }];

    // host_renders_catalogue = true, no synthesized tools, tool_choice::Auto:
    // `block` stays empty, so the rewrite leaves `messages` untouched.
    dialect.apply_to_request(&mut request, true, &[]);

    assert!(
        !matches!(request.messages[0], Message::System(_)),
        "no system message was actually inserted: {:?}",
        request.messages
    );
    assert!(
        request.cache_segments.is_empty(),
        "the stale tools segment must be dropped without inventing a system \
         segment for a message that does not exist: {:?}",
        request.cache_segments
    );

    super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
    assert_eq!(
        request.prompt_fingerprint, None,
        "an empty declared layout with no system messages fingerprints as \
         nothing (the trivial stable-prefix case), not a whole-request digest"
    );
}
