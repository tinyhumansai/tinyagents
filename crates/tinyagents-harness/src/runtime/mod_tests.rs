//! Tests for the [`AgentHarness`] builder and [`RunPolicy`].

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use crate::context::{RunConfig, RunContext};
use crate::host::{
    AgentMemory, AllowAllSecurityGate, BudgetGate, CallEstimate, CompressionHint, ContextComposer,
    ContextState, ErrorFieldClassifier, ExperienceStore, FixedModelResolver, GateDecision,
    InMemoryAgentMemory, InMemoryExperienceStore, LearningSink, MemoryId, ModelResolver,
    NoopLearningSink, OutcomeClass, Permit, ProgressSink, RecordingProgressSink, ScreenOutcome,
    SecurityGate, StaticContextComposer, ToolCallRequest, ToolOutcomeClassifier,
    UnlimitedBudgetGate,
};
use crate::limits::RunLimits;
use crate::middleware::{LoggingMiddleware, Middleware, ModelFallbackMiddleware};
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, AgentInvocation, AgentTurnRequest, RunPolicy};
use crate::testkit::ScriptedModel;
use futures::StreamExt;
use tinyagents_definition::{
    AgentDefinition, DefinitionRegistry, DefinitionRegistryError, InMemoryDefinitionRegistry,
};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::{
    model::{ChatModel, ModelRequest, ModelResponse, ResponseFormat},
    usage::Usage,
};
use tinytools::{Tool, ToolResult};

use async_trait::async_trait;
use serde_json::json;

struct NoopTool;

struct BlockedTool;

struct AnswerCollisionTool;

struct DenyToolGate;

struct DenyThenAllowGate {
    denials_remaining: AtomicUsize,
}

struct RedactJsonUserGate;

struct RedactDeltaMiddleware;

/// A resolver which has definitely entered its future before it stays pending.
/// This lets boundary tests cancel or time out the actual resolver work rather
/// than racing a pre-resolution checkpoint.
struct PendingResolver {
    started: Arc<tokio::sync::Notify>,
}

struct FirstThenPendingResolver {
    initial: Arc<dyn ChatModel<()>>,
    started: Arc<tokio::sync::Notify>,
    calls: AtomicUsize,
}

struct HostFallbackResolver {
    primary: Arc<dyn ChatModel<()>>,
    fallback: Arc<dyn ChatModel<()>>,
    requested_pins: Mutex<Vec<Option<String>>>,
}

struct RetryableFailingModel;

struct SecretProviderModel;

struct UsageReportingModel;

struct BorrowedStateModel;

struct SecretRecordBudget;

struct SecretAfterModelMiddleware;

struct FallbackGuidance;

#[async_trait]
impl Middleware<()> for FallbackGuidance {
    fn name(&self) -> &str {
        "fallback_guidance"
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> crate::error::Result<()> {
        crate::middleware::push_ephemeral_instruction(
            request,
            "fallback guidance",
            ctx.model_profile.as_ref(),
        );
        Ok(())
    }
}

struct RecordingArgumentGate {
    seen: Mutex<Vec<serde_json::Value>>,
}

struct BlockExtensionGate;

struct InjectedArgumentTool {
    executed: Arc<Mutex<Vec<serde_json::Value>>>,
}

struct RetryableClassifier;

struct LeadRecordingResolver {
    model: Arc<dyn ChatModel<()>>,
    team_lead_flags: Mutex<Vec<bool>>,
    model_pins: Mutex<Vec<Option<String>>>,
}

struct RecordingBudget {
    hint: CompressionHint,
    records: Mutex<Vec<Usage>>,
}

/// A permissive budget that records every `estimate.estimated_input_tokens`
/// it is asked to admit, so a test can assert the preflight estimate saw the
/// request the provider actually receives — not a smaller one taken before a
/// later rewrite grew it.
struct EstimateRecordingBudget {
    estimates: Mutex<Vec<u64>>,
}

impl EstimateRecordingBudget {
    fn new() -> Self {
        Self {
            estimates: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl BudgetGate for EstimateRecordingBudget {
    async fn acquire(&self, estimate: &CallEstimate) -> crate::error::Result<Permit> {
        self.estimates
            .lock()
            .expect("estimate budget lock")
            .push(estimate.estimated_input_tokens);
        Ok(Permit::unlimited())
    }

    async fn record(&self, _usage: &Usage) -> crate::error::Result<()> {
        Ok(())
    }

    fn compression_hint(&self, _state: &ContextState) -> CompressionHint {
        CompressionHint::None
    }
}

/// Per-invocation trace used by the overlap test below.  Each adapter writes
/// both its capability name and the identity-bearing value it received, so a
/// capability bundle accidentally borrowed from the other root is observable
/// rather than merely inferred from a final answer.
#[derive(Debug)]
struct TaggedTrace {
    tag: &'static str,
    events: Mutex<Vec<String>>,
}

impl TaggedTrace {
    fn new(tag: &'static str) -> Self {
        Self {
            tag,
            events: Mutex::new(Vec::new()),
        }
    }

    fn mark(&self, capability: &str, detail: impl std::fmt::Display) {
        self.events
            .lock()
            .expect("tag trace lock")
            .push(format!("{capability}:{detail}"));
    }

    fn has_capability(&self, capability: &str) -> bool {
        self.events
            .lock()
            .expect("tag trace lock")
            .iter()
            .any(|event| event.starts_with(&format!("{capability}:")))
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().expect("tag trace lock").clone()
    }
}

struct TaggedContext {
    trace: Arc<TaggedTrace>,
}

#[async_trait]
impl ContextComposer for TaggedContext {
    async fn compose_system_prompt(
        &self,
        request: &crate::host::TurnContextRequest,
    ) -> crate::error::Result<String> {
        self.trace.mark("context", &request.user_text);
        Ok(format!("system bundle {}", self.trace.tag))
    }

    async fn preamble(
        &self,
        request: &crate::host::TurnContextRequest,
    ) -> crate::error::Result<Vec<tinyinference_llm::message::Message>> {
        self.trace.mark("context-preamble", &request.user_text);
        Ok(Vec::new())
    }
}

struct TaggedDefinitions {
    trace: Arc<TaggedTrace>,
    inner: InMemoryDefinitionRegistry,
}

#[async_trait]
impl DefinitionRegistry for TaggedDefinitions {
    async fn resolve(
        &self,
        id: &str,
    ) -> std::result::Result<Option<AgentDefinition>, DefinitionRegistryError> {
        self.trace.mark("definition", id);
        self.inner.resolve(id).await
    }

    async fn list(&self) -> std::result::Result<Vec<AgentDefinition>, DefinitionRegistryError> {
        self.trace.mark("definition-list", self.trace.tag);
        self.inner.list().await
    }

    async fn delegates_for(
        &self,
        id: &str,
    ) -> std::result::Result<Vec<String>, DefinitionRegistryError> {
        self.trace.mark("definition-delegates", id);
        self.inner.delegates_for(id).await
    }
}

struct TaggedSecurity {
    trace: Arc<TaggedTrace>,
    denials_remaining: AtomicUsize,
}

#[async_trait]
impl SecurityGate for TaggedSecurity {
    async fn authorize_tool(&self, call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        if take_one(&self.denials_remaining) {
            self.trace
                .mark("security", format!("deny:{}", call.tool_name));
            Ok(GateDecision::deny("tagged approval denied"))
        } else {
            self.trace
                .mark("security", format!("allow:{}", call.tool_name));
            Ok(GateDecision::Allow)
        }
    }

    async fn screen_input(
        &self,
        text: &str,
        _origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        self.trace.mark("security-screen", text);
        Ok(ScreenOutcome::Pass)
    }
}

struct TaggedResolver {
    trace: Arc<TaggedTrace>,
    first_resolution_barrier: Arc<tokio::sync::Barrier>,
    resolution_count: AtomicUsize,
    model: Arc<dyn ChatModel<()>>,
}

#[async_trait]
impl ModelResolver<()> for TaggedResolver {
    async fn resolve(
        &self,
        request: &crate::host::ModelResolveRequest,
    ) -> crate::error::Result<Arc<dyn ChatModel<()>>> {
        self.trace.mark("model", &request.agent_id);
        // The first provider selection in each root waits for the other one.
        // This makes the two invocations genuinely overlap at the exact point
        // where a harness-level capability registry used to be consulted.
        if self.resolution_count.fetch_add(1, Ordering::SeqCst) == 0 {
            self.first_resolution_barrier.wait().await;
        }
        Ok(self.model.clone())
    }
}

struct TaggedMemory {
    trace: Arc<TaggedTrace>,
}

#[async_trait]
impl AgentMemory for TaggedMemory {
    async fn recall(
        &self,
        request: crate::host::RecallRequest,
    ) -> crate::error::Result<Vec<crate::host::MemoryItem>> {
        self.trace
            .mark("memory", format!("recall:{}", request.query));
        Ok(Vec::new())
    }

    async fn remember(&self, _item: crate::host::NewMemory) -> crate::error::Result<MemoryId> {
        self.trace.mark("memory", "remember");
        Ok(MemoryId::new(self.trace.tag))
    }

    async fn thread_summary(
        &self,
        thread: &crate::ids::ThreadId,
    ) -> crate::error::Result<Option<String>> {
        self.trace.mark("memory", format!("summary:{thread}"));
        Ok(None)
    }
}

struct TaggedBudget {
    trace: Arc<TaggedTrace>,
}

#[async_trait]
impl BudgetGate for TaggedBudget {
    async fn acquire(&self, _estimate: &CallEstimate) -> crate::error::Result<Permit> {
        self.trace.mark("budget", "acquire");
        Ok(Permit::unlimited())
    }

    async fn record(&self, _usage: &Usage) -> crate::error::Result<()> {
        self.trace.mark("budget", "record");
        Ok(())
    }

    fn compression_hint(&self, _state: &ContextState) -> CompressionHint {
        self.trace.mark("budget", "hint");
        CompressionHint::None
    }
}

struct TaggedProgress {
    trace: Arc<TaggedTrace>,
}

#[async_trait]
impl ProgressSink for TaggedProgress {
    async fn emit(&self, event: crate::host::ProgressEvent) {
        self.trace.mark("progress", event.run_id());
    }
}

struct TaggedLearning {
    trace: Arc<TaggedTrace>,
}

#[async_trait]
impl LearningSink for TaggedLearning {
    async fn on_turn_complete(
        &self,
        summary: &crate::host::TurnSummary,
    ) -> crate::error::Result<()> {
        self.trace.mark("learning", &summary.input);
        Ok(())
    }
}

struct TaggedOutcomeClassifier {
    trace: Arc<TaggedTrace>,
}

impl ToolOutcomeClassifier for TaggedOutcomeClassifier {
    fn classify(&self, name: &str, _result: &ToolResult) -> OutcomeClass {
        self.trace.mark("outcome", name);
        OutcomeClass::Success
    }
}

struct TaggedExperience {
    trace: Arc<TaggedTrace>,
}

#[async_trait]
impl ExperienceStore for TaggedExperience {
    async fn record(&self, experience: &crate::host::Experience) -> crate::error::Result<()> {
        self.trace
            .mark("experience", format!("record:{}", experience.task));
        Ok(())
    }

    async fn recall_for(
        &self,
        _agent_id: &str,
        task: &str,
    ) -> crate::error::Result<Vec<crate::host::Experience>> {
        self.trace.mark("experience", format!("recall:{task}"));
        Ok(Vec::new())
    }
}

impl RecordingBudget {
    fn hard() -> Self {
        Self {
            hint: CompressionHint::Hard,
            records: Mutex::new(Vec::new()),
        }
    }

    fn soft() -> Self {
        Self {
            hint: CompressionHint::Soft,
            records: Mutex::new(Vec::new()),
        }
    }

    fn permissive() -> Self {
        Self {
            hint: CompressionHint::None,
            records: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl BudgetGate for RecordingBudget {
    async fn acquire(&self, _estimate: &CallEstimate) -> crate::error::Result<Permit> {
        Ok(Permit::unlimited())
    }

    async fn record(&self, usage: &Usage) -> crate::error::Result<()> {
        self.records.lock().expect("budget lock").push(*usage);
        Ok(())
    }

    fn compression_hint(&self, _state: &ContextState) -> CompressionHint {
        self.hint
    }
}

#[async_trait]
impl BudgetGate for SecretRecordBudget {
    async fn acquire(&self, _estimate: &CallEstimate) -> crate::error::Result<Permit> {
        Ok(Permit::unlimited())
    }

    async fn record(&self, _usage: &Usage) -> crate::error::Result<()> {
        Err(crate::TinyAgentsError::Model(
            "budget-stream-secret".to_string(),
        ))
    }

    fn compression_hint(&self, _state: &ContextState) -> CompressionHint {
        CompressionHint::None
    }
}

#[derive(Default)]
struct PartialThenPendingModel {
    calls: AtomicUsize,
}

#[async_trait]
impl ChatModel<()> for PartialThenPendingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let mut response = ModelResponse::assistant("");
            response
                .message
                .tool_calls
                .push(tinyinference_llm::tool::ToolCall::new(
                    "partial-tool",
                    "noop",
                    json!({}),
                ));
            response.usage = Some(Usage {
                input_tokens: 3,
                output_tokens: 2,
                total_tokens: 5,
                ..Usage::default()
            });
            Ok(response)
        } else {
            std::future::pending().await
        }
    }
}

impl crate::host::ToolOutcomeClassifier for RetryableClassifier {
    fn classify(&self, _name: &str, _result: &ToolResult) -> crate::host::OutcomeClass {
        crate::host::OutcomeClass::RetryableFailure
    }
}

#[async_trait]
impl crate::host::ModelResolver<()> for LeadRecordingResolver {
    async fn resolve(
        &self,
        request: &crate::host::ModelResolveRequest,
    ) -> crate::error::Result<Arc<dyn ChatModel<()>>> {
        self.team_lead_flags
            .lock()
            .expect("resolver lock")
            .push(request.is_team_lead);
        self.model_pins
            .lock()
            .expect("resolver lock")
            .push(request.model_pin.clone());
        Ok(Arc::clone(&self.model))
    }
}

#[async_trait]
impl crate::host::ModelResolver<()> for PendingResolver {
    async fn resolve(
        &self,
        _request: &crate::host::ModelResolveRequest,
    ) -> crate::error::Result<Arc<dyn ChatModel<()>>> {
        self.started.notify_one();
        std::future::pending().await
    }
}

#[async_trait]
impl crate::host::ModelResolver<()> for FirstThenPendingResolver {
    async fn resolve(
        &self,
        _request: &crate::host::ModelResolveRequest,
    ) -> crate::error::Result<Arc<dyn ChatModel<()>>> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok(self.initial.clone())
        } else {
            self.started.notify_one();
            std::future::pending().await
        }
    }
}

#[async_trait]
impl crate::host::ModelResolver<()> for HostFallbackResolver {
    async fn resolve(
        &self,
        request: &crate::host::ModelResolveRequest,
    ) -> crate::error::Result<Arc<dyn ChatModel<()>>> {
        self.requested_pins
            .lock()
            .expect("resolver lock")
            .push(request.model_pin.clone());
        Ok(if request.model_pin.as_deref() == Some("host-backup") {
            self.fallback.clone()
        } else {
            self.primary.clone()
        })
    }
}

#[async_trait]
impl ChatModel<()> for RetryableFailingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Err(tinyinference_llm::Error::Model(
            "transient failure".to_string(),
        ))
    }
}

#[async_trait]
impl ChatModel<()> for SecretProviderModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Err(tinyinference_llm::Error::Model(
            "provider-stream-secret".to_string(),
        ))
    }
}

#[async_trait]
impl ChatModel<()> for UsageReportingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let mut response = ModelResponse::assistant("ok");
        response.usage = Some(Usage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            ..Usage::default()
        });
        Ok(response)
    }
}

#[async_trait]
impl<'state> ChatModel<&'state str> for BorrowedStateModel {
    async fn invoke(
        &self,
        state: &&'state str,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Ok(ModelResponse::assistant(*state))
    }
}

#[derive(Default)]
struct RecordingLearning {
    summaries: Mutex<Vec<crate::host::TurnSummary>>,
}

#[async_trait]
impl crate::host::LearningSink for RecordingLearning {
    async fn on_turn_complete(
        &self,
        summary: &crate::host::TurnSummary,
    ) -> crate::error::Result<()> {
        self.summaries
            .lock()
            .expect("learning lock")
            .push(summary.clone());
        Ok(())
    }
}

#[derive(Default)]
struct RecordingExperience {
    records: Mutex<Vec<crate::host::Experience>>,
}

#[derive(Default)]
struct RecordingMemory {
    items: Mutex<Vec<crate::host::NewMemory>>,
}

#[async_trait]
impl AgentMemory for RecordingMemory {
    async fn recall(
        &self,
        _req: crate::host::RecallRequest,
    ) -> crate::error::Result<Vec<crate::host::MemoryItem>> {
        Ok(Vec::new())
    }

    async fn remember(
        &self,
        item: crate::host::NewMemory,
    ) -> crate::error::Result<crate::host::MemoryId> {
        self.items.lock().expect("memory lock").push(item);
        Ok(crate::host::MemoryId::new("recorded"))
    }

    async fn thread_summary(
        &self,
        _thread: &crate::ids::ThreadId,
    ) -> crate::error::Result<Option<String>> {
        Ok(None)
    }
}

#[async_trait]
impl crate::host::ExperienceStore for RecordingExperience {
    async fn record(&self, exp: &crate::host::Experience) -> crate::error::Result<()> {
        self.records
            .lock()
            .expect("experience lock")
            .push(exp.clone());
        Ok(())
    }

    async fn recall_for(
        &self,
        _agent_id: &str,
        _task: &str,
    ) -> crate::error::Result<Vec<crate::host::Experience>> {
        Ok(Vec::new())
    }
}

async fn yield_until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !predicate() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background terminal finalizer did not complete within one second");
}

#[async_trait]
impl SecurityGate for DenyToolGate {
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        Ok(GateDecision::deny("host denied this tool"))
    }

    async fn screen_input(
        &self,
        _text: &str,
        _origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        Ok(ScreenOutcome::Pass)
    }
}

#[async_trait]
impl SecurityGate for DenyThenAllowGate {
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        if take_one(&self.denials_remaining) {
            Ok(GateDecision::deny("approval declined"))
        } else {
            Ok(GateDecision::Allow)
        }
    }

    async fn screen_input(
        &self,
        _text: &str,
        _origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        Ok(ScreenOutcome::Pass)
    }
}

#[async_trait]
impl SecurityGate for RedactJsonUserGate {
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        Ok(GateDecision::Allow)
    }

    async fn screen_input(
        &self,
        text: &str,
        origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        if origin == crate::host::ContentOrigin::User && text.contains("secret") {
            Ok(ScreenOutcome::Redacted(r#"{"safe":true}"#.to_string()))
        } else {
            Ok(ScreenOutcome::Pass)
        }
    }
}

#[async_trait]
impl SecurityGate for BlockExtensionGate {
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        Ok(GateDecision::Allow)
    }

    async fn screen_input(
        &self,
        text: &str,
        _origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        if text.contains("secret") {
            Ok(ScreenOutcome::Block {
                reason: "blocked extension".to_string(),
            })
        } else {
            Ok(ScreenOutcome::Pass)
        }
    }
}

#[async_trait]
impl SecurityGate for RecordingArgumentGate {
    async fn authorize_tool(&self, call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        self.seen
            .lock()
            .expect("gate lock")
            .push(call.arguments.clone());
        Ok(GateDecision::Allow)
    }

    async fn screen_input(
        &self,
        _text: &str,
        _origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        Ok(ScreenOutcome::Pass)
    }
}

#[async_trait]
impl crate::middleware::Middleware<(), ()> for RedactDeltaMiddleware {
    fn name(&self) -> &str {
        "redact-delta"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut tinyinference_llm::model::ModelDelta,
    ) -> crate::error::Result<()> {
        delta.content = "[redacted]".to_string();
        Ok(())
    }
}

#[async_trait]
impl crate::middleware::Middleware<(), ()> for SecretAfterModelMiddleware {
    fn name(&self) -> &str {
        "secret-after-model"
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> crate::error::Result<()> {
        Err(crate::TinyAgentsError::Middleware(
            "middleware-stream-secret".to_string(),
        ))
    }
}

#[async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "does nothing"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

#[async_trait]
impl Tool for BlockedTool {
    fn name(&self) -> &str {
        "blocked"
    }

    fn description(&self) -> &str {
        "must not be available to this agent"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        panic!("a definition-disallowed tool must never execute")
    }
}

#[async_trait]
impl Tool for AnswerCollisionTool {
    fn name(&self) -> &str {
        "answer"
    }

    fn description(&self) -> &str {
        "a tool intentionally hidden from the hosted definition"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        panic!("a definition-hidden collision tool must never execute")
    }
}

#[async_trait]
impl Tool for InjectedArgumentTool {
    fn name(&self) -> &str {
        "injected"
    }

    fn description(&self) -> &str {
        "records its prepared arguments"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "call_id": {"type": "string"}
            },
            "required": ["text", "call_id"]
        })
    }

    fn injected_arguments(&self) -> Vec<tinytools::ToolInjectedArgument> {
        vec![tinytools::ToolInjectedArgument::tool_call_id("call_id")]
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.executed.lock().expect("tool lock").push(arguments);
        Ok(ToolResult::success("executed"))
    }
}

#[test]
fn new_harness_is_empty_with_default_policy() {
    let harness: AgentHarness<()> = AgentHarness::new();
    assert!(harness.models().default_name().is_none());
    assert_eq!(harness.tools().names().len(), 0);
    assert!(harness.middleware().is_empty());
    assert_eq!(harness.policy(), &RunPolicy::default());
}

#[test]
fn register_first_model_becomes_default() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("a", Arc::new(MockModel::constant("a")))
        .register_model("b", Arc::new(MockModel::constant("b")));
    assert_eq!(harness.models().default_name(), Some("a"));
    harness.set_default_model("b");
    assert_eq!(harness.models().default_name(), Some("b"));
}

#[test]
fn register_tool_and_push_middleware() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.push_middleware(Arc::new(LoggingMiddleware::new()));
    assert_eq!(harness.tools().names(), vec!["noop".to_string()]);
    assert_eq!(harness.middleware().len(), 1);
}

#[test]
fn with_policy_replaces_policy() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let policy = RunPolicy {
        limits: RunLimits::default().with_max_model_calls(3),
        retry: RetryPolicy::default().with_max_attempts(1),
        fallback: Some(FallbackPolicy::new(["a", "b"])),
        default_response_format: None,
        ..RunPolicy::default()
    };
    harness.with_policy(policy.clone());
    assert_eq!(harness.policy(), &policy);
    assert_eq!(harness.policy().limits.max_model_calls, 3);
}

#[test]
fn default_matches_new() {
    let harness: AgentHarness<()> = AgentHarness::default();
    assert!(harness.models().default_name().is_none());
}

#[tokio::test]
async fn host_driven_turn_resolves_and_composes_without_touching_explicit_sdk_defaults() {
    let model = Arc::new(ScriptedModel::replies(vec!["host reply"]));
    let memory = Arc::new(InMemoryAgentMemory::default());
    memory
        .remember(crate::host::NewMemory::new("remembered preference").with_agent("helper"))
        .await
        .expect("seed memory");
    let progress = Arc::new(RecordingProgressSink::new());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::new("host system")),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            AgentDefinition::new("helper", "Helper", "test helper").with_tools(["verbose_lookup"]),
        ])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_memory(memory)
    .with_budget(Arc::new(UnlimitedBudgetGate))
    .with_progress(progress.clone())
    .with_learning(Arc::new(NoopLearningSink))
    .with_tool_outcomes(Arc::new(ErrorFieldClassifier))
    .with_experience(Arc::new(InMemoryExperienceStore::default()));
    let harness: AgentHarness<()> = AgentHarness::new();

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host.clone(),
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("preference")],
                ),
                RunContext::new(RunConfig::new("host-run").with_thread("thread"), ()),
            ),
            &(),
        )
        .await
        .expect("host turn succeeds");

    assert_eq!(run.text().as_deref(), Some("host reply"));
    let request = model.requests().pop().expect("model receives a request");
    assert_eq!(request.messages[0].text(), "host system");
    assert!(
        request
            .messages
            .iter()
            .any(|message| message.text() == "remembered preference")
    );
    yield_until(|| progress.len() == 2).await;
    assert!(
        harness.models().default_name().is_none(),
        "host resolution does not mutate SDK model defaults"
    );
}

#[tokio::test]
async fn initial_host_model_resolution_is_cancelled_while_the_resolver_is_pending() {
    let started = Arc::new(tokio::sync::Notify::new());
    let token = crate::CancellationToken::new();
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(PendingResolver {
            started: started.clone(),
        }),
    );
    let harness: AgentHarness<()> = AgentHarness::new();

    let invocation = harness.invoke_agent(
        AgentInvocation::new(
            host.clone(),
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("go")],
            ),
            RunContext::new(RunConfig::new("host-resolve-cancel"), ())
                .with_cancellation(token.clone()),
        ),
        &(),
    );
    tokio::pin!(invocation);

    let error = tokio::select! {
        _ = started.notified() => {
            token.cancel();
            tokio::time::timeout(Duration::from_secs(1), &mut invocation)
                .await
                .expect("cancellation drops the resolver future")
                .expect_err("cancelled resolution cannot complete")
        }
        result = &mut invocation => panic!("pending resolver unexpectedly finished: {result:?}"),
    };
    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Cancelled);
    assert_eq!(error.timeout_bound, None);
}

#[tokio::test]
async fn policy_only_deadline_bounds_initial_host_resolution_with_a_timeout_error() {
    let started = Arc::new(tokio::sync::Notify::new());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(PendingResolver { started }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_wall_clock_ms(Some(5)),
        ..RunPolicy::default()
    });

    let error = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("policy-host-resolve-timeout"), ()),
            ),
            &(),
        )
        .await
        .expect_err("policy deadline must bound a host resolver without a RunConfig timeout");
    // `HostedError` intentionally sanitizes the message to a fixed string per
    // `kind` (I-6) — the detailed "exceeded its remaining wall-clock budget"
    // text is still available on the run's internal `TinyAgentsError` (see
    // the non-hosted equivalents of this test), just not leaked here.
    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Timeout);
    assert_eq!(
        error.timeout_bound,
        Some(crate::runtime::TimeoutBound::Run),
        "the run's own budget is the run bound"
    );
}

#[tokio::test]
async fn per_model_call_limit_bounds_initial_host_resolution() {
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(PendingResolver {
            started: Arc::new(tokio::sync::Notify::new()),
        }),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_call_ms(Some(5)),
        ..RunPolicy::default()
    });

    let error = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("per-call-host-resolve-timeout"), ()),
            ),
            &(),
        )
        .await
        .expect_err("per-model-call cap must bound host resolution");
    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Timeout);
    assert_eq!(
        error.timeout_bound,
        Some(crate::runtime::TimeoutBound::PerModelCall)
    );
}

#[test]
fn hosted_timeout_round_trip_preserves_the_bound() {
    use crate::error::TinyAgentsError;
    use crate::runtime::{HostedError, HostedErrorKind, TimeoutBound};
    let hosted = |bound| HostedError {
        kind: HostedErrorKind::Timeout,
        message: "timed out".to_string(),
        timeout_bound: bound,
        run: None,
    };
    assert!(matches!(
        TinyAgentsError::from(hosted(Some(TimeoutBound::PerModelCall))),
        TinyAgentsError::CallTimeout(_)
    ));
    assert!(matches!(
        TinyAgentsError::from(hosted(Some(TimeoutBound::Run))),
        TinyAgentsError::Timeout(_)
    ));
    assert!(matches!(
        TinyAgentsError::from(hosted(None)),
        TinyAgentsError::Timeout(_)
    ));

    let hosted_kind = |kind| HostedError {
        kind,
        message: "sanitized failure".to_string(),
        timeout_bound: None,
        run: None,
    };
    assert!(matches!(
        TinyAgentsError::from(hosted_kind(HostedErrorKind::RateLimited)),
        TinyAgentsError::RateLimited(_)
    ));
    assert!(matches!(
        TinyAgentsError::from(hosted_kind(HostedErrorKind::StreamIdleTimeout)),
        TinyAgentsError::StreamIdleTimeout(_)
    ));
}

/// A model whose every call (streaming included) never answers.
struct StalledModel;

#[async_trait]
impl ChatModel<()> for StalledModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        std::future::pending().await
    }
}

/// The public hosted stream sanitizes every terminal failure to one fixed
/// string, so a host draining it cannot tell a wedged call from a provider
/// failure. `invoke_agent_streaming` keeps the typed kind: a per-model-call
/// ceiling that stops a stalled streaming call must come back as
/// [`crate::runtime::HostedErrorKind::Timeout`].
#[tokio::test]
async fn invoke_agent_streaming_preserves_the_timeout_kind_of_a_stalled_model_call() {
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(Arc::new(StalledModel))),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_call_ms(Some(20)),
        // One attempt: the point is the kind of the terminal error.
        retry: RetryPolicy::default().with_max_attempts(1),
        ..RunPolicy::default()
    });

    let error = harness
        .invoke_agent_streaming(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("streaming-call-timeout"), ()),
            ),
            &(),
        )
        .await
        .expect_err("a stalled model call must hit the per-model-call ceiling");
    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Timeout);
    assert_eq!(error.message, "hosted agent invocation timed out");
    assert_eq!(
        error.timeout_bound,
        Some(crate::runtime::TimeoutBound::PerModelCall)
    );
}

async fn assert_rebound_host_resolution_stops(
    token: Option<crate::CancellationToken>,
    policy_timeout: Option<u64>,
) -> crate::error::TinyAgentsError {
    let started = Arc::new(tokio::sync::Notify::new());
    let resolver = Arc::new(FirstThenPendingResolver {
        initial: Arc::new(RetryableFailingModel),
        started: started.clone(),
        calls: AtomicUsize::new(0),
    });
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        resolver,
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.push_model_middleware(Arc::new(ModelFallbackMiddleware::new(["fallback"])));
    if let Some(timeout) = policy_timeout {
        harness.with_policy(RunPolicy {
            limits: RunLimits::default().with_max_wall_clock_ms(Some(timeout)),
            ..RunPolicy::default()
        });
    }

    let mut context = RunContext::new(RunConfig::new("rebind-host-resolution"), ());
    if let Some(token) = token.clone() {
        context = context.with_cancellation(token);
    }
    let invocation = harness.invoke_agent(
        AgentInvocation::new(
            host,
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("go")],
            ),
            context,
        ),
        &(),
    );
    tokio::pin!(invocation);
    tokio::select! {
        biased;
        _ = started.notified() => {
            if let Some(token) = token {
                token.cancel();
            }
            tokio::time::timeout(Duration::from_secs(1), &mut invocation)
                .await
                .expect("rebound resolver must not hang")
                .expect_err("the rebinding resolver remains pending")
                .into()
        }
        result = &mut invocation => panic!("rebind resolver unexpectedly finished: {result:?}"),
    }
}

#[tokio::test]
async fn middleware_rebinding_cancels_a_pending_host_resolver() {
    let error =
        assert_rebound_host_resolution_stops(Some(crate::CancellationToken::new()), None).await;
    assert!(matches!(error, crate::error::TinyAgentsError::Cancelled));
}

#[tokio::test]
async fn middleware_rebinding_applies_the_host_resolution_deadline() {
    // `assert_rebound_host_resolution_stops` round-trips through the hosted
    // entry point (`AgentHarness::invoke_agent`), which now classifies and
    // sanitizes via `HostedError` (I-6) before converting back to
    // `TinyAgentsError` for this helper's declared return type — so the
    // detailed "host model resolution ... remaining wall-clock budget" text
    // is intentionally no longer observable here; only the `Timeout`
    // classification survives the round trip.
    let error = assert_rebound_host_resolution_stops(None, Some(5)).await;
    assert!(matches!(error, crate::error::TinyAgentsError::Timeout(_)));
}

#[tokio::test]
async fn hosted_model_fallback_rebinds_through_the_host_resolver_not_the_local_registry() {
    let host_backup = Arc::new(
        ScriptedModel::replies(vec!["host authority won"]).with_profile(
            tinyinference_llm::model::ModelProfile {
                hoists_system_messages: true,
                ..Default::default()
            },
        ),
    );
    let resolver = Arc::new(HostFallbackResolver {
        primary: Arc::new(RetryableFailingModel),
        fallback: host_backup.clone(),
        requested_pins: Mutex::new(Vec::new()),
    });
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        resolver.clone(),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("host-backup", Arc::new(MockModel::constant("local bypass")));
    harness.push_middleware(Arc::new(FallbackGuidance));
    harness.push_model_middleware(Arc::new(ModelFallbackMiddleware::new(["host-backup"])));

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("hosted-fallback"), ()),
            ),
            &(),
        )
        .await
        .expect("host fallback resolves and succeeds");
    assert_eq!(run.text().as_deref(), Some("host authority won"));
    assert_eq!(host_backup.requests().len(), 1);
    let fallback_request = host_backup.requests().pop().unwrap();
    assert!(
        fallback_request
            .messages
            .iter()
            .any(|message| message.text().contains("fallback guidance"))
    );
    assert!(fallback_request.messages.iter().all(|message| {
        !matches!(message, tinyinference_llm::message::Message::System(_))
            || !message.text().contains("fallback guidance")
    }));
    assert_eq!(
        *resolver.requested_pins.lock().expect("resolver lock"),
        vec![None, Some("host-backup".to_string())],
        "the fallback name is passed back to the host resolver"
    );
}

#[tokio::test]
async fn hosted_turn_screens_and_redacts_json_user_blocks_before_model_submission() {
    let model = Arc::new(ScriptedModel::replies(vec!["ok"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(RedactJsonUserGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let harness: AgentHarness<()> = AgentHarness::new();

    let message =
        tinyinference_llm::message::Message::User(tinyinference_llm::message::UserMessage {
            content: vec![tinyinference_llm::message::ContentBlock::Json(
                json!({"secret": "do not forward"}),
            )],
        });

    harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", vec![message]),
                RunContext::new(RunConfig::new("json-screen"), ()),
            ),
            &(),
        )
        .await
        .expect("hosted turn succeeds");

    let request = model.requests().pop().expect("model request");
    let user = request
        .messages
        .iter()
        .find_map(|message| match message {
            tinyinference_llm::message::Message::User(user) => Some(user),
            _ => None,
        })
        .expect("user message retained");
    assert_eq!(
        user.content,
        vec![tinyinference_llm::message::ContentBlock::Json(
            json!({"safe": true})
        )]
    );
}

#[tokio::test]
async fn hosted_turn_screens_and_redacts_thinking_user_blocks_before_model_submission() {
    let model = Arc::new(ScriptedModel::replies(vec!["ok"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(RedactJsonUserGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let harness: AgentHarness<()> = AgentHarness::new();

    let message =
        tinyinference_llm::message::Message::User(tinyinference_llm::message::UserMessage {
            content: vec![tinyinference_llm::message::ContentBlock::Thinking {
                text: "secret reasoning must not cross the host boundary".to_string(),
                signature: None,
            }],
        });

    harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", vec![message]),
                RunContext::new(RunConfig::new("thinking-screen"), ()),
            ),
            &(),
        )
        .await
        .expect("hosted turn succeeds");

    let request = model.requests().pop().expect("model request");
    let user = request
        .messages
        .iter()
        .find_map(|message| match message {
            tinyinference_llm::message::Message::User(user) => Some(user),
            _ => None,
        })
        .expect("user message retained");
    assert_eq!(
        user.content,
        vec![tinyinference_llm::message::ContentBlock::Thinking {
            text: r#"{"safe":true}"#.to_string(),
            signature: None,
        }]
    );
}

#[tokio::test]
async fn hosted_turn_screens_and_redacts_provider_extension_user_blocks() {
    let model = Arc::new(ScriptedModel::replies(vec!["ok"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(RedactJsonUserGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let harness: AgentHarness<()> = AgentHarness::new();

    let message =
        tinyinference_llm::message::Message::User(tinyinference_llm::message::UserMessage {
            content: vec![tinyinference_llm::message::ContentBlock::ProviderExtension(
                json!({"secret": "do not forward"}),
            )],
        });

    harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", vec![message]),
                RunContext::new(RunConfig::new("extension-redaction"), ()),
            ),
            &(),
        )
        .await
        .expect("redacted extension is safe to forward");

    let request = model.requests().pop().expect("model request");
    let user = request
        .messages
        .iter()
        .find_map(|message| match message {
            tinyinference_llm::message::Message::User(user) => Some(user),
            _ => None,
        })
        .expect("user message retained");
    assert_eq!(
        user.content,
        vec![tinyinference_llm::message::ContentBlock::ProviderExtension(
            json!({"safe": true})
        )]
    );
}

#[tokio::test]
async fn hosted_turn_blocks_provider_extension_user_blocks_before_model_submission() {
    let model = Arc::new(ScriptedModel::replies(vec!["must not run"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(BlockExtensionGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let harness: AgentHarness<()> = AgentHarness::new();

    let message =
        tinyinference_llm::message::Message::User(tinyinference_llm::message::UserMessage {
            content: vec![tinyinference_llm::message::ContentBlock::ProviderExtension(
                json!({"secret": "block me"}),
            )],
        });

    let error = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", vec![message]),
                RunContext::new(RunConfig::new("extension-block"), ()),
            ),
            &(),
        )
        .await
        .expect_err("blocked extensions must not reach the provider");
    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Policy);
    assert_eq!(
        error.to_string(),
        "hosted agent invocation was rejected by policy"
    );
    assert!(!error.to_string().contains("secret"));
    assert!(model.requests().is_empty());
}

#[tokio::test]
async fn hosted_turn_screens_only_the_new_user_input_not_replayed_history() {
    // A text-dialect host replays prior tool results as user rows. They were
    // screened when admitted; re-screening them every turn bricks the thread
    // the moment one scores over the threshold (openhuman#6710).
    let model = Arc::new(ScriptedModel::replies(vec!["answer"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(BlockExtensionGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let harness: AgentHarness<()> = AgentHarness::new();
    // Also covers a turn that failed after a tool round: the host persists the
    // accepted request, so history ends on the results row and the new input
    // follows it directly, with no assistant row between them.
    let messages = vec![
        tinyinference_llm::message::Message::user("[Tool results]\nsecret-looking payload"),
        tinyinference_llm::message::Message::assistant("summary of the results"),
        tinyinference_llm::message::Message::user("look again"),
        tinyinference_llm::message::Message::assistant("<tool_call>…</tool_call>"),
        tinyinference_llm::message::Message::user("[Tool results]\nmore secret-looking payload"),
        tinyinference_llm::message::Message::user("?"),
    ];

    harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", messages),
                RunContext::new(RunConfig::new("history-not-rescreened"), ()),
            )
            .with_replayed_prefix(5),
            &(),
        )
        .await
        .expect("replayed history must not be re-screened as new user input");
    assert_eq!(model.requests().len(), 1);
}

#[tokio::test]
async fn hosted_turn_still_blocks_new_user_input_and_screens_all_by_default() {
    // The first turn of a new thread, and a new message after replayed
    // history: the turn's own input is still screened as user input.
    for (messages, replayed_prefix) in [
        (vec![tinyinference_llm::message::Message::user("secret")], 0),
        (
            vec![
                tinyinference_llm::message::Message::user("hello"),
                tinyinference_llm::message::Message::assistant("hi"),
                tinyinference_llm::message::Message::user("secret"),
            ],
            2,
        ),
        // Without a boundary every user message is screened, as before: a
        // host that never opts in loses no screening (openhuman#6710).
        (
            vec![
                tinyinference_llm::message::Message::user("secret"),
                tinyinference_llm::message::Message::assistant("hi"),
                tinyinference_llm::message::Message::user("hello"),
            ],
            0,
        ),
    ] {
        let model = Arc::new(ScriptedModel::replies(vec!["must not run"]));
        let host = crate::host::HostCapabilities::new(
            Arc::new(StaticContextComposer::empty()),
            Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
                "helper",
                "Helper",
                "test helper",
            )])),
            Arc::new(BlockExtensionGate),
            Arc::new(FixedModelResolver::new(model.clone())),
        );
        let error = AgentHarness::<()>::new()
            .invoke_agent(
                AgentInvocation::new(
                    host,
                    AgentTurnRequest::new("helper", messages),
                    RunContext::new(RunConfig::new("new-input-block"), ()),
                )
                .with_replayed_prefix(replayed_prefix),
                &(),
            )
            .await
            .expect_err("the new user input must still be screened");
        assert_eq!(error.kind, crate::runtime::HostedErrorKind::Policy);
        assert!(model.requests().is_empty());
    }
}

fn replay_host(
    gate: Arc<dyn SecurityGate>,
    model: Arc<ScriptedModel>,
) -> crate::host::HostCapabilities<()> {
    crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        gate,
        Arc::new(FixedModelResolver::new(model)),
    )
}

#[tokio::test]
async fn hosted_turn_rejects_a_replayed_prefix_past_the_messages() {
    // A prefix past the end is always a host bug; clamping it would silently
    // skip screening of the whole request.
    let model = Arc::new(ScriptedModel::replies(vec!["must not run"]));
    let error = AgentHarness::<()>::new()
        .invoke_agent(
            AgentInvocation::new(
                replay_host(Arc::new(BlockExtensionGate), model.clone()),
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("secret")],
                ),
                RunContext::new(RunConfig::new("prefix-past-end"), ()),
            )
            .with_replayed_prefix(2),
            &(),
        )
        .await
        .expect_err("an out-of-range replayed prefix is rejected");
    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Policy);
    assert!(model.requests().is_empty());
}

#[tokio::test]
async fn hosted_turn_accepts_a_replayed_prefix_covering_every_message() {
    // Equal to the length is legitimate (a deferred resume has no new input);
    // it is logged when the last message is a user message, not rejected.
    let model = Arc::new(ScriptedModel::replies(vec!["answer"]));
    AgentHarness::<()>::new()
        .invoke_agent(
            AgentInvocation::new(
                replay_host(Arc::new(BlockExtensionGate), model.clone()),
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("hello")],
                ),
                RunContext::new(RunConfig::new("prefix-at-end"), ()),
            )
            .with_replayed_prefix(1),
            &(),
        )
        .await
        .expect("a prefix equal to the message count is accepted");
    assert_eq!(model.requests().len(), 1);
}

#[tokio::test]
async fn hosted_turn_replays_the_redacted_form_of_an_admitted_user_row() {
    // The replayed_prefix contract: a host replays rows as the gate returned
    // them. The run's transcript carries the redacted form, so a host that
    // persists it never sends the raw secret on a later turn.
    let model = Arc::new(ScriptedModel::replies(vec!["first", "second"]));
    let harness = AgentHarness::<()>::new();
    let first = harness
        .invoke_agent(
            AgentInvocation::new(
                replay_host(Arc::new(RedactJsonUserGate), model.clone()),
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("my secret")],
                ),
                RunContext::new(RunConfig::new("redact-first"), ()),
            ),
            &(),
        )
        .await
        .expect("first turn succeeds");

    let mut replay = first.messages.clone();
    let replayed = replay.len();
    replay.push(tinyinference_llm::message::Message::user("and now?"));
    harness
        .invoke_agent(
            AgentInvocation::new(
                replay_host(Arc::new(RedactJsonUserGate), model.clone()),
                AgentTurnRequest::new("helper", replay),
                RunContext::new(RunConfig::new("redact-replay"), ()),
            )
            .with_replayed_prefix(replayed),
            &(),
        )
        .await
        .expect("replayed turn succeeds");

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    let second = format!("{:?}", requests[1].messages);
    assert!(
        second.contains("and now?"),
        "the new input reached the model"
    );
    assert!(
        !second.contains("secret"),
        "the replayed row must stay redacted: {second}"
    );
}

#[tokio::test]
async fn hosted_model_resolution_marks_only_root_contexts_as_team_leads() {
    let model = Arc::new(ScriptedModel::replies(vec!["root", "child"]));
    let resolver = Arc::new(LeadRecordingResolver {
        model: model.clone(),
        team_lead_flags: Mutex::new(Vec::new()),
        model_pins: Mutex::new(Vec::new()),
    });
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        resolver.clone(),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.push_middleware(Arc::new(LoggingMiddleware::new()));
    harness.push_middleware(Arc::new(LoggingMiddleware::with_label("second")));

    harness
        .invoke_agent(
            AgentInvocation::new(
                host.clone(),
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("root")],
                ),
                RunContext::new(RunConfig::new("root").with_max_depth(2), ()),
            ),
            &(),
        )
        .await
        .expect("root turn succeeds");

    let parent: RunContext<()> = RunContext::new(RunConfig::new("parent").with_max_depth(2), ());
    let child = parent
        .child(RunConfig::new("child"), ())
        .expect("child context is valid");
    harness
        .invoke_agent(
            AgentInvocation::new(
                host.clone(),
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("child")],
                ),
                child,
            ),
            &(),
        )
        .await
        .expect("child turn succeeds");

    assert_eq!(
        *resolver.team_lead_flags.lock().expect("resolver lock"),
        vec![true, false],
        "each hosted model call resolves once despite multiple middleware hooks"
    );
}

/// I-6 regression: a hosted invocation that exhausts a configured run limit
/// must classify as `HostedErrorKind::LimitExceeded`, distinguishable from
/// other hosted failure modes (here, a policy rejection) rather than every
/// non-cancel/timeout failure collapsing into one generic
/// `Model("hosted agent invocation failed")`.
#[tokio::test]
async fn hosted_limit_exceeded_is_distinguishable_from_other_hosted_errors() {
    // A model that always requests the same tool call, so the run never
    // finishes on its own and must hit `max_model_calls`.
    let looping_model = Arc::new(ScriptedModel::new(
        std::iter::repeat_with(|| {
            let mut response = ModelResponse::assistant("");
            response
                .message
                .tool_calls
                .push(tinyinference_llm::tool::ToolCall::new(
                    "call",
                    "noop",
                    json!({}),
                ));
            response
        })
        .take(8)
        .collect(),
    ));
    let definition = AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]);
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![definition])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(looping_model)),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));

    let limit_error = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("limit-exceeded").with_max_model_calls(1), ()),
            ),
            &(),
        )
        .await
        .expect_err("the model-call cap must eventually fail the run");
    assert_eq!(
        limit_error.kind,
        crate::runtime::HostedErrorKind::LimitExceeded
    );
    // The run accumulated before failing is still available.
    assert!(limit_error.run.is_some());

    // A different hosted failure mode (a security-gate denial of the user's
    // input) classifies differently, proving `kind` genuinely discriminates
    // rather than every non-cancel/timeout error collapsing together.
    let denied_definition =
        AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]);
    let denied_host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![denied_definition])),
        Arc::new(BlockExtensionGate),
        Arc::new(FixedModelResolver::new(Arc::new(ScriptedModel::replies(
            vec!["unused"],
        )))),
    );
    let mut denied_harness: AgentHarness<()> = AgentHarness::new();
    denied_harness.register_tool(Arc::new(NoopTool));
    let policy_error = denied_harness
        .invoke_agent(
            AgentInvocation::new(
                denied_host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::User(
                        tinyinference_llm::message::UserMessage {
                            content: vec![
                                tinyinference_llm::message::ContentBlock::ProviderExtension(
                                    json!({"secret": "block me"}),
                                ),
                            ],
                        },
                    )],
                ),
                RunContext::new(RunConfig::new("policy-denied"), ()),
            ),
            &(),
        )
        .await
        .expect_err("the security gate must deny this input");
    assert_eq!(policy_error.kind, crate::runtime::HostedErrorKind::Policy);

    assert_ne!(
        limit_error.kind, policy_error.kind,
        "distinct hosted failure modes must classify to distinct kinds"
    );
}

#[tokio::test]
async fn hosted_definition_tool_allowlist_filters_schemas_and_rejects_fabricated_calls() {
    let mut blocked_call = ModelResponse::assistant("");
    blocked_call
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "blocked-call",
            "blocked",
            json!({}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        blocked_call,
        ModelResponse::assistant("recovered"),
    ]));
    let definition = AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]);
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![definition])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.register_tool(Arc::new(BlockedTool));

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("allowlist"), ()),
            ),
            &(),
        )
        .await
        .expect("the model recovers after its denied call");

    assert_eq!(run.text().as_deref(), Some("recovered"));
    assert!(
        run.messages
            .iter()
            .any(|message| message.text().contains("unknown tool `blocked`"))
    );
    let requests = model.requests();
    assert_eq!(
        requests[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["noop"]
    );
}

/// I-9 regression: a definition that declares **no** tools (an empty list —
/// `AgentDefinition::new` without `with_tools`) must deny every registered
/// tool, not grant the whole catalogue. Before the fix, `HashSet::is_empty()`
/// was read as "unrestricted" instead of "nothing authorized", so a
/// definition whose author simply forgot to declare tools (or a host that
/// failed to populate the field) silently ran with every tool available.
#[tokio::test]
async fn hosted_definition_with_no_declared_tools_denies_every_tool() {
    let mut fabricated_call = ModelResponse::assistant("");
    fabricated_call
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "call-1",
            "noop",
            json!({}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        fabricated_call,
        ModelResponse::assistant("recovered"),
    ]));
    // No `.with_tools(...)`: the definition declares nothing.
    let definition = AgentDefinition::new("helper", "Helper", "test helper");
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![definition])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("empty-allowlist"), ()),
            ),
            &(),
        )
        .await
        .expect("the model recovers after its denied call");

    assert_eq!(run.text().as_deref(), Some("recovered"));
    assert!(
        run.messages
            .iter()
            .any(|message| message.text().contains("unknown tool `noop`")),
        "a registered tool the definition never declared must be rejected, not silently run"
    );
    // No tool schema at all is offered to the provider — the registered
    // catalogue is not leaked to a definition that declared nothing.
    assert!(model.requests()[0].tools.is_empty());
}

#[tokio::test]
async fn hosted_structured_schema_rejects_hidden_registered_tool_collision() {
    // `answer` is registered globally but deliberately not allowed for this
    // hosted definition. It still cannot be used as a synthetic structured
    // tool name: a provider response would make the schema/tool attribution
    // ambiguous before the allowlist reaches dispatch.
    let model = Arc::new(ScriptedModel::replies(vec![r#"{"ok":true}"#]));
    let definition = AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]);
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![definition])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.register_tool(Arc::new(AnswerCollisionTool));
    harness.with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::json_schema(
            "answer",
            json!({"type": "object"}),
        )),
        ..RunPolicy::default()
    });

    let error = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("allowlisted-schema-collision"), ()),
            ),
            &(),
        )
        .await
        .expect_err("a hidden registered tool still collides with the schema");

    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Policy);
    assert_eq!(
        error.to_string(),
        "hosted agent invocation was rejected by policy"
    );
    assert!(model.requests().is_empty(), "provider was not contacted");
}

#[tokio::test]
async fn host_security_denial_returns_a_tool_message_without_executing_the_tool() {
    let mut tool_response = tinyinference_llm::model::ModelResponse::assistant("");
    tool_response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "call-1",
            "noop",
            json!({}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response,
        tinyinference_llm::model::ModelResponse::assistant("recovered"),
    ]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]),
        ])),
        Arc::new(DenyToolGate),
        Arc::new(FixedModelResolver::new(model)),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("denied"), ()),
            ),
            &(),
        )
        .await
        .expect("the model receives the denial and can finish");

    assert_eq!(run.text().as_deref(), Some("recovered"));
    assert!(
        run.messages
            .iter()
            .any(|message| message.text().contains("host denied this tool"))
    );
}

#[tokio::test]
async fn security_gate_sees_raw_provider_arguments_while_tools_receive_prepared_values() {
    let mut tool_response = ModelResponse::assistant("");
    tool_response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "real-call",
            "injected",
            json!({"text": "safe", "call_id": "forged-by-provider"}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response,
        ModelResponse::assistant("done"),
    ]));
    let gate = Arc::new(RecordingArgumentGate {
        seen: Mutex::new(Vec::new()),
    });
    let executed = Arc::new(Mutex::new(Vec::new()));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            AgentDefinition::new("helper", "Helper", "test helper").with_tools(["injected"]),
        ])),
        gate.clone(),
        Arc::new(FixedModelResolver::new(model)),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(InjectedArgumentTool {
        executed: executed.clone(),
    }));

    harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("raw-provider-args"), ()),
            ),
            &(),
        )
        .await
        .expect("the allowed tool completes");

    assert_eq!(
        *gate.seen.lock().expect("gate lock"),
        vec![json!({"text": "safe", "call_id": "forged-by-provider"})],
        "the host authorizes the unmodified provider payload"
    );
    assert_eq!(
        *executed.lock().expect("tool lock"),
        vec![json!({"text": "safe", "call_id": "real-call"})],
        "only the trusted call id reaches execution"
    );
}

/// A deferred tool that echoes its argument, used to exercise a by-name
/// deferred call under host authorization.
struct DeferredEchoTool;

#[async_trait]
impl Tool for DeferredEchoTool {
    fn name(&self) -> &str {
        "quote"
    }

    fn description(&self) -> &str {
        "echoes its argument"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        })
    }

    fn exposure(&self) -> tinytools::ToolExposure {
        tinytools::ToolExposure::Deferred
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success(arguments.to_string()))
    }
}

/// A deferred tool is called by its own name, with no bridge wrapper, so the
/// host's `SecurityGate` must see exactly the arguments the model sent and
/// validation and execution use.
#[tokio::test]
async fn security_gate_sees_the_arguments_of_a_deferred_call_by_its_own_name() {
    let mut tool_response = ModelResponse::assistant("");
    tool_response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "real-call",
            "quote",
            json!({"symbol": "ACME"}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response,
        ModelResponse::assistant("done"),
    ]));
    let gate = Arc::new(RecordingArgumentGate {
        seen: Mutex::new(Vec::new()),
    });
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            AgentDefinition::new("helper", "Helper", "test helper").with_tools(["quote"]),
        ])),
        gate.clone(),
        Arc::new(FixedModelResolver::new(model)),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(DeferredEchoTool));

    harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("deferred-by-name-args"), ()),
            ),
            &(),
        )
        .await
        .expect("the deferred tool call completes");

    assert_eq!(
        *gate.seen.lock().expect("gate lock"),
        vec![json!({"symbol": "ACME"})],
        "the host must authorize the real arguments of the deferred call"
    );
}

#[tokio::test]
async fn denied_tool_calls_release_their_reserved_limit_for_a_later_approval() {
    fn call(id: &str) -> ModelResponse {
        let mut response = ModelResponse::assistant("");
        response
            .message
            .tool_calls
            .push(tinyinference_llm::tool::ToolCall::new(
                id,
                "noop",
                json!({}),
            ));
        response
    }

    let model = Arc::new(ScriptedModel::new(vec![
        call("denied-one"),
        call("denied-two"),
        call("allowed"),
        ModelResponse::assistant("completed after approval"),
    ]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]),
        ])),
        Arc::new(DenyThenAllowGate {
            denials_remaining: AtomicUsize::new(2),
        }),
        Arc::new(FixedModelResolver::new(model)),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(
                    RunConfig::new("denial-limit-release").with_max_tool_calls(1),
                    (),
                ),
            ),
            &(),
        )
        .await
        .expect("two denials do not spend the only executable tool slot");

    assert_eq!(run.text().as_deref(), Some("completed after approval"));
    assert_eq!(run.executed_tools, vec!["noop"]);
}

#[tokio::test]
async fn hosted_streams_finalize_success_failure_and_drop_with_terminal_host_records() {
    async fn run_terminal_case(
        model: Arc<ScriptedModel>,
        run_id: &str,
        drain: bool,
    ) -> (
        Arc<RecordingLearning>,
        Arc<RecordingExperience>,
        Arc<RecordingProgressSink>,
    ) {
        let learning = Arc::new(RecordingLearning::default());
        let experience = Arc::new(RecordingExperience::default());
        let progress = Arc::new(RecordingProgressSink::new());
        let host = crate::host::HostCapabilities::new(
            Arc::new(StaticContextComposer::empty()),
            Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
                "helper",
                "Helper",
                "test helper",
            )])),
            Arc::new(AllowAllSecurityGate),
            Arc::new(FixedModelResolver::new(model)),
        )
        .with_progress(progress.clone())
        .with_learning(learning.clone())
        .with_experience(experience.clone());
        let harness: AgentHarness<()> = AgentHarness::new();

        let context = RunContext::new(RunConfig::new(run_id), ());
        let mut stream = harness
            .invoke_agent_stream(
                AgentInvocation::new(
                    host,
                    AgentTurnRequest::new(
                        "helper",
                        vec![tinyinference_llm::message::Message::user("go")],
                    ),
                    context,
                ),
                &(),
            )
            .await
            .expect("stream starts");
        if drain {
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if !matches!(item, crate::agent_loop::AgentStreamItem::Event(_)) {
                    terminal = Some(item);
                    break;
                }
            }
            assert!(terminal.is_some(), "stream reaches terminal item");
        }
        drop(stream);
        yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
        yield_until(|| experience.records.lock().expect("experience lock").len() == 1).await;
        yield_until(|| {
            progress
                .events()
                .iter()
                .any(crate::host::ProgressEvent::is_terminal)
        })
        .await;
        (learning, experience, progress)
    }

    let (learning, experience, progress) = run_terminal_case(
        Arc::new(ScriptedModel::replies(vec!["ok"])),
        "stream-success",
        true,
    )
    .await;
    assert!(experience.records.lock().expect("experience lock")[0].success);
    assert_eq!(learning.summaries.lock().expect("learning lock").len(), 1);
    assert!(matches!(
        progress.events().last(),
        Some(crate::host::ProgressEvent::Finished { .. })
    ));

    let (_learning, experience, progress) =
        run_terminal_case(Arc::new(ScriptedModel::new(vec![])), "stream-error", true).await;
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    assert!(
        progress
            .events()
            .iter()
            .any(|event| matches!(event, crate::host::ProgressEvent::Error { .. }))
    );
    assert_eq!(
        progress
            .events()
            .iter()
            .filter(|event| event.is_terminal())
            .count(),
        1,
        "a failed turn has one terminal progress event"
    );

    let (_learning, experience, progress) = run_terminal_case(
        Arc::new(ScriptedModel::replies(vec!["unused"])),
        "stream-cancel",
        false,
    )
    .await;
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    assert!(
        progress
            .events()
            .iter()
            .any(|event| matches!(event, crate::host::ProgressEvent::Error { .. }))
    );
    assert_eq!(
        progress
            .events()
            .iter()
            .filter(|event| event.is_terminal())
            .count(),
        1,
        "a cancelled turn has one terminal progress event"
    );
}

#[tokio::test]
async fn dropped_host_invocations_finalize_the_actual_partial_run_once() {
    async fn host_with_partial_model(
        model: Arc<PartialThenPendingModel>,
    ) -> (
        Arc<AgentHarness<()>>,
        crate::host::HostCapabilities<()>,
        Arc<RecordingLearning>,
        Arc<RecordingExperience>,
        Arc<RecordingMemory>,
        Arc<RecordingProgressSink>,
    ) {
        let learning = Arc::new(RecordingLearning::default());
        let experience = Arc::new(RecordingExperience::default());
        let memory = Arc::new(RecordingMemory::default());
        let progress = Arc::new(RecordingProgressSink::new());
        let host = crate::host::HostCapabilities::new(
            Arc::new(StaticContextComposer::empty()),
            Arc::new(InMemoryDefinitionRegistry::new(vec![
                AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]),
            ])),
            Arc::new(AllowAllSecurityGate),
            Arc::new(FixedModelResolver::new(model)),
        )
        .with_learning(learning.clone())
        .with_experience(experience.clone())
        .with_memory(memory.clone())
        .with_progress(progress.clone());
        let mut harness = AgentHarness::new();
        harness.register_tool(Arc::new(NoopTool));

        (
            Arc::new(harness),
            host,
            learning,
            experience,
            memory,
            progress,
        )
    }

    async fn wait_for_second_call(model: &PartialThenPendingModel) {
        yield_until(|| model.calls.load(Ordering::SeqCst) >= 2).await;
    }

    let model = Arc::new(PartialThenPendingModel::default());
    let (harness, host, learning, experience, memory, progress) =
        host_with_partial_model(model.clone()).await;
    let task_harness = harness.clone();
    let task = tokio::spawn(async move {
        task_harness
            .invoke_agent(
                AgentInvocation::new(
                    host,
                    AgentTurnRequest::new(
                        "helper",
                        vec![tinyinference_llm::message::Message::user("partial")],
                    ),
                    RunContext::new(RunConfig::new("unary-drop"), ()),
                ),
                &(),
            )
            .await
    });
    wait_for_second_call(&model).await;
    task.abort();
    let _ = task.await;
    yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0]
            .usage
            .total_tokens,
        5
    );
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0].tools_invoked,
        ["noop"]
    );
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    assert_eq!(memory.items.lock().expect("memory lock").len(), 1);
    let terminals: Vec<_> = progress
        .events()
        .into_iter()
        .filter(|event| event.is_terminal())
        .collect();
    assert!(matches!(
        terminals.as_slice(),
        [crate::host::ProgressEvent::Error { .. }]
    ));

    let model = Arc::new(PartialThenPendingModel::default());
    let (harness, host, learning, experience, _memory, progress) =
        host_with_partial_model(model.clone()).await;
    let mut stream = harness
        .invoke_agent_stream(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("partial")],
                ),
                RunContext::new(RunConfig::new("stream-drop"), ()),
            ),
            &(),
        )
        .await
        .expect("stream starts");
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while model.calls.load(Ordering::SeqCst) < 2 {
            assert!(
                stream.next().await.is_some(),
                "stream ended before second model call"
            );
        }
    })
    .await
    .expect("stream did not reach its second model call within one second");
    drop(stream);
    yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0]
            .usage
            .total_tokens,
        5
    );
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0].tools_invoked,
        ["noop"]
    );
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    let terminals: Vec<_> = progress
        .events()
        .into_iter()
        .filter(|event| event.is_terminal())
        .collect();
    assert!(matches!(
        terminals.as_slice(),
        [crate::host::ProgressEvent::Error { .. }]
    ));
}

#[tokio::test]
async fn denied_tool_calls_do_not_enter_terminal_executed_tool_summary() {
    let mut tool_response = tinyinference_llm::model::ModelResponse::assistant("");
    tool_response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "call-1",
            "noop",
            json!({}),
        ));
    let learning = Arc::new(RecordingLearning::default());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]),
        ])),
        Arc::new(DenyToolGate),
        Arc::new(FixedModelResolver::new(Arc::new(ScriptedModel::new(vec![
            tool_response,
            tinyinference_llm::model::ModelResponse::assistant("recovered"),
        ])))),
    )
    .with_learning(learning.clone());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));

    harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("denied-summary"), ()),
            ),
            &(),
        )
        .await
        .expect("denial is recoverable");
    yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
    let summaries = learning.summaries.lock().expect("learning lock");
    assert!(
        summaries[0].tools_invoked.is_empty(),
        "denied calls never reached a tool executor"
    );
}

#[tokio::test]
async fn retryable_classifier_changes_the_model_visible_result_without_redispatching() {
    let mut tool_response = tinyinference_llm::model::ModelResponse::assistant("");
    tool_response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "call-1",
            "noop",
            json!({}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response,
        tinyinference_llm::model::ModelResponse::assistant("model chose to continue"),
    ]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_tool_outcomes(Arc::new(RetryableClassifier));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("retryable-result"), ()),
            ),
            &(),
        )
        .await
        .expect("retryable result is recoverable");
    assert_eq!(
        run.tool_calls, 1,
        "the runtime never silently repeats an action"
    );
    assert_eq!(
        model.requests().len(),
        2,
        "the model, not the runtime, selected the next step"
    );
    assert!(
        run.messages
            .iter()
            .any(|message| message.text().contains("retryable tool failure"))
    );
}

#[tokio::test]
async fn explicit_model_paths_accept_borrowed_state() {
    async fn exercise<'state>(state: &'state str) {
        let mut harness: AgentHarness<&'state str> = AgentHarness::new();
        harness.register_model("borrowed", Arc::new(BorrowedStateModel));

        let unary = harness
            .invoke(
                &state,
                (),
                RunConfig::new("borrowed-unary"),
                vec![tinyinference_llm::message::Message::user("hello")],
            )
            .await
            .expect("explicit unary invocation accepts borrowed state");
        assert_eq!(unary.text().as_deref(), Some(state));

        let streaming = harness
            .invoke_streaming(
                &state,
                (),
                RunConfig::new("borrowed-streaming"),
                vec![tinyinference_llm::message::Message::user("hello")],
            )
            .await
            .expect("explicit streaming invocation accepts borrowed state");
        assert_eq!(streaming.text().as_deref(), Some(state));
    }

    let owned = String::from("borrowed state remains valid");
    exercise(owned.as_str()).await;
}

#[tokio::test]
async fn concurrent_roots_keep_every_invocation_capability_bundle_isolated() {
    fn tool_round(answer: &str) -> Vec<ModelResponse> {
        let mut tool_call = ModelResponse::assistant("");
        tool_call
            .message
            .tool_calls
            .push(tinyinference_llm::tool::ToolCall::new(
                "call",
                "noop",
                json!({}),
            ));
        vec![tool_call, ModelResponse::assistant(answer)]
    }

    fn bundle(
        trace: Arc<TaggedTrace>,
        barrier: Arc<tokio::sync::Barrier>,
        model: Arc<dyn ChatModel<()>>,
        denials: usize,
    ) -> crate::host::HostCapabilities<()> {
        crate::host::HostCapabilities::new(
            Arc::new(TaggedContext {
                trace: trace.clone(),
            }),
            Arc::new(TaggedDefinitions {
                trace: trace.clone(),
                inner: InMemoryDefinitionRegistry::new(vec![
                    AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]),
                ]),
            }),
            Arc::new(TaggedSecurity {
                trace: trace.clone(),
                denials_remaining: AtomicUsize::new(denials),
            }),
            Arc::new(TaggedResolver {
                trace: trace.clone(),
                first_resolution_barrier: barrier,
                resolution_count: AtomicUsize::new(0),
                model,
            }),
        )
        .with_memory(Arc::new(TaggedMemory {
            trace: trace.clone(),
        }))
        .with_budget(Arc::new(TaggedBudget {
            trace: trace.clone(),
        }))
        .with_progress(Arc::new(TaggedProgress {
            trace: trace.clone(),
        }))
        .with_learning(Arc::new(TaggedLearning {
            trace: trace.clone(),
        }))
        .with_tool_outcomes(Arc::new(TaggedOutcomeClassifier {
            trace: trace.clone(),
        }))
        .with_experience(Arc::new(TaggedExperience { trace }))
    }

    let alpha_trace = Arc::new(TaggedTrace::new("alpha"));
    let bravo_trace = Arc::new(TaggedTrace::new("bravo"));
    let first_resolution_barrier = Arc::new(tokio::sync::Barrier::new(2));
    let alpha_host = bundle(
        alpha_trace.clone(),
        first_resolution_barrier.clone(),
        Arc::new(ScriptedModel::new(tool_round("alpha answer"))),
        0,
    );
    // Bravo first denies a tool, then permits the model's second attempt.  The
    // differing security outcome proves that policy is invocation-local while
    // still exercising the outcome classifier in both roots.
    let mut bravo_round = tool_round("bravo answer");
    bravo_round.insert(1, bravo_round[0].clone());
    let bravo_host = bundle(
        bravo_trace.clone(),
        first_resolution_barrier,
        Arc::new(ScriptedModel::new(bravo_round)),
        1,
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));

    let first = harness.invoke_agent(
        AgentInvocation::new(
            alpha_host,
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("alpha input")],
            ),
            RunContext::new(RunConfig::new("root-alpha"), ()),
        ),
        &(),
    );
    let second = harness.invoke_agent(
        AgentInvocation::new(
            bravo_host,
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("bravo input")],
            ),
            RunContext::new(RunConfig::new("root-bravo"), ()),
        ),
        &(),
    );
    let (first, second) = tokio::join!(first, second);
    let first = first.expect("alpha invocation succeeds");
    let second = second.expect("bravo invocation recovers from its one denial");
    assert_eq!(first.executed_tools, ["noop"]);
    assert_eq!(second.executed_tools, ["noop"]);
    assert_eq!(first.text().as_deref(), Some("alpha answer"));
    assert_eq!(second.text().as_deref(), Some("bravo answer"));

    // The terminal observer is asynchronous.  Wait until both roots have
    // touched each of the ten concrete host adapters before inspecting their
    // tags, rather than relying on scheduling luck or a sleep.
    const ALL_CAPABILITIES: [&str; 10] = [
        "context",
        "definition",
        "security",
        "model",
        "memory",
        "budget",
        "progress",
        "learning",
        "outcome",
        "experience",
    ];
    yield_until(|| {
        ALL_CAPABILITIES
            .iter()
            .all(|capability| alpha_trace.has_capability(capability))
            && ALL_CAPABILITIES
                .iter()
                .all(|capability| bravo_trace.has_capability(capability))
    })
    .await;

    let alpha_events = alpha_trace.events();
    let bravo_events = bravo_trace.events();
    assert!(
        alpha_events
            .iter()
            .any(|event| event == "context:alpha input")
            && alpha_events
                .iter()
                .any(|event| event == "progress:root-alpha")
            && alpha_events
                .iter()
                .any(|event| event == "learning:alpha input")
            && alpha_events
                .iter()
                .any(|event| event == "security:allow:noop"),
        "alpha only used its own context, progress, learning, and security adapters: {alpha_events:?}"
    );
    assert!(
        bravo_events
            .iter()
            .any(|event| event == "context:bravo input")
            && bravo_events
                .iter()
                .any(|event| event == "progress:root-bravo")
            && bravo_events
                .iter()
                .any(|event| event == "learning:bravo input")
            && bravo_events
                .iter()
                .any(|event| event == "security:deny:noop")
            && bravo_events
                .iter()
                .any(|event| event == "security:allow:noop"),
        "bravo retained its own deny-then-allow policy: {bravo_events:?}"
    );
    assert!(
        alpha_events.iter().all(|event| !event.contains("bravo")),
        "alpha must never receive data from bravo's invocation bundle: {alpha_events:?}"
    );
    assert!(
        bravo_events.iter().all(|event| !event.contains("alpha")),
        "bravo must never receive data from alpha's invocation bundle: {bravo_events:?}"
    );
}

#[tokio::test]
async fn hard_budget_compression_hint_reduces_context_before_the_provider_call() {
    let model = Arc::new(ScriptedModel::new(vec![
        ModelResponse::assistant("reduced").with_usage(Usage {
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: 10,
            ..Usage::default()
        }),
    ]));
    let budget = Arc::new(RecordingBudget::hard());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_budget(budget.clone());
    let harness: AgentHarness<()> = AgentHarness::new();

    let mut prior_tool_call = ModelResponse::assistant("");
    prior_tool_call
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "prior-lookup",
            "lookup",
            json!({"query": "old context"}),
        ));
    let original_messages = vec![
        tinyinference_llm::message::Message::system("system instruction one: preserve exactly"),
        tinyinference_llm::message::Message::system("system instruction two: preserve exactly"),
        tinyinference_llm::message::Message::user("old context ".repeat(80)),
        tinyinference_llm::message::Message::Assistant(prior_tool_call.message),
        tinyinference_llm::message::Message::tool("prior-lookup", "old lookup result ".repeat(4)),
        tinyinference_llm::message::Message::user("current task ".repeat(20)),
    ];
    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", original_messages.clone()),
                RunContext::new(RunConfig::new("hard-compression"), ()),
            ),
            &(),
        )
        .await
        .expect("hard compression reduces a multi-turn request before calling the provider");
    assert_eq!(run.text().as_deref(), Some("reduced"));
    let request = model.requests().pop().expect("provider was called once");
    assert!(
        request.messages.len() < original_messages.len(),
        "hard compression sent fewer messages to the provider"
    );
    let preserved_system: Vec<_> = request
        .messages
        .iter()
        .filter(|message| matches!(message, tinyinference_llm::message::Message::System(_)))
        .cloned()
        .collect();
    assert_eq!(
        preserved_system,
        original_messages[..2],
        "every system instruction survives byte-for-byte"
    );
    assert!(
        crate::summarization::tool_pairing_is_intact(&request.messages),
        "hard budget trimming leaves a provider-valid tool transcript"
    );
    assert!(
        request.messages.iter().any(|message| {
            matches!(message, tinyinference_llm::message::Message::Assistant(assistant) if !assistant.tool_calls.is_empty())
        }) && request
            .messages
            .iter()
            .any(|message| matches!(message, tinyinference_llm::message::Message::Tool(_))),
        "the reduced request retains a complete user/tool conversational payload"
    );
    assert_eq!(budget.records.lock().expect("budget lock").len(), 1);
}

/// A tool with a deliberately long description, so folding its catalogue
/// entry into the system prompt (as a forced text dialect does) is a large,
/// easily distinguished jump in estimated prompt size.
struct VerboseTool;

#[async_trait]
impl Tool for VerboseTool {
    fn name(&self) -> &str {
        "verbose_lookup"
    }

    fn description(&self) -> &str {
        // ~2 KiB: large enough that folding it into the system prompt moves
        // the token estimate by hundreds of tokens under the `chars / 4`
        // heuristic `token_estimation::estimate_slice_tokens` uses.
        "look something up in the verbose index. "
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "q": { "type": "string", "description": "x".repeat(2000) } },
            "required": ["q"]
        })
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("verbose-output"))
    }
}

#[tokio::test]
async fn budget_preflight_estimate_reflects_the_dialect_rewritten_request() {
    // A forced text dialect (`Xml` here) folds the protocol block and full
    // tool catalogue into `request.messages` and clears `request.tools`.
    // `token_estimation::estimate_slice_tokens` only looks at
    // `request.messages`, so the host budget preflight estimate has to be
    // taken *after* that rewrite or it silently estimates a request far
    // smaller than the one actually sent to the provider — the whole point
    // of a pre-call budget limit defeated by the rewrite arriving late.
    let model = Arc::new(ScriptedModel::replies(vec!["done"]));
    let budget = Arc::new(EstimateRecordingBudget::new());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            AgentDefinition::new("helper", "Helper", "test helper").with_tools(["verbose_lookup"]),
        ])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_budget(budget.clone());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_tool(Arc::new(VerboseTool))
        .with_policy(RunPolicy {
            tool_dialect: crate::config::ToolDispatcher::Xml,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                RunContext::new(RunConfig::new("dialect-budget"), ()),
            ),
            &(),
        )
        .await
        .expect("run succeeds");
    assert_eq!(run.text().as_deref(), Some("done"));

    // The request the provider actually received carries the rendered
    // catalogue, not a schema — confirming the dialect rewrite did happen
    // before this call.
    let request = model.requests().pop().expect("provider was called once");
    assert!(request.tools.is_empty(), "no schema goes on the wire");
    let system = request
        .messages
        .iter()
        .find(|m| matches!(m, tinyinference_llm::message::Message::System(_)))
        .expect("a system turn carries the protocol")
        .text();
    assert!(system.contains("verbose_lookup"), "{system}");

    let estimates = budget.estimates.lock().expect("estimate budget lock");
    assert_eq!(estimates.len(), 1);
    // The bare user turn ("go") alone estimates to a handful of tokens; the
    // rewritten request additionally carries the ~2 KiB tool description
    // folded into the system prompt. A stale pre-rewrite estimate would stay
    // near the former; this asserts it reflects the latter.
    assert!(
        estimates[0] > 300,
        "preflight estimate ({}) does not reflect the dialect-rewritten request",
        estimates[0]
    );
}

#[tokio::test]
async fn public_hosted_stream_sanitizes_provider_middleware_and_budget_failures() {
    fn host(model: Arc<dyn ChatModel<()>>) -> crate::host::HostCapabilities<()> {
        crate::host::HostCapabilities::new(
            Arc::new(StaticContextComposer::empty()),
            Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
                "helper",
                "Helper",
                "test helper",
            )])),
            Arc::new(AllowAllSecurityGate),
            Arc::new(FixedModelResolver::new(model)),
        )
    }

    async fn collect(
        harness: &AgentHarness<()>,
        host: crate::host::HostCapabilities<()>,
    ) -> Vec<crate::agent_loop::AgentStreamItem> {
        let mut stream = harness
            .invoke_agent_stream(
                AgentInvocation::new(
                    host,
                    AgentTurnRequest::new(
                        "helper",
                        vec![tinyinference_llm::message::Message::user("go")],
                    ),
                    RunContext::new(RunConfig::new("public-hosted-stream"), ()),
                ),
                &(),
            )
            .await
            .expect("hosted stream starts");
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        items
    }

    fn assert_sanitized(items: &[crate::agent_loop::AgentStreamItem], secret: &str) {
        assert!(
            !format!("{items:#?}").contains(secret),
            "public hosted stream leaked its raw failure detail"
        );
        assert!(items.iter().any(|item| matches!(
            item,
            crate::agent_loop::AgentStreamItem::Failed { error, .. }
                if error == "hosted agent invocation failed"
        )));
        assert!(items.iter().any(|item| matches!(
            item,
            crate::agent_loop::AgentStreamItem::Event(record)
                if matches!(&record.event, crate::events::AgentEvent::RunFailed { error, .. }
                    if error == "hosted agent invocation failed")
        )));
    }

    let provider = AgentHarness::new();
    let provider_items = collect(&provider, host(Arc::new(SecretProviderModel))).await;
    assert_sanitized(&provider_items, "provider-stream-secret");

    let mut middleware = AgentHarness::new();
    middleware.push_middleware(Arc::new(SecretAfterModelMiddleware));
    assert_sanitized(
        &collect(&middleware, host(Arc::new(UsageReportingModel))).await,
        "middleware-stream-secret",
    );

    let budget = AgentHarness::new();
    let budget_items = collect(
        &budget,
        host(Arc::new(UsageReportingModel)).with_budget(Arc::new(SecretRecordBudget)),
    )
    .await;
    assert_sanitized(&budget_items, "budget-stream-secret");
    assert!(budget_items.iter().any(|item| matches!(
        item,
        crate::agent_loop::AgentStreamItem::Event(record)
            if matches!(&record.event, crate::events::AgentEvent::ModelFailed { error, .. }
                if error == "hosted model invocation failed")
    )));
}

#[tokio::test]
async fn hard_budget_compression_fails_closed_when_only_system_instructions_remain() {
    let model = Arc::new(ScriptedModel::replies(vec!["must not run"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_budget(Arc::new(RecordingBudget::hard()));
    let harness: AgentHarness<()> = AgentHarness::new();

    let error = harness
        .invoke_agent(
            AgentInvocation::new(
                host.clone(),
                AgentTurnRequest::new(
                    "helper",
                    vec![
                        tinyinference_llm::message::Message::system(
                            "do not remove this instruction",
                        ),
                        tinyinference_llm::message::Message::system("nor this instruction"),
                    ],
                ),
                RunContext::new(RunConfig::new("hard-system-only"), ()),
            ),
            &(),
        )
        .await
        .expect_err("hard pressure cannot discard sole system instructions");
    assert_eq!(error.kind, crate::runtime::HostedErrorKind::Policy);
    assert_eq!(
        error.to_string(),
        "hosted agent invocation was rejected by policy",
        "hosted callers receive no internal budget diagnostic"
    );
    assert!(model.requests().is_empty(), "provider was never called");
}

#[tokio::test]
async fn soft_budget_compression_hint_reduces_multiturn_context_without_blocking() {
    let model = Arc::new(ScriptedModel::replies(vec!["soft reduced"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_budget(Arc::new(RecordingBudget::soft()));
    let harness: AgentHarness<()> = AgentHarness::new();

    harness
        .invoke_agent(
            AgentInvocation::new(
                host.clone(),
                AgentTurnRequest::new(
                    "helper",
                    vec![
                        tinyinference_llm::message::Message::user("one ".repeat(20)),
                        tinyinference_llm::message::Message::assistant("two ".repeat(20)),
                        tinyinference_llm::message::Message::user("three ".repeat(20)),
                        tinyinference_llm::message::Message::assistant("four ".repeat(20)),
                    ],
                ),
                RunContext::new(RunConfig::new("soft-compression"), ()),
            ),
            &(),
        )
        .await
        .expect("soft compression still invokes the provider");

    assert!(
        model.requests()[0].messages.len() < 4,
        "soft compression reduced the provider request"
    );
}

#[tokio::test]
async fn cached_streaming_deltas_reach_events_and_progress_after_middleware() {
    let model = Arc::new(ScriptedModel::replies(vec!["secret"]));
    let cache = Arc::new(crate::cache::InMemoryResponseCache::new());
    let definition = || {
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )]))
    };
    let seed_host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        definition(),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let mut seed: AgentHarness<()> = AgentHarness::new();
    seed.with_response_cache(cache.clone());

    let mut seeded = seed
        .invoke_agent_stream(
            AgentInvocation::new(
                seed_host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("same")],
                ),
                RunContext::new(RunConfig::new("cache-seed"), ()),
            ),
            &(),
        )
        .await
        .expect("cache seed starts");
    while seeded.next().await.is_some() {}

    let progress = Arc::new(RecordingProgressSink::new());
    let replay_host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        definition(),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_progress(progress.clone());
    let mut replay_harness: AgentHarness<()> = AgentHarness::new();
    replay_harness.with_response_cache(cache);

    replay_harness.push_middleware(Arc::new(RedactDeltaMiddleware));
    let mut replay = replay_harness
        .invoke_agent_stream(
            AgentInvocation::new(
                replay_host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("same")],
                ),
                RunContext::new(RunConfig::new("cache-replay"), ()),
            ),
            &(),
        )
        .await
        .expect("cache replay starts");
    let mut replayed_events = Vec::new();
    let mut replayed_run = None;
    while let Some(item) = replay.next().await {
        match item {
            crate::agent_loop::AgentStreamItem::Event(event) => replayed_events.push(event),
            crate::agent_loop::AgentStreamItem::Completed(run) => replayed_run = Some(run),
            crate::agent_loop::AgentStreamItem::Failed { error, .. } => {
                panic!("cache replay failed unexpectedly: {error}")
            }
        }
    }

    assert_eq!(
        model.requests().len(),
        1,
        "replay does not invoke the model"
    );
    assert!(replayed_events.iter().any(
        |record| matches!(&record.event, crate::events::AgentEvent::ModelDelta { delta, .. } if delta.text == "[redacted]")
    ));
    assert!(!replayed_events.iter().any(
        |record| matches!(&record.event, crate::events::AgentEvent::ModelDelta { delta, .. } if delta.text == "secret")
    ));
    assert_eq!(
        replayed_run.as_ref().and_then(|run| run.text()).as_deref(),
        Some("[redacted]"),
        "the replayed terminal response follows transformed deltas"
    );
    yield_until(|| {
        progress.events().iter().any(
            |event| matches!(event, crate::host::ProgressEvent::Token { text, .. } if text == "[redacted]")
        )
    })
    .await;
    assert!(!progress.events().iter().any(
        |event| matches!(event, crate::host::ProgressEvent::Token { text, .. } if text == "secret")
    ));
}

#[tokio::test]
async fn cached_host_response_does_not_re_record_provider_usage() {
    let model = Arc::new(ScriptedModel::new(vec![
        ModelResponse::assistant("cached").with_usage(Usage {
            input_tokens: 2,
            output_tokens: 3,
            total_tokens: 5,
            ..Usage::default()
        }),
    ]));
    let budget = Arc::new(RecordingBudget::permissive());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_budget(budget.clone());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_response_cache(Arc::new(crate::cache::InMemoryResponseCache::new()));

    for run_id in ["cache-one", "cache-two"] {
        harness
            .invoke_agent(
                AgentInvocation::new(
                    host.clone(),
                    AgentTurnRequest::new(
                        "helper",
                        vec![tinyinference_llm::message::Message::user("same")],
                    ),
                    RunContext::new(RunConfig::new(run_id), ()),
                ),
                &(),
            )
            .await
            .expect("cached run succeeds");
    }
    assert_eq!(model.requests().len(), 1, "second call is cache-served");
    assert_eq!(budget.records.lock().expect("budget lock").len(), 1);
}

#[test]
fn host_invocation_binding_fails_closed_on_a_state_mismatch() {
    struct OtherState;

    let host = Arc::new(crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "parent", "Parent", "hosted",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(Arc::new(ScriptedModel::replies(
            vec!["unused"],
        )))),
    ));
    let mut context: RunContext<()> = RunContext::new(RunConfig::new("state-mismatch"), ());
    context.host_agent_id = Some("parent".to_string());
    context.host_authority = Some(Arc::new(
        crate::runtime::HostInvocationAuthority::<(), ()> {
            binding: Arc::new(crate::runtime::HostInvocationBinding {
                host,
                agent_id: "parent".to_string(),
                model_pin: None,
                role: None,
                allowed_tools: None,
                tool_rules: None,
                progress: None,
                runtime: None,
            }),
        },
    ));

    // Reading it back with the *same* `State`/`Ctx` the authority was
    // installed for succeeds.
    assert!(
        crate::runtime::host_invocation_binding::<(), ()>(&context)
            .expect("matching State/Ctx must not be rejected")
            .is_some()
    );

    // Reading the same context with a *different* `State` must fail closed
    // rather than transmute the wrong `HostInvocationBinding<_>` out of the
    // erased authority.
    let mismatched = crate::runtime::host_invocation_binding::<OtherState, ()>(&context);
    assert!(
        matches!(
            mismatched,
            Err(crate::error::TinyAgentsError::Validation(_))
        ),
        "expected a fail-closed Validation error"
    );
}

/// C-1 regression: `RunContext::child_with_data` (the only primitive that
/// changes `Ctx`) must never propagate host authority, closing the other
/// half of the C-1 repro (a child built with a different `Ctx` type
/// inheriting a parent's hosted authority for the wrong `Ctx`).
#[test]
fn child_with_data_never_propagates_host_authority() {
    let host = Arc::new(crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "parent", "Parent", "hosted",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(Arc::new(ScriptedModel::replies(
            vec!["unused"],
        )))),
    ));
    let mut parent: RunContext<()> = RunContext::new(RunConfig::new("ctx-change-parent"), ());
    parent.host_authority = Some(Arc::new(
        crate::runtime::HostInvocationAuthority::<(), ()> {
            binding: Arc::new(crate::runtime::HostInvocationBinding {
                host,
                agent_id: "parent".to_string(),
                model_pin: None,
                role: None,
                allowed_tools: None,
                tool_rules: None,
                progress: None,
                runtime: None,
            }),
        },
    ));
    assert!(parent.host_authority.is_some());

    // Same-`Ctx` `child` propagates authority.
    let same_ctx_child = parent.child(RunConfig::new("same-ctx"), ()).unwrap();
    assert!(same_ctx_child.host_authority.is_some());

    // Different-`Ctx` `child_with_data` never does, regardless of the
    // authority the parent carries.
    let different_ctx_child = parent
        .child_with_data(RunConfig::new("different-ctx"), "child-data")
        .unwrap();
    assert!(different_ctx_child.host_authority.is_none());
}

/// Atomically decrement `counter` if it is non-zero, returning whether it was.
///
/// A compare-exchange loop rather than `fetch_update`: that method was renamed
/// `try_update` on newer toolchains (deprecating the old name, which clippy
/// `-D warnings` rejects) while `try_update` does not exist at the workspace
/// MSRV, so neither spelling compiles cleanly on both.
fn take_one(counter: &AtomicUsize) -> bool {
    let mut current = counter.load(Ordering::SeqCst);
    while let Some(next) = current.checked_sub(1) {
        match counter.compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return true,
            Err(actual) => current = actual,
        }
    }
    false
}

#[tokio::test]
async fn hosted_error_run_terminal_outcome_message_is_sanitized() {
    struct LeakyModel;
    #[async_trait]
    impl tinyinference_llm::model::ChatModel<()> for LeakyModel {
        async fn invoke(
            &self,
            _: &(),
            _: tinyinference_llm::model::ModelRequest,
        ) -> tinyinference_llm::Result<tinyinference_llm::model::ModelResponse> {
            Err(tinyinference_llm::Error::Model(
                "secret-provider-detail".into(),
            ))
        }
    }
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(Arc::new(LeakyModel))),
    );
    let harness: AgentHarness<()> = AgentHarness::new();
    let error = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("hi")],
                ),
                RunContext::new(RunConfig::new("leaky"), ()),
            ),
            &(),
        )
        .await
        .expect_err("the provider fails");
    let run = error.run.expect("partial run");
    let outcome = run.terminal.expect("typed outcome");
    assert!(!outcome.message.contains("secret"), "{}", outcome.message);
}
