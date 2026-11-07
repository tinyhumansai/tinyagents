//! Root-turn observer hooks. Scopes attach once during harness construction;
//! payload capture is opt-in and raw errors/provider options are never exposed.
use super::physical_dispatch::{
    PhysicalDispatchMetadata, PhysicalDispatchObserver, PhysicalDispatchOutcome,
    PhysicalDispatchStatus, PhysicalDispatchTerminal,
};
use crate::context::RunContext;
use crate::events::{AgentEvent, EventListener, EventRecord, EventSink};
use crate::middleware::{MiddlewareModelOutcome, ModelHandler, ModelMiddleware};
use crate::runtime::AgentHarness;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tinyinference_llm::model::ModelRequest;

/// Explicit consent for capturing model messages and tool input/output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TraceContent {
    /// Only identifiers, durations, token counts, costs, and success/failure.
    #[default]
    MetadataOnly,
    /// Include messages and tool payloads. Hosts own their retention policy.
    Include,
}
/// Provider-reported usage; absent costs remain unknown.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObservedUsage {
    /// Prompt tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Cached prompt tokens.
    pub cached_tokens: u64,
    /// Reasoning tokens.
    pub reasoning_tokens: u64,
    /// Provider charge in USD, never a local estimate.
    pub cost_usd: Option<f64>,
}
/// A named step a host runs between receiving a turn and its first model
/// request. Names are fixed strings, never user data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SetupPhase {
    /// Re-reading the saved transcript into a cold agent.
    TranscriptResume,
    /// Refreshing the system prompt.
    PromptRefresh,
    /// Rendering the memory block.
    MemoryBlock,
    /// Opening or constructing the agent.
    AgentOpen,
    /// A host-defined phase; the name must be a static label, not user data.
    Other(&'static str),
}
impl SetupPhase {
    /// Stable snake_case name, suitable for a span name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TranscriptResume => "transcript_resume",
            Self::PromptRefresh => "prompt_refresh",
            Self::MemoryBlock => "memory_block",
            Self::AgentOpen => "agent_open",
            Self::Other(name) => name,
        }
    }
}
/// How a setup phase ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupOutcome {
    /// The phase ran to completion.
    Completed,
    /// The phase failed; the turn went on without its result.
    Failed,
    /// The phase was skipped, for example a block that missed its deadline.
    Skipped,
}
/// A curated model/tool observation. No raw errors or arbitrary metadata.
///
/// Not `#[non_exhaustive]`: matches written before [`Self::Setup`] existed must
/// add an arm (a `Setup { .. }` arm, or a wildcard).
#[derive(Clone, Debug, PartialEq)]
pub enum TurnObservation {
    /// One model pipeline returned, even if later answer validation rejects it.
    Model {
        /// Harness run identifier.
        run_id: String,
        /// Model call identifier when supplied by the harness.
        call_id: Option<String>,
        /// Requested model identifier.
        requested_model: Option<String>,
        /// Provider-reported actual answering model.
        answered_model: Option<String>,
        /// Provider finish reason.
        finish_reason: Option<String>,
        /// Upstream provider that served the call, when a routing gateway
        /// reports it (OpenRouter's top-level `provider`, e.g. `"DeepInfra"`).
        served_by: Option<String>,
        /// Provider's id for the response (`chatcmpl-…`, OpenRouter's
        /// `gen-…`, `msg_…`), to look the call up with the provider.
        response_id: Option<String>,
        /// Wall-clock time in milliseconds.
        duration_ms: u64,
        /// Milliseconds from the call's start to the first streamed output
        /// (text, reasoning or a tool call) of the attempt that produced the
        /// result. `None` for a buffered call or one that streamed nothing.
        first_token_ms: Option<u64>,
        /// Whether the pipeline failed.
        failed: bool,
        /// Provider usage.
        usage: Option<ObservedUsage>,
        /// Conversation messages, only with explicit capture consent.
        input: Option<Value>,
        /// Assistant message, only with explicit capture consent.
        output: Option<Value>,
    },
    /// One admitted physical provider invocation, independent of logical retries.
    ModelDispatch {
        /// Harness run identifier.
        run_id: String,
        /// Model call identifier when supplied by the harness.
        call_id: Option<String>,
        /// Requested model identifier.
        requested_model: Option<String>,
        /// Provider-reported answering model.
        answered_model: Option<String>,
        /// Provider finish reason.
        finish_reason: Option<String>,
        /// Upstream provider that served the call, when a routing gateway
        /// reports it (OpenRouter's top-level `provider`, e.g. `"DeepInfra"`).
        served_by: Option<String>,
        /// Provider's id for the response (`chatcmpl-…`, OpenRouter's
        /// `gen-…`, `msg_…`), to look the call up with the provider.
        response_id: Option<String>,
        /// Wall-clock time in milliseconds.
        duration_ms: u64,
        /// Milliseconds from the invocation's start to its first streamed
        /// output (text, reasoning or a tool call). `None` for a buffered call
        /// or a stream that produced no output.
        first_token_ms: Option<u64>,
        /// Terminal outcome label.
        status: PhysicalDispatchStatus,
        /// Whether the provider reported a refusal response.
        refused: bool,
        /// Provider-reported usage.
        usage: Option<ObservedUsage>,
    },
    /// A tool started or returned; `failed=None` denotes its start.
    Tool {
        /// Harness run identifier, once its start was observed.
        run_id: Option<String>,
        /// Call identifier, pairs start with completion.
        call_id: String,
        /// Tool name.
        name: String,
        /// Terminal failure status, never the raw error text.
        failed: Option<bool>,
        /// Terminal duration if available.
        duration_ms: Option<u64>,
        /// Arguments, only with explicit capture consent.
        input: Option<Value>,
        /// Result, only with explicit capture consent.
        output: Option<Value>,
        /// Size in bytes of the serialized arguments, set on completion.
        /// Reported whatever the capture consent: a size is not content.
        input_bytes: Option<u64>,
        /// Size in bytes of the result text the model is shown, set on
        /// completion. Reported whatever the capture consent, and equal to the
        /// Langfuse exporter's `output_bytes` for the same call.
        output_bytes: Option<u64>,
    },
    /// A host-reported setup phase that ran before the first model request.
    /// Carries no message text, identifiers or user data.
    Setup {
        /// Harness run identifier, once the run has started. Setup normally
        /// precedes the run, so hosts correlate by order and by their own scope.
        run_id: Option<String>,
        /// Which phase this was.
        phase: SetupPhase,
        /// Wall-clock time in milliseconds.
        duration_ms: u64,
        /// Whether the phase failed.
        failed: bool,
        /// Whether the phase was skipped.
        skipped: bool,
    },
}
/// Synchronous, nonblocking host callback. Queue export work in the host.
pub trait Observer: Send + Sync {
    /// Called once per observed model outcome or tool event.
    fn on_event(&self, event: &TurnObservation);
}
/// Observer and capture consent shared by one turn's harness.
pub struct ObserverScope {
    observer: Arc<dyn Observer>,
    capture: TraceContent,
    run_id: Mutex<Option<String>>,
}
impl ObserverScope {
    /// Construct a turn-local observer scope.
    pub fn new(observer: Arc<dyn Observer>, capture: TraceContent) -> Arc<Self> {
        Arc::new(Self {
            observer,
            capture,
            run_id: Mutex::default(),
        })
    }
    /// Harness run identifier observed during this scope's dispatch.
    pub fn run_id(&self) -> Option<String> {
        self.run_id
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    /// Report one setup phase the host ran before the first model request.
    /// Call it once per phase, in the order the phases ran; each call emits one
    /// [`TurnObservation::Setup`] to the host observer immediately.
    pub fn report_setup(&self, phase: SetupPhase, duration: Duration, outcome: SetupOutcome) {
        self.observer.on_event(&TurnObservation::Setup {
            run_id: self.run_id(),
            phase,
            duration_ms: millis(duration),
            failed: outcome == SetupOutcome::Failed,
            skipped: outcome == SetupOutcome::Skipped,
        });
    }
    fn payload(&self, value: &Option<Value>) -> Option<Value> {
        (self.capture == TraceContent::Include)
            .then(|| value.clone())
            .flatten()
    }
}
tokio::task_local! { static OBSERVER: Arc<ObserverScope>; }
/// Capture the current scope before dispatch crosses a runtime/task boundary.
pub fn current_scope() -> Option<Arc<ObserverScope>> {
    OBSERVER.try_with(Arc::clone).ok()
}

/// Report a setup phase to the current turn's observer scope, if a host
/// scoped one around this task. Quietly does nothing otherwise.
pub fn report_setup(phase: SetupPhase, duration: Duration, outcome: SetupOutcome) {
    if let Some(scope) = current_scope() {
        scope.report_setup(phase, duration, outcome);
    }
}

impl std::fmt::Debug for ObserverScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObserverScope")
            .field("capture", &self.capture)
            .finish_non_exhaustive()
    }
}

/// Scope an observation context around turn dispatch.
pub async fn with_observer<F: std::future::Future>(
    scope: Arc<ObserverScope>,
    future: F,
) -> F::Output {
    OBSERVER.scope(scope, Box::pin(future)).await
}
/// Attach to a root harness and its event stream if a host scoped an observer.
pub fn install<Ctx: Send + Sync + 'static>(
    harness: &mut AgentHarness<(), Ctx>,
    events: &EventSink,
    root: bool,
) {
    if !root {
        return;
    }
    if let Ok(scope) = OBSERVER.try_with(Arc::clone) {
        if scope.capture == TraceContent::Include {
            let mut policy = harness.policy().clone();
            policy.capture.tool_io = true;
            harness.with_policy(policy);
        }
        harness.push_model_middleware(Arc::new(InstalledObserverMiddleware(scope.clone())));
        events.subscribe(scope);
    }
}

/// Restores the captured observer task-local while an installed model hook runs.
struct InstalledObserverMiddleware(Arc<ObserverScope>);

#[async_trait]
impl<Ctx: Send + Sync + 'static> ModelMiddleware<(), Ctx> for InstalledObserverMiddleware {
    fn name(&self) -> &str {
        "turn_observer"
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), Ctx>,
    ) -> crate::error::Result<MiddlewareModelOutcome> {
        with_observer(
            Arc::clone(&self.0),
            self.0.wrap_model(ctx, state, request, next),
        )
        .await
    }
}
impl EventListener for ObserverScope {
    fn on_event(&self, record: &EventRecord) {
        if let AgentEvent::RunStarted { run_id, .. } = &record.event {
            *self
                .run_id
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(run_id.to_string());
        }
        let (call_id, name, failed, duration_ms, input, output, input_bytes, output_bytes) =
            match &record.event {
                AgentEvent::ToolStarted {
                    call_id,
                    tool_name,
                    input,
                    ..
                } => (
                    call_id,
                    tool_name,
                    None,
                    None,
                    self.payload(input),
                    None,
                    None,
                    None,
                ),
                AgentEvent::ToolCompleted {
                    call_id,
                    tool_name,
                    error,
                    duration_ms,
                    input,
                    output,
                    input_bytes,
                    output_bytes,
                    ..
                } => (
                    call_id,
                    tool_name,
                    Some(error.is_some()),
                    *duration_ms,
                    self.payload(input),
                    self.payload(output),
                    *input_bytes,
                    *output_bytes,
                ),
                AgentEvent::ToolFailed {
                    call_id,
                    tool_name,
                    duration_ms,
                    ..
                } => (
                    call_id,
                    tool_name,
                    Some(true),
                    *duration_ms,
                    None,
                    None,
                    None,
                    None,
                ),
                // Deltas, errors, metadata, and custom events can contain payloads.
                _ => return,
            };
        self.observer.on_event(&TurnObservation::Tool {
            run_id: self.run_id(),
            call_id: call_id.to_string(),
            name: name.clone(),
            failed,
            duration_ms,
            input,
            output,
            input_bytes,
            output_bytes,
        });
    }
}

impl PhysicalDispatchObserver for ObserverScope {
    fn on_start(&self, metadata: &PhysicalDispatchMetadata) -> Box<dyn PhysicalDispatchTerminal> {
        Box::new(DispatchTerminal {
            observer: Arc::clone(&self.observer),
            run_id: self.run_id().unwrap_or_default(),
            metadata: metadata.clone(),
        })
    }
}

struct DispatchTerminal {
    observer: Arc<dyn Observer>,
    run_id: String,
    metadata: PhysicalDispatchMetadata,
}

impl PhysicalDispatchTerminal for DispatchTerminal {
    fn on_terminal(self: Box<Self>, outcome: PhysicalDispatchOutcome) {
        let usage = outcome.usage.map(|usage| ObservedUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_tokens: usage.cached_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            cost_usd: usage.cost_usd,
        });
        self.observer.on_event(&TurnObservation::ModelDispatch {
            run_id: self.run_id,
            call_id: self.metadata.call_id,
            requested_model: self.metadata.requested_model,
            answered_model: outcome.answered_model,
            finish_reason: outcome.finish_reason,
            served_by: outcome.served_by,
            response_id: outcome.response_id,
            duration_ms: millis(outcome.duration),
            first_token_ms: outcome.first_output.map(millis),
            status: outcome.status,
            refused: outcome.refused,
            usage,
        });
    }
}

/// Wrap a provider so a turn's task-local observer receives physical dispatch events.
pub fn observe_physical_dispatches(
    model: Arc<dyn tinyinference_llm::model::ChatModel<()>>,
) -> Arc<dyn tinyinference_llm::model::ChatModel<()>> {
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(CurrentScopeDispatchObserver);
    Arc::new(super::physical_dispatch::ObservedChatModel::new(
        model, observer,
    ))
}

struct CurrentScopeDispatchObserver;

impl PhysicalDispatchObserver for CurrentScopeDispatchObserver {
    fn on_start(&self, metadata: &PhysicalDispatchMetadata) -> Box<dyn PhysicalDispatchTerminal> {
        match current_scope() {
            Some(scope) => scope.on_start(metadata),
            None => Box::new(NoopDispatchTerminal),
        }
    }
}

struct NoopDispatchTerminal;

impl PhysicalDispatchTerminal for NoopDispatchTerminal {
    fn on_terminal(self: Box<Self>, _: PhysicalDispatchOutcome) {}
}
#[async_trait]
impl<Ctx: Send + Sync + 'static> ModelMiddleware<(), Ctx> for ObserverScope {
    fn name(&self) -> &str {
        "turn_observer"
    }
    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), Ctx>,
    ) -> crate::error::Result<MiddlewareModelOutcome> {
        let run_id = ctx.run_id().to_string();
        let call_id = request
            .correlation
            .as_ref()
            .map(|correlation| correlation.call_id.clone());
        let requested_model = request.model.clone();
        // Serialize messages only: provider_options may contain credentials.
        let input = (self.capture == TraceContent::Include)
            .then(|| serde_json::to_value(&request.messages).ok())
            .flatten();
        let started = Instant::now();
        let result = next.run(ctx, state, request).await;
        let response = match &result {
            Ok(MiddlewareModelOutcome::Response(response)) => Some(response),
            _ => None,
        };
        let raw = response.and_then(|response| response.raw.as_ref());
        let answered_model = raw
            .and_then(|raw| raw.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                response
                    .and_then(|response| response.resolved_model.as_ref())
                    .map(|model| model.name.clone())
            });
        let raw_cost = raw
            .and_then(|raw| raw.pointer("/usage/cost"))
            .and_then(Value::as_f64);
        let usage = response
            .and_then(|response| response.usage.as_ref())
            .map(|usage| ObservedUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cached_tokens: usage.cache_read_tokens,
                reasoning_tokens: usage.reasoning_tokens,
                cost_usd: usage
                    .charged_amount
                    .as_ref()
                    .map(|amount| amount.micros as f64 / 1_000_000.0)
                    .or(raw_cost),
            })
            .or_else(|| {
                raw_cost.map(|cost_usd| ObservedUsage {
                    cost_usd: Some(cost_usd),
                    ..ObservedUsage::default()
                })
            });
        let output = if self.capture == TraceContent::Include {
            response.and_then(|response| serde_json::to_value(&response.message).ok())
        } else {
            None
        };
        // A time before `started` belongs to an earlier call, left behind when
        // a wrap middleware answered this one without reaching the model.
        let first_token_ms = ctx
            .call_first_output_at
            .and_then(|at| at.checked_duration_since(started))
            .map(millis);
        self.observer.on_event(&TurnObservation::Model {
            run_id,
            call_id,
            requested_model,
            answered_model,
            finish_reason: response.and_then(|response| response.finish_reason.clone()),
            served_by: response.and_then(super::response_facts::served_by),
            response_id: response.and_then(super::response_facts::response_id),
            duration_ms: millis(started.elapsed()),
            first_token_ms,
            failed: result.is_err(),
            usage,
            input,
            output,
        });
        result
    }
}
/// `duration` in whole milliseconds, saturating at `u64::MAX`.
fn millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

/// Existing TinyAgents exporter; this module adds no transport implementation.
///
/// Traces go out as OTLP: build requests from `TraceSpan`s with
/// `otlp_requests` and send them with `LangfuseClient::send_otlp`, the
/// supported Langfuse trace path. `create_score` still posts scores to the
/// batch API, which remains supported for scores.
#[cfg(feature = "langfuse")]
pub mod langfuse {
    pub use crate::observability::{
        LangfuseAuth, LangfuseClient, LangfuseScore, LangfuseScoreValue, LangfuseTraceConfig,
        trace_export::{ExportBrand, SpanKind, SpanStatus, TraceSpan, otlp::otlp_requests},
    };
}

#[cfg(test)]
#[path = "turn_observer_tests.rs"]
mod tests;
