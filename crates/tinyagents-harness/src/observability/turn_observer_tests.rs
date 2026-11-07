use super::*;
use crate::ids::{CallId, RunId};

#[derive(Default)]
struct Recorder(Mutex<Vec<TurnObservation>>);

impl Observer for Recorder {
    fn on_event(&self, event: &TurnObservation) {
        self.0.lock().unwrap().push(event.clone());
    }
}

fn started(input: Option<Value>) -> AgentEvent {
    AgentEvent::ToolStarted {
        call_id: CallId::new("c1"),
        tool_name: "shell".into(),
        input,
        parent_call_id: None,
    }
}

fn emit_all(scope: &Arc<ObserverScope>, events: Vec<AgentEvent>) {
    let sink = EventSink::new();
    sink.subscribe(scope.clone());
    for event in events {
        sink.emit(event);
    }
}

#[test]
fn metadata_only_scope_never_exposes_tool_payloads() {
    let rec = Arc::new(Recorder::default());
    let scope = ObserverScope::new(rec.clone(), TraceContent::MetadataOnly);
    emit_all(
        &scope,
        vec![
            AgentEvent::RunStarted {
                run_id: RunId::new("r1"),
                thread_id: None,
            },
            started(Some(serde_json::json!({"secret": 1}))),
        ],
    );
    let seen = rec.0.lock().unwrap();
    assert_eq!(seen.len(), 1);
    match &seen[0] {
        TurnObservation::Tool {
            run_id,
            call_id,
            name,
            failed,
            input,
            ..
        } => {
            assert_eq!(run_id.as_deref(), Some("r1"));
            assert_eq!(call_id, "c1");
            assert_eq!(name, "shell");
            assert_eq!(*failed, None);
            assert!(input.is_none());
        }
        other => panic!("unexpected observation {other:?}"),
    }
}

#[test]
fn include_scope_forwards_tool_input() {
    let rec = Arc::new(Recorder::default());
    let scope = ObserverScope::new(rec.clone(), TraceContent::Include);
    emit_all(&scope, vec![started(Some(serde_json::json!({"a": 1})))]);
    let seen = rec.0.lock().unwrap();
    assert!(matches!(
        &seen[0],
        TurnObservation::Tool { input: Some(v), .. } if v == &serde_json::json!({"a": 1})
    ));
}

#[test]
fn events_without_tool_identity_are_ignored() {
    let rec = Arc::new(Recorder::default());
    let scope = ObserverScope::new(rec.clone(), TraceContent::Include);
    emit_all(&scope, vec![AgentEvent::MemoryLoaded]);
    assert!(rec.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn scope_is_visible_only_inside_with_observer() {
    assert!(current_scope().is_none());
    let scope = ObserverScope::new(Arc::new(Recorder::default()), TraceContent::default());
    let inside = with_observer(scope, async { current_scope().is_some() }).await;
    assert!(inside);
    assert!(current_scope().is_none());
}

#[tokio::test]
async fn install_attaches_only_for_root_turns_with_a_scope() {
    let scope = ObserverScope::new(Arc::new(Recorder::default()), TraceContent::Include);
    let mut harness: AgentHarness<(), ()> = AgentHarness::new();
    let events = EventSink::new();
    install(&mut harness, &events, true);
    assert!(!harness.policy().capture.tool_io, "no scope, no capture");
    with_observer(scope.clone(), async {
        install(&mut harness, &events, false);
        assert!(!harness.policy().capture.tool_io, "non-root is skipped");
        install(&mut harness, &events, true);
    })
    .await;
    assert!(harness.policy().capture.tool_io);
}

struct PlainResponseModel;

#[async_trait::async_trait]
impl tinyinference_llm::model::ChatModel<()> for PlainResponseModel {
    async fn invoke(
        &self,
        _: &(),
        _: tinyinference_llm::model::ModelRequest,
    ) -> tinyinference_llm::Result<tinyinference_llm::model::ModelResponse> {
        Ok(tinyinference_llm::model::ModelResponse::assistant("ok"))
    }
}

struct PhysicalDispatchBase(Arc<dyn tinyinference_llm::model::ChatModel<()>>);

impl crate::middleware::ModelBaseCall<(), ()> for PhysicalDispatchBase {
    fn call<'a>(
        &'a self,
        _: &'a mut RunContext,
        state: &'a (),
        request: ModelRequest,
    ) -> crate::middleware::BoxModelFuture<'a> {
        Box::pin(async move { Ok(self.0.invoke(state, request).await?) })
    }
}

#[tokio::test]
async fn installed_model_observer_restores_scope_after_caller_scope_ends() {
    use crate::middleware::MiddlewareStack;

    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), TraceContent::MetadataOnly);
    let provider = observe_physical_dispatches(Arc::new(PlainResponseModel));
    let base = PhysicalDispatchBase(provider);
    let mut stack = MiddlewareStack::new();
    with_observer(scope.clone(), async {
        stack.push_model_middleware(Arc::new(InstalledObserverMiddleware(scope)));
    })
    .await;

    let mut ctx = RunContext::new(crate::context::RunConfig::new("scope-test"), ());
    stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect("model call");

    let events = recorder.0.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, TurnObservation::ModelDispatch { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, TurnObservation::Model { .. }))
            .count(),
        1
    );
}

struct NeverCalled;

#[async_trait::async_trait]
impl tinyinference_llm::model::ChatModel<()> for NeverCalled {
    async fn invoke(
        &self,
        _: &(),
        _: tinyinference_llm::model::ModelRequest,
    ) -> tinyinference_llm::Result<tinyinference_llm::model::ModelResponse> {
        panic!("budget refusal must happen before provider invocation")
    }
}

#[tokio::test]
async fn budget_refusal_emits_no_physical_dispatch_event() {
    use tinyinference_llm::model::budget::{Budget, CallBudget, SpendLimits};

    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), TraceContent::MetadataOnly);
    let budget = crate::cost::ModelBudget::new(
        Budget::new(SpendLimits {
            tokens: Some(0),
            cost_micros: None,
        }),
        CallBudget {
            input_tokens: 10_000,
            output_tokens: 100,
            cost_micros: 0,
        },
    );
    let model = budget.wrap_model(Arc::new(NeverCalled));
    let result = with_observer(scope, async {
        model
            .invoke(&(), tinyinference_llm::model::ModelRequest::new(Vec::new()))
            .await
    })
    .await;

    assert!(result.is_err());
    assert!(recorder.0.lock().unwrap().is_empty());
}

/// A base call that answers like a gateway-routed provider and, when
/// `streams`, records its first output the way the agent loop does.
struct GatewayBase {
    streams: bool,
}

impl crate::middleware::ModelBaseCall<(), ()> for GatewayBase {
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext,
        _: &'a (),
        _: ModelRequest,
    ) -> crate::middleware::BoxModelFuture<'a> {
        Box::pin(async move {
            if self.streams {
                ctx.call_first_output_at = Some(Instant::now());
            }
            let mut response = tinyinference_llm::model::ModelResponse::assistant("ok");
            response.message.id = Some("gen-7".into());
            response.raw = Some(serde_json::json!({"provider": "Wafer"}));
            Ok(response)
        })
    }
}

async fn observe_one_model_call(
    base: GatewayBase,
    ctx: &mut RunContext,
) -> Option<TurnObservation> {
    use crate::middleware::MiddlewareStack;

    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), TraceContent::MetadataOnly);
    let mut stack = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(InstalledObserverMiddleware(scope)));
    stack
        .run_wrapped_model(ctx, &(), ModelRequest::default(), &base)
        .await
        .expect("model call");
    let events = recorder.0.lock().unwrap();
    events
        .iter()
        .find(|event| matches!(event, TurnObservation::Model { .. }))
        .cloned()
}

#[tokio::test]
async fn model_observation_reports_who_served_it_and_its_first_token_time() {
    let mut ctx = RunContext::new(crate::context::RunConfig::new("facts"), ());
    let event = observe_one_model_call(GatewayBase { streams: true }, &mut ctx).await;
    let Some(TurnObservation::Model {
        served_by,
        response_id,
        first_token_ms,
        duration_ms,
        ..
    }) = event
    else {
        panic!("no model observation")
    };
    assert_eq!(served_by.as_deref(), Some("Wafer"));
    assert_eq!(response_id.as_deref(), Some("gen-7"));
    assert!(first_token_ms.is_some_and(|first| first <= duration_ms));
}

#[tokio::test]
async fn an_earlier_calls_first_output_is_not_reported_as_this_calls() {
    let mut ctx = RunContext::new(crate::context::RunConfig::new("stale"), ());
    // Left behind by a previous call; this one never reaches the stream.
    ctx.call_first_output_at = Some(Instant::now());
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    let event = observe_one_model_call(GatewayBase { streams: false }, &mut ctx).await;
    assert!(matches!(
        event,
        Some(TurnObservation::Model {
            first_token_ms: None,
            ..
        })
    ));
}

/// Runs one real tool call through the agent loop with an observer scope on the
/// run's event stream. Returns the observations and the journal-shaped events.
async fn run_observed_tool(
    capture: TraceContent,
) -> (Vec<TurnObservation>, Vec<crate::events::AgentEvent>) {
    use crate::events::RecordingListener;
    use crate::testkit::{FakeTool, ScriptedModel, text_response, tool_call_response};
    use tinyinference_llm::tool::ToolCall;

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(ScriptedModel::new(vec![
            tool_call_response(ToolCall::new(
                "c1",
                "lookup",
                serde_json::json!({"q": "weather"}),
            )),
            text_response("done"),
        ])),
    )));
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "sunny day")));

    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), capture);
    let events = Arc::new(RecordingListener::new());
    let ctx = RunContext::new(crate::context::RunConfig::new("sizes"), ());
    ctx.events.subscribe(scope);
    ctx.events.subscribe(events.clone());
    harness
        .invoke_in_context(
            &(),
            ctx,
            vec![tinyinference_llm::message::Message::user("go")],
        )
        .await
        .expect("run succeeds");
    let observed = recorder.0.lock().unwrap().clone();
    let journal = events.events().into_iter().map(|r| r.event).collect();
    (observed, journal)
}

fn completed_tool_observation(seen: &[TurnObservation]) -> TurnObservation {
    seen.iter()
        .find(|o| {
            matches!(
                o,
                TurnObservation::Tool {
                    failed: Some(_),
                    ..
                }
            )
        })
        .cloned()
        .expect("a completed tool observation")
}

/// `{"q":"weather"}` is 15 bytes; `sunny day` is 9.
const ARGUMENT_BYTES: u64 = 15;
const RESULT_BYTES: u64 = 9;

#[tokio::test]
async fn tool_observation_carries_argument_and_result_sizes() {
    let (seen, _) = run_observed_tool(TraceContent::Include).await;
    let TurnObservation::Tool {
        input_bytes,
        output_bytes,
        ..
    } = completed_tool_observation(&seen)
    else {
        unreachable!()
    };
    assert_eq!(input_bytes, Some(ARGUMENT_BYTES));
    assert_eq!(output_bytes, Some(RESULT_BYTES));
}

#[tokio::test]
async fn tool_observation_reports_sizes_without_content_when_capture_is_off() {
    let (seen, _) = run_observed_tool(TraceContent::MetadataOnly).await;
    let TurnObservation::Tool {
        input,
        output,
        input_bytes,
        output_bytes,
        ..
    } = completed_tool_observation(&seen)
    else {
        unreachable!()
    };
    assert_eq!(input_bytes, Some(ARGUMENT_BYTES));
    assert_eq!(output_bytes, Some(RESULT_BYTES));
    assert!(input.is_none() && output.is_none(), "sizes are not content");
}

#[tokio::test]
async fn tool_start_observation_has_no_sizes() {
    let (seen, _) = run_observed_tool(TraceContent::MetadataOnly).await;
    let start = seen
        .iter()
        .find(|o| matches!(o, TurnObservation::Tool { failed: None, .. }))
        .expect("a tool start observation");
    assert!(matches!(
        start,
        TurnObservation::Tool {
            input_bytes: None,
            output_bytes: None,
            ..
        }
    ));
}

#[tokio::test]
async fn langfuse_output_bytes_equals_the_observations() {
    use crate::observability::AgentObservation;
    use crate::observability::langfuse::{LangfuseClient, LangfuseTraceConfig};

    let (seen, journal) = run_observed_tool(TraceContent::MetadataOnly).await;
    let TurnObservation::Tool { output_bytes, .. } = completed_tool_observation(&seen) else {
        unreachable!()
    };
    let completed = journal
        .into_iter()
        .find(|e| matches!(e, AgentEvent::ToolCompleted { .. }))
        .expect("a ToolCompleted event");
    let client =
        LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t").unwrap();
    let batch = client
        .build_ingestion_batch(
            LangfuseTraceConfig::default(),
            &[AgentObservation {
                event_id: crate::ids::EventId::new("evt-1"),
                run_id: RunId::new("sizes"),
                parent_run_id: None,
                root_run_id: RunId::new("sizes"),
                offset: 0,
                ts_ms: 1_704_067_200_000,
                event: completed,
            }],
        )
        .unwrap();
    let span = batch["batch"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["type"] == "span-create" && e["body"]["name"] == "lookup")
        .expect("the tool span");
    assert_eq!(
        span["body"]["metadata"]["output_bytes"].as_u64(),
        output_bytes
    );
    assert_eq!(output_bytes, Some(RESULT_BYTES));
}

fn setup_phases(seen: &[TurnObservation]) -> Vec<(&'static str, bool, bool)> {
    seen.iter()
        .filter_map(|o| match o {
            TurnObservation::Setup {
                phase,
                failed,
                skipped,
                ..
            } => Some((phase.as_str(), *failed, *skipped)),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn setup_phases_arrive_in_order_before_the_first_model_event() {
    use crate::testkit::{ScriptedModel, text_response};
    use std::time::Duration;

    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), TraceContent::MetadataOnly);
    // The host times its own setup, then reports each phase on the turn's scope.
    scope.report_setup(
        SetupPhase::TranscriptResume,
        Duration::from_millis(12),
        SetupOutcome::Completed,
    );
    scope.report_setup(
        SetupPhase::MemoryBlock,
        Duration::from_millis(5_000),
        SetupOutcome::Skipped,
    );
    scope.report_setup(
        SetupPhase::Other("warm_cache"),
        Duration::from_millis(3),
        SetupOutcome::Failed,
    );

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(ScriptedModel::new(vec![text_response("done")])),
    )));
    harness.push_model_middleware(Arc::new(InstalledObserverMiddleware(scope.clone())));
    let ctx = RunContext::new(crate::context::RunConfig::new("setup"), ());
    harness
        .invoke_in_context(
            &(),
            ctx,
            vec![tinyinference_llm::message::Message::user("go")],
        )
        .await
        .expect("run succeeds");

    let seen = recorder.0.lock().unwrap().clone();
    assert_eq!(
        setup_phases(&seen),
        vec![
            ("transcript_resume", false, false),
            ("memory_block", false, true),
            ("warm_cache", true, false),
        ],
        "one event per phase, in the order reported"
    );
    let first_model = seen
        .iter()
        .position(|o| matches!(o, TurnObservation::Model { .. }))
        .expect("a model observation");
    assert_eq!(first_model, 3, "every setup event precedes the first model");
    assert!(matches!(
        &seen[1],
        TurnObservation::Setup {
            duration_ms: 5_000,
            run_id: None,
            ..
        }
    ));
}

#[tokio::test]
async fn setup_reported_through_the_task_local_scope_reaches_the_host() {
    use std::time::Duration;

    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), TraceContent::MetadataOnly);
    // No scope: reporting is a quiet no-op.
    report_setup(
        SetupPhase::AgentOpen,
        Duration::ZERO,
        SetupOutcome::Completed,
    );
    with_observer(scope, async {
        report_setup(
            SetupPhase::PromptRefresh,
            Duration::from_millis(7),
            SetupOutcome::Completed,
        );
    })
    .await;
    let seen = recorder.0.lock().unwrap().clone();
    assert_eq!(setup_phases(&seen), vec![("prompt_refresh", false, false)]);
}

#[test]
fn setup_duration_saturates_instead_of_wrapping() {
    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), TraceContent::MetadataOnly);
    scope.report_setup(
        SetupPhase::AgentOpen,
        std::time::Duration::MAX,
        SetupOutcome::Completed,
    );
    assert!(matches!(
        &recorder.0.lock().unwrap()[0],
        TurnObservation::Setup {
            duration_ms: u64::MAX,
            ..
        }
    ));
}
