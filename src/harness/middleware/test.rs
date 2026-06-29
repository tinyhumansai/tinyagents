//! Tests for the middleware stack and built-in middleware.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::*;
use crate::error::{Result, TinyAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, RecordingListener};
use crate::harness::message::{AssistantMessage, ContentBlock, Message, UserMessage};
use crate::harness::model::{ModelRequest, ModelResponse, PromptSegment, SegmentRole};
use crate::harness::summarization::TrimStrategy;
use crate::harness::usage::Usage;

// ── helpers ───────────────────────────────────────────────────────────────────

fn ctx() -> RunContext {
    RunContext::new(RunConfig::new("test-run"), ())
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![ContentBlock::Text(text.to_string())],
    })
}

fn response_with_usage(usage: Usage) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("ok".to_string())],
            tool_calls: Vec::new(),
            usage: None,
        },
        usage: Some(usage),
        finish_reason: None,
        raw: None,
        resolved_model: None,
    }
}

fn segment(id: &str, role: SegmentRole, cacheable: bool) -> PromptSegment {
    PromptSegment {
        id: id.to_string(),
        role,
        cacheable,
    }
}

/// Records hook firing order into a shared log for ordering assertions.
struct OrderRecorder {
    label: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<()> for OrderRecorder {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:before", self.label));
        Ok(())
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:after", self.label));
        Ok(())
    }
}

/// Always fails its `before_model` hook to exercise short-circuiting.
struct FailingMiddleware;

#[async_trait]
impl Middleware<()> for FailingMiddleware {
    fn name(&self) -> &str {
        "failing"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        Err(TinyAgentsError::Middleware("boom".to_string()))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn before_runs_forward_after_runs_reverse() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(OrderRecorder {
        label: "a",
        log: log.clone(),
    }));
    stack.push(Arc::new(OrderRecorder {
        label: "b",
        log: log.clone(),
    }));

    let mut c = ctx();
    let mut request = ModelRequest::default();
    let mut response = response_with_usage(Usage::new(1, 1));

    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();
    stack
        .run_after_model(&mut c, &(), &mut response)
        .await
        .unwrap();

    let order = log.lock().unwrap().clone();
    assert_eq!(
        order,
        vec!["a:before", "b:before", "b:after", "a:after"],
        "before runs in registration order, after runs reversed"
    );
}

#[tokio::test]
async fn error_short_circuits_and_invokes_on_error() {
    let logging = Arc::new(LoggingMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(logging.clone());
    stack.push(Arc::new(FailingMiddleware));
    // This third middleware must never run because the second one fails first.
    let never = Arc::new(LoggingMiddleware::with_label("never"));
    stack.push(never.clone());

    let mut c = ctx();
    let mut request = ModelRequest::default();
    let result = stack.run_before_model(&mut c, &(), &mut request).await;

    assert!(matches!(result, Err(TinyAgentsError::Middleware(_))));
    // on_error fanned out to the whole stack, so the first logging mw saw it.
    assert_eq!(logging.counts().on_error, 1);
    // The first logging mw's before_model ran; the one after the failure did not.
    assert_eq!(logging.counts().before_model, 1);
    assert_eq!(never.counts().before_model, 0);
}

#[tokio::test]
async fn emits_started_and_completed_events() {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(LoggingMiddleware::new()));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest::default();
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let kinds: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert_eq!(
        kinds,
        vec![
            AgentEvent::MiddlewareStarted {
                name: "logging".to_string()
            },
            AgentEvent::MiddlewareCompleted {
                name: "logging".to_string()
            },
        ]
    );
}

#[tokio::test]
async fn message_trim_middleware_shrinks_request() {
    let mw = MessageTrimMiddleware::new(TrimStrategy::KeepLast(1));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut request = ModelRequest {
        messages: vec![user("one"), user("two"), user("three")],
        ..Default::default()
    };
    let mut c = ctx();
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0], user("three"));
}

#[tokio::test]
async fn usage_accounting_accumulates_across_calls() {
    let mw = Arc::new(UsageAccountingMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    let mut r1 = response_with_usage(Usage::new(10, 5));
    let mut r2 = response_with_usage(Usage::new(3, 2));
    stack.run_after_model(&mut c, &(), &mut r1).await.unwrap();
    stack.run_after_model(&mut c, &(), &mut r2).await.unwrap();

    let totals = mw.totals();
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.input_tokens, 13);
    assert_eq!(totals.usage.output_tokens, 7);
    assert_eq!(totals.usage.total_tokens, 20);
}

#[tokio::test]
async fn prompt_cache_guard_detects_prefix_change() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();

    // First call establishes a cacheable prefix [sys].
    let mut req1 = ModelRequest {
        cache_segments: vec![segment("sys", SegmentRole::System, true)],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut req1)
        .await
        .unwrap();
    assert!(mw.layout_events().is_empty(), "no prior layout to compare");

    // Second call changes the stable prefix -> a layout event is recorded.
    let mut req2 = ModelRequest {
        cache_segments: vec![segment("sys2", SegmentRole::System, true)],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut req2)
        .await
        .unwrap();

    let events = mw.layout_events();
    assert_eq!(events.len(), 1);
    assert!(events[0].changed_prefix);
    assert_eq!(events[0].segment_ids_before, vec!["sys".to_string()]);
    assert_eq!(events[0].segment_ids_after, vec!["sys2".to_string()]);
}

#[tokio::test]
async fn agent_run_text_reflects_final_response() {
    let mut run = AgentRun::new();
    assert_eq!(run.text(), None);
    run.final_response = Some(response_with_usage(Usage::new(1, 1)));
    assert_eq!(run.text(), Some("ok".to_string()));
}
