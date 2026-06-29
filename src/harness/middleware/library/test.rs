//! Tests for the built-in middleware library.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::json;

use super::*;
use crate::error::{Result, TinyAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, EventRecord, RecordingListener};
use crate::harness::message::Message;
use crate::harness::middleware::{BoxModelFuture, MiddlewareStack, ModelBaseCall};
use crate::harness::model::{ModelRequest, ModelResponse, ResponseFormat};
use crate::harness::retry::{RateLimiter, RetryPolicy};
use crate::harness::tool::{ToolCall, ToolResult, ToolSchema};

// ── helpers ───────────────────────────────────────────────────────────────────

fn ctx_with_recorder() -> (RunContext, Arc<RecordingListener>) {
    let ctx = RunContext::new(RunConfig::new("test-run"), ());
    let recorder = Arc::new(RecordingListener::new());
    ctx.events.subscribe(recorder.clone());
    (ctx, recorder)
}

fn events(recorder: &RecordingListener) -> Vec<AgentEvent> {
    recorder
        .events()
        .into_iter()
        .map(|r: EventRecord| r.event)
        .collect()
}

fn ok_response() -> ModelResponse {
    ModelResponse::assistant("ok")
}

fn tool_call(name: &str) -> ToolCall {
    ToolCall {
        id: "call-1".to_string(),
        name: name.to_string(),
        arguments: json!({}),
    }
}

/// A configurable [`ModelBaseCall`] for driving wrap middleware in isolation.
struct FakeModelBase {
    calls: AtomicUsize,
    #[allow(clippy::type_complexity)]
    behavior: Box<dyn Fn(usize, &ModelRequest) -> Result<ModelResponse> + Send + Sync>,
}

impl FakeModelBase {
    fn new<F>(behavior: F) -> Self
    where
        F: Fn(usize, &ModelRequest) -> Result<ModelResponse> + Send + Sync + 'static,
    {
        Self {
            calls: AtomicUsize::new(0),
            behavior: Box::new(behavior),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ModelBaseCall<(), ()> for FakeModelBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let result = (self.behavior)(n, &request);
        Box::pin(async move { result })
    }
}

// ── RetryMiddleware ─────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_middleware_retries_then_succeeds() {
    let (mut ctx, recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(RetryMiddleware::new(
        RetryPolicy::default().with_max_attempts(4),
    )));

    // Fail (retryable) the first two attempts, then succeed.
    let base = FakeModelBase::new(|n, _req| {
        if n < 2 {
            Err(TinyAgentsError::Model("transient".to_string()))
        } else {
            Ok(ok_response())
        }
    });

    let outcome = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect("retry should eventually succeed");
    assert_eq!(outcome.into_response().text(), "ok");
    assert_eq!(base.calls(), 3);

    let scheduled = events(&recorder)
        .iter()
        .filter(|e| matches!(e, AgentEvent::RetryScheduled { .. }))
        .count();
    assert_eq!(scheduled, 2);
}

#[tokio::test]
async fn retry_middleware_does_not_retry_non_retryable() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(RetryMiddleware::with_default_policy()));

    let base =
        FakeModelBase::new(|_n, _req| Err(TinyAgentsError::Validation("bad input".to_string())));

    let err = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect_err("validation errors are not retryable");
    assert!(matches!(err, TinyAgentsError::Validation(_)));
    assert_eq!(base.calls(), 1);
}

// ── TimeoutMiddleware ───────────────────────────────────────────────────────

#[tokio::test]
async fn timeout_middleware_passes_fast_call() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(TimeoutMiddleware::from_millis(1_000)));

    let base = FakeModelBase::new(|_n, _req| Ok(ok_response()));
    let outcome = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect("fast call within timeout");
    assert_eq!(outcome.into_response().text(), "ok");
}

#[tokio::test(start_paused = true)]
async fn timeout_middleware_times_out_slow_call() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(TimeoutMiddleware::from_millis(10)));

    // A base that "hangs" far longer than the timeout. Under paused time the
    // runtime auto-advances to the timeout deadline without real sleeping.
    struct SlowBase;
    impl ModelBaseCall<(), ()> for SlowBase {
        fn call<'a>(
            &'a self,
            _ctx: &'a mut RunContext,
            _state: &'a (),
            _request: ModelRequest,
        ) -> BoxModelFuture<'a> {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(ok_response())
            })
        }
    }

    let err = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &SlowBase)
        .await
        .expect_err("slow call should time out");
    assert!(matches!(err, TinyAgentsError::Timeout(_)));
}

// ── ModelFallbackMiddleware ─────────────────────────────────────────────────

#[tokio::test]
async fn model_fallback_switches_model_on_error() {
    let (mut ctx, recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(ModelFallbackMiddleware::new([
        "backup-a", "backup-b",
    ])));

    // Only "backup-b" succeeds; the primary (None) and "backup-a" fail.
    let base = FakeModelBase::new(|_n, req| match req.model.as_deref() {
        Some("backup-b") => Ok(ok_response()),
        _ => Err(TinyAgentsError::Model("model down".to_string())),
    });

    let outcome = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect("fallback should reach a working model");
    assert_eq!(outcome.into_response().text(), "ok");
    assert_eq!(base.calls(), 3); // primary + backup-a + backup-b

    let selections: Vec<(String, String)> = events(&recorder)
        .into_iter()
        .filter_map(|e| match e {
            AgentEvent::FallbackSelected { from, to } => Some((from, to)),
            _ => None,
        })
        .collect();
    assert_eq!(
        selections,
        vec![
            (String::new(), "backup-a".to_string()),
            ("backup-a".to_string(), "backup-b".to_string()),
        ]
    );
}

#[tokio::test]
async fn model_fallback_returns_last_error_when_all_fail() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(ModelFallbackMiddleware::new(["backup"])));

    let base = FakeModelBase::new(|_n, _req| Err(TinyAgentsError::Model("down".to_string())));
    let err = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect_err("all models fail");
    assert!(matches!(err, TinyAgentsError::Model(_)));
    assert_eq!(base.calls(), 2);
}

// ── RateLimitMiddleware ─────────────────────────────────────────────────────

#[tokio::test]
async fn rate_limit_error_when_bucket_empty() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let limiter = Arc::new(RateLimiter::new(1, 0.0)); // capacity 1, no refill
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(
        RateLimitMiddleware::new(limiter.clone()).with_behavior(RateLimitBehavior::Error),
    ));

    let base = FakeModelBase::new(|_n, _req| Ok(ok_response()));

    // First call consumes the only token.
    stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect("first call admitted");
    // Second call finds the bucket empty and errors.
    let err = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect_err("second call rate limited");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)));
    assert_eq!(base.calls(), 1);
}

#[tokio::test]
async fn rate_limit_wait_then_proceed_with_advancing_clock() {
    let (mut ctx, recorder) = ctx_with_recorder();

    // Drain the bucket at a known base instant.
    let base_instant = Instant::now();
    let limiter = Arc::new(RateLimiter::new(1, 1000.0));
    assert!(limiter.try_acquire(1, base_instant));

    // Clock advances by one second on each read: the first acquire in the wrap
    // sees an empty bucket (elapsed 0), the second sees a refill.
    let counter = Arc::new(AtomicUsize::new(0));
    let clock_counter = counter.clone();
    let now: NowFn = Arc::new(move || {
        let n = clock_counter.fetch_add(1, Ordering::SeqCst) as u64;
        base_instant + Duration::from_secs(n)
    });

    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(
        RateLimitMiddleware::new(limiter)
            .waiting(Duration::ZERO) // zero poll interval => no real sleeping
            .with_clock(now),
    ));

    let model_base = FakeModelBase::new(|_n, _req| Ok(ok_response()));
    stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &model_base)
        .await
        .expect("call proceeds once the bucket refills");

    let waited = events(&recorder)
        .iter()
        .filter(|e| matches!(e, AgentEvent::RateLimitWaited { .. }))
        .count();
    assert_eq!(waited, 1);
    assert_eq!(model_base.calls(), 1);
}

// ── ToolAllowlistMiddleware ─────────────────────────────────────────────────

#[tokio::test]
async fn tool_allowlist_rejects_unlisted_tool() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(ToolAllowlistMiddleware::new(["search"])));

    let mut allowed = tool_call("search");
    stack
        .run_before_tool(&mut ctx, &(), &mut allowed)
        .await
        .expect("listed tool admitted");

    let mut blocked = tool_call("delete_everything");
    let err = stack
        .run_before_tool(&mut ctx, &(), &mut blocked)
        .await
        .expect_err("unlisted tool rejected");
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

// ── DynamicToolSelectionMiddleware ──────────────────────────────────────────

#[tokio::test]
async fn dynamic_tool_selection_filters_request_tools() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(DynamicToolSelectionMiddleware::allowing(["keep"])));

    let schema = |name: &str| ToolSchema {
        name: name.to_string(),
        description: String::new(),
        parameters: json!({}),
    };
    let mut request = ModelRequest::new(Vec::new()).with_tools(vec![
        schema("keep"),
        schema("drop"),
        schema("keep"),
    ]);

    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .expect("selection runs");
    assert_eq!(request.tools.len(), 2);
    assert!(request.tools.iter().all(|t| t.name == "keep"));
}

// ── HumanApprovalMiddleware ─────────────────────────────────────────────────

#[tokio::test]
async fn human_approval_interrupts_without_callback() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(HumanApprovalMiddleware::new(["wire_transfer"])));

    let mut call = tool_call("wire_transfer");
    let err = stack
        .run_before_tool(&mut ctx, &(), &mut call)
        .await
        .expect_err("flagged tool requires approval");
    assert!(matches!(err, TinyAgentsError::Interrupted { .. }));
}

#[tokio::test]
async fn human_approval_consults_callback() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    let approve: ApprovalFn = Arc::new(|call: &ToolCall| call.name == "wire_transfer");
    stack.push(Arc::new(
        HumanApprovalMiddleware::new(["wire_transfer", "delete"]).with_approval(approve),
    ));

    let mut approved = tool_call("wire_transfer");
    stack
        .run_before_tool(&mut ctx, &(), &mut approved)
        .await
        .expect("callback approves wire_transfer");

    let mut rejected = tool_call("delete");
    let err = stack
        .run_before_tool(&mut ctx, &(), &mut rejected)
        .await
        .expect_err("callback rejects delete");
    assert!(matches!(err, TinyAgentsError::Interrupted { .. }));
}

// ── StructuredOutputValidatorMiddleware ─────────────────────────────────────

#[tokio::test]
async fn structured_validator_rejects_non_json() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(StructuredOutputValidatorMiddleware::new(
        ResponseFormat::JsonObject,
    )));

    let mut bad = ModelResponse::assistant("not json");
    let err = stack
        .run_after_model(&mut ctx, &(), &mut bad)
        .await
        .expect_err("invalid JSON rejected");
    assert!(matches!(err, TinyAgentsError::StructuredOutput(_)));

    let mut good = ModelResponse::assistant(r#"{"ok":true}"#);
    stack
        .run_after_model(&mut ctx, &(), &mut good)
        .await
        .expect("valid JSON passes");
}

#[tokio::test]
async fn structured_validator_checks_json_schema() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    let schema = json!({ "type": "object" });
    stack.push(Arc::new(StructuredOutputValidatorMiddleware::new(
        ResponseFormat::json_schema("answer", schema),
    )));

    let mut good = ModelResponse::assistant(r#"{"value":1}"#);
    stack
        .run_after_model(&mut ctx, &(), &mut good)
        .await
        .expect("schema response parses");
}

// ── DynamicPromptMiddleware ─────────────────────────────────────────────────

#[tokio::test]
async fn dynamic_prompt_injects_system_message() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(DynamicPromptMiddleware::<(), ()>::from_fn(
        |_state, config| Some(format!("run is {}", config.run_id)),
    )));

    let mut request = ModelRequest::new(vec![Message::user("hi")]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .expect("prompt runs");
    assert_eq!(request.messages.len(), 2);
    assert!(matches!(request.messages[0], Message::System(_)));
    assert!(request.messages[0].text().contains("test-run"));
}

#[tokio::test]
async fn dynamic_prompt_skips_when_none() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(DynamicPromptMiddleware::<(), ()>::from_fn(
        |_state, _config| None,
    )));

    let mut request = ModelRequest::new(vec![Message::user("hi")]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .expect("prompt runs");
    assert_eq!(request.messages.len(), 1);
}

// ── RedactionMiddleware ─────────────────────────────────────────────────────

#[tokio::test]
async fn redaction_masks_response_and_tool_text() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let redaction = Arc::new(RedactionMiddleware::new(["sk-secret", "hunter2"]));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(redaction.clone());

    let mut response = ModelResponse::assistant("key is sk-secret and pw hunter2");
    stack
        .run_after_model(&mut ctx, &(), &mut response)
        .await
        .expect("redaction runs");
    assert_eq!(response.text(), "key is [REDACTED] and pw [REDACTED]");

    let mut result = ToolResult {
        call_id: "c".to_string(),
        name: "t".to_string(),
        content: "token sk-secret".to_string(),
        raw: None,
        error: None,
        elapsed_ms: 0,
    };
    stack
        .run_after_tool(&mut ctx, &(), &mut result)
        .await
        .expect("redaction runs on tool");
    assert_eq!(result.content, "token [REDACTED]");
    assert_eq!(redaction.redactions(), 3);
}

// ── TracingMiddleware ───────────────────────────────────────────────────────

#[tokio::test]
async fn tracing_records_phase_boundaries_and_counts() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let tracing = Arc::new(TracingMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(tracing.clone());

    stack.run_before_agent(&mut ctx, &()).await.unwrap();
    let mut request = ModelRequest::default();
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();
    let mut response = ok_response();
    stack
        .run_after_model(&mut ctx, &(), &mut response)
        .await
        .unwrap();
    let mut call = tool_call("t");
    stack
        .run_before_tool(&mut ctx, &(), &mut call)
        .await
        .unwrap();
    let mut result = ToolResult {
        call_id: "c".to_string(),
        name: "t".to_string(),
        content: String::new(),
        raw: None,
        error: None,
        elapsed_ms: 0,
    };
    stack
        .run_after_tool(&mut ctx, &(), &mut result)
        .await
        .unwrap();
    let mut run = crate::harness::middleware::AgentRun::new();
    stack
        .run_after_agent(&mut ctx, &(), &mut run)
        .await
        .unwrap();

    let counts = tracing.counts();
    assert_eq!(counts.agent, 1);
    assert_eq!(counts.model, 1);
    assert_eq!(counts.tool, 1);

    let records = tracing.records();
    assert_eq!(records.first().unwrap().phase, "agent");
    assert_eq!(records.first().unwrap().boundary, TraceBoundary::Begin);
    assert_eq!(records.last().unwrap().phase, "agent");
    assert_eq!(records.last().unwrap().boundary, TraceBoundary::End);
}
