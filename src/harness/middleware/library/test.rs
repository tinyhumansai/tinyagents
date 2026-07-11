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
        invalid: None,
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

#[tokio::test(start_paused = true)]
async fn retry_middleware_sleeps_the_documented_backoff_schedule() {
    // Regression test: the middleware used to compute the backoff from the
    // *post-increment* attempt number, so the first retry's sleep skipped
    // `initial_backoff_ms` entirely and the whole exponential schedule was
    // shifted one step higher than `RetryPolicy::backoff_for_attempt`
    // documents. With `initial_backoff_ms = 100`, `multiplier = 2.0`, no
    // jitter: attempt 0 -> 100ms, attempt 1 -> 200ms.
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(RetryMiddleware::new(
        RetryPolicy::default()
            .with_max_attempts(3)
            .with_initial_backoff_ms(100)
            .with_multiplier(2.0)
            .with_jitter(false)
            .with_backoff_sleep(true),
    )));

    let timestamps = Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = timestamps.clone();
    let base = FakeModelBase::new(move |_n, _req| {
        recorded.lock().unwrap().push(tokio::time::Instant::now());
        Err(TinyAgentsError::Model("transient".to_string()))
    });

    stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect_err("all 3 attempts fail");

    let timestamps = timestamps.lock().unwrap().clone();
    assert_eq!(timestamps.len(), 3, "expected exactly max_attempts calls");
    let gaps: Vec<Duration> = timestamps
        .windows(2)
        .map(|w| w[1].duration_since(w[0]))
        .collect();
    assert_eq!(
        gaps,
        vec![Duration::from_millis(100), Duration::from_millis(200)],
        "backoff schedule does not match RetryPolicy::backoff_for_attempt"
    );
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
async fn model_fallback_does_not_switch_on_non_retryable() {
    let (mut ctx, recorder) = ctx_with_recorder();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(ModelFallbackMiddleware::new([
        "backup-a", "backup-b",
    ])));

    // A non-retryable error (validation) will fail identically on every model,
    // so the middleware must not burn quota switching backends.
    let base =
        FakeModelBase::new(|_n, _req| Err(TinyAgentsError::Validation("bad input".to_string())));
    let err = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &base)
        .await
        .expect_err("validation errors are not retryable");
    assert!(matches!(err, TinyAgentsError::Validation(_)));
    // Only the primary call happens; no fallback attempts.
    assert_eq!(base.calls(), 1);
    let selections = events(&recorder)
        .into_iter()
        .filter(|e| matches!(e, AgentEvent::FallbackSelected { .. }))
        .count();
    assert_eq!(selections, 0);
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

    // `waited_ms` reports the actual wall-clock wait per the injected clock:
    // the bucket was empty at `base` and refilled at `base + 1s`.
    assert!(
        events(&recorder)
            .iter()
            .any(|e| matches!(e, AgentEvent::RateLimitWaited { waited_ms } if *waited_ms == 1_000))
    );
}

/// A `Wait` gate over a bucket that never refills must fail fast instead of
/// livelocking in the poll loop forever.
#[tokio::test]
async fn rate_limit_wait_errors_when_bucket_can_never_refill() {
    let (mut ctx, recorder) = ctx_with_recorder();

    // Drain a zero-refill bucket, then gate a call on it.
    let base_instant = Instant::now();
    let limiter = Arc::new(RateLimiter::new(1, 0.0));
    assert!(limiter.try_acquire(1, base_instant));

    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(
        RateLimitMiddleware::new(limiter).waiting(Duration::ZERO),
    ));

    let model_base = FakeModelBase::new(|_n, _req| Ok(ok_response()));
    let err = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &model_base)
        .await
        .expect_err("a never-refilling bucket must fail fast, not spin");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)), "{err:?}");
    assert!(err.to_string().contains("can never succeed"), "{err}");
    assert_eq!(model_base.calls(), 0);
    assert!(
        !events(&recorder)
            .iter()
            .any(|e| matches!(e, AgentEvent::RateLimitWaited { .. })),
        "no wait event when the call is rejected"
    );
}

/// A `Wait` gate requesting more tokens than the bucket capacity must also
/// fail fast: no amount of refill can ever admit the call.
#[tokio::test]
async fn rate_limit_wait_errors_when_tokens_exceed_capacity() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let limiter = Arc::new(RateLimiter::new(1, 10.0));

    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(
        RateLimitMiddleware::new(limiter)
            .with_tokens(2)
            .waiting(Duration::ZERO),
    ));

    let model_base = FakeModelBase::new(|_n, _req| Ok(ok_response()));
    let err = stack
        .run_wrapped_model(&mut ctx, &(), ModelRequest::default(), &model_base)
        .await
        .expect_err("requesting more tokens than capacity must fail fast");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)), "{err:?}");
    assert_eq!(model_base.calls(), 0);
}

// ── ContextualToolSelectionMiddleware ───────────────────────────────────────

fn schema_named(name: &str) -> ToolSchema {
    ToolSchema {
        name: name.to_string(),
        description: String::new(),
        parameters: json!({}),
        format: crate::harness::tool::ToolFormat::Json,
    }
}

#[tokio::test]
async fn contextual_selection_from_lists_denies_and_fails_closed() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    // allow=[a,b], deny=[b] -> only `a` survives; unknown `c` is fail-closed out.
    let mw = ContextualToolSelectionMiddleware::from_lists(Some(["a", "b"]), ["b"]);
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut request = ModelRequest::new(Vec::new()).with_tools(vec![
        schema_named("a"),
        schema_named("b"),
        schema_named("c"),
    ]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "a");
}

#[tokio::test]
async fn contextual_selection_emits_exposure_event() {
    let (mut ctx, recorder) = ctx_with_recorder();
    let mw = ContextualToolSelectionMiddleware::from_lists(Some(["a"]), ["b"]);
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut request = ModelRequest::new(Vec::new()).with_tools(vec![
        schema_named("a"),
        schema_named("b"),
        schema_named("c"),
    ]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();

    let filtered = recorder.events().into_iter().find_map(|r| match r.event {
        AgentEvent::ToolsFiltered {
            excluded,
            remaining,
            ..
        } => Some((excluded, remaining)),
        _ => None,
    });
    let (excluded, remaining) = filtered.expect("an exposure event should be emitted");
    assert_eq!(remaining, 1);
    assert_eq!(excluded, vec!["b".to_string(), "c".to_string()]);
}

#[tokio::test]
async fn contextual_selection_inheriting_narrows_never_widens() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    // Parent allows [a,b,c]; child tries to allow [b,c,d] and un-deny nothing.
    // Parent denies [c]. Effective: allow = {a,b,c} ∩ {b,c,d} = {b,c}; deny adds
    // parent's c -> {c}. So only `b` survives; `d` (never parent-allowed) and
    // `c` (parent-denied) are withheld.
    let mw = ContextualToolSelectionMiddleware::inheriting(
        Some(["a", "b", "c"]),
        ["c"],
        Some(["b", "c", "d"]),
        Vec::<String>::new(),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut request = ModelRequest::new(Vec::new()).with_tools(vec![
        schema_named("a"),
        schema_named("b"),
        schema_named("c"),
        schema_named("d"),
    ]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();
    let names: Vec<_> = request.tools.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["b"]);
}

#[tokio::test]
async fn contextual_selection_varies_by_depth() {
    use crate::harness::context::{RunConfig, RunContext};
    // Sub-agent depth (>0) hides the privileged tool.
    let ctx_data = RunContext::new(RunConfig::new("deep").with_depth(2), ());
    let mut ctx = ctx_data;
    let mw = ContextualToolSelectionMiddleware::new(Arc::new(|schema: &ToolSchema, sel| {
        schema.name != "privileged" || sel.depth == 0
    }));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut request = ModelRequest::new(Vec::new())
        .with_tools(vec![schema_named("safe"), schema_named("privileged")]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "safe");
}

// ── BudgetMiddleware ────────────────────────────────────────────────────────

fn response_with_usage(model: &str, input: u64, output: u64) -> ModelResponse {
    use crate::harness::model::{ModelResolutionSource, ResolvedModel};
    use crate::harness::usage::Usage;
    let mut response = ModelResponse::assistant("ok");
    response.usage = Some(Usage::new(input, output));
    response.resolved_model = Some(ResolvedModel {
        name: model.to_string(),
        requested: None,
        source: ModelResolutionSource::RegistryDefault,
    });
    response
}

#[tokio::test]
async fn budget_warns_then_blocks_on_token_exhaustion() {
    let (mut ctx, recorder) = ctx_with_recorder();
    let mw = BudgetMiddleware::new(BudgetLimits {
        max_total_tokens: Some(10),
        warn_fraction: Some(0.5),
        ..BudgetLimits::default()
    });
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    // First spend: 8 tokens crosses the 0.5*10=5 warning threshold, not exceeded.
    let mut resp = response_with_usage("m", 5, 3);
    stack
        .run_after_model(&mut ctx, &(), &mut resp)
        .await
        .unwrap();
    assert!(
        events(&recorder)
            .iter()
            .any(|e| matches!(e, AgentEvent::BudgetWarning { .. }))
    );

    // Preflight still admits the next call (8 < 10).
    let mut req = ModelRequest::new(vec![Message::user("go")]);
    stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .unwrap();

    // Second spend pushes cumulative tokens to 13 (>= 10): exceeded event.
    let mut resp2 = response_with_usage("m", 3, 2);
    stack
        .run_after_model(&mut ctx, &(), &mut resp2)
        .await
        .unwrap();
    assert!(
        events(&recorder)
            .iter()
            .any(|e| matches!(e, AgentEvent::BudgetExceeded { blocked: false, .. }))
    );

    // Now preflight fails closed.
    let err = stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect_err("budget exhausted should block");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)));
}

#[tokio::test]
async fn budget_prices_usage_and_enforces_cost() {
    use crate::registry::catalog::ModelPricing;
    let (mut ctx, recorder) = ctx_with_recorder();
    let mut pricing = std::collections::HashMap::new();
    pricing.insert(
        "m".to_string(),
        ModelPricing {
            input_per_token: Some(1.0),
            output_per_token: Some(1.0),
            ..ModelPricing::default()
        },
    );
    let mw = BudgetMiddleware::new(BudgetLimits {
        max_cost: Some(5.0),
        ..BudgetLimits::default()
    })
    .with_pricing(pricing);
    let tracker = mw.tracker();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    // 4 in + 2 out at 1.0/token = 6.0 cost >= 5.0 budget.
    let mut resp = response_with_usage("m", 4, 2);
    stack
        .run_after_model(&mut ctx, &(), &mut resp)
        .await
        .unwrap();
    assert!((tracker.snapshot().cost.total_cost - 6.0).abs() < 1e-9);
    assert!(
        events(&recorder)
            .iter()
            .any(|e| matches!(e, AgentEvent::CostRecorded { .. }))
    );

    let mut req = ModelRequest::new(vec![Message::user("go")]);
    let err = stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect_err("cost budget exhausted should block");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)));
}

#[tokio::test]
async fn budget_enforces_cached_input_token_limit() {
    use crate::harness::usage::Usage;
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mw = BudgetMiddleware::new(BudgetLimits {
        max_cached_input_tokens: Some(10),
        ..BudgetLimits::default()
    });
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    // A response reporting 12 cache-read tokens exhausts the cached budget.
    let mut resp = response_with_usage("m", 2, 1);
    if let Some(u) = resp.usage.as_mut() {
        *u = Usage {
            cache_read_tokens: 12,
            ..*u
        };
    }
    stack
        .run_after_model(&mut ctx, &(), &mut resp)
        .await
        .unwrap();

    let mut req = ModelRequest::new(vec![Message::user("next")]);
    let err = stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect_err("cached input budget exhausted should block");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)));
}

#[tokio::test]
async fn budget_preflight_reserves_and_reconciles() {
    let (mut ctx, recorder) = ctx_with_recorder();
    // A tiny input budget so a large prompt trips the reservation preflight.
    let mw = BudgetMiddleware::new(BudgetLimits {
        max_input_tokens: Some(5),
        ..BudgetLimits::default()
    });
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    // A long prompt estimates well over 5 input tokens -> reservation blocks it
    // before any call is dispatched.
    let big = "word ".repeat(200);
    let mut req = ModelRequest::new(vec![Message::user(big)]);
    let err = stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect_err("reservation should block an oversized call");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)));
    assert!(
        events(&recorder)
            .iter()
            .any(|e| matches!(e, AgentEvent::BudgetExceeded { blocked: true, .. }))
    );

    // A small prompt reserves, then reconciles against the actual usage.
    let mut small = ModelRequest::new(vec![Message::user("hi")]);
    stack
        .run_before_model(&mut ctx, &(), &mut small)
        .await
        .expect("small call fits the reservation");
    let mut resp = response_with_usage("m", 1, 1);
    stack
        .run_after_model(&mut ctx, &(), &mut resp)
        .await
        .unwrap();

    let evs = events(&recorder);
    assert!(
        evs.iter()
            .any(|e| matches!(e, AgentEvent::BudgetReserved { .. }))
    );
    assert!(evs.iter().any(
        |e| matches!(e, AgentEvent::BudgetReconciled { actual_input_tokens, .. } if *actual_input_tokens == 1)
    ));
}

#[tokio::test]
async fn reservation_is_released_even_when_response_carries_no_usage() {
    // A call whose response never reports `usage` (provider error, dropped
    // stream, etc.) must still release its preflight reservation in
    // `after_model` — otherwise the reservation leaks forever and
    // permanently starves later calls on this (or a shared) tracker.
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mw = BudgetMiddleware::new(BudgetLimits {
        max_input_tokens: Some(5),
        ..BudgetLimits::default()
    });
    let tracker = mw.tracker();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut req = ModelRequest::new(vec![Message::user("hi")]);
    stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect("small call fits the reservation");
    assert!(tracker.snapshot().reserved_input_total > 0);

    let mut resp = ModelResponse::assistant("ok"); // usage: None
    stack
        .run_after_model(&mut ctx, &(), &mut resp)
        .await
        .unwrap();
    assert_eq!(tracker.snapshot().reserved_input_total, 0);
}

#[tokio::test]
async fn reservation_is_released_when_model_call_errors() {
    // A model call that ultimately errors (retries/fallback exhausted, hard
    // provider error, timeout, ...) never reaches `after_model` — the wrap
    // onion returns `Err` and the harness propagates it with `?` before the
    // lifecycle `after_model` hook runs. The reservation `before_model` added
    // must still be released (via `on_error`), or it leaks forever and
    // permanently starves every later call on a shared tracker.
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mw = BudgetMiddleware::new(BudgetLimits {
        max_input_tokens: Some(5),
        ..BudgetLimits::default()
    });
    let tracker = mw.tracker();
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut req = ModelRequest::new(vec![Message::user("hi")]);
    stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect("small call fits the reservation");
    assert!(tracker.snapshot().reserved_input_total > 0);

    // Simulate the model call failing before `after_model` ever runs: the
    // harness surfaces the failure to every middleware via `on_error`
    // instead.
    let error = TinyAgentsError::Model("provider down".to_string());
    stack.run_on_error(&mut ctx, &error).await.unwrap();

    assert_eq!(tracker.snapshot().reserved_input_total, 0);

    // The tracker must have recovered: a fresh call reserving the same
    // amount should not be rejected as if capacity were still consumed.
    let mut req2 = ModelRequest::new(vec![Message::user("hi")]);
    stack
        .run_before_model(&mut ctx, &(), &mut req2)
        .await
        .expect("reservation was released so a new call fits the budget again");
}

#[test]
fn poisoned_tracker_stays_fail_closed() {
    // A poisoned mutex still holds a valid last-written spend value; a
    // `snapshot()` that defaulted to zero on poison would make every
    // subsequent budget check see an empty budget and admit calls forever
    // (fail-open). Recovering the poisoned guard must keep the previously
    // accumulated spend intact.
    use crate::harness::cost::CostTotals;
    use crate::harness::usage::Usage;

    let tracker = BudgetTracker::new();
    tracker.record(Usage::new(100, 100), CostTotals::default());
    assert!(!tracker.inner.is_poisoned());

    let poison_tracker = tracker.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poison_tracker.inner.lock().unwrap();
        panic!("simulated panic while holding the tracker lock");
    })
    .join();
    assert!(tracker.inner.is_poisoned());

    // The accumulated spend from before the panic must still be visible,
    // not silently reset to zero.
    let spend = tracker.snapshot();
    assert_eq!(spend.usage.usage.input_tokens, 100);
    assert_eq!(spend.usage.usage.output_tokens, 100);

    let limits = BudgetLimits {
        max_total_tokens: Some(200),
        ..BudgetLimits::default()
    };
    assert!(limits.exceeded_reason(&spend).is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn shared_tracker_reservation_is_atomic_under_concurrency() {
    // N concurrent "runs" (their own `BudgetMiddleware`, sharing one
    // `BudgetTracker` — the documented sub-agent pattern) all reserve the
    // same input-token estimate at once. A racy check-then-act preflight
    // would let every one of them observe spare capacity and reserve past
    // it; an atomic check-and-reserve admits only as many as the budget
    // actually allows.
    use crate::harness::middleware::Middleware;

    let tracker = BudgetTracker::new();
    let per_call_tokens = 10u64; // "x" * 40 chars / 4 == 10 estimated tokens.
    let concurrent_capacity = 4u64;
    let attempts = 10usize;
    let limits = BudgetLimits {
        max_input_tokens: Some(per_call_tokens * concurrent_capacity),
        ..BudgetLimits::default()
    };

    let barrier = Arc::new(tokio::sync::Barrier::new(attempts));
    let mut handles = Vec::new();
    for _ in 0..attempts {
        let mw = BudgetMiddleware::new(limits).with_tracker(tracker.clone());
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            let mut ctx: RunContext = RunContext::new(RunConfig::new("test-run"), ());
            let mut request = ModelRequest::new(vec![Message::user("x".repeat(40))]);
            barrier.wait().await;
            mw.before_model(&mut ctx, &(), &mut request).await
        }));
    }

    let mut ok_count = 0usize;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            ok_count += 1;
        }
    }

    // Exactly the number of reservations the budget can hold succeed; the
    // rest are rejected before dispatch. A racy implementation would let
    // more than `concurrent_capacity` through.
    assert_eq!(ok_count, concurrent_capacity as usize);
    assert_eq!(
        tracker.snapshot().reserved_input_total,
        per_call_tokens * concurrent_capacity
    );
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
        format: crate::harness::tool::ToolFormat::Json,
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

// ── ToolPolicyMiddleware ────────────────────────────────────────────────────

#[tokio::test]
async fn tool_policy_strict_hides_and_rejects_unclassified() {
    use crate::harness::tool::ToolPolicy;
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut policies = std::collections::HashMap::new();
    policies.insert("safe".to_string(), ToolPolicy::read_only());
    // "risky" is present but unclassified.
    policies.insert("risky".to_string(), ToolPolicy::default());

    let mw = ToolPolicyMiddleware::strict(policies);
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let schema = |name: &str| ToolSchema {
        name: name.to_string(),
        description: String::new(),
        parameters: json!({}),
        format: crate::harness::tool::ToolFormat::Json,
    };
    let mut request = ModelRequest::new(Vec::new()).with_tools(vec![
        schema("safe"),
        schema("risky"),
        schema("unknown"),
    ]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .expect("exposure filter runs");
    // Only the classified read-only tool survives exposure.
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "safe");

    // Execution guard agrees with exposure.
    let mut safe = tool_call("safe");
    stack
        .run_before_tool(&mut ctx, &(), &mut safe)
        .await
        .expect("classified tool admitted");
    let mut risky = tool_call("risky");
    let err = stack
        .run_before_tool(&mut ctx, &(), &mut risky)
        .await
        .expect_err("unclassified tool rejected");
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[tokio::test]
async fn tool_policy_denies_declared_side_effect() {
    use crate::harness::tool::{ToolPolicy, ToolSideEffects};
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut policies = std::collections::HashMap::new();
    policies.insert(
        "charge".to_string(),
        ToolPolicy::classified().with_side_effects(ToolSideEffects {
            payment: true,
            ..ToolSideEffects::default()
        }),
    );
    let mw = ToolPolicyMiddleware::new(policies).deny_side_effects(ToolSideEffects {
        payment: true,
        ..ToolSideEffects::default()
    });
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut call = tool_call("charge");
    let err = stack
        .run_before_tool(&mut ctx, &(), &mut call)
        .await
        .expect_err("payment tool denied");
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[tokio::test]
async fn tool_policy_blocks_unapproved_approval_required_tool() {
    use crate::harness::tool::{ToolAccess, ToolPolicy};
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut policies = std::collections::HashMap::new();
    policies.insert(
        "deploy".to_string(),
        ToolPolicy::classified().with_access(ToolAccess {
            approval_required: true,
            ..ToolAccess::default()
        }),
    );
    // Approval is required but only "other" is approved -> deploy is blocked.
    let mw = ToolPolicyMiddleware::new(policies).require_approval(["other"]);
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut call = tool_call("deploy");
    let err = stack
        .run_before_tool(&mut ctx, &(), &mut call)
        .await
        .expect_err("approval-required tool blocked without grant");
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[tokio::test]
async fn tool_policy_requires_sandbox_for_sandboxed_tool() {
    use crate::harness::context::{RunConfig, RunContext};
    use crate::harness::tool::{SandboxMode, ToolPolicy, ToolRuntime};
    use crate::harness::workspace::WorkspaceDescriptor;

    let mut policies = std::collections::HashMap::new();
    policies.insert(
        "shell".to_string(),
        ToolPolicy::classified().with_runtime(ToolRuntime {
            sandbox: SandboxMode::Required,
            ..ToolRuntime::default()
        }),
    );
    let mw = Arc::new(ToolPolicyMiddleware::new(policies).require_sandbox(true));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw);

    // No workspace -> the sandboxed tool is blocked (fail closed).
    let mut bare: RunContext = RunContext::new(RunConfig::new("no-sandbox"), ());
    let mut call = tool_call("shell");
    assert!(
        stack
            .run_before_tool(&mut bare, &(), &mut call)
            .await
            .is_err()
    );

    // A sandboxed workspace satisfies the requirement.
    let mut sandboxed: RunContext = RunContext::new(RunConfig::new("sandboxed"), ())
        .with_workspace(WorkspaceDescriptor::new("/work").with_sandbox(SandboxMode::Required));
    let mut call = tool_call("shell");
    stack
        .run_before_tool(&mut sandboxed, &(), &mut call)
        .await
        .expect("sandboxed run admits the tool");
}

#[tokio::test]
async fn tool_policy_truncates_oversized_results() {
    use crate::harness::tool::{ToolPolicy, ToolResult, ToolRuntime};
    let (mut ctx, _recorder) = ctx_with_recorder();
    let mut policies = std::collections::HashMap::new();
    policies.insert(
        "reader".to_string(),
        ToolPolicy::classified().with_runtime(ToolRuntime {
            max_result_bytes: Some(4),
            ..ToolRuntime::default()
        }),
    );
    let mw = Arc::new(ToolPolicyMiddleware::new(policies).enforce_result_bytes(true));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw);

    let mut result = ToolResult {
        call_id: "c1".into(),
        name: "reader".into(),
        content: "abcdefgh".into(),
        raw: None,
        error: None,
        elapsed_ms: 0,
    };
    stack
        .run_after_tool(&mut ctx, &(), &mut result)
        .await
        .expect("after_tool runs");
    assert_eq!(result.content, "abcd");
    assert!(result.error.unwrap().contains("max_result_bytes"));
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

/// Redaction must be idempotent and never self-matching: a pattern that occurs
/// inside the mask text (here `RED` inside `[REDACTED]`) must not corrupt the
/// mask, and re-running the middleware over already-redacted text is a no-op.
#[tokio::test]
async fn redaction_is_idempotent_and_never_matches_inside_mask() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let redaction = Arc::new(RedactionMiddleware::new(["RED", "secret"]));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(redaction.clone());

    let mut response = ModelResponse::assistant("the secret is RED");
    stack
        .run_after_model(&mut ctx, &(), &mut response)
        .await
        .expect("redaction runs");
    assert_eq!(
        response.text(),
        "the [REDACTED] is [REDACTED]",
        "patterns must not match inside mask text introduced by a replacement"
    );
    assert_eq!(redaction.redactions(), 2);

    // Second application over already-redacted text is a no-op.
    stack
        .run_after_model(&mut ctx, &(), &mut response)
        .await
        .expect("redaction runs again");
    assert_eq!(response.text(), "the [REDACTED] is [REDACTED]");
    assert_eq!(redaction.redactions(), 2, "re-run must not re-count");
}

/// Tool-call arguments (before the tool runs and in the model response) and
/// raw payloads must be scrubbed, not just plain text content.
#[tokio::test]
async fn redaction_scrubs_tool_call_arguments_and_raw_payloads() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let redaction = Arc::new(RedactionMiddleware::new(["sk-secret"]));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(redaction.clone());

    // Model-authored tool call arguments and the raw provider payload.
    let mut response = ModelResponse::assistant("ok");
    response.message.tool_calls.push(ToolCall {
        id: "c1".to_string(),
        name: "http".to_string(),
        arguments: json!({"header": "Bearer sk-secret", "nested": ["sk-secret"]}),
        invalid: None,
    });
    response.raw = Some(json!({"choices": [{"text": "token sk-secret"}]}));
    stack
        .run_after_model(&mut ctx, &(), &mut response)
        .await
        .expect("redaction runs");
    assert_eq!(
        response.message.tool_calls[0].arguments,
        json!({"header": "Bearer [REDACTED]", "nested": ["[REDACTED]"]})
    );
    assert_eq!(
        response.raw,
        Some(json!({"choices": [{"text": "token [REDACTED]"}]}))
    );

    // Outbound tool-call arguments via before_tool.
    let mut call = ToolCall {
        id: "c2".to_string(),
        name: "http".to_string(),
        arguments: json!({"key": "sk-secret"}),
        invalid: None,
    };
    stack
        .run_before_tool(&mut ctx, &(), &mut call)
        .await
        .expect("redaction runs before tool");
    assert_eq!(call.arguments, json!({"key": "[REDACTED]"}));

    // Tool result raw payload and error message via after_tool.
    let mut result = ToolResult {
        call_id: "c2".to_string(),
        name: "http".to_string(),
        content: "done".to_string(),
        raw: Some(json!({"echo": "sk-secret"})),
        error: Some("auth failed for sk-secret".to_string()),
        elapsed_ms: 0,
    };
    stack
        .run_after_tool(&mut ctx, &(), &mut result)
        .await
        .expect("redaction runs after tool");
    assert_eq!(result.raw, Some(json!({"echo": "[REDACTED]"})));
    assert_eq!(result.error.as_deref(), Some("auth failed for [REDACTED]"));
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

#[tokio::test]
async fn tracing_records_are_bounded_by_max_records() {
    let (mut ctx, _recorder) = ctx_with_recorder();
    let tracing = Arc::new(TracingMiddleware::new().with_max_records(3));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(tracing.clone());

    // Each `before_agent`/`after_agent` pair pushes 2 records; run enough
    // iterations to far exceed the cap of 3 and confirm memory stays bounded
    // rather than growing without limit.
    for _ in 0..50 {
        stack.run_before_agent(&mut ctx, &()).await.unwrap();
        let mut run = crate::harness::middleware::AgentRun::new();
        stack
            .run_after_agent(&mut ctx, &(), &mut run)
            .await
            .unwrap();
    }

    let records = tracing.records();
    assert_eq!(records.len(), 3);
    // The oldest entries were evicted; only the most recent boundary pair
    // (plus one) survives.
    assert_eq!(records.last().unwrap().phase, "agent");
    assert_eq!(records.last().unwrap().boundary, TraceBoundary::End);

    // Counts are unaffected by the record cap — they track unboundedly by
    // design (a single `usize` per phase, not a growing collection).
    let counts = tracing.counts();
    assert_eq!(counts.agent, 50);
}
