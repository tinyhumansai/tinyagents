//! End-to-end and integration coverage for
//! [`BudgetMiddleware`][tinyagents::harness::middleware::BudgetMiddleware].
//!
//! These tests exercise the four core behaviors of the budget middleware:
//!
//! 1. **Token enforcement in a live harness run** — a multi-call agent loop
//!    accumulates token usage across model calls until the preflight (`before_model`)
//!    trips and fails the run with
//!    [`TinyAgentsError::LimitExceeded`][tinyagents::TinyAgentsError::LimitExceeded],
//!    emitting both a `BudgetWarning` and a `BudgetExceeded` event along the way.
//! 2. **Shared-tracker rollup** — a single [`BudgetTracker`] shared by two
//!    middleware instances (simulating a parent + sub-agent run tree) accumulates
//!    spend across two separate harness runs and eventually blocks.
//! 3. **Cost pricing + enforcement** — driven at the middleware level through a
//!    [`MiddlewareStack`] because the harness does not stamp `resolved_model` on
//!    scripted responses; pricing turns usage into cost and the money budget is
//!    enforced by the preflight.
//! 4. **Below-threshold** — a small run under budget completes normally and emits
//!    no `BudgetExceeded` event.
//!
//! All assertions use structural signals (events, tracker snapshots, run counts)
//! rather than model prose.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use tinyagents::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::AgentEvent;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::middleware::{
    BudgetLimits, BudgetMiddleware, BudgetTracker, MiddlewareStack,
};
use tinyagents::harness::model::{
    ModelRequest, ModelResolutionSource, ModelResponse, ResolvedModel,
};
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, FakeTool};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;
use tinyagents::registry::catalog::ModelPricing;

// ── Helpers (copied verbatim per the task brief) ──────────────────────────────

/// A model response that requests a single tool call, carrying the given usage.
fn tool_call_response(id: &str, name: &str, input: u64, output: u64) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, json!({}))],
            usage: Some(Usage::new(input, output)),
        },
        usage: Some(Usage::new(input, output)),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
    }
}

/// A final text response carrying the given usage.
fn text_response(text: &str, input: u64, output: u64) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(input, output)),
        },
        usage: Some(Usage::new(input, output)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
    }
}

/// True when the recorder captured at least one event matching `pred`.
fn any_event(recorder: &EventRecorder, pred: impl Fn(&AgentEvent) -> bool) -> bool {
    recorder.events().iter().any(pred)
}

// ── Test 1: token budget stops a multi-call harness run ───────────────────────

/// A live multi-call run accumulates token usage until the budget preflight
/// blocks it, and the recorder observes the warn-then-exceed progression.
///
/// Each scripted model response requests the `noop` tool (usage `(8, 0)`), so the
/// loop keeps issuing model calls. With `max_total_tokens = 20`:
///
/// - preflight call 1 (spend 0) admits → records 8
/// - preflight call 2 (spend 8) admits → records 16 (crosses `0.5 * 20 = 10` warn)
/// - preflight call 3 (spend 16) admits → records 24 (crosses 20 → exceeded, unblocked)
/// - preflight call 4 (spend 24) **blocks** with `LimitExceeded`
#[tokio::test]
async fn token_budget_blocks_multi_call_run() {
    let recorder = EventRecorder::new();

    let model = MockModel::with_responses(vec![
        tool_call_response("c1", "noop", 8, 0),
        tool_call_response("c2", "noop", 8, 0),
        tool_call_response("c3", "noop", 8, 0),
    ]);

    let limits = BudgetLimits {
        max_total_tokens: Some(20),
        warn_fraction: Some(0.5),
        ..Default::default()
    };
    // Capture the shared tracker before moving the middleware into the stack.
    let mw = BudgetMiddleware::new(limits);
    let tracker = mw.tracker();

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(model))
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("noop", "ok")))
        .push_middleware(Arc::new(mw));

    let ctx = RunContext::new(RunConfig::new("budget-tokens"), ()).with_events(recorder.sink());
    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect_err("the accumulated token budget must block the run");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );

    // The warn threshold (10) and the exceed threshold (20) were both crossed.
    assert!(
        any_event(&recorder, |e| matches!(e, AgentEvent::BudgetWarning { .. })),
        "a BudgetWarning must be emitted as usage crosses warn_fraction"
    );
    assert!(
        any_event(&recorder, |e| matches!(
            e,
            AgentEvent::BudgetExceeded { blocked: true, .. }
        )),
        "the blocking preflight must emit BudgetExceeded {{ blocked: true }}"
    );

    // Three model calls recorded 8 tokens each before the fourth preflight blocked.
    assert_eq!(
        tracker.snapshot().usage.usage.effective_total(),
        24,
        "three recorded calls of 8 tokens each"
    );
    assert_eq!(
        tracker.snapshot().usage.calls,
        3,
        "three calls were recorded"
    );
}

// ── Test 2: shared-tracker rollup across two middleware instances ─────────────

/// One [`BudgetTracker`] shared by two middleware instances (a parent + a child)
/// accumulates spend across two independent harness runs, and the second run
/// blocks once the shared cumulative usage crosses the limit.
#[tokio::test]
async fn shared_tracker_rolls_up_and_blocks_across_runs() {
    let tracker = BudgetTracker::new();
    let limits = BudgetLimits {
        max_total_tokens: Some(30),
        ..Default::default()
    };

    // ── First harness: completes under budget, leaving 16 tokens on the tracker.
    let model_a = MockModel::with_responses(vec![
        tool_call_response("a1", "noop", 8, 0),
        text_response("done", 8, 0),
    ]);
    let mut harness_a: AgentHarness<()> = AgentHarness::new();
    harness_a
        .register_model("mock", Arc::new(model_a))
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("noop", "ok")))
        .push_middleware(Arc::new(
            BudgetMiddleware::new(limits).with_tracker(tracker.clone()),
        ));

    let run_a = harness_a
        .invoke_in_context(
            &(),
            RunContext::new(RunConfig::new("budget-parent"), ()),
            vec![Message::user("parent")],
        )
        .await
        .expect("the first run stays under the shared budget");
    assert!(run_a.final_response.is_some(), "first run completed");
    assert_eq!(
        tracker.snapshot().usage.usage.effective_total(),
        16,
        "the shared tracker holds the first run's spend"
    );

    // ── Second harness: shares the same tracker (already at 16) and pushes over.
    // preflight (16) admits → 24; preflight (24) admits → 32; preflight (32) blocks.
    let model_b = MockModel::with_responses(vec![
        tool_call_response("b1", "noop", 8, 0),
        tool_call_response("b2", "noop", 8, 0),
    ]);
    let mut harness_b: AgentHarness<()> = AgentHarness::new();
    harness_b
        .register_model("mock", Arc::new(model_b))
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("noop", "ok")))
        .push_middleware(Arc::new(
            BudgetMiddleware::new(limits).with_tracker(tracker.clone()),
        ));

    let err = harness_b
        .invoke_in_context(
            &(),
            RunContext::new(RunConfig::new("budget-child"), ()),
            vec![Message::user("child")],
        )
        .await
        .expect_err("the shared budget must block the second run");
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );

    // Both runs rolled into the single tracker: 16 (parent) + 16 (child) = 32.
    assert_eq!(
        tracker.snapshot().usage.usage.effective_total(),
        32,
        "spend from both runs accumulates in the shared tracker"
    );
    assert_eq!(
        tracker.snapshot().usage.calls,
        4,
        "four calls total across runs"
    );
}

// ── Test 3: cost pricing + enforcement via direct middleware calls ────────────

/// Cost enforcement is exercised at the middleware level: `after_model` prices a
/// response carrying a `resolved_model`, and a later preflight blocks once the
/// accumulated cost meets the money budget.
///
/// This must be driven directly through a [`MiddlewareStack`] because the harness
/// does not stamp `resolved_model` onto scripted responses, so the pricing table
/// would never match.
#[tokio::test]
async fn cost_pricing_records_and_enforces_money_budget() {
    let recorder = EventRecorder::new();
    let mut ctx = RunContext::new(RunConfig::new("budget-cost"), ()).with_events(recorder.sink());

    let mut pricing: HashMap<String, ModelPricing> = HashMap::new();
    pricing.insert(
        "m".to_string(),
        ModelPricing {
            input_per_token: Some(1.0),
            output_per_token: Some(1.0),
            ..Default::default()
        },
    );

    let mw = BudgetMiddleware::new(BudgetLimits {
        max_cost: Some(5.0),
        ..Default::default()
    })
    .with_pricing(pricing);
    let tracker = mw.tracker();

    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    // 4 input + 2 output at 1.0/token = 6.0 cost, which meets the 5.0 budget.
    let mut resp = ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("priced".into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, 2)),
        },
        usage: Some(Usage::new(4, 2)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: Some(ResolvedModel {
            name: "m".into(),
            requested: None,
            source: ModelResolutionSource::RegistryDefault,
        }),
    };

    stack
        .run_after_model(&mut ctx, &(), &mut resp)
        .await
        .expect("recording spend does not fail");

    // Cost was priced and folded into the shared tracker.
    assert!(
        (tracker.snapshot().cost.total_cost - 6.0).abs() < 1e-9,
        "4+2 tokens at 1.0/token should cost 6.0, got {}",
        tracker.snapshot().cost.total_cost
    );
    assert!(
        any_event(&recorder, |e| matches!(e, AgentEvent::CostRecorded { .. })),
        "a CostRecorded event must be emitted when priced cost is positive"
    );
    assert!(
        any_event(&recorder, |e| matches!(e, AgentEvent::UsageRecorded { .. })),
        "a UsageRecorded event accompanies the recorded usage"
    );

    // The next preflight blocks because 6.0 >= the 5.0 cost budget.
    let mut req = ModelRequest::new(vec![Message::user("go")]);
    let err = stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect_err("the cost budget must block the next model call");
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );
    assert!(
        any_event(&recorder, |e| matches!(
            e,
            AgentEvent::BudgetExceeded { blocked: true, .. }
        )),
        "the blocking preflight emits BudgetExceeded {{ blocked: true }}"
    );
}

// ── Test 4: below-threshold run completes cleanly ─────────────────────────────

/// A small run comfortably under the budget completes normally and emits no
/// `BudgetExceeded` event.
#[tokio::test]
async fn below_threshold_run_completes_without_exceeding() {
    let recorder = EventRecorder::new();

    let model = MockModel::with_responses(vec![
        tool_call_response("s1", "noop", 5, 0),
        text_response("all good", 5, 0),
    ]);

    let mw = BudgetMiddleware::new(BudgetLimits {
        max_total_tokens: Some(100),
        warn_fraction: Some(0.9),
        ..Default::default()
    });
    let tracker = mw.tracker();

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(model))
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("noop", "ok")))
        .push_middleware(Arc::new(mw));

    let ctx = RunContext::new(RunConfig::new("budget-under"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a run under budget completes normally");

    assert!(
        run.final_response.is_some(),
        "the run produced a final response"
    );
    assert_eq!(run.model_calls, 2, "two model turns completed");
    assert_eq!(run.tool_calls, 1, "the noop tool ran once");

    // 10 total tokens, well under the 100 budget and its 90-token warn threshold.
    assert_eq!(tracker.snapshot().usage.usage.effective_total(), 10);
    assert!(
        !any_event(&recorder, |e| matches!(
            e,
            AgentEvent::BudgetExceeded { .. }
        )),
        "no BudgetExceeded event should be emitted under budget"
    );
    assert!(
        !any_event(&recorder, |e| matches!(e, AgentEvent::BudgetWarning { .. })),
        "no BudgetWarning event should be emitted well under the warn threshold"
    );
}

// ── Test 5: input-token reservation blocks an oversized call in a live run ─────

/// With `max_input_tokens` set, the preflight estimates the request's input
/// tokens and reserves against the budget. A prompt that estimates over the
/// budget is blocked *before* any model call is dispatched, failing the run
/// with `LimitExceeded` and emitting `BudgetExceeded { blocked: true }`.
#[tokio::test]
async fn input_reservation_blocks_oversized_call_in_a_live_run() {
    let recorder = EventRecorder::new();

    let mw = BudgetMiddleware::new(BudgetLimits {
        max_input_tokens: Some(5),
        ..Default::default()
    });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(MockModel::constant("unused")))
        .set_default_model("mock")
        .push_middleware(Arc::new(mw));

    // A long prompt estimates well over the 5-token input reservation budget.
    let big = "word ".repeat(200);
    let ctx = RunContext::new(RunConfig::new("reserve-block"), ()).with_events(recorder.sink());
    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user(big)])
        .await
        .expect_err("the reservation preflight must block an oversized prompt");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );
    assert!(
        any_event(&recorder, |e| matches!(
            e,
            AgentEvent::BudgetExceeded { blocked: true, .. }
        )),
        "the blocking reservation preflight must emit BudgetExceeded {{ blocked: true }}"
    );
}

// ── Test 6: a fitting call reserves then reconciles against actual usage ───────

/// A run whose prompt fits the input reservation budget reserves input tokens
/// on the preflight (`BudgetReserved`) and reconciles against the actual
/// reported usage after the model responds (`BudgetReconciled`).
#[tokio::test]
async fn fitting_call_emits_reserved_and_reconciled_events() {
    let recorder = EventRecorder::new();

    let mw = BudgetMiddleware::new(BudgetLimits {
        max_input_tokens: Some(1_000),
        ..Default::default()
    });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![text_response("done", 3, 1)])),
        )
        .set_default_model("mock")
        .push_middleware(Arc::new(mw));

    let ctx = RunContext::new(RunConfig::new("reserve-ok"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("a small prompt fits the reservation and completes");
    assert!(run.final_response.is_some(), "the run produced a response");

    assert!(
        any_event(&recorder, |e| matches!(
            e,
            AgentEvent::BudgetReserved { .. }
        )),
        "the preflight must emit BudgetReserved when max_input_tokens is set"
    );
    assert!(
        any_event(&recorder, |e| matches!(
            e,
            AgentEvent::BudgetReconciled {
                actual_input_tokens: 3,
                ..
            }
        )),
        "after_model must reconcile the reservation against the actual 3 input tokens"
    );
}

// ── Test 7: cached-input token budget enforcement via the middleware stack ─────

/// The `max_cached_input_tokens` budget tracks `cache_read_tokens` reported in
/// usage and blocks the next preflight once the cached-input budget is
/// exhausted. Driven directly through a [`MiddlewareStack`] so the response can
/// carry a non-zero `cache_read_tokens` count.
#[tokio::test]
async fn cached_input_budget_blocks_next_call() {
    let recorder = EventRecorder::new();
    let mut ctx = RunContext::new(RunConfig::new("cached-budget"), ()).with_events(recorder.sink());

    let mw = BudgetMiddleware::new(BudgetLimits {
        max_cached_input_tokens: Some(10),
        ..Default::default()
    });
    let tracker = mw.tracker();

    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    // A response reporting 12 cache-read tokens exhausts the 10-token cached budget.
    let mut resp = ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("cached".into())],
            tool_calls: Vec::new(),
            usage: Some(Usage {
                cache_read_tokens: 12,
                ..Usage::new(2, 1)
            }),
        },
        usage: Some(Usage {
            cache_read_tokens: 12,
            ..Usage::new(2, 1)
        }),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
    };
    stack
        .run_after_model(&mut ctx, &(), &mut resp)
        .await
        .expect("recording cached usage does not fail");

    assert_eq!(
        tracker.snapshot().usage.usage.cache_read_tokens,
        12,
        "the tracker accumulates the reported cache-read tokens"
    );

    // The next preflight blocks because 12 >= the 10-token cached-input budget.
    let mut req = ModelRequest::new(vec![Message::user("next")]);
    let err = stack
        .run_before_model(&mut ctx, &(), &mut req)
        .await
        .expect_err("the cached-input budget must block the next model call");
    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded, got {err:?}"
    );
    assert!(
        any_event(&recorder, |e| matches!(
            e,
            AgentEvent::BudgetExceeded { blocked: true, .. }
        )),
        "the blocking preflight emits BudgetExceeded {{ blocked: true }}"
    );
}
