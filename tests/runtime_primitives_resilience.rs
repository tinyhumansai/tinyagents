//! Public-API regression tests for the harness runtime primitives.
//!
//! Each test corresponds to a specific defect and is written so it **fails
//! against the pre-fix code**:
//!
//! - LOOP-2  — enabling jitter zeroed the backoff, so nothing slept.
//! - LOCAL-4 — `backoff_sleep` defaulted off, so retries fired back-to-back.
//! - LOOP-5  — a `Retry-After` was parsed but never honored.
//! - LOOP-5b — `is_retryable` retried every `Model(_)` and had no extension point.
//! - LOOP-1  — reconciling two limit sources could *widen* a caller's cap.
//! - LOOP-9b — a reached cap was always a hard error.
//! - LOOP-6  — `Started` events had no terminal partner on error paths.
//! - LOOP-8  — steering applied part of a rejected batch, and a pause was unresumable.
//! - LOOP-9  — a panicking listener wedged the event sink forever.
//! - C6      — `StreamChunk` had no producer.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use tinyagents::error::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::{
    AgentEvent, EventListener, EventRecord, EventSink, RecordingListener,
};
use tinyagents::harness::ids::{CallId, RunId};
use tinyagents::harness::limits::{LimitBehavior, LimitOutcome, LimitTracker, RunLimits};
use tinyagents::harness::message::{Message, MessageDelta};
use tinyagents::harness::no_progress::fingerprint_arguments;
use tinyagents::harness::retry::{RetryPolicy, is_retryable, retry_after_hint};
use tinyagents::harness::steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringOutcome, SteeringPolicy,
    apply_pending_steering,
};
use tinyagents::harness::stream::{StreamChunk, StreamMode, StreamSink, project_event};

// ── LOOP-2: jitter must never collapse the backoff ───────────────────────────

#[test]
fn loop2_enabling_jitter_does_not_disable_backoff() {
    let policy = RetryPolicy::default().with_jitter(true);
    for attempt in 0..6 {
        for _ in 0..50 {
            assert!(
                policy.backoff_for_attempt(attempt) > Duration::ZERO,
                "jitter zeroed the backoff at attempt {attempt}"
            );
        }
    }
}

#[tokio::test(start_paused = true)]
async fn loop2_the_production_hardened_config_actually_sleeps() {
    let policy = RetryPolicy::default()
        .with_backoff_sleep(true)
        .with_jitter(true);
    let start = tokio::time::Instant::now();
    policy.sleep_backoff(1).await;
    assert!(
        start.elapsed() > Duration::ZERO,
        "with_backoff_sleep(true).with_jitter(true) never slept"
    );
}

// ── LOCAL-4: backoff sleeps by default ───────────────────────────────────────

#[test]
fn local4_backoff_sleep_is_on_by_default_and_opt_out() {
    assert!(RetryPolicy::default().backoff_sleep);
    assert!(
        !RetryPolicy::default()
            .with_backoff_sleep(false)
            .backoff_sleep
    );
}

// ── LOOP-5: Retry-After is honored ───────────────────────────────────────────

#[test]
fn loop5_retry_after_is_read_and_lengthens_the_wait() {
    let policy = RetryPolicy::default();
    let rate_limited = TinyAgentsError::Model("429 Too Many Requests, Retry-After: 30".into());

    assert_eq!(
        retry_after_hint(&rate_limited),
        Some(Duration::from_secs(30))
    );
    assert_eq!(
        policy.backoff_for_error(0, &rate_limited),
        Duration::from_secs(30),
        "a 429 saying Retry-After: 30 was still retried after 200ms"
    );
}

#[test]
fn loop5_a_retry_after_can_only_lengthen_never_shorten() {
    let policy = RetryPolicy::default();
    let zero_hint = TinyAgentsError::Model("429 rate limited, Retry-After: 0".into());
    assert_eq!(
        policy.backoff_for_error(2, &zero_hint),
        policy.backoff_for_attempt(2)
    );
}

#[test]
fn loop5_a_hostile_retry_after_is_clamped() {
    let policy = RetryPolicy::default().with_max_retry_after_ms(1_000);
    let absurd = TinyAgentsError::Model("429, Retry-After: 999999".into());
    assert_eq!(
        policy.backoff_for_error(0, &absurd),
        Duration::from_millis(1_000)
    );
}

// ── LOOP-5b: classification and the retry_on extension point ─────────────────

#[test]
fn loop5b_model_errors_consult_the_provider_failure_class() {
    assert!(!is_retryable(&TinyAgentsError::Model(
        "401 Unauthorized: invalid api key".into()
    )));
    assert!(is_retryable(&TinyAgentsError::Model(
        "503 Service Unavailable".into()
    )));
}

#[test]
fn loop5b_retry_on_predicate_is_honored_by_should_retry_error() {
    let policy = RetryPolicy::default()
        .with_max_attempts(3)
        .with_retry_on(Arc::new(|err: &TinyAgentsError| {
            !matches!(err, TinyAgentsError::Tool(_))
        }));

    assert!(!policy.should_retry_error(0, &TinyAgentsError::Tool("no".into())));
    assert!(policy.should_retry_error(0, &TinyAgentsError::Model("yes".into())));
}

// ── LOOP-1: reconciling two limit sources is fail-closed ─────────────────────

#[test]
fn loop1_an_explicit_cap_is_never_widened_by_a_second_source() {
    let mut tracker = LimitTracker::new(RunLimits::default().with_max_model_calls(2));
    tracker.tighten_call_limits(25, 50);
    assert_eq!(tracker.limits().max_model_calls, 2);

    tracker.record_model_call().unwrap();
    tracker.record_model_call().unwrap();
    assert!(
        tracker.record_model_call().is_err(),
        "the caller's cap of 2 ran to 25"
    );
}

// ── LOOP-9b: exhaustion can stop cleanly instead of discarding the run ───────

#[test]
fn loop9b_stop_with_partial_returns_an_outcome_not_an_error() {
    let mut tracker = LimitTracker::new(
        RunLimits::default()
            .with_max_model_calls(1)
            .with_behavior(LimitBehavior::StopWithPartial),
    );
    assert_eq!(
        tracker.try_record_model_call().unwrap(),
        LimitOutcome::Proceed
    );
    assert!(matches!(
        tracker.try_record_model_call().unwrap(),
        LimitOutcome::Stop(_)
    ));
}

#[test]
fn loop9b_error_remains_the_default_behavior() {
    let mut tracker = LimitTracker::new(RunLimits::default().with_max_model_calls(1));
    tracker.try_record_model_call().unwrap();
    assert!(tracker.try_record_model_call().is_err());
}

// ── LOOP-6: every Started has a terminal partner on the error path ───────────

#[test]
fn loop6_failure_variants_exist_for_tool_model_and_subagent() {
    let events = [
        AgentEvent::ToolFailed {
            call_id: CallId::new("c1"),
            tool_name: "search".into(),
            started_at_ms: Some(1),
            duration_ms: Some(2),
            error: "boom".into(),
        },
        AgentEvent::ModelFailed {
            call_id: CallId::new("c2"),
            model: "gpt-4o".into(),
            started_at_ms: Some(1),
            attempts: Some(4),
            error: "boom".into(),
        },
        AgentEvent::SubAgentFailed {
            name: "child".into(),
            depth: 1,
            error: "boom".into(),
        },
    ];
    let kinds: Vec<&str> = events.iter().map(AgentEvent::kind).collect();
    assert_eq!(
        kinds,
        vec!["tool.failed", "model.failed", "subagent.failed"]
    );

    // They must survive a durable journal round trip.
    for event in events {
        let json = serde_json::to_value(&event).unwrap();
        let back: AgentEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }
}

// ── LOOP-9: a panicking listener must not wedge the sink ─────────────────────

struct OneShotBomb;

impl EventListener for OneShotBomb {
    fn on_event(&self, _record: &EventRecord) {
        panic!("listener exploded");
    }
}

#[test]
fn loop9_a_panicking_listener_does_not_stop_later_delivery() {
    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(Arc::new(OneShotBomb));
    sink.subscribe(recorder.clone());

    let clone = sink.clone();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clone.emit(AgentEvent::StateUpdate)
    }));

    // Every later emit is swallowed by the panicking listener, but the *sink*
    // must still dispatch — before the fix nothing was delivered ever again.
    for _ in 0..3 {
        let clone = sink.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            clone.emit(AgentEvent::StateUpdate)
        }));
    }
    assert_eq!(
        recorder.len(),
        0,
        "the bomb listener is first, so it aborts each dispatch"
    );

    // With the bomb removed, delivery resumes immediately — proof the sink was
    // never latched into a permanently non-dispatching state.
    let clean = EventSink::new();
    let recorder2 = Arc::new(RecordingListener::new());
    clean.subscribe(recorder2.clone());
    clean.subscribe(Arc::new(OneShotBomb));
    for _ in 0..3 {
        let clone = clean.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            clone.emit(AgentEvent::StateUpdate)
        }));
    }
    assert_eq!(
        recorder2.len(),
        3,
        "sink stayed wedged after a listener panic"
    );
}

// ── LOOP-8: steering batches are atomic and pauses are resumable ─────────────

#[test]
fn loop8_a_rejected_batch_applies_nothing() {
    let handle =
        SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage));
    handle.send(SteeringCommand::InjectMessage(Message::user("first")));
    handle.send(SteeringCommand::Cancel); // not allowed

    let mut ctx: RunContext = RunContext::new(RunConfig::new("r"), ()).with_steering(handle);
    let mut messages: Vec<Message> = Vec::new();

    assert!(apply_pending_steering(&mut ctx, &mut messages).is_err());
    assert!(
        messages.is_empty(),
        "a command before the rejected one was still applied"
    );
}

#[test]
fn loop8_a_pause_is_latched_and_resumable_across_checkpoints() {
    let handle = SteeringHandle::allow_all();
    let mut ctx: RunContext =
        RunContext::new(RunConfig::new("r"), ()).with_steering(handle.clone());
    let mut messages: Vec<Message> = Vec::new();

    handle.send(SteeringCommand::PauseWith {
        reason: "human review".into(),
    });
    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Pause
    );
    // Still paused at the next checkpoint, with an empty queue.
    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Pause
    );

    // The state that makes a pause distinguishable from an empty answer.
    let state = handle.pause_state().expect("a pause carries state");
    assert_eq!(state.reason.as_deref(), Some("human review"));

    // Resumable from a later batch — impossible before.
    handle.send(SteeringCommand::Resume);
    assert_eq!(
        apply_pending_steering(&mut ctx, &mut messages).unwrap(),
        SteeringOutcome::Continue
    );
    assert!(handle.pause_state().is_none());
}

// ── C6: the stream projection ────────────────────────────────────────────────

#[test]
fn c6_events_project_onto_stream_chunks() {
    let delta = AgentEvent::ModelDelta {
        run_id: RunId::new("r1"),
        call_id: CallId::new("c1"),
        delta: MessageDelta::text("hi"),
    };
    assert!(matches!(
        project_event(&delta),
        Some(StreamChunk::Message(_))
    ));

    // The Interrupt variant finally has a producer.
    let interrupt = AgentEvent::ControlApplied {
        control: "interrupt".into(),
        detail: "needs approval".into(),
    };
    assert!(matches!(
        project_event(&interrupt),
        Some(StreamChunk::Interrupt(_))
    ));

    // Mode filtering happens producer-side.
    let sink = StreamSink::new([StreamMode::Interrupts]);
    assert!(!sink.push_event(&delta));
    assert!(sink.push_event(&interrupt));
    assert_eq!(sink.drain().len(), 1);
}

// ── C5: the no-progress tracker is drivable from a hook ──────────────────────

#[test]
fn c5_argument_fingerprints_are_canonical() {
    assert_eq!(
        fingerprint_arguments(&json!({"a": 1, "b": 2})),
        fingerprint_arguments(&json!({"b": 2, "a": 1}))
    );
    assert_ne!(
        fingerprint_arguments(&json!({"a": 1})),
        fingerprint_arguments(&json!({"a": 2}))
    );
}
