//! Unit tests for the run-context module.

use super::*;
use crate::harness::events::{AgentEvent, RecordingListener};
use std::sync::Arc;

#[test]
fn run_config_defaults_are_sensible() {
    let config = RunConfig::new("run-1");
    assert_eq!(config.run_id.as_str(), "run-1");
    assert!(config.thread_id.is_none());
    assert!(config.tags.is_empty());
    assert_eq!(config.metadata, serde_json::Value::Null);
    assert!(config.timeout_ms.is_none());
    assert_eq!(config.max_model_calls, 25);
    assert_eq!(config.max_tool_calls, 50);
}

#[test]
fn run_config_builders_compose() {
    let config = RunConfig::new("run-2")
        .with_thread("thread-9")
        .with_tag("a")
        .with_tag("b")
        .with_metadata(serde_json::json!({"k": "v"}))
        .with_timeout_ms(1234)
        .with_max_model_calls(3)
        .with_max_tool_calls(4);

    assert_eq!(config.thread_id.as_ref().unwrap().as_str(), "thread-9");
    assert_eq!(config.tags, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(config.metadata["k"], serde_json::json!("v"));
    assert_eq!(config.timeout_ms, Some(1234));
    assert_eq!(config.max_model_calls, 3);
    assert_eq!(config.max_tool_calls, 4);
}

#[test]
fn run_config_round_trips_through_json() {
    let config = RunConfig::new("run-3").with_thread("t1").with_tag("x");
    let json = serde_json::to_string(&config).unwrap();
    let back: RunConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.run_id.as_str(), "run-3");
    assert_eq!(back.thread_id.unwrap().as_str(), "t1");
    assert_eq!(back.tags, vec!["x".to_string()]);
}

#[test]
fn context_exposes_run_and_thread_ids() {
    let config = RunConfig::new("run-4").with_thread("thread-4");
    let ctx: RunContext = RunContext::new(config, ());
    assert_eq!(ctx.run_id().as_str(), "run-4");
    assert_eq!(ctx.thread_id().unwrap().as_str(), "thread-4");
}

#[test]
fn context_carries_generic_user_data() {
    let config = RunConfig::new("run-5");
    let ctx: RunContext<u32> = RunContext::new(config, 42);
    assert_eq!(ctx.data, 42);
}

#[test]
fn context_records_calls_and_enforces_limits() {
    let config = RunConfig::new("run-6")
        .with_max_model_calls(1)
        .with_max_tool_calls(1);
    let mut ctx: RunContext = RunContext::new(config, ());

    ctx.record_model_call().expect("first model call ok");
    assert_eq!(ctx.limits.model_calls(), 1);
    assert!(ctx.record_model_call().is_err(), "second exceeds cap");

    ctx.record_tool_call().expect("first tool call ok");
    assert_eq!(ctx.limits.tool_calls(), 1);
    assert!(ctx.record_tool_call().is_err(), "second exceeds cap");
}

#[test]
fn context_emit_delegates_to_event_sink() {
    let config = RunConfig::new("run-7");
    let ctx: RunContext = RunContext::new(config, ());

    let recorder = Arc::new(RecordingListener::new());
    ctx.events.subscribe(recorder.clone());

    let record = ctx.emit(AgentEvent::RunStarted {
        run_id: ctx.run_id().clone(),
        thread_id: None,
    });
    assert_eq!(record.offset, 0);
    assert_eq!(recorder.events().len(), 1);
}

#[test]
fn context_check_deadline_passes_without_timeout() {
    let config = RunConfig::new("run-8");
    let mut ctx: RunContext = RunContext::new(config, ());
    assert!(ctx.check_deadline().is_ok());
}

#[test]
fn with_events_shares_sink() {
    let shared = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    shared.subscribe(recorder.clone());

    let ctx: RunContext = RunContext::new(RunConfig::new("run-9"), ()).with_events(shared.clone());
    ctx.emit(AgentEvent::StateUpdate);
    assert_eq!(recorder.events().len(), 1);
    assert_eq!(shared.len(), 1);
}

#[test]
fn request_control_keeps_highest_precedence() {
    let ctx: RunContext = RunContext::new(RunConfig::new("run-ctrl"), ());

    // A stronger Interrupt is not downgraded by a later, weaker StopWithFinal.
    ctx.request_control(MiddlewareControl::Interrupt {
        node: "review".into(),
        message: "hold".into(),
    });
    ctx.request_control(MiddlewareControl::StopWithFinal("stop".into()));
    assert!(matches!(
        ctx.take_control(),
        Some(MiddlewareControl::Interrupt { .. })
    ));
    assert!(
        ctx.take_control().is_none(),
        "take_control clears the request"
    );

    // A stronger request does replace a weaker pending one.
    ctx.request_control(MiddlewareControl::StopWithFinal("stop".into()));
    ctx.request_control(MiddlewareControl::Interrupt {
        node: "review".into(),
        message: "hold".into(),
    });
    assert!(matches!(
        ctx.take_control(),
        Some(MiddlewareControl::Interrupt { .. })
    ));
}

#[test]
fn checked_child_depth_is_the_shared_depth_guard() {
    use crate::error::TinyAgentsError;

    // Below the cap: returns parent_depth + 1.
    assert_eq!(RunConfig::checked_child_depth(0, 8).unwrap(), 1);
    assert_eq!(RunConfig::checked_child_depth(7, 8).unwrap(), 8);

    // At the boundary (child would be max_depth + 1): fail closed with the cap.
    match RunConfig::checked_child_depth(8, 8) {
        Err(TinyAgentsError::SubAgentDepth(cap)) => assert_eq!(cap, 8),
        other => panic!("expected SubAgentDepth(8), got {other:?}"),
    }
}
