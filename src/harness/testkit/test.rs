//! Tests for the harness testkit.
//!
//! Exercises every double and the trajectory assertions with synthetic inputs,
//! verifying that all public API contracts are met without hitting live
//! providers.

use futures::StreamExt;

use crate::harness::events::AgentEvent;
use crate::harness::ids::{CallId, RunId};
use crate::harness::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStreamItem, collect_model_stream,
};
use crate::harness::testkit::{
    DeterministicClock, DeterministicIds, EventRecorder, FakeTool, ScriptedModel, StreamingMock,
    Trajectory,
};
use crate::harness::tool::{Tool, ToolCall};
use crate::harness::usage::Usage;

// ---------------------------------------------------------------------------
// StreamingMock
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_mock_yields_started_deltas_and_completed() {
    let model = StreamingMock::from_text_chunks(["Hello", ", ", "world"]);
    let stream = ChatModel::<()>::stream(&model, &(), ModelRequest::default())
        .await
        .unwrap();
    let items: Vec<ModelStreamItem> = stream.collect().await;

    assert!(matches!(items.first(), Some(ModelStreamItem::Started)));
    assert!(matches!(items.last(), Some(ModelStreamItem::Completed(_))));
    let delta_count = items
        .iter()
        .filter(|item| matches!(item, ModelStreamItem::MessageDelta(_)))
        .count();
    assert_eq!(delta_count, 3, "one message delta per text chunk");
}

#[tokio::test]
async fn streaming_mock_accumulates_to_full_text() {
    let model = StreamingMock::from_text_chunks(["foo", "bar", "baz"]);
    let stream = ChatModel::<()>::stream(&model, &(), ModelRequest::default())
        .await
        .unwrap();
    let merged = collect_model_stream(stream).await.unwrap();
    assert_eq!(merged.text(), "foobarbaz");

    // The unary path returns the same merged response.
    let invoked = ChatModel::<()>::invoke(&model, &(), ModelRequest::default())
        .await
        .unwrap();
    assert_eq!(invoked.text(), "foobarbaz");
    assert_eq!(model.call_count(), 2);
}

// ---------------------------------------------------------------------------
// ScriptedModel
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scripted_model_returns_responses_in_order() {
    let model = ScriptedModel::new(vec![
        ModelResponse::assistant("first"),
        ModelResponse::assistant("second"),
    ]);

    let state = ();
    let req = ModelRequest::default();

    let r1 = model.invoke(&state, req.clone()).await.unwrap();
    assert_eq!(r1.text(), "first");

    let r2 = model.invoke(&state, req.clone()).await.unwrap();
    assert_eq!(r2.text(), "second");
}

#[tokio::test]
async fn scripted_model_errors_when_exhausted() {
    let model = ScriptedModel::new(vec![ModelResponse::assistant("only")]);
    let state = ();
    let req = ModelRequest::default();

    model.invoke(&state, req.clone()).await.unwrap();
    let err = model.invoke(&state, req.clone()).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("exhausted"),
        "expected 'exhausted' in error, got: {msg}"
    );
}

#[tokio::test]
async fn scripted_model_records_received_requests() {
    let model = ScriptedModel::replies(vec!["a", "b"]);
    let state = ();

    let req1 = ModelRequest::new(vec![crate::harness::message::Message::user("hello")]);
    let req2 = ModelRequest::new(vec![crate::harness::message::Message::user("world")]);

    model.invoke(&state, req1).await.unwrap();
    model.invoke(&state, req2).await.unwrap();

    let received = model.requests();
    assert_eq!(received.len(), 2);
    assert_eq!(received[0].messages[0].text(), "hello");
    assert_eq!(received[1].messages[0].text(), "world");
}

#[tokio::test]
async fn scripted_model_replies_constructor() {
    let model = ScriptedModel::replies(vec!["hello", "world"]);
    let state = ();
    let req = ModelRequest::default();

    let r1 = model.invoke(&state, req.clone()).await.unwrap();
    assert_eq!(r1.text(), "hello");

    let r2 = model.invoke(&state, req.clone()).await.unwrap();
    assert_eq!(r2.text(), "world");
}

#[tokio::test]
async fn scripted_model_with_usage() {
    let usage = Usage {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        ..Default::default()
    };
    let model = ScriptedModel::new(vec![ModelResponse::assistant("hi").with_usage(usage)]);
    let state = ();
    let response = model.invoke(&state, ModelRequest::default()).await.unwrap();
    assert_eq!(response.usage.unwrap().input_tokens, 10);
}

// ---------------------------------------------------------------------------
// FakeTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fake_tool_returning_produces_text_result() {
    let tool = FakeTool::returning("search", "42");
    let state = ();
    let call = ToolCall::new("c1", "search", serde_json::json!({}));
    let result = tool.call(&state, call).await.unwrap();
    assert_eq!(result.content, "42");
    assert!(result.error.is_none());
}

#[tokio::test]
async fn fake_tool_failing_returns_error() {
    let tool = FakeTool::failing("explode", "boom");
    let state = ();
    let call = ToolCall::new("c1", "explode", serde_json::json!({}));
    let err = tool.call(&state, call).await.unwrap_err();
    assert!(err.to_string().contains("boom"));
}

#[tokio::test]
async fn fake_tool_new_returns_empty_result() {
    let tool = FakeTool::new("noop");
    let state = ();
    let call = ToolCall::new("c1", "noop", serde_json::json!({}));
    let result = tool.call(&state, call).await.unwrap();
    assert_eq!(result.content, "");
}

#[tokio::test]
async fn fake_tool_records_calls() {
    let tool = FakeTool::returning("search", "ok");
    let state = ();

    let c1 = ToolCall::new("id1", "search", serde_json::json!({"q": "rust"}));
    let c2 = ToolCall::new("id2", "search", serde_json::json!({"q": "cargo"}));

    tool.call(&state, c1.clone()).await.unwrap();
    tool.call(&state, c2.clone()).await.unwrap();

    let calls = tool.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "id1");
    assert_eq!(calls[1].id, "id2");
}

#[tokio::test]
async fn fake_tool_name_and_description() {
    use crate::harness::tool::Tool;
    let tool = FakeTool::returning("my_tool", "res");
    // Access via explicit trait disambiguation to resolve the generic State.
    assert_eq!(<FakeTool as Tool<()>>::name(&tool), "my_tool");
    assert!(<FakeTool as Tool<()>>::description(&tool).contains("my_tool"));
}

#[tokio::test]
async fn fake_tool_schema_is_valid() {
    use crate::harness::tool::Tool;
    let tool = FakeTool::new("echo");
    let schema = <FakeTool as Tool<()>>::schema(&tool);
    assert_eq!(schema.name, "echo");
}

// ---------------------------------------------------------------------------
// DeterministicClock
// ---------------------------------------------------------------------------

#[test]
fn deterministic_clock_starts_at_given_millis() {
    let clock = DeterministicClock::new(1_000);
    assert_eq!(clock.now_millis(), 1_000);
}

#[test]
fn deterministic_clock_advances_correctly() {
    let clock = DeterministicClock::new(0);
    clock.advance(250);
    assert_eq!(clock.now_millis(), 250);
    clock.advance(750);
    assert_eq!(clock.now_millis(), 1_000);
}

#[test]
fn deterministic_clock_default_starts_at_zero() {
    let clock = DeterministicClock::default();
    assert_eq!(clock.now_millis(), 0);
}

// ---------------------------------------------------------------------------
// DeterministicIds
// ---------------------------------------------------------------------------

#[test]
fn deterministic_ids_sequence() {
    let ids = DeterministicIds::new("run");
    assert_eq!(ids.next(), "run-0");
    assert_eq!(ids.next(), "run-1");
    assert_eq!(ids.next(), "run-2");
}

#[test]
fn deterministic_ids_different_prefixes() {
    let call_ids = DeterministicIds::new("call");
    let run_ids = DeterministicIds::new("run");

    assert_eq!(call_ids.next(), "call-0");
    assert_eq!(run_ids.next(), "run-0");
    assert_eq!(call_ids.next(), "call-1");
    assert_eq!(run_ids.next(), "run-1");
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------

#[test]
fn event_recorder_captures_events() {
    let recorder = EventRecorder::new();
    let sink = recorder.sink();

    sink.emit(AgentEvent::RunStarted {
        run_id: RunId::new("r1"),
        thread_id: None,
    });
    sink.emit(AgentEvent::ModelStarted {
        call_id: CallId::new("c1"),
        model: "gpt".into(),
    });

    let events = recorder.events();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
    assert!(matches!(events[1], AgentEvent::ModelStarted { .. }));
}

#[test]
fn event_recorder_kinds() {
    let recorder = EventRecorder::new();
    let sink = recorder.sink();

    sink.emit(AgentEvent::RunStarted {
        run_id: RunId::new("r1"),
        thread_id: None,
    });
    sink.emit(AgentEvent::RunCompleted {
        run_id: RunId::new("r1"),
    });

    let kinds = recorder.kinds();
    assert_eq!(kinds, vec!["run.started", "run.completed"]);
}

#[test]
fn event_recorder_default_is_empty() {
    let recorder = EventRecorder::default();
    assert!(recorder.events().is_empty());
}

#[test]
fn event_recorder_sink_clones_share_listener() {
    let recorder = EventRecorder::new();
    let sink_a = recorder.sink();
    let sink_b = recorder.sink(); // second clone, shares listeners

    sink_a.emit(AgentEvent::StateUpdate);
    sink_b.emit(AgentEvent::StateUpdate);

    // Both clones emit through the same shared inner, so recorder sees both.
    assert_eq!(recorder.events().len(), 2);
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

fn make_trajectory() -> Vec<AgentEvent> {
    vec![
        AgentEvent::RunStarted {
            run_id: RunId::new("r1"),
            thread_id: None,
        },
        AgentEvent::ModelStarted {
            call_id: CallId::new("c1"),
            model: "gpt-4".into(),
        },
        AgentEvent::ModelCompleted {
            call_id: CallId::new("c1"),
            usage: None,
            input: None,
            output: None,
        },
        AgentEvent::ToolStarted {
            call_id: CallId::new("t1"),
            tool_name: "search".into(),
        },
        AgentEvent::ToolCompleted {
            call_id: CallId::new("t1"),
            tool_name: "search".into(),
            input: None,
            output: None,
        },
        AgentEvent::ModelStarted {
            call_id: CallId::new("c2"),
            model: "gpt-4".into(),
        },
        AgentEvent::ModelCompleted {
            call_id: CallId::new("c2"),
            usage: None,
            input: None,
            output: None,
        },
        AgentEvent::RunCompleted {
            run_id: RunId::new("r1"),
        },
    ]
}

#[test]
fn trajectory_tool_was_called() {
    let traj = Trajectory::from_events(make_trajectory());
    assert!(traj.tool_was_called("search"));
    assert!(!traj.tool_was_called("nonexistent"));
}

#[test]
fn trajectory_assert_tool_called_passes() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_tool_called("search"); // should not panic
}

#[test]
#[should_panic(expected = "search2")]
fn trajectory_assert_tool_called_panics_when_missing() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_tool_called("search2");
}

#[test]
fn trajectory_tool_call_count() {
    let mut events = make_trajectory();
    // Add a second call to 'search'.
    events.push(AgentEvent::ToolStarted {
        call_id: CallId::new("t2"),
        tool_name: "search".into(),
    });
    let traj = Trajectory::from_events(events);
    assert_eq!(traj.tool_call_count("search"), 2);
    assert_eq!(traj.tool_call_count("other"), 0);
}

#[test]
fn trajectory_model_call_count() {
    let traj = Trajectory::from_events(make_trajectory());
    assert_eq!(traj.model_call_count(), 2);
}

#[test]
fn trajectory_assert_model_called_times_passes() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_model_called_times(2); // should not panic
}

#[test]
#[should_panic(expected = "expected 3 model call(s) but found 2")]
fn trajectory_assert_model_called_times_panics_on_mismatch() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_model_called_times(3);
}

#[test]
fn trajectory_completed_is_true_when_run_completed_present() {
    let traj = Trajectory::from_events(make_trajectory());
    assert!(traj.completed());
}

#[test]
fn trajectory_completed_is_false_when_no_run_completed() {
    let events = vec![AgentEvent::ModelStarted {
        call_id: CallId::new("c1"),
        model: "x".into(),
    }];
    let traj = Trajectory::from_events(events);
    assert!(!traj.completed());
}

#[test]
fn trajectory_assert_completed_passes() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_completed(); // should not panic
}

#[test]
#[should_panic(expected = "RunCompleted")]
fn trajectory_assert_completed_panics_when_missing() {
    let traj = Trajectory::from_events(vec![AgentEvent::StateUpdate]);
    traj.assert_completed();
}

#[test]
fn trajectory_failed_is_true_when_run_failed_present() {
    let events = vec![AgentEvent::RunFailed {
        run_id: RunId::new("r1"),
        error: "oops".into(),
    }];
    let traj = Trajectory::from_events(events);
    assert!(traj.failed());
}

#[test]
fn trajectory_failed_is_false_when_absent() {
    let traj = Trajectory::from_events(make_trajectory());
    assert!(!traj.failed());
}

#[test]
fn trajectory_assert_order_by_kind() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_order(&[
        "run.started",
        "model.started",
        "tool.started",
        "run.completed",
    ])
    .expect("order assertion should pass");
}

#[test]
fn trajectory_assert_order_by_tool_name() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_order(&["model.started", "search", "model.started"])
        .expect("tool name order assertion should pass");
}

#[test]
fn trajectory_assert_order_fails_when_not_subsequence() {
    let traj = Trajectory::from_events(make_trajectory());
    let result = traj.assert_order(&["run.completed", "run.started"]); // wrong order
    assert!(result.is_err(), "should fail because order is reversed");
    assert!(
        result.unwrap_err().to_string().contains("run.started"),
        "error should mention the missing label"
    );
}

#[test]
fn trajectory_assert_order_fails_for_nonexistent_label() {
    let traj = Trajectory::from_events(make_trajectory());
    let result = traj.assert_order(&["model.started", "nonexistent_tool"]);
    assert!(result.is_err());
}

#[test]
fn trajectory_assert_order_empty_labels_always_passes() {
    let traj = Trajectory::from_events(make_trajectory());
    traj.assert_order(&[])
        .expect("empty label list should always pass");
}
