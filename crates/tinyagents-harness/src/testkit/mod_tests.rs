//! Tests for the harness testkit.
//!
//! Exercises every double and the trajectory assertions with synthetic inputs,
//! verifying that all public API contracts are met without hitting live
//! providers.

use futures::StreamExt;

use crate::events::AgentEvent;
use crate::ids::{CallId, RunId};
use crate::testkit::{
    DeterministicClock, DeterministicIds, EventRecorder, FakeTool, SchemaDrivenModel,
    ScriptedModel, StreamingMock, Trajectory, generate_args_from_schema,
};
use tinyinference_llm::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStreamItem, collect_model_stream,
};
use tinyinference_llm::tool::ToolSchema;
use tinyinference_llm::usage::Usage;
use tinytools::Tool;

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

    let req1 = ModelRequest::new(vec![tinyinference_llm::message::Message::user("hello")]);
    let req2 = ModelRequest::new(vec![tinyinference_llm::message::Message::user("world")]);

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
    let result = tool.execute(serde_json::json!({})).await.unwrap();
    assert_eq!(result.output(), "42");
    assert!(!result.is_error);
}

#[tokio::test]
async fn fake_tool_failing_returns_error() {
    let tool = FakeTool::failing("explode", "boom");
    let err = tool.execute(serde_json::json!({})).await.unwrap_err();
    assert!(err.to_string().contains("boom"));
}

#[tokio::test]
async fn fake_tool_new_returns_empty_result() {
    let tool = FakeTool::new("noop");
    let result = tool.execute(serde_json::json!({})).await.unwrap();
    assert_eq!(result.output(), "");
}

#[tokio::test]
async fn fake_tool_records_calls() {
    let tool = FakeTool::returning("search", "ok");
    tool.execute(serde_json::json!({"q": "rust"}))
        .await
        .unwrap();
    tool.execute(serde_json::json!({"q": "cargo"}))
        .await
        .unwrap();

    let calls = tool.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["q"], "rust");
    assert_eq!(calls[1]["q"], "cargo");
}

#[tokio::test]
async fn fake_tool_name_and_description() {
    let tool = FakeTool::returning("my_tool", "res");
    assert_eq!(<FakeTool as Tool>::name(&tool), "my_tool");
    assert!(<FakeTool as Tool>::description(&tool).contains("my_tool"));
}

#[tokio::test]
async fn fake_tool_schema_is_valid() {
    let tool = FakeTool::new("echo");
    assert_eq!(<FakeTool as Tool>::spec(&tool).name, "echo");
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
        outcome: None,
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
            started_at_ms: None,
            usage: None,
            input: None,
            output: None,
        },
        AgentEvent::ToolStarted {
            parent_call_id: None,
            call_id: CallId::new("t1"),
            tool_name: "search".into(),
            input: None,
        },
        AgentEvent::ToolCompleted {
            parent_call_id: None,
            call_id: CallId::new("t1"),
            tool_name: "search".into(),
            started_at_ms: None,
            input: None,
            output: None,
            duration_ms: None,
            input_bytes: None,
            output_bytes: None,
            error: None,
            metadata: None,
        },
        AgentEvent::ModelStarted {
            call_id: CallId::new("c2"),
            model: "gpt-4".into(),
        },
        AgentEvent::ModelCompleted {
            call_id: CallId::new("c2"),
            started_at_ms: None,
            usage: None,
            input: None,
            output: None,
        },
        AgentEvent::RunCompleted {
            run_id: RunId::new("r1"),
            outcome: None,
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
        parent_call_id: None,
        call_id: CallId::new("t2"),
        tool_name: "search".into(),
        input: None,
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
        outcome: None,
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

// ---------------------------------------------------------------------------
// SchemaDrivenModel
// ---------------------------------------------------------------------------

#[test]
fn generate_args_from_schema_fills_declared_properties_by_type() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "enabled": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "nested": {
                "type": "object",
                "properties": {"inner": {"type": "string"}}
            }
        }
    });
    let args = generate_args_from_schema(&schema);
    assert_eq!(args["name"], "test");
    assert_eq!(args["count"], 0);
    assert_eq!(args["ratio"], 0.0);
    assert_eq!(args["enabled"], false);
    assert_eq!(args["tags"], serde_json::json!(["test"]));
    assert_eq!(args["nested"]["inner"], "test");
}

#[test]
fn generate_args_from_schema_treats_untyped_properties_schema_as_object() {
    let schema = serde_json::json!({"properties": {"a": {"type": "string"}}});
    let args = generate_args_from_schema(&schema);
    assert_eq!(args["a"], "test");
}

#[test]
fn generate_args_from_schema_empty_object_schema_is_empty_object() {
    let schema = serde_json::json!({"type": "object"});
    assert_eq!(generate_args_from_schema(&schema), serde_json::json!({}));
}

fn tool_schema(name: &str, params: serde_json::Value) -> ToolSchema {
    ToolSchema::new(name, format!("{name} description"), params)
}

#[tokio::test]
async fn schema_driven_model_calls_every_declared_tool_once_then_final_response() {
    let model = SchemaDrivenModel::with_final_text("all tools called");
    let tools = vec![
        tool_schema(
            "search",
            serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        ),
        tool_schema(
            "count",
            serde_json::json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
        ),
    ];

    // Call 0: expect a tool call for `search` with a schema-generated `query`.
    let request = ModelRequest::new(vec![]).with_tools(tools.clone());
    let response = model.invoke(&(), request).await.unwrap();
    assert_eq!(response.tool_calls().len(), 1);
    assert_eq!(response.tool_calls()[0].name, "search");
    assert_eq!(response.tool_calls()[0].arguments["query"], "test");

    // Call 1: expect a tool call for `count`.
    let request = ModelRequest::new(vec![]).with_tools(tools.clone());
    let response = model.invoke(&(), request).await.unwrap();
    assert_eq!(response.tool_calls()[0].name, "count");
    assert_eq!(response.tool_calls()[0].arguments["n"], 0);

    // Call 2: every declared tool has been called once; return the final
    // configured response instead.
    let request = ModelRequest::new(vec![]).with_tools(tools.clone());
    let response = model.invoke(&(), request).await.unwrap();
    assert!(response.tool_calls().is_empty());
    assert_eq!(response.text(), "all tools called");

    assert_eq!(model.call_count(), 3);
    assert_eq!(model.requests().len(), 3);
}

#[tokio::test]
async fn schema_driven_model_with_no_tools_returns_final_response_immediately() {
    let model = SchemaDrivenModel::with_final_text("no tools needed");
    let response = model.invoke(&(), ModelRequest::new(vec![])).await.unwrap();
    assert!(response.tool_calls().is_empty());
    assert_eq!(response.text(), "no tools needed");
}

#[test]
fn canned_responses_carry_their_finish_reason_and_content() {
    use super::{text_response, tool_call_response};
    use serde_json::json;
    use tinyinference_llm::tool::ToolCall;
    use tinyinference_llm::usage::Usage;

    let text = text_response("done").with_usage(Usage::new(3, 2));
    assert_eq!(text.text(), "done");
    assert_eq!(text.finish_reason.as_deref(), Some("stop"));
    assert_eq!(text.usage, Some(Usage::new(3, 2)));
    assert_eq!(text.message.usage, Some(Usage::new(3, 2)));

    let call = tool_call_response(ToolCall::new("c1", "lookup", json!({"q": 1})));
    assert_eq!(call.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(call.message.id.as_deref(), Some("msg-c1"));
    assert!(call.message.content.is_empty());
    assert_eq!(call.tool_calls().len(), 1);
    assert_eq!(call.tool_calls()[0].name, "lookup");
}
