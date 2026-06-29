//! Tests for the `providers` module.
//!
//! All tests are deterministic and have no network dependencies.

use serde_json::json;

use crate::harness::message::Message;
use crate::harness::model::{ChatModel, ModelRequest};
use crate::harness::providers::MockModel;

// ---------------------------------------------------------------------------
// Shared helper: a unit state used throughout
// ---------------------------------------------------------------------------

/// Minimal application state for generic impls that don't need state.
struct NoState;

// ---------------------------------------------------------------------------
// MockModel::echo
// ---------------------------------------------------------------------------

#[tokio::test]
async fn echo_returns_last_user_message() {
    let model = MockModel::echo();
    let request = ModelRequest::new(vec![
        Message::system("You are helpful."),
        Message::user("Hello, mock!"),
    ]);

    let response = model.invoke(&NoState, request).await.unwrap();
    assert_eq!(response.text(), "Hello, mock!");
}

#[tokio::test]
async fn echo_returns_last_user_when_multiple_turns() {
    let model = MockModel::echo();
    let request = ModelRequest::new(vec![
        Message::user("first turn"),
        Message::assistant("reply"),
        Message::user("second turn"),
    ]);

    let response = model.invoke(&NoState, request).await.unwrap();
    assert_eq!(response.text(), "second turn");
}

#[tokio::test]
async fn echo_returns_empty_string_when_no_user_message() {
    let model = MockModel::echo();
    let request = ModelRequest::new(vec![Message::system("only system")]);
    let response = model.invoke(&NoState, request).await.unwrap();
    assert_eq!(response.text(), "");
}

// ---------------------------------------------------------------------------
// MockModel::constant
// ---------------------------------------------------------------------------

#[tokio::test]
async fn constant_always_returns_fixed_text() {
    let model = MockModel::constant("always this");
    for _ in 0..3 {
        let response = model
            .invoke(&NoState, ModelRequest::new(vec![Message::user("anything")]))
            .await
            .unwrap();
        assert_eq!(response.text(), "always this");
    }
}

// ---------------------------------------------------------------------------
// MockModel::with_responses (scripted sequence)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scripted_returns_responses_in_order() {
    let model = MockModel::with_responses(vec![
        MockModel::text_response("first"),
        MockModel::text_response("second"),
        MockModel::text_response("third"),
    ]);

    let r1 = model
        .invoke(&NoState, ModelRequest::new(vec![]))
        .await
        .unwrap();
    let r2 = model
        .invoke(&NoState, ModelRequest::new(vec![]))
        .await
        .unwrap();
    let r3 = model
        .invoke(&NoState, ModelRequest::new(vec![]))
        .await
        .unwrap();

    assert_eq!(r1.text(), "first");
    assert_eq!(r2.text(), "second");
    assert_eq!(r3.text(), "third");

    // After exhaustion the sequence cycles back to the first response.
    let r4 = model
        .invoke(&NoState, ModelRequest::new(vec![]))
        .await
        .unwrap();
    assert_eq!(
        r4.text(),
        "first",
        "scripted sequence should cycle after exhaustion"
    );
}

#[test]
#[should_panic(expected = "responses must not be empty")]
fn scripted_panics_on_empty_vec() {
    MockModel::with_responses(vec![]);
}

// ---------------------------------------------------------------------------
// MockModel::with_tool_call
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_call_response_carries_correct_fields() {
    let model = MockModel::with_tool_call("search", json!({"query": "rust agents"}));
    let response = model
        .invoke(&NoState, ModelRequest::new(vec![Message::user("go")]))
        .await
        .unwrap();

    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(
        response.text(),
        "",
        "tool-call response should have no text content"
    );

    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[0].arguments["query"], "rust agents");
    assert!(
        !calls[0].id.is_empty(),
        "tool call must have a non-empty id"
    );
}

// ---------------------------------------------------------------------------
// call_count
// ---------------------------------------------------------------------------

#[tokio::test]
async fn call_count_tracks_invocations() {
    let model = MockModel::echo();
    assert_eq!(model.call_count(), 0);

    model
        .invoke(&NoState, ModelRequest::new(vec![Message::user("a")]))
        .await
        .unwrap();
    assert_eq!(model.call_count(), 1);

    model
        .invoke(&NoState, ModelRequest::new(vec![Message::user("b")]))
        .await
        .unwrap();
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn stream_also_increments_call_count() {
    let model = MockModel::constant("hello");
    assert_eq!(model.call_count(), 0);

    model
        .stream(&NoState, ModelRequest::new(vec![Message::user("x")]))
        .await
        .unwrap();
    assert_eq!(
        model.call_count(),
        1,
        "stream should increment call_count via invoke"
    );
}

// ---------------------------------------------------------------------------
// Streaming deltas
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_splits_text_into_two_deltas() {
    let model = MockModel::constant("hello world");
    let deltas = model
        .stream(&NoState, ModelRequest::new(vec![Message::user("hi")]))
        .await
        .unwrap();

    assert_eq!(deltas.len(), 2, "constant text should produce two deltas");

    let combined: String = deltas.iter().map(|d| d.content.as_str()).collect();
    assert_eq!(
        combined, "hello world",
        "deltas should reconstruct the full text"
    );

    // Both deltas share the same call_id.
    assert_eq!(deltas[0].call_id, deltas[1].call_id);
}

#[tokio::test]
async fn stream_tool_call_response_returns_single_empty_delta() {
    let model = MockModel::with_tool_call("do_thing", json!({}));
    let deltas = model
        .stream(&NoState, ModelRequest::new(vec![Message::user("run")]))
        .await
        .unwrap();

    assert_eq!(
        deltas.len(),
        1,
        "tool-call responses have no text → one empty delta"
    );
    assert_eq!(deltas[0].content, "");
}

// ---------------------------------------------------------------------------
// Usage estimates are non-zero
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_is_attached_to_echo_response() {
    let model = MockModel::echo();
    let response = model
        .invoke(
            &NoState,
            ModelRequest::new(vec![Message::user("hello world")]),
        )
        .await
        .unwrap();

    let usage = response.usage.expect("echo should attach usage");
    assert!(usage.input_tokens > 0, "input_tokens should be non-zero");
    assert!(usage.output_tokens > 0, "output_tokens should be non-zero");
    assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
}

#[tokio::test]
async fn usage_is_attached_to_tool_call_response() {
    let model = MockModel::with_tool_call("noop", json!(null));
    let response = model
        .invoke(&NoState, ModelRequest::new(vec![Message::user("go")]))
        .await
        .unwrap();

    let usage = response
        .usage
        .expect("tool-call response should attach usage");
    assert!(usage.input_tokens > 0);
    // output_tokens is a fixed estimate of 5 for tool calls
    assert_eq!(usage.output_tokens, 5);
}

// ---------------------------------------------------------------------------
// Message id stamping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_have_non_empty_message_id() {
    for model in [
        MockModel::echo(),
        MockModel::constant("x"),
        MockModel::with_tool_call("t", json!({})),
    ] {
        let response = model
            .invoke(&NoState, ModelRequest::new(vec![Message::user("ping")]))
            .await
            .unwrap();
        assert!(
            response
                .message
                .id
                .as_deref()
                .map(|s| !s.is_empty())
                .unwrap_or(false),
            "every response should carry a non-empty message id"
        );
    }
}
