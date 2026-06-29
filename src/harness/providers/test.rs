//! Tests for the `providers` module.
//!
//! All tests are deterministic and have no network dependencies.

use futures::StreamExt;
use serde_json::json;

use crate::harness::message::{Message, MessageDelta};
use crate::harness::model::{ChatModel, ModelRequest, ModelStreamItem};
use crate::harness::providers::{MockModel, ProviderKind, ProviderSpec};

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

    let stream = model
        .stream(&NoState, ModelRequest::new(vec![Message::user("x")]))
        .await
        .unwrap();
    let _items: Vec<ModelStreamItem> = stream.collect().await;
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
    let stream = model
        .stream(&NoState, ModelRequest::new(vec![Message::user("hi")]))
        .await
        .unwrap();
    let items: Vec<ModelStreamItem> = stream.collect().await;

    let message_deltas: Vec<String> = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        message_deltas.len(),
        2,
        "constant text should produce two message deltas"
    );

    let combined: String = message_deltas.concat();
    assert_eq!(
        combined, "hello world",
        "deltas should reconstruct the full text"
    );

    assert!(matches!(items.first(), Some(ModelStreamItem::Started)));
    assert!(matches!(items.last(), Some(ModelStreamItem::Completed(_))));
}

#[tokio::test]
async fn stream_tool_call_response_returns_single_empty_delta() {
    let model = MockModel::with_tool_call("do_thing", json!({}));
    let stream = model
        .stream(&NoState, ModelRequest::new(vec![Message::user("run")]))
        .await
        .unwrap();
    let items: Vec<ModelStreamItem> = stream.collect().await;

    let message_deltas: Vec<&MessageDelta> = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta),
            _ => None,
        })
        .collect();
    assert_eq!(
        message_deltas.len(),
        1,
        "tool-call responses have no text → one empty delta"
    );
    assert_eq!(message_deltas[0].text, "");
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

#[test]
fn provider_kind_infers_langchain_style_model_names() {
    assert_eq!(
        ProviderKind::infer("openai:gpt-4.1-mini"),
        Some(ProviderKind::OpenAi)
    );
    assert_eq!(
        ProviderKind::infer("anthropic:claude-sonnet-4"),
        Some(ProviderKind::Anthropic)
    );
    assert_eq!(
        ProviderKind::infer("ollama:llama3.2"),
        Some(ProviderKind::Ollama)
    );
    assert_eq!(
        ProviderKind::infer("gpt-4.1-mini"),
        Some(ProviderKind::OpenAi)
    );
    assert_eq!(
        ProviderKind::infer("claude-sonnet-4"),
        Some(ProviderKind::Anthropic)
    );
    assert_eq!(
        ProviderKind::infer("mistral-small-latest"),
        Some(ProviderKind::Mistral)
    );
    assert_eq!(ProviderKind::infer("unknown-model"), None);
}

#[test]
fn provider_spec_defaults_and_overrides_are_normalized() {
    let spec = ProviderSpec::for_kind(ProviderKind::Ollama)
        .with_model("qwen2.5")
        .with_base_url("http://localhost:11434/v1/")
        .with_provider("local-ollama");

    assert_eq!(spec.kind, ProviderKind::Ollama);
    assert_eq!(spec.provider, "local-ollama");
    assert_eq!(spec.model, "qwen2.5");
    assert_eq!(spec.base_url, "http://localhost:11434/v1");
    assert!(!spec.requires_api_key);

    let openai = ProviderSpec::for_kind(ProviderKind::OpenAi);
    assert_eq!(openai.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
    assert!(openai.requires_api_key);
}
