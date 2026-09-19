//! LIVE end-to-end: stream a short real OpenAI completion and assert that
//! incremental deltas arrived and the merged final text is non-empty.
//!
//! This drives [`ChatModel::stream`] directly against a real
//! [`OpenAiModel`], folding the [`ModelStreamItem`]s into a final
//! [`ModelResponse`] with a [`StreamAccumulator`] while counting the message
//! deltas observed along the way.
//!
//! # Skips gracefully
//!
//! This test is `#[ignore]`d and only runs opted in via
//! `tests/common/live.rs::require_live`, so `cargo test` passes with no key
//! configured and never dials a real provider by accident.

mod common;

#[tokio::test]
#[ignore = "network: set TINYAGENTS_LIVE=1 and run with --ignored"]
async fn live_openai_streams_deltas_and_final_text() {
    use futures::StreamExt;

    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::{ChatModel, ModelRequest, ModelStreamItem, StreamAccumulator};
    use tinyinference_llm::providers::openai::OpenAiModel;

    if !common::live::require_live(&["OPENAI_API_KEY"]) {
        return;
    }

    let model = OpenAiModel::from_env().expect("OPENAI_API_KEY present");

    let request = ModelRequest {
        messages: vec![Message::user("Reply with exactly the single word: hello")],
        max_tokens: Some(16),
        ..ModelRequest::default()
    };

    let mut stream = model
        .stream(&(), request)
        .await
        .expect("opening the live stream succeeds");

    let mut delta_count = 0usize;
    let mut accumulator = StreamAccumulator::new();
    while let Some(item) = stream.next().await {
        if matches!(item, ModelStreamItem::MessageDelta(_)) {
            delta_count += 1;
        }
        accumulator.push(&item);
    }

    let response = accumulator.finish().expect("stream merges into a response");

    assert!(
        delta_count > 0,
        "expected at least one streamed message delta, got {delta_count}"
    );
    assert!(
        !response.text().trim().is_empty(),
        "expected non-empty final streamed text"
    );
}
