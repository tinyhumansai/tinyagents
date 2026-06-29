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
//! The OpenAI-using code is gated behind `#[cfg(feature = "openai")]`, and the
//! test returns early (after an `eprintln!`) when `OPENAI_API_KEY` is unset, so
//! `cargo test --features openai` passes with no key configured. Without the
//! `openai` feature the test is a trivial no-op.

#[cfg(feature = "openai")]
#[tokio::test]
async fn live_openai_streams_deltas_and_final_text() {
    use futures::StreamExt;

    use tinyagents::harness::message::Message;
    use tinyagents::harness::model::{ChatModel, ModelRequest, ModelStreamItem, StreamAccumulator};
    use tinyagents::harness::providers::openai::OpenAiModel;

    // Load .env so `cargo test` picks up local credentials.
    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping live_openai_streams_deltas_and_final_text: OPENAI_API_KEY is not set");
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
