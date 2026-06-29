//! End-to-end coverage for the streaming invoke path and cooperative
//! cancellation, exercised entirely offline through the public harness API.
//!
//! These tests drive [`AgentHarness::invoke_streaming`] with a deterministic
//! [`StreamingMock`] so no network access is required. They assert that:
//!
//! - every streamed message delta fires the
//!   [`Middleware::on_model_delta`] hook (once per delta), and
//! - the deltas accumulate back into the correct final response, and
//! - a run started with a pre-cancelled [`CancellationToken`] unwinds with
//!   [`TinyAgentsError::Cancelled`] before any model call happens.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use tinyagents::CancellationToken;
use tinyagents::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::message::Message;
use tinyagents::harness::middleware::Middleware;
use tinyagents::harness::model::ModelDelta;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, StreamingMock};

/// Middleware that records the text of every `on_model_delta` it observes.
struct DeltaRecorder {
    texts: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<(), ()> for DeltaRecorder {
    fn name(&self) -> &str {
        "delta-recorder"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut ModelDelta,
    ) -> tinyagents::Result<()> {
        self.texts.lock().unwrap().push(delta.content.clone());
        Ok(())
    }
}

#[tokio::test]
async fn streaming_invoke_fires_on_model_delta_and_accumulates_final_response() {
    let texts = Arc::new(Mutex::new(Vec::new()));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::from_text_chunks(["Hel", "lo, ", "world"])),
    );
    harness.push_middleware(Arc::new(DeltaRecorder {
        texts: texts.clone(),
    }));

    // Subscribe an event recorder so we can also confirm `model.delta` events.
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("stream-run"), ()).with_events(recorder.sink());

    let run = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("streaming run succeeds");

    // The merged response equals the concatenated chunks.
    assert_eq!(run.model_calls, 1);
    assert_eq!(run.text(), Some("Hello, world".to_string()));

    // on_model_delta fired exactly once per streamed message delta, in order.
    assert_eq!(
        *texts.lock().unwrap(),
        vec!["Hel".to_string(), "lo, ".to_string(), "world".to_string()]
    );

    // One observability `model.delta` event per streamed delta.
    let delta_events = recorder
        .kinds()
        .into_iter()
        .filter(|k| k == "model.delta")
        .count();
    assert_eq!(delta_events, 3, "one model.delta event per streamed delta");
}

#[tokio::test]
async fn pre_cancelled_token_yields_cancelled_on_streaming_path() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "stream",
        Arc::new(StreamingMock::from_text_chunks(["never", "delivered"])),
    );

    // Pre-cancel the token before the run starts.
    let token = CancellationToken::new();
    token.cancel();
    let ctx = RunContext::new(RunConfig::new("cancel-stream"), ()).with_cancellation(token);

    let err = harness
        .invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect_err("a pre-cancelled streaming run must not complete");

    assert!(matches!(err, TinyAgentsError::Cancelled), "got {err:?}");
}
