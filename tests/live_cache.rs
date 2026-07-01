//! LIVE end-to-end: prove the harness response cache captures a real provider
//! response and serves an identical follow-up request entirely from cache.
//!
//! A small counting [`ChatModel`] decorator wraps a real [`OpenAiModel`] and
//! increments an [`AtomicUsize`] on every `invoke`/`stream` call. The decorator
//! is registered on an [`AgentHarness`] that also has an
//! [`InMemoryResponseCache`] attached. Running the *same* question twice must:
//!
//! 1. hit the real OpenAI API exactly **once** (the decorator's count is 1),
//! 2. return the same answer text on both runs, and
//! 3. emit a `cache.hit` event on the second run (captured via an
//!    [`EventRecorder`] attached through `invoke_in_context`).
//!
//! # Skips gracefully
//!
//! The whole test returns early (after an `eprintln!`) when `OPENAI_API_KEY`
//! is unset, so the default `cargo test` passes with no key configured.


use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use tinyagents::Result;
use tinyagents::harness::cache::InMemoryResponseCache;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::message::Message;
use tinyagents::harness::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStream,
};
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::EventRecorder;

/// Wraps an inner [`ChatModel`] and counts how many times the *underlying*
/// provider is actually contacted via `invoke` or `stream`.
struct CountingModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for CountingModel<State> {
    fn profile(&self) -> Option<&ModelProfile> {
        self.inner.profile()
    }

    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.invoke(state, request).await
    }

    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.stream(state, request).await
    }
}

#[tokio::test]
async fn live_openai_response_cache_hits_on_repeated_question() {
    // Load .env so `cargo test` picks up local credentials.
    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!(
            "skipping live_openai_response_cache_hits_on_repeated_question: \
             OPENAI_API_KEY is not set"
        );
        return;
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let inner: Arc<dyn ChatModel<()>> =
        Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present"));
    let counting = Arc::new(CountingModel {
        inner,
        calls: calls.clone(),
    });

    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("openai", counting);
    harness.with_response_cache(cache);

    // A short, deterministic question. The harness builds the `ModelRequest`
    // internally; identical inputs yield the same first request and thus the
    // same cache key, so the second run is served from cache.
    let question = || {
        vec![Message::user(
            "Reply with exactly the single lowercase word: hello",
        )]
    };

    // First run: real API call, cached afterwards.
    let recorder1 = EventRecorder::new();
    let ctx1 = RunContext::new(RunConfig::new("live-cache-1"), ()).with_events(recorder1.sink());
    let run1 = harness
        .invoke_in_context(&(), ctx1, question())
        .await
        .expect("first live run succeeds");
    let answer1 = run1.text().expect("first run produced text");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the real OpenAI API must be hit exactly once on the first run"
    );
    assert!(
        recorder1.kinds().iter().any(|k| k == "cache.miss"),
        "first run should record a cache miss"
    );

    // Second run with the SAME input: served from cache, no new API call.
    let recorder2 = EventRecorder::new();
    let ctx2 = RunContext::new(RunConfig::new("live-cache-2"), ()).with_events(recorder2.sink());
    let run2 = harness
        .invoke_in_context(&(), ctx2, question())
        .await
        .expect("second live run succeeds");
    let answer2 = run2.text().expect("second run produced text");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the second run must be served from cache without a new API call"
    );
    assert_eq!(
        answer1, answer2,
        "the cached response text must match the first run"
    );
    assert!(
        recorder2.kinds().iter().any(|k| k == "cache.hit"),
        "second run should record a cache hit"
    );
}
