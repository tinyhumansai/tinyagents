//! Wave 2 — agent-loop regressions around the response cache and the model
//! wrap onion.
//!
//! Covers CACHE-2 (fallback answers cached under the primary's key), CACHE-3
//! (cache hits re-billing tokens), CACHE-4 (a streaming cache hit emitted zero
//! deltas), CACHE-5 (a cache read/write failure killing the run) and LOOP-3
//! (`ModelFallbackMiddleware` could not actually switch models).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use tinyagents_graph::*;
use tinyagents_harness::cache::{InMemoryResponseCache, ResponseCache};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::middleware::ModelFallbackMiddleware;
use tinyagents_harness::retry::{FallbackPolicy, RetryPolicy};
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::EventRecorder;
use tinyagents_harness::*;
use tinyagents_language::*;
use tinyagents_registry::*;
use tinyinference::message::Message;
use tinyinference::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference::usage::Usage;

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// A model that answers with a fixed text plus a fixed usage, counting calls.
struct FixedModel {
    identity: &'static str,
    answer: &'static str,
    usage: Option<Usage>,
    calls: Arc<AtomicUsize>,
}

impl FixedModel {
    fn new(identity: &'static str, answer: &'static str, calls: Arc<AtomicUsize>) -> Self {
        Self {
            identity,
            answer,
            usage: Some(Usage::new(100, 50)),
            calls,
        }
    }
}

#[async_trait]
impl ChatModel<()> for FixedModel {
    fn cache_identity(&self) -> Option<String> {
        Some(self.identity.to_string())
    }

    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference::Result<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut response = ModelResponse::assistant(self.answer);
        response.usage = self.usage;
        Ok(response)
    }
}

/// A model that always fails with a retryable provider-shaped error.
struct AlwaysFailing {
    identity: &'static str,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ChatModel<()> for AlwaysFailing {
    fn cache_identity(&self) -> Option<String> {
        Some(self.identity.to_string())
    }

    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference::Result<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(tinyinference::Error::Model(
            "openai returned HTTP 503: service unavailable".to_string(),
        ))
    }
}

/// A [`ResponseCache`] whose every operation fails, standing in for a poisoned
/// mutex or an unavailable third-party backend.
struct BrokenCache {
    gets: Arc<AtomicUsize>,
    puts: Arc<AtomicUsize>,
}

#[async_trait]
impl ResponseCache for BrokenCache {
    async fn get(&self, _key: &str) -> Result<Option<ModelResponse>> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        Err(TinyAgentsError::Validation(
            "cache lock poisoned".to_string(),
        ))
    }

    async fn put(&self, _key: &str, _value: ModelResponse) -> Result<()> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        Err(TinyAgentsError::Validation(
            "cache lock poisoned".to_string(),
        ))
    }
}

// ── CACHE-2 ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_fallback_answer_is_never_cached_under_the_primary_key() {
    // The primary always fails, so the harness-level fallback chain answers.
    // Writing that answer under the primary's key poisons it permanently (no
    // TTL), so every later run of the primary silently gets the fallback's
    // answer while `ModelStarted` announced the primary.
    let primary_calls = Arc::new(AtomicUsize::new(0));
    let backup_calls = Arc::new(AtomicUsize::new(0));
    let cache = Arc::new(InMemoryResponseCache::new());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "primary",
        Arc::new(AlwaysFailing {
            identity: "primary-identity",
            calls: primary_calls.clone(),
        }),
    );
    harness.register_model(
        "backup",
        Arc::new(FixedModel::new(
            "backup-identity",
            "backup answer",
            backup_calls.clone(),
        )),
    );
    harness.set_default_model("primary");
    harness.with_response_cache(cache.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(1),
        fallback: Some(FallbackPolicy {
            models: vec!["primary".to_string(), "backup".to_string()],
        }),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("run falls back to the backup");
    assert_eq!(run.text().as_deref(), Some("backup answer"));

    // The backup's answer must not be sitting under the primary's key: a second
    // run has to try the primary again (and fall back again).
    let before = primary_calls.load(Ordering::SeqCst);
    let run2 = harness
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("second run");
    assert_eq!(run2.text().as_deref(), Some("backup answer"));
    assert!(
        primary_calls.load(Ordering::SeqCst) > before,
        "the primary must be retried; a fallback answer must not be cached under its key"
    );
    assert_eq!(
        cache.stats().writes,
        0,
        "no fallback response may be written under the primary's key"
    );
}

// ── CACHE-3 ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_cache_hit_is_marked_served_from_cache() {
    // Accounting needs to tell a replay from a real call: the cached response
    // retains the provider's `usage`, and re-billing it prices spend that never
    // happened (which a cost budget can abort a run over).
    let calls = Arc::new(AtomicUsize::new(0));
    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "chat",
        Arc::new(FixedModel::new("id", "answer", calls.clone())),
    );
    harness.with_response_cache(cache);

    let cold = harness
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("cold run");
    let warm = harness
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("warm run");

    assert_eq!(calls.load(Ordering::SeqCst), 1, "warm run hit the cache");
    let cold_response = cold.final_response.expect("cold response");
    let warm_response = warm.final_response.expect("warm response");
    assert!(
        !cold_response.served_from_cache,
        "a real provider call is not served from cache"
    );
    assert!(
        warm_response.served_from_cache,
        "a cache hit must be marked so accounting does not re-bill its tokens"
    );
    // The usage itself is preserved so a caller can still inspect it; only the
    // accounting sites are expected to skip it.
    assert!(warm_response.usage.is_some());
}

// ── CACHE-4 ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_streaming_cache_hit_still_emits_deltas() {
    // A hit returned before the streaming path was reached, so a warm streaming
    // run emitted zero `ModelDelta` events — a UI concatenating deltas rendered
    // nothing at all, contradicting the documented streaming contract.
    let calls = Arc::new(AtomicUsize::new(0));
    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "chat",
        Arc::new(FixedModel::new("id", "streamed answer", calls.clone())),
    );
    harness.with_response_cache(cache);

    let cold_events = EventRecorder::new();
    let cold_ctx =
        RunContext::new(RunConfig::new("stream-cold"), ()).with_events(cold_events.sink());
    harness
        .invoke_streaming_in_context(&(), cold_ctx, vec![Message::user("q")])
        .await
        .expect("cold streaming run");

    let warm_events = EventRecorder::new();
    let warm_ctx =
        RunContext::new(RunConfig::new("stream-warm"), ()).with_events(warm_events.sink());
    let warm = harness
        .invoke_streaming_in_context(&(), warm_ctx, vec![Message::user("q")])
        .await
        .expect("warm streaming run");

    assert_eq!(calls.load(Ordering::SeqCst), 1, "warm run hit the cache");
    assert_eq!(warm.text().as_deref(), Some("streamed answer"));

    let cold_deltas = cold_events
        .kinds()
        .iter()
        .filter(|k| *k == "model.delta")
        .count();
    let warm_deltas = warm_events
        .kinds()
        .iter()
        .filter(|k| *k == "model.delta")
        .count();
    assert!(cold_deltas > 0, "cold streaming run emits deltas");
    assert!(
        warm_deltas > 0,
        "a warm streaming run must replay the cached response as deltas so warm and \
         cold runs are observationally identical"
    );
    assert!(
        warm_events.kinds().iter().any(|k| k == "cache.hit"),
        "the warm run really was served from cache"
    );
}

#[tokio::test]
async fn a_streaming_run_does_not_reuse_a_unary_cache_entry() {
    // `streaming` is a parameter of the call, not a field of the request, so it
    // never reached the key. Deliberate sharing is a caller decision, not a
    // silent default.
    let calls = Arc::new(AtomicUsize::new(0));
    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "chat",
        Arc::new(FixedModel::new("id", "answer", calls.clone())),
    );
    harness.with_response_cache(cache);

    harness
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("unary run");
    harness
        .invoke_streaming_default(&(), vec![Message::user("q")])
        .await
        .expect("streaming run");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a streaming call must not be served an entry written by a unary call"
    );
}

// ── CACHE-5 ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_failing_cache_never_fails_the_run() {
    // Both the read and the write used `?`. The write case is the worse one:
    // the provider call already succeeded and was paid for, and its answer was
    // discarded because the cache was unavailable.
    let gets = Arc::new(AtomicUsize::new(0));
    let puts = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "chat",
        Arc::new(FixedModel::new("id", "answer", calls.clone())),
    );
    harness.with_response_cache(Arc::new(BrokenCache {
        gets: gets.clone(),
        puts: puts.clone(),
    }));

    let run = harness
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("a broken cache must not fail the run");
    assert_eq!(run.text().as_deref(), Some("answer"));
    assert_eq!(gets.load(Ordering::SeqCst), 1, "the read was attempted");
    assert_eq!(puts.load(Ordering::SeqCst), 1, "the write was attempted");
}

// ── LOOP-3 ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn model_fallback_middleware_actually_switches_models() {
    // This is deliberately driven through the REAL `ModelCallBase` (the
    // innermost base of the model-wrap onion) rather than a `FakeModelBase`
    // that dispatches on `req.model`. Every pre-existing test used such a fake,
    // which is exactly why the bug shipped: the real base rebuilt its binding
    // from fields captured *before* the wrap onion ran and never re-resolved
    // `request.model`, so the "fallback" re-invoked the same failing model once
    // per configured fallback name and returned the same error.
    let primary_calls = Arc::new(AtomicUsize::new(0));
    let backup_calls = Arc::new(AtomicUsize::new(0));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "primary",
        Arc::new(AlwaysFailing {
            identity: "primary-identity",
            calls: primary_calls.clone(),
        }),
    );
    harness.register_model(
        "backup",
        Arc::new(FixedModel::new(
            "backup-identity",
            "backup answer",
            backup_calls.clone(),
        )),
    );
    harness.set_default_model("primary");
    // No harness-level `FallbackPolicy` — the switch must come from the wrap
    // middleware alone, which steers by mutating `request.model`.
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });
    harness.push_model_middleware(Arc::new(ModelFallbackMiddleware::new(["backup"])));

    let events = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("fallback"), ()).with_events(events.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("q")])
        .await
        .expect("the wrap-layer fallback must reach the backup model");

    assert_eq!(run.text().as_deref(), Some("backup answer"));
    assert!(
        backup_calls.load(Ordering::SeqCst) >= 1,
        "the backup model must actually be invoked, not merely announced"
    );
    assert!(
        events
            .kinds()
            .iter()
            .any(|k| k == "model.fallback_selected"),
        "the fallback event is still emitted"
    );
}

#[tokio::test]
async fn an_unresolvable_wrap_override_keeps_the_resolved_binding() {
    // Fail-closed: naming a model the registry cannot resolve must not silently
    // substitute a different one — it keeps the resolved binding and makes the
    // skip observable, matching what `run_loop` already does for a pre-wrap
    // override.
    let primary_calls = Arc::new(AtomicUsize::new(0));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "primary",
        Arc::new(AlwaysFailing {
            identity: "primary-identity",
            calls: primary_calls.clone(),
        }),
    );
    harness.set_default_model("primary");
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });
    harness.push_model_middleware(Arc::new(ModelFallbackMiddleware::new(["nonexistent"])));

    let events = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("bad-override"), ()).with_events(events.sink());
    let outcome = harness
        .invoke_in_context(&(), ctx, vec![Message::user("q")])
        .await;

    assert!(outcome.is_err(), "no model could answer");
    assert!(
        events.kinds().iter().any(|k| k == "model.override_skipped"),
        "an unresolvable wrap-layer override must be observable"
    );
}
