//! Wave 2 — response-cache **key** regressions.
//!
//! Covers CACHE-1 (the key carried no provider/model identity), CACHE-4's key
//! half (`streaming` was not in the key), CACHE-7 (the envelope was
//! over-inclusive) and CACHE-8 (a non-array `messages`/`tools` was dropped
//! without being hashed).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use tinyagents_harness::Result;
use tinyagents_harness::cache::{
    CachePolicy, InMemoryResponseCache, cache_key, credential_fingerprint, model_cache_identity,
    scoped_cache_key,
};
use tinyinference::message::Message;
use tinyinference::model::{ChatModel, ModelRequest, ModelResponse};
use tinyagents_harness::runtime::AgentHarness;

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// A model that always answers `answer` and reports a distinct cache identity.
struct IdentifiedModel {
    identity: String,
    answer: String,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ChatModel<()> for IdentifiedModel {
    fn cache_identity(&self) -> Option<String> {
        Some(self.identity.clone())
    }

    async fn invoke(&self, _state: &(), _request: ModelRequest) -> tinyinference::Result<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ModelResponse::assistant(self.answer.clone()))
    }
}

fn request(prompt: &str) -> ModelRequest {
    ModelRequest::new(vec![Message::user(prompt)])
}

// ── CACHE-1 ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn shared_cache_does_not_cross_serve_between_two_providers() {
    // Two harnesses — think "hosted" and "local" — sharing one cache, asked the
    // same question. Before the identity was folded into the key, the second
    // harness was served the first harness's answer: `request.model` is never
    // set by the loop, and the endpoint/credentials live inside the
    // `Arc<dyn ChatModel>`, so nothing in the key distinguished them.
    let cache = Arc::new(InMemoryResponseCache::new());
    let hosted_calls = Arc::new(AtomicUsize::new(0));
    let local_calls = Arc::new(AtomicUsize::new(0));

    let mut hosted: AgentHarness<()> = AgentHarness::new();
    hosted.register_model(
        "chat",
        Arc::new(IdentifiedModel {
            identity: "openai|gpt-5|https://api.openai.com/v1||abc".to_string(),
            answer: "hosted answer".to_string(),
            calls: hosted_calls.clone(),
        }),
    );
    hosted.with_response_cache(cache.clone());

    let mut local: AgentHarness<()> = AgentHarness::new();
    local.register_model(
        "chat",
        Arc::new(IdentifiedModel {
            identity: "ollama|llama3.2|http://localhost:11434/v1||no-credential".to_string(),
            answer: "local answer".to_string(),
            calls: local_calls.clone(),
        }),
    );
    local.with_response_cache(cache.clone());

    let hosted_run = hosted
        .invoke_default(&(), vec![Message::user("what is 2+2?")])
        .await
        .expect("hosted run");
    let local_run = local
        .invoke_default(&(), vec![Message::user("what is 2+2?")])
        .await
        .expect("local run");

    assert_eq!(hosted_run.text().as_deref(), Some("hosted answer"));
    assert_eq!(
        local_run.text().as_deref(),
        Some("local answer"),
        "the local harness must not be served the hosted harness's cached answer"
    );
    assert_eq!(local_calls.load(Ordering::SeqCst), 1, "local really ran");
}

#[tokio::test]
async fn identical_identity_still_shares_the_cache() {
    // The fix must not disable caching: two harnesses on the *same* model
    // identity still share entries.
    let cache = Arc::new(InMemoryResponseCache::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let identity = "openai|gpt-5|https://api.openai.com/v1||abc";

    let build = |calls: Arc<AtomicUsize>| {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model(
            "chat",
            Arc::new(IdentifiedModel {
                identity: identity.to_string(),
                answer: "same answer".to_string(),
                calls,
            }),
        );
        harness.with_response_cache(cache.clone());
        harness
    };

    let first = build(calls.clone());
    let second = build(calls.clone());

    first
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("first run");
    second
        .invoke_default(&(), vec![Message::user("q")])
        .await
        .expect("second run");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the second harness must be served from cache"
    );
}

#[test]
fn scoped_key_separates_identity_streaming_and_namespace() {
    let base = cache_key(&request("hello"));

    let a = scoped_cache_key(&base, Some("provider-a"), false, None);
    let b = scoped_cache_key(&base, Some("provider-b"), false, None);
    let anon = scoped_cache_key(&base, None, false, None);
    let streamed = scoped_cache_key(&base, Some("provider-a"), true, None);
    let namespaced = scoped_cache_key(&base, Some("provider-a"), false, Some("tenant-7"));

    assert_ne!(a, b, "two identities must not share a key");
    assert_ne!(
        a, anon,
        "an anonymous model must not collide with a named one"
    );
    assert_ne!(
        a, streamed,
        "streaming is a call parameter and must be keyed"
    );
    assert_ne!(a, namespaced, "the policy namespace must be keyed");
    assert_eq!(
        a,
        scoped_cache_key(&base, Some("provider-a"), false, None),
        "the composition must stay deterministic"
    );
    assert_eq!(a.len(), 64);
}

#[test]
fn model_identity_never_carries_the_raw_credential() {
    // The identity ends up folded into keys that reach logs, events, and
    // durable cache files, so a raw key must never survive into it.
    let secret = "sk-super-secret-value";
    let identity =
        model_cache_identity("openai", "gpt-5", "https://api.openai.com/v1", None, secret);
    assert!(
        !identity.contains(secret),
        "the raw credential leaked into the cache identity: {identity}"
    );
    assert!(identity.contains(&credential_fingerprint(secret)));
    // Two different credentials must still be distinguishable.
    assert_ne!(
        credential_fingerprint("key-one"),
        credential_fingerprint("key-two")
    );
    assert_eq!(credential_fingerprint(""), "no-credential");
}

// ── CACHE-7 ──────────────────────────────────────────────────────────────────

#[test]
fn key_ignores_fields_that_cannot_change_the_answer() {
    let base = request("hello");
    let key = cache_key(&base);

    // `metadata` is free-form and its natural use is a run id. Folding it in
    // gave such a caller a permanent 0% hit rate with no diagnostic.
    let mut with_metadata = base.clone();
    with_metadata.metadata = serde_json::json!({ "run_id": "run-1234" });
    assert_eq!(cache_key(&with_metadata), key, "metadata must not be keyed");

    // Documented as "propagated to events and traces".
    let mut with_tags = base.clone();
    with_tags.tags = vec!["trace-a".to_string()];
    assert_eq!(cache_key(&with_tags), key, "tags must not be keyed");

    // A transport deadline cannot change what the model says.
    let mut with_timeout = base.clone();
    with_timeout.timeout_ms = Some(30_000);
    assert_eq!(
        cache_key(&with_timeout),
        key,
        "timeout_ms must not be keyed"
    );

    // The policy selects *whether* to cache. Folding it in meant flipping the
    // (previously dead) `protect_prompt_prefix` flag invalidated every entry.
    let mut with_policy = base.clone();
    with_policy.cache_policy = Some(CachePolicy {
        response_cache_enabled: true,
        protect_prompt_prefix: true,
        ..CachePolicy::default()
    });
    assert_eq!(
        cache_key(&with_policy),
        key,
        "cache_policy must not be keyed"
    );

    // Derived from the messages that are already folded.
    let mut with_fingerprint = base.clone();
    with_fingerprint.prompt_fingerprint = Some("deadbeef".to_string());
    assert_eq!(
        cache_key(&with_fingerprint),
        key,
        "prompt_fingerprint is derived and must not be keyed"
    );
}

#[test]
fn key_still_reflects_every_behaviour_affecting_field() {
    let base = request("hello");
    let key = cache_key(&base);

    let mut hotter = base.clone();
    hotter.temperature = Some(0.9);
    assert_ne!(cache_key(&hotter), key, "temperature changes the answer");

    let mut capped = base.clone();
    capped.max_tokens = Some(16);
    assert_ne!(cache_key(&capped), key, "max_tokens changes the answer");

    let mut seeded = base.clone();
    seeded.seed = Some(7);
    assert_ne!(cache_key(&seeded), key, "seed changes the answer");

    let mut opted = base.clone();
    opted.provider_options = serde_json::json!({ "hotness": 3 });
    assert_ne!(cache_key(&opted), key, "provider_options change the answer");

    let mut stopped = base.clone();
    stopped.stop_sequences = vec!["STOP".to_string()];
    assert_ne!(cache_key(&stopped), key, "stop_sequences change the answer");

    let mut continued = base.clone();
    continued.continuation_id = Some("resp_123".to_string());
    assert_ne!(
        cache_key(&continued),
        key,
        "continuation_id changes provider state"
    );

    let mut named = base.clone();
    named.model = Some("model-b".to_string());
    assert_ne!(
        cache_key(&named),
        key,
        "an explicit model override is keyed"
    );

    let mut longer = base.clone();
    longer.messages.push(Message::user("and one more"));
    assert_ne!(cache_key(&longer), key, "messages are keyed");
}

// ── CACHE-8 ──────────────────────────────────────────────────────────────────

#[test]
fn every_message_and_tool_participates_in_the_key() {
    // The old envelope removed `messages`/`tools` from a serialized `Value`
    // with `map.remove(..)` *inside* an `if let Some(Value::Array(..))`: the
    // removal ran unconditionally and the value was dropped unhashed when the
    // pattern did not match. Hashing the typed fields directly removes the
    // shape assumption; assert the guarantee the docstring promised.
    let one = ModelRequest::new(vec![Message::user("a")]);
    let two = ModelRequest::new(vec![Message::user("a"), Message::user("b")]);
    let swapped = ModelRequest::new(vec![Message::user("b"), Message::user("a")]);

    assert_ne!(cache_key(&one), cache_key(&two));
    assert_ne!(cache_key(&two), cache_key(&swapped), "order is significant");

    // An empty transcript and empty tools still produce a well-defined key.
    let empty = ModelRequest::new(vec![]);
    assert_eq!(cache_key(&empty).len(), 64);
    assert_eq!(cache_key(&empty), cache_key(&ModelRequest::new(vec![])));
}
