//! End-to-end contract coverage for public support APIs that recursive graph
//! and harness runs depend on.
//!
//! These tests intentionally sit in `tests/` rather than module-local unit tests
//! so the e2e coverage pass exercises the same public surface downstream users
//! call: cache keys, prompt assembly, memory/store persistence, retry policies,
//! channel reducers, and deterministic summarization.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use tinyagents::TinyAgentsError;
use tinyagents::graph::{
    Barrier, BinaryAggregate, ChannelSet, ChannelState, ChannelUpdate, Delta, Ephemeral, LastValue,
    Messages, NamedBarrier, Topic, Untracked,
};
use tinyagents::harness::cache::{
    CacheLayoutEvent, InMemoryResponseCache, PromptCacheLayout, ResponseCache, cache_key,
};
use tinyagents::harness::memory::{
    ChatHistory, InMemoryChatHistory, MemoryScope, ShortTermMemory, StoreChatHistory,
};
use tinyagents::harness::message::Message;
use tinyagents::harness::model::{ModelRequest, ModelResponse, ResponseFormat};
use tinyagents::harness::prompt::{MessagesTemplate, PromptBuilder, PromptTemplate, TemplateRole};
use tinyagents::harness::retry::{FallbackPolicy, RateLimiter, RetryPolicy, is_retryable};
use tinyagents::harness::store::{
    AppendStore, FileStore, InMemoryAppendStore, InMemoryStore, JsonlAppendStore, Store,
    StoreRegistry,
};
use tinyagents::harness::summarization::{
    ConcatSummarizer, SummarizationPolicy, Summarizer, TrimStrategy, estimate_tokens, trim_messages,
};
use tinyagents::harness::tool::ToolSchema;

#[tokio::test]
async fn cache_and_prompt_contracts_produce_stable_behavior_keys() {
    let mut vars = Map::new();
    vars.insert("task".to_string(), json!("classify"));
    vars.insert("count".to_string(), json!(3));

    let system = PromptTemplate::new("Task: {task}; keep {{literal}}; count={count}");
    assert_eq!(
        system.render(&vars).expect("template renders"),
        "Task: classify; keep {literal}; count=3"
    );
    assert!(PromptTemplate::new("missing {name}").render(&vars).is_err());

    let mut messages = MessagesTemplate::new();
    messages
        .push(TemplateRole::System, system)
        .push(TemplateRole::User, PromptTemplate::new("input {task}"));
    let rendered = messages.render(&vars).expect("messages render");
    assert_eq!(rendered.len(), 2);
    assert!(matches!(rendered[0], Message::System(_)));
    assert!(matches!(rendered[1], Message::User(_)));

    let tool = ToolSchema::new(
        "lookup",
        "lookup records",
        json!({
            "type": "object",
            "required": ["q"],
            "properties": { "q": { "type": "string" } }
        }),
    );
    let mut builder = PromptBuilder::new();
    builder
        .push_system("sys", vec![rendered[0].clone()])
        .push_tools_segment("tools", vec![tool])
        .push_history("history", vec![Message::assistant("prior answer")])
        .push_volatile("retrieval", vec![Message::system("retrieved context")])
        .with_response_format(ResponseFormat::JsonObject);
    let req = builder
        .build(vec![Message::user("now")])
        .with_model("model-a");

    assert_eq!(req.messages.len(), 4);
    assert_eq!(req.cacheable_prefix_ids(), vec!["sys", "tools"]);
    assert_eq!(req.tools.len(), 1);
    assert_eq!(req.response_format, Some(ResponseFormat::JsonObject));
    assert!(req.prompt_fingerprint.is_some());

    let key = cache_key(&req);
    assert_eq!(key.len(), 64);
    assert_eq!(key, cache_key(&req));
    assert_ne!(key, cache_key(&req.clone().with_model("model-b")));

    let layout = PromptCacheLayout::from_request(&req);
    assert_eq!(layout.prefix_ids(), &["sys", "tools"]);
    assert_eq!(layout.fingerprint().len(), 16);

    let changed = ModelRequest::new(vec![]).with_cache_segments(vec![]);
    let changed_layout = PromptCacheLayout::from_request(&changed);
    let event = CacheLayoutEvent::new(&layout, &changed_layout);
    assert!(event.changed_prefix);
    assert!(event.volatile_only);

    let cache = InMemoryResponseCache::new();
    assert!(cache.get(&key).await.expect("cache get").is_none());
    cache
        .put(&key, ModelResponse::assistant("cached"))
        .await
        .expect("cache put");
    assert_eq!(
        cache
            .get(&key)
            .await
            .expect("cache get")
            .expect("cache hit")
            .text(),
        "cached"
    );
}

#[tokio::test]
async fn stores_and_memory_round_trip_across_ephemeral_and_file_backends() {
    let memory_store = InMemoryStore::new();
    memory_store
        .put("ns", "a", json!({ "value": 1 }))
        .await
        .expect("put");
    memory_store
        .put("ns", "b", json!({ "value": 2 }))
        .await
        .expect("put");
    let mut keys = memory_store.list("ns").await.expect("list");
    keys.sort();
    assert_eq!(keys, vec!["a", "b"]);
    assert_eq!(
        memory_store
            .get("ns", "a")
            .await
            .expect("get")
            .expect("value"),
        json!({ "value": 1 })
    );
    memory_store.delete("ns", "a").await.expect("delete");
    assert!(memory_store.get("ns", "a").await.expect("get").is_none());

    let dir =
        std::env::temp_dir().join(format!("tinyagents-e2e-public-api-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("tempdir");
    let file_store = FileStore::new(dir.join("kv"));
    file_store.put("safe", "key", json!("value")).await.unwrap();
    assert_eq!(
        file_store.get("safe", "key").await.unwrap(),
        Some(json!("value"))
    );
    assert!(file_store.put("../bad", "key", json!(null)).await.is_err());

    let append = InMemoryAppendStore::new();
    assert_eq!(append.append("events", json!("a")).await.unwrap(), 0);
    assert_eq!(append.append("events", json!("b")).await.unwrap(), 1);
    assert_eq!(append.len("events").await.unwrap(), 2);
    assert_eq!(
        append.read_from("events", 1).await.unwrap(),
        vec![(1, json!("b"))]
    );

    let jsonl_a = JsonlAppendStore::new(dir.join("append"));
    assert_eq!(
        jsonl_a.append("stream", json!({ "n": 1 })).await.unwrap(),
        0
    );
    let jsonl_b = JsonlAppendStore::new(dir.join("append"));
    assert_eq!(
        jsonl_b.append("stream", json!({ "n": 2 })).await.unwrap(),
        1
    );
    assert_eq!(jsonl_a.len("stream").await.unwrap(), 2);
    assert!(jsonl_a.append("../escape", json!(0)).await.is_err());

    let mut registry = StoreRegistry::new();
    registry.register("file", Arc::new(file_store.clone()));
    assert!(registry.get("file").is_some());
    registry
        .default_store()
        .put("default", "x", json!(true))
        .await
        .unwrap();
    assert_eq!(
        registry.default_store().get("default", "x").await.unwrap(),
        Some(json!(true))
    );

    assert_eq!(
        serde_json::to_value(MemoryScope::ShortTerm).unwrap(),
        "short_term"
    );
    let history = InMemoryChatHistory::new();
    history.append("t1", Message::user("hello")).await.unwrap();
    history
        .append("t1", Message::assistant("world"))
        .await
        .unwrap();
    assert_eq!(history.messages("t1").await.unwrap().len(), 2);

    let memory = ShortTermMemory::new(history, "t1").with_trim(|messages| {
        messages
            .into_iter()
            .rev()
            .take(1)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    });
    assert_eq!(memory.thread_id(), "t1");
    assert_eq!(memory.load().await.unwrap().len(), 1);
    memory
        .save(vec![Message::user("old"), Message::assistant("new")])
        .await
        .unwrap();
    assert_eq!(memory.load().await.unwrap()[0].text(), "new");
    memory.clear().await.unwrap();
    assert!(memory.load().await.unwrap().is_empty());

    let store_history = StoreChatHistory::new(InMemoryStore::new());
    store_history
        .append("persisted", Message::user("stored"))
        .await
        .unwrap();
    assert_eq!(
        store_history.messages("persisted").await.unwrap()[0].text(),
        "stored"
    );
    assert!(
        store_history
            .store()
            .get(StoreChatHistory::<InMemoryStore>::NAMESPACE, "persisted")
            .await
            .unwrap()
            .is_some()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn channel_state_contracts_handle_merges_barriers_and_conflicts() {
    let mut set = ChannelSet::new()
        .with_channel("last", LastValue)
        .with_channel("topic", Topic)
        .with_channel("delta", Delta)
        .with_channel("messages", Messages)
        .with_channel("ephemeral", Ephemeral)
        .with_channel("scratch", Untracked)
        .with_channel("barrier", Barrier::new(2))
        .with_channel("named", NamedBarrier::new(["left", "right"]))
        .with_channel(
            "sum",
            BinaryAggregate::new(|current, incoming| {
                Ok(Value::from(
                    current.as_i64().unwrap_or(0) + incoming.as_i64().unwrap_or(0),
                ))
            }),
        );

    assert!(set.contains("last"));
    assert!(!set.allows_concurrent("last").unwrap());
    assert!(set.allows_concurrent("topic").unwrap());
    assert!(!set.is_ready("barrier").unwrap());

    set.apply_update("last", json!("a")).unwrap();
    set.apply_update("topic", json!("first")).unwrap();
    set.apply_update("topic", json!(["second", "third"]))
        .unwrap();
    set.apply_update("delta", json!(2)).unwrap();
    set.apply_update("delta", json!(3)).unwrap();
    set.apply_update("messages", json!({ "id": "m1", "text": "old" }))
        .unwrap();
    set.apply_update("messages", json!({ "id": "m1", "text": "new" }))
        .unwrap();
    set.apply_update("ephemeral", json!("one-shot")).unwrap();
    set.apply_update("scratch", json!("hidden")).unwrap();
    set.apply_update("barrier", json!("left")).unwrap();
    set.apply_update("barrier", json!("right")).unwrap();
    set.apply_update("named", json!({ "left": true })).unwrap();
    set.apply_update("named", json!({ "right": true })).unwrap();
    set.apply_update("sum", json!(5)).unwrap();
    set.apply_update("sum", json!(7)).unwrap();

    assert_eq!(set.get("delta"), Some(&json!(5)));
    assert_eq!(set.get("sum"), Some(&json!(12)));
    assert!(set.is_ready("barrier").unwrap());
    assert!(set.is_ready("named").unwrap());
    assert!(!set.snapshot().contains_key("scratch"));

    let state = ChannelState::new()
        .with_channel("last", LastValue)
        .with_channel("topic", Topic)
        .with_channel("delta", Delta)
        .with_channel("messages", Messages)
        .with_channel("ephemeral", Ephemeral)
        .with_channel("scratch", Untracked)
        .with_channel("barrier", Barrier::new(2))
        .with_channel("named", NamedBarrier::new(["left", "right"]))
        .with_channel(
            "sum",
            BinaryAggregate::new(|current, incoming| {
                Ok(Value::from(
                    current.as_i64().unwrap_or(0) + incoming.as_i64().unwrap_or(0),
                ))
            }),
        );
    let state = state
        .merge(
            ChannelUpdate::new()
                .set("last", json!("step-1"))
                .set("ephemeral", json!("flash"))
                .at_step(1),
        )
        .unwrap();
    assert_eq!(state.channels().get("ephemeral"), Some(&json!("flash")));
    let state = state
        .merge(ChannelUpdate::new().set("last", json!("step-2")).at_step(2))
        .unwrap();
    assert!(state.channels().get("ephemeral").is_none());

    let conflict = state
        .clone()
        .merge(ChannelUpdate::new().set("last", json!("a")).at_step(3))
        .unwrap()
        .merge(ChannelUpdate::new().set("last", json!("b")).at_step(3))
        .unwrap_err();
    assert!(matches!(
        conflict,
        TinyAgentsError::InvalidConcurrentUpdate(_)
    ));
}

#[tokio::test]
async fn retry_rate_limit_and_summarization_contracts_are_deterministic() {
    let retry = RetryPolicy::default()
        .with_max_attempts(3)
        .with_initial_backoff_ms(50)
        .with_multiplier(2.0)
        .with_max_backoff_ms(90)
        .with_jitter(false);
    assert!(retry.should_retry(0));
    assert!(retry.should_retry(1));
    assert!(!retry.should_retry(2));
    assert_eq!(retry.backoff_for_attempt(0), Duration::from_millis(50));
    assert_eq!(retry.backoff_for_attempt(2), Duration::from_millis(90));
    let jittered = retry.clone().with_jitter(true);
    assert_eq!(
        jittered.backoff_for_attempt_with(1, 0.5),
        Duration::from_millis(45)
    );

    assert!(is_retryable(&TinyAgentsError::Model("timeout".into())));
    assert!(is_retryable(&TinyAgentsError::Tool("temporary".into())));
    assert!(!is_retryable(&TinyAgentsError::Validation("bad".into())));

    let fallback = FallbackPolicy::new(["primary", "backup", "final"]);
    assert_eq!(fallback.next_after("primary"), Some("backup"));
    assert_eq!(fallback.next_after("final"), None);
    assert_eq!(fallback.next_after("missing"), None);

    let limiter = RateLimiter::new(2, 1.0);
    let now = Instant::now();
    assert_eq!(limiter.available(now), 2);
    assert!(limiter.try_acquire(2, now));
    assert!(!limiter.try_acquire(1, now));
    assert_eq!(limiter.available(now + Duration::from_secs(1)), 1);

    assert_eq!(estimate_tokens(""), 0);
    assert_eq!(estimate_tokens("abcd"), 1);

    let messages = vec![
        Message::system("rules"),
        Message::user("first question"),
        Message::assistant("first answer"),
        Message::user("second question"),
        Message::assistant("second answer"),
    ];
    let keep_last = trim_messages(&messages, &TrimStrategy::KeepLast(2));
    assert_eq!(keep_last.len(), 3);
    assert!(matches!(keep_last[0], Message::System(_)));

    let first_last = trim_messages(
        &messages,
        &TrimStrategy::KeepFirstAndLast { first: 1, last: 1 },
    );
    assert_eq!(first_last.len(), 3);
    let max_tokens = trim_messages(&messages, &TrimStrategy::MaxTokens(3));
    assert!(!max_tokens.is_empty());

    let policy = SummarizationPolicy {
        trigger_tokens: 1,
        keep_last: 2,
        ..SummarizationPolicy::default()
    }
    .with_context_window(20)
    .with_threshold_fraction(0.5);
    assert_eq!(policy.trigger_budget(), 10);
    assert!(policy.should_summarize(&messages));
    let (to_summarize, to_keep) = policy.plan(&messages);
    assert_eq!(to_summarize.len(), 2);
    assert_eq!(to_keep.len(), 3);

    let summary = ConcatSummarizer
        .summarize(&to_summarize)
        .await
        .expect("summary");
    assert!(summary.summary.text().contains("Conversation Summary"));
    assert_eq!(summary.provenance.source_ids, vec!["msg-0", "msg-1"]);
    assert!(ConcatSummarizer.summarize(&[]).await.is_err());
}
