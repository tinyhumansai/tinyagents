//! Unit tests for the harness memory module.

use super::*;
use crate::harness::message::Message;
use crate::harness::store::InMemoryStore;

#[tokio::test]
async fn in_memory_history_round_trips() {
    let history = InMemoryChatHistory::new();
    assert!(history.messages("t1").await.unwrap().is_empty());

    history.append("t1", Message::user("hi")).await.unwrap();
    history
        .append("t1", Message::assistant("hello"))
        .await
        .unwrap();

    let msgs = history.messages("t1").await.unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].text(), "hi");
    assert_eq!(msgs[1].text(), "hello");
}

#[tokio::test]
async fn in_memory_history_threads_are_isolated() {
    let history = InMemoryChatHistory::new();
    history.append("a", Message::user("for-a")).await.unwrap();
    history.append("b", Message::user("for-b")).await.unwrap();

    assert_eq!(history.messages("a").await.unwrap().len(), 1);
    assert_eq!(history.messages("b").await.unwrap()[0].text(), "for-b");
}

#[tokio::test]
async fn in_memory_history_clear_removes_thread() {
    let history = InMemoryChatHistory::new();
    history.append("t", Message::user("x")).await.unwrap();
    history.clear("t").await.unwrap();
    assert!(history.messages("t").await.unwrap().is_empty());
    // Clearing an empty thread is a no-op, not an error.
    history.clear("never").await.unwrap();
}

#[tokio::test]
async fn store_history_persists_through_store() {
    let store = InMemoryStore::new();
    let history = StoreChatHistory::new(store.clone());

    history.append("t1", Message::user("one")).await.unwrap();
    history
        .append("t1", Message::assistant("two"))
        .await
        .unwrap();

    // A second view over the same store sees the persisted history.
    let view = StoreChatHistory::new(store.clone());
    let msgs = view.messages("t1").await.unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].text(), "one");
    assert_eq!(msgs[1].text(), "two");

    // The data really lives in the store under the expected namespace.
    let raw = store
        .get(StoreChatHistory::<InMemoryStore>::NAMESPACE, "t1")
        .await
        .unwrap();
    assert!(raw.is_some());

    history.clear("t1").await.unwrap();
    assert!(view.messages("t1").await.unwrap().is_empty());
}

#[tokio::test]
async fn short_term_memory_load_append_save() {
    let history = InMemoryChatHistory::new();
    let mem = ShortTermMemory::new(history, "thread-1");
    assert_eq!(mem.thread_id(), "thread-1");

    mem.append(Message::user("a")).await.unwrap();
    mem.append(Message::user("b")).await.unwrap();
    let loaded = mem.load().await.unwrap();
    assert_eq!(loaded.len(), 2);

    // Save replaces the stored history.
    mem.save(vec![Message::system("only")]).await.unwrap();
    let after = mem.load().await.unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].text(), "only");

    mem.clear().await.unwrap();
    assert!(mem.load().await.unwrap().is_empty());
}

#[tokio::test]
async fn short_term_memory_trim_hook_applies() {
    let history = InMemoryChatHistory::new();
    // Keep only the last message on load and save.
    let mem = ShortTermMemory::new(history, "t").with_trim(|mut msgs| {
        if msgs.len() > 1 {
            msgs = msgs.split_off(msgs.len() - 1);
        }
        msgs
    });

    mem.append(Message::user("1")).await.unwrap();
    mem.append(Message::user("2")).await.unwrap();
    mem.append(Message::user("3")).await.unwrap();

    let loaded = mem.load().await.unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].text(), "3");
}

#[test]
fn memory_scope_serializes_snake_case() {
    assert_eq!(
        serde_json::to_string(&MemoryScope::ShortTerm).unwrap(),
        "\"short_term\""
    );
    assert_eq!(
        serde_json::to_string(&MemoryScope::LongTerm).unwrap(),
        "\"long_term\""
    );
}
