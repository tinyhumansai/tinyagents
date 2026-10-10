//! Unit tests for `SessionStore` persistence and UUID v4 validation.

use super::*;
use tempfile::tempdir;

#[test]
fn uuid_v4_format() {
    let id = generate_uuid_v4();
    assert!(is_uuid_v4(&id), "generated id should be v4: {id}");
}

#[test]
fn rejects_non_v4() {
    assert!(!is_uuid_v4("not-a-uuid"));
    assert!(!is_uuid_v4("cc_abc123"));
    // version 1 uuid (nibble at 14 is '1')
    assert!(!is_uuid_v4("00000000-0000-1000-8000-000000000000"));
}

#[test]
fn roundtrip_set_and_get() {
    let dir = tempdir().unwrap();
    let store = SessionStore::open(dir.path());
    assert!(store.get("thread_a").is_none());
    store.set("thread_a", "abc").unwrap();
    let reopened = SessionStore::open(dir.path());
    assert_eq!(reopened.get("thread_a").as_deref(), Some("abc"));
}

#[test]
fn delivered_turns_persist_and_reset_with_a_new_session() {
    let dir = tempdir().unwrap();
    let store = SessionStore::open(dir.path());
    store.set("t", "uuid-1").unwrap();
    store
        .record_delivered("t", &["a".into(), "b".into()])
        .unwrap();
    store
        .record_delivered("t", &["b".into(), "c".into()])
        .unwrap();

    let reopened = SessionStore::open(dir.path());
    assert_eq!(reopened.delivered("t").len(), 3);

    // Same uuid keeps the record; a replacement session starts empty.
    reopened.set("t", "uuid-1").unwrap();
    assert_eq!(reopened.delivered("t").len(), 3);
    reopened.set("t", "uuid-2").unwrap();
    assert!(reopened.delivered("t").is_empty());
}

#[test]
fn delivered_record_is_bounded() {
    let dir = tempdir().unwrap();
    let store = SessionStore::open(dir.path());
    let many: Vec<String> = (0..MAX_DELIVERED_PER_THREAD + 10)
        .map(|i| i.to_string())
        .collect();
    store.record_delivered("t", &many).unwrap();
    let kept = store.delivered("t");
    assert_eq!(kept.len(), MAX_DELIVERED_PER_THREAD);
    assert!(kept.contains(&(MAX_DELIVERED_PER_THREAD + 9).to_string()));
    assert!(!kept.contains("0"));
}
