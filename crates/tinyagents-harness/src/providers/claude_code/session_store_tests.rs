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
    for i in 0..MAX_DELIVERED_PER_THREAD + 10 {
        store.record_delivered("t", &[i.to_string()]).unwrap();
    }
    let kept = store.delivered("t");
    assert_eq!(kept.len(), MAX_DELIVERED_PER_THREAD);
    assert!(kept.contains(&(MAX_DELIVERED_PER_THREAD + 9).to_string()));
    assert!(!kept.contains("0"));
}

#[test]
fn delivered_record_never_forgets_the_batch_just_recorded() {
    let dir = tempdir().unwrap();
    let store = SessionStore::open(dir.path());
    let many: Vec<String> = (0..MAX_DELIVERED_PER_THREAD + 10)
        .map(|i| i.to_string())
        .collect();
    store.record_delivered("t", &many).unwrap();
    assert_eq!(store.delivered("t").len(), many.len());
}

#[test]
fn instances_on_one_workspace_see_each_others_deliveries() {
    let dir = tempdir().unwrap();
    let a = SessionStore::open(dir.path());
    let b = SessionStore::open(dir.path());
    a.set("t", "uuid-1").unwrap();
    a.record_delivered("t", &["x".into()]).unwrap();
    assert!(b.delivered("t").contains("x"));
    b.record_delivered("t", &["y".into()]).unwrap();
    let both = a.delivered("t");
    assert!(both.contains("x") && both.contains("y"));
}

#[test]
fn claim_is_exclusive_and_released_on_failure() {
    let dir = tempdir().unwrap();
    let store = Arc::new(SessionStore::open(dir.path()));
    let fps = vec!["turn".to_string()];

    let (already, first) = store.claim_delivered("t", &fps);
    assert!(already.is_empty());
    let (already, second) = store.claim_delivered("t", &fps);
    assert!(already.contains("turn"), "second caller must not re-send");
    drop(second); // owns nothing, releases nothing
    assert!(store.delivered("t").contains("turn"));

    drop(first); // failed turn: reservation released
    assert!(store.delivered("t").is_empty());

    let (_, ok) = store.claim_delivered("t", &fps);
    ok.commit();
    assert!(store.delivered("t").contains("turn"));
}
