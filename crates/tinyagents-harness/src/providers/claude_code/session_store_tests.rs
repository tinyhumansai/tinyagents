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
fn remove_if_forgets_matching_mapping_across_reopen() {
    let dir = tempdir().unwrap();
    let store = SessionStore::open(dir.path());
    store.set("thread_a", "session-a").unwrap();

    assert!(store.remove_if("thread_a", "session-a").unwrap());

    assert!(store.get("thread_a").is_none());
    assert!(SessionStore::open(dir.path()).get("thread_a").is_none());
}

#[test]
fn remove_if_keeps_a_newer_mapping() {
    let dir = tempdir().unwrap();
    let store = SessionStore::open(dir.path());
    store.set("thread_a", "session-new").unwrap();

    assert!(!store.remove_if("thread_a", "session-old").unwrap());
    assert_eq!(store.get("thread_a").as_deref(), Some("session-new"));
}

#[test]
fn remove_if_keeps_mapping_when_persistence_fails() {
    let dir = tempdir().unwrap();
    let store = SessionStore::open(dir.path());
    store.set("thread_a", "session-a").unwrap();
    // Make the store file unwritable by replacing it with a directory.
    let path = dir.path().join("claude-code-sessions.json");
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();

    assert!(store.remove_if("thread_a", "session-a").is_err());
    assert_eq!(store.get("thread_a").as_deref(), Some("session-a"));
}
