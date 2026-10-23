use super::*;
use crate::testkit::MemoryStorage;
use serde_json::{Value, json};
use tinyagents_storage_kit::{CollectionSpec, Scope, ScopedStorage};
use tinystoragedrivers_api::StorageBackend;

// Explicit fixture preparation is owning bootstrap authority, not actor DDL.
const PENDING_COLLECTION: &str = "conversations.appends";

fn request(id: &str) -> CreateConversationThread {
    CreateConversationThread {
        id: id.into(),
        title: "Original".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        parent_thread_id: None,
        labels: None,
        personality_id: None,
        working_dir: None,
    }
}

fn message(id: &str, content: &str) -> ThreadMessage {
    ThreadMessage {
        id: id.into(),
        content: content.into(),
        message_type: "text".into(),
        extra_metadata: json!({}),
        sender: "assistant".into(),
        created_at: "2026-02-01T00:00:00Z".into(),
    }
}

async fn prepared_append(backend: &dyn StorageBackend) {
    let alice = Scope::new("alice").unwrap();
    let bob = Scope::new("bob").unwrap();
    let raw = backend.for_scope(&alice).unwrap();
    DriverConversationStore::prepare_append_schema(raw.documents().as_ref())
        .await
        .unwrap();
    raw.documents()
        .ensure_collection(&CollectionSpec::new(PENDING_COLLECTION))
        .await
        .unwrap();
    let a =
        DriverConversationStore::new_validate_only(ScopedStorage::new(raw.clone(), alice.clone()));
    let b =
        DriverConversationStore::new_validate_only(ScopedStorage::new(raw.clone(), alice.clone()));
    let bob_storage = backend.for_scope(&bob).unwrap();
    DriverConversationStore::prepare_append_schema(bob_storage.documents().as_ref())
        .await
        .unwrap();
    let other = DriverConversationStore::new_validate_only(ScopedStorage::new(bob_storage, bob));
    a.ensure_thread(request("same")).await.unwrap();
    other.ensure_thread(request("same")).await.unwrap();

    let first = message("agent:run", "first original 東京");
    assert_eq!(
        a.append_message("same", first.clone()).await.unwrap(),
        first
    );
    let mut retry = first.clone();
    retry.content = "different retry must return original".into();
    assert_eq!(b.append_message("same", retry).await.unwrap(), first);

    let mut channel = message("channel-row", "channel original");
    channel.extra_metadata = json!({"scope": "channel"});
    assert_eq!(
        a.append_message("same", channel.clone()).await.unwrap(),
        channel
    );
    let mut channel_retry = channel.clone();
    channel_retry.content = "different channel retry".into();
    assert_eq!(
        b.append_message("same", channel_retry).await.unwrap(),
        channel
    );

    // Ordinary repeated IDs retain classic append semantics; no generic
    // exactly-once promise is introduced by the native outbox.
    let mut ordinary = message("ordinary-row", "ordinary repeats append");
    ordinary.created_at = "2026-01-02T00:00:00Z".into();
    a.append_message("same", ordinary.clone()).await.unwrap();
    b.update_thread_title("same", "Edited", "now")
        .await
        .unwrap();
    b.append_message("same", ordinary.clone()).await.unwrap();

    let reopened = DriverConversationStore::new_validate_only(ScopedStorage::new(
        backend.for_scope(&alice).unwrap(),
        alice,
    ));
    assert_eq!(
        reopened.get_messages("same").await.unwrap(),
        vec![first, channel, ordinary.clone(), ordinary.clone()]
    );
    let thread = reopened.get_thread("same").await.unwrap().unwrap();
    assert_eq!(thread.title, "Edited");
    assert_eq!(thread.created_at, "2026-01-01T00:00:00Z");
    assert_eq!(thread.message_count, 4);
    assert_eq!(thread.last_message_at, ordinary.created_at);
    assert!(other.get_messages("same").await.unwrap().is_empty());
    assert_eq!(
        other
            .get_thread("same")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        0
    );
    assert_eq!(
        raw.documents()
            .count(PENDING_COLLECTION, &Filter::All)
            .await
            .unwrap(),
        0,
        "successful appends retire their exact pending intent"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn memory_prepared_native_append_reopens_and_preserves_classic_duplicate_semantics() {
    prepared_append(&MemoryStorage::new()).await;
}

#[tokio::test(flavor = "current_thread")]
async fn reconstructed_memory_prepared_append_reopens_and_preserves_classic_duplicate_semantics() {
    let backend = MemoryStorage::new();
    prepared_append(&backend).await;
}

#[tokio::test]
async fn append_checks_conditional_stream_capability_on_the_stream_port() {
    let store = actor(&MemoryStorage::new()).await;
    let stream_capability_only = fixtures::hide_document_conditional_stream_append(&store);
    let message = message("agent:capability", "conditional stream append is supported");

    assert_eq!(
        stream_capability_only
            .append_message("same", message.clone())
            .await
            .unwrap(),
        message
    );
}

#[path = "driver_append_fixtures.rs"]
mod fixtures;
use fixtures::{Action, Fault, Phase, actor, faulted};

async fn canceled(backend: &dyn StorageBackend, phase: Phase) {
    let store = actor(backend).await;
    let fault = Fault::new(phase, Action::Pause);
    let interrupted = faulted(&store, fault.clone());
    let mut future =
        Box::pin(interrupted.append_message("same", message("agent:first", "accepted")));
    tokio::select! {
        _ = fault.reached.notified() => {},
        result = &mut future => panic!("phase never paused: {result:?}"),
    }
    assert!(
        store.get_messages("same").await.unwrap().is_empty(),
        "unpublished rows must remain invisible"
    );
    assert_eq!(
        store
            .get_thread("same")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        0
    );
    drop(future);
    let resumed = store.clone();
    resumed
        .append_message("same", message("agent:second", "next"))
        .await
        .unwrap();
    let rows = resumed.get_messages("same").await.unwrap();
    let expected = if phase == Phase::Probe { 1 } else { 2 };
    assert_eq!(rows.len(), expected);
    if phase != Phase::Probe {
        assert_eq!(rows[0].content, "accepted");
    }
    assert_eq!(
        resumed
            .docs
            .count(PENDING_COLLECTION, &Filter::All)
            .await
            .unwrap(),
        0
    );
}
#[tokio::test]
async fn canceled_memory_writers_keep_reserved_and_appended_rows_invisible_until_recovery() {
    for phase in [Phase::Probe, Phase::Reserve, Phase::Append] {
        canceled(&MemoryStorage::new(), phase).await;
    }
}
#[tokio::test]
async fn canceled_reconstructed_memory_writers_keep_reserved_and_appended_rows_invisible_until_recovery()
 {
    for phase in [Phase::Probe, Phase::Reserve, Phase::Append] {
        let backend = MemoryStorage::new();
        canceled(&backend, phase).await;
    }
}
#[tokio::test]
async fn lost_native_acknowledgments_reconcile_exact_reserved_attempt_without_duplicates() {
    for phase in [Phase::Reserve, Phase::Append, Phase::Finalize] {
        let store = actor(&MemoryStorage::new()).await;
        let interrupted = faulted(&store, Fault::new(phase, Action::LoseAck));
        let original = message("ordinary", "one actual attempt");
        assert_eq!(
            interrupted
                .append_message("same", original.clone())
                .await
                .unwrap(),
            original
        );
        assert_eq!(store.get_messages("same").await.unwrap(), vec![original]);
        assert_eq!(
            store
                .docs
                .count(PENDING_COLLECTION, &Filter::All)
                .await
                .unwrap(),
            0
        );
    }
}
#[tokio::test]
async fn unobserved_ordinary_reservation_returns_typed_uncertainty_with_original_native_source() {
    use std::error::Error;
    let store = actor(&MemoryStorage::new()).await;
    let interrupted = faulted(&store, Fault::new(Phase::Reserve, Action::FailBefore));
    let error = interrupted
        .append_message("same", message("ordinary", "not accepted"))
        .await
        .unwrap_err();
    let ConversationError::Uncertain { source, .. } = error else {
        panic!("expected typed uncertainty")
    };
    assert_eq!(source.kind(), ErrorKind::Unavailable);
    assert_eq!(
        source
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .to_string(),
        "original native cause"
    );
    assert!(store.get_messages("same").await.unwrap().is_empty());
    assert_eq!(
        store
            .docs
            .count(PENDING_COLLECTION, &Filter::All)
            .await
            .unwrap(),
        0
    );
}
#[tokio::test]
async fn counterfeit_document_capabilities_do_not_reserve_when_actual_stream_probe_refuses() {
    let store = actor(&MemoryStorage::new()).await;
    let interrupted = faulted(&store, Fault::new(Phase::Probe, Action::Unsupported));
    let error = interrupted
        .append_message("same", message("agent:a", "refused"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, ConversationError::Driver(ref e) if matches!(e.kind(), ErrorKind::Unsupported(tinystoragedrivers_types::Capability::ConditionalStreamAppend)))
    );
    assert!(store.get_messages("same").await.unwrap().is_empty());
    assert_eq!(
        store
            .docs
            .count(PENDING_COLLECTION, &Filter::All)
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn cold_helpers_publish_once_and_preserve_concurrent_title_and_duplicate_original() {
    let store = actor(&MemoryStorage::new()).await;
    let fault = Fault::new(Phase::Reserve, Action::Pause);
    let interrupted = faulted(&store, fault.clone());
    let original = message("agent:race", "original");
    let mut writer = Box::pin(interrupted.append_message("same", original.clone()));
    tokio::select! {
        _ = fault.reached.notified() => {},
        result = &mut writer => panic!("reservation never paused: {result:?}"),
    }
    store
        .update_thread_title("same", "Concurrent title", "later")
        .await
        .unwrap();
    let finalization = Fault::new(Phase::Finalize, Action::Barrier);
    let b = faulted(&store, finalization.clone());
    let c = b.clone();
    let (b_result, c_result) = tokio::join!(
        b.append_message("same", message("agent:race", "different retry B")),
        c.append_message("same", message("agent:race", "different retry C")),
    );
    assert_eq!(b_result.unwrap(), original);
    assert_eq!(c_result.unwrap(), original);
    assert!(
        finalization
            .conflicts
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0,
        "two cold helpers must actually race the native finalization CAS"
    );
    fault.release.notify_one();
    assert_eq!(writer.await.unwrap(), original);
    assert_eq!(store.get_messages("same").await.unwrap(), vec![original]);
    assert_eq!(
        store.get_thread("same").await.unwrap().unwrap().title,
        "Concurrent title"
    );
}
#[tokio::test]
async fn byte_identical_ordinary_competitor_cannot_prove_unobserved_reservation_ownership() {
    let store = actor(&MemoryStorage::new()).await;
    let fault = Fault::new(Phase::Reserve, Action::PauseLoseAck);
    let interrupted = faulted(&store, fault.clone());
    let identical = message("ordinary", "identical bytes from two callers");
    let mut first = Box::pin(interrupted.append_message("same", identical.clone()));
    tokio::select! {
        _ = fault.reached.notified() => {},
        result = &mut first => panic!("reserve never paused: {result:?}"),
    }
    assert_eq!(
        store
            .append_message("same", identical.clone())
            .await
            .unwrap(),
        identical
    );
    fault.release.notify_one();
    assert!(
        matches!(first.await, Err(ConversationError::Uncertain { ref source, .. }) if source.kind() == ErrorKind::Unavailable)
    );
    assert_eq!(
        store.get_messages("same").await.unwrap(),
        vec![identical.clone(), identical]
    );
    assert_eq!(
        store
            .get_thread("same")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        2,
        "uncertain caller must not silently reserve another UUID append"
    );
}

#[tokio::test]
async fn noncanonical_pending_record_refuses_without_appending_or_publishing_prefix() {
    let store = actor(&MemoryStorage::new()).await;
    let fault = Fault::new(Phase::Reserve, Action::Pause);
    let interrupted = faulted(&store, fault.clone());
    let mut writer =
        Box::pin(interrupted.append_message("same", message("agent:corrupt", "accepted")));
    tokio::select! {
        _ = fault.reached.notified() => {},
        result = &mut writer => panic!("reserve never paused: {result:?}"),
    }
    drop(writer);
    let current = store.load("same").await.unwrap().unwrap();
    let key = pending::AppendIntent::key("same", &current.doc.incarnation).unwrap();
    let row = store
        .docs
        .get(PENDING_COLLECTION, &key)
        .await
        .unwrap()
        .unwrap();
    let mut envelope = row.doc.clone();
    envelope["record"]["unexpected"] = json!("must not disappear silently");
    store
        .docs
        .put(PENDING_COLLECTION, &key, envelope.clone(), row.unchanged())
        .await
        .unwrap();
    let error = store
        .append_message("same", message("agent:next", "next"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, ConversationError::Driver(ref e) if e.kind()==ErrorKind::Serialization)
    );
    let stream = current.doc.stream().unwrap();
    assert_eq!(store.streams.len(&stream).await.unwrap(), 0);
    assert!(store.get_messages("same").await.unwrap().is_empty());
    assert_eq!(
        store
            .get_thread("same")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        0
    );
    assert_eq!(
        store
            .docs
            .get(PENDING_COLLECTION, &key)
            .await
            .unwrap()
            .unwrap()
            .doc,
        envelope
    );
}

#[tokio::test]
async fn competing_same_id_reservations_force_cas_conflict_then_restart_original_dedup() {
    use std::sync::atomic::Ordering;
    let store = actor(&MemoryStorage::new()).await;
    let fault = Fault::new(Phase::Reserve, Action::Barrier);
    let a = faulted(&store, fault.clone());
    let b = a.clone();
    let (a_result, b_result) = tokio::join!(
        a.append_message("same", message("agent:same", "A")),
        b.append_message("same", message("agent:same", "B")),
    );
    let a_result = a_result.unwrap();
    assert_eq!(b_result.unwrap(), a_result);
    assert!(
        fault.conflicts.load(Ordering::SeqCst) > 0,
        "real native checked-batch CAS conflict must have occurred"
    );
    assert_eq!(store.get_messages("same").await.unwrap(), vec![a_result]);
}

#[tokio::test]
async fn cancellation_after_finalization_does_not_hide_committed_prefix_or_repeat_deterministic_message()
 {
    let store = actor(&MemoryStorage::new()).await;
    let fault = Fault::new(Phase::Finalize, Action::Pause);
    let interrupted = faulted(&store, fault.clone());
    let original = message("agent:finalized", "committed before cancellation");
    let mut writer = Box::pin(interrupted.append_message("same", original.clone()));
    tokio::select! {
        _ = fault.reached.notified() => {},
        result = &mut writer => panic!("finalization never paused: {result:?}"),
    }
    assert_eq!(
        store.get_messages("same").await.unwrap(),
        vec![original.clone()]
    );
    drop(writer);
    assert_eq!(
        store
            .append_message("same", message("agent:finalized", "changed retry"))
            .await
            .unwrap(),
        original
    );
    assert_eq!(
        store
            .get_thread("same")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        1
    );
    assert_eq!(
        store
            .docs
            .count(PENDING_COLLECTION, &Filter::All)
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn mismatching_actual_stream_row_cannot_publish_reserved_message_or_retire_pending() {
    let store = actor(&MemoryStorage::new()).await;
    let fault = Fault::new(Phase::Reserve, Action::Pause);
    let interrupted = faulted(&store, fault.clone());
    let mut writer =
        Box::pin(interrupted.append_message("same", message("agent:reserved", "expected")));
    tokio::select! {
        _ = fault.reached.notified() => {},
        result = &mut writer => panic!("reservation never paused: {result:?}"),
    }
    drop(writer);
    let current = store.load("same").await.unwrap().unwrap();
    let stream = current.doc.stream().unwrap();
    store
        .streams
        .append_batch_if_offset(
            &stream,
            0,
            vec![
                serde_json::to_value(
                    MessageDto::from_message(&message("agent:foreign", "wrong row"), to_ms)
                        .unwrap(),
                )
                .unwrap(),
            ],
        )
        .await
        .unwrap();
    let error = store
        .append_message("same", message("agent:next", "next"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, ConversationError::Driver(ref e) if e.kind() == ErrorKind::Serialization)
    );
    assert!(store.get_messages("same").await.unwrap().is_empty());
    assert_eq!(
        store
            .get_thread("same")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        0
    );
    assert_eq!(
        store
            .docs
            .count(PENDING_COLLECTION, &Filter::All)
            .await
            .unwrap(),
        1
    );
    assert_eq!(store.streams.len(&stream).await.unwrap(), 1);
}

#[path = "driver_append_corruption_tests.rs"]
mod corruption_tests;

#[path = "driver_append_review_tests.rs"]
mod review_tests;
