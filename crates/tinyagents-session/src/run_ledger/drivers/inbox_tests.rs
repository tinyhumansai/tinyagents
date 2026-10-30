use super::*;
use crate::run_ledger::inbox::InputMode;
use crate::testkit::MemoryStorage;

/// Read every admitted input across the maximum page boundary without consuming it.
#[tokio::test]
async fn pending_inputs_pages_past_five_hundred_with_exclusive_cursor() {
    let ledger = ledger();
    prepare(&ledger).await;
    let mut receipts = Vec::new();
    for index in 0..503 {
        receipts.push(
            ledger
                .enqueue_input(input(&format!("page-{index}")))
                .await
                .unwrap(),
        );
    }
    let first = ledger
        .pending_inputs("root", "child", None, u32::MAX)
        .await
        .unwrap();
    assert_eq!(first, receipts[..500]);
    let cursor = first.last().unwrap().sequence;
    let rest = ledger
        .pending_inputs("root", "child", Some(cursor), 500)
        .await
        .unwrap();
    assert_eq!(rest, receipts[500..]);
    assert_eq!(
        ledger
            .pending_inputs("root", "child", Some(cursor), 0)
            .await
            .unwrap(),
        receipts[500..501]
    );
    assert!(
        ledger
            .pending_inputs("root", "child", Some(u64::MAX), 500)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        ledger
            .pending_inputs("other", "child", Some(u64::MAX), 500)
            .await
            .is_err()
    );
}

/// Create an isolated in-memory ledger for admission tests.
fn ledger() -> DriverRunLedger {
    DriverRunLedger::new(
        MemoryStorage::new()
            .for_scope(&Scope::new("inbox-tests").unwrap())
            .unwrap(),
    )
}

/// Construct an awaiting run with explicit persisted ancestry.
fn run(id: &str, parent: Option<&str>, thread: Option<&str>) -> AgentRunUpsert {
    AgentRunUpsert {
        id: id.into(),
        kind: AgentRunKind::Subagent,
        parent_run_id: parent.map(str::to_owned),
        parent_thread_id: thread.map(str::to_owned),
        agent_id: None,
        status: AgentRunStatus::AwaitingUser,
        prompt_ref: None,
        worker_thread_id: None,
        checkpoint_path: None,
        checkpoint: None,
        summary: None,
        error: None,
        metadata: serde_json::json!({}),
        started_at: None,
        completed_at: None,
    }
}

/// Address an exact text payload to the prepared child run.
fn input(id: &str) -> AddressedInput {
    AddressedInput {
        submission_id: id.into(),
        root_thread_id: "root".into(),
        recipient_run_id: "child".into(),
        mode: InputMode::Answer,
        text: "Keep this exact answer\nwith its line break.".into(),
    }
}

/// Persist the root-to-parent-to-child lineage used by admission tests.
async fn prepare(ledger: &DriverRunLedger) {
    ledger
        .upsert_agent_run(run("parent", None, Some("root")))
        .await
        .unwrap();
    ledger
        .upsert_agent_run(run("child", Some("parent"), Some("worker-thread")))
        .await
        .unwrap();
}

#[tokio::test]
/// Reopening a ledger preserves the original receipt and rejects id reuse.
async fn duplicate_admission_survives_handle_recreation_and_records_one_event() {
    let ledger = ledger();
    prepare(&ledger).await;
    let message = input("submission-1");
    let first = ledger.enqueue_input(message.clone()).await.unwrap();
    let reopened = DriverRunLedger::new(ledger.scoped.clone());
    assert_eq!(
        reopened.enqueue_input(message.clone()).await.unwrap(),
        first
    );
    assert_eq!(
        reopened
            .pending_inputs("root", "child", None, 20)
            .await
            .unwrap(),
        vec![first]
    );
    let events = reopened
        .list_recent_run_events(&RunEventListRequest {
            run_id: "child".into(),
            after_sequence: None,
            limit: None,
        })
        .await
        .unwrap();
    assert_eq!(events.events.len(), 1);
    assert_eq!(events.events[0].event_type, "input_queued");
    let mut changed = message;
    changed.text = "Different content".into();
    assert!(reopened.enqueue_input(changed).await.is_err());
}

#[tokio::test]
/// Concurrent retries commit once and later admissions remain ordered.
async fn concurrent_duplicate_admissions_share_one_receipt_and_preserve_order() {
    let ledger = ledger();
    prepare(&ledger).await;
    let (left, right) = tokio::join!(
        ledger.enqueue_input(input("same")),
        ledger.enqueue_input(input("same"))
    );
    assert_eq!(left.unwrap(), right.unwrap());
    ledger.enqueue_input(input("second")).await.unwrap();
    let queued = ledger
        .pending_inputs("root", "child", None, 20)
        .await
        .unwrap();
    assert_eq!(queued.len(), 2);
    assert!(queued[0].sequence < queued[1].sequence);
    assert_eq!(queued[0].input.submission_id, "same");
    assert_eq!(queued[1].input.submission_id, "second");
}

#[tokio::test]
/// Malformed payloads and invalid ancestry cannot enter the durable inbox.
async fn cross_root_cyclic_missing_and_blank_inputs_are_rejected() {
    let ledger = ledger();
    prepare(&ledger).await;
    let mut wrong_root = input("wrong");
    wrong_root.root_thread_id = "other".into();
    assert!(ledger.enqueue_input(wrong_root).await.is_err());
    assert!(
        ledger
            .pending_inputs("other", "child", None, 20)
            .await
            .is_err()
    );
    let mut missing = input("missing");
    missing.recipient_run_id = "absent".into();
    assert!(ledger.enqueue_input(missing).await.is_err());
    let mut blank = input("blank");
    blank.text = " \n ".into();
    assert!(ledger.enqueue_input(blank).await.is_err());
    ledger
        .upsert_agent_run(run("parent", Some("child"), Some("root")))
        .await
        .unwrap();
    assert!(ledger.enqueue_input(input("cycle")).await.is_err());
}
