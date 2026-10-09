use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use super::*;

fn record(task: &str, parent: &str) -> CompletionRecord {
    CompletionRecord::new(
        task,
        parent,
        "worker",
        CompletionStatus::Success,
        CompletionResult::text(format!("result of {task}")),
    )
}

fn open_router(path: &std::path::Path) -> CompletionRouter {
    CompletionRouter::new(Arc::new(JsonlCompletionStore::open(path).unwrap()))
}

#[tokio::test]
async fn pending_completions_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    {
        let router = open_router(&path);
        router.record(record("t1", "p")).await.unwrap();
        router
            .record(record("t2", "p").with_notify_mode(NotifyMode::Off))
            .await
            .unwrap();
        router.record(record("t3", "other")).await.unwrap();
    }
    let router = open_router(&path);
    let pending = router.pending_for("p");
    assert_eq!(
        pending
            .iter()
            .map(|r| r.task_id.as_str())
            .collect::<Vec<_>>(),
        ["t1", "t2"]
    );
    assert_eq!(pending[0].result.text, "result of t1");
    let batch = router.claim_pending("p", 10).unwrap();
    assert_eq!(batch.len(), 1, "the redelivery path finds t1 again");
}

#[tokio::test]
async fn attempts_and_give_up_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    {
        let router = open_router(&path).with_max_attempts(2);
        router.record(record("t1", "p")).await.unwrap();
        router.claim_pending("p", 1).unwrap();
        router.mark_failed(&["t1"]).unwrap();
    }
    let router = open_router(&path).with_max_attempts(2);
    let batch = router.claim_pending("p", 1).unwrap();
    assert_eq!(batch[0].attempts, 2, "the earlier attempt was kept");
    assert_eq!(router.mark_failed(&["t1"]).unwrap().len(), 1);
    drop(router);
    let router = open_router(&path);
    assert!(router.pending_for("p").is_empty());
    assert!(router.claim_pending("p", 1).unwrap().is_empty());
}

#[tokio::test]
async fn delivered_and_tombstoned_stay_deduped_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    {
        let router = open_router(&path);
        router.record(record("done", "p")).await.unwrap();
        router.claim_pending("p", 1).unwrap();
        router.mark_delivered(&["done"]).unwrap();
        router.record(record("waited", "p")).await.unwrap();
        router.tombstone("waited").unwrap();
        router.tombstone("early").unwrap();
    }
    let router = open_router(&path);
    for task in ["done", "waited", "early"] {
        let outcome = router.record(record(task, "p")).await.unwrap();
        assert!(
            matches!(
                outcome,
                RecordOutcome::Duplicate | RecordOutcome::Suppressed
            ),
            "{task}: {outcome:?}"
        );
    }
    assert!(router.pending_for("p").is_empty());
}

#[tokio::test]
async fn a_torn_final_line_is_dropped_and_the_log_stays_appendable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    {
        let router = open_router(&path);
        router.record(record("t1", "p")).await.unwrap();
    }
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(br#"{"task_id":"t2","parent_key":"p","agent_i"#)
        .unwrap();
    let router = open_router(&path);
    assert_eq!(router.pending_for("p").len(), 1);
    router.record(record("t3", "p")).await.unwrap();
    drop(router);
    let router = open_router(&path);
    let mut ids: Vec<_> = router
        .pending_for("p")
        .into_iter()
        .map(|r| r.task_id)
        .collect();
    ids.sort();
    assert_eq!(ids, ["t1", "t3"]);
}

#[tokio::test]
async fn a_whole_record_missing_its_newline_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    let line = serde_json::to_string(&record("t1", "p")).unwrap();
    std::fs::write(&path, line).unwrap();
    let router = open_router(&path);
    assert_eq!(router.pending_for("p").len(), 1);
    router.record(record("t2", "p")).await.unwrap();
    drop(router);
    assert_eq!(open_router(&path).pending_for("p").len(), 2);
}

#[tokio::test]
async fn an_unreadable_middle_line_does_not_block_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    let good = |t: &str| serde_json::to_string(&record(t, "p")).unwrap();
    std::fs::write(&path, format!("{}\nnot json\n{}\n", good("t1"), good("t2"))).unwrap();
    assert_eq!(open_router(&path).pending_for("p").len(), 2);
}

#[tokio::test]
async fn the_log_replays_to_the_latest_record_per_task() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    {
        let router = open_router(&path);
        router.record(record("t1", "p")).await.unwrap();
        router.claim_pending("p", 1).unwrap();
        router.mark_delivered(&["t1"]).unwrap();
    }
    let lines = std::fs::read_to_string(&path).unwrap().lines().count();
    assert_eq!(lines, 3, "append-only: insert, claim, delivered");
    let store = JsonlCompletionStore::open(&path).unwrap();
    assert_eq!(store.list(None).len(), 1);
    assert_eq!(store.get("t1").unwrap().state, CompletionState::Delivered);
}

#[tokio::test]
async fn compaction_rewrites_the_log_and_keeps_pending_work() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    let router = open_router(&path);
    router.record(record("old", "p")).await.unwrap();
    router.claim_pending("p", 1).unwrap();
    router.mark_delivered(&["old"]).unwrap();
    router.record(record("live", "p")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(router.compact(Duration::from_millis(1)).unwrap(), 1);
    router.record(record("after", "p")).await.unwrap();
    drop(router);
    let router = open_router(&path);
    let mut ids: Vec<_> = router
        .pending_for("p")
        .into_iter()
        .map(|r| r.task_id)
        .collect();
    ids.sort();
    assert_eq!(ids, ["after", "live"]);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap().lines().count(),
        2,
        "one line per surviving task"
    );
}

#[test]
fn in_memory_store_compacts_settled_records_only() {
    let store = InMemoryCompletionStore::new();
    let mut done = record("done", "p");
    done.state = CompletionState::Delivered;
    done.updated_at -= Duration::from_secs(60);
    let mut pending = record("pending", "p");
    pending.updated_at -= Duration::from_secs(60);
    store.put(&done).unwrap();
    store.put(&pending).unwrap();
    assert_eq!(store.compact(Duration::from_secs(1)).unwrap(), 1);
    assert!(store.get("pending").is_some());
    assert!(store.get("done").is_none());
}

#[tokio::test]
async fn the_log_stays_appendable_after_compaction_in_the_same_process() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    let router = open_router(&path);
    router.record(record("a", "p")).await.unwrap();
    router.compact(Duration::from_secs(3600)).unwrap();
    router.record(record("b", "p")).await.unwrap();
    drop(router);
    assert_eq!(open_router(&path).pending_for("p").len(), 2);
}

#[tokio::test]
async fn a_cancelled_parent_stays_cancelled_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    {
        let router = open_router(&path);
        router.cancel_parent("p").unwrap();
    }
    let router = open_router(&path);
    assert_eq!(
        router.record(record("late", "p")).await.unwrap(),
        RecordOutcome::Suppressed
    );
    router.resume_parent("p");
    drop(router);
    let router = open_router(&path);
    assert!(matches!(
        router.record(record("later", "p")).await.unwrap(),
        RecordOutcome::Recorded { .. }
    ));
}

#[tokio::test]
async fn an_unterminated_whitespace_tail_is_repaired() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    let line = serde_json::to_string(&record("t1", "p")).unwrap();
    std::fs::write(&path, format!("{line}\n   ")).unwrap();
    let router = open_router(&path);
    router.record(record("t2", "p")).await.unwrap();
    drop(router);
    assert_eq!(open_router(&path).pending_for("p").len(), 2);
}

#[tokio::test]
async fn compaction_leaves_an_unrelated_tmp_file_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completions.jsonl");
    let unrelated = dir.path().join("completions.jsonl.tmp");
    std::fs::write(&unrelated, "not ours").unwrap();
    let router = open_router(&path);
    router.record(record("a", "p")).await.unwrap();
    router.compact(Duration::from_secs(3600)).unwrap();
    assert_eq!(std::fs::read_to_string(&unrelated).unwrap(), "not ours");
}
