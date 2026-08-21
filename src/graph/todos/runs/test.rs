//! Unit tests for the task-run claim/heartbeat/reclaim layer.

use std::sync::Arc;

use super::store::{
    complete_run, count_reclaims_for_card, create_run, find_stale_runs, get_run, list_runs,
    reclaim_stale, update_heartbeat,
};
use super::types::{RunLimits, RunOutcome, TaskRun, staleness_reason};
use crate::graph::todos::store as board;
use crate::graph::todos::types::TaskCardStatus;
use crate::harness::store::{InMemoryStore, Store};

fn store() -> Arc<dyn Store> {
    Arc::new(InMemoryStore::new())
}

/// A run with hand-set timestamps, so staleness is testable without waiting.
fn run_at(run_id: &str, card_id: &str, started_ms: u64, heartbeat_ms: u64) -> TaskRun {
    TaskRun {
        run_id: run_id.to_string(),
        card_id: card_id.to_string(),
        claimed_by: "worker".to_string(),
        claim_token: "token".to_string(),
        started_at: started_ms.to_string(),
        last_heartbeat_at: heartbeat_ms.to_string(),
        completed_at: None,
        outcome: None,
        error: None,
        evidence: Vec::new(),
    }
}

async fn seed_card(store: &Arc<dyn Store>, thread_id: &str, title: &str) -> String {
    let snapshot = board::add(store, thread_id, title, Default::default())
        .await
        .expect("add card");
    snapshot.cards.last().expect("card added").id.clone()
}

#[tokio::test]
async fn create_records_an_active_run_listable_by_card() {
    let store = store();
    let run = create_run(&store, "thread-1", None, "task-1", "worker-a")
        .await
        .unwrap();

    assert_eq!(run.card_id, "task-1");
    assert_eq!(run.claimed_by, "worker-a");
    assert!(run.is_active());
    assert!(!run.claim_token.is_empty());
    assert!(!run.run_id.is_empty());

    assert_eq!(list_runs(&store, "thread-1", None).await.unwrap().len(), 1);
    assert_eq!(
        list_runs(&store, "thread-1", Some("task-1"))
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        list_runs(&store, "thread-1", Some("task-other"))
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn runs_are_scoped_to_their_thread() {
    let store = store();
    create_run(&store, "thread-a", Some("run-a"), "task-1", "worker")
        .await
        .unwrap();
    create_run(&store, "thread-b", Some("run-b"), "task-1", "worker")
        .await
        .unwrap();

    let a = list_runs(&store, "thread-a", None).await.unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].run_id, "run-a");
    assert!(
        get_run(&store, "thread-a", "run-b")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn a_supplied_run_id_cannot_be_reused_on_one_thread() {
    let store = store();
    create_run(&store, "thread-1", Some("run-1"), "task-1", "worker")
        .await
        .unwrap();
    let error = create_run(&store, "thread-1", Some("run-1"), "task-2", "worker")
        .await
        .expect_err("duplicate run id must be rejected");
    assert!(error.to_string().contains("already exists"));
}

#[tokio::test]
async fn create_rejects_a_blank_thread_id() {
    let store = store();
    assert!(
        create_run(&store, "   ", None, "task-1", "worker")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn heartbeat_advances_the_liveness_stamp() {
    let store = store();
    let run = create_run(&store, "thread-1", None, "task-1", "worker")
        .await
        .unwrap();

    update_heartbeat(&store, "thread-1", &run.run_id)
        .await
        .unwrap();

    let after = get_run(&store, "thread-1", &run.run_id)
        .await
        .unwrap()
        .unwrap();
    assert!(after.last_heartbeat_at >= run.last_heartbeat_at);
    assert!(after.is_active());
}

#[tokio::test]
async fn heartbeat_and_completion_reject_a_finished_run() {
    let store = store();
    let run = create_run(&store, "thread-1", None, "task-1", "worker")
        .await
        .unwrap();
    complete_run(
        &store,
        "thread-1",
        &run.run_id,
        RunOutcome::Success,
        None,
        vec!["pr #12".to_string()],
    )
    .await
    .unwrap();

    // A finished run is not resurrected, and it is not completed twice.
    assert!(
        update_heartbeat(&store, "thread-1", &run.run_id)
            .await
            .is_err()
    );
    assert!(
        complete_run(
            &store,
            "thread-1",
            &run.run_id,
            RunOutcome::Failed,
            None,
            vec![]
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn completion_records_outcome_error_and_evidence() {
    let store = store();
    let run = create_run(&store, "thread-1", None, "task-1", "worker")
        .await
        .unwrap();
    let done = complete_run(
        &store,
        "thread-1",
        &run.run_id,
        RunOutcome::Failed,
        Some("provider timed out".to_string()),
        vec!["log line".to_string()],
    )
    .await
    .unwrap();

    assert!(!done.is_active());
    assert_eq!(done.outcome, Some(RunOutcome::Failed));
    assert_eq!(done.error.as_deref(), Some("provider timed out"));
    assert_eq!(done.evidence, vec!["log line".to_string()]);
    assert!(done.completed_at.is_some());
}

#[test]
fn staleness_reports_ttl_before_heartbeat() {
    let limits = RunLimits {
        heartbeat_stale_secs: 10,
        claim_ttl_secs: 60,
        max_reclaim_count: 3,
    };
    let now = 1_000_000u64;

    // Healthy: young claim, fresh heartbeat.
    let healthy = run_at("r", "c", now - 5_000, now - 1_000);
    assert!(staleness_reason(&healthy, now, &limits).is_none());

    // Silent worker, claim still inside its TTL.
    let silent = run_at("r", "c", now - 30_000, now - 30_000);
    let reason = staleness_reason(&silent, now, &limits).expect("stale heartbeat");
    assert!(reason.contains("heartbeat stale"), "{reason}");

    // Both aged out: the TTL is the reason reported.
    let expired = run_at("r", "c", now - 120_000, now - 120_000);
    let reason = staleness_reason(&expired, now, &limits).expect("expired claim");
    assert!(reason.contains("claim TTL expired"), "{reason}");
}

#[test]
fn staleness_treats_an_unparsable_stamp_as_healthy() {
    // A corrupt record must never cause a live worker's card to be yanked away.
    let mut run = run_at("r", "c", 0, 0);
    run.started_at = "not-a-timestamp".to_string();
    assert!(staleness_reason(&run, 9_999_999, &RunLimits::default()).is_none());
}

#[tokio::test]
async fn find_stale_runs_ignores_completed_runs() {
    let store = store();
    let run = create_run(&store, "thread-1", None, "task-1", "worker")
        .await
        .unwrap();
    complete_run(
        &store,
        "thread-1",
        &run.run_id,
        RunOutcome::Success,
        None,
        vec![],
    )
    .await
    .unwrap();

    // Zero limits would make any *active* run stale; this one is finished.
    let limits = RunLimits {
        heartbeat_stale_secs: 0,
        claim_ttl_secs: 0,
        max_reclaim_count: 3,
    };
    assert!(
        find_stale_runs(&store, "thread-1", &limits)
            .await
            .unwrap()
            .is_empty()
    );
}

/// Force `run_id` to look ancient by rewriting its stamps in place.
async fn age_run(store: &Arc<dyn Store>, thread_id: &str, run_id: &str) {
    let mut runs = list_runs(store, thread_id, None).await.unwrap();
    for run in runs.iter_mut().filter(|r| r.run_id == run_id) {
        run.started_at = "0".to_string();
        run.last_heartbeat_at = "0".to_string();
    }
    let key: String = thread_id
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    store
        .put(
            super::store::RUNS_NAMESPACE,
            &key,
            serde_json::to_value(&runs).unwrap(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn reclaim_returns_a_wedged_card_to_the_queue() {
    let store = store();
    let thread_id = "thread-reclaim";
    let card_id = seed_card(&store, thread_id, "ship the thing").await;
    board::claim_card(
        &store,
        thread_id,
        &card_id,
        &[TaskCardStatus::Todo],
        TaskCardStatus::InProgress,
    )
    .await
    .unwrap();
    let run = create_run(&store, thread_id, None, &card_id, "worker")
        .await
        .unwrap();
    age_run(&store, thread_id, &run.run_id).await;

    let result = reclaim_stale(&store, thread_id, &RunLimits::default())
        .await
        .unwrap();

    assert_eq!(result.reclaimed_count, 1);
    assert_eq!(result.blocked_count, 0);
    assert_eq!(result.details.len(), 1);
    assert_eq!(result.details[0].card_id, card_id);
    assert_eq!(result.details[0].new_card_status, "todo");

    let closed = get_run(&store, thread_id, &run.run_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(closed.outcome, Some(RunOutcome::Reclaimed));
    assert!(closed.error.is_some(), "the reason is recorded on the run");

    let snapshot = board::list(&store, thread_id).await.unwrap();
    assert_eq!(snapshot.cards[0].status, TaskCardStatus::Todo);
}

#[tokio::test]
async fn a_card_that_keeps_wedging_workers_parks_as_blocked() {
    let store = store();
    let thread_id = "thread-poison";
    let card_id = seed_card(&store, thread_id, "poison card").await;
    let limits = RunLimits {
        max_reclaim_count: 2,
        ..RunLimits::default()
    };

    // Two reclaims are tolerated; the second reaches the limit and parks it.
    for expected_status in ["todo", "blocked"] {
        board::claim_card(
            &store,
            thread_id,
            &card_id,
            &[TaskCardStatus::Todo, TaskCardStatus::Blocked],
            TaskCardStatus::InProgress,
        )
        .await
        .unwrap();
        let run = create_run(&store, thread_id, None, &card_id, "worker")
            .await
            .unwrap();
        age_run(&store, thread_id, &run.run_id).await;
        let result = reclaim_stale(&store, thread_id, &limits).await.unwrap();
        assert_eq!(result.details[0].new_card_status, expected_status);
    }

    assert_eq!(
        count_reclaims_for_card(&store, thread_id, &card_id)
            .await
            .unwrap(),
        2
    );
    let snapshot = board::list(&store, thread_id).await.unwrap();
    assert_eq!(snapshot.cards[0].status, TaskCardStatus::Blocked);
    let blocker = snapshot.cards[0].blocker.as_deref().unwrap_or_default();
    assert!(blocker.contains("exceeding limit of 2"), "{blocker}");
}

#[tokio::test]
async fn reclaim_leaves_a_healthy_run_alone() {
    let store = store();
    let thread_id = "thread-healthy";
    let card_id = seed_card(&store, thread_id, "in flight").await;
    board::claim_card(
        &store,
        thread_id,
        &card_id,
        &[TaskCardStatus::Todo],
        TaskCardStatus::InProgress,
    )
    .await
    .unwrap();
    create_run(&store, thread_id, None, &card_id, "worker")
        .await
        .unwrap();

    let result = reclaim_stale(&store, thread_id, &RunLimits::default())
        .await
        .unwrap();

    assert_eq!(result, Default::default());
    let snapshot = board::list(&store, thread_id).await.unwrap();
    assert_eq!(snapshot.cards[0].status, TaskCardStatus::InProgress);
}

#[tokio::test]
async fn reclaim_on_a_thread_with_no_runs_is_a_no_op() {
    let store = store();
    let result = reclaim_stale(&store, "quiet-thread", &RunLimits::default())
        .await
        .unwrap();
    assert_eq!(result.reclaimed_count, 0);
    assert!(result.details.is_empty());
}

#[test]
fn a_run_round_trips_through_json() {
    let run = run_at("run-1", "task-1", 10, 20);
    let json = serde_json::to_value(&run).unwrap();
    // camelCase on the wire, so a host's UI and RPC layer read it directly.
    assert_eq!(json["runId"], "run-1");
    assert_eq!(json["cardId"], "task-1");
    assert!(
        json.get("completedAt").is_none(),
        "absent fields stay absent"
    );
    assert_eq!(serde_json::from_value::<TaskRun>(json).unwrap(), run);
}

#[test]
fn default_limits_are_the_documented_policy() {
    let limits = RunLimits::default();
    assert_eq!(
        limits.heartbeat_stale_secs,
        super::DEFAULT_HEARTBEAT_STALE_SECS
    );
    assert_eq!(limits.claim_ttl_secs, super::DEFAULT_CLAIM_TTL_SECS);
    assert_eq!(limits.max_reclaim_count, super::DEFAULT_MAX_RECLAIM_COUNT);
    assert!(limits.claim_ttl_secs > limits.heartbeat_stale_secs);
}

#[tokio::test]
async fn import_if_absent_never_replaces_an_existing_log() {
    let store = store();
    let imported = vec![run_at("legacy-run", "task-1", 10, 20)];
    assert!(
        super::store::import_if_absent(&store, "thread-1", imported.clone())
            .await
            .unwrap()
    );
    assert_eq!(list_runs(&store, "thread-1", None).await.unwrap(), imported);

    // A second import is refused, so a re-run of a migration cannot duplicate
    // the reclaim history the sweep's budget is counted from.
    assert!(
        !super::store::import_if_absent(&store, "thread-1", vec![run_at("other", "task-2", 1, 2)])
            .await
            .unwrap()
    );
    assert_eq!(list_runs(&store, "thread-1", None).await.unwrap(), imported);
}
