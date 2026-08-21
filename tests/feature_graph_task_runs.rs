//! Feature coverage for the task-run layer (`graph::todos::runs`) through the
//! public crate surface.
//!
//! The unit tests in `src/graph/todos/runs/test.rs` cover each operation in
//! isolation. These exercise the *feature* a host actually depends on: a card
//! is claimed, worked, and either finished or — when its worker dies without
//! saying so — handed back to the queue by a sweep, with a poisonous card
//! eventually parked instead of cycling forever.

use std::sync::Arc;
use std::time::Duration;

use tinyagents::harness::store::{InMemoryStore, Store};
use tinyagents::{
    CardPatch, RunLimits, RunOutcome, TaskCardStatus, TaskRun, staleness_reason, task_run_store,
    todo_store,
};

fn store() -> Arc<dyn Store> {
    Arc::new(InMemoryStore::new())
}

/// Add one card and return its id.
async fn seed_card(store: &Arc<dyn Store>, thread_id: &str, title: &str) -> String {
    todo_store::add(store, thread_id, title, CardPatch::default())
        .await
        .expect("add card")
        .cards
        .last()
        .expect("card present")
        .id
        .clone()
}

/// Rewrite a run's timestamps so it looks abandoned, without waiting for real
/// time to pass.
async fn abandon(store: &Arc<dyn Store>, thread_id: &str, run_id: &str) {
    let mut runs = task_run_store::list_runs(store, thread_id, None)
        .await
        .expect("list runs");
    for run in runs.iter_mut().filter(|run| run.run_id == run_id) {
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
            task_run_store::RUNS_NAMESPACE,
            &key,
            serde_json::to_value(&runs).expect("serialize runs"),
        )
        .await
        .expect("write runs");
}

#[tokio::test]
async fn a_worker_claims_works_and_finishes_a_card() {
    let store = store();
    let thread = "feature-happy-path";
    let card_id = seed_card(&store, thread, "write the release notes").await;

    // Claim: the card moves to InProgress and a run opens alongside it.
    let card = todo_store::claim_card(
        &store,
        thread,
        &card_id,
        &[TaskCardStatus::Todo],
        TaskCardStatus::InProgress,
    )
    .await
    .expect("claim card");
    assert_eq!(card.status, TaskCardStatus::InProgress);

    let run = task_run_store::create_run(&store, thread, None, &card_id, "writer-agent")
        .await
        .expect("open run");
    assert!(run.is_active());

    // Work: heartbeats keep the claim alive, so no sweep touches it.
    for _ in 0..3 {
        task_run_store::update_heartbeat(&store, thread, &run.run_id)
            .await
            .expect("heartbeat");
    }
    let swept = task_run_store::reclaim_stale(&store, thread, &RunLimits::default())
        .await
        .expect("sweep");
    assert_eq!(swept.reclaimed_count, 0);
    assert_eq!(swept.blocked_count, 0);

    // Finish: the run closes with its evidence and the card is marked done.
    let closed = task_run_store::complete_run(
        &store,
        thread,
        &run.run_id,
        RunOutcome::Success,
        None,
        vec!["notes.md".to_string()],
    )
    .await
    .expect("close run");
    assert_eq!(closed.outcome, Some(RunOutcome::Success));
    assert_eq!(closed.evidence, vec!["notes.md".to_string()]);

    todo_store::update_status(&store, thread, &card_id, TaskCardStatus::Done)
        .await
        .expect("mark done");
    let board = todo_store::list(&store, thread).await.expect("list board");
    assert_eq!(board.cards[0].status, TaskCardStatus::Done);

    // The run log is the audit trail: one attempt, and it succeeded.
    let history = task_run_store::list_runs(&store, thread, Some(&card_id))
        .await
        .expect("history");
    assert_eq!(history.len(), 1);
    assert!(!history[0].is_active());
}

#[tokio::test]
async fn a_dead_worker_hands_its_card_back_to_the_queue() {
    let store = store();
    let thread = "feature-dead-worker";
    let card_id = seed_card(&store, thread, "reindex the archive").await;
    todo_store::claim_card(
        &store,
        thread,
        &card_id,
        &[TaskCardStatus::Todo],
        TaskCardStatus::InProgress,
    )
    .await
    .expect("claim card");
    let run = task_run_store::create_run(&store, thread, None, &card_id, "worker-a")
        .await
        .expect("open run");

    // The worker dies mid-card: no completion, no more heartbeats.
    abandon(&store, thread, &run.run_id).await;

    let swept = task_run_store::reclaim_stale(&store, thread, &RunLimits::default())
        .await
        .expect("sweep");
    assert_eq!(swept.reclaimed_count, 1);
    assert_eq!(swept.details[0].new_card_status, "todo");

    // The card is dispatchable again, and a second worker can take it cleanly.
    let board = todo_store::list(&store, thread).await.expect("list board");
    assert_eq!(board.cards[0].status, TaskCardStatus::Todo);

    todo_store::claim_card(
        &store,
        thread,
        &card_id,
        &[TaskCardStatus::Todo],
        TaskCardStatus::InProgress,
    )
    .await
    .expect("second claim");
    let second = task_run_store::create_run(&store, thread, None, &card_id, "worker-b")
        .await
        .expect("second run");
    task_run_store::complete_run(
        &store,
        thread,
        &second.run_id,
        RunOutcome::Success,
        None,
        vec![],
    )
    .await
    .expect("second run finishes");

    let history = task_run_store::list_runs(&store, thread, Some(&card_id))
        .await
        .expect("history");
    assert_eq!(history.len(), 2, "both attempts are recorded");
    assert_eq!(history[0].outcome, Some(RunOutcome::Reclaimed));
    assert_eq!(history[1].outcome, Some(RunOutcome::Success));
}

#[tokio::test]
async fn a_card_that_keeps_killing_workers_stops_cycling() {
    let store = store();
    let thread = "feature-poison";
    let card_id = seed_card(&store, thread, "run the cursed migration").await;
    let limits = RunLimits {
        max_reclaim_count: 2,
        ..RunLimits::default()
    };

    let mut statuses = Vec::new();
    for _ in 0..3 {
        todo_store::claim_card(
            &store,
            thread,
            &card_id,
            &[TaskCardStatus::Todo, TaskCardStatus::Blocked],
            TaskCardStatus::InProgress,
        )
        .await
        .expect("claim card");
        let run = task_run_store::create_run(&store, thread, None, &card_id, "worker")
            .await
            .expect("open run");
        abandon(&store, thread, &run.run_id).await;
        let swept = task_run_store::reclaim_stale(&store, thread, &limits)
            .await
            .expect("sweep");
        statuses.push(swept.details[0].new_card_status.clone());
    }

    // Two attempts get the card back; the one that reaches the limit parks it,
    // and it stays parked from then on.
    assert_eq!(statuses, vec!["todo", "blocked", "blocked"]);
    let board = todo_store::list(&store, thread).await.expect("list board");
    assert_eq!(board.cards[0].status, TaskCardStatus::Blocked);
    assert!(
        board.cards[0]
            .blocker
            .as_deref()
            .unwrap_or_default()
            .contains("exceeding limit of 2")
    );
}

#[tokio::test]
async fn one_thread_sweep_does_not_disturb_another() {
    let store = store();
    let quiet_card = seed_card(&store, "thread-quiet", "leave me alone").await;
    todo_store::claim_card(
        &store,
        "thread-quiet",
        &quiet_card,
        &[TaskCardStatus::Todo],
        TaskCardStatus::InProgress,
    )
    .await
    .expect("claim quiet card");
    task_run_store::create_run(&store, "thread-quiet", None, &quiet_card, "worker")
        .await
        .expect("quiet run");

    let noisy_card = seed_card(&store, "thread-noisy", "abandoned work").await;
    todo_store::claim_card(
        &store,
        "thread-noisy",
        &noisy_card,
        &[TaskCardStatus::Todo],
        TaskCardStatus::InProgress,
    )
    .await
    .expect("claim noisy card");
    let noisy_run = task_run_store::create_run(&store, "thread-noisy", None, &noisy_card, "worker")
        .await
        .expect("noisy run");
    abandon(&store, "thread-noisy", &noisy_run.run_id).await;

    let swept = task_run_store::reclaim_stale(&store, "thread-noisy", &RunLimits::default())
        .await
        .expect("sweep");
    assert_eq!(swept.reclaimed_count, 1);

    let quiet = todo_store::list(&store, "thread-quiet")
        .await
        .expect("quiet board");
    assert_eq!(
        quiet.cards[0].status,
        TaskCardStatus::InProgress,
        "a healthy run on another thread is untouched"
    );
}

/// Give a spawned task room to wake, do its store write, and park again.
async fn settle() {
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
}

#[tokio::test(start_paused = true)]
async fn a_heartbeat_task_keeps_a_long_run_alive_and_stops_on_cancel() {
    let store = store();
    let thread = "feature-heartbeat";
    let run = task_run_store::create_run(&store, thread, None, "task-1", "worker")
        .await
        .expect("open run");
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

    task_run_store::spawn_heartbeat_task(
        store.clone(),
        thread.to_string(),
        run.run_id.clone(),
        cancel_rx,
        Duration::from_secs(30),
    );

    // Staleness is judged against the wall clock, which a paused tokio timer
    // does not move — so the run is aged by rewriting its stamps, and a tick
    // proves itself by writing a fresh one back.
    let limits = RunLimits {
        heartbeat_stale_secs: 60,
        claim_ttl_secs: u64::MAX,
        max_reclaim_count: 3,
    };
    for _ in 0..3 {
        abandon(&store, thread, &run.run_id).await;
        tokio::time::advance(Duration::from_secs(30)).await;
        settle().await;
        assert!(
            task_run_store::find_stale_runs(&store, thread, &limits)
                .await
                .expect("stale check")
                .is_empty(),
            "a heartbeating run is never stale"
        );
    }

    // Once cancelled the ticks stop, so the next silence is not papered over.
    cancel_tx.send(true).expect("cancel heartbeat");
    settle().await;
    abandon(&store, thread, &run.run_id).await;
    tokio::time::advance(Duration::from_secs(120)).await;
    settle().await;
    assert_eq!(
        task_run_store::find_stale_runs(&store, thread, &limits)
            .await
            .expect("stale check")
            .len(),
        1
    );
}

#[test]
fn the_staleness_policy_is_pure_and_clock_injected() {
    // A host can evaluate the policy against its own clock — no store, no wait.
    let run = TaskRun {
        run_id: "run-1".to_string(),
        card_id: "task-1".to_string(),
        claimed_by: "worker".to_string(),
        claim_token: "token".to_string(),
        started_at: "0".to_string(),
        last_heartbeat_at: "0".to_string(),
        completed_at: None,
        outcome: None,
        error: None,
        evidence: Vec::new(),
    };
    let limits = RunLimits {
        heartbeat_stale_secs: 10,
        claim_ttl_secs: 100,
        max_reclaim_count: 3,
    };

    assert!(staleness_reason(&run, 5_000, &limits).is_none());
    assert!(
        staleness_reason(&run, 20_000, &limits)
            .expect("silent worker")
            .contains("heartbeat stale")
    );
    assert!(
        staleness_reason(&run, 200_000, &limits)
            .expect("expired claim")
            .contains("claim TTL expired")
    );
}
