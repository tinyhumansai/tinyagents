//! End-to-end coverage for two independent harness building blocks:
//!
//! - **Part A** exercises the ordered, bounded [`map_reduce`] helper driving
//!   *real* async agent work: each mapped item spins up (or shares) an
//!   [`AgentHarness`] and performs a genuine `invoke_default` model call. We
//!   assert input-order preservation, skewed-completion ordering, every failure
//!   policy, and concurrency bounding.
//! - **Part B** feeds a durable [`InMemoryEventJournal`] from a real recorded
//!   harness run and exercises windowed / filtered reads, then validates run
//!   lineage queries over an [`InMemoryStatusStore`].

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::HarnessRunStatus;
use tinyagents::harness::ids::{ComponentId, EventId, HarnessPhase, RunId, ThreadId};
use tinyagents::harness::message::Message;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::EventRecorder;
use tinyagents::{
    AgentObservation, FailurePolicy, HarnessEventJournal, HarnessStatusStore, InMemoryEventJournal,
    InMemoryStatusStore, ParallelOptions, TinyAgentsError, map_reduce,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Builds a harness whose default model always replies with `reply`.
fn constant_harness(reply: &str) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("m", Arc::new(MockModel::constant(reply)));
    harness
}

// ---------------------------------------------------------------------------
// Part A — parallel map/reduce over real harness invocations
// ---------------------------------------------------------------------------

/// Test 1: N=5 real harness invocations run concurrently through `map_reduce`
/// (CollectAll). All five succeed and the outcomes stay in input order, each
/// carrying the index of the input that produced it.
#[tokio::test]
async fn map_reduce_runs_real_harness_invocations_in_input_order() {
    let harness = Arc::new(constant_harness("reply"));
    let items: Vec<usize> = (0..5).collect();

    let opts = ParallelOptions::default().with_failure_policy(FailurePolicy::CollectAll);

    let outcome = map_reduce(items, opts, move |index, item| {
        let harness = harness.clone();
        async move {
            // Distinct, index-derived prompt so each unit of work is unique.
            let prompt = format!("question #{item}");
            let run = harness
                .invoke_default(&(), vec![Message::user(prompt)])
                .await?;
            let text = run.text().unwrap_or_default();
            Ok::<_, TinyAgentsError>(format!("{index}:{text}"))
        }
    })
    .await
    .expect("map_reduce should not error under CollectAll");

    assert_eq!(outcome.success_count(), 5);
    assert_eq!(outcome.failure_count(), 0);

    // Outcomes preserve input order: outcome[i].index == i and the value was
    // produced by the i-th closure invocation.
    for (i, item) in outcome.outcomes.iter().enumerate() {
        assert_eq!(item.index, i, "outcome index must match input position");
        assert!(item.is_ok());
        assert_eq!(item.result.as_ref().unwrap(), &format!("{i}:reply"));
    }

    // The borrowed successes are likewise in input order.
    let successes = outcome.successes();
    assert_eq!(successes.len(), 5);
    for (i, text) in successes.iter().enumerate() {
        assert_eq!(*text, &format!("{i}:reply"));
    }
}

/// Test 2: items sleep for descending durations (so completion order is the
/// reverse of input order), yet results still land in input order.
#[tokio::test]
async fn map_reduce_preserves_input_order_under_skewed_completion() {
    let harness = Arc::new(constant_harness("done"));
    let items: Vec<usize> = (0..5).collect();
    let n = items.len();

    let outcome = map_reduce(items, ParallelOptions::default(), move |index, item| {
        let harness = harness.clone();
        async move {
            // Earlier items sleep longest, so they complete last.
            let millis = ((n - index) as u64) * 20;
            tokio::time::sleep(Duration::from_millis(millis)).await;
            let run = harness
                .invoke_default(&(), vec![Message::user(format!("item {item}"))])
                .await?;
            Ok::<_, TinyAgentsError>(format!("{index}:{}", run.text().unwrap_or_default()))
        }
    })
    .await
    .expect("map_reduce should succeed");

    assert_eq!(outcome.success_count(), 5);
    let ordered = outcome.into_successes();
    let expected: Vec<String> = (0..5).map(|i| format!("{i}:done")).collect();
    assert_eq!(
        ordered, expected,
        "results must follow input order, not completion order"
    );
}

/// Test 3: each failure policy behaves as documented when roughly half the
/// items fail. Odd indices fail; even indices succeed (3 successes of 5).
#[tokio::test]
async fn map_reduce_honors_each_failure_policy() {
    // Closure factory: even index succeeds, odd index fails.
    fn work(
        harness: Arc<AgentHarness<()>>,
    ) -> impl Fn(
        usize,
        usize,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = tinyagents::Result<String>> + Send>,
    > {
        move |index, _item| {
            let harness = harness.clone();
            Box::pin(async move {
                if index % 2 == 1 {
                    return Err(TinyAgentsError::Graph(format!("item {index} failed")));
                }
                let run = harness
                    .invoke_default(&(), vec![Message::user(format!("ok {index}"))])
                    .await?;
                Ok::<_, TinyAgentsError>(format!("{index}:{}", run.text().unwrap_or_default()))
            })
        }
    }

    let harness = Arc::new(constant_harness("ok"));

    // CollectAll: never errors; records per-item successes and failures.
    let collect = map_reduce(
        (0..5).collect::<Vec<_>>(),
        ParallelOptions::default().with_failure_policy(FailurePolicy::CollectAll),
        work(harness.clone()),
    )
    .await
    .expect("CollectAll never errors");
    assert_eq!(collect.success_count(), 3);
    assert_eq!(collect.failure_count(), 2);
    // Failures are recorded, not raised, and remain index-tagged in order.
    for item in &collect.outcomes {
        assert_eq!(item.is_ok(), item.index % 2 == 0);
    }

    // FailFast: returns the first error (in input order → index 1).
    let fail_fast = map_reduce(
        (0..5).collect::<Vec<_>>(),
        ParallelOptions::default().with_failure_policy(FailurePolicy::FailFast),
        work(harness.clone()),
    )
    .await;
    match fail_fast {
        Err(TinyAgentsError::Graph(msg)) => assert_eq!(msg, "item 1 failed"),
        other => panic!("FailFast should return the first item error, got {other:?}"),
    }

    // Quorum below the success count errors.
    let quorum_high = map_reduce(
        (0..5).collect::<Vec<_>>(),
        ParallelOptions::default().with_failure_policy(FailurePolicy::Quorum(4)),
        work(harness.clone()),
    )
    .await;
    assert!(
        matches!(quorum_high, Err(TinyAgentsError::Graph(_))),
        "Quorum(4) with only 3 successes must error"
    );

    // Quorum at/below the success count is Ok.
    let quorum_ok = map_reduce(
        (0..5).collect::<Vec<_>>(),
        ParallelOptions::default().with_failure_policy(FailurePolicy::Quorum(3)),
        work(harness.clone()),
    )
    .await
    .expect("Quorum(3) with 3 successes must be Ok");
    assert_eq!(quorum_ok.success_count(), 3);

    // BestEffort: Ok, keeping only the successful outputs.
    let best = map_reduce(
        (0..5).collect::<Vec<_>>(),
        ParallelOptions::default().with_failure_policy(FailurePolicy::BestEffort),
        work(harness),
    )
    .await
    .expect("BestEffort never errors");
    assert_eq!(best.success_count(), 3);
    assert_eq!(
        best.failure_count(),
        0,
        "BestEffort drops failures entirely"
    );
    let kept = best.into_successes();
    assert_eq!(
        kept,
        vec!["0:ok".to_string(), "2:ok".to_string(), "4:ok".to_string()]
    );
}

/// Test 4: `with_max_concurrency(2)` bounds the number of simultaneously
/// executing closures. A live counter tracks in-flight work and a peak counter
/// records the high-water mark, which must never exceed 2.
#[tokio::test]
async fn map_reduce_bounds_concurrency() {
    let harness = Arc::new(constant_harness("x"));
    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));

    let opts = ParallelOptions::default().with_max_concurrency(2);

    let live_c = live.clone();
    let peak_c = peak.clone();
    let outcome = map_reduce((0..8).collect::<Vec<_>>(), opts, move |index, _item| {
        let harness = harness.clone();
        let live = live_c.clone();
        let peak = peak_c.clone();
        async move {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            // Hold the slot long enough for overlap to be observable.
            tokio::time::sleep(Duration::from_millis(15)).await;
            let run = harness
                .invoke_default(&(), vec![Message::user(format!("n{index}"))])
                .await?;
            live.fetch_sub(1, Ordering::SeqCst);
            Ok::<_, TinyAgentsError>(run.text().unwrap_or_default())
        }
    })
    .await
    .expect("bounded map_reduce should succeed");

    assert_eq!(outcome.success_count(), 8);
    let observed_peak = peak.load(Ordering::SeqCst);
    assert!(
        observed_peak <= 2,
        "peak concurrency {observed_peak} must not exceed the configured bound of 2"
    );
    assert!(
        observed_peak >= 1,
        "at least one closure must have run (peak was {observed_peak})"
    );
}

// ---------------------------------------------------------------------------
// Part B — journals fed from a real run, plus status lineage
// ---------------------------------------------------------------------------

/// Test 5: run a real harness with an [`EventRecorder`], wrap each recorded
/// [`AgentEvent`] into an [`AgentObservation`] (increasing offset, shared
/// run id) and append them to an [`InMemoryEventJournal`]. Then exercise
/// full, windowed, and kind-filtered reads.
#[tokio::test]
async fn journal_windowed_and_filtered_reads_from_real_run() {
    let harness = constant_harness("hello journal");
    let recorder = EventRecorder::new();

    let run_id_str = "journal-run";
    let ctx: RunContext<()> =
        RunContext::new(RunConfig::new(run_id_str), ()).with_events(recorder.sink());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("real harness run should succeed");

    let events = recorder.events();
    assert!(
        events.len() >= 2,
        "a real run should emit at least run.started and run.completed"
    );

    // Feed the durable journal from the real recorded events.
    let journal = InMemoryEventJournal::default();
    let run_id = RunId::new(run_id_str);
    for (offset, event) in events.iter().cloned().enumerate() {
        let obs = AgentObservation {
            event_id: EventId::new(format!("e{offset}")),
            run_id: run_id.clone(),
            parent_run_id: None,
            root_run_id: run_id.clone(),
            offset: offset as u64,
            ts_ms: offset as u64,
            event,
        };
        journal.append(obs).await.expect("append should succeed");
    }

    // read_from(0) replays the entire run.
    let all = journal
        .read_from(run_id_str, 0)
        .await
        .expect("read_from should succeed");
    assert_eq!(all.len(), events.len());

    // read_window truncates to at most `limit`.
    let window = journal
        .read_window(run_id_str, 0, 2)
        .await
        .expect("read_window should succeed");
    assert!(window.len() <= 2, "window must honor the limit");
    assert_eq!(window.len(), 2.min(events.len()));

    // read_filtered returns only the requested kinds. A real run always emits
    // run.started and run.completed.
    let kinds = ["run.started", "run.completed"];
    let filtered = journal
        .read_filtered(run_id_str, 0, &kinds)
        .await
        .expect("read_filtered should succeed");
    assert!(
        !filtered.is_empty(),
        "run.started/run.completed should be present in a real run"
    );
    for obs in &filtered {
        assert!(
            obs.event.kind().starts_with("run."),
            "filtered observation kind {:?} should start with run.",
            obs.event.kind()
        );
        assert!(kinds.contains(&obs.event.kind()));
    }
    // Both terminal-lifecycle kinds should be captured.
    assert!(filtered.iter().any(|o| o.event.kind() == "run.started"));
    assert!(filtered.iter().any(|o| o.event.kind() == "run.completed"));

    // An unknown run reads back empty.
    let missing = journal
        .read_from("no-such-run", 0)
        .await
        .expect("reading an unknown run is Ok");
    assert!(missing.is_empty());
}

/// Test 6: build a small run tree in an [`InMemoryStatusStore`] and validate
/// lineage queries: `list_by_root` returns every descendant of a root, while
/// `list_active` excludes terminal (completed) runs.
#[tokio::test]
async fn status_lineage_by_root_and_active() {
    let store = InMemoryStatusStore::default();
    let component = ComponentId::new("agent_loop");
    let root = RunId::new("root");

    // Parent (the root itself), running.
    let mut parent = HarnessRunStatus::new(root.clone(), component.clone())
        .with_thread(ThreadId::new("thread-1"));
    parent.mark_running(HarnessPhase::Model);

    // Child under the root, running.
    let mut child = HarnessRunStatus::new(RunId::new("child"), component.clone())
        .with_parent(root.clone(), root.clone());
    child.mark_running(HarnessPhase::Tools);

    // Completed sibling under the same root.
    let mut sibling = HarnessRunStatus::new(RunId::new("sibling"), component.clone())
        .with_parent(root.clone(), root.clone());
    sibling.mark_completed();

    // Unrelated run under a different root.
    let mut unrelated = HarnessRunStatus::new(RunId::new("other"), component.clone());
    unrelated.mark_running(HarnessPhase::Model);

    for status in [parent, child, sibling, unrelated] {
        store.put_status(status).await.expect("put_status");
    }

    // list_by_root returns the 3 descendants (root, child, sibling).
    let mut lineage = store.list_by_root("root").await.expect("list_by_root");
    lineage.sort_by(|a, b| a.run_id.as_str().cmp(b.run_id.as_str()));
    let lineage_ids: Vec<&str> = lineage.iter().map(|s| s.run_id.as_str()).collect();
    assert_eq!(lineage_ids, vec!["child", "root", "sibling"]);
    assert!(
        !lineage_ids.contains(&"other"),
        "the unrelated run must not appear under this root"
    );

    // list_active excludes the completed sibling but keeps the running runs.
    let active = store.list_active().await.expect("list_active");
    let active_ids: Vec<&str> = active.iter().map(|s| s.run_id.as_str()).collect();
    assert!(active_ids.contains(&"root"));
    assert!(active_ids.contains(&"child"));
    assert!(active_ids.contains(&"other"));
    assert!(
        !active_ids.contains(&"sibling"),
        "the completed sibling must be excluded from active runs"
    );

    // list_by_thread finds the thread-tagged root.
    let by_thread = store
        .list_by_thread("thread-1")
        .await
        .expect("list_by_thread");
    assert_eq!(by_thread.len(), 1);
    assert_eq!(by_thread[0].run_id.as_str(), "root");

    // get_status round-trips a known run.
    let got = store.get_status("child").await.expect("get_status");
    assert!(got.is_some());
}
