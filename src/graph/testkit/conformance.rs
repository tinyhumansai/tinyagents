//! Reusable storage **conformance** (contract) suites.
//!
//! Durable graph stores are hard to migrate safely without a shared contract:
//! two backends that both implement a trait should behave identically. These
//! functions encode that contract once so any backend — the built-in ones or a
//! caller-supplied adapter — can be certified by running the same assertions.
//!
//! Each function panics with a descriptive message on the first violation, so
//! call them from a `#[tokio::test]` / `#[test]`.

use crate::graph::checkpoint::{Checkpoint, Checkpointer};
use crate::graph::orchestration::{
    OrchestrationTaskFilter, OrchestrationTaskKind, OrchestrationTaskResult,
    OrchestrationTaskStatus, TaskStore,
};
use crate::harness::ids::{NodeId, TaskId};

fn contract_checkpoint(
    thread: &str,
    id: &str,
    parent: Option<&str>,
    step: usize,
) -> Checkpoint<i32> {
    Checkpoint {
        thread_id: thread.to_string(),
        checkpoint_id: id.to_string(),
        run_id: None,
        parent_checkpoint_id: parent.map(str::to_string),
        namespace: vec![],
        state: step as i32,
        next_nodes: vec![NodeId::from("n")],
        completed_tasks: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        pending_activations: None,
        barrier_arrivals: vec![],
        metadata: serde_json::json!({ "source": "loop", "step": step }),
    }
}

/// Runs the [`Checkpointer`] contract against `cp`.
///
/// Covers put/get (latest and specific), insertion-order listing, unknown-thread
/// misses, `list_threads`, `delete_thread`, and `prune`. Any backend that
/// passes this behaves interchangeably for the durable graph runtime.
pub async fn checkpointer_contract<C>(cp: C)
where
    C: Checkpointer<i32>,
{
    cp.put(contract_checkpoint("t1", "c1", None, 1))
        .await
        .expect("put c1");
    cp.put(contract_checkpoint("t1", "c2", Some("c1"), 2))
        .await
        .expect("put c2");

    // Latest wins.
    let latest = cp.get("t1", None).await.expect("get latest").expect("some");
    assert_eq!(latest.checkpoint_id, "c2", "latest checkpoint id");
    assert_eq!(latest.state, 2, "latest state");

    // Specific lookup.
    let first = cp
        .get("t1", Some("c1"))
        .await
        .expect("get specific")
        .expect("some");
    assert_eq!(first.checkpoint_id, "c1", "specific checkpoint id");

    // Duplicate-id lookup: every backend resolves a re-used checkpoint id to
    // the *last* written record (matching the append-only history and
    // `get(None)`), so the three backends stay interchangeable.
    cp.put(contract_checkpoint("dup", "d1", None, 1))
        .await
        .expect("put dup first");
    cp.put(contract_checkpoint("dup", "d1", None, 9))
        .await
        .expect("put dup second");
    let dup = cp
        .get("dup", Some("d1"))
        .await
        .expect("get dup")
        .expect("some");
    assert_eq!(
        dup.state, 9,
        "duplicate checkpoint id resolves to the last written record"
    );

    // Unknown thread / checkpoint.
    assert!(cp.get("nope", None).await.expect("get miss").is_none());
    assert!(
        cp.get("t1", Some("missing"))
            .await
            .expect("get miss")
            .is_none()
    );

    // Namespace-scoped lookup: checkpoints sharing a thread but differing in
    // namespace never resolve to each other. `get_scoped(None, ns)` returns the
    // latest in that namespace, not the global latest for the thread.
    let mut root = contract_checkpoint("ns", "root1", None, 1);
    root.namespace = vec![];
    cp.put(root).await.expect("put root ns");
    let mut child = contract_checkpoint("ns", "child1", None, 5);
    child.namespace = vec!["child".to_string()];
    cp.put(child).await.expect("put child ns");
    // The global latest is the child checkpoint, but the root namespace still
    // resolves to root1 (and vice versa).
    let scoped_root = cp
        .get_scoped("ns", None, &[])
        .await
        .expect("scoped root")
        .expect("some");
    assert_eq!(
        scoped_root.checkpoint_id, "root1",
        "root namespace ignores the child-namespace checkpoint"
    );
    let scoped_child = cp
        .get_scoped("ns", None, &["child".to_string()])
        .await
        .expect("scoped child")
        .expect("some");
    assert_eq!(
        scoped_child.checkpoint_id, "child1",
        "child namespace resolves to its own checkpoint"
    );

    // Listing preserves insertion order and projects lineage.
    let list = cp.list("t1").await.expect("list");
    assert_eq!(list.len(), 2, "listed count");
    assert_eq!(list[0].checkpoint_id, "c1", "list order[0]");
    assert_eq!(
        list[1].parent_checkpoint_id.as_deref(),
        Some("c1"),
        "list lineage"
    );

    // Thread listing includes the written thread.
    let threads = cp.list_threads().await.expect("list_threads");
    assert!(
        threads.iter().any(|t| t == "t1"),
        "list_threads contains t1"
    );

    // Prune keeps a window (and the ancestor chain of what it keeps).
    let second_thread = "t2";
    for i in 1..=3 {
        let parent = (i > 1).then(|| format!("c{}", i - 1));
        cp.put(contract_checkpoint(
            second_thread,
            &format!("c{i}"),
            parent.as_deref(),
            i,
        ))
        .await
        .expect("put for prune");
    }
    cp.prune(second_thread, 1).await.expect("prune");
    let pruned = cp.list(second_thread).await.expect("list pruned");
    assert!(!pruned.is_empty(), "prune keeps at least the window");

    // Delete removes the thread entirely.
    cp.delete_thread("t1").await.expect("delete_thread");
    assert!(
        cp.get("t1", None)
            .await
            .expect("get after delete")
            .is_none()
    );
}

/// Runs the [`TaskStore`] contract against `store`.
///
/// Covers the full lifecycle state machine (pending → running → complete /
/// fail / timeout), cooperative cancellation, kill, deadline updates, filtering
/// by status, and terminal-transition rejection. Any backend that passes this
/// behaves interchangeably for orchestration.
pub fn taskstore_contract<S>(store: S)
where
    S: TaskStore,
{
    let spec = |id: &str| {
        crate::graph::orchestration::OrchestrationTaskSpec::new(
            id,
            OrchestrationTaskKind::SubAgent {
                agent: "worker".into(),
            },
        )
    };

    // Happy path: pending → running → completed.
    let happy = TaskId::new("happy");
    let record = store.insert(spec("happy")).expect("insert");
    assert_eq!(record.status, OrchestrationTaskStatus::Pending);
    assert_eq!(
        store.mark_running(&happy).expect("running").status,
        OrchestrationTaskStatus::Running
    );
    let done = store
        .complete(&happy, OrchestrationTaskResult::text("ok"))
        .expect("complete");
    assert_eq!(done.status, OrchestrationTaskStatus::Completed);
    assert!(done.ended_at.is_some(), "completed sets ended_at");

    // Terminal tasks reject further control.
    assert!(
        store.request_cancel(&happy).is_err(),
        "terminal task rejects cancel"
    );

    // Failure path.
    let bad = TaskId::new("bad");
    store.insert(spec("bad")).expect("insert bad");
    assert_eq!(
        store.fail(&bad, "boom".into()).expect("fail").status,
        OrchestrationTaskStatus::Failed
    );

    // Timeout path.
    let slow = TaskId::new("slow");
    store.insert(spec("slow")).expect("insert slow");
    assert_eq!(
        store
            .timeout(&slow, "deadline".into())
            .expect("timeout")
            .status,
        OrchestrationTaskStatus::TimedOut
    );

    // Cooperative cancellation: request then mark.
    let cancelable = TaskId::new("cancelable");
    store.insert(spec("cancelable")).expect("insert cancelable");
    store.request_cancel(&cancelable).expect("request_cancel");
    assert_eq!(
        store.get(&cancelable).expect("get").status,
        OrchestrationTaskStatus::CancelRequested
    );
    assert_eq!(
        store.mark_cancelled(&cancelable).expect("mark").status,
        OrchestrationTaskStatus::Cancelled
    );

    // Kill abandons a live task.
    let doomed = TaskId::new("doomed");
    store.insert(spec("doomed")).expect("insert doomed");
    store.kill(&doomed).expect("kill");
    assert_eq!(
        store.get(&doomed).expect("get").status,
        OrchestrationTaskStatus::Abandoned
    );

    // Deadline update on a live task.
    let timed = TaskId::new("timed");
    store.insert(spec("timed")).expect("insert timed");
    let updated = store.set_timeout_ms(&timed, 1234).expect("set_timeout_ms");
    assert_eq!(updated.spec.timeout_ms, Some(1234));

    // Filtering by status returns only matches.
    let completed = store.list(OrchestrationTaskFilter {
        status: Some(OrchestrationTaskStatus::Completed),
        ..OrchestrationTaskFilter::default()
    });
    assert_eq!(completed.len(), 1, "one completed task");
    assert_eq!(completed[0].spec.task_id.as_str(), "happy");
}

/// Concurrent-write contract for a [`Checkpointer`]: many tasks put distinct
/// checkpoints on one shared instance concurrently; every put must be durable
/// and retrievable with no lost writes.
pub async fn checkpointer_concurrent_contract<C>(cp: std::sync::Arc<C>)
where
    C: Checkpointer<i32> + 'static,
{
    const WRITERS: usize = 8;
    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let cp = cp.clone();
        handles.push(tokio::spawn(async move {
            cp.put(contract_checkpoint("shared", &format!("c{w}"), None, w))
                .await
                .expect("concurrent put");
        }));
    }
    for h in handles {
        h.await.expect("writer task joins");
    }

    // Every concurrently-written checkpoint is retrievable by id.
    for w in 0..WRITERS {
        let got = cp.get("shared", Some(&format!("c{w}"))).await.expect("get");
        assert!(got.is_some(), "checkpoint c{w} survived concurrent writes");
    }
}

/// Concurrent-write contract: many threads insert and advance distinct tasks
/// against one shared store; every write must land exactly once with no lost
/// updates. Any backend that passes is safe to share across a run tree.
pub fn taskstore_concurrent_contract<S>(store: std::sync::Arc<S>)
where
    S: TaskStore + 'static,
{
    const WRITERS: usize = 8;
    std::thread::scope(|scope| {
        for w in 0..WRITERS {
            let store = store.clone();
            scope.spawn(move || {
                let id = TaskId::new(format!("task-{w}"));
                store
                    .insert(crate::graph::orchestration::OrchestrationTaskSpec::new(
                        id.as_str(),
                        OrchestrationTaskKind::SubAgent {
                            agent: "worker".into(),
                        },
                    ))
                    .expect("concurrent insert");
                store.mark_running(&id).expect("concurrent mark_running");
            });
        }
    });

    let all = store.list(OrchestrationTaskFilter::default());
    assert_eq!(all.len(), WRITERS, "every concurrent insert must land once");
    assert!(
        all.iter()
            .all(|r| r.status == OrchestrationTaskStatus::Running),
        "every concurrently-advanced task reached Running"
    );
}

/// Durable replay contract: state written through one handle must survive
/// re-opening the backing store through `reopen`. Only meaningful for durable
/// backends (call it for those; an in-memory store would drop state on reopen).
pub fn taskstore_replay_contract<S, F>(reopen: F)
where
    S: TaskStore,
    F: Fn() -> S,
{
    let id = TaskId::new("survivor");
    let spec = || {
        crate::graph::orchestration::OrchestrationTaskSpec::new(
            "survivor",
            OrchestrationTaskKind::SubAgent {
                agent: "worker".into(),
            },
        )
    };
    {
        let store = reopen();
        store.insert(spec()).expect("insert");
        store.mark_running(&id).expect("running");
        store
            .complete(&id, OrchestrationTaskResult::text("done"))
            .expect("complete");
    }
    // Re-open: the terminal state and full transition history are reconstructed.
    let reopened = reopen();
    let record = reopened.get(&id).expect("task survives reopen");
    assert_eq!(record.status, OrchestrationTaskStatus::Completed);
    assert_eq!(
        reopened.history(&id).len(),
        3,
        "pending → running → completed history replays"
    );
}
