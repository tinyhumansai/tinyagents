//! Durable executor tests: the interrupt→resume and failure→retry scenarios
//! from `test.rs` (`interrupt_then_resume_reruns_node`,
//! `exhausted_retries_leave_a_resumable_failure_checkpoint`), replayed against
//! real on-disk checkpointers instead of [`InMemoryCheckpointer`].
//!
//! Each scenario simulates a process restart between its write half and its
//! read/resume half: the checkpointer used for the first half is dropped
//! entirely, and a **fresh** checkpointer instance is opened against the same
//! on-disk location (the same directory for [`FileCheckpointer`], the same
//! database file for [`SqliteCheckpointer`]) for the second half. This
//! exercises the "close it, come back, resume from disk" path rather than
//! merely calling `.resume()`/`.retry()` on an already-warm in-process
//! checkpointer.

use super::*;
use crate::builder::{GraphBuilder, NodeContext};
#[cfg(feature = "sqlite")]
use crate::checkpoint::SqliteCheckpointer;
use crate::checkpoint::{Checkpointer, FileCheckpointer};
use crate::command::{Command, Interrupt, NodeResult};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use tinyagents_harness::ids::ExecutionStatus;

/// Same "human approval" graph shape as `interrupt_then_resume_reruns_node`:
/// pauses on first run, applies a resume-supplied bump on the second.
fn approve_graph() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("approve", |s, ctx: NodeContext| async move {
            match ctx.resume {
                Some(value) => {
                    let bump = value.get("bump").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                    Ok(NodeResult::Update(s + bump))
                }
                None => Ok(NodeResult::Interrupt(Interrupt::new(
                    "approve",
                    json!({ "ask": "approve?" }),
                ))),
            }
        })
        .add_node("done", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("approve")
        .add_edge("approve", "done")
        .set_finish("done")
        .compile()
        .unwrap()
}

/// Same "flaky node" shape as `flaky_graph` in `test.rs`: fails the first
/// `fail_times` invocations, then succeeds with `+1`.
fn flaky_graph(fail_times: usize, attempts: Arc<AtomicUsize>) -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("flaky", move |s, _c: NodeContext| {
            let attempts = attempts.clone();
            async move {
                let n = attempts.fetch_add(1, AtomicOrdering::SeqCst);
                if n < fail_times {
                    Err(TinyAgentsError::Model(format!("transient blip {n}")))
                } else {
                    Ok(NodeResult::Update(s + 1))
                }
            }
        })
        .set_entry("flaky")
        .set_finish("flaky")
        .compile()
        .unwrap()
}

// ── FileCheckpointer ─────────────────────────────────────────────────────

#[tokio::test]
async fn file_backend_interrupt_then_restart_then_resume() {
    let dir = tempfile::tempdir().unwrap();

    // Write phase: a fresh `FileCheckpointer` opened on the temp directory
    // runs the graph to its interrupt point.
    {
        let cp: Arc<dyn Checkpointer<i32>> = Arc::new(FileCheckpointer::<i32>::new(dir.path()));
        let graph = approve_graph().with_checkpointer(cp);
        let paused = graph.run_with_thread("hitl", 10).await.unwrap();
        assert!(paused.is_interrupted());
        assert_eq!(paused.status.status, ExecutionStatus::Interrupted);
        assert_eq!(paused.interrupts.len(), 1);
        // `graph` (and the checkpointer it owns) is dropped at the end of
        // this block, simulating the process exiting.
    }

    // Read/resume phase: a brand-new `FileCheckpointer` instance, opened
    // fresh against the same directory, resumes the run from disk.
    let cp2: Arc<dyn Checkpointer<i32>> = Arc::new(FileCheckpointer::<i32>::new(dir.path()));
    let graph2 = approve_graph().with_checkpointer(cp2);
    let resumed = graph2
        .resume("hitl", Command::resume(json!({ "bump": 5 })))
        .await
        .unwrap();
    assert!(!resumed.is_interrupted());
    assert_eq!(resumed.state, 15);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
}

#[tokio::test]
async fn file_backend_failure_then_restart_then_retry() {
    let dir = tempfile::tempdir().unwrap();
    let attempts = Arc::new(AtomicUsize::new(0));

    // Write phase: the node fails once and the run aborts, leaving a
    // resumable failure checkpoint on disk.
    {
        let cp: Arc<dyn Checkpointer<i32>> = Arc::new(FileCheckpointer::<i32>::new(dir.path()));
        let graph = flaky_graph(1, attempts.clone()).with_checkpointer(cp);
        let err = graph.run_with_thread("net", 100).await.unwrap_err();
        assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        // `graph` (and the checkpointer it owns) is dropped here, simulating
        // the process exiting after the failure.
    }

    // Read/retry phase: a fresh `FileCheckpointer` instance, opened against
    // the same directory, retries the failed node from disk. The transient
    // condition has cleared by the time this attempt runs.
    let cp2: Arc<dyn Checkpointer<i32>> = Arc::new(FileCheckpointer::<i32>::new(dir.path()));
    let graph2 = flaky_graph(1, attempts.clone()).with_checkpointer(cp2);
    let resumed = graph2.retry("net").await.unwrap();
    assert_eq!(resumed.state, 101);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);
}

// ── SqliteCheckpointer (feature = "sqlite") ──────────────────────────────

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_backend_interrupt_then_restart_then_resume() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("checkpoints.db");

    // Write phase: a fresh `SqliteCheckpointer` opened on the db file runs
    // the graph to its interrupt point.
    {
        let cp: Arc<dyn Checkpointer<i32>> =
            Arc::new(SqliteCheckpointer::<i32>::open(&db_path).unwrap());
        let graph = approve_graph().with_checkpointer(cp);
        let paused = graph.run_with_thread("hitl", 10).await.unwrap();
        assert!(paused.is_interrupted());
        assert_eq!(paused.status.status, ExecutionStatus::Interrupted);
        assert_eq!(paused.interrupts.len(), 1);
        // `graph` (and the `Connection` it owns) is dropped at the end of
        // this block, simulating the process exiting.
    }

    // Read/resume phase: a brand-new `Connection`/`SqliteCheckpointer`,
    // opened fresh against the same database file, resumes from disk.
    let cp2: Arc<dyn Checkpointer<i32>> =
        Arc::new(SqliteCheckpointer::<i32>::open(&db_path).unwrap());
    let graph2 = approve_graph().with_checkpointer(cp2);
    let resumed = graph2
        .resume("hitl", Command::resume(json!({ "bump": 5 })))
        .await
        .unwrap();
    assert!(!resumed.is_interrupted());
    assert_eq!(resumed.state, 15);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_backend_failure_then_restart_then_retry() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("checkpoints.db");
    let attempts = Arc::new(AtomicUsize::new(0));

    // Write phase: the node fails once and the run aborts, leaving a
    // resumable failure checkpoint on disk.
    {
        let cp: Arc<dyn Checkpointer<i32>> =
            Arc::new(SqliteCheckpointer::<i32>::open(&db_path).unwrap());
        let graph = flaky_graph(1, attempts.clone()).with_checkpointer(cp);
        let err = graph.run_with_thread("net", 100).await.unwrap_err();
        assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        // `graph` (and the `Connection` it owns) is dropped here, simulating
        // the process exiting after the failure.
    }

    // Read/retry phase: a fresh `Connection`/`SqliteCheckpointer`, opened
    // against the same database file, retries the failed node from disk.
    let cp2: Arc<dyn Checkpointer<i32>> =
        Arc::new(SqliteCheckpointer::<i32>::open(&db_path).unwrap());
    let graph2 = flaky_graph(1, attempts.clone()).with_checkpointer(cp2);
    let resumed = graph2.retry("net").await.unwrap();
    assert_eq!(resumed.state, 101);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);
}
