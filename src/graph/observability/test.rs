//! Unit tests for the graph durable observability layer: journaling sinks,
//! offset-addressable replay, status-store lifecycle, store-backed journals, and
//! namespaced subgraph observations.

use super::*;
use crate::error::TinyAgentsError;
use crate::graph::builder::{GraphBuilder, NodeContext};
use crate::graph::command::NodeResult;
use crate::graph::compiled::CompiledGraph;
use crate::graph::stream::{CollectingSink, GraphEvent, GraphEventSink};
use crate::harness::ids::{ExecutionStatus, GraphId, NodeId, RunId};
use crate::harness::store::InMemoryAppendStore;
use std::sync::Arc;

/// A two-node line graph over `i32` with overwrite semantics: `a -> b`.
fn line_graph() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap()
}

/// A single-node graph whose node always errors.
fn failing_graph() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("boom", |_s, _c: NodeContext| async move {
            Err::<NodeResult<i32>, _>(TinyAgentsError::Validation("boom".to_string()))
        })
        .set_entry("boom")
        .set_finish("boom")
        .compile()
        .unwrap()
}

fn graph_obs(run: &str, offset: u64, step: usize, event: GraphEvent) -> GraphObservation {
    GraphObservation {
        event_id: crate::harness::ids::EventId::new(format!("g-evt-{offset}")),
        run_id: RunId::new(run),
        root_run_id: RunId::new(run),
        parent_run_id: None,
        thread_id: None,
        graph_id: GraphId::new("graph-latency"),
        checkpoint_id: None,
        namespace: Vec::new(),
        step,
        offset,
        ts_ms: 1_000 + offset,
        event,
    }
}

#[test]
fn graph_latency_metrics_include_run_steps_and_nodes() {
    let run_id = RunId::new("run-latency");
    let node_a = NodeId::new("a");
    let node_b = NodeId::new("b");
    let observations = vec![
        graph_obs(
            run_id.as_str(),
            0,
            0,
            GraphEvent::RunStarted {
                run_id: run_id.clone(),
            },
        ),
        graph_obs(
            run_id.as_str(),
            10,
            1,
            GraphEvent::StepStarted {
                step: 1,
                active: vec![node_a.clone(), node_b.clone()],
            },
        ),
        graph_obs(
            run_id.as_str(),
            20,
            1,
            GraphEvent::NodeStarted {
                node: node_a.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run_id.as_str(),
            45,
            1,
            GraphEvent::NodeCompleted {
                node: node_a.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run_id.as_str(),
            50,
            1,
            GraphEvent::NodeStarted {
                node: node_b.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run_id.as_str(),
            80,
            1,
            GraphEvent::NodeFailed {
                node: node_b.clone(),
                step: 1,
                error: "boom".to_string(),
            },
        ),
        graph_obs(
            run_id.as_str(),
            90,
            1,
            GraphEvent::StepCompleted { step: 1 },
        ),
        graph_obs(
            run_id.as_str(),
            100,
            1,
            GraphEvent::RunFailed {
                run_id: run_id.clone(),
                error: "boom".to_string(),
            },
        ),
    ];

    let metrics = GraphLatencyMetrics::from_observations(&observations);
    assert_eq!(metrics.run_elapsed_ms, Some(100));
    assert_eq!(metrics.steps.len(), 1);
    assert_eq!(metrics.steps[0].step, 1);
    assert_eq!(metrics.steps[0].elapsed_ms, 80);
    assert_eq!(metrics.total_step_ms, 80);
    assert_eq!(metrics.average_step_ms(), Some(80));

    assert_eq!(metrics.nodes.len(), 2);
    assert_eq!(metrics.nodes[0].node, node_a);
    assert_eq!(metrics.nodes[0].elapsed_ms, 25);
    assert!(!metrics.nodes[0].failed);
    assert_eq!(metrics.nodes[1].node, node_b);
    assert_eq!(metrics.nodes[1].elapsed_ms, 30);
    assert!(metrics.nodes[1].failed);
    assert_eq!(metrics.total_node_ms, 55);
    assert_eq!(metrics.max_node_ms, 30);
    assert_eq!(metrics.average_node_ms(), Some(27));
}

#[test]
fn graph_health_summary_counts_node_outcomes() {
    let run = "run-health";
    let a = NodeId::new("a");
    let b = NodeId::new("b");
    let observations = vec![
        graph_obs(
            run,
            0,
            0,
            GraphEvent::RunStarted {
                run_id: RunId::new(run),
            },
        ),
        graph_obs(
            run,
            1,
            1,
            GraphEvent::NodeStarted {
                node: a.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run,
            2,
            1,
            GraphEvent::NodeCompleted {
                node: a.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run,
            3,
            1,
            GraphEvent::NodeStarted {
                node: b.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run,
            4,
            1,
            GraphEvent::NodeFailed {
                node: b.clone(),
                step: 1,
                error: "boom".to_string(),
            },
        ),
        graph_obs(
            run,
            5,
            1,
            GraphEvent::RunFailed {
                run_id: RunId::new(run),
                error: "boom".to_string(),
            },
        ),
    ];

    let health = GraphHealthSummary::from_observations(&observations);
    assert_eq!(health.total_started, 2);
    assert_eq!(health.total_completed, 1);
    assert_eq!(health.total_failed, 1);
    assert!(health.run_failed);
    assert!(!health.is_healthy());
    assert_eq!(health.failure_rate(), 0.5);

    // Sorted by node id, with per-node health surfaced.
    assert_eq!(health.nodes.len(), 2);
    assert_eq!(health.nodes[0].node, a);
    assert!(health.nodes[0].is_healthy());
    assert_eq!(health.nodes[1].node, b);
    assert!(!health.nodes[1].is_healthy());
    assert_eq!(health.nodes[1].failure_rate(), 1.0);

    let unhealthy: Vec<_> = health.unhealthy_nodes().map(|n| n.node.clone()).collect();
    assert_eq!(unhealthy, vec![b]);
}

#[test]
fn healthy_run_reports_no_failures() {
    let run = "run-ok";
    let a = NodeId::new("a");
    let observations = vec![
        graph_obs(
            run,
            0,
            1,
            GraphEvent::NodeStarted {
                node: a.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run,
            1,
            1,
            GraphEvent::NodeCompleted {
                node: a.clone(),
                step: 1,
            },
        ),
        graph_obs(
            run,
            2,
            1,
            GraphEvent::RunCompleted {
                run_id: RunId::new(run),
                steps: 1,
            },
        ),
    ];

    let health = GraphHealthSummary::from_observations(&observations);
    assert!(health.is_healthy());
    assert_eq!(health.failure_rate(), 0.0);
    assert_eq!(health.unhealthy_nodes().count(), 0);
}

#[tokio::test]
async fn run_with_journal_records_replayable_observations() {
    let journal = Arc::new(InMemoryGraphEventJournal::new());
    let graph = line_graph().with_event_journal(journal.clone());

    let run = graph.run(0).await.unwrap();
    let run_id = run.status.run_id.as_str().to_string();

    let all = journal.read_from(&run_id, 0).await.unwrap();
    assert!(
        all.len() >= 3,
        "expected several observations, got {}",
        all.len()
    );

    // Offsets are dense and monotonic from 0, and lineage/graph are stamped.
    for (i, obs) in all.iter().enumerate() {
        assert_eq!(obs.offset, i as u64);
        assert_eq!(obs.run_id.as_str(), run_id);
        assert_eq!(obs.root_run_id.as_str(), run_id);
        assert_eq!(obs.graph_id, *graph.graph_id());
    }

    // The run lifecycle bookends the stream.
    assert!(matches!(
        all.first().unwrap().event,
        GraphEvent::RunStarted { .. }
    ));
    assert!(
        all.iter()
            .any(|o| matches!(o.event, GraphEvent::RunCompleted { steps: 2, .. }))
    );

    // Replay from a mid-stream offset returns exactly the tail.
    let tail = journal.read_from(&run_id, 2).await.unwrap();
    assert_eq!(tail.len(), all.len() - 2);
    assert_eq!(tail.first().unwrap().offset, 2);

    // Reading an unknown run is empty, not an error.
    assert!(journal.read_from("nope", 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn status_store_reflects_run_lifecycle() {
    let store = Arc::new(InMemoryGraphStatusStore::new());
    let graph = line_graph().with_status_store(store.clone());

    let run = graph.run_with_thread("t-1", 0).await.unwrap();
    let run_id = run.status.run_id.as_str().to_string();

    let status = store.get_status(&run_id).await.unwrap().unwrap();
    assert_eq!(status.status, ExecutionStatus::Completed);
    assert_eq!(status.current_step, 2);
    assert!(status.ended_at.is_some());
    assert!(status.error.is_none());

    // Indexed by thread for supervisor queries.
    let by_thread = store.list_by_thread("t-1").await.unwrap();
    assert_eq!(by_thread.len(), 1);
    assert_eq!(by_thread[0].run_id.as_str(), run_id);
}

#[tokio::test]
async fn status_store_records_failed_runs() {
    let store = Arc::new(InMemoryGraphStatusStore::new());
    let graph = failing_graph().with_status_store(store.clone());

    let err = graph.run_with_thread("t-fail", 0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Validation(_)));

    let failed = store.list_by_thread("t-fail").await.unwrap();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].status, ExecutionStatus::Failed);
    assert!(failed[0].error.as_deref().unwrap().contains("boom"));
    assert!(failed[0].ended_at.is_some());
}

#[tokio::test]
async fn failed_run_journals_run_failed_event() {
    let journal = Arc::new(InMemoryGraphEventJournal::new());
    let store = Arc::new(InMemoryGraphStatusStore::new());
    let graph = failing_graph()
        .with_event_journal(journal.clone())
        .with_status_store(store.clone());

    let _ = graph.run_with_thread("t-boom", 0).await.unwrap_err();

    // Recover the executor-assigned run id via the thread-indexed status store.
    let status = store.list_by_thread("t-boom").await.unwrap();
    let run_id = status[0].run_id.as_str().to_string();

    let obs = journal.read_from(&run_id, 0).await.unwrap();
    assert!(matches!(
        obs.first().unwrap().event,
        GraphEvent::RunStarted { .. }
    ));
    assert!(obs.iter().any(
        |o| matches!(&o.event, GraphEvent::RunFailed { error, .. } if error.contains("boom"))
    ));
}

#[tokio::test]
async fn namespaced_subgraph_observations_carry_child_namespace() {
    let journal = Arc::new(InMemoryGraphEventJournal::new());
    let child_ns = vec!["parent".to_string(), "child".to_string()];
    let graph = line_graph()
        .with_namespace(child_ns.clone())
        .with_event_journal(journal.clone());

    let run = graph.run(0).await.unwrap();
    let obs = journal
        .read_from(run.status.run_id.as_str(), 0)
        .await
        .unwrap();

    assert!(!obs.is_empty());
    assert!(
        obs.iter().all(|o| o.namespace == child_ns),
        "every observation should carry the child namespace"
    );
}

#[tokio::test]
async fn journal_sink_used_directly_forwards_to_inner() {
    // Drive the public JournalGraphSink as the live event sink: it both journals
    // observations under its configured run id and forwards to an inner sink.
    let journal = Arc::new(InMemoryGraphEventJournal::new());
    let collector = Arc::new(CollectingSink::new());
    let sink = JournalGraphSink::new(journal.clone(), RunId::new("fixed-run"), GraphId::new("g"))
        .with_inner(collector.clone());

    sink.emit(GraphEvent::StepStarted {
        step: 1,
        active: Vec::new(),
    });
    sink.emit(GraphEvent::RouteSelected {
        node: "a".into(),
        target: "b".into(),
    });
    sink.emit(GraphEvent::StepCompleted { step: 1 });

    // Forwarded to the live sink.
    assert_eq!(collector.len(), 3);

    // Persistence is asynchronous; block until the durable log catches up.
    sink.flush();

    // Journaled with dense offsets; the step is carried forward onto the
    // step-less RouteSelected event.
    let obs = journal.read_from("fixed-run", 0).await.unwrap();
    assert_eq!(obs.len(), 3);
    assert_eq!(obs[0].step, 1);
    assert_eq!(obs[1].step, 1); // RouteSelected inherits the last seen step
    assert_eq!(obs[2].offset, 2);
}

#[tokio::test]
async fn store_backed_journal_round_trips() {
    let store = StoreGraphEventJournal::new(InMemoryAppendStore::new());
    let obs = GraphObservation {
        event_id: crate::harness::ids::EventId::new("e0"),
        run_id: RunId::new("r0"),
        root_run_id: RunId::new("r0"),
        parent_run_id: None,
        thread_id: None,
        graph_id: GraphId::new("g0"),
        checkpoint_id: None,
        namespace: vec!["ns".to_string()],
        step: 1,
        offset: 0,
        ts_ms: 42,
        event: GraphEvent::RunStarted {
            run_id: RunId::new("r0"),
        },
    };
    let off = store.append(obs.clone()).await.unwrap();
    assert_eq!(off, 0);

    let read = store.read_from("r0", 0).await.unwrap();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0], obs);
}

#[test]
fn graph_event_kind_and_step_are_stable() {
    assert_eq!(
        GraphEvent::RunStarted {
            run_id: RunId::new("r")
        }
        .kind(),
        "run.started"
    );
    assert_eq!(
        GraphEvent::SubgraphStarted {
            node: "n".into(),
            namespace: vec![]
        }
        .kind(),
        "subgraph.started"
    );
    let forked = GraphEvent::ContextForked {
        node: "n".into(),
        fork: 0,
        step: 3,
    };
    assert_eq!(forked.kind(), "context.forked");
    assert_eq!(forked.step(), Some(3));
    assert_eq!(GraphEvent::RecursionDepthChanged { depth: 2 }.step(), None);
}

// ── InMemoryGraphStatusStore indexing and retention ───────────────────────────

/// Builds a status for `run_id` on `thread_id` in the given lifecycle state.
fn status_on_thread(run_id: &str, thread_id: &str, status: ExecutionStatus) -> GraphRunStatus {
    let mut s = GraphRunStatus::new(RunId::new(run_id), GraphId::new("g"), status);
    s.thread_id = Some(crate::harness::ids::ThreadId::new(thread_id));
    s
}

#[tokio::test]
async fn status_store_thread_index_tracks_overwrites() {
    let store = InMemoryGraphStatusStore::new();
    store
        .put_status(status_on_thread("r-1", "t-a", ExecutionStatus::Running))
        .await
        .unwrap();
    store
        .put_status(status_on_thread("r-2", "t-a", ExecutionStatus::Running))
        .await
        .unwrap();
    store
        .put_status(status_on_thread("r-3", "t-b", ExecutionStatus::Running))
        .await
        .unwrap();

    assert_eq!(store.list_by_thread("t-a").await.unwrap().len(), 2);
    assert_eq!(store.list_by_thread("t-b").await.unwrap().len(), 1);
    assert!(store.list_by_thread("t-missing").await.unwrap().is_empty());

    // Overwriting a run on the same thread does not duplicate the index entry.
    store
        .put_status(status_on_thread("r-1", "t-a", ExecutionStatus::Completed))
        .await
        .unwrap();
    assert_eq!(store.list_by_thread("t-a").await.unwrap().len(), 2);

    // Re-homing a run to a different thread moves it in the index.
    store
        .put_status(status_on_thread("r-2", "t-b", ExecutionStatus::Running))
        .await
        .unwrap();
    assert_eq!(store.list_by_thread("t-a").await.unwrap().len(), 1);
    assert_eq!(store.list_by_thread("t-b").await.unwrap().len(), 2);
    assert_eq!(store.len(), 3);
}

#[tokio::test]
async fn status_store_cap_evicts_oldest_terminal_first() {
    let store = InMemoryGraphStatusStore::new().with_max_runs(2);
    store
        .put_status(status_on_thread("r-live", "t", ExecutionStatus::Running))
        .await
        .unwrap();
    store
        .put_status(status_on_thread("r-done", "t", ExecutionStatus::Completed))
        .await
        .unwrap();
    // Third run exceeds the cap: the terminal `r-done` goes first even though
    // `r-live` is older.
    store
        .put_status(status_on_thread("r-new", "t", ExecutionStatus::Running))
        .await
        .unwrap();

    assert_eq!(store.len(), 2);
    assert!(store.get_status("r-done").await.unwrap().is_none());
    assert!(store.get_status("r-live").await.unwrap().is_some());
    assert!(store.get_status("r-new").await.unwrap().is_some());

    // The thread index no longer serves the evicted run.
    let by_thread = store.list_by_thread("t").await.unwrap();
    assert_eq!(by_thread.len(), 2);
    assert!(by_thread.iter().all(|s| s.run_id.as_str() != "r-done"));
}

#[tokio::test]
async fn status_store_cap_falls_back_to_oldest_live_run() {
    let store = InMemoryGraphStatusStore::new().with_max_runs(2);
    for id in ["r-1", "r-2", "r-3"] {
        store
            .put_status(status_on_thread(id, "t", ExecutionStatus::Running))
            .await
            .unwrap();
    }
    // No terminal run to prefer, so the oldest live run is evicted.
    assert_eq!(store.len(), 2);
    assert!(store.get_status("r-1").await.unwrap().is_none());
    assert!(store.get_status("r-2").await.unwrap().is_some());
    assert!(store.get_status("r-3").await.unwrap().is_some());
    assert_eq!(store.list_by_thread("t").await.unwrap().len(), 2);
}

#[tokio::test]
async fn status_store_overwrite_never_evicts() {
    let store = InMemoryGraphStatusStore::new().with_max_runs(2);
    store
        .put_status(status_on_thread("r-1", "t", ExecutionStatus::Running))
        .await
        .unwrap();
    store
        .put_status(status_on_thread("r-2", "t", ExecutionStatus::Running))
        .await
        .unwrap();
    // Updating an existing run at capacity is not an insertion.
    store
        .put_status(status_on_thread("r-1", "t", ExecutionStatus::Completed))
        .await
        .unwrap();
    assert_eq!(store.len(), 2);
    assert!(store.get_status("r-1").await.unwrap().is_some());
    assert!(store.get_status("r-2").await.unwrap().is_some());
}
