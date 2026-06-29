//! Unit tests for the graph durable observability layer: journaling sinks,
//! offset-addressable replay, status-store lifecycle, store-backed journals, and
//! namespaced subgraph observations.

use super::*;
use crate::error::TinyAgentsError;
use crate::graph::builder::{GraphBuilder, NodeContext};
use crate::graph::command::NodeResult;
use crate::graph::compiled::CompiledGraph;
use crate::graph::stream::{CollectingSink, GraphEvent, GraphEventSink};
use crate::harness::ids::{ExecutionStatus, GraphId, RunId};
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
