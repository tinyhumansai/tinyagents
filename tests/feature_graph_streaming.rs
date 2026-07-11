//! Feature tests for the graph event stream and run-status snapshot.
//!
//! Exercises the live [`GraphEvent`] surface projected through
//! [`StreamCollector`] (node order, state writes, selected routes, checkpoint
//! boundaries, custom writes) and the compact [`GraphRunStatus`] snapshot on a
//! completed run. Complements `tests/e2e_graph_support_contracts.rs`, which
//! smoke-tests the testkit helpers, by asserting the concrete event lifecycle
//! and ordering.

use std::sync::Arc;

use tinyagents::harness::ids::ExecutionStatus;
use tinyagents::{
    END, GraphBuilder, GraphEvent, InMemoryCheckpointer, NodeContext, NodeResult, StreamCollector,
    run_recorded,
};

/// A linear two-node graph plus a conditional edge, so the stream carries both
/// static and conditional `RouteSelected` events.
fn pipeline() -> tinyagents::CompiledGraph<i64, i64> {
    GraphBuilder::<i64, i64>::overwrite()
        .add_node("seed", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("double", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s * 2))
        })
        .add_node("finish", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 100))
        })
        .set_entry("seed")
        .add_edge("seed", "double")
        .add_conditional_edges(
            "double",
            |n: &i64| if *n > 0 { "go" } else { "stop" },
            [("go", "finish"), ("stop", END)],
        )
        .set_finish("finish")
        .compile()
        .expect("pipeline compiles")
}

#[tokio::test]
async fn event_stream_reports_the_full_run_lifecycle() {
    let run = run_recorded(&pipeline(), None, 0)
        .await
        .expect("run succeeds");

    let kinds = run.events.iter().map(|e| e.kind()).collect::<Vec<_>>();

    // The run brackets every step with start/complete, and every executed node
    // reports started/completed with a state write.
    assert_eq!(kinds.first(), Some(&"run.started"));
    assert_eq!(kinds.last(), Some(&"run.completed"));
    for expected in [
        "step.started",
        "node.started",
        "node.completed",
        "state.updated",
        "route.selected",
        "step.completed",
    ] {
        assert!(
            kinds.contains(&expected),
            "expected a `{expected}` event, saw {kinds:?}"
        );
    }
}

#[tokio::test]
async fn stream_collector_projects_node_order_updates_and_routes() {
    let run = run_recorded(&pipeline(), None, 0)
        .await
        .expect("run succeeds");
    let collector = StreamCollector::new(run.events.clone());

    let order: Vec<String> = collector
        .node_order()
        .iter()
        .map(|n| n.as_str().to_string())
        .collect();
    assert_eq!(order, vec!["seed", "double", "finish"]);

    let updates: Vec<String> = collector
        .updates()
        .iter()
        .map(|n| n.as_str().to_string())
        .collect();
    assert_eq!(updates, vec!["seed", "double", "finish"]);

    // Both the static edge (seed -> double) and the conditional edge
    // (double -> finish) surface as selected routes; the final -> END edge does
    // not emit a route.
    let routes: Vec<(String, String)> = collector
        .routes()
        .iter()
        .map(|(f, t)| (f.as_str().to_string(), t.as_str().to_string()))
        .collect();
    assert!(routes.contains(&("seed".to_string(), "double".to_string())));
    assert!(routes.contains(&("double".to_string(), "finish".to_string())));

    // 0 -> +1 -> *2 -> +100.
    assert_eq!(run.execution.state, 102);
}

#[tokio::test]
async fn checkpoint_saved_events_track_every_superstep_boundary() {
    let graph = pipeline().with_checkpointer(Arc::new(InMemoryCheckpointer::<i64>::new()));

    let run = run_recorded(&graph, Some("stream-thread"), 0)
        .await
        .expect("threaded run succeeds");

    let collector = StreamCollector::new(run.events.clone());
    // One checkpoint per executed superstep (seed, double, finish).
    assert_eq!(collector.checkpoint_count(), 3);
    assert_eq!(run.execution.steps, 3);
    // The recorded checkpoint history mirrors the boundary count.
    assert_eq!(run.history.len(), 3);
}

#[tokio::test]
async fn run_status_snapshot_reflects_a_completed_run() {
    let run = run_recorded(&pipeline(), None, 0)
        .await
        .expect("run succeeds");
    let status = &run.execution.status;

    assert_eq!(status.status, ExecutionStatus::Completed);
    assert!(status.is_terminal());
    assert_eq!(status.current_step, run.execution.steps);
    assert!(
        status.active_nodes.is_empty(),
        "a completed run has no active nodes, saw {:?}",
        status.active_nodes
    );
    assert!(status.pending_interrupts.is_empty());
    assert!(
        status.ended_at.is_some(),
        "a terminal run records an end time"
    );
    assert!(status.error.is_none());
}

#[tokio::test]
async fn a_failed_node_emits_run_failed_and_a_failed_status() {
    let graph = GraphBuilder::<i64, i64>::overwrite()
        .add_node("boom", tinyagents::failing_node("kaboom"))
        .set_entry("boom")
        .set_finish("boom")
        .compile()
        .expect("graph compiles");

    // Wire the recorder manually so we can inspect events even though the run
    // returns an error.
    let recorder = tinyagents::GraphEventRecorder::new();
    let graph = graph.with_event_sink(recorder.sink());
    let err = graph.run(0).await.expect_err("failing node aborts the run");
    assert!(err.to_string().contains("kaboom"), "got: {err}");

    let kinds = recorder.kinds();
    assert!(kinds.contains(&"node.failed".to_string()), "saw {kinds:?}");
    assert!(kinds.contains(&"run.failed".to_string()), "saw {kinds:?}");
    // No run.completed event is emitted for a failed run.
    assert!(
        !recorder
            .events()
            .iter()
            .any(|e| matches!(e, GraphEvent::RunCompleted { .. }))
    );
}
