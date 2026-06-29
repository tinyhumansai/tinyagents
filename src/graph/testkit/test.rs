//! Unit tests for the graph testkit: every node double and every assertion.

use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::graph::builder::{END, GraphBuilder};
use crate::graph::checkpoint::InMemoryCheckpointer;
use crate::graph::command::{Command, NodeResult};
use crate::graph::reducer::ClosureStateReducer;
use crate::harness::ids::NodeId;
use crate::harness::usage::UsageTotals;

fn ids(values: &[&str]) -> Vec<NodeId> {
    values.iter().map(|v| NodeId::from(*v)).collect()
}

// ── noop_node ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn noop_node_passes_through_without_updating_state() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", noop_node())
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap();

    let run = run_recorded(&graph, None, 7).await.unwrap();
    assert_eq!(run.execution.state, 7);
    assert_graph(&run).visited(["a"]).completed();
}

// ── scripted_update_node ─────────────────────────────────────────────────────

#[tokio::test]
async fn scripted_update_node_emits_queued_updates_in_order() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", scripted_update_node([10]))
        .add_node("b", scripted_update_node([20]))
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap();

    let run = run_recorded(&graph, None, 0).await.unwrap();
    assert_eq!(run.execution.state, 20);
    assert_graph(&run).visited(["a", "b"]).completed();
}

#[tokio::test]
async fn scripted_update_node_saturates_last_update_in_a_loop() {
    // The "loop" node re-emits its last scripted update on the third visit.
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("loop", scripted_update_node([1, 2]))
        .add_node(
            "gate",
            scripted_route_node(vec![vec!["loop"], vec!["loop"], vec![END]]),
        )
        .set_entry("loop")
        .add_edge("loop", "gate")
        .compile()
        .unwrap();

    let run = run_recorded(&graph, None, 0).await.unwrap();
    // loop visited 3 times: updates 1, 2, then saturated 2.
    assert_eq!(run.execution.state, 2);
    assert_graph(&run).visited(["loop", "gate", "loop", "gate", "loop", "gate"]);
}

#[tokio::test]
async fn scripted_update_node_without_updates_fails() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", scripted_update_node(Vec::<i32>::new()))
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap();

    assert!(graph.run(0).await.is_err());
}

// ── scripted_route_node ──────────────────────────────────────────────────────

#[tokio::test]
async fn scripted_route_node_routes_to_queued_targets() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", scripted_route_node(vec![vec!["b"], vec![END]]))
        .add_node("b", noop_node())
        .set_entry("a")
        .add_edge("b", "a")
        .compile()
        .unwrap();

    let run = run_recorded(&graph, None, 0).await.unwrap();
    assert_graph(&run)
        .visited(["a", "b", "a"])
        .routed("a", "b")
        .routed("b", "a");
}

// ── fanout_node ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn fanout_node_schedules_target_once_per_arg() {
    // Sum reducer so each worker's send_arg folds into the final state.
    let graph = GraphBuilder::<i64, i64>::new()
        .set_reducer(ClosureStateReducer::new(|s: i64, u: i64| Ok(s + u)))
        .add_node("fan", fanout_node("work", [json!(1), json!(2), json!(3)]))
        .add_node("work", |_s, ctx| async move {
            let n = ctx.send_arg.and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(NodeResult::Update(n))
        })
        .set_entry("fan")
        .set_finish("work")
        .compile()
        .unwrap();

    let run = run_recorded(&graph, None, 0).await.unwrap();
    // fan + three work activations.
    let work_visits = run
        .execution
        .visited
        .iter()
        .filter(|n| n.as_str() == "work")
        .count();
    assert_eq!(work_visits, 3);
    assert_eq!(run.execution.state, 6);
}

// ── failing_node ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn failing_node_aborts_the_run() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("boom", failing_node("kaboom"))
        .set_entry("boom")
        .set_finish("boom")
        .compile()
        .unwrap();

    let err = graph.run(0).await.unwrap_err();
    assert!(err.to_string().contains("kaboom"), "got: {err}");
}

// ── RetryCountingNode ────────────────────────────────────────────────────────

#[tokio::test]
async fn retry_counting_node_counts_and_eventually_succeeds() {
    let counter = RetryCountingNode::new(0);
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", counter.handler(99))
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap();

    let run = run_recorded(&graph, None, 0).await.unwrap();
    assert_eq!(run.execution.state, 99);
    assert_eq!(counter.attempts(), 1);
}

#[tokio::test]
async fn retry_counting_node_fails_first_attempts() {
    let counter = RetryCountingNode::new(1);
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", counter.handler(99))
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap();

    // The single failing attempt aborts the run.
    assert!(graph.run(0).await.is_err());
    assert_eq!(counter.attempts(), 1);
}

// ── interrupting_node ────────────────────────────────────────────────────────

#[tokio::test]
async fn interrupting_node_pauses_then_resumes() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("hitl", interrupting_node(json!("approve?"), 42))
        .set_entry("hitl")
        .set_finish("hitl")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let paused = graph.run_with_thread("t1", 0).await.unwrap();
    assert_graph(&GraphRun::new(paused)).interrupted();

    let resumed = graph
        .resume("t1", Command::resume(json!(true)))
        .await
        .unwrap();
    let final_state = resumed.state;
    assert_graph(&GraphRun::new(resumed)).completed();
    assert_eq!(final_state, 42);
}

// ── subgraph_test_node ───────────────────────────────────────────────────────

#[tokio::test]
async fn subgraph_test_node_embeds_a_child_graph() {
    let child = GraphBuilder::<i32, i32>::overwrite()
        .add_node(
            "double",
            |s, _| async move { Ok(NodeResult::Update(s * 2)) },
        )
        .set_entry("double")
        .set_finish("double")
        .compile()
        .unwrap();

    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("sub", subgraph_test_node(child))
        .set_entry("sub")
        .set_finish("sub")
        .compile()
        .unwrap();

    let run = run_recorded(&parent, None, 5).await.unwrap();
    assert_eq!(run.execution.state, 10);
    assert_graph(&run).visited(["sub"]).completed();
    assert_eq!(run.execution.child_runs.len(), 1);
    assert_eq!(run.execution.child_runs[0].node.as_str(), "sub");
}

// ── subagent_fake_node ───────────────────────────────────────────────────────

#[tokio::test]
async fn subagent_fake_node_records_a_child_run() {
    let usage = UsageTotals::default();
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("ask", subagent_fake_node("helper", 1, usage))
        .set_entry("ask")
        .set_finish("ask")
        .compile()
        .unwrap();

    let run = run_recorded(&graph, None, 0).await.unwrap();
    assert_eq!(run.execution.child_runs.len(), 1);
    let child = &run.execution.child_runs[0];
    assert_eq!(child.node.as_str(), "ask");
    assert_eq!(child.graph_id.as_str(), "agent:helper");
    // The child preserves the parent run as the recursion-tree root.
    assert_eq!(child.root_run_id, run.execution.run_id);
}

// ── GraphEventRecorder + StreamCollector ─────────────────────────────────────

#[tokio::test]
async fn event_recorder_captures_events_and_collector_projects_them() {
    let recorder = GraphEventRecorder::new();
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", scripted_update_node([1]))
        .add_node("b", noop_node())
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap()
        .with_event_sink(recorder.sink());

    graph.run(0).await.unwrap();

    let kinds = recorder.kinds();
    assert!(kinds.contains(&"run.started".to_string()));
    assert!(kinds.contains(&"run.completed".to_string()));

    let collector = recorder.collector();
    assert_eq!(collector.node_order(), ids(&["a", "b"]));
    assert_eq!(collector.updates(), ids(&["a"]));
    assert_eq!(
        collector.routes(),
        vec![(NodeId::from("a"), NodeId::from("b"))]
    );
    assert!(collector.interrupts().is_empty());
}

// ── assert_graph: the exact doc example, plus checkpoint assertions ──────────

#[tokio::test]
async fn assert_graph_doc_example_with_checkpoints() {
    let cp = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("agent", scripted_route_node(vec![vec!["tools"], vec![END]]))
        .add_node("tools", noop_node())
        .set_entry("agent")
        .add_edge("tools", "agent")
        .compile()
        .unwrap()
        .with_checkpointer(cp);

    let run = run_recorded(&graph, Some("t1"), 0).await.unwrap();

    assert_graph(&run)
        .visited(["agent", "tools", "agent"])
        .routed("agent", "tools")
        .checkpoint_count(3)
        .completed();

    // state_history + checkpoint assertions read the durable snapshots.
    assert_graph(&run)
        .state_history(|history| assert_eq!(history.len(), 3))
        .checkpoint(|latest| assert_eq!(latest.values, 0));
}

#[tokio::test]
async fn routed_falls_back_to_visited_adjacency_without_events() {
    // A GraphRun assembled by hand (no events) routes off visited adjacency.
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", scripted_update_node([1]))
        .add_node("b", noop_node())
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap();
    let execution = graph.run(0).await.unwrap();
    let run = GraphRun::new(execution);
    assert_graph(&run).routed("a", "b");
}
