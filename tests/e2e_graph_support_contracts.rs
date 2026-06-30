//! End-to-end contracts for graph support surfaces: file-backed checkpoints and
//! the public graph testkit helpers.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

use tinyagents::graph::testkit::{
    GraphEventRecorder, RetryCountingNode, assert_graph, fanout_node, interrupting_node, noop_node,
    run_recorded, scripted_route_node, scripted_update_node, subagent_fake_node,
    subgraph_test_node,
};
use tinyagents::harness::ids::ExecutionStatus;
use tinyagents::harness::usage::UsageTotals;
use tinyagents::{
    CheckpointConfig, CheckpointSource, Checkpointer, Command, DurabilityMode, FileCheckpointer,
    GraphBuilder, InMemoryCheckpointer, NodeContext, NodeResult, TinyAgentsError,
};

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tinyagents-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[tokio::test]
async fn file_checkpointer_persists_lists_copies_prunes_and_deletes_threads() {
    let dir = temp_path("file-checkpointer");
    let checkpointer = std::sync::Arc::new(FileCheckpointer::<i32>::new(&dir));
    assert_eq!(checkpointer.base_dir(), dir.as_path());

    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .expect("graph compiles")
        .with_checkpointer(checkpointer.clone());

    let run = graph
        .run_with_thread("thread/with space", 0)
        .await
        .expect("threaded file-backed run succeeds");
    assert_eq!(run.state, 2);
    assert_eq!(run.status.status, ExecutionStatus::Completed);

    let list = checkpointer.list("thread/with space").await.expect("list");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].source, CheckpointSource::Loop);
    assert_eq!(
        CheckpointSource::parse("fork"),
        Some(CheckpointSource::Fork)
    );
    assert_eq!(CheckpointSource::Update.to_string(), "update");

    let latest = checkpointer
        .get("thread/with space", None)
        .await
        .expect("get latest")
        .expect("latest checkpoint");
    assert_eq!(latest.state, 2);
    assert_eq!(
        latest.parent_checkpoint_id.as_deref(),
        Some(list[0].checkpoint_id.as_str())
    );

    let tuple = checkpointer
        .get_tuple(CheckpointConfig::latest("thread/with space"))
        .await
        .expect("tuple")
        .expect("tuple present");
    assert_eq!(tuple.config.thread_id, "thread/with space");
    assert!(tuple.parent_config.is_some());

    checkpointer
        .copy_thread("thread/with space", "copy-target")
        .await
        .expect("copy thread");
    let copied = checkpointer.list("copy-target").await.expect("copied list");
    assert_eq!(copied.len(), 2);
    assert!(copied.iter().all(|m| m.thread_id == "copy-target"));

    let threads = checkpointer.list_threads().await.expect("threads");
    assert!(threads.contains(&"thread/with space".to_string()));
    assert!(threads.contains(&"copy-target".to_string()));

    let removed = checkpointer
        .delete_checkpoints("copy-target", &[copied[0].checkpoint_id.clone()])
        .await
        .expect("delete one");
    assert_eq!(removed, 1);
    assert_eq!(checkpointer.list("copy-target").await.unwrap().len(), 1);

    let pruned = checkpointer
        .prune("thread/with space", 1)
        .await
        .expect("prune keeps latest and ancestors");
    assert_eq!(pruned, 0);

    checkpointer
        .delete_thread("copy-target")
        .await
        .expect("delete thread");
    assert!(checkpointer.list("copy-target").await.unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn graph_testkit_helpers_drive_recorded_graphs_and_assertions() {
    let checkpointer = std::sync::Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("seed", scripted_update_node([1]))
        .add_node("route", scripted_route_node([["fanout"]]))
        .add_node("fanout", fanout_node("worker", [json!(2), json!(3)]))
        .add_node("worker", |s: i32, ctx: NodeContext| async move {
            let arg = ctx.send_arg.and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            Ok(NodeResult::Update(s + arg))
        })
        .add_node("agent", subagent_fake_node("fake", 10, UsageTotals::new()))
        .add_node("noop", noop_node())
        .set_entry("seed")
        .add_edge("seed", "route")
        .mark_command_routing("route")
        .mark_command_routing("fanout")
        .add_waiting_edge("worker", "agent")
        .add_edge("agent", "noop")
        .set_finish("noop")
        .compile()
        .expect("graph compiles")
        .with_checkpointer(checkpointer);

    let recorded = run_recorded(&graph, Some("graph-testkit"), 0)
        .await
        .expect("recorded run");
    assert_eq!(recorded.execution.state, 10);
    assert_eq!(recorded.execution.child_runs.len(), 1);
    assert_graph(&recorded)
        .visited([
            "seed", "route", "fanout", "worker", "worker", "agent", "noop",
        ])
        .routed("seed", "route")
        .checkpoint_count(6)
        .state_history(|history| assert_eq!(history.len(), 6))
        .checkpoint(|latest| {
            assert_eq!(latest.values, 10);
            assert!(latest.next_nodes.is_empty());
        })
        .completed();

    let collector = recorded.collector();
    assert!(collector.events().len() >= 5);
    assert!(
        collector
            .node_order()
            .iter()
            .any(|n| n.as_str() == "worker")
    );
    assert!(collector.updates().iter().any(|n| n.as_str() == "agent"));
    assert!(
        collector
            .routes()
            .iter()
            .any(|(from, to)| { from.as_str() == "seed" && to.as_str() == "route" })
    );
    assert_eq!(collector.checkpoint_count(), 6);
    assert!(collector.interrupts().is_empty());
    assert!(collector.custom().is_empty());

    let manual_recorder = GraphEventRecorder::new();
    let manual_graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("one", scripted_update_node([1]))
        .set_entry("one")
        .set_finish("one")
        .compile()
        .unwrap()
        .with_event_sink(manual_recorder.sink());
    manual_graph.run(0).await.unwrap();
    assert!(
        manual_recorder
            .kinds()
            .contains(&"node.completed".to_string())
    );
    assert!(!manual_recorder.collector().events().is_empty());
}

#[tokio::test]
async fn graph_testkit_interrupt_retry_subgraph_and_failure_paths_are_public_contracts() {
    let retry = RetryCountingNode::new(1);
    let retry_graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("retry", retry.handler(7))
        .set_entry("retry")
        .set_finish("retry")
        .compile()
        .unwrap();
    let err = retry_graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Graph(_)));
    assert_eq!(retry.attempts(), 1);

    let retry_success = GraphBuilder::<i32, i32>::overwrite()
        .add_node("retry", retry.handler(7))
        .set_entry("retry")
        .set_finish("retry")
        .compile()
        .unwrap();
    assert_eq!(retry_success.run(0).await.unwrap().state, 7);
    assert_eq!(retry.attempts(), 2);

    let child = GraphBuilder::<i32, i32>::overwrite()
        .add_node("add", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 5))
        })
        .set_entry("add")
        .set_finish("add")
        .compile()
        .unwrap();
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", subgraph_test_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap();
    assert_eq!(parent.run(1).await.unwrap().state, 6);

    let interrupting = GraphBuilder::<i32, i32>::overwrite()
        .add_node("hitl", interrupting_node(json!({ "ask": true }), 11))
        .set_entry("hitl")
        .set_finish("hitl")
        .compile()
        .unwrap()
        .with_checkpointer(std::sync::Arc::new(InMemoryCheckpointer::<i32>::new()))
        .with_durability(DurabilityMode::Exit);
    let paused = interrupting.run_with_thread("hitl", 0).await.unwrap();
    assert_graph(&tinyagents::graph::testkit::GraphRun::new(paused)).interrupted();
    let resumed = interrupting
        .resume("hitl", Command::resume(json!(true)))
        .await
        .unwrap();
    assert_eq!(resumed.state, 11);
}
