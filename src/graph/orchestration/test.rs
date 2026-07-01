use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::harness::ids::TaskId;
use crate::harness::tool::{Tool, ToolCall, ToolRegistry};

fn graph_spec(id: &str) -> OrchestrationTaskSpec {
    OrchestrationTaskSpec::new(
        id,
        OrchestrationTaskKind::Graph {
            graph_id: "child".into(),
        },
    )
}

#[test]
fn in_memory_store_tracks_task_lifecycle() {
    let store = InMemoryTaskStore::new();
    let task_id = TaskId::new("task-a");

    let pending = store.insert(graph_spec(task_id.as_str())).unwrap();
    assert_eq!(pending.status, OrchestrationTaskStatus::Pending);

    let running = store.mark_running(&task_id).unwrap();
    assert_eq!(running.status, OrchestrationTaskStatus::Running);
    assert!(running.started_at.is_some());

    let completed = store
        .complete(&task_id, OrchestrationTaskResult::text("done"))
        .unwrap();
    assert_eq!(completed.status, OrchestrationTaskStatus::Completed);
    assert!(completed.ended_at.is_some());
    assert!(completed.is_terminal());
}

fn unique_log_path(tag: &str) -> std::path::PathBuf {
    // Deterministic-per-test path in the system temp dir (no clock/random ids).
    std::env::temp_dir().join(format!("tinyagents-taskstore-{tag}.jsonl"))
}

#[test]
fn jsonl_store_survives_restart_and_keeps_history() {
    let path = unique_log_path("restart");
    let _ = std::fs::remove_file(&path);
    let task_id = TaskId::new("task-a");

    {
        let store = JsonlTaskStore::open(&path).unwrap();
        store.insert(graph_spec(task_id.as_str())).unwrap();
        store.mark_running(&task_id).unwrap();
        store
            .complete(&task_id, OrchestrationTaskResult::text("done"))
            .unwrap();
        // Full lifecycle history is retained (pending → running → completed).
        let history = store.history(&task_id);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].status, OrchestrationTaskStatus::Pending);
        assert_eq!(history[2].status, OrchestrationTaskStatus::Completed);
    }

    // Re-open: state and history are reconstructed from the append log.
    let reopened = JsonlTaskStore::open(&path).unwrap();
    let record = reopened.get(&task_id).expect("task survives restart");
    assert_eq!(record.status, OrchestrationTaskStatus::Completed);
    assert_eq!(reopened.history(&task_id).len(), 3);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn filter_by_kind_and_created_window() {
    let store = InMemoryTaskStore::new();
    store.insert(graph_spec("g1")).unwrap();
    store
        .insert(OrchestrationTaskSpec::new(
            "s1",
            OrchestrationTaskKind::SubAgent {
                agent: "worker".into(),
            },
        ))
        .unwrap();

    let sub_agents = store.list(OrchestrationTaskFilter::default().with_kind("sub_agent"));
    assert_eq!(sub_agents.len(), 1);
    assert_eq!(sub_agents[0].spec.task_id.as_str(), "s1");

    // A created-before bound in the past excludes everything.
    let none = store.list(
        OrchestrationTaskFilter::default().created_between(None, Some(std::time::UNIX_EPOCH)),
    );
    assert!(none.is_empty());
    // A created-after bound in the past includes everything.
    let all = store.list(
        OrchestrationTaskFilter::default().created_between(Some(std::time::UNIX_EPOCH), None),
    );
    assert_eq!(all.len(), 2);
}

#[test]
fn terminal_tasks_reject_further_control() {
    let store = InMemoryTaskStore::new();
    let task_id = TaskId::new("task-a");

    store.insert(graph_spec(task_id.as_str())).unwrap();
    store
        .fail(&task_id, "child failed".to_string())
        .expect("live task can fail");

    let err = store
        .request_cancel(&task_id)
        .expect_err("terminal task cannot be cancelled");
    assert!(err.to_string().contains("cannot transition"));
}

#[test]
fn register_orchestration_tools_adds_normal_tool_names() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let mut registry: ToolRegistry<()> = ToolRegistry::new();

    register_orchestration_tools(&mut registry, store);

    assert!(registry.get("orchestrate_spawn").is_some());
    assert!(registry.get("orchestrate_cancel").is_some());
    assert!(registry.names().contains(&"orchestrate_status".to_string()));
}

#[tokio::test]
async fn spawn_and_status_run_through_tool_trait() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store.clone());
    let status = OrchestrationTool::new(OrchestrationToolKind::Status, store);

    let spawned = spawn
        .call(
            &(),
            ToolCall::new(
                "call-1",
                "orchestrate_spawn",
                json!({
                    "kind": "graph",
                    "target": "planner",
                    "timeout_ms": 1000
                }),
            ),
        )
        .await
        .unwrap();
    let task_id = spawned.raw.unwrap()["spec"]["task_id"]
        .as_str()
        .unwrap()
        .to_string();

    let inspected = status
        .call(
            &(),
            ToolCall::new(
                "call-2",
                "orchestrate_status",
                json!({ "task_id": task_id }),
            ),
        )
        .await
        .unwrap();

    assert_eq!(inspected.raw.unwrap()["status"], "pending");
}
