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

#[tokio::test]
async fn list_tool_honors_created_window_and_kind() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store.clone());
    let list = OrchestrationTool::new(OrchestrationToolKind::List, store);

    spawn
        .call(
            &(),
            ToolCall::new(
                "s1",
                "orchestrate_spawn",
                json!({ "kind": "graph", "target": "planner" }),
            ),
        )
        .await
        .unwrap();
    spawn
        .call(
            &(),
            ToolCall::new(
                "s2",
                "orchestrate_spawn",
                json!({ "kind": "sub_agent", "target": "writer" }),
            ),
        )
        .await
        .unwrap();

    // Kind filter routes through the tool.
    let sub_agents = list
        .call(
            &(),
            ToolCall::new("l1", "orchestrate_list", json!({ "kind": "sub_agent" })),
        )
        .await
        .unwrap();
    let sub_agents = sub_agents.raw.unwrap();
    assert_eq!(sub_agents.as_array().unwrap().len(), 1);
    assert_eq!(sub_agents[0]["spec"]["kind"]["type"], "sub_agent");

    // An impossibly-early upper bound excludes everything created just now.
    let none = list
        .call(
            &(),
            ToolCall::new("l2", "orchestrate_list", json!({ "created_before_ms": 0 })),
        )
        .await
        .unwrap();
    assert!(none.raw.unwrap().as_array().unwrap().is_empty());

    // A window opening at the epoch includes both tasks.
    let all = list
        .call(
            &(),
            ToolCall::new("l3", "orchestrate_list", json!({ "created_after_ms": 0 })),
        )
        .await
        .unwrap();
    assert_eq!(all.raw.unwrap().as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn steer_tool_delivers_command_through_steering_registry() {
    use crate::harness::steering::{SteeringCommand, SteeringHandle};

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();

    // Spawn a task and mark it running; register a steering handle for it.
    let task_id = TaskId::new("child-1");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());

    let steer = OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone())
        .with_steering(steering.clone());

    let result = steer
        .call(
            &(),
            ToolCall::new(
                "call-steer",
                "orchestrate_steer",
                json!({ "task_id": task_id.as_str(), "command": "pause" }),
            ),
        )
        .await
        .unwrap();

    // The command was accepted and actually delivered to the live handle.
    assert_eq!(result.raw.as_ref().unwrap()["accepted"], true);
    let drained = handle.drain();
    assert_eq!(drained.len(), 1);
    assert!(matches!(drained[0], SteeringCommand::Pause));
}

#[tokio::test]
async fn steer_tool_reports_not_delivered_without_registered_handle() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let task_id = TaskId::new("child-2");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();

    // No steering registry attached -> recorded but not delivered.
    let steer = OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone());
    let result = steer
        .call(
            &(),
            ToolCall::new(
                "call-steer",
                "orchestrate_steer",
                json!({ "task_id": task_id.as_str(), "command": "pause" }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(result.raw.as_ref().unwrap()["accepted"], false);
}

#[tokio::test]
async fn steer_tool_delivers_redirect_via_payload() {
    use crate::harness::steering::{SteeringCommand, SteeringHandle};

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-r");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());

    let steer = OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone())
        .with_steering(steering.clone());

    // redirect carries its instruction in the schema-allowed `payload` field.
    let result = steer
        .call(
            &(),
            ToolCall::new(
                "call-steer",
                "orchestrate_steer",
                json!({
                    "task_id": task_id.as_str(),
                    "command": "redirect",
                    "payload": "go north"
                }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(result.raw.as_ref().unwrap()["accepted"], true);
    let drained = handle.drain();
    assert!(matches!(
        &drained[0],
        SteeringCommand::Redirect { instruction } if instruction == "go north"
    ));
}

#[tokio::test]
async fn steer_tool_redirect_without_payload_is_rejected() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-r2");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    steering.register(
        task_id.clone(),
        crate::harness::steering::SteeringHandle::allow_all(),
    );

    let steer =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone()).with_steering(steering);
    let err = steer
        .call(
            &(),
            ToolCall::new(
                "call-steer",
                "orchestrate_steer",
                json!({ "task_id": task_id.as_str(), "command": "redirect" }),
            ),
        )
        .await
        .expect_err("redirect without payload is rejected");
    assert!(matches!(err, crate::TinyAgentsError::Validation(_)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn jsonl_store_persists_inside_multi_thread_runtime() {
    // Exercises the `block_in_place` path: persist runs on a tokio worker.
    let path = unique_log_path("multi-thread-runtime");
    let _ = std::fs::remove_file(&path);
    let task_id = TaskId::new("task-mt");

    let store = JsonlTaskStore::open(&path).unwrap();
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    store
        .complete(&task_id, OrchestrationTaskResult::text("done"))
        .unwrap();

    assert_eq!(store.history(&task_id).len(), 3);
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn jsonl_store_persists_inside_current_thread_runtime() {
    // A current-thread runtime must not panic (block_in_place is skipped).
    let path = unique_log_path("current-thread-runtime");
    let _ = std::fs::remove_file(&path);
    let task_id = TaskId::new("task-ct");

    let store = JsonlTaskStore::open(&path).unwrap();
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();

    assert_eq!(store.history(&task_id).len(), 2);
    let _ = std::fs::remove_file(&path);
}
