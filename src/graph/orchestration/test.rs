use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::CancellationToken;
use crate::harness::ids::TaskId;
use crate::harness::steering::SteeringHandle;
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum RuntimeStatus {
    Running,
    Completed(String),
}

fn runtime_registry() -> DetachedTaskRegistry<String, RuntimeStatus> {
    DetachedTaskRegistry::new(SteeringRegistry::new(), 2, |status| {
        matches!(status, RuntimeStatus::Completed(_))
    })
}

fn detached_handles() -> (
    tokio::sync::watch::Sender<RuntimeStatus>,
    tokio::sync::watch::Receiver<RuntimeStatus>,
    CancellationToken,
    tokio::task::JoinHandle<()>,
) {
    let (tx, rx) = tokio::sync::watch::channel(RuntimeStatus::Running);
    let cancellation = CancellationToken::new();
    let join = tokio::spawn(std::future::pending());
    (tx, rx, cancellation, join)
}

#[tokio::test]
async fn detached_registry_enforces_owner_and_waits_for_terminal_status() {
    let registry = runtime_registry();
    let task_id = TaskId::new("detached-wait");
    let (tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id.clone(),
            "parent-a",
            "researcher".to_string(),
            rx,
            cancellation,
            join.abort_handle(),
        )
        .unwrap();

    assert_eq!(
        registry.snapshot(&task_id, "parent-b").unwrap_err(),
        DetachedTaskRegistryError::NotOwned
    );
    assert_eq!(
        registry.snapshot(&task_id, "parent-a").unwrap().metadata,
        "researcher"
    );

    assert_eq!(
        registry
            .wait(&task_id, "parent-b", std::time::Duration::from_millis(1))
            .await,
        Err(DetachedTaskRegistryError::NotOwned)
    );
    tx.send(RuntimeStatus::Completed("done".into())).unwrap();
    assert_eq!(
        registry
            .wait(&task_id, "parent-a", std::time::Duration::from_secs(1))
            .await
            .unwrap(),
        DetachedTaskWaitOutcome::Terminal(RuntimeStatus::Completed("done".into()))
    );
    assert!(registry.is_empty().unwrap());
    join.abort();
}

#[tokio::test]
async fn detached_registry_timeout_keeps_task_registered() {
    let registry = runtime_registry();
    let task_id = TaskId::new("detached-timeout");
    let (_tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id.clone(),
            "parent",
            "worker".to_string(),
            rx,
            cancellation,
            join.abort_handle(),
        )
        .unwrap();

    assert_eq!(
        registry
            .wait(&task_id, "parent", std::time::Duration::from_millis(1))
            .await
            .unwrap(),
        DetachedTaskWaitOutcome::TimedOut(RuntimeStatus::Running)
    );
    assert_eq!(registry.len().unwrap(), 1);
    join.abort();
}

#[tokio::test]
async fn detached_registry_cancel_is_cooperative_then_aborts_and_returns_metadata() {
    let registry = runtime_registry();
    let task_id = TaskId::new("detached-cancel");
    let (_tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id.clone(),
            "parent",
            "worker-meta".to_string(),
            rx,
            cancellation.clone(),
            join.abort_handle(),
        )
        .unwrap();

    let cancelled = registry.cancel(&task_id, "parent").unwrap();
    assert_eq!(cancelled.metadata, "worker-meta");
    assert!(cancellation.is_cancelled());
    assert!(join.await.unwrap_err().is_cancelled());
    assert!(registry.is_empty().unwrap());
}

#[tokio::test]
async fn detached_registry_uses_shared_steering_and_sweeps_terminal_at_soft_cap() {
    let steering = SteeringRegistry::new();
    let registry = DetachedTaskRegistry::new(steering.clone(), 1, |status: &RuntimeStatus| {
        matches!(status, RuntimeStatus::Completed(_))
    });
    let first = TaskId::new("detached-first");
    let (first_tx, first_rx, first_cancel, first_join) = detached_handles();
    registry
        .register(
            first.clone(),
            "parent",
            "first".to_string(),
            first_rx,
            first_cancel,
            first_join.abort_handle(),
        )
        .unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(first.clone(), handle);
    assert!(registry.steering_handle(&first, "parent").is_ok());
    first_tx
        .send(RuntimeStatus::Completed("done".into()))
        .unwrap();

    let second = TaskId::new("detached-second");
    let (_second_tx, second_rx, second_cancel, second_join) = detached_handles();
    registry
        .register(
            second.clone(),
            "parent",
            "second".to_string(),
            second_rx,
            second_cancel,
            second_join.abort_handle(),
        )
        .unwrap();

    assert_eq!(registry.len().unwrap(), 1);
    assert!(steering.get(&first).is_none());
    assert_eq!(
        registry.snapshots(Some("parent")).unwrap()[0].task_id,
        second
    );
    first_join.abort();
    second_join.abort();
}

#[derive(Debug)]
struct PanickingMetadata;

impl Clone for PanickingMetadata {
    fn clone(&self) -> Self {
        panic!("poison registry lock during metadata clone")
    }
}

#[tokio::test]
async fn detached_registry_reports_a_poisoned_lock() {
    let registry =
        DetachedTaskRegistry::new(SteeringRegistry::new(), 2, |status: &RuntimeStatus| {
            matches!(status, RuntimeStatus::Completed(_))
        });
    let task_id = TaskId::new("detached-poison");
    let (_tx, rx, cancellation, join) = detached_handles();
    registry
        .register(
            task_id,
            "parent",
            PanickingMetadata,
            rx,
            cancellation,
            join.abort_handle(),
        )
        .unwrap();

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = registry.snapshots(None);
    }));
    assert!(panic.is_err());
    assert_eq!(registry.len(), Err(DetachedTaskRegistryError::LockPoisoned));
    join.abort();
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
async fn spawn_tool_preserves_every_task_kind_input_and_timeout() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store);

    let cases = [
        ("graph", "planner", "graph", "graph_id"),
        ("sub_agent", "writer", "sub_agent", "agent"),
        ("tool", "search", "tool", "tool"),
        (
            "external_process",
            "sandboxed-worker",
            "external_process",
            "label",
        ),
    ];

    for (idx, (kind, target, serialized_kind, target_field)) in cases.into_iter().enumerate() {
        let result = spawn
            .call(
                &(),
                ToolCall::new(
                    format!("spawn-{idx}"),
                    "orchestrate_spawn",
                    json!({
                        "kind": kind,
                        "target": target,
                        "input": { "topic": target },
                        "timeout_ms": 250
                    }),
                ),
            )
            .await
            .unwrap();
        let raw = result.raw.unwrap();
        assert_eq!(raw["status"], "pending");
        assert_eq!(raw["spec"]["kind"]["type"], serialized_kind);
        assert_eq!(raw["spec"]["kind"][target_field], target);
        assert_eq!(raw["spec"]["input"]["topic"], target);
        assert_eq!(raw["spec"]["timeout_ms"], 250);
    }
}

#[tokio::test]
async fn await_cancel_kill_timeout_and_yield_tools_return_control_records() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let task_a = TaskId::new("task-a");
    let task_b = TaskId::new("task-b");
    let task_c = TaskId::new("task-c");
    store.insert(graph_spec(task_a.as_str())).unwrap();
    store.insert(graph_spec(task_b.as_str())).unwrap();
    store.insert(graph_spec(task_c.as_str())).unwrap();

    let timeout = OrchestrationTool::new(OrchestrationToolKind::Timeout, store.clone());
    let awaited = OrchestrationTool::new(OrchestrationToolKind::Await, store.clone());
    let cancel = OrchestrationTool::new(OrchestrationToolKind::Cancel, store.clone());
    let kill = OrchestrationTool::new(OrchestrationToolKind::Kill, store.clone());
    let yield_interrupt = OrchestrationTool::new(OrchestrationToolKind::YieldInterrupt, store);

    let timed = timeout
        .call(
            &(),
            ToolCall::new(
                "timeout",
                "orchestrate_timeout",
                json!({ "task_id": task_a.as_str(), "timeout_ms": 500 }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(timed.raw.as_ref().unwrap()["spec"]["timeout_ms"], 500);

    let records = awaited
        .call(
            &(),
            ToolCall::new(
                "await",
                "orchestrate_await",
                json!({
                    "task_ids": [task_a.as_str(), task_b.as_str()],
                    "timeout_ms": 50,
                    "mode": "all"
                }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(records.raw.unwrap().as_array().unwrap().len(), 2);

    let cancelled = cancel
        .call(
            &(),
            ToolCall::new(
                "cancel",
                "orchestrate_cancel",
                json!({ "task_id": task_b.as_str() }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(
        cancelled.raw.as_ref().unwrap()["status"],
        "cancel_requested"
    );
    assert_eq!(
        cancelled.raw.as_ref().unwrap()["message"],
        "cancellation requested"
    );

    let killed = kill
        .call(
            &(),
            ToolCall::new(
                "kill",
                "orchestrate_kill",
                json!({ "task_id": task_c.as_str() }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(killed.raw.as_ref().unwrap()["status"], "abandoned");

    let yielded = yield_interrupt
        .call(
            &(),
            ToolCall::new(
                "yield",
                "orchestrate_yield",
                json!({
                    "message": "need human input",
                    "resume_schema": { "type": "object" }
                }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(
        yielded.raw.as_ref().unwrap()["status"],
        "interrupt_requested"
    );
    assert_eq!(yielded.raw.as_ref().unwrap()["message"], "need human input");
}

#[tokio::test]
async fn race_tool_reports_completed_winner_and_cancels_live_losers() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let winner = TaskId::new("winner");
    let loser = TaskId::new("loser");
    let terminal_loser = TaskId::new("terminal-loser");
    store.insert(graph_spec(winner.as_str())).unwrap();
    store.insert(graph_spec(loser.as_str())).unwrap();
    store.insert(graph_spec(terminal_loser.as_str())).unwrap();
    store.mark_running(&winner).unwrap();
    store.mark_running(&loser).unwrap();
    store.mark_running(&terminal_loser).unwrap();
    store
        .complete(&winner, OrchestrationTaskResult::text("done"))
        .unwrap();
    store
        .fail(&terminal_loser, "already failed".to_string())
        .unwrap();

    let race = OrchestrationTool::new(OrchestrationToolKind::Race, store.clone());
    let result = race
        .call(
            &(),
            ToolCall::new(
                "race",
                "orchestrate_race",
                json!({
                    "task_ids": [loser.as_str(), winner.as_str(), terminal_loser.as_str()],
                    "cancel_losers": true
                }),
            ),
        )
        .await
        .unwrap();
    let raw = result.raw.unwrap();
    assert_eq!(raw["winner"]["spec"]["task_id"], winner.as_str());
    assert_eq!(
        store.get(&loser).unwrap().status,
        OrchestrationTaskStatus::CancelRequested
    );
    assert_eq!(
        store.get(&terminal_loser).unwrap().status,
        OrchestrationTaskStatus::Failed
    );
}

#[tokio::test]
async fn orchestration_tool_validation_rejects_bad_model_arguments() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let spawn = OrchestrationTool::new(OrchestrationToolKind::Spawn, store.clone());
    let awaited = OrchestrationTool::new(OrchestrationToolKind::Await, store.clone());
    let timeout = OrchestrationTool::new(OrchestrationToolKind::Timeout, store);

    let err = spawn
        .call(
            &(),
            ToolCall::new(
                "bad-spawn",
                "orchestrate_spawn",
                json!({ "kind": "unknown", "target": "x" }),
            ),
        )
        .await
        .expect_err("schema rejects unsupported task kind enum");
    assert!(err.to_string().contains("kind"));

    let err = awaited
        .call(
            &(),
            ToolCall::new(
                "empty-await",
                "orchestrate_await",
                json!({ "task_ids": [] }),
            ),
        )
        .await
        .expect_err("empty task list is rejected");
    assert!(err.to_string().contains("at least one task id"));

    let err = timeout
        .call(
            &(),
            ToolCall::new(
                "bad-timeout",
                "orchestrate_timeout",
                json!({ "task_id": "task-a", "timeout_ms": "soon" }),
            ),
        )
        .await
        .expect_err("schema rejects wrong timeout type");
    assert!(err.to_string().contains("timeout_ms"));
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
async fn steer_tool_delivers_inject_message_and_metadata_payloads() {
    use crate::harness::steering::{SteeringCommand, SteeringHandle};

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-payloads");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store.mark_running(&task_id).unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());
    let steer =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone()).with_steering(steering);

    steer
        .call(
            &(),
            ToolCall::new(
                "inject",
                "orchestrate_steer",
                json!({
                    "task_id": task_id.as_str(),
                    "command": "inject_message",
                    "payload": { "content": "new user hint" }
                }),
            ),
        )
        .await
        .unwrap();
    steer
        .call(
            &(),
            ToolCall::new(
                "metadata",
                "orchestrate_steer",
                json!({
                    "task_id": task_id.as_str(),
                    "command": "set_metadata",
                    "payload": { "priority": "high" }
                }),
            ),
        )
        .await
        .unwrap();

    let drained = handle.drain();
    assert!(matches!(
        &drained[0],
        SteeringCommand::InjectMessage(message) if message.text() == "new user hint"
    ));
    assert!(matches!(
        &drained[1],
        SteeringCommand::SetMetadata { metadata } if metadata["priority"] == "high"
    ));
}

#[tokio::test]
async fn steer_tool_accepts_terminal_task_but_does_not_deliver() {
    use crate::harness::steering::SteeringHandle;

    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
    let steering = SteeringRegistry::new();
    let task_id = TaskId::new("child-done");
    store.insert(graph_spec(task_id.as_str())).unwrap();
    store
        .complete(&task_id, OrchestrationTaskResult::text("done"))
        .unwrap();
    let handle = SteeringHandle::allow_all();
    steering.register(task_id.clone(), handle.clone());

    let steer =
        OrchestrationTool::new(OrchestrationToolKind::Steer, store.clone()).with_steering(steering);
    let result = steer
        .call(
            &(),
            ToolCall::new(
                "terminal-steer",
                "orchestrate_steer",
                json!({ "task_id": task_id.as_str(), "command": "cancel" }),
            ),
        )
        .await
        .unwrap();

    assert_eq!(result.raw.as_ref().unwrap()["accepted"], false);
    assert!(handle.drain().is_empty());
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
