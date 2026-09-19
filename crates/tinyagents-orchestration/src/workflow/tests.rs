//! Tests for workflow execution: scheduling, phase transitions, concurrency,
//! and error handling.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use parking_lot::Mutex;
use serde_json::json;
use tinyagents_graph::CollectingSink;
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{
    WorkflowLeaseClaim, WorkflowRun, WorkflowRunStatus, WorkflowRunUpsert,
};

use super::engine::PhaseRegistration;
use super::state::set_phase_status;
use super::*;

fn definition() -> WorkflowDefinition {
    WorkflowDefinition {
        id: "test".into(),
        name: "Test".into(),
        description: "test workflow".into(),
        phases: vec![
            WorkflowPhase {
                name: "plan".into(),
                description: "plan".into(),
                agent_ids: vec!["planner".into()],
                depends_on: vec![],
            },
            WorkflowPhase {
                name: "research".into(),
                description: "research".into(),
                agent_ids: vec!["researcher".into(), "researcher".into()],
                depends_on: vec!["plan".into()],
            },
            WorkflowPhase {
                name: "synthesize".into(),
                description: "synthesize".into(),
                agent_ids: vec!["writer".into()],
                depends_on: vec!["research".into()],
            },
        ],
        default_concurrency: 2,
        max_children: 8,
        extensions: BTreeMap::new(),
    }
}

#[derive(Default)]
struct MemoryStore(Mutex<HashMap<String, WorkflowRun>>);

impl WorkflowStore for MemoryStore {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError> {
        Ok(self.0.lock().get(id).cloned())
    }

    fn upsert(&self, update: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError> {
        let now = Utc::now();
        let prior = self.0.lock().get(&update.id).cloned();
        let row = WorkflowRun {
            id: update.id.clone(),
            definition_id: update.definition_id,
            parent_thread_id: update.parent_thread_id,
            input: update.input,
            phase_states: update.phase_states,
            child_run_ids: update.child_run_ids,
            status: update.status,
            summary: update
                .summary
                .or_else(|| prior.as_ref().and_then(|row| row.summary.clone())),
            started_at: update
                .started_at
                .unwrap_or_else(|| prior.as_ref().map(|row| row.started_at).unwrap_or(now)),
            updated_at: now,
            completed_at: update
                .completed_at
                .or_else(|| prior.as_ref().and_then(|row| row.completed_at)),
            revision: prior.as_ref().map_or(0, |row| row.revision + 1),
            lease_owner: prior.as_ref().and_then(|row| row.lease_owner.clone()),
            lease_expires_at: prior.as_ref().and_then(|row| row.lease_expires_at),
        };
        self.0.lock().insert(row.id.clone(), row.clone());
        Ok(row)
    }

    fn claim(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError> {
        let mut rows = self.0.lock();
        let Some(row) = rows.get_mut(id) else {
            return Ok(WorkflowLeaseClaim::Missing);
        };
        let now = Utc::now();
        if row
            .lease_owner
            .as_deref()
            .is_some_and(|current| current != owner)
            && row.lease_expires_at.is_some_and(|until| until > now)
        {
            return Ok(WorkflowLeaseClaim::Busy(row.clone()));
        }
        row.lease_owner = Some(owner.to_owned());
        row.lease_expires_at = chrono::Duration::from_std(lease_for)
            .ok()
            .map(|duration| now + duration);
        row.revision += 1;
        Ok(WorkflowLeaseClaim::Acquired(row.clone()))
    }

    fn compare_and_swap(
        &self,
        update: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError> {
        let mut rows = self.0.lock();
        let Some(prior) = rows.get(&update.id).cloned() else {
            return Ok(None);
        };
        if prior.revision != expected_revision || prior.lease_owner.as_deref() != Some(owner) {
            return Ok(None);
        }
        let now = Utc::now();
        let row = WorkflowRun {
            id: update.id,
            definition_id: update.definition_id,
            parent_thread_id: update.parent_thread_id,
            input: update.input,
            phase_states: update.phase_states,
            child_run_ids: update.child_run_ids,
            status: update.status,
            summary: update.summary.or(prior.summary),
            started_at: update.started_at.unwrap_or(prior.started_at),
            updated_at: now,
            completed_at: update.completed_at.or(prior.completed_at),
            revision: prior.revision + 1,
            lease_owner: (!update.status.is_terminal()).then(|| owner.to_owned()),
            lease_expires_at: (!update.status.is_terminal())
                .then(|| {
                    chrono::Duration::from_std(lease_for)
                        .ok()
                        .map(|duration| now + duration)
                })
                .flatten(),
        };
        rows.insert(row.id.clone(), row.clone());
        Ok(Some(row))
    }

    fn renew(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<bool, OrchestrationError> {
        let mut rows = self.0.lock();
        let Some(row) = rows.get_mut(id) else {
            return Ok(false);
        };
        let now = Utc::now();
        if row.lease_owner.as_deref() != Some(owner)
            || row.lease_expires_at.is_none_or(|expires| expires <= now)
        {
            return Ok(false);
        }
        row.lease_expires_at = chrono::Duration::from_std(lease_for)
            .ok()
            .map(|duration| now + duration);
        Ok(true)
    }
}

#[derive(Default)]
struct RenewFailStore(MemoryStore);

impl WorkflowStore for RenewFailStore {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError> {
        self.0.load(id)
    }

    fn upsert(&self, update: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError> {
        self.0.upsert(update)
    }

    fn claim(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError> {
        self.0.claim(id, owner, lease_for)
    }

    fn compare_and_swap(
        &self,
        update: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError> {
        self.0
            .compare_and_swap(update, expected_revision, owner, lease_for)
    }

    fn renew(
        &self,
        _id: &str,
        _owner: &str,
        _lease_for: Duration,
    ) -> Result<bool, OrchestrationError> {
        Ok(false)
    }
}

#[derive(Default)]
struct FakeExecutor {
    calls: Mutex<Vec<WorkflowChildRequest>>,
    active: AtomicUsize,
    peak: AtomicUsize,
    fail_agent: Mutex<Option<String>>,
    cancelled: AtomicUsize,
}

#[derive(Default)]
struct BlockingExecutor {
    started: tokio::sync::Notify,
    cancelled: Mutex<Vec<String>>,
    calls: AtomicUsize,
}

#[async_trait]
impl WorkflowExecutor for BlockingExecutor {
    async fn execute(
        &self,
        request: WorkflowChildRequest,
        cancel: CancellationToken,
        registration: Arc<dyn WorkflowChildRegistration>,
    ) -> Result<WorkflowChildResult, OrchestrationError> {
        let id = format!("live-{}-{}", request.phase, request.index_in_phase);
        registration.register(id.clone())?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        cancel.cancelled().await;
        Err(OrchestrationError("cancelled while child was live".into()))
    }

    async fn cancel_children(&self, child_ids: &[String]) {
        self.cancelled.lock().extend(child_ids.iter().cloned());
    }
}

#[async_trait]
impl WorkflowExecutor for FakeExecutor {
    async fn execute(
        &self,
        request: WorkflowChildRequest,
        cancel: CancellationToken,
        registration: Arc<dyn WorkflowChildRegistration>,
    ) -> Result<WorkflowChildResult, OrchestrationError> {
        if cancel.is_cancelled() {
            return Err(OrchestrationError("cancelled".into()));
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        self.calls.lock().push(request.clone());
        registration.register(format!("{}-{}", request.phase, request.index_in_phase))?;
        tokio::task::yield_now().await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        if self.fail_agent.lock().as_deref() == Some(request.agent_id.as_str()) {
            return Err(OrchestrationError("child failure".into()));
        }
        Ok(WorkflowChildResult {
            child_id: format!("{}-{}", request.phase, request.index_in_phase),
            output: json!(format!("{} output", request.phase)),
        })
    }

    async fn cancel_children(&self, _child_ids: &[String]) {
        self.cancelled.fetch_add(1, Ordering::SeqCst);
    }
}

fn engine() -> (
    Arc<MemoryStore>,
    Arc<FakeExecutor>,
    WorkflowEngine<MemoryStore, FakeExecutor>,
) {
    let store = Arc::new(MemoryStore::default());
    let executor = Arc::new(FakeExecutor::default());
    let engine = WorkflowEngine::new(store.clone(), executor.clone());
    (store, executor, engine)
}

#[test]
fn structural_validation_covers_invalid_definitions() {
    let mut empty = definition();
    empty.phases.clear();
    assert_eq!(validate_structure(&empty), vec![DefinitionError::NoPhases]);
    let mut bad = definition();
    bad.default_concurrency = 0;
    bad.max_children = 0;
    bad.phases[1].name = "plan".into();
    bad.phases[2].depends_on = vec!["missing".into()];
    let errors = validate_structure(&bad);
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, DefinitionError::DuplicatePhase { .. }))
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, DefinitionError::UnknownDependency { .. }))
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, DefinitionError::InvalidConcurrency { .. }))
    );
    let mut cyclic = definition();
    cyclic.phases[0].depends_on = vec!["synthesize".into()];
    assert!(validate_structure(&cyclic).contains(&DefinitionError::CyclicDependency));
}

#[test]
fn scheduler_topology_preview_exposes_dispatch_run_and_done() {
    let topology = scheduler_topology_preview().expect("topology");
    let nodes = topology
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<Vec<_>>();
    assert!(nodes.contains(&"dispatch") && nodes.contains(&"run_phase") && nodes.contains(&"done"));
}

#[tokio::test]
async fn engine_runs_in_deterministic_dependency_order_and_threads_context() {
    let (store, executor, engine) = engine();
    let def = definition();
    engine
        .initialise("run".into(), &def, json!({"question":"q"}), None)
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    let run = store.load("run").unwrap().unwrap();
    assert_eq!(run.status, WorkflowRunStatus::Completed);
    let calls = executor.calls.lock();
    assert_eq!(
        calls
            .iter()
            .map(|call| call.phase.as_str())
            .collect::<Vec<_>>(),
        vec!["plan", "research", "research", "synthesize"]
    );
    assert!(
        calls
            .last()
            .unwrap()
            .prompt
            .contains("Context from prior phases")
    );
    assert!(run.summary.unwrap().contains("synthesize output"));
}

#[tokio::test]
async fn engine_respects_concurrency_global_cap_and_partial_failure() {
    let (store, executor, engine) = engine();
    let mut def = definition();
    def.default_concurrency = 1;
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    assert!(executor.peak.load(Ordering::SeqCst) <= 1);
    let mut cap = definition();
    cap.max_children = 2;
    engine
        .initialise("cap".into(), &cap, json!("q"), None)
        .unwrap();
    engine
        .drive("cap", &cap, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        store.load("cap").unwrap().unwrap().status,
        WorkflowRunStatus::Failed
    );
    *executor.fail_agent.lock() = Some("planner".into());
    let failing = definition();
    engine
        .initialise("failed".into(), &failing, json!("q"), None)
        .unwrap();
    engine
        .drive("failed", &failing, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        store.load("failed").unwrap().unwrap().status,
        WorkflowRunStatus::Failed
    );
}

#[tokio::test]
async fn cancellation_and_resume_do_not_repeat_completed_phases() {
    let (store, executor, engine) = engine();
    let def = definition();
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    engine.drive("run", &def, cancel).await.unwrap();
    assert_eq!(
        store.load("run").unwrap().unwrap().status,
        WorkflowRunStatus::Interrupted
    );
    let mut states = init_phase_states(&def);
    set_phase_status(
        &mut states,
        "plan",
        PhaseStatus::Completed,
        Some(json!([{ "output": "already" }])),
    );
    let run = store.load("run").unwrap().unwrap();
    store
        .upsert(WorkflowRunUpsert {
            id: run.id,
            definition_id: run.definition_id,
            parent_thread_id: run.parent_thread_id,
            input: run.input,
            phase_states: states,
            child_run_ids: vec!["old".into()],
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: Some(run.started_at),
            completed_at: None,
        })
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    assert!(
        !executor
            .calls
            .lock()
            .iter()
            .any(|call| call.phase == "plan")
    );
    assert_eq!(
        store.load("run").unwrap().unwrap().status,
        WorkflowRunStatus::Completed
    );
}

#[tokio::test]
async fn cancellation_after_workers_start_cancels_durably_registered_children() {
    let store = Arc::new(MemoryStore::default());
    let executor = Arc::new(BlockingExecutor::default());
    let engine = Arc::new(WorkflowEngine::new(store.clone(), executor.clone()));
    let mut def = definition();
    def.phases = vec![WorkflowPhase {
        name: "live".into(),
        description: "live".into(),
        agent_ids: vec!["one".into(), "two".into()],
        depends_on: vec![],
    }];
    def.default_concurrency = 2;
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    let cancel = CancellationToken::new();
    let drive = {
        let engine = engine.clone();
        let def = def.clone();
        let cancel = cancel.clone();
        tokio::spawn(async move { engine.drive("run", &def, cancel).await })
    };
    executor.started.notified().await;
    // The notification occurs only after register(), so this races precisely
    // the old orphan window between real child spawn and ledger persistence.
    cancel.cancel();
    drive.await.unwrap().unwrap();
    let run = store.load("run").unwrap().unwrap();
    assert_eq!(run.status, WorkflowRunStatus::Interrupted);
    assert_eq!(
        run.phase_states["live"]["status"],
        json!("pending"),
        "an interrupted in-flight phase must be retryable on resume"
    );
    assert!(!run.child_run_ids.is_empty());
    let cancelled = executor.cancelled.lock().clone();
    for id in &run.child_run_ids {
        assert!(cancelled.contains(id), "missing cancellation for {id}");
    }
}

#[tokio::test]
async fn concurrent_drives_acquire_one_lease_and_do_not_duplicate_children() {
    let (store, executor, engine) = engine();
    let def = definition();
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    let engine = Arc::new(engine);
    let first = {
        let engine = engine.clone();
        let def = def.clone();
        tokio::spawn(async move { engine.drive("run", &def, CancellationToken::new()).await })
    };
    let second = {
        let engine = engine.clone();
        let def = def.clone();
        tokio::spawn(async move { engine.drive("run", &def, CancellationToken::new()).await })
    };
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    assert_eq!(
        store.load("run").unwrap().unwrap().status,
        WorkflowRunStatus::Completed
    );
    assert_eq!(executor.calls.lock().len(), 4, "one driver owns all phases");
}

#[tokio::test]
async fn expired_owner_takeover_resets_running_phase_and_retries_once() {
    let (store, executor, engine) = engine();
    let mut def = definition();
    def.phases = vec![WorkflowPhase {
        name: "recover".into(),
        description: "recover".into(),
        agent_ids: vec!["worker".into()],
        depends_on: vec![],
    }];
    engine
        .initialise("takeover".into(), &def, json!("q"), None)
        .unwrap();

    // Simulate a process death after the owner has persisted `running`, but
    // before it can complete or reset the phase.
    let old = match store
        .claim("takeover", "crashed-owner", Duration::from_millis(3))
        .unwrap()
    {
        WorkflowLeaseClaim::Acquired(run) => run,
        other => panic!("expected lease, got {other:?}"),
    };
    store
        .compare_and_swap(
            WorkflowRunUpsert {
                id: old.id.clone(),
                definition_id: old.definition_id.clone(),
                parent_thread_id: old.parent_thread_id.clone(),
                input: old.input.clone(),
                phase_states: json!({
                    "recover": {"status": "running", "outputs": []}
                }),
                child_run_ids: vec!["recover-0".into()],
                status: WorkflowRunStatus::Running,
                summary: None,
                started_at: Some(old.started_at),
                completed_at: None,
            },
            old.revision,
            "crashed-owner",
            Duration::from_millis(3),
        )
        .unwrap()
        .expect("crashed owner persists phase start");
    tokio::time::sleep(Duration::from_millis(10)).await;

    engine
        .drive("takeover", &def, CancellationToken::new())
        .await
        .expect("new owner retries reclaimed phase");

    let run = store.load("takeover").unwrap().expect("run remains");
    assert_eq!(run.status, WorkflowRunStatus::Completed);
    assert_eq!(run.phase_states["recover"]["status"], json!("completed"));
    assert_eq!(
        executor
            .calls
            .lock()
            .iter()
            .filter(|call| call.phase == "recover")
            .count(),
        1,
        "lease takeover must schedule the reclaimed phase once"
    );
    assert_eq!(
        run.child_run_ids
            .iter()
            .filter(|id| id.as_str() == "recover-0")
            .count(),
        1,
        "the retry must not duplicate a durably registered child id"
    );
}

#[tokio::test]
async fn heartbeat_renews_a_short_lease_while_a_child_is_running() {
    let store = Arc::new(MemoryStore::default());
    let executor = Arc::new(BlockingExecutor::default());
    let engine = Arc::new(
        WorkflowEngine::new(store.clone(), executor.clone())
            .with_lease_duration(Duration::from_millis(30)),
    );
    let mut def = definition();
    def.phases = vec![WorkflowPhase {
        name: "live".into(),
        description: "live".into(),
        agent_ids: vec!["one".into()],
        depends_on: vec![],
    }];
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    let cancel = CancellationToken::new();
    let first = {
        let engine = engine.clone();
        let def = def.clone();
        let cancel = cancel.clone();
        tokio::spawn(async move { engine.drive("run", &def, cancel).await })
    };
    executor.started.notified().await;
    tokio::time::sleep(Duration::from_millis(75)).await;
    // This exceeds the original lease, so it only remains busy if the
    // in-flight driver's heartbeat kept renewing it.
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    cancel.cancel();
    first.await.unwrap().unwrap();
}

#[tokio::test]
async fn lost_heartbeat_cancels_registered_children_and_fails_closed() {
    let store = Arc::new(RenewFailStore::default());
    let executor = Arc::new(BlockingExecutor::default());
    let engine = Arc::new(
        WorkflowEngine::new(store.clone(), executor.clone())
            .with_lease_duration(Duration::from_millis(30)),
    );
    let mut def = definition();
    def.phases = vec![WorkflowPhase {
        name: "live".into(),
        description: "live".into(),
        agent_ids: vec!["one".into()],
        depends_on: vec![],
    }];
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    let drive = {
        let engine = engine.clone();
        let def = def.clone();
        tokio::spawn(async move { engine.drive("run", &def, CancellationToken::new()).await })
    };
    executor.started.notified().await;
    let error = drive
        .await
        .unwrap()
        .expect_err("renewal loss must fail closed");
    assert!(error.0.contains("lease renewal failed"));
    assert!(
        executor
            .cancelled
            .lock()
            .iter()
            .any(|id| id == "live-live-0"),
        "the child registered before waiting must be cancelled"
    );
}

#[tokio::test]
async fn terminal_events_are_truthful_and_flushed() {
    let (store, executor, engine) = engine();
    let sink = Arc::new(CollectingSink::new());
    let engine = engine.with_event_sink(sink.clone());
    let def = definition();
    engine
        .initialise("ok".into(), &def, json!("q"), None)
        .unwrap();
    engine
        .drive("ok", &def, CancellationToken::new())
        .await
        .unwrap();
    assert!(
        sink.events()
            .iter()
            .any(|event| matches!(event, tinyagents_graph::GraphEvent::RunCompleted { .. }))
    );

    *executor.fail_agent.lock() = Some("planner".into());
    engine
        .initialise("failed".into(), &def, json!("q"), None)
        .unwrap();
    engine
        .drive("failed", &def, CancellationToken::new())
        .await
        .unwrap();
    assert!(sink.events().iter().any(|event| matches!(
        event,
        tinyagents_graph::GraphEvent::RunFailed { error, .. } if error.contains("child failure")
    )));
    assert_eq!(
        store.load("failed").unwrap().unwrap().status,
        WorkflowRunStatus::Failed
    );
}

#[tokio::test]
async fn fenced_driver_exits_silently_when_a_replacement_is_running() {
    let store = Arc::new(RenewFailStore::default());
    let executor = Arc::new(BlockingExecutor::default());
    let sink = Arc::new(CollectingSink::new());
    let engine = Arc::new(
        WorkflowEngine::new(store.clone(), executor.clone())
            .with_lease_duration(Duration::from_millis(30))
            .with_event_sink(sink.clone()),
    );
    let mut def = definition();
    def.phases = vec![WorkflowPhase {
        name: "live".into(),
        description: "live".into(),
        agent_ids: vec!["one".into()],
        depends_on: vec![],
    }];
    engine
        .initialise("handoff".into(), &def, json!("q"), None)
        .unwrap();
    let first = {
        let engine = engine.clone();
        let def = def.clone();
        tokio::spawn(async move {
            engine
                .drive("handoff", &def, CancellationToken::new())
                .await
        })
    };
    executor.started.notified().await;

    // A replacement owner acquires after expiry before the old driver's
    // heartbeat sees its loss. The old loop must not flush a false failure.
    let current = store.load("handoff").unwrap().unwrap();
    store
        .compare_and_swap(
            WorkflowRunUpsert {
                id: current.id.clone(),
                definition_id: current.definition_id.clone(),
                parent_thread_id: current.parent_thread_id.clone(),
                input: current.input.clone(),
                phase_states: current.phase_states.clone(),
                child_run_ids: current.child_run_ids.clone(),
                status: WorkflowRunStatus::Running,
                summary: None,
                started_at: Some(current.started_at),
                completed_at: None,
            },
            current.revision,
            current.lease_owner.as_deref().unwrap(),
            Duration::from_millis(3),
        )
        .unwrap()
        .unwrap();
    tokio::time::sleep(Duration::from_millis(5)).await;
    match store
        .claim("handoff", "replacement", Duration::from_secs(1))
        .unwrap()
    {
        WorkflowLeaseClaim::Acquired(run) => {
            assert_eq!(run.lease_owner.as_deref(), Some("replacement"));
        }
        other => panic!("expected replacement lease, got {other:?}"),
    }
    first.await.unwrap().expect("fenced driver exits cleanly");
    assert!(
        !sink
            .events()
            .iter()
            .any(|event| matches!(event, tinyagents_graph::GraphEvent::RunFailed { .. })),
        "a fenced driver must not report the active replacement as failed"
    );
}

#[test]
fn structured_outputs_are_preserved_in_context_and_summary() {
    let def = definition();
    let mut states = init_phase_states(&def);
    set_phase_status(
        &mut states,
        "plan",
        PhaseStatus::Completed,
        Some(json!([{
            "output": { "claims": ["a", "b"], "score": 7 }
        }])),
    );
    let upstream = upstream_outputs(&def.phases[1], &states);
    let prompt = phase_prompt(&json!("q"), &def.phases[1], 0, &upstream);
    assert!(prompt.contains(r#"{"claims":["a","b"],"score":7}"#));
    set_phase_status(
        &mut states,
        "synthesize",
        PhaseStatus::Completed,
        Some(json!([{
            "output": { "answer": "kept" }
        }])),
    );
    assert_eq!(
        synthesize_summary(&def, &states).as_deref(),
        Some(r#"{"answer":"kept"}"#)
    );
}

#[tokio::test]
async fn output_wire_shape_remains_compatible_while_json_stays_lossless() {
    let (store, _executor, engine) = engine();
    let mut def = definition();
    def.phases = vec![WorkflowPhase {
        name: "only".into(),
        description: "only".into(),
        agent_ids: vec!["planner".into()],
        depends_on: vec![],
    }];
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    let output = store.load("run").unwrap().unwrap().phase_states["only"]["outputs"][0].clone();
    assert_eq!(output["agentId"], json!("planner"));
    assert!(output["output"].is_string());
    assert_eq!(output["metadata"]["version"], json!(2));
    assert_eq!(output["metadata"]["rawOutput"], json!("only output"));
}

/// M11 regression: `PhaseRegistration::register` no longer holds its
/// `parking_lot::Mutex` across the blocking DB CAS (see `engine.rs`'s
/// `WorkflowChildRegistration for PhaseRegistration` impl). This exercises
/// the CAS semantics that guarded property depends on, from a real tokio
/// async context (tasks actually spawned onto the runtime, not just
/// sequential `.await`s), so a reintroduced deadlock or a lost write under
/// contention would show up here rather than only in production.
#[tokio::test]
async fn phase_registration_register_is_idempotent_and_survives_concurrent_registration() {
    let store = Arc::new(MemoryStore::default());
    let seed = store
        .upsert(WorkflowRunUpsert {
            id: "run-1".into(),
            definition_id: "test".into(),
            parent_thread_id: None,
            input: json!({}),
            phase_states: json!({}),
            child_run_ids: vec![],
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: None,
            completed_at: None,
        })
        .unwrap();
    let owner = "owner-1".to_owned();
    let claimed = match store
        .claim(&seed.id, &owner, Duration::from_secs(60))
        .unwrap()
    {
        WorkflowLeaseClaim::Acquired(run) => run,
        other => panic!("expected to acquire the lease, got {other:?}"),
    };

    let registration = Arc::new(PhaseRegistration::new(
        store.clone(),
        owner,
        claimed,
        json!({}),
        Duration::from_secs(60),
    ));

    // Double registration of the same id must stay idempotent: no error, no
    // duplicate entry — this is the pre-existing contract `register`'s
    // duplicate check preserves.
    registration.register("child-a".into()).unwrap();
    registration.register("child-a".into()).unwrap();

    // Concurrent registrations of *distinct* ids from spawned tokio tasks
    // race the same CAS loop this refactor changed. None may be lost, none
    // may deadlock (the test's own timeout — the harness default — is the
    // deadlock detector: a regression here hangs instead of failing fast).
    const CONCURRENT: usize = 16;
    let mut handles = Vec::with_capacity(CONCURRENT);
    for index in 0..CONCURRENT {
        let registration = registration.clone();
        handles.push(tokio::spawn(async move {
            registration.register(format!("child-concurrent-{index}"))
        }));
    }
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let final_run = store.load("run-1").unwrap().expect("run still present");
    let mut unique_ids = final_run.child_run_ids.clone();
    unique_ids.sort();
    unique_ids.dedup();
    assert_eq!(
        final_run.child_run_ids.len(),
        unique_ids.len(),
        "concurrent registration must not duplicate a child id: {:?}",
        final_run.child_run_ids
    );
    assert_eq!(
        final_run.child_run_ids.len(),
        1 + CONCURRENT,
        "every registration (the duplicate `child-a` call collapses to one) \
         must land durably: {:?}",
        final_run.child_run_ids
    );
    assert!(final_run.child_run_ids.contains(&"child-a".to_owned()));
    for index in 0..CONCURRENT {
        assert!(
            final_run
                .child_run_ids
                .contains(&format!("child-concurrent-{index}"))
        );
    }
}
