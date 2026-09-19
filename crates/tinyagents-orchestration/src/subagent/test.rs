use std::time::Duration;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use tinyagents_harness::{
    CancellationToken,
    context::{RunConfig, RunContext},
};
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::{message::Message, usage::UsageTotals};

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Action {
    Load,
    Prepare,
    Execute,
    Pause,
    Terminal(SubagentStatusName),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SubagentStatusName {
    Completed,
    Incomplete,
    Cancelled,
}

#[derive(Clone, Copy)]
enum ExecutorMode {
    Completed,
    Incomplete,
    Pause,
    WaitForCancellation,
    CancelAfterExecution,
    Error,
}

struct FakePlanner {
    calls: Mutex<usize>,
    saw_resume: Mutex<bool>,
    seen_resumes: Mutex<Vec<bool>>,
    reject: bool,
    actions: Arc<Mutex<Vec<Action>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HostRequest {
    label: String,
}

struct PayloadPlanner {
    payloads: Mutex<Vec<HostRequest>>,
}

#[async_trait]
impl SubagentPlanner<String, HostRequest> for PayloadPlanner {
    async fn prepare(
        &self,
        request: SubagentRequest<String, HostRequest>,
    ) -> Result<PreparedSubagent<String>, SubagentError> {
        let request = request.into_parts();
        self.payloads.lock().unwrap().push(request.host_request);
        Ok(PreparedSubagent {
            task_id: request.task_key.task_id,
            agent_key: "payload-agent".into(),
            input: vec![Message::user(request.input)],
            tools: ToolSnapshot::new(vec![]).unwrap(),
            run_context: request.run_context,
        })
    }
}

#[async_trait]
impl SubagentPlanner<String> for FakePlanner {
    async fn prepare(
        &self,
        request: SubagentRequest<String>,
    ) -> Result<PreparedSubagent<String>, SubagentError> {
        let request = request.into_parts();
        *self.calls.lock().unwrap() += 1;
        *self.saw_resume.lock().unwrap() = request.resume.is_some();
        self.seen_resumes
            .lock()
            .unwrap()
            .push(request.resume.is_some());
        self.actions.lock().unwrap().push(Action::Prepare);
        if self.reject {
            return Err(SubagentError::Planning("rejected".into()));
        }
        Ok(PreparedSubagent {
            task_id: request.task_key.task_id,
            agent_key: "resolved-agent".into(),
            input: vec![Message::user(request.input)],
            tools: ToolSnapshot::new(vec![]).unwrap(),
            run_context: request.run_context,
        })
    }
}

struct FakeExecutor {
    calls: Mutex<usize>,
    context_ids: Mutex<Vec<u64>>,
    root_run_ids: Mutex<Vec<String>>,
    context_cancellations: Mutex<Vec<CancellationToken>>,
    mode: ExecutorMode,
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    actions: Arc<Mutex<Vec<Action>>>,
}

#[async_trait]
impl SubagentExecutor<String> for FakeExecutor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        *self.calls.lock().unwrap() += 1;
        self.context_ids
            .lock()
            .unwrap()
            .push(execution.prepared.run_context.instance_id());
        self.root_run_ids.lock().unwrap().push(
            execution
                .prepared
                .run_context
                .lineage()
                .root_run_id
                .as_str()
                .to_owned(),
        );
        self.context_cancellations
            .lock()
            .unwrap()
            .push(execution.prepared.run_context.cancellation.clone());
        self.actions.lock().unwrap().push(Action::Execute);
        if matches!(self.mode, ExecutorMode::WaitForCancellation) {
            if let Some(sender) = self.started.lock().unwrap().take() {
                let _ = sender.send(());
            }
            execution.cancellation.cancelled().await;
        }
        if matches!(self.mode, ExecutorMode::CancelAfterExecution) {
            execution.cancellation.cancel();
        }
        if matches!(self.mode, ExecutorMode::Error) {
            return Err(SubagentError::Execution("executor failed".into()));
        }
        let status = match self.mode {
            ExecutorMode::Completed
            | ExecutorMode::WaitForCancellation
            | ExecutorMode::CancelAfterExecution => SubagentStatus::Completed,
            ExecutorMode::Incomplete => SubagentStatus::Incomplete(SubagentIncomplete {
                reason: "budget exhausted".into(),
            }),
            ExecutorMode::Pause => SubagentStatus::AwaitingInput(SubagentPause {
                reason: "need approval".into(),
                resume: SubagentResume::default(),
            }),
            ExecutorMode::Error => unreachable!(),
        };
        Ok(SubagentOutcome {
            task_id: execution.prepared.task_id,
            output: "result".into(),
            history: vec![Message::assistant("result")],
            status,
            usage: UsageTotals {
                calls: 7,
                ..UsageTotals::default()
            },
            artifacts: vec![ArtifactReference {
                id: "artifact-1".into(),
                ..ArtifactReference::default()
            }],
        })
    }
}

enum LoadMode {
    Empty,
    Resume,
    Error,
}

type Fakes = (
    Arc<FakePlanner>,
    Arc<FakeExecutor>,
    Arc<FakePersistence>,
    Arc<Mutex<Vec<Action>>>,
);

struct FakePersistence {
    actions: Arc<Mutex<Vec<Action>>>,
    load_mode: LoadMode,
    pause_error: bool,
    terminal_error: bool,
    outcomes: Mutex<Vec<SubagentOutcome>>,
    terminals: Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>,
    pauses: Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>,
    saved_pause: Mutex<Option<SubagentResume>>,
    keys: Mutex<Vec<SubagentTaskKey>>,
}

#[async_trait]
impl SubagentPersistence for FakePersistence {
    async fn load_terminal(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        self.keys.lock().unwrap().push(key.clone());
        Ok(self.terminals.lock().unwrap().get(key).cloned())
    }
    async fn load(&self, key: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError> {
        self.actions.lock().unwrap().push(Action::Load);
        self.keys.lock().unwrap().push(key.clone());
        match self.load_mode {
            LoadMode::Empty => Ok(self.saved_pause.lock().unwrap().clone()),
            LoadMode::Resume => Ok(Some(SubagentResume {
                checkpoint: Some("saved".into()),
                ..SubagentResume::default()
            })),
            LoadMode::Error => Err(SubagentError::Persistence("load failed".into())),
        }
    }

    async fn load_pause(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(self.pauses.lock().unwrap().get(key).cloned())
    }

    async fn save_pause(
        &self,
        pause: PersistedSubagentPause,
    ) -> Result<SubagentPausePersistenceDisposition, SubagentError> {
        self.actions.lock().unwrap().push(Action::Pause);
        self.keys.lock().unwrap().push(pause.key.clone());
        if self.pause_error {
            Err(SubagentError::Persistence("pause save failed".into()))
        } else {
            let mut pauses = self.pauses.lock().unwrap();
            match pauses.get(&pause.key) {
                None if pause.replaces.is_none() => {
                    let resume = match &pause.outcome.status {
                        SubagentStatus::AwaitingInput(pause) => pause.resume.clone(),
                        _ => unreachable!("fake only persists awaiting outcomes"),
                    };
                    *self.saved_pause.lock().unwrap() = Some(resume);
                    pauses.insert(pause.key, pause.outcome);
                    Ok(SubagentPausePersistenceDisposition::Inserted)
                }
                Some(current)
                    if pause.replaces.as_ref().is_some_and(|expected| {
                        matches!(
                            &current.status,
                            SubagentStatus::AwaitingInput(current_pause)
                                if current_pause.resume == *expected
                        )
                    }) =>
                {
                    let resume = match &pause.outcome.status {
                        SubagentStatus::AwaitingInput(pause) => pause.resume.clone(),
                        _ => unreachable!("fake only persists awaiting outcomes"),
                    };
                    *self.saved_pause.lock().unwrap() = Some(resume);
                    pauses.insert(pause.key, pause.outcome);
                    Ok(SubagentPausePersistenceDisposition::Replaced)
                }
                _ => Ok(SubagentPausePersistenceDisposition::Existing),
            }
        }
    }

    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
        replaces: Option<&SubagentResume>,
    ) -> Result<SubagentTerminalPersistenceDisposition, SubagentError> {
        self.actions
            .lock()
            .unwrap()
            .push(Action::Terminal(match outcome.status {
                SubagentStatus::Completed => SubagentStatusName::Completed,
                SubagentStatus::Incomplete(_) => SubagentStatusName::Incomplete,
                SubagentStatus::Cancelled => SubagentStatusName::Cancelled,
                SubagentStatus::AwaitingInput(_) => unreachable!(),
            }));
        self.keys.lock().unwrap().push(key.clone());
        if self.terminal_error {
            Err(SubagentError::Persistence("terminal save failed".into()))
        } else {
            let mut terminals = self.terminals.lock().unwrap();
            if terminals.contains_key(key) {
                Ok(SubagentTerminalPersistenceDisposition::Existing)
            } else if self.pauses.lock().unwrap().get(key).is_some_and(|paused| {
                !replaces.is_some_and(|expected| {
                    matches!(
                        &paused.status,
                        SubagentStatus::AwaitingInput(pause) if pause.resume == *expected
                    )
                })
            }) {
                Ok(SubagentTerminalPersistenceDisposition::PauseExisting)
            } else {
                terminals.insert(key.clone(), outcome.clone());
                self.pauses.lock().unwrap().remove(key);
                self.outcomes.lock().unwrap().push(outcome.clone());
                Ok(SubagentTerminalPersistenceDisposition::Inserted)
            }
        }
    }
}

fn request(task_id: &str, data: &str) -> SubagentRequest<String> {
    let parent = RunContext::new(RunConfig::new(format!("parent-{task_id}")), data.to_owned());
    let child = parent
        .child(RunConfig::new(format!("run-{task_id}")), data.to_owned())
        .unwrap();
    SubagentRequest::fresh_from_parent(&parent, child, task_id, (), "do work", None).unwrap()
}

fn request_with_parent(task_id: &str, run_context: RunContext<String>) -> SubagentRequest<String> {
    let task_key = SubagentTaskKey::from_context(&run_context, task_id, Some("thread-1".into()));
    SubagentRequest::continue_with_key(task_key, run_context, (), "do work", None).unwrap()
}

fn continuation_request(
    task_key: SubagentTaskKey,
    run_context: RunContext<String>,
) -> SubagentRequest<String> {
    SubagentRequest::continue_with_key(task_key, run_context, (), "do work", None).unwrap()
}

fn driver(
    planner: Arc<dyn SubagentPlanner<String>>,
    executor: Arc<dyn SubagentExecutor<String>>,
    persistence: Arc<dyn SubagentPersistence>,
) -> SubagentDriver<String> {
    SubagentDriver::new(SubagentCapabilities {
        planner: Some(planner),
        executor: Some(executor),
        persistence: Some(persistence),
    })
    .unwrap()
}

fn fakes(mode: ExecutorMode) -> Fakes {
    let actions = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(FakePlanner {
            calls: Mutex::new(0),
            saw_resume: Mutex::new(false),
            seen_resumes: Mutex::new(Vec::new()),
            reject: false,
            actions: actions.clone(),
        }),
        Arc::new(FakeExecutor {
            calls: Mutex::new(0),
            context_ids: Mutex::new(Vec::new()),
            root_run_ids: Mutex::new(Vec::new()),
            context_cancellations: Mutex::new(Vec::new()),
            mode,
            started: Mutex::new(None),
            actions: actions.clone(),
        }),
        Arc::new(FakePersistence {
            actions: actions.clone(),
            load_mode: LoadMode::Empty,
            pause_error: false,
            terminal_error: false,
            outcomes: Mutex::new(Vec::new()),
            terminals: Mutex::new(HashMap::new()),
            pauses: Mutex::new(HashMap::new()),
            saved_pause: Mutex::new(None),
            keys: Mutex::new(Vec::new()),
        }),
        actions,
    )
}

/// Persistence fake whose first selected operation cannot commit until the
/// test releases it. This makes the cancellation/commit boundary observable.
#[derive(Clone, Copy)]
enum BlockingStage {
    Pause,
    Terminal,
}

struct BlockingPersistence {
    stage: BlockingStage,
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    first: AtomicBool,
    outcomes: Mutex<Vec<SubagentOutcome>>,
}

#[async_trait]
impl SubagentPersistence for BlockingPersistence {
    async fn load_terminal(
        &self,
        _key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(None)
    }
    async fn load(&self, _: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError> {
        Ok(None)
    }

    async fn load_pause(
        &self,
        _: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError> {
        Ok(None)
    }

    async fn save_pause(
        &self,
        _: PersistedSubagentPause,
    ) -> Result<SubagentPausePersistenceDisposition, SubagentError> {
        if matches!(self.stage, BlockingStage::Pause) && self.first.swap(false, Ordering::AcqRel) {
            self.started.notify_one();
            self.release.notified().await;
        }
        Ok(SubagentPausePersistenceDisposition::Inserted)
    }

    async fn record_terminal(
        &self,
        _: &SubagentTaskKey,
        outcome: &SubagentOutcome,
        _: Option<&SubagentResume>,
    ) -> Result<SubagentTerminalPersistenceDisposition, SubagentError> {
        if matches!(self.stage, BlockingStage::Terminal) && self.first.swap(false, Ordering::AcqRel)
        {
            self.started.notify_one();
            self.release.notified().await;
        }
        self.outcomes.lock().unwrap().push(outcome.clone());
        Ok(SubagentTerminalPersistenceDisposition::Inserted)
    }
}

struct MismatchedPlanner;

#[async_trait]
impl SubagentPlanner<String> for MismatchedPlanner {
    async fn prepare(
        &self,
        request: SubagentRequest<String>,
    ) -> Result<PreparedSubagent<String>, SubagentError> {
        let request = request.into_parts();
        Ok(PreparedSubagent {
            task_id: "other-task".into(),
            agent_key: "resolved-agent".into(),
            input: vec![Message::user(request.input)],
            tools: ToolSnapshot::new(vec![]).unwrap(),
            run_context: request.run_context,
        })
    }
}

struct MismatchedExecutor;

#[async_trait]
impl SubagentExecutor<String> for MismatchedExecutor {
    async fn execute(
        &self,
        _: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        Ok(SubagentOutcome {
            task_id: "other-task".into(),
            output: String::new(),
            history: Vec::new(),
            status: SubagentStatus::Completed,
            usage: UsageTotals::default(),
            artifacts: Vec::new(),
        })
    }
}

struct NestedExecutor {
    driver: Mutex<Option<std::sync::Weak<SubagentDriver<String>>>>,
    calls: Mutex<Vec<String>>,
}

struct PauseThenCompleteExecutor {
    calls: Mutex<usize>,
    actions: Arc<Mutex<Vec<Action>>>,
}

#[async_trait]
impl SubagentExecutor<String> for PauseThenCompleteExecutor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        self.actions.lock().unwrap().push(Action::Execute);
        Ok(SubagentOutcome {
            task_id: execution.prepared.task_id,
            output: if call == 1 {
                "waiting".into()
            } else {
                "completed".into()
            },
            history: Vec::new(),
            status: if call == 1 {
                SubagentStatus::AwaitingInput(SubagentPause {
                    reason: "need input".into(),
                    resume: SubagentResume {
                        checkpoint: Some("resume-token".into()),
                        ..SubagentResume::default()
                    },
                })
            } else {
                SubagentStatus::Completed
            },
            usage: UsageTotals::default(),
            artifacts: Vec::new(),
        })
    }
}

#[async_trait]
impl SubagentExecutor<String> for NestedExecutor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        let task_id = execution.prepared.task_id.clone();
        self.calls.lock().unwrap().push(task_id.clone());
        if task_id == "parent" {
            let child_driver = self
                .driver
                .lock()
                .unwrap()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .expect("nested executor is attached to its driver");
            let child = child_driver
                .run(
                    SubagentRequest::continue_with_key(
                        SubagentTaskKey::from_context(
                            &execution.prepared.run_context,
                            "child",
                            None,
                        ),
                        execution.prepared.run_context,
                        (),
                        "nested work",
                        None,
                    )?,
                    execution.cancellation,
                )
                .await?;
            return Ok(SubagentOutcome {
                task_id,
                output: "parent result".into(),
                history: Vec::new(),
                status: SubagentStatus::Completed,
                // This models the host's parent-visible roll-up: the child is
                // added once alongside the parent's own model call.
                usage: UsageTotals {
                    calls: child.outcome.usage.calls + 1,
                    ..UsageTotals::default()
                },
                artifacts: Vec::new(),
            });
        }
        Ok(SubagentOutcome {
            task_id,
            output: "child result".into(),
            history: Vec::new(),
            status: SubagentStatus::Completed,
            usage: UsageTotals {
                calls: 7,
                ..UsageTotals::default()
            },
            artifacts: Vec::new(),
        })
    }
}

#[tokio::test]
async fn planner_rejection_does_not_execute_or_persist_terminal_state() {
    let (_, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    let rejecting = Arc::new(FakePlanner {
        calls: Mutex::new(0),
        saw_resume: Mutex::new(false),
        seen_resumes: Mutex::new(Vec::new()),
        reject: true,
        actions: actions.clone(),
    });
    let result = driver(rejecting.clone(), executor.clone(), persistence.clone())
        .run(request("task", "ctx"), CancellationToken::new())
        .await;

    assert_eq!(result, Err(SubagentError::Planning("rejected".into())));
    assert_eq!(*executor.calls.lock().unwrap(), 0);
    assert!(persistence.outcomes.lock().unwrap().is_empty());
    assert_eq!(
        *actions.lock().unwrap(),
        vec![Action::Load, Action::Prepare]
    );
}

#[tokio::test]
async fn prepared_context_identity_reaches_executor() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let incoming = request("task", "identity");
    let expected = incoming.run_context().instance_id();
    let outcome = driver(planner, executor.clone(), persistence)
        .run(incoming, CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(outcome.outcome.status, SubagentStatus::Completed);
    assert_eq!(*executor.context_ids.lock().unwrap(), vec![expected]);
}

#[tokio::test]
async fn caller_owned_child_context_keeps_its_lineage_into_execution() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let root = RunContext::new(RunConfig::new("root-lineage"), "root".to_owned());
    let child = root
        .child(RunConfig::new("owned-child"), "child".to_owned())
        .unwrap();
    driver(planner, executor.clone(), persistence)
        .run(
            request_with_parent("lineage", child),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(*executor.root_run_ids.lock().unwrap(), vec!["root-lineage"]);
}

#[tokio::test]
async fn opaque_host_request_reaches_the_planner_unchanged() {
    let (_, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let planner = Arc::new(PayloadPlanner {
        payloads: Mutex::new(Vec::new()),
    });
    let context = RunContext::new(RunConfig::new("payload-run"), "ctx".into());
    let key = SubagentTaskKey::from_context(&context, "payload", Some("thread-1".into()));
    let request = SubagentRequest::continue_with_key(
        key,
        context,
        HostRequest {
            label: "per-call options".into(),
        },
        "do work",
        None,
    )
    .unwrap();
    SubagentDriver::new(SubagentCapabilities {
        planner: Some(planner.clone()),
        executor: Some(executor),
        persistence: Some(persistence),
    })
    .unwrap()
    .run(request, CancellationToken::new())
    .await
    .unwrap();
    assert_eq!(
        *planner.payloads.lock().unwrap(),
        vec![HostRequest {
            label: "per-call options".into()
        }]
    );
}

#[tokio::test]
async fn completed_incomplete_and_pause_use_one_mutually_exclusive_persistence_action() {
    for (mode, expected) in [
        (
            ExecutorMode::Completed,
            Action::Terminal(SubagentStatusName::Completed),
        ),
        (
            ExecutorMode::Incomplete,
            Action::Terminal(SubagentStatusName::Incomplete),
        ),
        (ExecutorMode::Pause, Action::Pause),
    ] {
        let (planner, executor, persistence, actions) = fakes(mode);
        driver(planner, executor, persistence)
            .run(request("task", "ctx"), CancellationToken::new())
            .await
            .unwrap();
        let records = actions.lock().unwrap();
        assert_eq!(
            records
                .iter()
                .filter(|action| matches!(action, Action::Pause | Action::Terminal(_)))
                .count(),
            1
        );
        assert_eq!(records.last(), Some(&expected));
    }
}

#[tokio::test]
async fn load_pause_and_terminal_errors_remain_typed() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let load_failure = Arc::new(FakePersistence {
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Error,
        pause_error: false,
        terminal_error: false,
        outcomes: Mutex::new(Vec::new()),
        terminals: Mutex::new(HashMap::new()),
        pauses: Mutex::new(HashMap::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    assert_eq!(
        driver(planner.clone(), executor.clone(), load_failure)
            .run(request("load", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Persistence("load failed".into()))
    );

    let (planner, executor, persistence, _) = fakes(ExecutorMode::Pause);
    let pause_failure = Arc::new(FakePersistence {
        pause_error: true,
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Empty,
        terminal_error: false,
        outcomes: Mutex::new(Vec::new()),
        terminals: Mutex::new(HashMap::new()),
        pauses: Mutex::new(HashMap::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    assert_eq!(
        driver(planner, executor, pause_failure)
            .run(request("pause", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Persistence("pause save failed".into()))
    );

    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let terminal_failure = Arc::new(FakePersistence {
        terminal_error: true,
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Empty,
        pause_error: false,
        outcomes: Mutex::new(Vec::new()),
        terminals: Mutex::new(HashMap::new()),
        pauses: Mutex::new(HashMap::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    assert_eq!(
        driver(planner, executor, terminal_failure)
            .run(request("terminal", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Persistence("terminal save failed".into()))
    );
}

#[tokio::test]
async fn loaded_resume_reaches_planner_before_execution_and_execution_errors_do_not_persist() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    let resume_persistence = Arc::new(FakePersistence {
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Resume,
        pause_error: false,
        terminal_error: false,
        outcomes: Mutex::new(Vec::new()),
        terminals: Mutex::new(HashMap::new()),
        pauses: Mutex::new(HashMap::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    driver(planner.clone(), executor, resume_persistence)
        .run(request("resume", "ctx"), CancellationToken::new())
        .await
        .unwrap();
    assert!(*planner.saw_resume.lock().unwrap());
    assert_eq!(
        actions.lock().unwrap()[..2],
        [Action::Load, Action::Prepare]
    );

    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Error);
    assert_eq!(
        driver(planner, executor, persistence)
            .run(request("execution-error", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Execution("executor failed".into()))
    );
    assert_eq!(
        *actions.lock().unwrap(),
        vec![Action::Load, Action::Prepare, Action::Execute]
    );
}

#[tokio::test]
async fn duplicate_task_id_returns_recorded_outcome_without_second_execution_or_record() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let driver = driver(planner.clone(), executor.clone(), persistence.clone());
    let first = driver
        .run(request("same", "one"), CancellationToken::new())
        .await
        .unwrap();
    let second = driver
        .run(request("same", "two"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(first.outcome, second.outcome);
    assert_eq!(
        first.disposition,
        SubagentPersistenceDisposition::TerminalInserted
    );
    assert_eq!(
        second.disposition,
        SubagentPersistenceDisposition::TerminalExisting
    );
    assert_eq!(*planner.calls.lock().unwrap(), 1);
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn separate_drivers_share_terminal_winner_and_disposition() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let first = driver(planner.clone(), executor.clone(), persistence.clone())
        .run(request("shared-terminal", "one"), CancellationToken::new())
        .await
        .unwrap();
    let second = driver(planner, executor, persistence.clone())
        .run(request("shared-terminal", "two"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(
        first.disposition,
        SubagentPersistenceDisposition::TerminalInserted
    );
    assert_eq!(
        second.disposition,
        SubagentPersistenceDisposition::TerminalExisting
    );
    assert_eq!(first.outcome, second.outcome);
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_later_continuation_replaces_its_consumed_pause_and_owns_new_effects() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Pause);
    let first = driver(planner.clone(), executor.clone(), persistence.clone())
        .run(request("shared-pause", "one"), CancellationToken::new())
        .await
        .unwrap();
    let second = driver(planner, executor, persistence)
        .run(request("shared-pause", "two"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(
        first.disposition,
        SubagentPersistenceDisposition::PauseCommitted
    );
    assert_eq!(
        second.disposition,
        SubagentPersistenceDisposition::PauseReplaced
    );
    assert!(first.should_emit_host_effects());
    assert!(second.should_emit_host_effects());
}

#[tokio::test]
async fn fresh_request_rejects_an_unrelated_or_thread_mismatched_context() {
    let parent = RunContext::new(
        RunConfig::new("parent").with_thread("thread-a"),
        "parent".to_owned(),
    );
    let unrelated = RunContext::new(
        RunConfig::new("unrelated").with_thread("thread-a"),
        "other".to_owned(),
    );
    assert!(matches!(
        SubagentRequest::fresh_from_parent(&parent, unrelated, "task", (), "input", None),
        Err(SubagentError::InvalidRequest(_))
    ));

    let wrong_thread_child = parent
        .child(
            RunConfig::new("wrong-thread-child").with_thread("thread-b"),
            "child".to_owned(),
        )
        .unwrap();
    assert!(matches!(
        SubagentRequest::fresh_from_parent(&parent, wrong_thread_child, "task", (), "input", None,),
        Err(SubagentError::InvalidRequest(_))
    ));

    let key = SubagentTaskKey::from_context(&parent, "task", None);
    let fresh_wrong_thread = RunContext::new(
        RunConfig::new("continuation").with_thread("thread-b"),
        "fresh".to_owned(),
    );
    assert!(matches!(
        SubagentRequest::continue_with_key(key, fresh_wrong_thread, (), "input", None),
        Err(SubagentError::InvalidRequest(_))
    ));
}

#[tokio::test]
async fn concurrent_same_task_calls_coalesce_to_one_lifecycle() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(
        planner.clone(),
        executor.clone(),
        persistence.clone(),
    ));
    let cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);

    let first = tokio::spawn({
        let driver = driver.clone();
        let cancellation = cancellation.clone();
        async move {
            driver
                .run(request("same-in-flight", "one"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    let second = tokio::spawn({
        let driver = driver.clone();
        async move {
            driver
                .run(request("same-in-flight", "two"), CancellationToken::new())
                .await
        }
    });
    tokio::task::yield_now().await;
    cancellation.cancel();

    assert_eq!(
        first.await.unwrap().unwrap().outcome.status,
        SubagentStatus::Cancelled
    );
    assert_eq!(
        second.await.unwrap().unwrap().outcome.status,
        SubagentStatus::Cancelled
    );
    assert_eq!(*planner.calls.lock().unwrap(), 1);
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn cancelled_follower_returns_without_cancelling_the_leader_or_persisting() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(
        planner.clone(),
        executor.clone(),
        persistence.clone(),
    ));
    let leader_cancellation = CancellationToken::new();
    let follower_cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);

    let leader = tokio::spawn({
        let driver = driver.clone();
        let cancellation = leader_cancellation.clone();
        async move {
            driver
                .run(request("follower-cancellation", "leader"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    let follower = tokio::spawn({
        let driver = driver.clone();
        let cancellation = follower_cancellation.clone();
        async move {
            driver
                .run(request("follower-cancellation", "follower"), cancellation)
                .await
        }
    });
    follower_cancellation.cancel();

    let follower_outcome = tokio::time::timeout(Duration::from_secs(1), follower)
        .await
        .expect("cancelled follower must not wait for the leader")
        .unwrap()
        .unwrap();
    assert_eq!(follower_outcome.outcome.status, SubagentStatus::Cancelled);
    assert_eq!(
        follower_outcome.disposition,
        SubagentPersistenceDisposition::ObserverCancelled
    );
    assert!(!follower_outcome.should_emit_host_effects());
    assert!(!leader_cancellation.is_cancelled());
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert!(persistence.outcomes.lock().unwrap().is_empty());

    leader_cancellation.cancel();
    assert_eq!(
        leader.await.unwrap().unwrap().outcome.status,
        SubagentStatus::Cancelled
    );
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn same_task_id_from_distinct_parent_runs_never_shares_lifecycle_state() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let driver = driver(planner.clone(), executor.clone(), persistence.clone());
    let root = RunContext::new(RunConfig::new("root"), "root-data".to_owned());
    let first_parent = root
        .child(RunConfig::new("parent-one"), "first-parent".into())
        .unwrap();
    let second_parent = root
        .child(RunConfig::new("parent-two"), "second-parent".into())
        .unwrap();

    driver
        .run(
            request_with_parent("same-task", first_parent),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    driver
        .run(
            request_with_parent("same-task", second_parent),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_eq!(*planner.calls.lock().unwrap(), 2);
    assert_eq!(*executor.calls.lock().unwrap(), 2);
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 2);
    let keys = persistence.keys.lock().unwrap();
    let terminal_keys = keys
        .iter()
        .filter(|key| key.task_id == "same-task")
        .collect::<Vec<_>>();
    assert_eq!(
        terminal_keys.len(),
        6,
        "initial terminal lookup, pause lookup, and terminal persistence use each scoped key"
    );
    assert!(
        terminal_keys
            .iter()
            .any(|key| key.parent_run_id == "parent-one")
    );
    assert!(
        terminal_keys
            .iter()
            .any(|key| key.parent_run_id == "parent-two")
    );
    assert!(terminal_keys.iter().all(|key| key.root_run_id == "root"));
}

#[tokio::test]
async fn awaiting_input_is_not_cached_and_the_next_call_resumes_to_completion() {
    let (planner, _, persistence, actions) = fakes(ExecutorMode::Pause);
    let executor = Arc::new(PauseThenCompleteExecutor {
        calls: Mutex::new(0),
        actions: actions.clone(),
    });
    let driver = driver(planner.clone(), executor.clone(), persistence.clone());

    let first = driver
        .run(request("resumable", "first"), CancellationToken::new())
        .await
        .unwrap();
    let second = driver
        .run(request("resumable", "continued"), CancellationToken::new())
        .await
        .unwrap();

    assert!(matches!(
        first.outcome.status,
        SubagentStatus::AwaitingInput(_)
    ));
    assert_eq!(second.outcome.status, SubagentStatus::Completed);
    assert_eq!(*planner.calls.lock().unwrap(), 2);
    assert_eq!(*executor.calls.lock().unwrap(), 2);
    assert_eq!(*planner.seen_resumes.lock().unwrap(), vec![false, true]);
    assert_eq!(
        actions
            .lock()
            .unwrap()
            .iter()
            .filter(|action| matches!(action, Action::Pause | Action::Terminal(_)))
            .cloned()
            .collect::<Vec<_>>(),
        vec![
            Action::Pause,
            Action::Terminal(SubagentStatusName::Completed)
        ]
    );
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn continuation_keeps_original_key_with_a_fresh_owned_context() {
    let (planner, _, persistence, actions) = fakes(ExecutorMode::Pause);
    let executor = Arc::new(PauseThenCompleteExecutor {
        calls: Mutex::new(0),
        actions,
    });
    let lifecycle = driver(planner.clone(), executor, persistence.clone());
    let original = RunContext::new(RunConfig::new("original-run"), "first".into());
    let key = SubagentTaskKey::from_context(&original, "continued", Some("thread-1".into()));
    lifecycle
        .run(
            continuation_request(key.clone(), original),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let fresh = RunContext::new(RunConfig::new("fresh-turn-context"), "second".into());
    let completed = lifecycle
        .run(
            continuation_request(key.clone(), fresh),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(completed.outcome.status, SubagentStatus::Completed);
    assert_eq!(*planner.seen_resumes.lock().unwrap(), vec![false, true]);
    assert!(
        persistence
            .keys
            .lock()
            .unwrap()
            .iter()
            .all(|seen| seen == &key)
    );
}

#[tokio::test]
async fn driver_replaces_prepared_context_cancellation_with_execution_token() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(planner, executor.clone(), persistence));
    let cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);

    let task = tokio::spawn({
        let driver = driver.clone();
        let cancellation = cancellation.clone();
        async move {
            driver
                .run(request("shared-cancellation", "ctx"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    cancellation.cancel();
    let outcome = task.await.unwrap().unwrap();

    assert_eq!(outcome.outcome.status, SubagentStatus::Cancelled);
    assert!(executor.context_cancellations.lock().unwrap()[0].is_cancelled());
}

#[tokio::test]
async fn cancellation_before_persistence_commit_records_only_cancelled_terminal() {
    for (stage, mode, task_id) in [
        (
            BlockingStage::Pause,
            ExecutorMode::Pause,
            "cancel-save-pause",
        ),
        (
            BlockingStage::Terminal,
            ExecutorMode::Completed,
            "cancel-record-terminal",
        ),
    ] {
        let (planner, executor, _, _) = fakes(mode);
        let persistence = Arc::new(BlockingPersistence {
            stage,
            started: Arc::new(tokio::sync::Notify::new()),
            release: Arc::new(tokio::sync::Notify::new()),
            first: AtomicBool::new(true),
            outcomes: Mutex::new(Vec::new()),
        });
        let driver = Arc::new(driver(planner, executor, persistence.clone()));
        let cancellation = CancellationToken::new();
        let started = persistence.started.clone();
        let run = tokio::spawn({
            let driver = driver.clone();
            let cancellation = cancellation.clone();
            async move { driver.run(request(task_id, "ctx"), cancellation).await }
        });

        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("first persistence action must be pending");
        cancellation.cancel();
        let outcome = tokio::time::timeout(Duration::from_secs(1), run)
            .await
            .expect("cancellation must resolve the lifecycle")
            .unwrap()
            .unwrap();

        assert_eq!(outcome.outcome.status, SubagentStatus::Cancelled);
        assert_eq!(
            outcome.disposition,
            SubagentPersistenceDisposition::TerminalInserted
        );
        assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
        assert_eq!(
            persistence.outcomes.lock().unwrap()[0].status,
            SubagentStatus::Cancelled
        );
    }
}

#[tokio::test]
async fn planner_and_executor_task_id_mismatches_do_not_persist_or_cache() {
    let (_, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let planner_error = driver(Arc::new(MismatchedPlanner), executor, persistence.clone())
        .run(request("expected", "ctx"), CancellationToken::new())
        .await;
    assert_eq!(
        planner_error,
        Err(SubagentError::TaskIdMismatch {
            expected: "expected".into(),
            actual: "other-task".into(),
        })
    );
    assert!(persistence.outcomes.lock().unwrap().is_empty());

    let (planner, _, persistence, _) = fakes(ExecutorMode::Completed);
    let executor_error = driver(planner, Arc::new(MismatchedExecutor), persistence.clone())
        .run(request("expected", "ctx"), CancellationToken::new())
        .await;
    assert_eq!(
        executor_error,
        Err(SubagentError::TaskIdMismatch {
            expected: "expected".into(),
            actual: "other-task".into(),
        })
    );
    assert!(persistence.outcomes.lock().unwrap().is_empty());
}

#[tokio::test]
async fn nested_same_driver_task_uses_child_reservation_and_rolls_usage_up_once() {
    let (planner, _, persistence, _) = fakes(ExecutorMode::Completed);
    let executor = Arc::new(NestedExecutor {
        driver: Mutex::new(None),
        calls: Mutex::new(Vec::new()),
    });
    let driver = Arc::new(driver(planner, executor.clone(), persistence.clone()));
    *executor.driver.lock().unwrap() = Some(Arc::downgrade(&driver));

    let parent = tokio::time::timeout(
        Duration::from_secs(1),
        driver.run(request("parent", "ctx"), CancellationToken::new()),
    )
    .await
    .expect("a nested child task must not wait on a driver-global lock")
    .unwrap();

    assert_eq!(parent.outcome.usage.calls, 8);
    assert_eq!(*executor.calls.lock().unwrap(), vec!["parent", "child"]);
    let records = persistence.outcomes.lock().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(
        records
            .iter()
            .find(|outcome| outcome.task_id == "parent")
            .expect("parent outcome is persisted")
            .usage
            .calls,
        8
    );
}

#[tokio::test]
async fn cancellation_before_execution_records_one_truthful_terminal() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let outcome = driver(planner.clone(), executor.clone(), persistence)
        .run(request("cancel-before", "ctx"), cancellation)
        .await
        .unwrap();

    assert_eq!(outcome.outcome.status, SubagentStatus::Cancelled);
    assert_eq!(*planner.calls.lock().unwrap(), 0);
    assert_eq!(*executor.calls.lock().unwrap(), 0);
    assert_eq!(
        *actions.lock().unwrap(),
        vec![Action::Terminal(SubagentStatusName::Cancelled)]
    );
}

#[tokio::test]
async fn cancellation_during_execution_is_truthful_and_terminal_once() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(planner, executor.clone(), persistence));
    let cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);
    let task = tokio::spawn({
        let driver = driver.clone();
        let cancellation = cancellation.clone();
        async move {
            driver
                .run(request("cancel-during", "ctx"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    cancellation.cancel();
    let outcome = task.await.unwrap().unwrap();

    assert_eq!(outcome.outcome.status, SubagentStatus::Cancelled);
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert_eq!(
        actions.lock().unwrap().last(),
        Some(&Action::Terminal(SubagentStatusName::Cancelled))
    );
}

#[tokio::test]
async fn cancellation_after_execution_preserves_lossless_result_data() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::CancelAfterExecution);
    let outcome = driver(planner, executor, persistence)
        .run(request("cancel-after", "ctx"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(outcome.outcome.status, SubagentStatus::Cancelled);
    assert_eq!(outcome.outcome.output, "result");
    assert_eq!(outcome.outcome.history, vec![Message::assistant("result")]);
    assert_eq!(outcome.outcome.usage.calls, 7);
    assert_eq!(outcome.outcome.artifacts.len(), 1);
}

#[tokio::test]
async fn nested_usage_is_persisted_once_and_failure_order_is_load_prepare_execute_then_terminal() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    driver(planner, executor, persistence.clone())
        .run(request("usage", "ctx"), CancellationToken::new())
        .await
        .unwrap();

    let outcomes = persistence.outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].usage.calls, 7);
    assert_eq!(
        *actions.lock().unwrap(),
        vec![
            Action::Load,
            Action::Prepare,
            Action::Execute,
            Action::Terminal(SubagentStatusName::Completed),
        ]
    );
}

#[test]
fn absent_host_capabilities_fail_closed() {
    let result = SubagentDriver::<String>::new(SubagentCapabilities {
        planner: None,
        executor: None,
        persistence: None,
    });
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("missing host capabilities must fail closed"),
    };
    assert_eq!(error, SubagentError::MissingCapability("planner"));
}
