//! Task store implementations for graph orchestration.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::harness::ids::TaskId;
use crate::{Result, TinyAgentsError};

use super::types::*;

/// Store abstraction for managed orchestration tasks.
///
/// The trait is synchronous on purpose: task lifecycle updates happen at graph
/// execution boundaries and should be cheap/in-memory by default. Durable
/// backends can still implement this trait behind a lock or append log and keep
/// async I/O outside the model-visible control path.
pub trait TaskStore: Send + Sync {
    /// Inserts a pending task record.
    fn insert(&self, spec: OrchestrationTaskSpec) -> Result<OrchestrationTaskRecord>;

    /// Reads one task by id.
    fn get(&self, task_id: &TaskId) -> Option<OrchestrationTaskRecord>;

    /// Lists tasks matching `filter`.
    fn list(&self, filter: OrchestrationTaskFilter) -> Vec<OrchestrationTaskRecord>;

    /// Returns the lifecycle history (oldest → newest) for one task.
    ///
    /// The default returns just the current record (latest-only). Durable
    /// append-log backends such as [`JsonlTaskStore`] override this to return
    /// every recorded transition so supervisors can reconstruct the timeline.
    fn history(&self, task_id: &TaskId) -> Vec<OrchestrationTaskRecord> {
        self.get(task_id).into_iter().collect()
    }

    /// Marks a pending task as running.
    fn mark_running(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord>;

    /// Marks a task as awaiting a child task or external input.
    fn mark_awaiting(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord>;

    /// Completes a live task.
    fn complete(
        &self,
        task_id: &TaskId,
        result: OrchestrationTaskResult,
    ) -> Result<OrchestrationTaskRecord>;

    /// Fails a live task.
    fn fail(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord>;

    /// Marks a live task as timed out.
    fn timeout(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord>;

    /// Requests cooperative cancellation of a live task.
    fn request_cancel(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome>;

    /// Marks a cancellation-requested task as cancelled.
    fn mark_cancelled(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord>;

    /// Marks a live task abandoned after requesting cancellation.
    fn kill(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome>;

    /// Updates a non-terminal task deadline in milliseconds.
    fn set_timeout_ms(&self, task_id: &TaskId, timeout_ms: u64) -> Result<OrchestrationTaskRecord>;
}

/// Thread-safe in-memory [`TaskStore`].
#[derive(Clone, Debug, Default)]
pub struct InMemoryTaskStore {
    inner: Arc<Mutex<HashMap<TaskId, OrchestrationTaskRecord>>>,
}

impl InMemoryTaskStore {
    /// Creates an empty in-memory task store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Seeds a store from existing records (used by durable replay). Later
    /// records for the same task id overwrite earlier ones.
    pub fn from_records(records: impl IntoIterator<Item = OrchestrationTaskRecord>) -> Self {
        let store = Self::new();
        if let Ok(mut guard) = store.inner.lock() {
            for record in records {
                guard.insert(record.spec.task_id.clone(), record);
            }
        }
        store
    }

    fn with_task<R>(
        &self,
        task_id: &TaskId,
        f: impl FnOnce(&mut OrchestrationTaskRecord) -> Result<R>,
    ) -> Result<R> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| TinyAgentsError::Graph("orchestration task store lock poisoned".into()))?;
        let record = guard
            .get_mut(task_id)
            .ok_or_else(|| orchestration_not_found(task_id))?;
        f(record)
    }
}

impl TaskStore for InMemoryTaskStore {
    fn insert(&self, spec: OrchestrationTaskSpec) -> Result<OrchestrationTaskRecord> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| TinyAgentsError::Graph("orchestration task store lock poisoned".into()))?;
        if guard.contains_key(&spec.task_id) {
            return Err(TinyAgentsError::DuplicateComponent(format!(
                "orchestration task `{}`",
                spec.task_id
            )));
        }
        let record = OrchestrationTaskRecord::pending(spec);
        guard.insert(record.spec.task_id.clone(), record.clone());
        Ok(record)
    }

    fn get(&self, task_id: &TaskId) -> Option<OrchestrationTaskRecord> {
        self.inner
            .lock()
            .ok()
            .and_then(|guard| guard.get(task_id).cloned())
    }

    fn list(&self, filter: OrchestrationTaskFilter) -> Vec<OrchestrationTaskRecord> {
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let mut records: Vec<_> = guard
            .values()
            .filter(|record| filter.matches(record))
            .cloned()
            .collect();
        records.sort_by(|a, b| a.spec.task_id.as_str().cmp(b.spec.task_id.as_str()));
        records
    }

    fn mark_running(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        self.with_task(task_id, |record| {
            require_status(
                record,
                &[OrchestrationTaskStatus::Pending],
                OrchestrationTaskStatus::Running,
            )?;
            let now = SystemTime::now();
            record.status = OrchestrationTaskStatus::Running;
            record.started_at = Some(now);
            record.updated_at = now;
            Ok(record.clone())
        })
    }

    fn mark_awaiting(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        self.with_task(task_id, |record| {
            require_status(
                record,
                &[
                    OrchestrationTaskStatus::Pending,
                    OrchestrationTaskStatus::Running,
                ],
                OrchestrationTaskStatus::Awaiting,
            )?;
            record.status = OrchestrationTaskStatus::Awaiting;
            record.updated_at = SystemTime::now();
            Ok(record.clone())
        })
    }

    fn complete(
        &self,
        task_id: &TaskId,
        result: OrchestrationTaskResult,
    ) -> Result<OrchestrationTaskRecord> {
        self.with_task(task_id, |record| {
            require_live(record, OrchestrationTaskStatus::Completed)?;
            let now = SystemTime::now();
            record.status = OrchestrationTaskStatus::Completed;
            record.result = Some(result);
            record.error = None;
            record.updated_at = now;
            record.ended_at = Some(now);
            Ok(record.clone())
        })
    }

    fn fail(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord> {
        self.with_task(task_id, |record| {
            require_live(record, OrchestrationTaskStatus::Failed)?;
            let now = SystemTime::now();
            record.status = OrchestrationTaskStatus::Failed;
            record.result = None;
            record.error = Some(error);
            record.updated_at = now;
            record.ended_at = Some(now);
            Ok(record.clone())
        })
    }

    fn timeout(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord> {
        self.with_task(task_id, |record| {
            require_live(record, OrchestrationTaskStatus::TimedOut)?;
            let now = SystemTime::now();
            record.status = OrchestrationTaskStatus::TimedOut;
            record.result = None;
            record.error = Some(error);
            record.updated_at = now;
            record.ended_at = Some(now);
            Ok(record.clone())
        })
    }

    fn request_cancel(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome> {
        self.with_task(task_id, |record| {
            if record.status == OrchestrationTaskStatus::CancelRequested {
                return Ok(control_outcome(
                    record,
                    "cancellation was already requested",
                ));
            }
            require_live(record, OrchestrationTaskStatus::CancelRequested)?;
            record.status = OrchestrationTaskStatus::CancelRequested;
            record.updated_at = SystemTime::now();
            Ok(control_outcome(record, "cancellation requested"))
        })
    }

    fn mark_cancelled(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        self.with_task(task_id, |record| {
            require_status(
                record,
                &[OrchestrationTaskStatus::CancelRequested],
                OrchestrationTaskStatus::Cancelled,
            )?;
            let now = SystemTime::now();
            record.status = OrchestrationTaskStatus::Cancelled;
            record.updated_at = now;
            record.ended_at = Some(now);
            Ok(record.clone())
        })
    }

    fn kill(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome> {
        self.with_task(task_id, |record| {
            require_live(record, OrchestrationTaskStatus::Abandoned)?;
            let now = SystemTime::now();
            record.status = OrchestrationTaskStatus::Abandoned;
            record.error = Some("task abandoned after kill request".to_string());
            record.updated_at = now;
            record.ended_at = Some(now);
            Ok(control_outcome(record, "task abandoned after kill request"))
        })
    }

    fn set_timeout_ms(&self, task_id: &TaskId, timeout_ms: u64) -> Result<OrchestrationTaskRecord> {
        self.with_task(task_id, |record| {
            if record.is_terminal() {
                return Err(invalid_transition(record, OrchestrationTaskStatus::Running));
            }
            record.spec.timeout_ms = Some(timeout_ms);
            record.updated_at = SystemTime::now();
            Ok(record.clone())
        })
    }
}

/// Durable, append-only JSONL-backed [`TaskStore`].
///
/// Every lifecycle transition appends a full [`OrchestrationTaskRecord`] snapshot
/// as one JSON line to a log file, so:
///
/// - a process restart re-hydrates all task state via [`JsonlTaskStore::open`]
///   (replaying the log; the latest snapshot per task id wins), and
/// - [`TaskStore::history`] returns the complete oldest → newest timeline for a
///   task, not just its latest state.
///
/// The in-memory state machine (transition validation, filtering) is reused from
/// [`InMemoryTaskStore`]; this type layers durability and history on top.
pub struct JsonlTaskStore {
    inner: InMemoryTaskStore,
    file: Arc<Mutex<std::fs::File>>,
    history: Arc<Mutex<HashMap<TaskId, Vec<OrchestrationTaskRecord>>>>,
}

impl JsonlTaskStore {
    /// Opens (creating if necessary) a JSONL task log at `path`, replaying any
    /// existing records to reconstruct current state and per-task history.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::io::{BufRead, BufReader};

        let path = path.as_ref();
        let mut history: HashMap<TaskId, Vec<OrchestrationTaskRecord>> = HashMap::new();
        let mut latest: Vec<OrchestrationTaskRecord> = Vec::new();

        if path.exists() {
            let file = std::fs::File::open(path)
                .map_err(|e| TinyAgentsError::Graph(format!("open task log: {e}")))?;
            for line in BufReader::new(file).lines() {
                let line =
                    line.map_err(|e| TinyAgentsError::Graph(format!("read task log: {e}")))?;
                if line.trim().is_empty() {
                    continue;
                }
                let record: OrchestrationTaskRecord = serde_json::from_str(&line)
                    .map_err(|e| TinyAgentsError::Graph(format!("parse task log: {e}")))?;
                history
                    .entry(record.spec.task_id.clone())
                    .or_default()
                    .push(record.clone());
                latest.push(record);
            }
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| TinyAgentsError::Graph(format!("open task log for append: {e}")))?;

        Ok(Self {
            inner: InMemoryTaskStore::from_records(latest),
            file: Arc::new(Mutex::new(file)),
            history: Arc::new(Mutex::new(history)),
        })
    }

    fn persist(&self, record: &OrchestrationTaskRecord) -> Result<()> {
        use std::io::Write;

        let line = serde_json::to_string(record)
            .map_err(|e| TinyAgentsError::Graph(format!("serialize task record: {e}")))?;
        {
            let mut file = self
                .file
                .lock()
                .map_err(|_| TinyAgentsError::Graph("task log file lock poisoned".into()))?;
            writeln!(file, "{line}")
                .map_err(|e| TinyAgentsError::Graph(format!("append task log: {e}")))?;
            file.flush()
                .map_err(|e| TinyAgentsError::Graph(format!("flush task log: {e}")))?;
        }
        if let Ok(mut hist) = self.history.lock() {
            hist.entry(record.spec.task_id.clone())
                .or_default()
                .push(record.clone());
        }
        Ok(())
    }

    /// Persists the current snapshot of `task_id` after a control transition
    /// (used by `request_cancel`/`kill`, which return an outcome, not a record).
    fn persist_current(&self, task_id: &TaskId) -> Result<()> {
        if let Some(record) = self.inner.get(task_id) {
            self.persist(&record)?;
        }
        Ok(())
    }
}

impl TaskStore for JsonlTaskStore {
    fn insert(&self, spec: OrchestrationTaskSpec) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.insert(spec)?;
        self.persist(&record)?;
        Ok(record)
    }

    fn get(&self, task_id: &TaskId) -> Option<OrchestrationTaskRecord> {
        self.inner.get(task_id)
    }

    fn list(&self, filter: OrchestrationTaskFilter) -> Vec<OrchestrationTaskRecord> {
        self.inner.list(filter)
    }

    fn history(&self, task_id: &TaskId) -> Vec<OrchestrationTaskRecord> {
        self.history
            .lock()
            .ok()
            .and_then(|hist| hist.get(task_id).cloned())
            .unwrap_or_default()
    }

    fn mark_running(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.mark_running(task_id)?;
        self.persist(&record)?;
        Ok(record)
    }

    fn mark_awaiting(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.mark_awaiting(task_id)?;
        self.persist(&record)?;
        Ok(record)
    }

    fn complete(
        &self,
        task_id: &TaskId,
        result: OrchestrationTaskResult,
    ) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.complete(task_id, result)?;
        self.persist(&record)?;
        Ok(record)
    }

    fn fail(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.fail(task_id, error)?;
        self.persist(&record)?;
        Ok(record)
    }

    fn timeout(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.timeout(task_id, error)?;
        self.persist(&record)?;
        Ok(record)
    }

    fn request_cancel(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome> {
        let outcome = self.inner.request_cancel(task_id)?;
        self.persist_current(task_id)?;
        Ok(outcome)
    }

    fn mark_cancelled(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.mark_cancelled(task_id)?;
        self.persist(&record)?;
        Ok(record)
    }

    fn kill(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome> {
        let outcome = self.inner.kill(task_id)?;
        self.persist_current(task_id)?;
        Ok(outcome)
    }

    fn set_timeout_ms(&self, task_id: &TaskId, timeout_ms: u64) -> Result<OrchestrationTaskRecord> {
        let record = self.inner.set_timeout_ms(task_id, timeout_ms)?;
        self.persist(&record)?;
        Ok(record)
    }
}

pub(crate) fn orchestration_not_found(task_id: &TaskId) -> TinyAgentsError {
    TinyAgentsError::Graph(format!("orchestration task `{task_id}` does not exist"))
}

fn require_live(record: &OrchestrationTaskRecord, next: OrchestrationTaskStatus) -> Result<()> {
    if record.status.is_live() {
        Ok(())
    } else {
        Err(invalid_transition(record, next))
    }
}

fn require_status(
    record: &OrchestrationTaskRecord,
    allowed: &[OrchestrationTaskStatus],
    next: OrchestrationTaskStatus,
) -> Result<()> {
    if allowed.contains(&record.status) {
        Ok(())
    } else {
        Err(invalid_transition(record, next))
    }
}

fn invalid_transition(
    record: &OrchestrationTaskRecord,
    next: OrchestrationTaskStatus,
) -> TinyAgentsError {
    TinyAgentsError::Graph(format!(
        "cannot transition orchestration task `{}` from {:?} to {:?}",
        record.spec.task_id, record.status, next
    ))
}

fn control_outcome(
    record: &OrchestrationTaskRecord,
    message: impl Into<String>,
) -> OrchestrationControlOutcome {
    OrchestrationControlOutcome {
        task_id: record.spec.task_id.clone(),
        status: record.status,
        message: message.into(),
    }
}
