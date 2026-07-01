//! Type definitions for graph-level orchestration controls.
//!
//! This file owns the plain data model for managed child work: task identity,
//! task kinds, lifecycle status, task records, store filters, and the
//! model-visible orchestration tool descriptors. Implementations live in
//! `mod.rs` and tests live in `test.rs`.

use std::collections::BTreeMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::ids::{GraphId, NodeId, RunId, TaskId, ThreadId};

/// A managed unit of child work controlled through graph orchestration tools.
///
/// The enum is intentionally runtime-level rather than executor-specific:
/// graph runs, harness sub-agents, harness tools, and future external-process
/// adapters can all be represented without exposing raw executor handles to a
/// model. The first implementation slice records and controls task lifecycle;
/// actual execution adapters plug in at this boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum OrchestrationTaskKind {
    /// A child compiled graph run.
    Graph {
        /// Registered graph identifier.
        graph_id: GraphId,
    },
    /// A child harness sub-agent invocation.
    SubAgent {
        /// Registered agent name.
        agent: String,
    },
    /// A harness tool invocation managed as a child task.
    Tool {
        /// Registered tool name.
        tool: String,
    },
    /// A policy-gated external process placeholder.
    ///
    /// External processes are deliberately represented but not executed by this
    /// module; an adapter must enforce sandboxing and approval policy before
    /// constructing/running this task kind.
    ExternalProcess {
        /// Human-readable label for the process, not an executable command.
        label: String,
    },
}

impl OrchestrationTaskKind {
    /// Returns the stable lower-snake-case kind label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Graph { .. } => "graph",
            Self::SubAgent { .. } => "sub_agent",
            Self::Tool { .. } => "tool",
            Self::ExternalProcess { .. } => "external_process",
        }
    }
}

/// Lifecycle state of a managed orchestration task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationTaskStatus {
    /// Created but not started.
    Pending,
    /// Currently executing.
    Running,
    /// Waiting on a child task or external input.
    Awaiting,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Cooperative cancellation was requested but not yet observed.
    CancelRequested,
    /// The task cooperatively stopped.
    Cancelled,
    /// The task exceeded its deadline.
    TimedOut,
    /// The supervisor stopped waiting and marked the task abandoned.
    Abandoned,
}

impl OrchestrationTaskStatus {
    /// Returns `true` when this status is terminal.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::TimedOut | Self::Abandoned
        )
    }

    /// Returns `true` when this status can still transition due to live work.
    pub fn is_live(self) -> bool {
        !self.is_terminal()
    }
}

/// A request to create a managed orchestration task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationTaskSpec {
    /// Stable task id returned to the model/orchestrator.
    pub task_id: TaskId,
    /// Work kind this task represents.
    pub kind: OrchestrationTaskKind,
    /// Enclosing run id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<RunId>,
    /// Root run id of the run tree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_run_id: Option<RunId>,
    /// Thread id when checkpointing is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<ThreadId>,
    /// Node that spawned this task, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<NodeId>,
    /// Optional task deadline in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Optional structured task input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Sorted metadata for adapters, UIs, and audit trails.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl OrchestrationTaskSpec {
    /// Builds a task spec for `task_id` and `kind`.
    pub fn new(task_id: impl Into<TaskId>, kind: OrchestrationTaskKind) -> Self {
        Self {
            task_id: task_id.into(),
            kind,
            parent_run_id: None,
            root_run_id: None,
            thread_id: None,
            node_id: None,
            timeout_ms: None,
            input: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Sets parent/root run lineage, returning `self` for chaining.
    pub fn with_lineage(
        mut self,
        parent_run_id: impl Into<RunId>,
        root_run_id: impl Into<RunId>,
    ) -> Self {
        self.parent_run_id = Some(parent_run_id.into());
        self.root_run_id = Some(root_run_id.into());
        self
    }

    /// Sets the checkpoint thread id, returning `self` for chaining.
    pub fn with_thread(mut self, thread_id: impl Into<ThreadId>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Sets the spawning node id, returning `self` for chaining.
    pub fn with_node(mut self, node_id: impl Into<NodeId>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Sets the task timeout in milliseconds, returning `self` for chaining.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Sets structured input, returning `self` for chaining.
    pub fn with_input(mut self, input: Value) -> Self {
        self.input = Some(input);
        self
    }

    /// Adds sorted metadata, returning `self` for chaining.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Structured result produced by a managed orchestration task.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationTaskResult {
    /// Model-facing text summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Structured task output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
}

impl OrchestrationTaskResult {
    /// Builds a text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            output: None,
        }
    }

    /// Builds a structured result.
    pub fn output(output: Value) -> Self {
        Self {
            text: None,
            output: Some(output),
        }
    }
}

/// Durable lifecycle record for one managed orchestration task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationTaskRecord {
    /// Immutable task spec.
    pub spec: OrchestrationTaskSpec,
    /// Current lifecycle state.
    pub status: OrchestrationTaskStatus,
    /// Successful result, present only after completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<OrchestrationTaskResult>,
    /// Rendered error, present for failed/timed-out tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Creation time.
    pub created_at: SystemTime,
    /// Last lifecycle update time.
    pub updated_at: SystemTime,
    /// Start time, once the task starts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<SystemTime>,
    /// Terminal time, once the task ends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<SystemTime>,
}

impl OrchestrationTaskRecord {
    /// Creates a pending task record for `spec`.
    pub fn pending(spec: OrchestrationTaskSpec) -> Self {
        let now = SystemTime::now();
        Self {
            spec,
            status: OrchestrationTaskStatus::Pending,
            result: None,
            error: None,
            created_at: now,
            updated_at: now,
            started_at: None,
            ended_at: None,
        }
    }

    /// The task id.
    pub fn task_id(&self) -> &TaskId {
        &self.spec.task_id
    }

    /// Returns `true` if this task is terminal.
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }
}

/// Filter used when listing orchestration tasks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrchestrationTaskFilter {
    /// Restrict to tasks under this parent run.
    pub parent_run_id: Option<RunId>,
    /// Restrict to tasks under this root run.
    pub root_run_id: Option<RunId>,
    /// Restrict to a checkpoint thread.
    pub thread_id: Option<ThreadId>,
    /// Restrict to a spawning graph node.
    pub node_id: Option<NodeId>,
    /// Restrict to tasks currently in one status.
    pub status: Option<OrchestrationTaskStatus>,
    /// Restrict to a task-kind discriminant label (see
    /// [`OrchestrationTaskKind::as_str`]), for example `"sub_agent"`.
    pub kind: Option<String>,
    /// Restrict to tasks created at or after this instant (inclusive).
    pub created_after: Option<SystemTime>,
    /// Restrict to tasks created at or before this instant (inclusive).
    pub created_before: Option<SystemTime>,
}

impl OrchestrationTaskFilter {
    /// Restricts the filter to a task-kind discriminant label.
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    /// Restricts the filter to a created-at window (inclusive bounds).
    pub fn created_between(
        mut self,
        after: Option<SystemTime>,
        before: Option<SystemTime>,
    ) -> Self {
        self.created_after = after;
        self.created_before = before;
        self
    }

    /// Returns `true` when `record` matches every configured filter field.
    pub fn matches(&self, record: &OrchestrationTaskRecord) -> bool {
        if let Some(parent) = &self.parent_run_id
            && record.spec.parent_run_id.as_ref() != Some(parent)
        {
            return false;
        }
        if let Some(root) = &self.root_run_id
            && record.spec.root_run_id.as_ref() != Some(root)
        {
            return false;
        }
        if let Some(thread) = &self.thread_id
            && record.spec.thread_id.as_ref() != Some(thread)
        {
            return false;
        }
        if let Some(node) = &self.node_id
            && record.spec.node_id.as_ref() != Some(node)
        {
            return false;
        }
        if let Some(status) = self.status
            && record.status != status
        {
            return false;
        }
        if let Some(kind) = &self.kind
            && record.spec.kind.as_str() != kind
        {
            return false;
        }
        if let Some(after) = self.created_after
            && record.created_at < after
        {
            return false;
        }
        if let Some(before) = self.created_before
            && record.created_at > before
        {
            return false;
        }
        true
    }
}

/// Model-visible orchestration tool set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationToolKind {
    /// Start managed child work.
    Spawn,
    /// Wait for one or more tasks.
    Await,
    /// Request cooperative cancellation.
    Cancel,
    /// Mark live work abandoned after requesting cancellation.
    Kill,
    /// Inspect one task.
    Status,
    /// List tasks visible to the current run/thread.
    List,
    /// Apply or update a deadline.
    Timeout,
    /// Wait for the first acceptable task result.
    Race,
    /// Pause orchestration behind a durable interrupt.
    YieldInterrupt,
    /// Send policy-checked steering input to a child task.
    Steer,
}

impl OrchestrationToolKind {
    /// Every built-in orchestration tool in declaration order.
    pub const ALL: [Self; 10] = [
        Self::Spawn,
        Self::Await,
        Self::Cancel,
        Self::Kill,
        Self::Status,
        Self::List,
        Self::Timeout,
        Self::Race,
        Self::YieldInterrupt,
        Self::Steer,
    ];

    /// Stable model-visible tool name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Spawn => "orchestrate_spawn",
            Self::Await => "orchestrate_await",
            Self::Cancel => "orchestrate_cancel",
            Self::Kill => "orchestrate_kill",
            Self::Status => "orchestrate_status",
            Self::List => "orchestrate_list",
            Self::Timeout => "orchestrate_timeout",
            Self::Race => "orchestrate_race",
            Self::YieldInterrupt => "orchestrate_yield",
            Self::Steer => "orchestrate_steer",
        }
    }

    /// Short model-visible description.
    pub fn description(self) -> &'static str {
        match self {
            Self::Spawn => "Start managed child work and return a durable task id.",
            Self::Await => "Wait for one or more managed tasks, optionally with a timeout.",
            Self::Cancel => "Request cooperative cancellation of a managed task.",
            Self::Kill => {
                "Stop supervising a task by requesting cancellation and marking it abandoned."
            }
            Self::Status => "Read the current status of a managed task.",
            Self::List => "List managed tasks visible to the current orchestration scope.",
            Self::Timeout => "Set or update the deadline for a managed task.",
            Self::Race => {
                "Wait for the first acceptable task from a set, optionally cancelling losers."
            }
            Self::YieldInterrupt => {
                "Pause orchestration behind a durable interrupt/resume boundary."
            }
            Self::Steer => "Send policy-checked steering input to a managed child task.",
        }
    }
}

/// A tool call outcome returned by orchestration control methods.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationControlOutcome {
    /// Controlled task id.
    pub task_id: TaskId,
    /// Task status after the control was applied.
    pub status: OrchestrationTaskStatus,
    /// Human/model-readable summary.
    pub message: String,
}
