//! Serde record types for the run ledger: background agent runs, workflow
//! runs, run events/telemetry, and agent-team coordination (teams, members,
//! tasks).
//!
//! These mirror the tables created by migration 2 (and later) in
//! `crate::migrations`. Each persisted record type (`AgentRun`, `WorkflowRun`,
//! `RunEvent`, `RunTelemetry`, `AgentTeam`, `AgentTeamMember`,
//! `AgentTeamTask`) has a paired `*Upsert` type carrying only the fields a
//! caller supplies — timestamps and derived fields are filled in by
//! `super::ops` rather than the caller. List request/response pairs
//! (`*ListRequest` / `*ListResponse`) are the shapes `super::ops`'s listing
//! functions accept and return, and double as the RPC-facing schema for
//! hosts that expose the ledger over an API.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What kind of agent run an [`AgentRun`] represents.
///
/// Distinguishes the several call sites that create a run row so listings and
/// UI can filter by origin without inferring it from other fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentRunKind {
    /// A synchronous sub-agent invocation nested inside a parent run.
    Subagent,
    /// A background worker thread spawned to run independently of its caller.
    WorkerThread,
    /// A top-level background agent with no synchronous caller.
    BackgroundAgent,
    /// A foreground agent turn initiated directly by a user or host caller.
    InteractiveTurn,
    /// An agent acting as a member of an [`AgentTeam`].
    TeamMember,
    /// A run spawned as a child of a [`WorkflowRun`] phase.
    WorkflowChild,
}

impl AgentRunKind {
    /// Renders the kind as the string stored in the `kind` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Subagent => "subagent",
            Self::WorkerThread => "worker_thread",
            Self::BackgroundAgent => "background_agent",
            Self::InteractiveTurn => "interactive_turn",
            Self::TeamMember => "team_member",
            Self::WorkflowChild => "workflow_child",
        }
    }

    /// Parses a stored `kind` string, defaulting to [`AgentRunKind::Subagent`]
    /// for any unrecognized value.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "worker_thread" => Self::WorkerThread,
            "background_agent" => Self::BackgroundAgent,
            "interactive_turn" => Self::InteractiveTurn,
            "team_member" => Self::TeamMember,
            "workflow_child" => Self::WorkflowChild,
            _ => Self::Subagent,
        }
    }
}

/// Lifecycle status of a single [`AgentRun`].
///
/// Convertible to and from the canonical `tinyagents_tasks::OrchestrationTaskStatus`
/// (see `status_map.rs`); the serialized form is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentRunStatus {
    /// Created but not yet started.
    Pending,
    /// Actively executing.
    Running,
    /// Paused pending input from the user (e.g. a tool-approval gate).
    AwaitingUser,
    /// Explicitly paused, not awaiting user input.
    Paused,
    /// Finished successfully. Terminal.
    Completed,
    /// Finished with an error. Terminal.
    Failed,
    /// Cancelled before completion. Terminal.
    Cancelled,
    /// Interrupted by a process restart or `mark_interrupted`-style sweep.
    /// Terminal.
    Interrupted,
}

impl AgentRunStatus {
    /// Renders the status as the string stored in the `status` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::AwaitingUser => "awaiting_user",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }

    /// Parses a stored `status` string, defaulting to
    /// [`AgentRunStatus::Pending`] for any unrecognized value.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "running" => Self::Running,
            "awaiting_user" => Self::AwaitingUser,
            "paused" => Self::Paused,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            "interrupted" => Self::Interrupted,
            _ => Self::Pending,
        }
    }

    /// Returns `true` for a status that will never change again: `Completed`,
    /// `Failed`, `Cancelled`, or `Interrupted`.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Interrupted
        )
    }
}

/// Lifecycle status of a single [`WorkflowRun`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowRunStatus {
    /// Created but not yet started.
    Pending,
    /// Actively executing.
    Running,
    /// Finished successfully. Terminal.
    Completed,
    /// Finished with an error. Terminal.
    Failed,
    /// Cancelled before completion. Terminal.
    Cancelled,
    /// Interrupted by a process restart. Terminal.
    Interrupted,
}

impl WorkflowRunStatus {
    /// Renders the status as the string stored in the `status` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }

    /// Parses a stored `status` string, defaulting to
    /// [`WorkflowRunStatus::Pending`] for any unrecognized value.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "running" => Self::Running,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            "interrupted" => Self::Interrupted,
            _ => Self::Pending,
        }
    }

    /// Returns `true` for a status that will never change again: `Completed`,
    /// `Failed`, `Cancelled`, or `Interrupted`.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Interrupted
        )
    }
}

/// A persisted background agent run: a subagent, worker thread, background
/// agent, team member, or workflow child, as classified by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentRun {
    /// Caller-supplied stable identifier.
    pub id: String,
    /// What kind of run this is.
    pub kind: AgentRunKind,
    /// The run that spawned this one, if any.
    pub parent_run_id: Option<String>,
    /// The parent conversation thread this run is nested under, if any.
    pub parent_thread_id: Option<String>,
    /// Identifier of the agent definition driving this run.
    pub agent_id: Option<String>,
    /// Current lifecycle state.
    pub status: AgentRunStatus,
    /// Opaque reference to the prompt/instructions this run was given.
    pub prompt_ref: Option<String>,
    /// The worker thread id this run executes on, for `WorkerThread` runs.
    pub worker_thread_id: Option<String>,
    /// Filesystem path to a resumable checkpoint for this run, if persisted
    /// out-of-band from `checkpoint`.
    pub checkpoint_path: Option<String>,
    /// Inline resumable checkpoint state, if small enough to store directly.
    pub checkpoint: Option<Value>,
    /// Human-readable summary of the run's outcome, once known.
    pub summary: Option<String>,
    /// Error message, if the run ended in `Failed`.
    pub error: Option<String>,
    /// Arbitrary caller-supplied metadata.
    pub metadata: Value,
    /// Aggregated token/cost telemetry for this run, when joined from
    /// `run_telemetry`.
    pub telemetry: Option<RunTelemetry>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// When the run reached a terminal status, if it has.
    pub completed_at: Option<DateTime<Utc>>,
}

/// A persisted workflow run: durable state for a multi-phase workflow
/// definition, including a compare-and-swap driver lease.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowRun {
    /// Caller-supplied stable identifier.
    pub id: String,
    /// Identifier of the workflow definition being executed.
    pub definition_id: String,
    /// The parent conversation thread this workflow is nested under, if any.
    pub parent_thread_id: Option<String>,
    /// The workflow's input payload.
    pub input: Value,
    /// Per-phase state, keyed by phase name.
    pub phase_states: Value,
    /// Ids of [`AgentRun`]s spawned as children of this workflow.
    pub child_run_ids: Vec<String>,
    /// Current lifecycle state.
    pub status: WorkflowRunStatus,
    /// Human-readable summary of the workflow's outcome, once known.
    pub summary: Option<String>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// When the workflow reached a terminal status, if it has.
    pub completed_at: Option<DateTime<Utc>>,
    /// Monotonically increasing durable revision used for compare-and-swap
    /// workflow state transitions.
    pub revision: u64,
    /// The live workflow driver, when a driver currently owns this run.
    pub lease_owner: Option<String>,
    /// When the current driver lease may be taken over after a crash.
    pub lease_expires_at: Option<DateTime<Utc>>,
}

/// Result of atomically acquiring a workflow driver's lease.
///
/// A driver takes this lease before acting on a workflow so at most one
/// process is ever advancing a given run; see
/// `try_claim_workflow_run`.
#[derive(Debug, Clone, PartialEq, schemars::JsonSchema)]
pub enum WorkflowLeaseClaim {
    /// The lease was free (or expired) and is now held by the caller.
    Acquired(WorkflowRun),
    /// Another driver currently holds an unexpired lease.
    Busy(WorkflowRun),
    /// No workflow run matched the given id.
    Missing,
}

/// One entry in a run's append-only event log.
///
/// `sequence` is allocated by the insert itself (see `super::ops`) rather
/// than computed by the caller, so concurrent appenders cannot collide.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RunEvent {
    pub run_id: String,
    /// Per-run monotonically increasing sequence number; the primary key
    /// together with `run_id`.
    pub sequence: u64,
    /// Caller-defined event type discriminator.
    pub event_type: String,
    /// Arbitrary event payload.
    pub payload: Value,
    pub timestamp: DateTime<Utc>,
}

/// Aggregated token, cost, and status telemetry for a single run.
///
/// One row per `run_id`, updated in place as a run progresses (see
/// `super::ops::upsert_run_telemetry` for how partial updates are coalesced).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RunTelemetry {
    pub run_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cost_usd: f64,
    /// Total wall-clock time spent executing the run so far, in milliseconds.
    pub elapsed_ms: Option<u64>,
    /// Number of tool calls made so far.
    pub tool_count: u64,
    pub model: Option<String>,
    pub provider: Option<String>,
    /// Most recent error message, if any step of the run failed.
    pub error: Option<String>,
    pub updated_at: Option<DateTime<Utc>>,
}

/// Fields a caller supplies to create or update an [`AgentRun`].
///
/// `started_at` / `completed_at` are optional so an update can leave them
/// untouched; `super::ops` fills `started_at` with "now" on first insert.
#[derive(Debug, Clone, schemars::JsonSchema)]
pub struct AgentRunUpsert {
    pub id: String,
    pub kind: AgentRunKind,
    pub parent_run_id: Option<String>,
    pub parent_thread_id: Option<String>,
    pub agent_id: Option<String>,
    pub status: AgentRunStatus,
    pub prompt_ref: Option<String>,
    pub worker_thread_id: Option<String>,
    pub checkpoint_path: Option<String>,
    pub checkpoint: Option<Value>,
    pub summary: Option<String>,
    pub error: Option<String>,
    pub metadata: Value,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Fields a caller supplies to create or update a [`WorkflowRun`].
///
/// `started_at` / `completed_at` are optional so an update can leave them
/// untouched; `super::ops` fills `started_at` with "now" on first insert.
#[derive(Debug, Clone, schemars::JsonSchema)]
pub struct WorkflowRunUpsert {
    pub id: String,
    pub definition_id: String,
    pub parent_thread_id: Option<String>,
    pub input: Value,
    pub phase_states: Value,
    pub child_run_ids: Vec<String>,
    pub status: WorkflowRunStatus,
    pub summary: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Fields a caller supplies to append a new [`RunEvent`].
///
/// `sequence` and `timestamp` are not here: they are assigned by the insert
/// (see `super::ops::append_run_event`).
#[derive(Debug, Clone, schemars::JsonSchema)]
pub struct RunEventAppend {
    pub run_id: String,
    pub event_type: String,
    pub payload: Value,
}

/// Fields a caller supplies to update an [`RunTelemetry`] row.
///
/// Every counter is `Option` so a partial update (e.g. bumping only
/// `tool_count`) can leave the others alone; see `super::ops` for how the
/// upsert coalesces `None` against the existing stored value rather than a
/// column default.
#[derive(Debug, Clone, Default, schemars::JsonSchema)]
pub struct RunTelemetryUpsert {
    pub run_id: String,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    pub elapsed_ms: Option<u64>,
    pub tool_count: Option<u64>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub error: Option<String>,
}

/// Filter/pagination parameters for `super::ops::list_agent_runs`.
#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentRunListRequest {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub parent_run_id: Option<String>,
    #[serde(default)]
    pub parent_thread_id: Option<String>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
}

/// Response shape for `super::ops::list_agent_runs`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentRunListResponse {
    pub runs: Vec<AgentRun>,
    /// Number of runs in `runs` (not the total matching count).
    pub count: usize,
}

/// Filter/pagination parameters for `super::ops::list_workflow_runs`.
#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowRunListRequest {
    #[serde(default)]
    pub definition_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub parent_thread_id: Option<String>,
    /// `u64` to match the `TypeSchema::U64` the controller advertises (the RPC
    /// scalar-coercion layer only handles `U64`). Capped at 500 in `list_workflow_runs`.
    #[serde(default)]
    pub limit: Option<u64>,
    #[serde(default)]
    pub offset: Option<u64>,
}

/// Response shape for `super::ops::list_workflow_runs`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowRunListResponse {
    pub runs: Vec<WorkflowRun>,
    /// Number of runs in `runs` (not the total matching count).
    pub count: usize,
}

/// Filter/pagination parameters for `super::ops::list_recent_run_events`.
#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RunEventListRequest {
    pub run_id: String,
    /// Only return events with `sequence` strictly greater than this — the
    /// standard "give me what's new since my last poll" cursor.
    #[serde(default)]
    pub after_sequence: Option<u64>,
    #[serde(default)]
    pub limit: Option<u32>,
}

/// Response shape for `super::ops::list_recent_run_events`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RunEventListResponse {
    pub events: Vec<RunEvent>,
    /// Number of events in `events` (not the total matching count).
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Agent-team coordination (issue #3374)
// ---------------------------------------------------------------------------

/// Lifecycle of an agent team.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentTeamStatus {
    /// The team is coordinating tasks.
    Active,
    /// The team has finished and will not accept further task activity.
    Closed,
}

impl AgentTeamStatus {
    /// Renders the status as the string stored in the `status` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Closed => "closed",
        }
    }

    /// Parse a stored status string (named `parse`, not `from_str`, to match the
    /// run-ledger status-enum convention and avoid the `FromStr` clippy lint).
    pub fn parse(raw: &str) -> Self {
        match raw {
            "closed" => Self::Closed,
            _ => Self::Active,
        }
    }
}

/// Lifecycle of a single team member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentTeamMemberStatus {
    /// Registered but not yet running.
    Pending,
    /// Currently running a claimed task.
    Active,
    /// Registered and available, holding no claim.
    Idle,
    /// Shut down; will not take further tasks.
    Stopped,
}

impl AgentTeamMemberStatus {
    /// Renders the status as the string stored in the `member_status` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Idle => "idle",
            Self::Stopped => "stopped",
        }
    }

    /// Parses a stored `member_status` string, defaulting to
    /// [`AgentTeamMemberStatus::Pending`] for any unrecognized value.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "active" => Self::Active,
            "idle" => Self::Idle,
            "stopped" => Self::Stopped,
            _ => Self::Pending,
        }
    }
}

/// Lifecycle of a coordination task within a team.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentTeamTaskStatus {
    /// Not yet ready to be claimed (e.g. unmet dependencies).
    Todo,
    /// Dependencies satisfied; available to be claimed.
    Ready,
    /// Claimed by a member and being worked.
    InProgress,
    /// Blocked on something other than a dependency task.
    Blocked,
    /// Completed and passed its quality gate. Terminal.
    Done,
}

impl AgentTeamTaskStatus {
    /// Renders the status as the string stored in the `status` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Todo => "todo",
            Self::Ready => "ready",
            Self::InProgress => "in_progress",
            Self::Blocked => "blocked",
            Self::Done => "done",
        }
    }

    /// Parses a stored `status` string, defaulting to
    /// [`AgentTeamTaskStatus::Todo`] for any unrecognized value.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "ready" => Self::Ready,
            "in_progress" => Self::InProgress,
            "blocked" => Self::Blocked,
            "done" => Self::Done,
            _ => Self::Todo,
        }
    }
}

/// A persisted agent team: a lead agent coordinating a set of members against
/// a shared pool of tasks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentTeam {
    pub id: String,
    /// The parent conversation thread this team is nested under, if any.
    pub parent_thread_id: Option<String>,
    /// Identifier of the agent leading this team.
    pub lead_agent_id: String,
    pub status: AgentTeamStatus,
    /// Human-readable summary of the team's outcome, once known.
    pub summary: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// When the team's status transitioned to `Closed`, if it has.
    pub closed_at: Option<DateTime<Utc>>,
}

/// Fields a caller supplies to create or update an [`AgentTeam`].
#[derive(Debug, Clone, schemars::JsonSchema)]
pub struct AgentTeamUpsert {
    pub id: String,
    pub parent_thread_id: Option<String>,
    pub lead_agent_id: String,
    pub status: AgentTeamStatus,
    pub summary: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub closed_at: Option<DateTime<Utc>>,
}

/// A persisted member of an [`AgentTeam`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentTeamMember {
    pub id: String,
    pub team_id: String,
    /// Display name, unique within the team.
    pub name: String,
    /// Identifier of the agent definition driving this member.
    pub agent_id: Option<String>,
    pub member_status: AgentTeamMemberStatus,
    /// Id of the [`AgentTeamTask`] this member currently holds a claim on, if
    /// any.
    pub current_task_id: Option<String>,
    /// The worker thread id this member executes on, if any.
    pub worker_thread_id: Option<String>,
    /// Id of the [`AgentRun`] backing this member's current execution, if
    /// any.
    pub run_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Outcome of atomically reserving a team member for one task run.
#[derive(Debug, Clone, PartialEq, schemars::JsonSchema)]
pub enum AgentTeamMemberRunAdmission {
    /// A pending or idle member was atomically moved to active.
    Started(AgentTeamMember),
    /// The member already owns an active task.
    AlreadyActive(AgentTeamMember),
    /// The member exists but is stopped and cannot start a task.
    NotAdmissible(AgentTeamMember),
    /// No member matched the requested team and id.
    Missing,
}

/// Fields a caller supplies to create or update an [`AgentTeamMember`].
#[derive(Debug, Clone, schemars::JsonSchema)]
pub struct AgentTeamMemberUpsert {
    pub id: String,
    pub team_id: String,
    pub name: String,
    pub agent_id: Option<String>,
    pub member_status: AgentTeamMemberStatus,
    pub current_task_id: Option<String>,
    pub worker_thread_id: Option<String>,
    pub run_id: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
}

/// A persisted coordination task within an [`AgentTeam`].
///
/// `claimed_by_member_id` / `claim_token` are meaningful only while `status`
/// is `InProgress`; see `super::ops::claim_agent_team_task` and
/// `super::ops::complete_agent_team_task` for the claim/completion protocol
/// and why both fields are cleared whenever a task leaves that status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentTeamTask {
    pub id: String,
    pub team_id: String,
    pub title: String,
    pub objective: Option<String>,
    pub status: AgentTeamTaskStatus,
    /// Member responsible for the task's outcome, distinct from whoever
    /// currently claims it.
    pub owner_member_id: Option<String>,
    /// Member currently holding the claim, if `status` is `InProgress`.
    pub claimed_by_member_id: Option<String>,
    /// Opaque token identifying the current claim, checked by
    /// `complete_agent_team_task` / `release_agent_team_task` so a stale
    /// caller cannot act on a claim it no longer holds.
    pub claim_token: Option<String>,
    /// Ids of [`AgentTeamTask`]s that must be `Done` before this task can be
    /// claimed.
    pub depends_on: Vec<String>,
    /// Outcome of the task's quality gate (`"pending"`, `"passed"`,
    /// `"failed"`).
    pub gate_status: String,
    /// Human-readable reason(s) the gate failed, if it did.
    pub gate_reason: Option<String>,
    /// Evidence submitted across completion attempts. Accumulates rather than
    /// resets on a failed gate, so a retry after fixing one problem does not
    /// have to resubmit evidence already accepted.
    pub evidence: Vec<String>,
    /// Id of the [`AgentRun`] that produced this task, if it was created from
    /// one.
    pub source_run_id: Option<String>,
    /// Explicit sort position among sibling tasks.
    pub order_index: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Fields a caller supplies to create or update an [`AgentTeamTask`].
///
/// Does not include `claimed_by_member_id` / `claim_token`: those are managed
/// exclusively by the claim/completion/release operations in `super::ops`,
/// never by a direct upsert.
#[derive(Debug, Clone, schemars::JsonSchema)]
pub struct AgentTeamTaskUpsert {
    pub id: String,
    pub team_id: String,
    pub title: String,
    pub objective: Option<String>,
    pub status: AgentTeamTaskStatus,
    pub owner_member_id: Option<String>,
    pub depends_on: Vec<String>,
    pub gate_status: Option<String>,
    pub gate_reason: Option<String>,
    pub evidence: Vec<String>,
    pub source_run_id: Option<String>,
    pub order_index: i64,
    pub created_at: Option<DateTime<Utc>>,
}

/// Filter/pagination parameters for `super::ops::list_agent_teams`.
#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentTeamListRequest {
    #[serde(default)]
    pub parent_thread_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    /// `u64` to match the `TypeSchema::U64` the controller advertises (the RPC
    /// scalar-coercion layer only handles `U64`). Capped at 500 in
    /// `list_agent_teams`.
    #[serde(default)]
    pub limit: Option<u64>,
    #[serde(default)]
    pub offset: Option<u64>,
}

/// Response shape for `super::ops::list_agent_teams`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentTeamListResponse {
    pub teams: Vec<AgentTeam>,
    /// Total number of teams matching the request filters, before pagination.
    pub count: usize,
}

/// Outcome of an atomic claim attempt on a team task.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ClaimOutcome {
    /// The claim succeeded; carries the freshly-claimed task. Boxed to keep the
    /// enum small (the task payload dwarfs the other variants).
    Claimed(Box<AgentTeamTask>),
    /// Another member already holds the claim.
    AlreadyClaimed,
    /// One or more dependency tasks are not yet `done`.
    Blocked { unmet: Vec<String> },
    /// No task matched the given team + task id.
    UnknownTask,
}

/// Outcome of a completion attempt on a team task.
///
/// Completion gates a task's transition to `done` behind quality invariants
/// (dependencies done, claimer owns the task, evidence present when required).
/// A failed gate leaves the task `in_progress` with `gate_status = "failed"`
/// and the reasons recorded, so a teammate can fix and retry.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum CompletionOutcome {
    /// The task passed its quality gate and is now `done`. Boxed to keep the
    /// enum small (the task payload dwarfs the other variants).
    Completed(Box<AgentTeamTask>),
    /// One or more quality-gate invariants failed; carries human-readable
    /// reasons for each unmet invariant.
    GateFailed { reasons: Vec<String> },
    /// The task is not claimed by the completing member, or is not in progress.
    NotClaimed,
    /// No task matched the given team + task id.
    UnknownTask,
}

/// Result of [`RunLedgerStore::transition_agent_run_status_from`](super::RunLedgerStore::transition_agent_run_status_from).
#[derive(Debug, Clone, schemars::JsonSchema)]
pub enum RunTransition {
    /// The run was still in the expected status; it moved and the event (if
    /// any) was recorded. Carries the run as stored afterwards.
    Applied(AgentRun),
    /// The run had already left the expected status; nothing was written.
    /// Carries the run as it is now.
    StatusChanged(AgentRun),
    /// No run has this id.
    NotFound,
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
