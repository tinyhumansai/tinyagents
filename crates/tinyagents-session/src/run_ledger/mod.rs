//! Durable run ledger for agent and workflow execution state.
//!
//! Extends [`super`] with a queryable, restart-survivable ledger for background
//! agent and workflow runs. Conversation transcripts remain in the session
//! store; this ledger holds compact run metadata, child lineage, events,
//! telemetry, and checkpoint references — enough for a host to reconstruct
//! what was in flight after a crash and resume or interrupt it.
//!
//! Shares the session database and connection helper with `super::store`, so
//! a run and the session that produced it are queryable together.

pub mod command_center;
pub mod drivers;
pub mod error;
pub mod history;
pub mod import;
pub mod inbox;
pub mod port;
pub mod recovery;
mod status_map;
pub mod tool_effects;
pub mod types;

pub use drivers::DriverRunLedger;
pub use error::{LedgerError, LedgerResult};
pub use history::{
    MAX_TOOL_OUTPUT_BYTES, NewSessionMessage, NewSessionToolCall, SessionEnd, SessionHistoryImport,
    SessionHistoryImportReport, SessionHistoryStore, SessionStart,
};
pub use import::{
    AgentFamilyImport, AgentFamilyImportReport, ImportCounts, LegacyLedgerRows,
    LegacyLedgerRowsReport,
};
pub use port::RunLedgerStore;

pub use recovery::{
    CallRecovery, DanglingToolCall, MissingEffectRow, RecoveryClass, classify_recovery,
    classify_recovery_with, overall_recovery,
};
pub use tool_effects::{
    RunLedgerToolEffects, ToolEffectRow, ToolEffectSettle, ToolEffectStart, ToolEffectStatus,
};
pub use types::{
    AgentRun, AgentRunKind, AgentRunListRequest, AgentRunListResponse, AgentRunStatus,
    AgentRunUpsert, AgentTeam, AgentTeamListRequest, AgentTeamListResponse, AgentTeamMember,
    AgentTeamMemberRunAdmission, AgentTeamMemberStatus, AgentTeamMemberUpsert, AgentTeamStatus,
    AgentTeamTask, AgentTeamTaskStatus, AgentTeamTaskUpsert, AgentTeamUpsert, ClaimOutcome,
    CompletionOutcome, RunEvent, RunEventAppend, RunEventListRequest, RunEventListResponse,
    RunTelemetry, RunTelemetryUpsert, RunTransition, WorkflowLeaseClaim, WorkflowRun,
    WorkflowRunListRequest, WorkflowRunListResponse, WorkflowRunStatus, WorkflowRunUpsert,
};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
