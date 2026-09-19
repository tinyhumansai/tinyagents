//! Host-neutral subagent lifecycle orchestration.
//!
//! A host resolves its own policy, agent definition, prompt, tool allowlist,
//! and persistence implementation through the three object-safe seams exposed
//! here. This module only orders the lifecycle: resume loading, preparation,
//! execution, and one mutually exclusive persistence action. It never creates
//! a context, selects a model, interprets an artifact path, or applies host
//! policy.
//!
//! Dependency direction remains `orchestration -> runtime -> {harness,
//! session}`. In particular, lower TinyAgents layers and hosts must not depend
//! on this lifecycle module.

mod driver;
mod executor;
mod persistence;
mod planner;
mod types;

pub use driver::{SubagentCapabilities, SubagentDriver};
pub use executor::SubagentExecutor;
pub use persistence::SubagentPersistence;
pub use planner::SubagentPlanner;
pub use types::{
    ArtifactReference, PersistedSubagentPause, PreparedSubagent, SubagentError, SubagentExecution,
    SubagentIncomplete, SubagentOutcome, SubagentPause, SubagentPausePersistenceDisposition,
    SubagentPersistenceDisposition, SubagentRequest, SubagentRequestParts, SubagentResume,
    SubagentRunResult, SubagentStatus, SubagentTaskKey, SubagentTerminalPersistenceDisposition,
};

#[cfg(test)]
mod test;
