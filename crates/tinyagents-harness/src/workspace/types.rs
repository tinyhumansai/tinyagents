//! Workspace isolation and sandbox types.
//!
//! These are the SDK-owned, application-policy-neutral hooks agents use when
//! their tools run over real files or command executors: a
//! [`WorkspaceDescriptor`] tells a tool which filesystem root it may touch, and
//! a [`WorkspaceIsolation`] provider prepares and tears down per-agent
//! worktrees/sandboxes. TinyAgents does not own any concrete policy; it owns the
//! interface so parallel agents can be isolated consistently.

use async_trait::async_trait;

use crate::Result;

// The descriptor is tool vocabulary — it tells a tool which filesystem root it
// may touch — so it is defined in `tinytools` alongside the trait that reads
// it, and re-exported here at its historical path. `WorkspaceIsolation` stays:
// preparing and tearing down a worktree is harness work, and it returns this
// crate's `Result`.
pub use tinytools::WorkspaceDescriptor;

/// Prepares and tears down per-agent execution environments.
///
/// Implementations create a worktree/sandbox for one agent run and clean it up
/// afterward. The returned [`WorkspaceDescriptor`] is what the run threads into
/// tool execution contexts.
#[async_trait]
pub trait WorkspaceIsolation: Send + Sync {
    /// Prepares an environment for `run_id` (optionally on behalf of a named
    /// `agent`).
    async fn prepare(&self, run_id: &str, agent: Option<&str>) -> Result<WorkspaceDescriptor>;

    /// Cleans up a previously prepared environment.
    async fn cleanup(&self, descriptor: &WorkspaceDescriptor) -> Result<()>;
}
