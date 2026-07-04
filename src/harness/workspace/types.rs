//! Workspace isolation and sandbox types.
//!
//! These are the SDK-owned, application-policy-neutral hooks agents use when
//! their tools run over real files or command executors: a
//! [`WorkspaceDescriptor`] tells a tool which filesystem root it may touch, and
//! a [`WorkspaceIsolation`] provider prepares and tears down per-agent
//! worktrees/sandboxes. TinyAgents does not own any concrete policy; it owns the
//! interface so parallel agents can be isolated consistently.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::harness::tool::SandboxMode;

/// Describes the isolated execution environment a tool is allowed to operate in.
///
/// A tool discovers its allowed root from this descriptor (via
/// [`ToolExecutionContext::workspace`][crate::harness::tool::ToolExecutionContext::workspace])
/// instead of reaching for an application global, and a policy engine can call
/// [`allows`](Self::allows) to block unsafe paths before execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceDescriptor {
    /// The primary root the agent/tool may read and write under.
    pub root: PathBuf,
    /// Additional roots the tool is explicitly trusted to touch.
    #[serde(default)]
    pub trusted_roots: Vec<PathBuf>,
    /// Identity of the policy that produced this descriptor (for audit).
    #[serde(default)]
    pub policy_id: String,
    /// How strictly the environment is sandboxed.
    #[serde(default)]
    pub sandbox: SandboxMode,
}

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
