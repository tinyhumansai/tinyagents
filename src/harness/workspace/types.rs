//! Workspace isolation and sandbox types.
//!
//! These are the SDK-owned, application-policy-neutral hooks agents use when
//! their tools run over real files or command executors: a
//! [`WorkspaceDescriptor`] tells a tool which filesystem root it may touch, and
//! a [`WorkspaceIsolation`] provider prepares and tears down per-agent
//! worktrees/sandboxes. TinyAgents does not own any concrete policy; it owns the
//! interface so parallel agents can be isolated consistently.

use std::path::{Path, PathBuf};

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

impl WorkspaceDescriptor {
    /// Creates a descriptor rooted at `root` with no extra trusted roots.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            trusted_roots: Vec::new(),
            policy_id: String::new(),
            sandbox: SandboxMode::Inherit,
        }
    }

    /// Adds a trusted root the tool may also touch.
    pub fn with_trusted_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.trusted_roots.push(root.into());
        self
    }

    /// Sets the audit policy identity.
    pub fn with_policy_id(mut self, id: impl Into<String>) -> Self {
        self.policy_id = id.into();
        self
    }

    /// Sets the sandbox mode.
    pub fn with_sandbox(mut self, sandbox: SandboxMode) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Returns `true` when `path` is contained within the root or any trusted
    /// root.
    ///
    /// Comparison is lexical (after normalizing `.`/`..` components) so it does
    /// not require the path to exist; it is a policy gate, not a canonicalizing
    /// filesystem call.
    pub fn allows(&self, path: &Path) -> bool {
        let candidate = normalize(path);
        std::iter::once(&self.root)
            .chain(self.trusted_roots.iter())
            .any(|root| candidate.starts_with(normalize(root)))
    }
}

/// Lexically normalizes a path by resolving `.` and `..` components without
/// touching the filesystem.
fn normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
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
