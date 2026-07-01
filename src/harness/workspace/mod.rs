//! Workspace isolation and sandbox hooks for tools that run over real files or
//! command executors.
//!
//! See [`types`] for the [`WorkspaceDescriptor`] (the allowed-root policy a tool
//! reads from its execution context) and the [`WorkspaceIsolation`] provider
//! trait (per-agent environment preparation/cleanup). This module ships one
//! trivial provider, [`SharedRootWorkspace`], which scopes every agent to a
//! single shared root without copying — a sensible default and a test double.
//! Application-specific worktree/sandbox providers implement
//! [`WorkspaceIsolation`] themselves.

mod types;

pub use types::*;

use std::path::PathBuf;

use async_trait::async_trait;

use crate::Result;
use crate::harness::tool::SandboxMode;

/// A [`WorkspaceIsolation`] provider that scopes every agent to one shared root
/// without creating per-agent copies.
///
/// `prepare` returns a descriptor rooted at the shared directory (tagged with
/// the run id as the policy identity) and `cleanup` is a no-op, since nothing
/// per-agent was created.
#[derive(Clone, Debug)]
pub struct SharedRootWorkspace {
    root: PathBuf,
    sandbox: SandboxMode,
}

impl SharedRootWorkspace {
    /// Creates a provider scoping agents to `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            sandbox: SandboxMode::Inherit,
        }
    }

    /// Sets the sandbox mode advertised on prepared descriptors.
    pub fn with_sandbox(mut self, sandbox: SandboxMode) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[async_trait]
impl WorkspaceIsolation for SharedRootWorkspace {
    async fn prepare(&self, run_id: &str, _agent: Option<&str>) -> Result<WorkspaceDescriptor> {
        Ok(WorkspaceDescriptor::new(self.root.clone())
            .with_policy_id(run_id)
            .with_sandbox(self.sandbox))
    }

    async fn cleanup(&self, _descriptor: &WorkspaceDescriptor) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test;
