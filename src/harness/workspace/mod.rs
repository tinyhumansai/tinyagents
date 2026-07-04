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

mod policy;
mod types;

pub use types::*;

use std::path::PathBuf;

use async_trait::async_trait;

use crate::Result;
use crate::harness::events::{AgentEvent, EventSink};
use crate::harness::tool::SandboxMode;

/// Prepares a per-agent environment through `isolation` and emits an
/// [`AgentEvent::WorkspacePrepared`] on the run's event sink so late observers
/// and journals record the isolation setup. Returns the descriptor to thread
/// into the run via [`RunContext::with_workspace`][crate::harness::context::RunContext::with_workspace].
///
/// A preparation failure is propagated to the caller (there is no partial
/// environment to clean up); the paired teardown is
/// [`cleanup_workspace`].
pub async fn prepare_workspace(
    isolation: &dyn WorkspaceIsolation,
    events: &EventSink,
    run_id: &str,
    agent: Option<&str>,
) -> Result<WorkspaceDescriptor> {
    let descriptor = isolation.prepare(run_id, agent).await?;
    events.emit(AgentEvent::WorkspacePrepared {
        policy_id: descriptor.policy_id.clone(),
        root: descriptor.root.display().to_string(),
    });
    Ok(descriptor)
}

/// Tears down a previously prepared environment through `isolation` and emits an
/// [`AgentEvent::WorkspaceCleanup`] (with `error` set when cleanup fails) so the
/// teardown is observable. The cleanup result is returned unchanged.
pub async fn cleanup_workspace(
    isolation: &dyn WorkspaceIsolation,
    events: &EventSink,
    descriptor: &WorkspaceDescriptor,
) -> Result<()> {
    let result = isolation.cleanup(descriptor).await;
    events.emit(AgentEvent::WorkspaceCleanup {
        policy_id: descriptor.policy_id.clone(),
        error: result.as_ref().err().map(|e| e.to_string()),
    });
    result
}

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
