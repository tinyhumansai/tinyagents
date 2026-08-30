//! The fail-closed path gate for a [`WorkspaceDescriptor`].
//!
//! The descriptor and its lexical `allows` check live in `tinytools`, which
//! owns the tool vocabulary. What stays here is the half that needs this
//! crate: emitting a [`WorkspaceViolation`][crate::events::AgentEvent::WorkspaceViolation]
//! and returning this crate's error type. It is a free function rather than an
//! inherent method because the descriptor is now a foreign type.

use std::path::Path;

use tinytools::WorkspaceDescriptor;

use crate::Result;
use crate::events::{AgentEvent, EventSink};

/// Fail-closed path gate to call *before* a tool touches `path`.
///
/// When the path is outside every allowed root, emits a
/// [`AgentEvent::WorkspaceViolation`] on `events` and returns a validation
/// error so the caller blocks the operation. Returns `Ok(())` when the path is
/// allowed.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation]
/// when `path` lies outside the descriptor's root and trusted roots.
pub fn enforce_workspace_path(
    workspace: &WorkspaceDescriptor,
    path: &Path,
    events: &EventSink,
) -> Result<()> {
    if workspace.allows(path) {
        return Ok(());
    }
    let rendered = path.display().to_string();
    events.emit(AgentEvent::WorkspaceViolation {
        path: rendered.clone(),
    });
    Err(crate::error::TinyAgentsError::Validation(format!(
        "path `{rendered}` is outside the allowed workspace roots"
    )))
}
