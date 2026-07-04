//! Path-gating policy for [`WorkspaceDescriptor`]: the `allows`/`enforce`
//! checks and the lexical path-normalization helpers they rely on.
//!
//! Split out of `workspace/types.rs`; kept separate from the plain type
//! definitions because this is where the fail-closed security guarantee
//! actually lives.

use std::path::{Path, PathBuf};

use crate::Result;
use crate::harness::tool::SandboxMode;
use crate::harness::workspace::types::WorkspaceDescriptor;

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
    /// filesystem call. Relative candidates and roots are first anchored to the
    /// current working directory so a relative path cannot use leading `..`
    /// components to spoof re-entry into a same-named sibling of the root. If
    /// the current directory cannot be read, the gate fails closed (`false`).
    pub fn allows(&self, path: &Path) -> bool {
        let Some(candidate) = anchored_normalize(path) else {
            return false;
        };
        std::iter::once(&self.root)
            .chain(self.trusted_roots.iter())
            .filter_map(|root| anchored_normalize(root))
            .any(|root| candidate.starts_with(&root))
    }

    /// Fail-closed path gate to call *before* a tool touches `path`: when the
    /// path is outside every allowed root, emits an
    /// [`AgentEvent::WorkspaceViolation`][crate::harness::events::AgentEvent::WorkspaceViolation]
    /// on `events` and returns a [`TinyAgentsError::Validation`] so the caller
    /// blocks the operation. Returns `Ok(())` when the path is allowed.
    pub fn enforce(&self, path: &Path, events: &crate::harness::events::EventSink) -> Result<()> {
        if self.allows(path) {
            return Ok(());
        }
        let rendered = path.display().to_string();
        events.emit(crate::harness::events::AgentEvent::WorkspaceViolation {
            path: rendered.clone(),
        });
        Err(crate::error::TinyAgentsError::Validation(format!(
            "path `{rendered}` is outside the allowed workspace roots"
        )))
    }
}

/// Anchors `path` to an absolute base (the current working directory when
/// relative) and lexically normalizes it. Returns `None` when a relative path
/// cannot be anchored because the current directory is unavailable, so callers
/// fail closed.
fn anchored_normalize(path: &Path) -> Option<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    Some(normalize(&absolute))
}

/// Lexically normalizes a path by resolving `.` and `..` components without
/// touching the filesystem.
///
/// A `..` only pops a preceding *named* segment; a `..` that would escape the
/// accumulated prefix (leading or after another `..`) is preserved rather than
/// discarded. Dropping such components would let a relative path like
/// `ws/../../ws/secret` collapse back onto `ws` and spoof re-entry into a
/// same-named sibling directory outside the workspace.
fn normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {
                    // At a filesystem root; `..` cannot go higher.
                }
                _ => out.push(Component::ParentDir),
            },
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}
