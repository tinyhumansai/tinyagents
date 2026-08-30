//! Path hardening for the artifact-offload convention.
//!
//! Everything an agent offloads resolves under
//! `<artifact root>/<outputs|workspace>`. The resolver is deliberately
//! fail-closed: it rejects absolute paths, `..` traversal, anything landing
//! outside its convention root after lexical normalization, and — when the host
//! supplies an [`ArtifactPathPolicy`] — anything reaching the host's internal
//! state.

use std::path::{Component, Path, PathBuf};

use super::policy::ArtifactPathPolicy;
use super::types::{ArtifactKind, OffloadError};

/// Maximum characters kept from a single path component when deriving a name
/// from an agent id / task id. Keeps generated names well inside filesystem
/// limits on every supported platform.
const MAX_COMPONENT_CHARS: usize = 80;

/// Reduce an arbitrary string to a safe single path component.
///
/// Anything that is not ASCII alphanumeric, `-`, or `_` becomes `_`, so a task
/// id like `sub-1a/2b` — or an agent id carrying a path separator — can never
/// introduce a directory level of its own.
pub fn sanitize_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len().min(MAX_COMPONENT_CHARS));
    for ch in value.chars().take(MAX_COMPONENT_CHARS) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

/// Lexically normalize `path` by dropping `.` components and popping a
/// directory for each `..`.
///
/// Purely lexical on purpose: the target usually does not exist yet, so
/// `canonicalize` is unavailable. [`resolve_artifact_path`] rejects `..`
/// outright before calling this; the popping here is defence in depth for any
/// future caller.
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Render `absolute` as a `root`-relative, `/`-separated path, so the string can
/// be pasted straight into a file-read call on any platform.
///
/// Falls back to the lossy display form when `absolute` is somehow not under
/// `root` — unreachable via [`resolve_artifact_path`], which validates
/// containment first, but reachable when a caller renders against a *different*
/// root than it wrote to (see `ArtifactOffload::with_render_root`). Falling back
/// to an absolute path is correct there: a wrong relative path would silently
/// resolve against the reader's own root and miss the file.
pub fn relative_to_root(root: &Path, absolute: &Path) -> String {
    match absolute.strip_prefix(root) {
        Ok(rel) => rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/"),
        Err(_) => absolute.to_string_lossy().to_string(),
    }
}

/// Resolve `relative` into an absolute offload target under
/// `root/<kind.subdir()>`, or explain why it is refused.
///
/// When `policy` is supplied the resolved path is additionally checked against
/// the host's internal state:
///
/// * anything [`ArtifactPathPolicy::is_internal_state`] flags is refused with
///   [`OffloadError::InternalState`];
/// * anything under [`ArtifactPathPolicy::internal_root`] is refused with
///   [`OffloadError::InternalRoot`].
///
/// The specific check runs **first** so its more precise error survives when a
/// host has configured the artifact root inside its internal root — reporting
/// the blanket containment failure there would lose the useful detail.
///
/// Passing `policy: None` skips only those two checks; traversal and
/// containment always run.
pub fn resolve_artifact_path(
    root: &Path,
    policy: Option<&dyn ArtifactPathPolicy>,
    kind: ArtifactKind,
    relative: &str,
) -> Result<PathBuf, OffloadError> {
    let trimmed = relative.trim();
    if trimmed.is_empty() {
        return Err(OffloadError::EmptyName);
    }

    let requested = Path::new(trimmed);
    let convention_root = root.join(kind.subdir());
    for component in requested.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(OffloadError::PathEscape {
                    root: convention_root.display().to_string(),
                    path: trimmed.to_string(),
                });
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(OffloadError::AbsolutePath {
                    path: trimmed.to_string(),
                });
            }
        }
    }

    let candidate = normalize_lexically(&convention_root.join(requested));
    // `normalize_lexically` cannot climb above the root given the `..`
    // rejection above, but assert containment anyway so a future relaxation of
    // that rejection cannot silently widen the write surface.
    if !candidate.starts_with(&convention_root) {
        return Err(OffloadError::PathEscape {
            root: convention_root.display().to_string(),
            path: candidate.display().to_string(),
        });
    }

    if let Some(policy) = policy {
        if policy.is_internal_state(&candidate) {
            return Err(OffloadError::InternalState {
                path: candidate.display().to_string(),
            });
        }
        if policy
            .internal_root()
            .is_some_and(|internal_root| candidate.starts_with(internal_root))
        {
            return Err(OffloadError::InternalRoot {
                path: candidate.display().to_string(),
            });
        }
    }

    Ok(candidate)
}
