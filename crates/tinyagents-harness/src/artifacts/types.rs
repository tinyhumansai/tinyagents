//! Inert value types for the artifact-offload convention.
//!
//! Dependency-free by design (std + `thiserror` only) so a host can build
//! against these without pulling the engine — RFC §2 rule 2.

use std::path::PathBuf;

/// Deliverables directory under the artifact root. Artifacts here are meant to
/// outlive the step that produced them and to be handed to a parent agent or a
/// later step **by path**, not by value.
pub const OUTPUTS_DIR: &str = "outputs";

/// Scratch directory under the artifact root. Intermediate files a worker needs
/// while it works but does not intend to hand back.
pub const SCRATCH_DIR: &str = "workspace";

/// Byte threshold above which a worker's final result is written to
/// [`OUTPUTS_DIR`] and replaced by a pointer + abstract.
///
/// ~2,000 tokens at the harness-wide 4-chars-per-token estimate. Below this,
/// inlining is cheaper than a file round-trip plus the pointer envelope.
pub const DEFAULT_OFFLOAD_THRESHOLD_BYTES: usize = 8_192;

/// Characters of the offloaded body reproduced as the abstract in the pointer.
/// Enough for a parent to decide whether to read the full artifact.
pub const ABSTRACT_BUDGET_CHARS: usize = 600;

/// Line prefix every pointer line carries. Grep-friendly, and the anchor
/// [`extract_artifact_paths`](super::extract_artifact_paths) keys off when
/// reading a handoff.
pub const ARTIFACT_POINTER_PREFIX: &str = "[artifact]";

/// Which of the two convention directories an artifact belongs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    /// A deliverable, written under [`OUTPUTS_DIR`].
    Output,
    /// Scratch, written under [`SCRATCH_DIR`].
    ///
    /// Note this is the artifact root's own `workspace` subdirectory, which is
    /// **not** a host's internal state root. The resolver refuses to place an
    /// artifact in the latter regardless of kind.
    Scratch,
}

impl ArtifactKind {
    /// Directory name under the artifact root for this kind.
    pub fn subdir(self) -> &'static str {
        match self {
            Self::Output => OUTPUTS_DIR,
            Self::Scratch => SCRATCH_DIR,
        }
    }

    /// Stable, log-friendly label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Output => "output",
            Self::Scratch => "scratch",
        }
    }
}

/// A worker artifact that was successfully written to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffloadedArtifact {
    /// Which convention directory it landed in.
    pub kind: ArtifactKind,
    /// Path relative to the render root, always `/`-separated so it can be
    /// pasted straight into a file-read call on any platform.
    pub relative_path: String,
    /// Absolute path on disk.
    pub absolute_path: PathBuf,
    /// Bytes actually stored (post-redaction).
    pub stored_bytes: usize,
    /// Bytes of the caller's original payload (pre-redaction).
    pub original_bytes: usize,
    /// Whether the host redactor rewrote the body before storage.
    pub redacted: bool,
}

/// Why an offload was refused.
///
/// Every variant is non-fatal at the call site: the caller keeps its inline
/// payload and whatever summarisation or truncation backstop it already had
/// still applies. An offload failure must never fail a turn.
#[derive(Debug, thiserror::Error)]
pub enum OffloadError {
    /// The requested relative path was empty or whitespace-only.
    #[error("artifact name is empty")]
    EmptyName,

    /// The relative path was absolute (or carried a Windows drive/UNC prefix).
    #[error("artifact path must be relative to the artifact root, got {path}")]
    AbsolutePath {
        /// The offending path, as supplied.
        path: String,
    },

    /// The relative path escaped its convention directory (`..` traversal).
    #[error("artifact path escapes {root}: {path}")]
    PathEscape {
        /// The convention root the path was resolved against.
        root: String,
        /// The offending path.
        path: String,
    },

    /// The resolved path landed inside the host's internal state root.
    ///
    /// Fail-closed: offload targets resolve under the artifact root, never a
    /// host's internal state.
    #[error(
        "artifact path resolves inside the host internal root, which agent writes may never reach: {path}"
    )]
    InternalRoot {
        /// The offending path.
        path: String,
    },

    /// The resolved path is a host-internal state location per
    /// [`ArtifactPathPolicy::is_internal_state`](super::ArtifactPathPolicy::is_internal_state).
    #[error("artifact path is host-internal state: {path}")]
    InternalState {
        /// The offending path.
        path: String,
    },

    /// The parent directory that actually materialised on disk resolves,
    /// through symlinks, to somewhere outside the convention root.
    ///
    /// The lexical checks in
    /// [`resolve_artifact_path`](super::resolve_artifact_path) cannot see this,
    /// because the target does not exist yet when they run.
    #[error("artifact parent escapes its root through a symlink: {path} resolves to {resolved}")]
    SymlinkEscape {
        /// The parent directory as requested.
        path: String,
        /// Where it actually resolves to.
        resolved: String,
    },

    /// Creating the parent directory or writing the file failed.
    #[error("artifact write failed: {0}")]
    Io(#[from] std::io::Error),
}
