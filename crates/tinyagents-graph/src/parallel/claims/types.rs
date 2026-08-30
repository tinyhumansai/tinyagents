//! Type definitions for shared-workspace claim arbitration.

use std::path::PathBuf;

/// One worker's declared relationship to a fan-out's shared workspace.
///
/// A worker is safe to run concurrently with its siblings when it either has a
/// workspace of its own ([`isolated`](Self::isolated)) or never mutates the one
/// it shares ([`writes`](Self::writes) is `false`). Anything else needs an
/// explicit, disjoint set of owned [`paths`](Self::paths) before it can be
/// scheduled at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceClaim {
    /// Caller-chosen identifier, echoed back in conflicts so the caller can
    /// phrase its own diagnostics.
    pub worker_id: String,
    /// The worker runs in its own isolated root and can never collide.
    pub isolated: bool,
    /// The worker can mutate files in whatever root it runs in.
    pub writes: bool,
    /// Relative paths this worker declares exclusive ownership of. Empty means
    /// "unbounded" — a writing, non-isolated worker with no claim yields
    /// [`ClaimConflict::UnboundedWrite`].
    pub paths: Vec<PathBuf>,
}

impl WorkspaceClaim {
    /// A worker with its own workspace root. Never conflicts, always parallel.
    pub fn isolated(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            isolated: true,
            writes: false,
            paths: Vec::new(),
        }
    }

    /// A worker sharing the workspace but never mutating it.
    pub fn read_only(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            isolated: false,
            writes: false,
            paths: Vec::new(),
        }
    }

    /// A worker sharing the workspace and mutating the given owned paths.
    ///
    /// An empty `paths` is accepted here and reported as
    /// [`ClaimConflict::UnboundedWrite`] at planning time, so callers can build
    /// claims uniformly and let the planner do the rejecting.
    pub fn writing(worker_id: impl Into<String>, paths: Vec<PathBuf>) -> Self {
        Self {
            worker_id: worker_id.into(),
            isolated: false,
            writes: true,
            paths,
        }
    }
}

/// Why a claim string could not be parsed into safe relative paths.
///
/// Carries the offending input rather than a rendered sentence: the phrasing a
/// host shows its users is the host's, not this crate's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimPathError {
    /// The path was absolute; claims are always relative to the shared root.
    Absolute {
        /// The offending entry, as written.
        raw: String,
    },
    /// The path escaped the shared root via `..` or a platform path prefix.
    Escaping {
        /// The offending entry, as written.
        raw: String,
    },
}

/// Why a worker cannot be scheduled against the shared workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimConflict {
    /// The worker mutates a shared workspace without declaring what it owns.
    UnboundedWrite {
        /// The worker that could not be scheduled.
        worker_id: String,
    },
    /// The worker's claim overlaps one already granted to an earlier worker.
    Overlap {
        /// The worker that could not be scheduled.
        worker_id: String,
        /// The earlier worker holding the conflicting claim.
        other_worker_id: String,
        /// The path both workers claim.
        path: PathBuf,
    },
}

impl ClaimConflict {
    /// The worker this conflict rejected.
    pub fn worker_id(&self) -> &str {
        match self {
            Self::UnboundedWrite { worker_id } => worker_id,
            Self::Overlap { worker_id, .. } => worker_id,
        }
    }
}

/// How a worker must be dispatched relative to its siblings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    /// Safe to run concurrently with every other worker.
    Parallel,
    /// Mutates the shared workspace; must not overlap another writer in time.
    Serial,
}

/// Which workers may run concurrently, in input order.
///
/// [`modes`](Self::modes) is index-aligned with the planner's input. Rejected
/// workers appear in [`conflicts`](Self::conflicts) keyed by that same index and
/// are **not** represented in `modes`, so the two are read together.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DispatchPlan {
    /// Per-worker dispatch mode, indexed by input position. `None` marks a
    /// worker that was rejected and must not run.
    pub modes: Vec<Option<DispatchMode>>,
    /// Rejections, paired with the input index of the worker they rejected.
    pub conflicts: Vec<(usize, ClaimConflict)>,
}

impl DispatchPlan {
    /// `true` when at least one schedulable worker must run serially.
    pub fn has_serial_work(&self) -> bool {
        self.modes
            .iter()
            .any(|mode| matches!(mode, Some(DispatchMode::Serial)))
    }

    /// Input indices of the workers that may run concurrently.
    pub fn parallel_indices(&self) -> Vec<usize> {
        self.indices_matching(DispatchMode::Parallel)
    }

    /// Input indices of the workers that must run one at a time, in input order.
    pub fn serial_indices(&self) -> Vec<usize> {
        self.indices_matching(DispatchMode::Serial)
    }

    fn indices_matching(&self, wanted: DispatchMode) -> Vec<usize> {
        self.modes
            .iter()
            .enumerate()
            .filter_map(|(index, mode)| (*mode == Some(wanted)).then_some(index))
            .collect()
    }
}
