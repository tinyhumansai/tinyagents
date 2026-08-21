//! Shared-workspace claim arbitration for parallel agent fan-out.
//!
//! [`map_reduce`](super::map_reduce) answers "run these N items with bounded
//! concurrency". It cannot answer whether running them concurrently is *safe*.
//! When fanned-out workers share one filesystem root, two of them mutating the
//! same file at the same time corrupts it silently — there is no error, just a
//! plausible-looking result built on a torn tree.
//!
//! This module owns that decision, and only that decision. Given a set of
//! [`WorkspaceClaim`]s it returns a [`DispatchPlan`] saying which workers may
//! run concurrently, which must be serialized, and which cannot be scheduled at
//! all. It performs no I/O, spawns nothing, and never renders a message.
//!
//! # The rule
//!
//! A worker is safe to run in parallel when it either has its own root
//! (`isolated`) or never writes (`!writes`). A worker that writes a *shared*
//! root must declare the paths it owns; it is then serialized, and its claim is
//! checked against every claim already granted.
//!
//! Claims are granted in **input order, first-writer-wins**. That ordering is
//! load-bearing: it makes the plan a pure function of the input, independent of
//! how quickly any worker happens to finish, so the same request always yields
//! the same rejection.
//!
//! # What stays with the caller
//!
//! The crate reports a [`ClaimConflict`] as data; it never decides what a host
//! does about one. Whether an unbounded write is a hard rejection or a warning,
//! and what sentence the user reads, are product decisions. Likewise the syntax
//! a claim arrives in: [`parse_relative_claim_paths`] takes the *body* of a
//! claim list, not whatever prefix or key a host wraps it in.

mod types;

pub use types::{
    ClaimConflict, ClaimPathError, DispatchMode, DispatchPlan, WorkspaceClaim,
};

use std::path::{Component, Path, PathBuf};

use crate::harness::tool::ToolSideEffects;

/// Parses a claim list into safe, sorted, deduplicated relative paths.
///
/// Entries are separated by commas or newlines and may carry a leading `-` or
/// `*` bullet. Empty entries are skipped, so a trailing separator is harmless.
///
/// # Errors
///
/// Returns [`ClaimPathError::Absolute`] for an absolute path and
/// [`ClaimPathError::Escaping`] for one containing `..` or a platform prefix —
/// either would let a claim reach outside the shared root it is meant to
/// partition.
pub fn parse_relative_claim_paths(spec: &str) -> Result<Vec<PathBuf>, ClaimPathError> {
    let mut paths = Vec::new();
    for raw in spec.split([',', '\n']) {
        let trimmed = raw.trim().trim_start_matches(['-', '*']).trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        if path.is_absolute() {
            return Err(ClaimPathError::Absolute {
                raw: trimmed.to_string(),
            });
        }
        if path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
        {
            return Err(ClaimPathError::Escaping {
                raw: trimmed.to_string(),
            });
        }
        paths.push(path);
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

/// `true` when two claimed paths cannot be held by different workers at once.
///
/// Comparison is **component-wise**, not textual: `src/a` and `src/ab` are
/// distinct, while `src/a` and `src/a/inner.rs` collide because one contains the
/// other.
pub fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

/// `true` when a tool's declared effects can mutate a shared workspace.
///
/// Reads the declaration rather than a name or a host permission enum, so any
/// host can derive a claim from the tool metadata it already publishes.
/// `read_only` wins over the mutating flags: a tool that declares both is taken
/// at its most conservative word.
pub fn writes_shared_workspace(effects: &ToolSideEffects) -> bool {
    if effects.read_only {
        return false;
    }
    effects.writes_files || effects.installs_dependencies || effects.destructive
}

/// Decides which workers may run concurrently against one shared workspace.
///
/// Claims are considered in input order; each writing worker's paths are checked
/// against those already granted, and on the first overlap that worker is
/// rejected (the earlier holder keeps its claim). Rejected workers get `None` in
/// [`DispatchPlan::modes`] and an entry in [`DispatchPlan::conflicts`].
pub fn plan_shared_workspace_dispatch(claims: &[WorkspaceClaim]) -> DispatchPlan {
    let mut plan = DispatchPlan {
        modes: Vec::with_capacity(claims.len()),
        conflicts: Vec::new(),
    };
    // Granted (path, worker_id) pairs, accumulated in input order.
    let mut granted: Vec<(PathBuf, String)> = Vec::new();

    for (index, claim) in claims.iter().enumerate() {
        if claim.isolated || !claim.writes {
            plan.modes.push(Some(DispatchMode::Parallel));
            continue;
        }

        if claim.paths.is_empty() {
            plan.modes.push(None);
            plan.conflicts.push((
                index,
                ClaimConflict::UnboundedWrite {
                    worker_id: claim.worker_id.clone(),
                },
            ));
            continue;
        }

        let overlap = claim.paths.iter().find_map(|path| {
            granted
                .iter()
                .find(|(claimed, _)| paths_overlap(path, claimed))
                .map(|(claimed, holder)| (claimed.clone(), holder.clone()))
        });

        match overlap {
            Some((path, other_worker_id)) => {
                plan.modes.push(None);
                plan.conflicts.push((
                    index,
                    ClaimConflict::Overlap {
                        worker_id: claim.worker_id.clone(),
                        other_worker_id,
                        path,
                    },
                ));
            }
            None => {
                for path in &claim.paths {
                    granted.push((path.clone(), claim.worker_id.clone()));
                }
                plan.modes.push(Some(DispatchMode::Serial));
            }
        }
    }

    plan
}

#[cfg(test)]
mod test;
