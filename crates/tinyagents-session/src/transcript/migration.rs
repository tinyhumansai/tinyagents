//! One-shot migration of legacy date-grouped transcript layouts.
//!
//! Older hosts wrote transcripts beneath `session_raw/DDMMYYYY/` and markdown
//! companions beneath `sessions/DDMMYYYY/`. [`migrate_layout_if_needed`]
//! moves those artifacts into the flat transcript layout without overwriting a
//! file already present at the destination. Its workspace-local marker makes
//! completed migrations idempotent.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Marker file that signals the v1 session-layout migration has run for a
/// workspace. It lives under `state/migrations/` to keep the workspace root
/// tidy.
const MIGRATION_MARKER: &str = "state/migrations/session_layout_v1.done";

/// Counts and non-fatal diagnostics from a session-layout migration.
#[derive(Debug, Default, Clone)]
pub struct TranscriptLayoutMigration {
    /// Legacy JSONL transcripts moved into flat `session_raw/`.
    pub jsonl_moved: usize,
    /// JSONL files retained in the legacy directory because the destination existed.
    pub jsonl_skipped: usize,
    /// Markdown files moved into ISO-date directories.
    pub md_moved: usize,
    /// Markdown files retained in the legacy directory because the destination existed.
    pub md_skipped: usize,
    /// Empty legacy date directories removed after moving their contents.
    pub legacy_dirs_pruned: usize,
    /// Whether the migration marker was present before this invocation.
    pub already_done: bool,
    /// Non-fatal filesystem diagnostics encountered while migrating.
    pub warnings: Vec<String>,
}

/// Migrate a workspace's legacy transcript layout if it has not run before.
///
/// Detects `session_raw/{DDMMYYYY}/...jsonl` and `sessions/{DDMMYYYY}/...md`,
/// moves JSONL files into flat `session_raw/`, and moves markdown files into
/// `sessions/{YYYY_MM_DD}/`. Existing destination files are never
/// overwritten: they are retained in the legacy location and reported in
/// [`TranscriptLayoutMigration::warnings`]. A successful run writes a marker, including
/// when there were no legacy artifacts, so later starts perform no scan.
///
/// Individual filesystem failures are collected as warnings rather than
/// returned, allowing a host to continue startup and use legacy read fallback.
pub fn migrate_layout_if_needed(workspace_dir: &Path) -> Result<TranscriptLayoutMigration> {
    let marker_path = workspace_dir.join(MIGRATION_MARKER);
    if marker_path.exists() {
        log::debug!(
            "[session-migration] marker present at {} — skipping",
            marker_path.display()
        );
        return Ok(TranscriptLayoutMigration {
            already_done: true,
            ..Default::default()
        });
    }

    let mut outcome = TranscriptLayoutMigration::default();

    let raw_root = workspace_dir.join("session_raw");
    if raw_root.is_dir() {
        migrate_raw_jsonl(&raw_root, &mut outcome)?;
    }

    let sessions_root = workspace_dir.join("sessions");
    if sessions_root.is_dir() {
        migrate_md_directories(&sessions_root, &mut outcome)?;
    }

    write_marker(&marker_path, &outcome).context("write session-migration marker")?;

    log::info!(
        "[session-migration] complete: jsonl moved={} skipped={}, md moved={} skipped={}, legacy dirs pruned={}, warnings={}",
        outcome.jsonl_moved,
        outcome.jsonl_skipped,
        outcome.md_moved,
        outcome.md_skipped,
        outcome.legacy_dirs_pruned,
        outcome.warnings.len(),
    );

    Ok(outcome)
}

fn migrate_raw_jsonl(raw_root: &Path, outcome: &mut TranscriptLayoutMigration) -> Result<()> {
    let entries = match fs::read_dir(raw_root) {
        Ok(it) => it,
        Err(err) => {
            outcome
                .warnings
                .push(format!("read_dir({}) failed: {err}", raw_root.display()));
            return Ok(());
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !is_ddmmyyyy(name) {
            continue;
        }
        move_jsonl_files_up(&path, raw_root, outcome);
        prune_if_empty(&path, outcome);
    }

    Ok(())
}

fn move_jsonl_files_up(
    legacy_dir: &Path,
    flat_dir: &Path,
    outcome: &mut TranscriptLayoutMigration,
) {
    let entries = match fs::read_dir(legacy_dir) {
        Ok(it) => it,
        Err(err) => {
            outcome
                .warnings
                .push(format!("read_dir({}) failed: {err}", legacy_dir.display()));
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(file_name) = path.file_name() else {
            continue;
        };
        let dest = flat_dir.join(file_name);
        match hard_link_then_remove(&path, &dest) {
            Ok(()) => {
                outcome.jsonl_moved += 1;
                log::debug!(
                    "[session-migration] moved {} → {}",
                    path.display(),
                    dest.display()
                );
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                outcome.jsonl_skipped += 1;
                outcome.warnings.push(format!(
                    "skip move: destination already exists at {} (legacy file kept at {})",
                    dest.display(),
                    path.display()
                ));
            }
            Err(err) => outcome.warnings.push(format!(
                "atomic move({} → {}) failed: {err}",
                path.display(),
                dest.display()
            )),
        }
    }
}

fn migrate_md_directories(
    sessions_root: &Path,
    outcome: &mut TranscriptLayoutMigration,
) -> Result<()> {
    let entries = match fs::read_dir(sessions_root) {
        Ok(it) => it,
        Err(err) => {
            outcome.warnings.push(format!(
                "read_dir({}) failed: {err}",
                sessions_root.display()
            ));
            return Ok(());
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(iso) = ddmmyyyy_to_yyyy_mm_dd(name) else {
            continue;
        };
        let dest = sessions_root.join(&iso);
        match reserve_md_destination(&dest) {
            Ok(created) => {
                let moved_before = outcome.md_moved;
                if created {
                    log::debug!(
                        "[session-migration] reserved markdown destination {}",
                        dest.display()
                    );
                }
                merge_md_dirs(&path, &dest, outcome);
                prune_if_empty(&path, outcome);
                if created && !path.exists() {
                    // The legacy implementation moved this entire directory in
                    // one rename and recorded one move regardless of its file
                    // count. Keep that marker/report contract stable for a
                    // complete fresh-directory migration; merging into an
                    // existing destination remains per-file.
                    outcome.md_moved = moved_before + 1;
                }
            }
            Err(err) => outcome.warnings.push(format!(
                "reserve markdown destination {} failed: {err}",
                dest.display()
            )),
        }
    }

    Ok(())
}

/// Reserve a markdown destination directory without replacing a concurrent
/// creator's directory. `create_dir` is the atomic no-replace operation: an
/// `AlreadyExists` result means another process (or an earlier migration) owns
/// the directory and its files must be merged rather than replaced.
fn reserve_md_destination(destination: &Path) -> std::io::Result<bool> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::create_dir(destination) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(err) => Err(err),
    }
}

fn merge_md_dirs(legacy: &Path, dest: &Path, outcome: &mut TranscriptLayoutMigration) {
    let entries = match fs::read_dir(legacy) {
        Ok(it) => it,
        Err(err) => {
            outcome
                .warnings
                .push(format!("read_dir({}) failed: {err}", legacy.display()));
            return;
        }
    };
    for entry in entries.flatten() {
        let src = entry.path();
        if !src.is_file() {
            continue;
        }
        let Some(file_name) = src.file_name() else {
            continue;
        };
        let target = dest.join(file_name);
        match hard_link_then_remove(&src, &target) {
            Ok(()) => outcome.md_moved += 1,
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                outcome.md_skipped += 1;
                outcome.warnings.push(format!(
                    "skip md merge: {} already exists (legacy at {} kept)",
                    target.display(),
                    src.display()
                ));
            }
            Err(err) => outcome.warnings.push(format!(
                "atomic md move({} → {}) failed: {err}",
                src.display(),
                target.display()
            )),
        }
    }
}

/// Atomically create `destination` as a second link to `source`, then remove
/// `source`. `hard_link` refuses an existing destination, so a competing
/// writer can never be replaced between discovery and transfer.
fn hard_link_then_remove(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::hard_link(source, destination)?;
    fs::remove_file(source)
}

fn prune_if_empty(dir: &Path, outcome: &mut TranscriptLayoutMigration) {
    match fs::read_dir(dir) {
        Ok(mut it) => {
            if it.next().is_some() {
                return;
            }
        }
        Err(_) => return,
    }
    if fs::remove_dir(dir).is_ok() {
        outcome.legacy_dirs_pruned += 1;
    }
}

fn write_marker(marker_path: &Path, outcome: &TranscriptLayoutMigration) -> Result<()> {
    if let Some(parent) = marker_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create marker dir {}", parent.display()))?;
    }
    let body = format!(
        "openhuman session_layout migration v1\nrun_at: {}\njsonl_moved: {}\nmd_moved: {}\nlegacy_dirs_pruned: {}\nwarnings: {}\n",
        chrono::Utc::now().to_rfc3339(),
        outcome.jsonl_moved,
        outcome.md_moved,
        outcome.legacy_dirs_pruned,
        outcome.warnings.len(),
    );
    fs::write(marker_path, body)
        .with_context(|| format!("write marker {}", marker_path.display()))?;
    Ok(())
}

fn is_ddmmyyyy(name: &str) -> bool {
    name.len() == 8 && name.chars().all(|c| c.is_ascii_digit())
}

fn ddmmyyyy_to_yyyy_mm_dd(name: &str) -> Option<String> {
    if !is_ddmmyyyy(name) {
        return None;
    }
    let dd = &name[0..2];
    let mm = &name[2..4];
    let yyyy = &name[4..8];
    Some(format!("{yyyy}_{mm}_{dd}"))
}

/// Return the migration marker path for `workspace_dir`.
#[cfg(test)]
fn marker_path_for(workspace_dir: &Path) -> std::path::PathBuf {
    workspace_dir.join(MIGRATION_MARKER)
}

#[cfg(test)]
#[path = "migration_test.rs"]
mod tests;
