//! Workspace prompt files: seeding a host's bundled default files into a
//! workspace directory, and injecting on-disk or inline content into a prompt
//! under a `###` heading with a character cap.
//!
//! The host owns the file contents, the names, and the character budget; this
//! module owns the on-disk contract and the rendered bytes:
//!
//! - [`sync_workspace_file`] writes a default file into the workspace and
//!   records the SHA-256 of that default in a `.{filename}.builtin-hash`
//!   sidecar ([`builtin_hash_sidecar_name`]). When the default changes it
//!   replaces the file only if the file still matches the default the sidecar
//!   recorded, so user edits survive releases.
//! - [`inject_inline_content`] and [`inject_workspace_file_capped`] render
//!   `### {label}\n\n{content}\n\n`, cut at `max_chars` characters with a
//!   marker that names the tool the model can use to read the whole file.

use std::fmt::Write;
use std::path::Path;

/// The sidecar file name that records the built-in hash for `filename`:
/// `.{filename}.builtin-hash`, a sibling of the file in the workspace.
#[must_use]
pub fn builtin_hash_sidecar_name(filename: &str) -> String {
    format!(".{filename}.builtin-hash")
}

/// The hash recorded in the `.{filename}.builtin-hash` sidecar: lowercase hex
/// SHA-256 of `content`'s UTF-8 bytes (64 characters).
///
/// SHA-256's output is fixed forever, unlike `DefaultHasher`, which a
/// toolchain bump may change and so make an edited file look untouched.
#[must_use]
pub fn builtin_hash(content: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(content.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Whether the on-disk `disk` content still matches the built-in recorded in
/// the `stored` sidecar.
///
/// Only a SHA-256 sidecar may authorize an overwrite. A legacy 16-hex
/// `DefaultHasher` sidecar is migrated by the caller, but is never trusted to
/// classify disk content as unedited: a collision or a changed `DefaultHasher`
/// implementation could otherwise overwrite a user's file.
fn matches_stored_builtin(disk: &str, stored: &str) -> bool {
    stored.len() == 64 && builtin_hash(disk) == stored
}

/// Ensure `workspace_dir/filename` is up to date with the host's
/// `default_content`.
///
/// On first install the file doesn't exist, so it is written. Every sync
/// stores the SHA-256 of `default_content` in the
/// [`builtin_hash_sidecar_name`] sidecar. When the default changes (the host
/// shipped a new one), the file is overwritten only if its current content
/// still hashes to the stored built-in, so prompt improvements ship
/// automatically while user edits between releases are preserved. An edited
/// file is left in place and the sidecar moves to the new hash, so the old
/// default is not re-compared on every sync.
///
/// Does nothing when `default_content` is empty (the host has no default for
/// this file) or when `workspace_dir` is not absolute: a relative path means
/// the caller has no workspace (typically `Path::new(".")` in a test
/// fixture), and seeding into it would drop the file and its sidecar into the
/// process's current directory.
///
/// Failures are logged, never returned: a failed file write logs a warning
/// and leaves the sidecar unchanged; a failed sidecar write is ignored.
pub fn sync_workspace_file(workspace_dir: &Path, filename: &str, default_content: &str) {
    if default_content.is_empty() {
        return;
    }

    if !workspace_dir.is_absolute() {
        tracing::debug!(
            "[workspace-sync] refusing to seed {filename} into non-absolute workspace {} \
             — no workspace configured",
            workspace_dir.display()
        );
        return;
    }

    let path = workspace_dir.join(filename);
    let hash_path = workspace_dir.join(builtin_hash_sidecar_name(filename));

    let current_hash = builtin_hash(default_content);

    // Read the last-written hash (if any).
    let stored_hash = std::fs::read_to_string(&hash_path).unwrap_or_default();
    let stored_hash = stored_hash.trim();

    if stored_hash == current_hash && path.exists() {
        // Built-in hasn't changed and file exists — nothing to do.
        return;
    }

    // Two safe cases to write the default:
    //   1. The file doesn't exist yet — first install.
    //   2. The file exists and its hash matches the stored built-in hash —
    //      the user hasn't edited it since it was last written.
    // Otherwise the file was hand-edited; keep it and only move the sidecar.
    let file_exists = path.exists();
    let user_unmodified = if file_exists {
        match std::fs::read_to_string(&path) {
            Ok(disk) => matches_stored_builtin(&disk, stored_hash),
            Err(_) => false,
        }
    } else {
        false
    };

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    if !file_exists || user_unmodified {
        if let Err(e) = std::fs::write(&path, default_content) {
            tracing::warn!("[agent:prompt] failed to write workspace file {filename}: {e}");
            return;
        }
        tracing::info!(
            "[agent:prompt] updated workspace file {filename} (builtin content changed)"
        );
    } else {
        tracing::info!(
            "[agent:prompt] keeping user-edited workspace file {filename} (builtin changed but disk contents diverge)"
        );
    }
    let _ = std::fs::write(&hash_path, &current_hash);
}

/// Inject `content` into `prompt` under a `### {label}` heading, capped at
/// `max_chars` characters (Unicode scalar values, not bytes).
///
/// Renders `### {label}\n\n` followed by the trimmed content and `\n\n`. When
/// the trimmed content is longer than `max_chars`, it is cut at that many
/// characters and followed by
/// ``\n\n[... truncated at {max_chars} chars — use `{read_tool}` for full file]\n\n``,
/// where `read_tool` is the host's name for the tool that reads a file.
///
/// Empty or whitespace-only content is silently skipped.
pub fn inject_inline_content(
    prompt: &mut String,
    label: &str,
    content: &str,
    max_chars: usize,
    read_tool: &str,
) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }
    let _ = writeln!(prompt, "### {label}\n");
    let truncated = if trimmed.chars().count() > max_chars {
        trimmed
            .char_indices()
            .nth(max_chars)
            .map(|(idx, _)| &trimmed[..idx])
            .unwrap_or(trimmed)
    } else {
        trimmed
    };
    prompt.push_str(truncated);
    if truncated.len() < trimmed.len() {
        let _ = writeln!(
            prompt,
            "\n\n[... truncated at {max_chars} chars — use `{read_tool}` for full file]\n"
        );
    } else {
        prompt.push_str("\n\n");
    }
}

/// Inject `workspace_dir/filename` into `prompt` under a `### {filename}`
/// heading, capped at `max_chars` characters.
///
/// The rendering is exactly [`inject_inline_content`] with `filename` as the
/// label. A missing or empty file is silently skipped, so callers can inject
/// optional files unconditionally; any other read error is logged at debug
/// level and skipped.
///
/// **KV-cache contract:** the output is a pure function of `filename`, the
/// file bytes at call time, `max_chars`, and `read_tool`. Read each file once
/// per session — re-reading mid-session can change the prompt prefix and
/// break an inference backend's prefix cache.
pub fn inject_workspace_file_capped(
    prompt: &mut String,
    workspace_dir: &Path,
    filename: &str,
    max_chars: usize,
    read_tool: &str,
) {
    let path = workspace_dir.join(filename);

    match std::fs::read_to_string(&path) {
        Ok(content) => inject_inline_content(prompt, filename, &content, max_chars, read_tool),
        Err(e) => match e.kind() {
            std::io::ErrorKind::NotFound => {
                // Missing optional files add no placeholder, keeping the
                // prompt focused.
            }
            _ => {
                tracing::debug!("[prompt] failed to read {}: {e}", path.display());
            }
        },
    }
}

#[cfg(test)]
#[path = "workspace_files_tests.rs"]
mod tests;
