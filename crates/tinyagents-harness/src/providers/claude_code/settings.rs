//! Persisted, self-contained Claude Code provider settings.
//!
//! The only Claude-Code-specific knob exposed to the user — full access
//! (`bypassPermissions` + full native toolset) vs the default `acceptEdits`
//! posture — lives in its own small JSON file rather than in a host's central
//! configuration. Keeping it module-local avoids threading a provider-only
//! flag through unrelated host configuration.
//!
//! The host chooses the directory passed to [`load`] and [`save`]. It should be
//! stable across launches and separate from the user's project root.
//!
//! The `OPENHUMAN_CLAUDE_CODE_PERMISSION_MODE` environment variable can
//! override this at the driver layer for debugging and power users.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// File name holding the persisted toggle, written into the directory the
/// caller passes.
const SETTINGS_FILE: &str = "claude_code_settings.json";

/// Persisted Claude Code provider settings. Defaults are the safe posture.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeCodeSettings {
    /// When true, Claude Code runs with `--permission-mode bypassPermissions`
    /// plus its complete native toolset (Bash/network/subagents). Default
    /// false → `acceptEdits` (auto-apply file edits, gate everything else).
    #[serde(default)]
    pub full_access: bool,
}

fn settings_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(SETTINGS_FILE)
}

/// Loads settings from the host-selected settings directory. A missing,
/// unreadable, or corrupt file yields safe defaults with full access disabled.
pub fn load(workspace_dir: &Path) -> ClaudeCodeSettings {
    let path = settings_path(workspace_dir);
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(
                "[claude-code][settings] corrupt {} ({e}); using safe defaults",
                path.display()
            );
            ClaudeCodeSettings::default()
        }),
        Err(e) => {
            tracing::debug!(
                "[claude-code][settings] no settings at {} ({e}); using defaults",
                path.display()
            );
            ClaudeCodeSettings::default()
        }
    }
}

/// Persist `settings` to `workspace_dir`, creating the directory if needed.
pub fn save(workspace_dir: &Path, settings: &ClaudeCodeSettings) -> std::io::Result<()> {
    let path = settings_path(workspace_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(settings).map_err(std::io::Error::other)?;
    std::fs::write(&path, json)?;
    tracing::debug!(
        "[claude-code][settings] saved full_access={} → {}",
        settings.full_access,
        path.display()
    );
    Ok(())
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
