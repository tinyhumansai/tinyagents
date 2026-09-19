//! Transcript path resolution: the flat `session_raw/{stem}.jsonl` layout,
//! the dated `.md` companion, legacy date-grouped fallbacks, and the
//! newest-transcript scan used for resume.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Resolve a transcript path under `session_raw/{stem}.jsonl` — a
/// *flat* directory keyed only by stem. Used by the session-key flow:
/// the stem is `"{unix_ts}_{agent_id}"` for a root session, or
/// `"{parent_chain}__{session_key}"` for a sub-agent, so nested
/// delegations still produce a single flat filename that encodes the
/// parent → child path.
///
/// Creates the directory if needed. Overwrites are intentional: the
/// `Agent` persists the same transcript file across every turn of a
/// session, and every sub-agent spawn gets a unique timestamp in its
/// own key so collisions are effectively impossible.
pub fn resolve_keyed_transcript_path(workspace_dir: &Path, stem: &str) -> Result<PathBuf> {
    let raw_dir = raw_session_dir(workspace_dir);
    resolve_keyed_transcript_path_in_dir(&raw_dir, stem)
}

pub fn resolve_keyed_transcript_path_in_dir(raw_dir: &Path, stem: &str) -> Result<PathBuf> {
    fs::create_dir_all(raw_dir)
        .with_context(|| format!("create session_raw dir {}", raw_dir.display()))?;
    let sanitized = sanitize_stem(stem);
    Ok(raw_dir.join(format!("{sanitized}.jsonl")))
}

/// Sanitize a user-supplied transcript stem so it never escapes the
/// `session_raw/` directory. Allows ASCII alphanumerics plus a small
/// punctuation set (`_`, `-`, `.`); every other byte is replaced with
/// `_`. Empty inputs fall back to `"session"`.
fn sanitize_stem(stem: &str) -> String {
    let cleaned: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "session".to_string()
    } else {
        cleaned
    }
}

/// Find the most recent transcript for `agent_name`.
///
/// **Primary**: scan the flat `session_raw/` directory and pick the
/// newest matching stem (root sessions only — sub-agents are skipped).
/// **Fallback**: scan the legacy `session_raw/DDMMYYYY/` dirs (today
/// and yesterday) and the legacy `sessions/DDMMYYYY/` markdown dirs so
/// users upgrading from the date-grouped layout don't lose resume.
/// The fallback is one-release transitional and can be removed once
/// existing transcripts have rolled forward.
pub fn find_latest_transcript(workspace_dir: &Path, agent_name: &str) -> Option<PathBuf> {
    let sanitized = sanitize_agent_name(agent_name);
    let raw_root = raw_session_dir(workspace_dir);
    let sessions_root = workspace_dir.join("sessions");

    // Primary path: flat session_raw/ directory. The stem-suffix scan
    // is naturally date-independent, so an idle thread resumes the same
    // way today as it did weeks ago.
    if raw_root.is_dir()
        && let Some(path) = latest_in_dir(&raw_root, &sanitized)
    {
        return Some(path);
    }

    // Fallback: legacy date-grouped layout (one-release migration
    // window). Today first, then yesterday — matches the previous
    // behaviour so we don't regress while users still have files in
    // the old structure.
    let today = chrono::Local::now().format("%d%m%Y").to_string();
    let yesterday = (chrono::Local::now() - chrono::Duration::days(1))
        .format("%d%m%Y")
        .to_string();

    for date_str in [&today, &yesterday] {
        let raw_dir = raw_root.join(date_str);
        if raw_dir.is_dir()
            && let Some(path) = latest_in_dir(&raw_dir, &sanitized)
        {
            return Some(path);
        }
        let legacy_dir = sessions_root.join(date_str);
        if legacy_dir.is_dir()
            && let Some(path) = latest_in_dir(&legacy_dir, &sanitized)
        {
            return Some(path);
        }
    }

    None
}

/// Flat directory for the JSONL source of truth, e.g.
/// `{workspace}/session_raw`. Stems start with `{unix_ts}` so the
/// listing is naturally time-ordered without a date subdirectory.
pub(super) fn raw_session_dir(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join("session_raw")
}

/// Given a `session_raw/{stem}.jsonl` path, derive the companion
/// `sessions/YYYY_MM_DD/{stem}.md` path. The date is taken from the
/// local clock at write time — fine for browsing because the source
/// of truth lives in the flat raw dir; the `.md` is purely a view.
///
/// Legacy `session_raw/DDMMYYYY/{stem}.jsonl` paths (still on disk
/// from older releases until they roll forward) keep their date
/// component when generating the companion so we don't accidentally
/// stamp old transcripts with today's date.
///
/// If no `session_raw` component is present (tests using a flat
/// tempdir), the companion sits alongside as a sibling `.md`.
pub(super) fn md_companion_path(jsonl_path: &Path) -> PathBuf {
    let components: Vec<_> = jsonl_path.components().collect();

    let raw_idx = components
        .iter()
        .position(|comp| matches!(comp, std::path::Component::Normal(s) if *s == "session_raw"));

    let Some(raw_idx) = raw_idx else {
        return jsonl_path.with_extension("md");
    };

    let mut out = PathBuf::new();
    for comp in &components[..raw_idx] {
        out.push(comp.as_os_str());
    }
    out.push("sessions");

    // Tail after `session_raw`:
    //   * Flat: ["{stem}.jsonl"] — prepend today's YYYY_MM_DD.
    //   * Legacy: ["DDMMYYYY", "{stem}.jsonl"] — keep the existing
    //     date dir so we don't relabel old transcripts.
    let tail = &components[raw_idx + 1..];
    if tail.len() <= 1 {
        out.push(chrono::Local::now().format("%Y_%m_%d").to_string());
    }
    for comp in tail {
        out.push(comp.as_os_str());
    }

    out.with_extension("md")
}

pub(super) fn sanitize_agent_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Find the latest transcript file for `agent_prefix` in `dir`.
///
/// Prefers `.jsonl` files; falls back to `.md` if no `.jsonl` exists
/// (legacy sessions). When both exist for the same index the `.jsonl`
/// wins.
pub(super) fn latest_in_dir(dir: &Path, agent_prefix: &str) -> Option<PathBuf> {
    // Two transcript-naming schemes coexist on disk:
    //   * Legacy: `{agent}_{index}.jsonl|.md` — strictly increasing
    //     index, used by the now-removed `resolve_new_transcript_path`.
    //   * Keyed: `{unix_ts}_{agent}.jsonl` (root session) or
    //     `{parent_chain}__{unix_ts}_{agent}.jsonl` (sub-agent). The
    //     root stem starts with `{unix_ts}_{agent}` and has no `__`
    //     prefix segment.
    //
    // For resume we only care about root sessions (sub-agents rebuild
    // from scratch), so we scan for filenames matching either scheme
    // and pick the newest. "Newest" is the largest sort key — indices
    // and unix timestamps both order naturally as integers.
    let legacy_prefix = format!("{}_", agent_prefix);
    let keyed_suffix = format!("_{}", agent_prefix);
    let mut best_jsonl: Option<(u64, PathBuf)> = None;
    let mut best_md: Option<(u64, PathBuf)> = None;

    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Extract the stem minus extension.
        let (stem, is_jsonl) = if let Some(s) = name_str.strip_suffix(".jsonl") {
            (s, true)
        } else if let Some(s) = name_str.strip_suffix(".md") {
            (s, false)
        } else {
            continue;
        };
        // Skip sub-agent transcripts — they carry at least one `__`
        // separator in their stem (e.g.
        // `{orch_key}__{planner_key}`). Root resume never targets a
        // sub-agent's transcript directly.
        if stem.contains("__") {
            continue;
        }
        // Determine sort key. Keyed filenames end with
        // `_{agent_prefix}`: everything before that is the unix
        // timestamp. Legacy filenames start with `{agent_prefix}_`:
        // everything after is the numeric index.
        let sort_key: u64 = if let Some(ts_part) = stem.strip_suffix(&keyed_suffix) {
            match ts_part.parse::<u64>() {
                Ok(ts) => ts,
                Err(_) => continue,
            }
        } else if let Some(idx_part) = stem.strip_prefix(&legacy_prefix) {
            match idx_part.parse::<u64>() {
                Ok(idx) => idx,
                Err(_) => continue,
            }
        } else {
            continue;
        };
        let slot = if is_jsonl {
            &mut best_jsonl
        } else {
            &mut best_md
        };
        if slot.as_ref().is_none_or(|(best, _)| sort_key > *best) {
            *slot = Some((sort_key, entry.path()));
        }
    }

    // Prefer the best .jsonl; fall back to .md if no .jsonl exists.
    match (best_jsonl, best_md) {
        (Some(jsonl), Some(md)) => {
            // Take the one with the higher index; on a tie prefer .jsonl.
            if md.0 > jsonl.0 {
                Some(md.1)
            } else {
                Some(jsonl.1)
            }
        }
        (Some(jsonl), None) => Some(jsonl.1),
        (None, Some(md)) => Some(md.1),
        (None, None) => None,
    }
}
