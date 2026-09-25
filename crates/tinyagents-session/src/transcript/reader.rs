//! Transcript readers: the model-context replay ([`read_transcript`]), the
//! display projection ([`read_transcript_display`]), and the cheap
//! meta-only / last-usage scans used by thread summaries.

use super::jsonl::{
    LineKind, MetaLine, classify_line, display_message_from_line, message_from_line,
    meta_from_payload,
};
use super::legacy_md::read_transcript_legacy_md;
use super::types::TranscriptMessage;
use super::types::{
    CompactionMarker, DisplayRecord, DisplaySessionTranscript, MessageUsage, SessionTranscript,
    TranscriptMeta,
};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Decode records independently so a cut multibyte character in one record
/// cannot make every earlier and later turn unreadable.
fn read_jsonl_lines(path: &Path) -> Result<Vec<Option<String>>> {
    let raw =
        fs::read(path).with_context(|| format!("read transcript jsonl {}", path.display()))?;
    Ok(raw
        .split(|byte| *byte == b'\n')
        .map(|line| std::str::from_utf8(line).ok().map(str::to_owned))
        .collect())
}

/// Read a session transcript.
///
/// **Primary path**: reads the `.jsonl` source of truth.
/// **Fallback**: if the `.jsonl` does not exist but the legacy `.md` does
/// (migration path — old sessions), reads it via the legacy HTML-comment
/// parser and returns a `SessionTranscript` with default meta where the
/// `.md` format didn't track a field.
pub fn read_transcript(path: &Path) -> Result<SessionTranscript> {
    // Route by extension first: a legacy `.md` path (returned by
    // `find_latest_transcript` when only legacy files exist) must go to
    // the legacy parser, never to the JSONL parser.
    if path.extension().and_then(|s| s.to_str()) == Some("md") {
        tracing::debug!(
            "[transcript] reading legacy .md transcript: {}",
            path.display()
        );
        return read_transcript_legacy_md(path);
    }

    if path.exists() {
        read_transcript_jsonl(path)
    } else {
        // Fallback: try the .md sibling (legacy one-release compat).
        let md_path = path.with_extension("md");
        if md_path.exists() {
            tracing::debug!(
                "[transcript] .jsonl not found, falling back to legacy .md: {}",
                md_path.display()
            );
            read_transcript_legacy_md(&md_path)
        } else {
            // Neither exists — propagate the original jsonl error.
            read_transcript_jsonl(path)
        }
    }
}

/// Replays a `.jsonl` transcript into the model-context [`SessionTranscript`]:
/// message lines accumulate, a `compaction` record replaces the accumulator
/// wholesale, and `interrupted` partials are skipped since they never entered
/// the model's context.
fn read_transcript_jsonl(path: &Path) -> Result<SessionTranscript> {
    let lines = read_jsonl_lines(path)?;

    let mut meta: Option<TranscriptMeta> = None;
    let mut messages: Vec<TranscriptMessage> = Vec::new();
    let mut tools: Option<serde_json::Value> = None;
    let mut compactions_replayed = 0usize;
    let mut interrupted_skipped = 0usize;

    // Append-only log replay (Phase A): the first non-empty line MUST be the
    // `_meta` header; subsequent lines are messages, `compaction` records
    // (which *replace* the accumulated context), interrupted partials (skipped
    // for the model-context path), or refreshed `_meta` lines (last wins).
    let mut seen_first = false;
    for (line_no, line) in lines.iter().enumerate() {
        let Some(line) = line else {
            if !seen_first {
                anyhow::bail!(
                    "first non-empty line of {} is not valid UTF-8",
                    path.display()
                );
            }
            tracing::warn!(
                "[transcript] skipping invalid UTF-8 record line {} in {}",
                line_no + 1,
                path.display()
            );
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if !seen_first {
            seen_first = true;
            let ml: MetaLine = serde_json::from_str(line).map_err(|err| {
                anyhow::anyhow!(
                    "first non-empty line of {} (line {}) is not a valid _meta object: {err}",
                    path.display(),
                    line_no + 1,
                )
            })?;
            meta = Some(meta_from_payload(ml.meta));
            continue;
        }

        match classify_line(line) {
            Ok(LineKind::Meta(ml)) => {
                // Refreshed cumulative meta — last one wins.
                meta = Some(meta_from_payload(ml.meta));
            }
            Ok(LineKind::Compaction(cl)) => {
                // Reduction record: the reduced context REPLACES everything
                // accumulated so far, exactly reproducing the old full-rewrite.
                let replacement: Vec<TranscriptMessage> =
                    cl.replacement.into_iter().map(message_from_line).collect();
                tracing::debug!(
                    "[transcript] replay: compaction at line {} replaces {} accumulated message(s) with {} (request_id={:?}) in {}",
                    line_no + 1,
                    messages.len(),
                    replacement.len(),
                    cl.request_id,
                    path.display()
                );
                messages = replacement;
                compactions_replayed += 1;
            }
            Ok(LineKind::Tools(tl)) => {
                // Declarations the session was last sent with — last wins.
                tools = Some(tl.tools);
            }
            Ok(LineKind::Message(ml)) => {
                if ml.interrupted {
                    // Display-only partial — never part of the model context.
                    interrupted_skipped += 1;
                    tracing::debug!(
                        "[transcript] replay: skipping interrupted partial line {} (display only) in {}",
                        line_no + 1,
                        path.display()
                    );
                    continue;
                }
                messages.push(message_from_line(ml));
            }
            Err(err) => {
                tracing::warn!(
                    "[transcript] skipping malformed/unknown record line {} in {}: {err}",
                    line_no + 1,
                    path.display()
                );
            }
        }
    }

    let meta = meta.with_context(|| {
        format!(
            "missing _meta header line in jsonl transcript {}",
            path.display()
        )
    })?;

    tracing::debug!(
        "[transcript] loaded {} messages (jsonl, {} compaction(s) replayed, {} interrupted skipped) from {}",
        messages.len(),
        compactions_replayed,
        interrupted_skipped,
        path.display()
    );

    Ok(SessionTranscript {
        meta,
        messages,
        tools,
    })
}

/// Read a transcript for **display**: returns *every* record in file order,
/// including pre-compaction history, compaction markers, and interrupted
/// partials — the counterpart to the model-context [`read_transcript`], which
/// collapses the log into the reduced context.
///
/// `meta` reflects the newest `_meta` line (cumulative totals stay current).
pub fn read_transcript_display(path: &Path) -> Result<DisplaySessionTranscript> {
    let lines = read_jsonl_lines(path)?;

    let mut meta: Option<TranscriptMeta> = None;
    let mut records: Vec<DisplayRecord> = Vec::new();
    let mut seen_first = false;

    for (line_no, line) in lines.iter().enumerate() {
        let Some(line) = line else {
            if !seen_first {
                anyhow::bail!(
                    "first non-empty line of {} is not valid UTF-8",
                    path.display()
                );
            }
            tracing::warn!(
                "[transcript] display: skipping invalid UTF-8 record line {} in {}",
                line_no + 1,
                path.display()
            );
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !seen_first {
            seen_first = true;
            let ml: MetaLine = serde_json::from_str(line).map_err(|err| {
                anyhow::anyhow!(
                    "first non-empty line of {} (line {}) is not a valid _meta object: {err}",
                    path.display(),
                    line_no + 1,
                )
            })?;
            meta = Some(meta_from_payload(ml.meta));
            continue;
        }
        match classify_line(line) {
            Ok(LineKind::Meta(ml)) => meta = Some(meta_from_payload(ml.meta)),
            // Request state, not a displayable record.
            Ok(LineKind::Tools(_)) => {}
            Ok(LineKind::Compaction(cl)) => {
                let replacement = cl
                    .replacement
                    .into_iter()
                    .map(display_message_from_line)
                    .collect();
                records.push(DisplayRecord::Compaction(CompactionMarker {
                    replacement,
                    ts: cl.ts,
                    request_id: cl.request_id,
                }));
            }
            Ok(LineKind::Message(ml)) => {
                records.push(DisplayRecord::Message(Box::new(display_message_from_line(
                    ml,
                ))));
            }
            Err(err) => {
                tracing::warn!(
                    "[transcript] display: skipping malformed/unknown record line {} in {}: {err}",
                    line_no + 1,
                    path.display()
                );
            }
        }
    }

    let meta = meta.with_context(|| {
        format!(
            "missing _meta header line in jsonl transcript {}",
            path.display()
        )
    })?;

    tracing::debug!(
        "[transcript] display-loaded {} record(s) from {}",
        records.len(),
        path.display()
    );

    Ok(DisplaySessionTranscript { meta, records })
}

/// Parse the authoritative `_meta` of a root transcript JSONL.
///
/// Append-only files carry the immutable header on line 1 plus a refreshed
/// `_meta` line per turn (cumulative totals). The **last** `_meta` line wins,
/// so a multi-turn session reports its running totals — not just the first
/// turn's. Falls back to line 1 for legacy single-header files.
pub(super) fn read_transcript_meta_only(path: &Path) -> Option<TranscriptMeta> {
    let lines = read_jsonl_lines(path).ok()?;
    let mut latest: Option<TranscriptMeta> = None;
    for line in &lines {
        let Some(line) = line else {
            latest.as_ref()?;
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(ml) = serde_json::from_str::<MetaLine>(line) {
            latest = Some(meta_from_payload(ml.meta));
        } else if latest.is_none() {
            // The first non-empty line must be a valid meta header.
            return None;
        }
    }
    latest
}

/// Extract the last assistant message's usage + model from a transcript JSONL.
/// Only the final assistant message of a turn carries these (see the JSONL
/// format docs at the top of this module). Compaction records and refreshed
/// `_meta` lines are skipped; a `compaction` record's `replacement` assistant
/// rows are considered so a compacted transcript still surfaces its latest
/// usage.
pub(super) fn read_last_assistant_usage(path: &Path) -> Option<(MessageUsage, Option<String>)> {
    let lines = read_jsonl_lines(path).ok()?;
    let mut result = None;
    let mut seen_first = false;
    for line in &lines {
        let Some(line) = line else {
            if !seen_first {
                return None;
            }
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !seen_first {
            seen_first = true; // first non-empty line is the `_meta` header
            continue;
        }
        match classify_line(line) {
            Ok(LineKind::Message(ml)) if ml.role == "assistant" && !ml.interrupted => {
                if let Some(usage) = ml.usage {
                    result = Some((usage, ml.model));
                }
            }
            Ok(LineKind::Compaction(cl)) => {
                for ml in &cl.replacement {
                    if ml.role == "assistant"
                        && let Some(usage) = ml.usage.clone()
                    {
                        result = Some((usage, ml.model.clone()));
                    }
                }
            }
            _ => {}
        }
    }
    result
}
