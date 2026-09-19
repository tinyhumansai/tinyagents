//! Transcript writers: the one-shot full rewrite, the append-only per-turn
//! delta (message tail or compaction record), interrupted partials, and the
//! derived `.md` companion.

use super::history::TranscriptPartial;
use super::jsonl::{
    COMPACTION_KIND, CompactionLine, MessageLine, build_message_line, meta_line_json,
    serialise_message_lines,
};
use super::markdown::render_markdown;
use super::paths::md_companion_path;
use super::types::TranscriptMessage;
use super::types::{TranscriptMeta, TurnUsage};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Write JSONL as source of truth **and** re-render the companion `.md`.
///
/// `jsonl_path` must end in `.jsonl`; the `.md` companion is derived by
/// swapping the extension. **Full rewrite** on every call — this is the
/// one-shot writer used by migrations, the sub-agent runners, and tests.
/// The incremental session-persistence path uses [`append_transcript_turn`]
/// instead, which never rewrites existing lines.
pub fn write_transcript(
    jsonl_path: &Path,
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
    last_assistant_turn_usage: Option<&TurnUsage>,
) -> Result<()> {
    if let Some(parent) = jsonl_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create transcript dir {}", parent.display()))?;
    }

    // ── JSONL ────────────────────────────────────────────────────────
    let mut jsonl_buf = String::new();
    jsonl_buf.push_str(&meta_line_json(meta)?);
    jsonl_buf.push('\n');
    serialise_message_lines(messages, last_assistant_turn_usage, None, &mut jsonl_buf)?;

    fs::write(jsonl_path, jsonl_buf.as_bytes())
        .with_context(|| format!("write transcript {}", jsonl_path.display()))?;

    log::debug!(
        "[transcript] wrote {} messages (jsonl, full rewrite) to {}",
        messages.len(),
        jsonl_path.display()
    );

    render_md_companion(jsonl_path, messages, meta, last_assistant_turn_usage);
    Ok(())
}

/// Append this turn's delta to an **append-only** transcript, never rewriting
/// existing lines.
///
/// `prev_persisted` is the logical message set the previous call left on disk
/// (empty on the first call for a fresh file). The incoming `messages` is the
/// current full logical set for this turn:
///
/// - **Pure extension** (`prev_persisted` is a prefix of `messages`): only the
///   new tail is appended as message lines.
/// - **Reduction / rewrite** (context reduction changed or dropped earlier
///   turns): a single `compaction` record carrying the full reduced
///   `messages` is appended; earlier lines are left untouched on disk.
///
/// A fresh `_meta` line is appended so cumulative totals stay current without a
/// full rewrite. The `.md` companion is re-rendered from `messages` (derived
/// view — always the reduced/current set). Returns nothing; the caller updates
/// its tracked `prev_persisted` to `messages` on success.
///
/// `request_id`, when supplied by the caller, is stamped on every appended
/// line as a turn boundary marker.
pub fn append_transcript_turn(
    jsonl_path: &Path,
    prev_persisted: &[TranscriptMessage],
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
    turn_usage: Option<&TurnUsage>,
    request_id: Option<&str>,
) -> Result<()> {
    append_transcript_turn_with_partial(
        jsonl_path,
        prev_persisted,
        messages,
        meta,
        turn_usage,
        request_id,
        None,
    )
}

/// Appends one logical turn and, when present, its display-only interruption
/// row from one serialized buffer and one file-write operation.
///
/// The partial is written after the logical delta and refreshed metadata. It
/// has `interrupted: true`, so the model-context reader skips it while the
/// display reader preserves it. Serialization happens before opening the file
/// for append, preventing a serialization failure from leaving a logical turn
/// without its associated display partial.
pub fn append_transcript_turn_with_partial(
    jsonl_path: &Path,
    prev_persisted: &[TranscriptMessage],
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
    turn_usage: Option<&TurnUsage>,
    request_id: Option<&str>,
    partial: Option<&TranscriptPartial>,
) -> Result<()> {
    if let Some(parent) = jsonl_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create transcript dir {}", parent.display()))?;
    }

    let file_exists = jsonl_path.exists();

    // First write for this file: create it with meta + all message lines.
    if !file_exists {
        let mut buf = String::new();
        buf.push_str(&meta_line_json(meta)?);
        buf.push('\n');
        serialise_message_lines(messages, turn_usage, request_id, &mut buf)?;
        serialise_interrupted_partial(partial, request_id, &mut buf)?;
        fs::write(jsonl_path, buf.as_bytes())
            .with_context(|| format!("create transcript {}", jsonl_path.display()))?;
        log::debug!(
            "[transcript] created append-only transcript with {} message(s) at {}",
            messages.len(),
            jsonl_path.display()
        );
        render_md_companion(jsonl_path, messages, meta, turn_usage);
        return Ok(());
    }

    // Subsequent writes: diff against the previously-persisted logical set.
    let common = common_prefix_len(prev_persisted, messages);
    let mut buf = String::new();

    if common == prev_persisted.len() {
        // Pure extension — append only the new tail.
        let tail = &messages[common..];
        log::debug!(
            "[transcript] append: extending on-disk set (prev={}, new={}, appending {} tail line(s)) {}",
            prev_persisted.len(),
            messages.len(),
            tail.len(),
            jsonl_path.display()
        );
        serialise_message_lines(tail, turn_usage, request_id, &mut buf)?;
    } else {
        // Reduction / rewrite — the on-disk set is no longer a prefix. Append a
        // compaction record carrying the full reduced context so the
        // model-context reader can replay it, without destroying earlier lines.
        log::debug!(
            "[transcript] append: context reduced (prev={}, new={}, common_prefix={}) — writing compaction record {}",
            prev_persisted.len(),
            messages.len(),
            common,
            jsonl_path.display()
        );
        let last_assistant_idx = messages.iter().rposition(|m| m.role == "assistant");
        let replacement: Vec<MessageLine> = messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let tu = if Some(i) == last_assistant_idx {
                    turn_usage.cloned().or_else(|| msg.turn_usage.clone())
                } else {
                    msg.turn_usage.clone()
                };
                build_message_line(msg, tu.as_ref(), request_id, false)
            })
            .collect();
        let compaction = CompactionLine {
            kind: COMPACTION_KIND.to_string(),
            replacement,
            ts: Some(chrono::Utc::now().to_rfc3339()),
            request_id: request_id.map(str::to_string),
            _extra: HashMap::new(),
        };
        let line = serde_json::to_string(&compaction).context("serialise compaction record")?;
        buf.push_str(&line);
        buf.push('\n');
    }

    // Refresh cumulative meta by appending a new `_meta` line (readers take the
    // last one). Keeps append-only + O(1)-per-turn (no full-file rewrite).
    buf.push_str(&meta_line_json(meta)?);
    buf.push('\n');
    serialise_interrupted_partial(partial, request_id, &mut buf)?;

    append_bytes(jsonl_path, buf.as_bytes())?;
    render_md_companion(jsonl_path, messages, meta, turn_usage);
    Ok(())
}

/// Appends the optional display-only row to an already assembled turn buffer.
fn serialise_interrupted_partial(
    partial: Option<&TranscriptPartial>,
    request_id: Option<&str>,
    buf: &mut String,
) -> Result<()> {
    let Some(partial) = partial.filter(|partial| !partial.content.is_empty()) else {
        return Ok(());
    };
    let mut line = build_message_line(
        &TranscriptMessage::assistant(&partial.content),
        None,
        request_id,
        true,
    );
    line.iteration = partial.iteration;
    line.reasoning_content = partial
        .reasoning_content
        .as_deref()
        .map(str::trim)
        .filter(|content| !content.is_empty())
        .map(str::to_owned);
    line.ts = Some(chrono::Utc::now().to_rfc3339());
    buf.push_str(&serde_json::to_string(&line).context("serialise interrupted partial line")?);
    buf.push('\n');
    Ok(())
}

/// Append a partial assistant answer, flagged `interrupted: true`, captured
/// when a streaming turn was cancelled/interrupted before completion.
///
/// **Display only**: the model-context reader skips interrupted lines, so a
/// resumed context never carries a truncated answer. Does not affect the
/// caller's tracked `prev_persisted` (nothing about the logical model context
/// changed). No-op when `partial_content` is empty.
pub fn append_interrupted_partial(
    jsonl_path: &Path,
    partial_content: &str,
    request_id: Option<&str>,
    iteration: Option<u32>,
    reasoning_content: Option<&str>,
) -> Result<()> {
    if partial_content.is_empty() {
        return Ok(());
    }
    if let Some(parent) = jsonl_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create transcript dir {}", parent.display()))?;
    }
    let mut buf = String::new();
    serialise_interrupted_partial(
        Some(&TranscriptPartial {
            content: partial_content.to_owned(),
            reasoning_content: reasoning_content.map(str::to_owned),
            iteration,
        }),
        request_id,
        &mut buf,
    )?;
    append_bytes(jsonl_path, buf.as_bytes())?;
    log::debug!(
        "[transcript] appended interrupted partial ({} chars, request_id={:?}) to {}",
        partial_content.len(),
        request_id,
        jsonl_path.display()
    );
    Ok(())
}

/// Longest common prefix length between two message slices, comparing on the
/// stable, serialised fields (`role`, `content`, `id`). `TranscriptMessage` does not
/// derive `PartialEq`, and `extra_metadata` is intentionally excluded because
/// it is enriched (turn usage) between the in-memory history and the persisted
/// line, which must not count as a divergence.
fn common_prefix_len(a: &[TranscriptMessage], b: &[TranscriptMessage]) -> usize {
    a.iter()
        .zip(b.iter())
        .take_while(|(x, y)| x.role == y.role && x.content == y.content && x.id == y.id)
        .count()
}

/// Append raw bytes to a file, opening in append mode (O(1), no read-back).
fn append_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open transcript for append {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("append transcript {}", path.display()))?;
    Ok(())
}

/// Re-render the derived `.md` companion from the current (reduced) message set.
///
/// Best-effort — the JSONL is the source of truth; a companion write failure is
/// logged and swallowed so it can never take down state persistence.
fn render_md_companion(
    jsonl_path: &Path,
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
    last_assistant_turn_usage: Option<&TurnUsage>,
) {
    let last_assistant_idx = messages.iter().rposition(|m| m.role == "assistant");
    let mut owned_usage: Vec<(usize, TurnUsage)> = Vec::new();
    for (idx, msg) in messages.iter().enumerate() {
        let usage = if Some(idx) == last_assistant_idx {
            last_assistant_turn_usage
                .cloned()
                .or_else(|| msg.turn_usage.clone())
        } else {
            msg.turn_usage.clone()
        };
        if let Some(usage) = usage {
            owned_usage.push((idx, usage));
        }
    }
    let per_msg_usage: HashMap<usize, &TurnUsage> = owned_usage
        .iter()
        .map(|(idx, usage)| (*idx, usage))
        .collect();

    let md_path = md_companion_path(jsonl_path);
    if let Some(parent) = md_path.parent()
        && let Err(err) = fs::create_dir_all(parent)
    {
        log::warn!(
            "[transcript] failed to create md companion dir {}: {err}",
            parent.display()
        );
        return;
    }
    let md = render_markdown(messages, meta, &per_msg_usage);
    if let Err(err) = fs::write(&md_path, md.as_bytes()) {
        log::warn!(
            "[transcript] failed to write markdown companion {}: {err}",
            md_path.display()
        );
        return;
    }
    log::debug!(
        "[transcript] wrote markdown companion to {}",
        md_path.display()
    );
}
