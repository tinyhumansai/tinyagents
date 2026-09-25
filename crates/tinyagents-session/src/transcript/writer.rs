//! Transcript writers: the one-shot full rewrite, the append-only per-turn
//! delta (message tail or compaction record), interrupted partials, and the
//! derived `.md` companion.

use super::history::TranscriptPartial;
use super::jsonl::{
    COMPACTION_KIND, CompactionLine, MessageLine, build_message_line, meta_line_json,
    serialise_message_lines, tools_line_json,
};
use super::markdown::render_markdown;
use super::paths::md_companion_path;
use super::types::TranscriptMessage;
use super::types::{TranscriptMeta, TurnUsage};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

    atomic_write(jsonl_path, jsonl_buf.as_bytes())
        .with_context(|| format!("write transcript {}", jsonl_path.display()))?;

    tracing::debug!(
        "[transcript] wrote {} messages (jsonl, full rewrite) to {}",
        messages.len(),
        jsonl_path.display()
    );

    render_md_companion(jsonl_path, messages, meta, last_assistant_turn_usage);
    Ok(())
}

/// Like [`write_transcript`], but never overwrites an existing destination —
/// see [`publish_transcript_if_absent`] for why adoption needs that instead
/// of the full-rewrite semantics every other `write_transcript` caller
/// wants. Returns `Ok(true)` when this call created `jsonl_path`, `Ok(false)`
/// when it already existed (some other write already won the race and this
/// call's `messages`/`meta` were discarded).
pub fn write_transcript_if_absent(
    jsonl_path: &Path,
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
) -> Result<bool> {
    write_transcript_if_absent_with_tools(jsonl_path, messages, meta, None)
}

/// Like [`write_transcript_if_absent`], additionally preserving the latest
/// recorded tool declarations when a transcript is migrated into a session.
pub fn write_transcript_if_absent_with_tools(
    jsonl_path: &Path,
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
    tools: Option<&serde_json::Value>,
) -> Result<bool> {
    if let Some(parent) = jsonl_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create transcript dir {}", parent.display()))?;
    }

    let mut jsonl_buf = String::new();
    jsonl_buf.push_str(&meta_line_json(meta)?);
    jsonl_buf.push('\n');
    serialise_message_lines(messages, None, None, &mut jsonl_buf)?;
    if let Some(tools) = tools {
        jsonl_buf.push_str(&tools_line_json(tools)?);
        jsonl_buf.push('\n');
    }

    let published = publish_transcript_if_absent(jsonl_path, jsonl_buf.as_bytes())
        .with_context(|| format!("publish transcript {}", jsonl_path.display()))?;

    if published {
        tracing::debug!(
            "[transcript] published {} messages (jsonl, create-if-absent) to {}",
            messages.len(),
            jsonl_path.display()
        );
        render_md_companion(jsonl_path, messages, meta, None);
    } else {
        tracing::debug!(
            "[transcript] create-if-absent lost the race, {} already exists",
            jsonl_path.display()
        );
    }
    Ok(published)
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
    append_transcript_turn_with_extras(
        jsonl_path,
        prev_persisted,
        messages,
        meta,
        turn_usage,
        request_id,
        AppendTranscriptExtras {
            partial: None,
            tools: None,
        },
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
    append_transcript_turn_with_extras(
        jsonl_path,
        prev_persisted,
        messages,
        meta,
        turn_usage,
        request_id,
        AppendTranscriptExtras {
            partial,
            tools: None,
        },
    )
}

/// Optional turn records serialized alongside a turn's logical delta.
pub(crate) struct AppendTranscriptExtras<'a> {
    pub partial: Option<&'a TranscriptPartial>,
    pub tools: Option<&'a serde_json::Value>,
}

/// Appends a turn, optional display partial, and optional tool declarations
/// from one serialized buffer and one file-write operation.
pub(crate) fn append_transcript_turn_with_extras(
    jsonl_path: &Path,
    prev_persisted: &[TranscriptMessage],
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
    turn_usage: Option<&TurnUsage>,
    request_id: Option<&str>,
    extras: AppendTranscriptExtras<'_>,
) -> Result<()> {
    let AppendTranscriptExtras { partial, tools } = extras;
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
        if let Some(tools) = tools {
            buf.push_str(&tools_line_json(tools)?);
            buf.push('\n');
        }
        // A direct write can publish a partial _meta header if the process
        // stops mid-write. Such a file exists but cannot be resumed at all.
        // Publish the complete first generation only after staging it, and
        // never replace a file another writer created meanwhile.
        anyhow::ensure!(
            publish_transcript_if_absent(jsonl_path, buf.as_bytes())?,
            "transcript appeared during initial creation: {}",
            jsonl_path.display()
        );
        tracing::debug!(
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
        tracing::debug!(
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
        tracing::debug!(
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
    if let Some(tools) = tools {
        buf.push_str(&tools_line_json(tools)?);
        buf.push('\n');
    }

    append_bytes(jsonl_path, buf.as_bytes())?;
    render_md_companion(jsonl_path, messages, meta, turn_usage);
    Ok(())
}

/// Appends a `{"kind":"tools"}` record naming the tool declarations the
/// session's latest turn was sent with. The file must already exist (its
/// first line is always `_meta`), so callers append this after the turn.
pub fn append_tools_record(jsonl_path: &Path, tools: &serde_json::Value) -> Result<()> {
    anyhow::ensure!(
        jsonl_path.is_file(),
        "transcript must exist before appending tool declarations: {}",
        jsonl_path.display()
    );
    let mut line = tools_line_json(tools)?;
    line.push('\n');
    append_bytes(jsonl_path, line.as_bytes())?;
    tracing::debug!(
        "[transcript] recorded tool declarations ({} bytes) in {}",
        line.len(),
        jsonl_path.display()
    );
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
    tracing::debug!(
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

/// A same-directory temp path for `path`, unique per call within this
/// process. Shared by [`atomic_write`] and [`publish_transcript_if_absent`],
/// both of which stage full contents in a temp file before publishing it
/// with one atomic filesystem operation.
fn unique_tmp_path(path: &Path) -> PathBuf {
    static NONCE: AtomicU64 = AtomicU64::new(0);

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("transcript");
    let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()))
}

/// Writes `contents` to `path` via a same-directory temp file and an atomic
/// rename, rather than truncating `path` in place.
///
/// [`write_transcript`] is a full rewrite of the source-of-truth JSONL — used
/// directly by migrations, sub-agent runners, and (through
/// [`publish_transcript_if_absent`]) adoption. A plain `fs::write` truncates
/// the destination before the new bytes land, so a process or filesystem
/// failure partway through leaves a truncated file that nonetheless
/// satisfies `path.exists()`. Writing to a temp file first and renaming it
/// into place means the destination only ever transitions from "absent"
/// straight to "complete"; there is no truncated intermediate state a crash
/// can strand callers on. `fs::rename` always **replaces** an existing
/// destination on both Unix and Windows (`MoveFileExW` with
/// `MOVEFILE_REPLACE_EXISTING`, with a `SetFileInformationByHandle` fallback
/// — see the `std::fs::rename` docs), which is exactly the full-rewrite
/// semantics this function's other callers want.
/// Writes `contents` to a fresh temp file at `tmp_path`, refusing to follow
/// (and so overwrite the target of) any pre-existing filesystem entry —
/// including a symlink — already at that path.
///
/// `unique_tmp_path` mints a name unique to this process and call, so this
/// should never race with a legitimate temp file of ours; a party able to
/// pre-create an entry at the exact predicted name is exactly the case this
/// guards against. `fs::write` alone would instead follow a pre-planted
/// symlink and write our transcript content through it into whatever the
/// symlink points at — a party with write access to this directory could aim
/// that at a file elsewhere the process can write but should not overwrite.
/// `create_new` fails instead, atomically, without ever opening whatever was
/// really there.
fn write_temp_file(tmp_path: &Path, contents: &[u8]) -> Result<()> {
    use std::io::Write;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(tmp_path)
        .with_context(|| format!("create temp transcript {}", tmp_path.display()))?;
    file.write_all(contents)
        .with_context(|| format!("write temp transcript {}", tmp_path.display()))?;
    Ok(())
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let tmp_path = unique_tmp_path(path);

    if let Err(error) = write_temp_file(&tmp_path, contents) {
        // A failure after `create_new` succeeded (a full disk, a signal
        // interruption partway through `write_all`) leaves the temp file
        // behind; nothing else ever looks for or cleans up a name only this
        // call ever mints, so it would otherwise orphan forever. A failure
        // from `create_new` itself (the guarded case above) means there is
        // nothing of ours to clean up.
        let _ = fs::remove_file(&tmp_path);
        return Err(error);
    }
    fs::rename(&tmp_path, path).with_context(|| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "rename temp transcript {} to {}",
            tmp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

/// Publishes `contents` at `path` only if nothing is there yet, atomically.
///
/// Unlike [`atomic_write`] (and [`write_transcript`], which uses it),
/// **never overwrites an existing destination**. `fs::rename` cannot express
/// "fail if the destination exists" — as documented on [`atomic_write`], it
/// always replaces on both platforms — so this uses `fs::hard_link` instead,
/// which fails with `AlreadyExists` without touching whatever is already at
/// `path`. That failure is reported by returning `Ok(false)` rather than an
/// error: it means some other write legitimately won the race, not that
/// anything went wrong.
///
/// Adoption is this function's one caller and the reason it exists: its
/// destination is the session's very first transcript, and a session's own
/// normal turn persistence can independently create that same file at any
/// point during adoption's scan. Adoption must publish only if it still
/// holds the honor of "first write" when it finishes — never clobber a
/// conversation's genuine first turn with an adoption fold that started
/// scanning before that turn existed.
fn publish_transcript_if_absent(path: &Path, contents: &[u8]) -> Result<bool> {
    let tmp_path = unique_tmp_path(path);

    if let Err(error) = write_temp_file(&tmp_path, contents) {
        // Same orphaned-temp-file / symlink hazard as `atomic_write` — see
        // `write_temp_file`'s comment.
        let _ = fs::remove_file(&tmp_path);
        return Err(error);
    }
    let published = match fs::hard_link(&tmp_path, path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(error) => {
            let _ = fs::remove_file(&tmp_path);
            return Err(error).with_context(|| format!("publish transcript {}", path.display()));
        }
    };
    // The temp file and its hard-linked destination share one inode; once
    // linked (or once we know we lost the race), the temp name itself has
    // no further purpose.
    let _ = fs::remove_file(&tmp_path);
    Ok(published)
}

/// Append raw bytes to a file, opening in append mode (O(1), no read-back).
fn append_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut file = fs::OpenOptions::new()
        .read(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open transcript for append {}", path.display()))?;
    // A previous write may have stopped in the middle of a JSONL record.
    // Separate that incomplete tail from the new turn, so readers can skip
    // the damaged line without also skipping every later record.
    let len = file.metadata()?.len();
    let mut last = [0u8; 1];
    if len > 0 {
        file.seek(SeekFrom::End(-1))?;
        file.read_exact(&mut last)?;
    }
    let mut append = Vec::with_capacity(bytes.len() + usize::from(len > 0 && last[0] != b'\n'));
    if len > 0 && last[0] != b'\n' {
        append.push(b'\n');
    }
    append.extend_from_slice(bytes);
    file.write_all(&append)
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
        tracing::warn!(
            "[transcript] failed to create md companion dir {}: {err}",
            parent.display()
        );
        return;
    }
    let md = render_markdown(messages, meta, &per_msg_usage);
    if let Err(err) = fs::write(&md_path, md.as_bytes()) {
        tracing::warn!(
            "[transcript] failed to write markdown companion {}: {err}",
            md_path.display()
        );
        return;
    }
    tracing::debug!(
        "[transcript] wrote markdown companion to {}",
        md_path.display()
    );
}
