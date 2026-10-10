//! Token/cost spend totals read back from persisted session transcripts.
//!
//! Every transcript records only what the agent that owns it spent, so walking a
//! thread's root transcripts and their `{root_stem}__{child}` descendants counts
//! each token exactly once. Pricing (re-auditing cost at current rates) is the
//! host's concern and stays out of here.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use super::{
    DisplayRecord, SessionTranscript, TranscriptMessage, TranscriptMeta, TurnUsage,
    find_root_transcripts_for_thread, read_transcript_display,
};

/// One transcript's own spend, summed from the per-turn `turn_usage` records
/// the codec attaches to its assistant rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscriptSpend {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cost_usd: f64,
    /// Turns that recorded usage. The codec emits one record per durable
    /// append, and omits an all-zero one, so this is "turns that spent".
    pub turns: usize,
    /// The newest record's model and window, for the caller's last-turn view.
    /// `last_input_tokens` / `last_output_tokens` are that turn's spend, summed
    /// over every call it made; they are not a context size.
    pub last_input_tokens: u64,
    pub last_output_tokens: u64,
    /// Tokens the context held after the newest turn's final call (its input
    /// plus its reply): the numerator of a context-window gauge. See
    /// [`context_tokens_of`] for records written before the per-call fields.
    pub last_context_tokens: u64,
    pub model: Option<String>,
    pub context_window: u64,
}

/// Context occupancy recorded by one turn's usage: the final call's input plus
/// its reply.
///
/// A record written before `last_call_input` existed carries only the turn's
/// summed spend. A single-call turn's sum *is* its call, so it is exact; a
/// multi-call turn's is divided by its call count (`iteration`), the mean
/// request size. That undercounts the final, largest request but stays inside
/// the window, where the raw sum reported a 72-call turn at 5.7M tokens of a
/// 1M window.
pub fn context_tokens_of(usage: &TurnUsage) -> u64 {
    let record = &usage.usage;
    if record.last_call_input > 0 {
        return record.last_call_input.saturating_add(record.last_call_output);
    }
    let calls = u64::from(usage.iteration.max(1));
    record.input.saturating_add(record.output) / calls
}

/// Total one transcript's own recorded spend over its logical message set.
///
/// The `_meta` header carries denormalised rollups for the same figures, but
/// nothing has written them on the root path since the TinyAgents runtime
/// cutover — every root transcript since reads `input_tokens: 0`
/// while its per-turn records hold the real numbers (#6460). The per-turn
/// records are the authoritative copy and the only one that is written on every
/// path, so this reads those and ignores the header.
///
/// This sums only what `transcript.messages` still holds: after a compaction
/// that is the reduced context, so it undercounts a compacted session. The
/// thread aggregate ([`thread_spend`]) reads the full append history instead.
pub fn transcript_spend(transcript: &SessionTranscript) -> TranscriptSpend {
    spend_from_usages(
        transcript
            .messages
            .iter()
            .filter_map(|message| message.turn_usage.as_ref()),
        transcript.meta.model.as_ref(),
    )
}

/// Sum a sequence of per-turn usage records, oldest first.
fn spend_from_usages<'a>(
    usages: impl IntoIterator<Item = &'a TurnUsage>,
    header_model: Option<&String>,
) -> TranscriptSpend {
    let mut spend = TranscriptSpend::default();
    for usage in usages {
        // A text-dialect tool round's issuing row carries a provenance-only
        // record (its calls, zero spend); the turn's spend is on its final row.
        // It is not a turn that spent, and must not become the "last" one.
        if !usage.tool_calls.is_empty()
            && usage.usage.input == 0
            && usage.usage.output == 0
            && usage.usage.cached_input == 0
            && usage.usage.cost_usd == 0.0
        {
            continue;
        }
        spend.input_tokens = spend.input_tokens.saturating_add(usage.usage.input);
        spend.output_tokens = spend.output_tokens.saturating_add(usage.usage.output);
        spend.cached_input_tokens = spend
            .cached_input_tokens
            .saturating_add(usage.usage.cached_input);
        spend.cost_usd += usage.usage.cost_usd;
        spend.turns += 1;
        spend.last_input_tokens = usage.usage.input;
        spend.last_output_tokens = usage.usage.output;
        spend.last_context_tokens = context_tokens_of(usage);
        if usage.usage.context_window > 0 {
            spend.context_window = usage.usage.context_window;
        }
        if !usage.model.is_empty() {
            spend.model = Some(usage.model.clone());
        }
    }
    // A transcript whose model never landed on a usage record still has one in
    // its header; prefer the record, fall back to the header.
    if spend.model.is_none() {
        spend.model = header_model.cloned();
    }
    spend
}

/// Spend of one transcript file over its whole append history.
///
/// Compaction never deletes lines: the earlier rows (and their usage) stay in
/// the file, and the compaction record carries the reduced context. Reading
/// the model-context view would drop every usage record a compaction summarised
/// away, so this walks the whole log instead:
///
/// - every appended message line is a distinct durable append and always
///   counts (interrupted partials carry no spend and are skipped);
/// - a compaction replacement mixes kept history with the rows of the turn
///   that wrote it. Kept rows were read back from the transcript and keep
///   their original `request_id`; the turn's own rows take the compaction's
///   `request_id`. So a replacement row's usage counts only when its
///   `request_id` is the compaction's and no appended line already counted
///   spend under that `request_id`. A compaction written without a
///   `request_id` falls back to the usage on its final assistant row.
///
/// `earlier_requests` holds the `request_id`s that earlier generation files of
/// the same conversation already counted. A successor generation starts by
/// writing the rows it retained from its parent, and those rows keep their
/// original `request_id`, so a row whose `request_id` is in that set is
/// carried history, not new spend. The ids this file counts are added to it.
fn file_spend(
    path: &Path,
    earlier_requests: &mut HashSet<String>,
) -> Option<(TranscriptSpend, TranscriptMeta)> {
    let display = match read_transcript_display(path) {
        Ok(display) => display,
        Err(err) => {
            tracing::warn!(
                "[transcript:spend] skipping unreadable transcript {}: {err}",
                path.display()
            );
            return None;
        }
    };
    let mut usages: Vec<&TurnUsage> = Vec::new();
    let mut counted_requests: HashSet<&str> = HashSet::new();
    // The logical context as of the previous record: what a compaction's kept
    // rows are carried over from.
    let mut context: Vec<&TranscriptMessage> = Vec::new();
    for record in &display.records {
        match record {
            DisplayRecord::Message(message) if !message.interrupted => {
                let carried = message
                    .request_id
                    .as_deref()
                    .is_some_and(|request_id| earlier_requests.contains(request_id));
                if !carried && let Some(usage) = message.message.turn_usage.as_ref() {
                    usages.push(usage);
                    if let Some(request_id) = message.request_id.as_deref() {
                        counted_requests.insert(request_id);
                    }
                }
                context.push(&message.message);
            }
            DisplayRecord::Message(_) => {}
            DisplayRecord::Compaction(marker) => {
                match marker.request_id.as_deref() {
                    Some(turn)
                        if !counted_requests.contains(turn) && !earlier_requests.contains(turn) =>
                    {
                        let fresh: Vec<&TurnUsage> = marker
                            .replacement
                            .iter()
                            .filter(|row| row.request_id.as_deref() == Some(turn))
                            .filter_map(|row| row.message.turn_usage.as_ref())
                            .collect();
                        if !fresh.is_empty() {
                            counted_requests.insert(turn);
                        }
                        usages.extend(fresh);
                    }
                    Some(_) => {}
                    None => {
                        // A compaction written without a `request_id` has no
                        // turn identity. Its writing turn's spend sits on the
                        // replacement's final assistant row; that row is fresh
                        // unless it was carried over from the context this
                        // compaction replaced (same position-free row:
                        // role, content and usage).
                        if let Some(row) = marker
                            .replacement
                            .iter()
                            .rev()
                            .find(|row| row.message.role == "assistant")
                            && let Some(usage) = row.message.turn_usage.as_ref()
                            && !context
                                .iter()
                                .any(|prior| carried_from(prior, &row.message))
                        {
                            usages.push(usage);
                        }
                    }
                }
                context = marker.replacement.iter().map(|row| &row.message).collect();
            }
        }
    }
    let spend = spend_from_usages(usages, display.meta.model.as_ref());
    earlier_requests.extend(counted_requests.into_iter().map(str::to_owned));
    Some((spend, display.meta))
}

/// Whether `row` in a compaction replacement is `prior` carried over: same
/// role, content, typed structure and usage record.
fn carried_from(prior: &TranscriptMessage, row: &TranscriptMessage) -> bool {
    prior.role == row.role
        && prior.content == row.content
        && prior.tool_calls == row.tool_calls
        && prior.tool_call_id == row.tool_call_id
        && prior.parts == row.parts
        && prior.turn_usage == row.turn_usage
}

/// Every descendant transcript of `root`, at any delegation depth.
///
/// Sub-agent transcripts are named `{root_stem}__{child}` (and a grandchild
/// chains another `__`), which is the *only* durable record of the parent→child
/// relation. Selecting them by `_meta.thread_id` instead — as the session
/// crate's own summary helper does — drops the majority of them: a delegation
/// gets its own worker thread, and `inherited_thread_id` lets that worker id win
/// over the parent's, so the child's header names a thread the caller never
/// asked about (#6460).
fn descendant_transcripts(root: &Path) -> Vec<PathBuf> {
    let Some(dir) = root.parent() else {
        return Vec::new();
    };
    let Some(root_stem) = root.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let prefix = format!("{root_stem}__");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut matches: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().and_then(|s| s.to_str()) == Some("jsonl")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|stem| stem.starts_with(&prefix))
        })
        .collect();
    matches.sort();
    matches
}

/// One thread's spend, split into the orchestrator's own and each sub-agent
/// archetype's, each counted exactly once.
#[derive(Debug, Clone, Default)]
pub struct ThreadSpend {
    pub root: TranscriptSpend,
    /// Keyed by archetype (`_meta.agent`), with a run count.
    pub subagents: BTreeMap<String, (TranscriptSpend, usize)>,
    pub updated: Option<String>,
    pub found_transcript: bool,
}

/// Walk a thread's root transcripts and their descendants, totalling each file's
/// own recorded spend.
///
/// Correct-by-construction because every transcript records only what the agent
/// that owns it spent (see the host's per-turn usage codec): the walk visits
/// each file once, so no token is counted twice however deep the delegation went,
/// and a child whose usage never reached its parent's in-turn ledger (#6459) is
/// still counted from its own file.
pub fn thread_spend(workspace_dir: &Path, thread_id: &str) -> ThreadSpend {
    let mut out = ThreadSpend::default();
    let roots = find_root_transcripts_for_thread(workspace_dir, thread_id);
    // Shared across the thread's root files (oldest first), so rows a later
    // generation carried over from an earlier one are not counted twice.
    let mut root_requests = HashSet::new();
    for root in &roots {
        out.found_transcript = true;
        if let Some((spend, meta)) = file_spend(root, &mut root_requests) {
            out.root.input_tokens = out.root.input_tokens.saturating_add(spend.input_tokens);
            out.root.output_tokens = out.root.output_tokens.saturating_add(spend.output_tokens);
            out.root.cached_input_tokens = out
                .root
                .cached_input_tokens
                .saturating_add(spend.cached_input_tokens);
            out.root.cost_usd += spend.cost_usd;
            out.root.turns += spend.turns;
            // `find_root_transcripts_for_thread` returns oldest first, so the
            // last root to report a turn owns the last-turn view.
            if spend.turns > 0 {
                out.root.last_input_tokens = spend.last_input_tokens;
                out.root.last_output_tokens = spend.last_output_tokens;
                out.root.last_context_tokens = spend.last_context_tokens;
            }
            if spend.context_window > 0 {
                out.root.context_window = spend.context_window;
            }
            if spend.model.is_some() {
                out.root.model = spend.model;
            }
            out.updated = Some(meta.updated);
        }
        for child in descendant_transcripts(root) {
            let Some((spend, child_meta)) = file_spend(&child, &mut HashSet::new()) else {
                continue;
            };
            out.found_transcript = true;
            let entry = out
                .subagents
                .entry(child_meta.agent_name.clone())
                .or_default();
            entry.0.input_tokens = entry.0.input_tokens.saturating_add(spend.input_tokens);
            entry.0.output_tokens = entry.0.output_tokens.saturating_add(spend.output_tokens);
            entry.0.cached_input_tokens = entry
                .0
                .cached_input_tokens
                .saturating_add(spend.cached_input_tokens);
            entry.0.cost_usd += spend.cost_usd;
            entry.0.turns += spend.turns;
            if entry.0.model.is_none() {
                entry.0.model = spend.model;
            }
            entry.1 += 1;
        }
    }
    out
}

#[cfg(test)]
#[path = "spend_tests.rs"]
mod tests;
