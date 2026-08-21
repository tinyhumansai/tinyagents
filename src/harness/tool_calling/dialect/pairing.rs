//! Dropping half-finished tool cycles before a transcript reaches the wire.
//!
//! # The failure this exists to prevent
//!
//! Providers reject an assistant message carrying `tool_calls` unless it is
//! immediately followed by tool messages answering **every** `tool_call_id` on
//! it. The error is a hard `400`, not a degraded reply, and it fails the whole
//! request — so one orphaned record poisons every subsequent turn of that
//! thread until the history is edited.
//!
//! Bisected cycles are not exotic. A cached transcript restore, an aborted
//! turn, and history compaction can each preserve the assistant half while the
//! results half is dropped or was never persisted. The repair therefore belongs
//! at the last possible moment — here, while serializing — not at the write
//! sites, which are many and none of which can see the final sequence.
//!
//! # Why adjacency alone is not the check
//!
//! The provider's complaint is about *coverage*: a following tool message set
//! whose ids do not cover every id on the opener fails the same way as no
//! following messages at all. So the opener's id set must **equal** the
//! follower's, not merely be adjacent to a non-empty one.
//!
//! The drop is symmetric. A `ToolResults` record whose opener was dropped is
//! itself dropped, because a `tool` role message answering a call the model
//! never sees is just as malformed.

use super::types::TranscriptEntry;

/// Return the entries safe to serialize, in order, dropping any tool cycle that
/// is not complete.
pub fn pair_tool_cycles(history: &[TranscriptEntry]) -> Vec<&TranscriptEntry> {
    let mut kept_indices: Vec<usize> = Vec::with_capacity(history.len());

    for (index, entry) in history.iter().enumerate() {
        match entry {
            TranscriptEntry::AssistantToolCalls { tool_calls, .. } => {
                let Some(TranscriptEntry::ToolResults(results)) = history.get(index + 1) else {
                    tracing::debug!(
                        index,
                        total = history.len(),
                        "[dialect][pairing] dropping unpaired assistant tool calls (no immediately \
                         following results — would trip the provider's 400)"
                    );
                    continue;
                };
                let opener_ids: std::collections::BTreeSet<&str> =
                    tool_calls.iter().map(|call| call.id.as_str()).collect();
                let result_ids: std::collections::BTreeSet<&str> = results
                    .iter()
                    .map(|result| result.tool_call_id.as_str())
                    .collect();
                if !opener_ids.is_empty() && opener_ids == result_ids {
                    kept_indices.push(index);
                } else {
                    tracing::debug!(
                        index,
                        ?opener_ids,
                        ?result_ids,
                        "[dialect][pairing] dropping assistant tool calls: call-id sets differ \
                         between the opener and its results"
                    );
                }
            }
            TranscriptEntry::ToolResults(_) => {
                let preceded_by_kept_opener = index > 0
                    && matches!(
                        history.get(index - 1),
                        Some(TranscriptEntry::AssistantToolCalls { .. })
                    )
                    && kept_indices.last() == Some(&(index - 1));
                if preceded_by_kept_opener {
                    kept_indices.push(index);
                } else {
                    tracing::debug!(
                        index,
                        "[dialect][pairing] dropping orphan tool results (their opener is not in \
                         the emitted sequence)"
                    );
                }
            }
            TranscriptEntry::Chat(_) => kept_indices.push(index),
        }
    }

    kept_indices.into_iter().map(|index| &history[index]).collect()
}
