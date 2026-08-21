//! Transcript rendering shared by the text dialects.
//!
//! The XML and p-format dialects differ only in how the model *asks* for a
//! tool. How results come back, and how a transcript is replayed onto the wire,
//! is identical for both: everything collapses to plain chat turns, because a
//! model being prompted in text has no structured tool channel to read from.
//!
//! Keeping that in one place is not just deduplication. The `<tool_result>`
//! envelope is advertised to the model in the protocol block; if the two
//! dialects rendered it separately, one of them could drift from what the
//! prompt promises and the model would be reading a format nothing emits.

use std::fmt::Write as _;

use super::types::{DialectMessage, ToolOutcome, TranscriptEntry};

/// Prefix of the synthetic user turn that carries tool results back to a
/// text-mode model.
pub const TOOL_RESULTS_PREFIX: &str = "[Tool results]\n";

/// Render freshly executed outcomes into the transcript entry a text dialect
/// appends after an assistant turn.
///
/// Results are keyed by **tool name** and carry an explicit `status`, because a
/// text-mode model has no call ids to correlate by — the name and the ordering
/// are all it has.
pub fn format_results(results: &[ToolOutcome]) -> TranscriptEntry {
    let mut content = String::new();
    for result in results {
        let status = if result.success { "ok" } else { "error" };
        let _ = writeln!(
            content,
            "<tool_result name=\"{}\" status=\"{}\">\n{}\n</tool_result>",
            result.name, status, result.output
        );
    }
    TranscriptEntry::Chat(DialectMessage::user(format!(
        "{TOOL_RESULTS_PREFIX}{content}"
    )))
}

/// Replay a transcript as flat chat messages.
///
/// Assistant tool calls degrade to their narrative text (the calls themselves
/// were already in that text, as tags), and persisted results are re-wrapped by
/// **id** — which is what the durable record stores — into one user turn.
pub fn to_provider_messages(history: &[TranscriptEntry]) -> Vec<DialectMessage> {
    history
        .iter()
        .flat_map(|entry| match entry {
            TranscriptEntry::Chat(chat) => vec![chat.clone()],
            TranscriptEntry::AssistantToolCalls {
                text,
                extra_metadata,
                ..
            } => vec![
                DialectMessage::assistant(text.clone().unwrap_or_default())
                    .with_metadata(extra_metadata.clone()),
            ],
            TranscriptEntry::ToolResults(results) => {
                let mut content = String::new();
                for result in results {
                    let _ = writeln!(
                        content,
                        "<tool_result id=\"{}\">\n{}\n</tool_result>",
                        result.tool_call_id, result.content
                    );
                }
                vec![DialectMessage::user(format!(
                    "{TOOL_RESULTS_PREFIX}{content}"
                ))]
            }
        })
        .collect()
}
