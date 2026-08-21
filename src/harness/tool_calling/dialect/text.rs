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

/// Escape XML metacharacters so a tool-controlled value cannot forge
/// `<tool_result>` protocol boundaries in the rendered transcript.
///
/// Tool names and outputs are model- and tool-controlled; a value containing a
/// literal `</tool_result>` (or a crafted `<tool_result name="forged" …>`)
/// would otherwise be indistinguishable from real envelope structure once
/// interpolated verbatim, letting tool output influence how the model reads
/// subsequent tool calls (CWE-74). Escaping `&`, `<`, `>`, and `"` neutralizes
/// both the tag delimiters and the attribute-value quote.
fn escape_xml(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Render freshly executed outcomes into the transcript entry a text dialect
/// appends after an assistant turn.
///
/// Results are keyed by **tool name** and carry an explicit `status`, because a
/// text-mode model has no call ids to correlate by — the name and the ordering
/// are all it has. The name and output are tool-controlled, so both are
/// escaped ([`escape_xml`]) before interpolation to keep them from forging
/// `<tool_result>` envelope boundaries.
pub fn format_results(results: &[ToolOutcome]) -> TranscriptEntry {
    let mut content = String::new();
    for result in results {
        let status = if result.success { "ok" } else { "error" };
        let _ = writeln!(
            content,
            "<tool_result name=\"{}\" status=\"{}\">\n{}\n</tool_result>",
            escape_xml(&result.name),
            status,
            escape_xml(&result.output)
        );
    }
    TranscriptEntry::Chat(DialectMessage::user(format!(
        "{TOOL_RESULTS_PREFIX}{content}"
    )))
}

/// Replay a transcript as flat chat messages.
///
/// Assistant tool calls degrade to `text` verbatim, and persisted results are
/// re-wrapped by **id** — which is what the durable record stores — into one
/// user turn.
///
/// `text` must be the **raw** model response the text dialect originally
/// parsed (tags and all), not the narrative-only text
/// [`ToolDialect::parse_response`](super::ToolDialect::parse_response) returns
/// with the `<tool_call>` tags stripped. A host that persists the parsed
/// narrative into `TranscriptEntry::AssistantToolCalls::text` replays an
/// assistant turn with no visible call, followed by a `<tool_result>` turn
/// that answers nothing the model can see.
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
                        escape_xml(&result.tool_call_id),
                        escape_xml(&result.content)
                    );
                }
                vec![DialectMessage::user(format!(
                    "{TOOL_RESULTS_PREFIX}{content}"
                ))]
            }
        })
        .collect()
}
