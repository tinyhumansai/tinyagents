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
//!
//! # Envelope integrity, without mangling the payload
//!
//! Tool names and outputs are tool-controlled, so neither can be interpolated
//! into the envelope unexamined — a body spelling `</tool_result>` would close
//! it early (CWE-74). The two get different rules because they land in
//! different places, and the difference is the point:
//!
//! * a **name** is an attribute value, so [`escape_attribute`] escapes the lot;
//! * an **output** is the body, so [`neutralize_protocol_tags`] rewrites only
//!   the protocol tag openers and lets everything else through byte-for-byte.
//!
//! Escaping the body wholesale is the obvious implementation and the wrong one:
//! tool output is usually source code, and this is the channel prompt-guided
//! models read it through, so `&lt;div&gt;` is a fidelity cost paid on every
//! result to defend against a handful of exact byte sequences.

use std::borrow::Cow;
use std::fmt::Write as _;

use super::types::{DialectMessage, ToolOutcome, TranscriptEntry};

/// Prefix of the synthetic user turn that carries tool results back to a
/// text-mode model.
pub const TOOL_RESULTS_PREFIX: &str = "[Tool results]\n";

/// Escape XML metacharacters in a value that goes inside an **attribute**.
///
/// Attribute values sit between quotes in `<tool_result name="…" status="…">`,
/// so a `"` ends the attribute and a `<`/`>` ends the tag. They are short
/// identifiers — a tool name, a call id — where escaping costs nothing legible,
/// so the blunt rule is the right one here.
fn escape_attribute(value: &str) -> String {
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

/// Tag names a tool-controlled **body** must not be able to spell literally.
///
/// `tool_result` is the envelope this module renders; `tool_call` is what the
/// parser recognizes in model output. Both are protocol structure, and a body
/// that can produce either verbatim can make the transcript read as though it
/// contains boundaries or calls that nothing emitted.
const PROTOCOL_TAGS: &[&str] = &["tool_result", "tool_call"];

/// Neutralize protocol tag openers in a tool-controlled body, and **only**
/// those.
///
/// # Why not just escape every metacharacter
///
/// Because the body is usually source code, and escaping all of `& < > "`
/// turns `<div className="x">` into `&lt;div className=&quot;x&quot;&gt;`
/// before the model ever sees it. That is the primary tool-output channel for
/// prompt-guided models — the p-format path local models use — so the cost
/// lands exactly where tool output matters most: the model reads mangled code,
/// and may write mangled code back.
///
/// The security property does not require that. What must not happen is a body
/// spelling a **protocol boundary**: a literal `</tool_result>` closing the
/// envelope early, or a crafted `<tool_result name="forged" status="ok">`
/// opening a fake one (CWE-74, the finding on PR #116). That is a handful of
/// exact byte sequences, not a character class. So only a `<` that begins one
/// of [`PROTOCOL_TAGS`] — with or without a leading `/`, and with the tag name
/// actually ending there — becomes `&lt;`; every other `<`, `>`, `&` and `"`
/// passes through untouched. The name-boundary check is what keeps
/// `<tool_calls>` and `<tool_resultant>` intact: each false positive is
/// fidelity spent for no security.
///
/// Matching is ASCII-case-insensitive: the protocol is lowercase, but a forgery
/// is not obliged to be, and the comparison is free.
///
/// # What this does not claim
///
/// Boundary integrity, not prompt-injection defence. A tool can still return
/// prose that *argues* the model should do something, and no escaping rule
/// fixes that — a file the agent legitimately reads can contain anything. This
/// guarantees only that tool output cannot masquerade as the transcript's own
/// protocol structure.
///
/// Returns [`Cow::Borrowed`] when nothing matched, which is the overwhelmingly
/// common case and makes "ordinary content is passed through unchanged" a
/// property of the type rather than a promise in a comment.
fn neutralize_protocol_tags(value: &str) -> Cow<'_, str> {
    let bytes = value.as_bytes();
    let mut out: Option<String> = None;
    let mut copied_to = 0;

    for (index, _) in value.char_indices().filter(|(_, ch)| *ch == '<') {
        let after_angle = &bytes[index + 1..];
        let rest = match after_angle.first() {
            Some(b'/') => &after_angle[1..],
            _ => after_angle,
        };
        let is_protocol_tag = PROTOCOL_TAGS.iter().any(|tag| {
            if rest.len() < tag.len() || !rest[..tag.len()].eq_ignore_ascii_case(tag.as_bytes()) {
                return false;
            }
            // The tag name must actually end here, on a real XML tag
            // terminator: whitespace, `>`, a self-closing `/`, or the end of
            // the body. Not merely "a byte outside `[A-Za-z0-9_-]`" — `.` and
            // `:` are valid XML name characters, so treating them as
            // boundaries would rewrite `<tool_result.debug>` and
            // `<tool_call:custom>`, neither of which is protocol.
            //
            // End of body counts, and that is the subtle one. The body is not
            // the end of the rendered text: the caller appends
            // "\n</tool_result>" straight after it, so a body ending in a bare
            // `<tool_result` is followed by a newline in the message the model
            // actually reads, and that opener swallows the real closing tag.
            // "No terminator yet, so not a protocol tag" reads correct in
            // isolation and is wrong in context.
            match rest.get(tag.len()) {
                None => true,
                Some(byte) => byte.is_ascii_whitespace() || matches!(*byte, b'>' | b'/'),
            }
        });
        if !is_protocol_tag {
            continue;
        }

        let buffer = out.get_or_insert_with(|| String::with_capacity(value.len() + 8));
        buffer.push_str(&value[copied_to..index]);
        buffer.push_str("&lt;");
        copied_to = index + 1;
    }

    match out {
        Some(mut buffer) => {
            buffer.push_str(&value[copied_to..]);
            Cow::Owned(buffer)
        }
        None => Cow::Borrowed(value),
    }
}

/// Render freshly executed outcomes into the transcript entry a text dialect
/// appends after an assistant turn.
///
/// Results are keyed by **tool name** and carry an explicit `status`, because a
/// text-mode model has no call ids to correlate by — the name and the ordering
/// are all it has.
///
/// Both the name and the output are tool-controlled, and each gets the rule
/// that fits where it lands: the name is an attribute value, so it is fully
/// escaped ([`escape_attribute`]); the output is the body, so only protocol
/// tag openers are neutralized ([`neutralize_protocol_tags`]) and the rest
/// reaches the model byte-for-byte.
pub fn format_results(results: &[ToolOutcome]) -> TranscriptEntry {
    let mut content = String::new();
    for result in results {
        let status = if result.success { "ok" } else { "error" };
        let _ = writeln!(
            content,
            "<tool_result name=\"{}\" status=\"{}\">\n{}\n</tool_result>",
            escape_attribute(&result.name),
            status,
            neutralize_protocol_tags(&result.output)
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
/// user turn, under the same envelope rules [`format_results`] applies: the id
/// is escaped as an attribute, the content only has protocol tag openers
/// neutralized.
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
                        escape_attribute(&result.tool_call_id),
                        neutralize_protocol_tags(&result.content)
                    );
                }
                vec![DialectMessage::user(format!(
                    "{TOOL_RESULTS_PREFIX}{content}"
                ))]
            }
        })
        .collect()
}
