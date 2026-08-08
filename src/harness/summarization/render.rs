//! Lossless-enough rendering of a transcript into summarizable text.
//!
//! # Why `Message::text()` is not enough
//!
//! [`Message::text`] returns only [`ContentBlock::Text`] blocks — by design, so
//! reasoning never leaks into visible assistant output. That makes it the wrong
//! function to summarize *with*: an assistant turn that only called tools has no
//! text, a tool result carrying JSON has no text, and reasoning has no text. A
//! summarizer built on `text()` therefore renders a tool-heavy transcript as a
//! column of bare role labels:
//!
//! ```text
//! [msg-2] assistant:
//! [msg-3] tool:
//! ```
//!
//! and compaction replaces real history with nothing. Since the crate's default
//! summarizer is exactly that, the out-of-the-box behaviour was to erase the
//! part of the history that was most expensive to produce.
//!
//! [`render_message_for_summary`] renders the whole message — text, reasoning,
//! tool calls with their arguments, tool results with their correlation id and
//! JSON content — in a compact tagged form. LangChain solves the same problem
//! with `get_buffer_string(..., format="xml")`.

use crate::harness::message::{ContentBlock, Message};

/// Maximum characters rendered for a single tool-call argument blob or tool
/// result body before it is elided.
///
/// A summary that faithfully reproduces a 2 MB tool result is not a summary.
/// The cap is generous enough to keep short structured results (ids, paths,
/// row counts) intact, which is what a later turn usually needs to refer back
/// to.
const MAX_RENDERED_PAYLOAD_CHARS: usize = 2_000;

/// The role label used when rendering a message for summarization.
fn role_label(message: &Message) -> &'static str {
    match message {
        Message::System(_) => "system",
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::Tool(_) => "tool",
    }
}

/// Truncates `text` to [`MAX_RENDERED_PAYLOAD_CHARS`], marking the elision so a
/// reader (or a downstream LLM summarizer) can tell content was dropped rather
/// than assuming the tool returned exactly that much.
fn elide(text: &str) -> String {
    let count = text.chars().count();
    if count <= MAX_RENDERED_PAYLOAD_CHARS {
        return text.to_string();
    }
    let kept: String = text.chars().take(MAX_RENDERED_PAYLOAD_CHARS).collect();
    format!(
        "{kept}… [{} chars elided]",
        count - MAX_RENDERED_PAYLOAD_CHARS
    )
}

/// Renders every content block that carries information a summary should keep:
/// visible text, structured JSON, reasoning, provider extensions, and a marker
/// for images (whose bytes are useless in a text summary but whose *presence*
/// is not).
fn render_content(content: &[ContentBlock]) -> Vec<String> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) if text.trim().is_empty() => None,
            ContentBlock::Text(text) => Some(text.clone()),
            ContentBlock::Json(value) => {
                Some(format!("<json>{}</json>", elide(&value.to_string())))
            }
            ContentBlock::Image(image) => Some(format!(
                "<image mime=\"{}\" />",
                image.mime_type.as_deref().unwrap_or("unknown")
            )),
            ContentBlock::Thinking { text, .. } if text.trim().is_empty() => None,
            ContentBlock::Thinking { text, .. } => {
                Some(format!("<reasoning>{}</reasoning>", elide(text)))
            }
            ContentBlock::RedactedThinking { .. } => Some("<reasoning redacted />".to_string()),
            ContentBlock::ProviderExtension(value) => Some(format!(
                "<provider_extension>{}</provider_extension>",
                elide(&value.to_string())
            )),
        })
        .collect()
}

/// Renders a single message into the text a summarizer should see.
///
/// The output is a single line per message where possible, in the shape
/// `<role>: <parts>`, with tool calls rendered as
/// `<tool_call id="…" name="…">{args}</tool_call>` and tool results as
/// `<tool_result id="…">…</tool_result>`. Large payloads are elided (see
/// [`MAX_RENDERED_PAYLOAD_CHARS`]).
pub fn render_message_for_summary(message: &Message) -> String {
    let mut parts: Vec<String> = match message {
        Message::System(m) => render_content(&m.content),
        Message::User(m) => render_content(&m.content),
        Message::Assistant(m) => render_content(&m.content),
        Message::Tool(m) => render_content(&m.content),
    };

    match message {
        Message::Assistant(assistant) => {
            for call in &assistant.tool_calls {
                let arguments = elide(&call.arguments.to_string());
                parts.push(format!(
                    "<tool_call id=\"{}\" name=\"{}\">{arguments}</tool_call>",
                    call.id, call.name
                ));
                if let Some(reason) = &call.invalid {
                    parts.push(format!("<tool_call_invalid>{reason}</tool_call_invalid>"));
                }
            }
        }
        Message::Tool(tool) => {
            // A tool result whose content rendered to nothing still needs its
            // correlation id recorded: "this call was answered" is itself the
            // fact a later turn reasons about.
            // Tool results are the one message kind whose *text* is elided as
            // well as its structured blocks: a tool that returns a 9 MB page
            // dump would otherwise be reproduced verbatim into the "summary".
            // User and assistant prose is left intact — that is the
            // conversation itself, and `ConcatSummarizer` promises verbatim
            // concatenation of it.
            let body = if parts.is_empty() {
                String::new()
            } else {
                elide(&parts.join(" "))
            };
            return format!(
                "tool: <tool_result id=\"{}\">{body}</tool_result>",
                tool.tool_call_id
            );
        }
        _ => {}
    }

    format!("{}: {}", role_label(message), parts.join(" "))
}
