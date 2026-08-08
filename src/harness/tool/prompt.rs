//! Provider-neutral prompt-guided (text-mode) tool calling for models without native tool support.
//!
//! Provider adapters whose model profile has `tool_calling = false` can use these
//! helpers to embed tool specs **in the system prompt** as a small protocol and
//! parse the model's `<tool_call>…</tool_call>` blocks back into
//! [`ToolCall`]s — so the agent loop sees tool calls identically to the native
//! path, without changing the harness loop.
//!
//! The `<tool_call>{"name":…,"arguments":…}</tool_call>` convention matches the
//! long-standing OpenHuman host format so models already prompted for it behave
//! identically after the crate cutover.

use std::fmt::Write as _;

use serde_json::{Map, Value};

use crate::harness::message::{ContentBlock, Message};
use crate::harness::model::ModelResponse;
use crate::harness::tool::{ToolCall, ToolSchema};

/// Opening / closing delimiters for a text-mode tool call.
const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";

/// Prefix of the opening delimiter, matched tolerantly so the attribute form
/// (`<tool_call id="call_0">`, as emitted by Hermes / DeepSeek chat templates)
/// and the pipe variant (`<tool_call|>`) are recognized — not just the bare
/// `<tool_call>` literal. The scanner matches this prefix, verifies the tag name
/// is properly delimited, and then consumes up to the tag's closing `>`.
const OPEN_PREFIX: &str = "<tool_call";

/// DeepSeek-R1 native tool-call delimiters, emitted verbatim as text by some
/// OpenAI-compatible routes. Matched as an alternative open/close pair so the
/// body between them is parsed (or dropped) instead of leaking to the caller.
const DS_OPEN: &str = "<｜tool▁call▁begin｜>";
const DS_CLOSE: &str = "<｜tool▁call▁end｜>";

/// Leading marker of the synthetic user turn that [`coalesce_prompt_tool_results`]
/// folds tool results into. Shared with [`ensure_resolvable_user_turn`], which
/// must be able to tell a folded result apart from a real user query — keeping
/// one const means the two can never drift.
const TOOL_RESULTS_MARKER: &str = "[Tool results]";

/// User turn synthesized by [`ensure_resolvable_user_turn`] when a request
/// carries none. Deliberately content-free: the actual task is in the system
/// prompt (or in the transcript that follows), and this exists to satisfy chat
/// templates that require a locatable user query, not to add instructions.
const CONTINUATION_USER_TURN: &str = "Continue with the task described above.";

/// Build the tool-use protocol block appended to the system prompt when native
/// tool calling is unavailable. Describes the `<tool_call>` convention and lists
/// each tool's name, description, and JSON-Schema parameters.
pub fn prompt_tool_instructions(tools: &[ToolSchema]) -> String {
    let mut out = String::new();
    out.push_str("## Tool Use Protocol\n\n");
    out.push_str("To use a tool, wrap a JSON object in <tool_call></tool_call> tags:\n\n");
    out.push_str(OPEN_TAG);
    out.push('\n');
    out.push_str(r#"{"name": "tool_name", "arguments": {"param": "value"}}"#);
    out.push('\n');
    out.push_str(CLOSE_TAG);
    out.push_str("\n\n");
    out.push_str("You may emit multiple tool calls in a single response. ");
    out.push_str("After execution, results appear in <tool_result> tags. ");
    out.push_str("Continue reasoning with the results until you can give a final answer.\n\n");
    out.push_str("### Available Tools\n\n");
    for tool in tools {
        let params = serde_json::to_string(&tool.parameters).unwrap_or_else(|_| "{}".to_string());
        // Infallible: writing to a String never errors.
        let _ = writeln!(out, "**{}**: {}", tool.name, tool.description);
        let _ = writeln!(out, "Parameters: `{params}`\n");
    }
    out
}

/// Return `messages` with the tool-use protocol appended to the system prompt:
/// the instructions are added as a trailing block on the first system message, or
/// a new leading system message when the request carries none. `tools` empty →
/// `messages` is returned unchanged (cloned).
pub fn with_prompt_tool_instructions(messages: &[Message], tools: &[ToolSchema]) -> Vec<Message> {
    if tools.is_empty() {
        return messages.to_vec();
    }
    let block = prompt_tool_instructions(tools);
    let mut out = messages.to_vec();
    if let Some(Message::System(system)) = out.iter_mut().find(|m| matches!(m, Message::System(_)))
    {
        // Append as a distinct text block so the original system prompt is intact.
        system
            .content
            .push(ContentBlock::Text(format!("\n\n{block}")));
    } else {
        out.insert(0, Message::system(block));
    }
    out
}

/// Convert structured assistant tool calls and native tool-result messages into
/// prompt-guided turns.
///
/// Models without native tool calling cannot consume provider assistant
/// `tool_calls` fields or a `tool` role. Assistant calls are rendered back into
/// `<tool_call>` blocks and cleared from the structured field; consecutive
/// results are folded into one `[Tool results]` user message with each result
/// wrapped in the advertised `<tool_result>` protocol. Other messages keep
/// their original order and type.
pub fn coalesce_prompt_tool_results(messages: &[Message]) -> Vec<Message> {
    let mut out = Vec::with_capacity(messages.len());
    let mut pending = Vec::new();

    fn flush(out: &mut Vec<Message>, pending: &mut Vec<String>) {
        if !pending.is_empty() {
            out.push(Message::user(format!(
                "{TOOL_RESULTS_MARKER}\n{}",
                std::mem::take(pending).join("\n")
            )));
        }
    }

    for message in messages {
        match message {
            Message::Tool(_) => {
                pending.push(format!("<tool_result>\n{}\n</tool_result>", message.text()));
            }
            Message::Assistant(assistant) if !assistant.tool_calls.is_empty() => {
                flush(&mut out, &mut pending);
                let mut assistant = assistant.clone();
                let mut rendered = String::new();
                if !assistant.content.is_empty() && !message.text().trim().is_empty() {
                    rendered.push('\n');
                }
                for call in &assistant.tool_calls {
                    let body = serde_json::json!({
                        "name": &call.name,
                        "arguments": &call.arguments,
                    });
                    let body = serde_json::to_string(&body).unwrap_or_else(|_| "{}".to_string());
                    let _ = writeln!(rendered, "{OPEN_TAG}{body}{CLOSE_TAG}");
                }
                assistant.content.push(ContentBlock::Text(rendered));
                assistant.tool_calls.clear();
                out.push(Message::Assistant(assistant));
            }
            _ => {
                flush(&mut out, &mut pending);
                out.push(message.clone());
            }
        }
    }
    flush(&mut out, &mut pending);
    out
}

/// Whether this message is a user turn a chat template can resolve as "the user
/// query".
///
/// A folded tool-result turn does not count. It carries the
/// [`TOOL_RESULTS_MARKER`] prefix, and templates that look for a user query are
/// looking for a request to answer, not for the transcript of a tool the model
/// itself invoked — Qwen 3's template makes the same distinction, skipping user
/// turns that are wholly a tool response. Neither does an empty or
/// whitespace-only turn. Non-text content (JSON, an image) does count: it is a
/// real user input the model is being asked about.
fn is_resolvable_user_query(message: &Message) -> bool {
    let Message::User(user) = message else {
        return false;
    };
    if message.text().trim_start().starts_with(TOOL_RESULTS_MARKER) {
        return false;
    }
    user.content.iter().any(|block| match block {
        ContentBlock::Text(text) => !text.trim().is_empty(),
        ContentBlock::Json(_) | ContentBlock::Image(_) => true,
        // Reasoning replay and opaque provider payloads are not user input.
        ContentBlock::Thinking { .. }
        | ContentBlock::RedactedThinking { .. }
        | ContentBlock::ProviderExtension(_) => false,
    })
}

/// Guarantee the outgoing list contains a user turn a chat template can resolve,
/// inserting one only when none is present.
///
/// Models without native tool calling are driven through their **own** chat
/// template by the serving runtime (LM Studio, llama.cpp, Ollama), and several
/// widely used templates hard-require a locatable user query. Qwen 3's raises
/// outright:
///
/// ```text
/// {%- if ns.multi_step_tool %}{{- raise_exception('No user query found in messages.') }}
/// ```
///
/// A prompt-guided tool loop can reach that state legitimately: once the real
/// user turn has aged out of the window — summarization, a resumed transcript,
/// a task delivered entirely through the system prompt — every remaining
/// non-system turn is an assistant continuation or a folded tool result, and the
/// template aborts the request with a 400 before the model is ever called
/// (tinyhumansai/openhuman#5291). Native-tool models are unaffected: they are
/// served through the provider's own tool protocol, not a Jinja template with
/// this guard.
///
/// The inserted turn goes directly after any leading system messages, so the
/// transcript still reads system → user → assistant. A request that already has
/// a real user turn is returned unchanged.
pub fn ensure_resolvable_user_turn(messages: &[Message]) -> Vec<Message> {
    if messages.iter().any(is_resolvable_user_query) {
        return messages.to_vec();
    }
    let mut out = messages.to_vec();
    let insert_at = out
        .iter()
        .position(|message| !matches!(message, Message::System(_)))
        .unwrap_or(out.len());
    out.insert(insert_at, Message::user(CONTINUATION_USER_TURN));
    out
}

/// Extract `<tool_call>…</tool_call>` blocks from `text`, parsing each inner JSON
/// object (`{"name":…,"arguments":…}`) into a [`ToolCall`]. Returns the text with
/// the blocks removed (trimmed) plus the parsed calls, in order.
///
/// The opening delimiter is matched tolerantly: the bare `<tool_call>` literal,
/// the attribute form `<tool_call id="…">` and pipe variant `<tool_call|>`
/// emitted by Hermes / DeepSeek chat templates, and the DeepSeek
/// `<｜tool▁call▁begin｜>` delimiter all open a block.
///
/// Robust to noise: a block whose inner text is not a JSON object with a string
/// `name` is dropped (its raw markup never leaks back into the text); a dangling
/// open tag with no close is left verbatim in the returned text; a prose mention
/// of `<tool_call` with no closing `>` (or the plural `<tool_calls>`) is not
/// treated as an opening tag.
pub fn parse_prompt_tool_calls_from_text(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = text;

    while let Some(open) = next_open(rest) {
        cleaned.push_str(&rest[..open.start]);
        let after_open = &rest[open.body_start..];
        let Some(end) = after_open.find(open.close) else {
            // Unterminated block: keep it (and everything after) as plain text.
            cleaned.push_str(&rest[open.start..]);
            return (cleaned.trim().to_string(), calls);
        };
        let inner = after_open[..end].trim();
        if let Some(call) = parse_one(inner, calls.len() + 1) {
            calls.push(call);
        }
        rest = &after_open[end + open.close.len()..];
    }
    cleaned.push_str(rest);
    (cleaned.trim().to_string(), calls)
}

/// A located opening tool-call delimiter.
struct OpenMatch {
    /// Byte offset of the opening `<`.
    start: usize,
    /// Byte offset where the inner body begins (just past the opening tag's `>`
    /// for the `<tool_call …>` family, or past the DeepSeek open delimiter).
    body_start: usize,
    /// Closing delimiter that terminates this block.
    close: &'static str,
}

/// Find the earliest opening tool-call delimiter in `text`, tolerant of the
/// attribute form (`<tool_call id="…">`), the pipe variant (`<tool_call|>`), and
/// the DeepSeek `<｜tool▁call▁begin｜>` delimiter. Returns `None` when no complete
/// opening tag is present. A bare `<tool_call` mention with no closing `>` (prose)
/// is not treated as an opening tag.
fn next_open(text: &str) -> Option<OpenMatch> {
    let mut best: Option<OpenMatch> = None;

    // DeepSeek native delimiter (literal open/close pair).
    if let Some(start) = text.find(DS_OPEN) {
        best = Some(OpenMatch {
            start,
            body_start: start + DS_OPEN.len(),
            close: DS_CLOSE,
        });
    }

    // `<tool_call …>` family: the first prefix occurrence whose tag name is
    // properly delimited (`>`, whitespace, or `|` — so `<tool_calls>` /
    // `<tool_callable>` do not match) and is closed by a `>`.
    let mut from = 0;
    while let Some(rel) = text[from..].find(OPEN_PREFIX) {
        let start = from + rel;
        let after_prefix = &text[start + OPEN_PREFIX.len()..];
        let delimited = match after_prefix.chars().next() {
            Some('>') | Some('|') => true,
            Some(c) => c.is_whitespace(),
            None => false,
        };
        if delimited && let Some(gt) = after_prefix.find('>') {
            let candidate = OpenMatch {
                start,
                body_start: start + OPEN_PREFIX.len() + gt + 1,
                close: CLOSE_TAG,
            };
            best = match best {
                Some(b) if b.start <= candidate.start => Some(b),
                _ => Some(candidate),
            };
            break;
        }
        // Not a usable open tag here (prose mention, or no closing `>`): keep
        // scanning past this occurrence. Advances by a non-zero amount.
        from = start + OPEN_PREFIX.len();
    }

    best
}

/// Byte index in `buf` from which the trailing bytes must be held back because
/// they could still grow into a tool-call open delimiter once more text arrives.
/// Returns `buf.len()` when the whole buffer is provably safe to surface now.
///
/// Two shapes are held: an in-progress `<tool_call …` open tag whose closing `>`
/// has not arrived yet (so [`next_open`] cannot see it), and a trailing byte run
/// that is a proper prefix of an open delimiter (`<tool_cal`, a partial DeepSeek
/// `<｜tool▁`, or a lone `<`). A complete open delimiter is never reported here —
/// [`next_open`] handles that case.
fn hold_from(buf: &str) -> usize {
    // In-progress `<tool_call …` tag: the last openable prefix occurrence whose
    // `>` has not yet arrived. `<tool_calls>` (name not delimited) is not openable.
    let mut from = 0;
    let mut incomplete: Option<usize> = None;
    while let Some(rel) = buf[from..].find(OPEN_PREFIX) {
        let start = from + rel;
        let after = &buf[start + OPEN_PREFIX.len()..];
        let openable = match after.chars().next() {
            None => true,
            Some('>') | Some('|') => true,
            Some(c) => c.is_whitespace(),
        };
        if openable && !after.contains('>') {
            incomplete = Some(start);
        }
        from = start + OPEN_PREFIX.len();
    }
    if let Some(start) = incomplete {
        return start;
    }

    // Trailing proper prefix of an open delimiter (a full delimiter would have
    // been reported by `next_open` or the in-progress branch above).
    let len = buf.len();
    for marker in [OPEN_PREFIX, DS_OPEN] {
        // A held tail can be as long as the whole buffer when the buffer is
        // shorter than the marker, so the range is inclusive of `max`.
        let max = marker.len().min(len);
        for k in (1..=max).rev() {
            if marker.is_char_boundary(k)
                && buf.is_char_boundary(len - k)
                && buf[len - k..] == marker[..k]
            {
                return len - k;
            }
        }
    }
    len
}

/// Streaming counterpart to [`parse_prompt_tool_calls_from_text`]: strips
/// `<tool_call>…</tool_call>` markup from a text stream *as fragments arrive*.
///
/// The terminal-response recovery ([`apply_prompt_tool_calls`]) only cleans the
/// aggregated answer, so a consumer that renders live [`MessageDelta`] text would
/// still see raw markup stream through. This scrubber closes that gap: it emits
/// only the text that is provably not part of a tool-call block, holding back any
/// tail that could still become one (a partial `<tool_call`, an open tag whose
/// `>` or matching close has not arrived, or a partial DeepSeek delimiter) until
/// more input resolves it.
///
/// It is stateful across fragments — feed each fragment through [`feed`](Self::feed)
/// and call [`flush`](Self::flush) once when the stream ends to drain the final
/// safe remainder. Only the visible-text channel is scrubbed; reasoning and
/// structured tool-call channels are unaffected.
#[derive(Debug, Default)]
pub struct ToolCallStreamScrubber {
    buf: String,
}

impl ToolCallStreamScrubber {
    /// Creates an empty scrubber.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds the next text fragment and returns the portion that is safe to emit
    /// now. Complete `<tool_call>` blocks are dropped; text that could still be
    /// the start of one is buffered for a later fragment.
    pub fn feed(&mut self, fragment: &str) -> String {
        self.buf.push_str(fragment);
        let mut out = String::new();
        loop {
            match next_open(&self.buf) {
                Some(open) => {
                    let after_open = &self.buf[open.body_start..];
                    if let Some(end) = after_open.find(open.close) {
                        // Complete block: the prefix before it is safe; drop the
                        // block and keep scanning the remainder.
                        out.push_str(&self.buf[..open.start]);
                        let next = open.body_start + end + open.close.len();
                        self.buf.drain(..next);
                        continue;
                    }
                    // Open delimiter located but not yet closed: emit the prefix,
                    // hold the unterminated block for later fragments.
                    out.push_str(&self.buf[..open.start]);
                    self.buf.drain(..open.start);
                    break;
                }
                None => {
                    // No complete open delimiter: emit everything except the tail
                    // that could still grow into one.
                    let hold = hold_from(&self.buf);
                    out.push_str(&self.buf[..hold]);
                    self.buf.drain(..hold);
                    break;
                }
            }
        }
        out
    }

    /// Drains the final safe remainder once no more fragments will arrive. A
    /// complete tool-call block still buffered is dropped; a dangling partial
    /// delimiter is surfaced verbatim (with the stream ended it was never a real
    /// call). Unlike [`parse_prompt_tool_calls_from_text`], the remainder is not
    /// trimmed — streamed whitespace is preserved.
    pub fn flush(&mut self) -> String {
        let mut out = String::new();
        loop {
            match next_open(&self.buf) {
                Some(open) => {
                    let after_open = &self.buf[open.body_start..];
                    if let Some(end) = after_open.find(open.close) {
                        out.push_str(&self.buf[..open.start]);
                        let next = open.body_start + end + open.close.len();
                        self.buf.drain(..next);
                        continue;
                    }
                    // Unterminated at end of stream: real text, emit verbatim.
                    out.push_str(&self.buf);
                    break;
                }
                None => {
                    out.push_str(&self.buf);
                    break;
                }
            }
        }
        self.buf.clear();
        out
    }
}

/// Whether text-mode `<tool_call>` recovery should run over a completed response.
///
/// * Prompt-guided models (`native == false`) always recover — the whole point of
///   the mode is that tool calls arrive as text.
/// * Native models recover only as a **fallback**: when they were offered tools
///   but returned an empty structured `tool_calls` array, some OpenAI-compatible
///   routes (Hermes / DeepSeek chat templates via OpenRouter) emit the call as
///   `<tool_call>…</tool_call>` text instead of the structured field — recovering
///   it keeps the raw markup from leaking to the caller as assistant content.
/// * When native tool calls came back structured (`structured_calls > 0`),
///   recovery is skipped so the native path stays byte-for-byte unchanged.
/// * When no tools were offered, there is nothing to recover.
pub fn should_recover(native: bool, has_tools: bool, structured_calls: usize) -> bool {
    has_tools && (!native || structured_calls == 0)
}

/// Extract prompt-guided `<tool_call>` blocks from a completed [`ModelResponse`]'s
/// text into `message.tool_calls`, replacing the message content with the cleaned
/// prose. No-op when the text carries no blocks — so a plain text answer is
/// untouched. Provider adapters should apply this to each completed response after
/// using [`with_prompt_tool_instructions`].
pub fn apply_prompt_tool_calls(mut response: ModelResponse) -> ModelResponse {
    let text = response.text();
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(&text);
    if calls.is_empty() {
        return response;
    }
    response.message.tool_calls.extend(calls);
    response.message.content = replace_text_blocks(response.message.content, cleaned);
    response
}

/// Rebuild a content vector, keeping every non-[`ContentBlock::Text`] block (e.g.
/// `Thinking`) in place and substituting the single cleaned text at the position
/// of the first original `Text` block. If the original content had no `Text`
/// block, the cleaned text (when non-empty) is appended; if `cleaned` is empty,
/// no text block is emitted at all.
fn replace_text_blocks(content: Vec<ContentBlock>, cleaned: String) -> Vec<ContentBlock> {
    let mut out = Vec::with_capacity(content.len());
    let mut inserted = false;
    for block in content {
        match block {
            ContentBlock::Text(_) => {
                if !inserted {
                    if !cleaned.is_empty() {
                        out.push(ContentBlock::Text(cleaned.clone()));
                    }
                    inserted = true;
                }
            }
            other => out.push(other),
        }
    }
    if !inserted && !cleaned.is_empty() {
        out.push(ContentBlock::Text(cleaned));
    }
    out
}

/// Parse a single tool-call body into a [`ToolCall`] with a synthetic 1-based id.
fn parse_one(inner: &str, index: usize) -> Option<ToolCall> {
    let value: Value = serde_json::from_str(inner).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    Some(ToolCall {
        id: format!("call_{index}"),
        name,
        arguments,
        invalid: None,
    })
}
