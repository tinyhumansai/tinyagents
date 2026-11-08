//! Replaces old tool results with a one-line note.
//!
//! A tool result is re-sent verbatim on every later call, long after the model
//! stopped needing it. [`expire_tool_results`] rewrites each result older than
//! the policy allows to `[used <tool>: <first 120 chars> …]`, so the stored
//! transcript, and every prompt built from it, stops carrying the body.
//!
//! The rewrite touches only the message content, in place. Everything before
//! the first rewritten message is byte for byte what it was, so a provider's
//! cached prefix stays valid up to it.

use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::tool::ToolCall;

use super::types::{EXPIRY_NOTE_CHARS, ToolResultExpiryPolicy};

/// The tool named in a note when no assistant message made the call.
const UNKNOWN_TOOL: &str = "tool";

/// Rewrites every tool result in `messages` that is older than
/// `policy.max_age_turns` user turns and returns how many it rewrote. Does
/// nothing when the policy is not enabled.
///
/// A result's age is the number of user messages after it. A result is left
/// alone when its tool is in `policy.exempt_tools`, when it is marked
/// `trusted_verbatim`, or when it is already a note, so the rewrite is
/// idempotent.
pub fn expire_tool_results(messages: &mut [Message], policy: &ToolResultExpiryPolicy) -> usize {
    if !policy.enabled {
        return 0;
    }
    let mut expired = 0;
    let mut age = 0usize;
    // Walk newest to oldest so the age is a running count of user messages.
    for index in (0..messages.len()).rev() {
        if matches!(messages[index], Message::User(_)) {
            age += 1;
            continue;
        }
        if age <= policy.max_age_turns {
            continue;
        }
        let Message::Tool(tool) = &messages[index] else {
            continue;
        };
        if tool.trusted_verbatim {
            continue;
        }
        let body = messages[index].text();
        if is_note(&body) {
            continue;
        }
        let name = effective_tool_name(messages, index);
        if policy.exempt_tools.iter().any(|exempt| exempt == &name) {
            continue;
        }
        let note = note_for(&name, &body);
        if let Message::Tool(tool) = &mut messages[index] {
            tool.content = vec![ContentBlock::Text(note)];
            expired += 1;
        }
    }
    if expired > 0 {
        tracing::debug!(
            target: "tinyagents::result_expiry",
            expired,
            max_age_turns = policy.max_age_turns,
            "[result_expiry] replaced old tool results with notes"
        );
    }
    expired
}

/// The assistant call that produced the tool result at `index`.
///
/// The call comes from the nearest earlier assistant message that holds one
/// with the result's id. When that message repeats the id, the result is
/// paired by occurrence: the n-th result carrying the id answers the n-th call
/// carrying it, so duplicate provider ids cannot point two results at one call.
pub(crate) fn call_for_result(messages: &[Message], index: usize) -> Option<&ToolCall> {
    let Message::Tool(result) = messages.get(index)? else {
        return None;
    };
    let id = result.tool_call_id.as_str();
    let (assistant_at, assistant) =
        messages[..index]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(at, message)| match message {
                Message::Assistant(assistant)
                    if assistant.tool_calls.iter().any(|call| call.id == id) =>
                {
                    Some((at, assistant))
                }
                _ => None,
            })?;
    let occurrence = messages[assistant_at + 1..index]
        .iter()
        .filter(|message| matches!(message, Message::Tool(tool) if tool.tool_call_id == id))
        .count();
    let mut matching = assistant.tool_calls.iter().filter(|call| call.id == id);
    let nth = matching.clone().nth(occurrence);
    nth.or_else(|| matching.next_back())
}

/// The name of the tool that actually ran for the result at `index`: the
/// target a pseudo `tool_call` wrapper was repaired to, else the called name.
pub(crate) fn effective_call_name(call: &ToolCall) -> String {
    if crate::agent_loop::tool_call_wrapper::is_wrapper_name(&call.name)
        && let Some((target, _)) =
            crate::agent_loop::tool_call_wrapper::unwrap_wrapped_call(&call.arguments, &|_| true)
    {
        return target;
    }
    call.name.clone()
}

fn effective_tool_name(messages: &[Message], index: usize) -> String {
    call_for_result(messages, index)
        .map(effective_call_name)
        .unwrap_or_else(|| UNKNOWN_TOOL.to_string())
}

/// `[used <tool>: <first 120 chars of body on one line>]`, with ` …` before the
/// bracket when the body was cut.
fn note_for(tool: &str, body: &str) -> String {
    let flat = body.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = flat.chars();
    let head: String = chars.by_ref().take(EXPIRY_NOTE_CHARS).collect();
    let cut = if chars.next().is_some() {
        " \u{2026}"
    } else {
        ""
    };
    format!("[used {tool}: {head}{cut}]")
}

/// Slack over the note head for the tool name and the `[used ` / ` …]` frame.
const NOTE_FRAME_CHARS: usize = 128;

/// Whether `body` is a note this module wrote. A note is one line no longer
/// than its head plus the frame, so a large body that merely starts with
/// `[used ` and ends with `]` is still expired.
fn is_note(body: &str) -> bool {
    body.starts_with("[used ")
        && body.ends_with(']')
        && !body.contains('\n')
        && body.chars().count() <= EXPIRY_NOTE_CHARS + NOTE_FRAME_CHARS
}

#[cfg(test)]
#[path = "expire_tests.rs"]
mod test;
