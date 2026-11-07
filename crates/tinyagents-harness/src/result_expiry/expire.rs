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
        let name = tool_name(&messages[..index], &tool.tool_call_id);
        if policy.exempt_tools.iter().any(|exempt| exempt == name) {
            continue;
        }
        let note = note_for(name, &body);
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

/// The name of the tool the call `tool_call_id` (made before `earlier`'s end)
/// invoked.
fn tool_name<'a>(earlier: &'a [Message], tool_call_id: &str) -> &'a str {
    earlier
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::Assistant(assistant) => assistant
                .tool_calls
                .iter()
                .find(|call| call.id == tool_call_id)
                .map(|call| call.name.as_str()),
            _ => None,
        })
        .unwrap_or(UNKNOWN_TOOL)
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

fn is_note(body: &str) -> bool {
    body.starts_with("[used ") && body.ends_with(']')
}

#[cfg(test)]
#[path = "expire_tests.rs"]
mod test;
